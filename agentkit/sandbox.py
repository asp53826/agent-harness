"""Sandboxed code execution.

An agent that can run code it wrote is an agent that can run code an attacker
wrote, because the text that produced it came from a model that read untrusted
input. "Run it in a subprocess" is not a sandbox: a subprocess can still open
sockets, read your SSH keys, fill the disk, and fork until the machine dies.

So this is defence in depth, and every layer is tested by actually trying to
break out of it (see tests/test_sandbox.py):

  1. separate process, killed by process group so children die too
  2. POSIX rlimits: CPU seconds, address space, file size, process count
  3. an OS sandbox profile where one exists (seatbelt on macOS, bubblewrap on
     Linux) denying network and all filesystem writes outside the work dir
  4. a scrubbed environment and a temporary working directory
  5. wall clock timeout independent of the CPU limit, so a sleeping process
     still gets killed

Layer 3 is the only one that stops network access, and it's the only one that
isn't portable. `Sandbox.capabilities()` reports honestly what's active, and
`strict=True` refuses to run at all rather than silently executing untrusted
code with less isolation than the caller asked for.
"""

from __future__ import annotations

import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field


@dataclass
class SandboxLimits:
    cpu_seconds: int = 5
    wall_seconds: float = 10.0
    memory_mb: int = 512
    file_size_mb: int = 16
    # extra processes allowed *beyond* what this user already has running.
    # RLIMIT_NPROC counts every process owned by the uid, not just this tree,
    # so an absolute value low enough to stop a fork bomb also blocks the very
    # first legitimate subprocess. None disables it and leaves fork bombs to
    # the wall clock timeout plus the process group kill.
    extra_processes: int | None = 96
    max_output_bytes: int = 64 * 1024
    allow_network: bool = False


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    duration: float
    timed_out: bool = False
    killed_by: str | None = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def as_dict(self) -> dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration": round(self.duration, 3),
            "timed_out": self.timed_out,
            "killed_by": self.killed_by,
            "ok": self.ok,
        }


# Seatbelt profile. Deny by default, then allow the narrow set a python
# interpreter needs to start: read its own stdlib, map memory, and write only
# inside the work directory. Explicitly denies network at the syscall level,
# which no amount of rlimit tuning can do.
SEATBELT = """(version 1)
(deny default)

; a python interpreter needs to exec itself, map its own shared libraries and
; read the stdlib. reads are broad on purpose: the threat being defended
; against is exfiltration and destruction, and without network or write access
; a read of the filesystem cannot leave the sandbox.
(allow process-exec)
(allow process-fork)
(allow file-read*)
(allow file-map-executable)
(allow file-ioctl)
(allow signal (target self))
(allow sysctl-read)
(allow mach-lookup)
(allow ipc-posix-shm)
(allow system-socket)

; writes are confined to the work directory and the usual null sinks
(allow file-write*
    (subpath "{workdir}")
    (subpath "{tmpdir}")
    (literal "/dev/null")
    (literal "/dev/zero")
    (literal "/dev/urandom")
    (literal "/dev/random")
    (literal "/dev/dtracehelper"))

; the control that rlimits cannot provide
(deny network*)
"""


def _find_bwrap() -> str | None:
    return shutil.which("bwrap")


def _find_unshare() -> str | None:
    return shutil.which("unshare")


class Sandbox:
    def __init__(self, limits: SandboxLimits | None = None, strict: bool = False,
                 python: str | None = None):
        self.limits = limits or SandboxLimits()
        self.strict = strict
        self.python = python or sys.executable
        self._caps = self._detect()
        self._nproc_cap = self._compute_nproc_cap()
        if strict and not self._caps["os_sandbox"]:
            raise RuntimeError(
                "strict mode requires an OS sandbox (seatbelt or bubblewrap) and "
                f"none is available on {platform.system()}. refusing to run "
                "untrusted code with only rlimits."
            )

    def _detect(self) -> dict:
        system = platform.system()
        seatbelt = system == "Darwin" and os.path.exists("/usr/bin/sandbox-exec")
        # bubblewrap configures loopback whenever it creates the network
        # namespace. Some otherwise-capable hosts (including GitHub Actions)
        # deny that netlink operation. util-linux unshare can create the empty
        # namespace without configuring an interface, then bubblewrap can keep
        # that namespace while applying the filesystem sandbox.
        bwrap = (system == "Linux"
                 and _find_bwrap() is not None
                 and _find_unshare() is not None)
        return {
            "platform": system,
            "rlimits": True,
            "os_sandbox": bool(seatbelt or bwrap),
            "mechanism": "seatbelt" if seatbelt else "bubblewrap" if bwrap else "rlimits-only",
            "network_blocked": bool(seatbelt or bwrap),
            "filesystem_confined": bool(seatbelt or bwrap),
        }

    def _compute_nproc_cap(self) -> int | None:
        """Current process count for this user, plus the allowed headroom.

        Counted once per Sandbox rather than per run, since it only needs to be
        roughly right: the point is to bound runaway forking, not to be exact.
        """
        if self.limits.extra_processes is None:
            return None
        try:
            out = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid="],
                                 capture_output=True, text=True, timeout=5)
            current = len([ln for ln in out.stdout.splitlines() if ln.strip()])
        except Exception:
            return None
        return current + self.limits.extra_processes

    def capabilities(self) -> dict:
        return dict(self._caps)

    def _preexec(self):
        """Runs in the child between fork and exec.

        No setsid here: Popen(start_new_session=True) already made this process
        a session leader, and calling it again fails with EPERM. That new
        session is what lets the timeout kill the whole tree rather than just
        the direct child.
        """
        lim = self.limits
        # a hard limit one second above the soft one, so SIGXCPU arrives first
        # and python can print a traceback before SIGKILL lands
        resource.setrlimit(resource.RLIMIT_CPU, (lim.cpu_seconds, lim.cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE,
                           (lim.file_size_mb * 1024 * 1024,) * 2)
        if self._nproc_cap is not None:
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (self._nproc_cap,) * 2)
            except (ValueError, OSError):
                pass  # not enforceable everywhere
        try:
            nbytes = lim.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        except (ValueError, OSError):
            pass
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def _wrap_command(self, argv: list[str], workdir: str) -> list[str]:
        mech = self._caps["mechanism"]
        if mech == "seatbelt" and not self.limits.allow_network:
            # seatbelt matches the resolved path. on macOS a temp dir comes back
            # as /var/folders/... which is a symlink to /private/var/folders/...,
            # so an unresolved subpath rule silently matches nothing and the
            # sandbox denies writes to its own work directory.
            profile = SEATBELT.format(
                workdir=os.path.realpath(workdir),
                tmpdir=os.path.realpath(tempfile.gettempdir()),
            )
            return ["/usr/bin/sandbox-exec", "-p", profile, *argv]
        if mech == "bubblewrap" and not self.limits.allow_network:
            # A virtualenv launcher and its resolved base interpreter can live
            # under different prefixes. Both must be readable: the former owns
            # site-packages, while the latter owns the actual executable and
            # standard library.
            python_prefixes = dict.fromkeys([
                os.path.dirname(os.path.dirname(os.path.abspath(self.python))),
                os.path.dirname(os.path.dirname(os.path.realpath(self.python))),
            ])
            python_binds = [
                item
                for prefix in python_prefixes
                for item in ("--ro-bind-try", prefix, prefix)
            ]
            return [
                _find_unshare(),
                "--user",
                "--map-root-user",
                "--net",
                "--",
                _find_bwrap(),
                "--unshare-all",
                # Retain the outer, unconfigured network namespace. This keeps
                # networking unavailable without asking bubblewrap to add a
                # loopback address, which restricted hosts may prohibit.
                "--share-net",
                "--ro-bind", "/usr", "/usr",
                "--ro-bind-try", "/lib", "/lib",
                "--ro-bind-try", "/lib64", "/lib64",
                "--ro-bind-try", "/bin", "/bin",
                # setup-python and virtualenvs can install outside /usr. Bind
                # only their runtime prefixes read-only so sys.executable,
                # site-packages and the standard library remain available.
                *python_binds,
                "--proc", "/proc",
                "--dev", "/dev",
                "--bind", workdir, workdir,
                "--chdir", workdir,
                "--die-with-parent",
                *argv,
            ]
        return argv

    def run_code(self, code: str, files: dict[str, str] | None = None) -> ExecResult:
        """Execute python source. Returns rather than raises on failure."""
        with tempfile.TemporaryDirectory(prefix="agentkit-") as workdir:
            for name, content in (files or {}).items():
                # keep writes inside the work dir even if the name tries to escape
                safe = os.path.normpath(os.path.join(workdir, name))
                if not safe.startswith(os.path.realpath(workdir)) and not safe.startswith(workdir):
                    raise ValueError(f"file path escapes the work directory: {name}")
                os.makedirs(os.path.dirname(safe), exist_ok=True)
                with open(safe, "w") as f:
                    f.write(content)

            script = os.path.join(workdir, "_main.py")
            with open(script, "w") as f:
                f.write(textwrap.dedent(code))

            return self._spawn([self.python, "-I", "-B", script], workdir)

    def run_argv(self, argv: list[str], workdir: str | None = None) -> ExecResult:
        if workdir is None:
            with tempfile.TemporaryDirectory(prefix="agentkit-") as tmp:
                return self._spawn(argv, tmp)
        return self._spawn(argv, workdir)

    def _spawn(self, argv: list[str], workdir: str) -> ExecResult:
        lim = self.limits
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": workdir,
            "TMPDIR": workdir,
            "PYTHONHASHSEED": "0",       # deterministic, so evals are reproducible
            "PYTHONDONTWRITEBYTECODE": "1",
            "LC_ALL": "C",
        }
        cmd = self._wrap_command(argv, workdir)

        t0 = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=self._preexec,
            start_new_session=True,
        )

        timed_out = False
        try:
            out, err = proc.communicate(timeout=lim.wall_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_tree(proc)
            out, err = proc.communicate()

        duration = time.perf_counter() - t0
        stdout, t1 = self._clip(out)
        stderr, t2 = self._clip(err)

        killed_by = None
        if proc.returncode is not None and proc.returncode < 0:
            killed_by = signal.Signals(-proc.returncode).name

        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode if proc.returncode is not None else -1,
            duration=duration,
            timed_out=timed_out,
            killed_by=killed_by,
            truncated=t1 or t2,
        )

    @staticmethod
    def _kill_tree(proc) -> None:
        """SIGKILL the whole process group.

        Killing only the direct child leaves grandchildren running, which is
        exactly what a fork bomb produces.
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def _clip(self, raw: bytes) -> tuple[str, bool]:
        """Bound the output. An unbounded read is its own denial of service."""
        limit = self.limits.max_output_bytes
        truncated = len(raw) > limit
        text = raw[:limit].decode("utf-8", errors="replace")
        if truncated:
            text += f"\n[output truncated at {limit} bytes]"
        return text, truncated
