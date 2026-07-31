"""Sandbox tests that actually try to escape.

A sandbox nobody attacked is a claim, not a control. Each of these runs real
hostile code and asserts it was contained: reaching the network, reading
outside the work directory, forking without bound, allocating without bound,
spinning forever, and writing a file large enough to fill a disk.

Anything requiring an OS sandbox is skipped rather than silently passing where
none exists, because a green tick on a machine that can't enforce the control
is worse than a skip.
"""

import os
import platform
import socket

import pytest

import agentkit.sandbox as sandbox_module
from agentkit.sandbox import ExecResult, Sandbox, SandboxLimits

HAS_OS_SANDBOX = Sandbox().capabilities()["os_sandbox"]
needs_os_sandbox = pytest.mark.skipif(
    not HAS_OS_SANDBOX,
    reason=f"no OS sandbox on {platform.system()}, rlimits alone can't enforce this",
)


def box(**kw):
    return Sandbox(SandboxLimits(**kw))


# ------------------------------------------------------------- it still works

def test_runs_ordinary_code():
    r = box().run_code("print(sum(range(100)))")
    assert r.ok and r.stdout.strip() == "4950"


def test_reports_a_nonzero_exit_without_raising():
    r = box().run_code("raise SystemExit(3)")
    assert r.exit_code == 3 and not r.ok


def test_captures_a_traceback_on_stderr():
    r = box().run_code("raise ValueError('boom')")
    assert "ValueError" in r.stderr and "boom" in r.stderr
    assert r.exit_code != 0


def test_input_files_are_readable():
    r = box().run_code("print(open('data.txt').read().strip())",
                       files={"data.txt": "hello from a file"})
    assert r.stdout.strip() == "hello from a file"


def test_can_write_inside_the_work_directory():
    r = box().run_code(
        "open('out.txt','w').write('x'*100)\n"
        "print(len(open('out.txt').read()))"
    )
    assert r.ok and r.stdout.strip() == "100"


def test_work_directory_is_cleaned_up():
    r = box().run_code("import os; print(os.getcwd())")
    assert not os.path.exists(r.stdout.strip())


def test_capabilities_are_reported_honestly():
    caps = Sandbox().capabilities()
    assert caps["rlimits"] is True
    assert caps["mechanism"] in ("seatbelt", "bubblewrap", "rlimits-only")
    # the claim and the mechanism must agree, no overselling
    assert caps["network_blocked"] == caps["os_sandbox"]


def test_bubblewrap_retains_an_outer_empty_network_namespace(monkeypatch):
    """Do not let bubblewrap configure loopback on restricted Linux hosts."""
    monkeypatch.setattr(sandbox_module, "_find_unshare", lambda: "/usr/bin/unshare")
    monkeypatch.setattr(sandbox_module, "_find_bwrap", lambda: "/usr/bin/bwrap")
    sb = box()
    sb._caps["mechanism"] = "bubblewrap"

    command = sb._wrap_command(["/runtime/bin/python", "script.py"], "/work")

    assert command[:6] == [
        "/usr/bin/unshare", "--user", "--map-root-user", "--net", "--",
        "/usr/bin/bwrap",
    ]
    assert "--unshare-all" in command
    assert "--share-net" in command
    assert "--unshare-net" not in command
    prefixes = {
        os.path.dirname(os.path.dirname(os.path.abspath(sb.python))),
        os.path.dirname(os.path.dirname(os.path.realpath(sb.python))),
    }
    triplets = [command[i:i + 3] for i in range(len(command) - 2)]
    for prefix in prefixes:
        assert ["--ro-bind-try", prefix, prefix] in triplets


# ------------------------------------------------------------- escape attempts

@needs_os_sandbox
def test_network_is_blocked():
    """The one an rlimit cannot stop. Exfiltration is the actual worst case."""
    r = box(wall_seconds=15).run_code("""
        import socket
        s = socket.socket()
        s.settimeout(5)
        s.connect(("1.1.1.1", 80))
        print("CONNECTED")
    """)
    assert "CONNECTED" not in r.stdout, "sandbox allowed an outbound connection"
    assert not r.ok


@needs_os_sandbox
def test_dns_is_blocked():
    r = box(wall_seconds=15).run_code("""
        import socket
        print("RESOLVED", socket.gethostbyname("example.com"))
    """)
    assert "RESOLVED" not in r.stdout
    assert not r.ok


@needs_os_sandbox
def test_host_loopback_is_blocked():
    """A private loopback socket must not reach services on the host."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        r = box().run_code(f"""
            import socket
            s = socket.socket()
            s.settimeout(2)
            s.connect(("127.0.0.1", {port}))
            print("CONNECTED")
        """)
    assert "CONNECTED" not in r.stdout, "sandbox reached a host loopback service"
    assert not r.ok


@needs_os_sandbox
def test_cannot_write_outside_the_work_directory():
    target = os.path.expanduser("~/agentkit_escape_probe.txt")
    r = box().run_code(f"""
        open({target!r}, "w").write("escaped")
        print("WROTE")
    """)
    assert "WROTE" not in r.stdout, "sandbox allowed a write to the home directory"
    assert not os.path.exists(target)


@needs_os_sandbox
def test_cannot_write_to_system_paths():
    r = box().run_code("""
        open("/tmp/../etc/agentkit_probe", "w").write("x")
        print("WROTE")
    """)
    assert "WROTE" not in r.stdout


def test_infinite_loop_is_killed_by_wall_clock():
    r = box(wall_seconds=2.0, cpu_seconds=30).run_code("while True: pass")
    assert r.timed_out
    assert r.duration < 6.0, "timeout did not fire promptly"


def test_sleeping_process_is_killed_too():
    """A CPU limit alone never fires on a process that isn't burning CPU."""
    r = box(wall_seconds=1.5, cpu_seconds=30).run_code("import time; time.sleep(600)")
    assert r.timed_out and r.duration < 6.0


def test_cpu_limit_kills_a_busy_loop():
    r = box(cpu_seconds=1, wall_seconds=30).run_code("""
        x = 0
        while True:
            x += 1
    """)
    assert not r.ok
    assert r.killed_by in ("SIGKILL", "SIGXCPU") or r.exit_code != 0


def test_memory_bomb_is_contained():
    """Must fail cleanly rather than taking the host into swap."""
    r = box(memory_mb=128, wall_seconds=20).run_code("""
        chunks = []
        while True:
            chunks.append(bytearray(20 * 1024 * 1024))
    """)
    assert not r.ok
    assert "MemoryError" in r.stderr or r.killed_by or r.timed_out


def test_fork_bomb_is_contained():
    """The tell is duration: if the host were struggling this would not
    return promptly, and the whole process group has to die with it."""
    r = box(wall_seconds=6.0, extra_processes=16).run_code("""
        import os
        while True:
            os.fork()
    """)
    assert not r.ok
    assert r.duration < 15.0


def test_child_processes_do_not_outlive_the_timeout():
    r = box(wall_seconds=2.0).run_code("""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        time.sleep(300)
    """)
    assert r.timed_out
    assert r.duration < 8.0


def test_giant_file_write_is_capped():
    r = box(file_size_mb=2, wall_seconds=20).run_code("""
        with open("big.bin", "wb") as f:
            for _ in range(200):
                f.write(b"0" * 1024 * 1024)
        print("WROTE ALL")
    """)
    assert "WROTE ALL" not in r.stdout


def test_output_flood_is_truncated():
    """An unbounded read into the parent is its own denial of service."""
    r = box(max_output_bytes=4096, wall_seconds=20).run_code(
        "print('A' * 10_000_000)"
    )
    assert r.truncated
    assert len(r.stdout) < 20_000


def test_environment_is_scrubbed():
    """Credentials in the parent's env must not reach the child."""
    os.environ["AGENTKIT_SECRET_PROBE"] = "topsecret"
    try:
        r = box().run_code("""
            import os
            print("SECRET" if "AGENTKIT_SECRET_PROBE" in os.environ else "CLEAN")
            print(len(os.environ))
        """)
        assert r.stdout.splitlines()[0] == "CLEAN"
    finally:
        os.environ.pop("AGENTKIT_SECRET_PROBE", None)


def test_input_file_cannot_escape_the_work_directory():
    with pytest.raises(ValueError, match="escapes"):
        box().run_code("pass", files={"../../evil.txt": "x"})


def test_strict_mode_refuses_when_it_cannot_enforce():
    """Better to refuse than to run untrusted code with less isolation than
    the caller asked for."""
    if HAS_OS_SANDBOX:
        Sandbox(strict=True)  # must not raise where it can be enforced
    else:
        with pytest.raises(RuntimeError, match="strict mode requires"):
            Sandbox(strict=True)


def test_repeated_runs_are_isolated_from_each_other():
    sb = box()
    sb.run_code("open('leak.txt','w').write('from run 1')")
    r = sb.run_code("""
        import os
        print("LEAKED" if os.path.exists("leak.txt") else "ISOLATED")
    """)
    assert r.stdout.strip() == "ISOLATED"


def test_result_serializes():
    r = box().run_code("print(1)")
    d = r.as_dict()
    assert d["ok"] is True and d["exit_code"] == 0
    assert isinstance(ExecResult(**{k: v for k, v in d.items() if k != "ok"}), ExecResult)
