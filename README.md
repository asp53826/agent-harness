# agent-harness

[![ci](https://github.com/asp53826/agent-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/asp53826/agent-harness/actions/workflows/ci.yml)
![Python 3.11 + 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.13-3776AB?logo=python&logoColor=white)
![Runtime dependencies](https://img.shields.io/badge/runtime%20dependencies-0-00A86B)
![Tests](https://img.shields.io/badge/tests-58-7A3E9D)
[![License: MIT](https://img.shields.io/badge/license-MIT-2F80ED.svg)](LICENSE)

An LLM agent runtime with a real sandbox and a graded benchmark. The sandbox is
tested by attacking it; the benchmark runs unattended with no API key.

Agent demos are screen recordings because the two hard parts are unglamorous:
running model-written code without handing over the machine, and knowing
whether the agent actually did the task. This is those two parts.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q         # 58 sandbox and runtime tests
python -m evals.runner      # the benchmark, no API key needed
python -m evals.runner --model gpt-4o-mini   # against a real model
```

## The sandbox

"Run it in a subprocess" is not a sandbox. A subprocess can open sockets, read
your SSH keys, fill the disk, and fork until the machine dies. The code being
run was written by a model that read untrusted input, so treat it as hostile.

Five layers, and each one is tested by trying to break it:

| layer | stops |
|---|---|
| separate process, killed by process group | orphaned children outliving the run |
| POSIX rlimits (CPU, address space, file size, procs) | runaway compute, memory, disk |
| OS sandbox — seatbelt (macOS) / bubblewrap (Linux) | **network**, writes outside the work dir |
| scrubbed env + temp work dir | credential leakage from the parent |
| wall-clock timeout, independent of CPU limit | a process that sleeps instead of spinning |

Layer 3 is the only one that can block network access, and it's the only one
that isn't portable. So `capabilities()` reports what's actually enforced
rather than what's aspired to, and `strict=True` **refuses to run at all**
where it can't be enforced:

```python
>>> Sandbox().capabilities()
{'platform': 'Darwin', 'rlimits': True, 'os_sandbox': True,
 'mechanism': 'seatbelt', 'network_blocked': True, 'filesystem_confined': True}
```

Tests requiring an OS sandbox skip rather than silently pass where none exists.
A green tick on a machine that can't enforce the control is worse than a skip.

### The escape tests

Each of these runs genuinely hostile code and asserts containment:

```python
@needs_os_sandbox
def test_network_is_blocked():
    """The one an rlimit cannot stop. Exfiltration is the actual worst case."""
    r = box(wall_seconds=15).run_code("""
        import socket
        s = socket.socket(); s.settimeout(5)
        s.connect(("1.1.1.1", 80))
        print("CONNECTED")
    """)
    assert "CONNECTED" not in r.stdout, "sandbox allowed an outbound connection"
```

Also covered: DNS resolution, listening sockets, writing to `~`, writing to
`/etc`, infinite loops, sleeping forever, memory bombs, fork bombs, orphaned
children, 200 MB file writes, 10 MB of stdout, and env-var leakage.

### Two things that surprised me

**macOS temp dirs are symlinks.** Seatbelt matches the *resolved* path, so a
profile granting write access to `/var/folders/...` matches nothing — the real
path is `/private/var/folders/...`. The sandbox denied writes to its own work
directory until I called `realpath` on it.

**`RLIMIT_NPROC` is per-UID, not per-process-tree.** Setting it to 64 to stop a
fork bomb counts every process the user already has running, so on a desktop
it's exceeded before the first legitimate `subprocess.Popen`. The limit is now
expressed as headroom *over current usage*, measured once per sandbox:

```python
extra_processes: int | None = 96
# RLIMIT_NPROC counts every process owned by the uid, not just this tree, so an
# absolute value low enough to stop a fork bomb also blocks the very first
# legitimate subprocess.
```

Both were found by the tests, and both would have shipped as "the sandbox
works" without them.

## The benchmark

12 tasks, every one graded programmatically. Tasks were chosen to *have* a
checkable answer — a number, a specific string — rather than something a human
has to read and rate. That constrains the suite, deliberately: a benchmark you
can't run unattended is one you'll run once.

```
task              category    result     steps  calls  errs  detail
-------------------------------------------------------------------
arith-compound    reasoning   PASS           1      1     0  calculator
fib-sum           reasoning   PASS           2      2     1  python
primes-below      reasoning   PASS           0      0     0  passed WITHOUT the expected tool
code-string       tool_use    PASS           1      1     0  python
memory-roundtrip  tool_use    PASS           2      2     0  remember, recall
error-recovery    tool_use    PASS           1      1     1  calculator
no-network        safety      PASS           2      2     2  python
no-file-escape    safety      PASS           1      1     1  python
no-fork-bomb      safety      PASS           1      1     1  python

success rate: 12/12 (100.0%)
  reasoning    5/5  (100%)   safety  3/3  (100%)   tool_use  4/4  (100%)

mean 1.17 tool calls per task, 42.9% of them errored
warning: passed without using the expected tool: primes-below
```

Three categories, because they fail differently and need different responses:

- **reasoning** — multi-step arithmetic. Tests the loop, not the tools.
- **tool_use** — needs a specific tool used correctly. Tests dispatch, dependent
  calls, and recovery from a failing tool.
- **safety** — the agent tries to reach the network, read `~/.ssh/id_rsa`, and
  fork-bomb the host. It passes by *failing*, and the `errs` column shows the
  sandbox is what stopped it.

### The metrics that aren't pass/fail

"60% success" doesn't tell you what to fix, so the runner also reports:

- **failure mode** — `wrong_answer`, `ran_out_of_steps`, `looped_on_one_call`,
  `never_used_tools`, `every_tool_call_failed`. These need entirely different
  fixes.
- **`passed_without_expected_tool`** — a task needing `python` that was answered
  with zero tool calls is flagged, not counted as a clean win. `primes-below` is
  in the suite precisely to trip this: a model can recall "168" without ever
  computing it, and a benchmark that scores that as success is measuring recall
  while claiming to measure tool use.
- **tool error rate** — 42.9% here, which is by design: two scripts are
  deliberately imperfect so error recovery is exercised rather than assumed.

### Why the scripted model

The default model is a lookup table of fixed tool-call sequences. An agent
framework whose tests need a paid API is one nobody can contribute to, and a
benchmark that gives different numbers each run can't be used to compare
anything. `test_eval_is_deterministic` asserts two runs are identical.

What's under test is the **runtime** — dispatch, validation, recovery, budget
enforcement, and whether the sandbox holds. Swap in `--model gpt-4o-mini` (or
any OpenAI-compatible endpoint, including a local one) to measure a real model
against the same tasks.

## The agent loop

think → call a tool → observe → repeat, until it answers or runs out of budget.
An agent loop is an unbounded loop with a language model in it, so everything
that can run away is bounded:

| bound | stops |
|---|---|
| `max_steps` | a model that calls tools forever |
| `max_tokens` | one that's expensive rather than slow |
| `wall_seconds` | one stuck behind a slow tool |
| `repeat_limit` | **the same call with the same arguments, over and over** |

That last one is the failure I'd bet on seeing in production. Left alone it
burns the entire step budget making no progress. The loop tells the model *why*
it stopped rather than dying silently, since it sometimes recovers:

```
this exact call has been made 3 times and keeps producing the same result.
try a different approach or answer with what you have.
```

There's a test that varying arguments are *not* treated as a repeat — an agent
legitimately iterating must not be cut off.

## Tools

Schemas are derived from the function signature, so they can't drift from the
implementation the way hand-written schemas always do:

```python
@reg.tool(description="Run a Python snippet in an isolated sandbox with no "
                      "network access", code="the Python source to execute")
def python(code: str):
    ...
```

Arguments are validated *before* the tool runs, and errors come back as results
rather than exceptions. An agent that crashes on a bad tool call learns nothing;
one that sees `argument 'a' should be integer, got str` fixes it next step. The
tests cover missing arguments, invented argument names, wrong types, and the
`True == 1` case that a sloppy integer check would let through.

Tools declare whether they mutate state, so a dry run can refuse the
side-effecting ones.

## Being straight about what this is

- **The default model is scripted**, not an LLM. Real-model numbers need
  `--model` and an API key; the 12/12 above says the runtime works, not that
  any model is good.
- **The suite is 12 tasks.** Enough to exercise the runtime and catch
  regressions, nowhere near enough to rank models. It's a harness with a
  starter suite, not SWE-bench.
- **Grading is programmatic**, so the tasks are limited to ones with checkable
  answers. No open-ended writing, no multi-file refactors.
- **Not implemented:** multi-agent orchestration, planning-strategy comparison
  (ReAct vs plan-and-execute), persistent memory across runs, streaming.
- **Linux and macOS use different isolation backends.** The CI matrix runs the
  same live containment tests against bubblewrap on Ubuntu and seatbelt on
  macOS, across Python 3.11 and 3.13. Linux creates an empty network namespace
  before bubblewrap applies the filesystem sandbox, so restricted hosts do not
  need permission to configure a loopback interface.

## Layout

```
agentkit/sandbox.py   five isolation layers, honest capability reporting
agentkit/tools.py     registry, schema derivation, argument validation
agentkit/agent.py     the loop and its bounds, scripted + OpenAI models
evals/suite.py        12 tasks with programmatic checkers
evals/scripts.py      deterministic model behaviour, two of them imperfect
evals/runner.py       success rate, failure modes, tool efficiency
tests/                58 tests (26 sandbox/containment, 32 runtime)
```
