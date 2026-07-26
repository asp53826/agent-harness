#!/usr/bin/env python3
"""Run the suite and report success rate.

Reports more than pass/fail, because "60% success" tells you nothing about what
to fix:

  success rate        by category, since safety failures and reasoning failures
                      need completely different responses
  tool efficiency     calls made against the minimum needed. an agent that gets
                      there in 8 steps instead of 2 is working and expensive.
  failure mode        why it failed — wrong answer, ran out of steps, looped on
                      the same call, or never used the tool the task required
  used_expected_tool  guards against a lucky guess. getting 168 primes from
                      memory is not the same as computing it.

Every run is seeded and the scripted model is deterministic, so two runs of the
same configuration give identical numbers.

  python -m evals.runner                      # scripted model, no API key
  python -m evals.runner --model gpt-4o-mini  # needs OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentkit.agent import Agent, AgentConfig  # noqa: E402
from agentkit.sandbox import Sandbox, SandboxLimits  # noqa: E402
from agentkit.tools import build_default_registry  # noqa: E402
from evals.suite import CATEGORIES, Task, get_tasks  # noqa: E402


@dataclass
class TaskResult:
    task_id: str
    category: str
    passed: bool
    answer: str
    failure_mode: str | None
    steps: int
    tool_calls: int
    tool_errors: int
    tools_used: list[str] = field(default_factory=list)
    used_expected_tool: bool = True
    stop_reason: str = ""
    seconds: float = 0.0


def classify_failure(task: Task, traj, passed: bool) -> str | None:
    if passed:
        return None
    if traj.stop_reason == "max_steps":
        return "ran_out_of_steps"
    if traj.stop_reason == "repeated_tool_call":
        return "looped_on_one_call"
    if traj.stop_reason in ("wall_clock", "token_budget"):
        return f"budget_{traj.stop_reason}"
    if not traj.answer:
        return "no_answer"
    if traj.num_tool_calls == 0 and task.expects_tools:
        return "never_used_tools"
    if traj.num_tool_errors and traj.num_tool_errors == traj.num_tool_calls:
        return "every_tool_call_failed"
    return "wrong_answer"


def run_task(agent: Agent, task: Task) -> TaskResult:
    agent.config.max_steps = task.max_steps
    t0 = time.perf_counter()
    traj = agent.run(task.prompt)
    passed = bool(task.check(traj.answer or "", traj))
    used = traj.tools_used()
    return TaskResult(
        task_id=task.id,
        category=task.category,
        passed=passed,
        answer=(traj.answer or "")[:400],
        failure_mode=classify_failure(task, traj, passed),
        steps=len(traj.steps),
        tool_calls=traj.num_tool_calls,
        tool_errors=traj.num_tool_errors,
        tools_used=used,
        # no expectation means nothing to check, which is the safety tasks
        used_expected_tool=(not task.expects_tools
                            or any(t in used for t in task.expects_tools)),
        stop_reason=traj.stop_reason,
        seconds=time.perf_counter() - t0,
    )


def summarize(results: list[TaskResult]) -> dict:
    if not results:
        return {}
    by_cat: dict[str, list[TaskResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    out = {
        "tasks": len(results),
        "passed": sum(r.passed for r in results),
        "success_rate": round(sum(r.passed for r in results) / len(results), 4),
        "mean_steps": round(sum(r.steps for r in results) / len(results), 2),
        "mean_tool_calls": round(sum(r.tool_calls for r in results) / len(results), 2),
        "tool_error_rate": round(
            sum(r.tool_errors for r in results) / max(sum(r.tool_calls for r in results), 1), 4),
        "by_category": {
            c: {"passed": sum(r.passed for r in rs), "total": len(rs),
                "success_rate": round(sum(r.passed for r in rs) / len(rs), 4)}
            for c, rs in sorted(by_cat.items())
        },
        "failure_modes": dict(Counter(r.failure_mode for r in results if r.failure_mode)),
        "total_seconds": round(sum(r.seconds for r in results), 2),
    }
    # a pass without the required tool is suspicious, not a win
    unearned = [r.task_id for r in results if r.passed and not r.used_expected_tool]
    if unearned:
        out["passed_without_expected_tool"] = unearned
    return out


def print_report(results: list[TaskResult], summary: dict) -> None:
    w = max(len(r.task_id) for r in results) + 2
    head = ("task".ljust(w) + "category".ljust(12) + "result".ljust(9)
            + "steps".rjust(7) + "calls".rjust(7) + "errs".rjust(6) + "  detail")
    print(head)
    print("-" * len(head))
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        detail = r.failure_mode or ", ".join(r.tools_used) or "-"
        if r.passed and not r.used_expected_tool:
            detail = "passed WITHOUT the expected tool"
        print(r.task_id.ljust(w) + r.category.ljust(12) + mark.ljust(9)
              + str(r.steps).rjust(7) + str(r.tool_calls).rjust(7)
              + str(r.tool_errors).rjust(6) + "  " + detail)

    print(f"\nsuccess rate: {summary['passed']}/{summary['tasks']} "
          f"({summary['success_rate']:.1%})")
    for cat, s in summary["by_category"].items():
        print(f"  {cat:<12} {s['passed']}/{s['total']}  ({s['success_rate']:.0%})")
    if summary["failure_modes"]:
        print("\nfailure modes:")
        for mode, n in sorted(summary["failure_modes"].items(), key=lambda kv: -kv[1]):
            print(f"  {mode:<26} {n}")
    print(f"\nmean {summary['mean_tool_calls']} tool calls per task, "
          f"{summary['tool_error_rate']:.1%} of them errored")
    if "passed_without_expected_tool" in summary:
        print("warning: passed without using the expected tool: "
              + ", ".join(summary["passed_without_expected_tool"]))


def build_agent(model_spec: str, max_steps: int, wall_seconds: float) -> Agent:
    sandbox = Sandbox(SandboxLimits(cpu_seconds=5, wall_seconds=10.0, memory_mb=512))
    tools = build_default_registry(sandbox)

    if model_spec == "scripted":
        from evals.scripts import build_scripted_model

        model = build_scripted_model()
    else:
        from agentkit.agent import OpenAIModel

        if not os.environ.get("OPENAI_API_KEY") and "localhost" not in os.environ.get(
                "OPENAI_BASE_URL", ""):
            sys.exit("set OPENAI_API_KEY, or use --model scripted")
        model = OpenAIModel(model_spec)

    return Agent(model, tools, AgentConfig(max_steps=max_steps, wall_seconds=wall_seconds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="scripted")
    ap.add_argument("--categories", default=None,
                    help=f"comma separated, from: {', '.join(CATEGORIES)}")
    ap.add_argument("--tasks", default=None, help="comma separated task ids")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--wall-seconds", type=float, default=120.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    tasks = get_tasks(
        args.categories.split(",") if args.categories else None,
        args.tasks.split(",") if args.tasks else None,
    )
    if not tasks:
        sys.exit("no tasks matched")

    agent = build_agent(args.model, args.max_steps, args.wall_seconds)
    caps = agent.tools.get("python") and Sandbox().capabilities()
    print(f"model: {agent.model.name}   tools: {', '.join(agent.tools.names)}")
    print(f"sandbox: {caps['mechanism']}, network_blocked={caps['network_blocked']}")
    print(f"{len(tasks)} tasks\n")

    results = [run_task(agent, t) for t in tasks]
    summary = summarize(results)
    print_report(results, summary)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"model": agent.model.name, "sandbox": caps,
                       "summary": summary,
                       "results": [asdict(r) for r in results]}, f, indent=2)
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
