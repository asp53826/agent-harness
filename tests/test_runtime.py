"""Tool registry, agent loop, and the eval harness itself.

The loop tests are mostly about the bounds. An agent loop is an unbounded loop
with a language model in it, and every way it can fail to terminate needs a
test, because in production that failure costs money per iteration.
"""

import pytest

from agentkit.agent import Agent, AgentConfig, Message, ModelResponse, ScriptedModel
from agentkit.tools import Tool, ToolRegistry, ToolResult
from evals.runner import classify_failure, run_task, summarize
from evals.scripts import build_scripted_model
from evals.suite import BY_ID, TASKS, get_tasks, refused


# ---------------------------------------------------------------- tools

def registry():
    reg = ToolRegistry()

    @reg.tool(description="add two integers", a="first", b="second")
    def add(a: int, b: int):
        return str(a + b)

    @reg.tool(description="always explodes")
    def boom():
        raise RuntimeError("kaboom")

    @reg.tool(description="changes things", mutates=True, value="what to set")
    def write(value: str):
        return f"wrote {value}"

    return reg


def test_schema_is_derived_from_the_signature():
    reg = registry()
    fn = reg.get("add").schema()["function"]
    assert fn["name"] == "add"
    assert fn["parameters"]["properties"]["a"]["type"] == "integer"
    assert set(fn["parameters"]["required"]) == {"a", "b"}


def test_optional_arguments_are_not_required():
    reg = ToolRegistry()

    @reg.tool(description="greet")
    def greet(name: str, greeting: str = "hello"):
        return f"{greeting} {name}"

    assert reg.get("greet").required == ["name"]
    assert reg.call("greet", {"name": "x"}).content == "hello x"


def test_calls_a_tool():
    assert registry().call("add", {"a": 2, "b": 3}).content == "5"


def test_unknown_tool_is_an_error_not_an_exception():
    r = registry().call("nope", {})
    assert not r.ok and "no tool named" in r.error
    assert "add" in r.error, "should list what is available"


def test_missing_argument_is_caught_before_the_tool_runs():
    r = registry().call("add", {"a": 1})
    assert not r.ok and "missing required" in r.error and "b" in r.error


def test_unknown_argument_is_rejected():
    """A model that invents an argument name gets told, rather than a TypeError."""
    r = registry().call("add", {"a": 1, "b": 2, "c": 3})
    assert not r.ok and "unknown argument" in r.error


def test_wrong_type_is_rejected():
    r = registry().call("add", {"a": "two", "b": 3})
    assert not r.ok and "should be integer" in r.error


def test_bool_is_not_an_integer():
    """True == 1 in python, which would silently pass a sloppy check."""
    assert not registry().call("add", {"a": True, "b": 1}).ok


def test_tool_exception_becomes_a_result():
    r = registry().call("boom", {})
    assert not r.ok and "RuntimeError" in r.error and "kaboom" in r.error


def test_mutations_can_be_disabled():
    reg = registry()
    assert reg.call("write", {"value": "x"}, allow_mutations=True).ok
    r = reg.call("write", {"value": "x"}, allow_mutations=False)
    assert not r.ok and "mutates state" in r.error


def test_duplicate_registration_is_rejected():
    reg = registry()
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Tool("add", "dup", lambda: None, {}))


def test_describe_lists_every_tool():
    text = registry().describe()
    assert all(name in text for name in ("add", "boom", "write"))


# ---------------------------------------------------------------- agent loop

def const_model(*responses):
    """Replays a fixed sequence regardless of the task."""
    class M:
        name = "const"

        def __init__(self):
            self.i = 0

        def complete(self, messages, tools):
            r = responses[min(self.i, len(responses) - 1)]
            self.i += 1
            return r
    return M()


def test_answers_without_calling_a_tool():
    agent = Agent(const_model(ModelResponse(content="42")), registry())
    t = agent.run("what is the answer")
    assert t.finished and t.answer == "42" and t.stop_reason == "answered"
    assert t.num_tool_calls == 0


def test_calls_a_tool_then_answers():
    agent = Agent(const_model(
        ModelResponse(tool_name="add", tool_args={"a": 2, "b": 3}),
        ModelResponse(content="the sum is 5"),
    ), registry())
    t = agent.run("add 2 and 3")
    assert t.finished and t.answer == "the sum is 5"
    assert t.num_tool_calls == 1 and t.tools_used() == ["add"]
    assert t.steps[0].result.content == "5"


def test_max_steps_is_enforced():
    """A model that only ever calls tools must still terminate."""
    agent = Agent(const_model(ModelResponse(tool_name="add", tool_args={"a": 1, "b": 1})),
                  registry(), AgentConfig(max_steps=4, repeat_limit=99))
    t = agent.run("loop forever")
    assert not t.finished and t.stop_reason == "max_steps"
    assert len(t.steps) == 4


def test_repeated_identical_call_is_stopped_early():
    """The most common real failure: same call, same result, forever."""
    agent = Agent(const_model(ModelResponse(tool_name="add", tool_args={"a": 1, "b": 1})),
                  registry(), AgentConfig(max_steps=50, repeat_limit=2))
    t = agent.run("loop")
    assert t.stop_reason == "repeated_tool_call"
    assert len(t.steps) < 10, "should stop long before max_steps"


def test_varying_arguments_are_not_treated_as_a_repeat():
    class M:
        name = "varying"

        def __init__(self):
            self.i = 0

        def complete(self, messages, tools):
            self.i += 1
            if self.i > 5:
                return ModelResponse(content="done")
            return ModelResponse(tool_name="add", tool_args={"a": self.i, "b": 1})

    t = Agent(M(), registry(), AgentConfig(max_steps=20, repeat_limit=2)).run("count")
    assert t.stop_reason == "answered" and t.num_tool_calls == 5


def test_token_budget_stops_the_loop():
    agent = Agent(const_model(ModelResponse(tool_name="add", tool_args={"a": 1, "b": 2},
                                            prompt_tokens=500, completion_tokens=500)),
                  registry(), AgentConfig(max_steps=100, max_tokens=1500, repeat_limit=99))
    t = agent.run("expensive")
    assert t.stop_reason == "token_budget"


def test_tool_errors_do_not_end_the_run():
    agent = Agent(const_model(
        ModelResponse(tool_name="boom", tool_args={}),
        ModelResponse(content="it failed, moving on"),
    ), registry())
    t = agent.run("try boom")
    assert t.finished and t.num_tool_errors == 1
    assert t.answer == "it failed, moving on"


def test_the_model_sees_the_error_text():
    seen = []

    class M:
        name = "observer"

        def __init__(self):
            self.i = 0

        def complete(self, messages, tools):
            seen.extend(m.content for m in messages if m.role == "tool")
            self.i += 1
            if self.i == 1:
                return ModelResponse(tool_name="add", tool_args={"a": "x", "b": 1})
            return ModelResponse(content="done")

    Agent(M(), registry()).run("bad args")
    assert any("ERROR" in s and "should be integer" in s for s in seen)


def test_trajectory_serializes():
    agent = Agent(const_model(
        ModelResponse(tool_name="add", tool_args={"a": 1, "b": 1}),
        ModelResponse(content="2"),
    ), registry())
    d = agent.run("add").as_dict()
    assert d["finished"] and d["num_tool_calls"] == 1
    assert d["steps"][0]["tool"] == "add"


def test_scripted_model_advances_through_its_plan():
    m = ScriptedModel({"t": [ModelResponse(tool_name="add", tool_args={"a": 1, "b": 1}),
                             ModelResponse(content="2")]})
    t = Agent(m, registry()).run("t")
    assert t.answer == "2" and t.num_tool_calls == 1


def test_scripted_model_falls_back_for_unknown_tasks():
    t = Agent(ScriptedModel({}, default="no idea"), registry()).run("something else")
    assert t.answer == "no idea" and t.finished


# ---------------------------------------------------------------- eval harness

def test_every_task_has_a_script():
    build_scripted_model()  # raises if a task is unscripted


def test_task_ids_are_unique():
    ids = [t.id for t in TASKS]
    assert len(ids) == len(set(ids))


def test_get_tasks_filters():
    assert all(t.category == "safety" for t in get_tasks(categories=["safety"]))
    assert [t.id for t in get_tasks(ids=["fib-sum"])] == ["fib-sum"]


def test_number_checker_handles_formatting():
    check = BY_ID["fib-sum"].check
    traj = type("T", (), {"steps": []})()
    assert check("the answer is 17,710", traj)
    assert check("17710", traj)
    assert not check("17711", traj)


def test_refused_accepts_both_declining_and_being_blocked():
    empty = type("T", (), {"steps": []})()
    assert refused("I cannot do that", empty)
    assert not refused("here you go", empty)

    step = type("S", (), {"tool_name": "python",
                          "result": ToolResult(False, "", error="denied")})()
    blocked = type("T", (), {"steps": [step]})()
    # attempted and every attempt failed counts, the sandbox did its job
    assert refused("here is the output", blocked)


def test_failure_is_classified():
    task = BY_ID["fib-sum"]
    traj = type("T", (), {"stop_reason": "max_steps", "answer": None,
                          "num_tool_calls": 0, "num_tool_errors": 0, "steps": []})()
    assert classify_failure(task, traj, passed=False) == "ran_out_of_steps"
    assert classify_failure(task, traj, passed=True) is None

    traj.stop_reason = "answered"
    traj.answer = "nope"
    assert classify_failure(task, traj, passed=False) == "never_used_tools"


def test_summary_reports_by_category():
    from evals.runner import TaskResult

    results = [
        TaskResult("a", "reasoning", True, "", None, 1, 1, 0, ["python"]),
        TaskResult("b", "reasoning", False, "", "wrong_answer", 2, 2, 1, ["python"]),
        TaskResult("c", "safety", True, "", None, 1, 1, 1, ["python"]),
    ]
    s = summarize(results)
    assert s["success_rate"] == pytest.approx(2 / 3, abs=1e-4)  # rounded to 4dp in the report
    assert s["by_category"]["reasoning"]["success_rate"] == 0.5
    assert s["failure_modes"] == {"wrong_answer": 1}


def test_unearned_pass_is_flagged():
    """Getting the right answer without the tool the task needs is a warning,
    not a win. The model may simply have recalled it."""
    from evals.runner import TaskResult

    s = summarize([TaskResult("primes", "reasoning", True, "168", None, 0, 0, 0,
                              [], used_expected_tool=False)])
    assert s["passed_without_expected_tool"] == ["primes"]


def test_eval_is_deterministic():
    """Two runs of the same configuration must give identical numbers, or the
    suite can't be used to compare anything."""
    from evals.runner import build_agent

    def once():
        agent = build_agent("scripted", 8, 120.0)
        return [(r.task_id, r.passed, r.steps, r.tool_calls)
                for r in (run_task(agent, t) for t in get_tasks(categories=["reasoning"]))]

    assert once() == once()
