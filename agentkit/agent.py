"""The agent loop.

think -> call a tool -> observe -> repeat, until it answers or runs out of budget.

Everything that can run away is bounded, because an agent loop is an unbounded
loop with a language model in it:

  max_steps         stops a model that keeps calling tools forever
  max_tokens        stops one that is expensive rather than slow
  wall_seconds      stops one that is stuck behind a slow tool
  repeat detection  stops the specific failure where a model calls the same
                    tool with the same arguments over and over. Left alone it
                    burns the entire step budget making no progress, and it is
                    by far the most common way these loops fail.

Every step is recorded. A trajectory you can't inspect is one you can't debug,
and "the agent got it wrong" is not a bug report.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from .tools import ToolRegistry, ToolResult


@dataclass
class Message:
    role: str          # system | user | assistant | tool
    content: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_call_id: str | None = None

    def as_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.tool_name:
            d["tool_name"] = self.tool_name
            d["tool_args"] = self.tool_args
        return d


@dataclass
class ModelResponse:
    """Either a final answer or a tool call, never both."""
    content: str = ""
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def is_tool_call(self) -> bool:
        return self.tool_name is not None


class Model(Protocol):
    name: str

    def complete(self, messages: list[Message], tools: ToolRegistry) -> ModelResponse: ...


@dataclass
class Step:
    index: int
    thought: str = ""
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    result: ToolResult | None = None
    duration: float = 0.0

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "thought": self.thought,
            "tool": self.tool_name,
            "args": self.tool_args,
            "result": self.result.as_dict() if self.result else None,
            "duration": round(self.duration, 3),
        }


@dataclass
class Trajectory:
    task: str
    steps: list[Step] = field(default_factory=list)
    answer: str | None = None
    finished: bool = False
    stop_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0

    @property
    def num_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.tool_name)

    @property
    def num_tool_errors(self) -> int:
        return sum(1 for s in self.steps if s.result and not s.result.ok)

    def tools_used(self) -> list[str]:
        seen = []
        for s in self.steps:
            if s.tool_name and s.tool_name not in seen:
                seen.append(s.tool_name)
        return seen

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "answer": self.answer,
            "finished": self.finished,
            "stop_reason": self.stop_reason,
            "steps": [s.as_dict() for s in self.steps],
            "num_tool_calls": self.num_tool_calls,
            "num_tool_errors": self.num_tool_errors,
            "tools_used": self.tools_used(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_seconds": round(self.wall_seconds, 3),
        }


@dataclass
class AgentConfig:
    max_steps: int = 10
    max_tokens: int = 100_000
    wall_seconds: float = 120.0
    allow_mutations: bool = True
    # how many identical (tool, args) calls before the loop gives up on it
    repeat_limit: int = 3
    system_prompt: str = (
        "You are a careful assistant with tools. Work step by step. "
        "Call one tool at a time and use its result. "
        "When you know the answer, state it plainly and stop."
    )


class Agent:
    def __init__(self, model: Model, tools: ToolRegistry, config: AgentConfig | None = None):
        self.model = model
        self.tools = tools
        self.config = config or AgentConfig()

    def run(self, task: str) -> Trajectory:
        cfg = self.config
        traj = Trajectory(task=task)
        messages = [Message("system", cfg.system_prompt), Message("user", task)]
        seen: dict[str, int] = {}
        t0 = time.perf_counter()

        for i in range(cfg.max_steps):
            if time.perf_counter() - t0 > cfg.wall_seconds:
                traj.stop_reason = "wall_clock"
                break
            if traj.prompt_tokens + traj.completion_tokens > cfg.max_tokens:
                traj.stop_reason = "token_budget"
                break

            response = self.model.complete(messages, self.tools)
            traj.prompt_tokens += response.prompt_tokens
            traj.completion_tokens += response.completion_tokens

            if not response.is_tool_call:
                traj.answer = response.content
                traj.finished = True
                traj.stop_reason = "answered"
                break

            step = Step(index=i, thought=response.content,
                        tool_name=response.tool_name, tool_args=response.tool_args)

            key = f"{response.tool_name}:{json.dumps(response.tool_args, sort_keys=True, default=str)}"
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > cfg.repeat_limit:
                # tell the model why, rather than silently stopping. sometimes
                # it recovers by trying something else.
                step.result = ToolResult(
                    False, "",
                    error=(f"this exact call has been made {seen[key]} times and keeps "
                           f"producing the same result. try a different approach or "
                           f"answer with what you have."),
                )
                traj.steps.append(step)
                messages.append(Message("tool", step.result.error,
                                        tool_name=response.tool_name))
                traj.stop_reason = "repeated_tool_call"
                break

            s0 = time.perf_counter()
            step.result = self.tools.call(response.tool_name, response.tool_args,
                                          allow_mutations=cfg.allow_mutations)
            step.duration = time.perf_counter() - s0
            traj.steps.append(step)

            messages.append(Message("assistant", response.content,
                                    tool_name=response.tool_name,
                                    tool_args=response.tool_args))
            observation = (step.result.content if step.result.ok
                           else f"ERROR: {step.result.error}")
            messages.append(Message("tool", observation, tool_name=response.tool_name))
        else:
            traj.stop_reason = "max_steps"

        traj.wall_seconds = time.perf_counter() - t0
        return traj


class ScriptedModel:
    """A model whose behaviour is a lookup table.

    The eval harness has to run in CI with no API key and give the same answer
    every time, and an agent framework whose tests need a paid API is a
    framework nobody can contribute to. This plays fixed sequences so the
    *runtime* is what's under test.
    """

    name = "scripted"

    def __init__(self, script: dict[str, list[ModelResponse]], default: str = "I don't know."):
        self.script = script
        self.default = default
        self.calls = 0

    def complete(self, messages: list[Message], tools: ToolRegistry) -> ModelResponse:
        self.calls += 1
        task = next((m.content for m in messages if m.role == "user"), "")
        plan = self.script.get(task)
        if not plan:
            return ModelResponse(content=self.default, completion_tokens=5)
        # how many assistant turns have already happened tells us where we are
        step = sum(1 for m in messages if m.role == "assistant")
        if step >= len(plan):
            return ModelResponse(content=self.default, completion_tokens=5)
        return plan[step]


class OpenAIModel:
    """Any OpenAI-compatible chat completions endpoint, including local ones.

    Point base_url at vllm-lite or ollama and it works the same. Kept
    dependency-free on purpose (urllib, not the openai package) so the runtime
    doesn't drag in a client library.
    """

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 timeout: float = 60.0):
        import os

        self.name = model
        self.model = model
        self.base_url = (base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, messages: list[Message], tools: ToolRegistry) -> ModelResponse:
        import urllib.request

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [self._render(m) for m in messages],
            "tools": tools.schemas(),
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json",
                     "authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.load(r)

        choice = data["choices"][0]["message"]
        usage = data.get("usage", {})
        calls = choice.get("tool_calls") or []
        if calls:
            fn = calls[0]["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            return ModelResponse(
                content=choice.get("content") or "",
                tool_name=fn["name"], tool_args=args,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
        return ModelResponse(
            content=choice.get("content") or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    @staticmethod
    def _render(m: Message) -> dict:
        if m.role == "tool":
            return {"role": "user", "content": f"[{m.tool_name} result]\n{m.content}"}
        if m.role == "assistant" and m.tool_name:
            return {"role": "assistant",
                    "content": f"{m.content}\ncalling {m.tool_name}({json.dumps(m.tool_args)})"}
        return {"role": m.role, "content": m.content}
