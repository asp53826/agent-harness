"""Tool registry.

Tools are the agent's entire capability surface, so this is also the security
boundary. Two rules that come from that:

  - a tool declares its schema, and arguments are validated against it before
    the tool runs. The model produces arguments from text it read, and some of
    that text came from the internet.
  - a tool declares whether it mutates anything. A runtime can then refuse
    side-effecting tools in a dry run, and an eval can measure a read-only
    trajectory without touching the world.

Errors are returned as results, not raised. An agent that crashes on a bad tool
call learns nothing; an agent that sees "TypeError: expected int" can fix it on
the next step, which is most of what makes the loop work.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

JSON_TYPES = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}


@dataclass
class ToolResult:
    ok: bool
    content: str
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "content": self.content, "error": self.error,
                **({"metadata": self.metadata} if self.metadata else {})}


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable
    parameters: dict
    required: list[str] = field(default_factory=list)
    mutates: bool = False

    def schema(self) -> dict:
        """OpenAI-style function schema, which most providers now accept."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    def validate(self, args: dict) -> str | None:
        """Returns an error message, or None if the arguments are usable."""
        if not isinstance(args, dict):
            return f"arguments must be an object, got {type(args).__name__}"
        missing = [r for r in self.required if r not in args]
        if missing:
            return f"missing required argument(s): {', '.join(missing)}"
        unknown = [k for k in args if k not in self.parameters]
        if unknown:
            return (f"unknown argument(s): {', '.join(sorted(unknown))}. "
                    f"expected one of: {', '.join(sorted(self.parameters))}")
        for key, value in args.items():
            want = self.parameters[key].get("type")
            if want and not _type_ok(value, want):
                return f"argument {key!r} should be {want}, got {type(value).__name__}"
        return None

    def call(self, args: dict) -> ToolResult:
        err = self.validate(args)
        if err:
            return ToolResult(False, "", error=err)
        try:
            out = self.fn(**args)
        except Exception as e:
            # surfaced to the model rather than raised, so it can correct itself
            return ToolResult(False, "", error=f"{type(e).__name__}: {e}")
        if isinstance(out, ToolResult):
            return out
        return ToolResult(True, out if isinstance(out, str) else json.dumps(out, default=str))


def _type_ok(value: Any, want: str) -> bool:
    if want == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if want == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, {"string": str, "boolean": bool,
                              "array": list, "object": dict}.get(want, object))


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name} is already registered")
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str | None = None, description: str = "",
             mutates: bool = False, **param_docs):
        """Decorator that derives the schema from the signature.

        Keeping the schema next to the function means they can't drift, which
        they always do when the schema is written out separately by hand.
        """
        def deco(fn):
            sig = inspect.signature(fn)
            params, required = {}, []
            for pname, p in sig.parameters.items():
                jtype = JSON_TYPES.get(p.annotation, "string")
                params[pname] = {"type": jtype,
                                 "description": param_docs.get(pname, "")}
                if p.default is inspect.Parameter.empty:
                    required.append(pname)
            self.register(Tool(name or fn.__name__,
                               description or (fn.__doc__ or "").strip(),
                               fn, params, required, mutates))
            return fn
        return deco

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def call(self, name: str, args: dict, allow_mutations: bool = True) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                False, "",
                error=f"no tool named {name!r}. available: {', '.join(self.names)}",
            )
        if tool.mutates and not allow_mutations:
            return ToolResult(False, "", error=f"tool {name!r} mutates state and "
                                               "mutations are disabled for this run")
        return tool.call(args)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict]:
        return [self._tools[n].schema() for n in self.names]

    def describe(self) -> str:
        """Plain text listing, for models without native tool calling."""
        lines = []
        for name in self.names:
            t = self._tools[name]
            args = ", ".join(
                f"{k}: {v['type']}" + ("" if k in t.required else " (optional)")
                for k, v in t.parameters.items()
            )
            lines.append(f"- {name}({args}): {t.description}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def build_default_registry(sandbox=None, workspace: dict | None = None) -> ToolRegistry:
    """The standard toolset: run code, do arithmetic, read and write notes."""
    from .sandbox import Sandbox

    reg = ToolRegistry()
    sandbox = sandbox or Sandbox()
    workspace = workspace if workspace is not None else {}

    @reg.tool(
        description="Run a Python snippet in an isolated sandbox with no network "
                    "access and return whatever it prints. Use print() to return "
                    "a value, the last expression is not echoed.",
        code="the Python source to execute",
    )
    def python(code: str):
        r = sandbox.run_code(code)
        if r.timed_out:
            return ToolResult(False, r.stdout,
                              error=f"timed out after {sandbox.limits.wall_seconds}s")
        if not r.ok:
            return ToolResult(False, r.stdout, error=r.stderr[-2000:] or "non-zero exit")
        if not r.stdout.strip():
            return ToolResult(True, "", metadata={"note": "nothing was printed"})
        return ToolResult(True, r.stdout.strip(), metadata={"duration": r.duration})

    @reg.tool(
        description="Evaluate a arithmetic expression. Faster than python for "
                    "simple sums.",
        expression="an arithmetic expression, eg (17 * 34) + 9",
    )
    def calculator(expression: str):
        # ast.literal_eval won't do arithmetic, and eval on model output is
        # exactly the thing this repo is about, so it goes through the sandbox
        r = sandbox.run_code(f"print({expression})")
        if not r.ok:
            return ToolResult(False, "", error=(r.stderr.strip().splitlines() or ["invalid"])[-1])
        return ToolResult(True, r.stdout.strip())

    @reg.tool(description="Store a value in the scratchpad for later steps.",
              mutates=True, key="name to store under", value="the value")
    def remember(key: str, value: str):
        workspace[key] = value
        return f"stored {key!r}"

    @reg.tool(description="Read a value previously stored with remember.",
              key="the name to look up")
    def recall(key: str):
        if key not in workspace:
            return ToolResult(False, "", error=f"nothing stored under {key!r}. "
                                               f"keys: {', '.join(sorted(workspace)) or 'none'}")
        return workspace[key]

    return reg
