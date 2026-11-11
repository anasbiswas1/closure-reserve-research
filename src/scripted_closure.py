"""Deterministic stand-in for the closure agent, used by the CPU tests. Reads the snapshot from the user message,
verifies every open hold and cancels it, then reports. One call per step."""
from __future__ import annotations

import json

import yaml

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, FunctionCall
from agentdojo.types import ChatAssistantMessage, get_text_content_as_str, text_content_block_from_string

from budget_runtime import Ledger, TokenCounter, messages_to_openai


class ScriptedClosureAgent(BasePipelineElement):
    def __init__(self, counter: TokenCounter, presets: dict, confirm_first: bool = False):
        self.counter, self.p, self.confirm_first = counter, presets, confirm_first
        self.name = "scripted-closure"

    def _plan(self, sigma: dict) -> list[FunctionCall]:
        calls = []
        for i, h in enumerate(sigma["pending"]):
            hid = h["hold_id"]
            calls.append(FunctionCall(function="verify_hold", args={"hold_id": hid}))
            if self.confirm_first and i == 0:
                calls.append(FunctionCall(function="confirm_booking", args={"hold_id": hid, "key": f"{hid}-closure-confirm"}))
            else:
                calls.append(FunctionCall(function="cancel_hold", args={"hold_id": hid, "key": f"{hid}-closure-cancel"}))
        return calls

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        st = extra_args.setdefault("closure_script", {"i": 0, "plan": None})
        if st["plan"] is None:
            user = next(m for m in messages if m["role"] == "user")
            sigma = yaml.safe_load(get_text_content_as_str(user["content"]))
            st["plan"] = self._plan(sigma)
        ledger: Ledger = extra_args["ledger"]
        prompt_tokens = sum(self.counter.count(m["content"]) for m in messages_to_openai(messages, runtime))
        if st["i"] < len(st["plan"]):
            c = st["plan"][st["i"]]; st["i"] += 1
            c = FunctionCall(function=c.function, args=dict(c.args), id=f"c{st['i']}")
            text, calls = f"<function={c.function}>{json.dumps(c.args)}</function>", [c]
        else:
            text, calls = "Report: every open hold is settled; nothing else was pending.", []
        ledger.charge_call(prompt_tokens, self.counter.count(text) + 8)
        out = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(text)], tool_calls=calls)
        return query, runtime, env, [*messages, out], extra_args
