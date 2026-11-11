"""Deterministic stand-in for the closure agent, used by the CPU tests. Reads the snapshot from the user message
and settles it correctly: verify then confirm a hold that a queued confirm names, verify then cancel any other
hold, verify then pay a queued payment, send a queued reply, then report. One call per step."""
from __future__ import annotations

import json

import yaml

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, FunctionCall
from agentdojo.types import ChatAssistantMessage, get_text_content_as_str, text_content_block_from_string

from budget_runtime import Ledger, TokenCounter, messages_to_openai


class ScriptedClosureAgent(BasePipelineElement):
    def __init__(self, counter: TokenCounter, presets: dict, sloppy: bool = False):
        self.counter, self.p, self.sloppy = counter, presets, sloppy
        self.name = "scripted-closure"

    def _plan(self, sigma: dict) -> list[FunctionCall]:
        calls = []
        confirm_for = {c["args"].get("hold_id"): c["args"] for c in sigma["queued"] if c["function"] == "confirm_booking"}
        for h in sigma["pending"]:
            hid = h["hold_id"]
            calls.append(FunctionCall(function="verify_hold", args={"hold_id": hid}))
            if hid in confirm_for:
                calls.append(FunctionCall(function="confirm_booking", args={"hold_id": hid, "key": confirm_for[hid].get("key", f"{hid}-confirm")}))
            elif not self.sloppy:
                calls.append(FunctionCall(function="cancel_hold", args={"hold_id": hid, "key": f"{hid}-cancel"}))
        for c in sigma["queued"]:
            if c["function"] == "send_money":
                calls.append(FunctionCall(function="verify_payee", args={"name": c["args"].get("payee_name", "")}))
                calls.append(FunctionCall(function="send_money", args={k: v for k, v in c["args"].items() if k != "payee_name"}))
            elif c["function"] == "reply_email":
                calls.append(FunctionCall(function="reply_email", args=dict(c["args"])))
            elif c["function"] == "cancel_hold":
                calls.append(FunctionCall(function="cancel_hold", args=dict(c["args"])))
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
            text, calls = "Report: every open hold is settled and the queued actions that could be verified were executed.", []
        ledger.charge_call(prompt_tokens, self.counter.count(text) + 8)
        out = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(text)], tool_calls=calls)
        return query, runtime, env, [*messages, out], extra_args
