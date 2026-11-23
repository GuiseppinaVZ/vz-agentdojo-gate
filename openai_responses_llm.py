"""
AgentDojo pipeline element that talks to OpenAI through the Responses API,
so models can use tools with reasoning switched on.

Reasoning items are kept (encrypted) and sent back on the next turn, so the
model keeps its own chain of thought across tool calls. Nothing is stored on
OpenAI's side (store=False).
"""

import json
import time
from collections.abc import Sequence

import openai

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string


def _tool_to_responses(f) -> dict:
    return {
        "type": "function",
        "name": f.name,
        "description": f.description,
        "parameters": f.parameters.model_json_schema(),
    }


def _messages_to_input(messages: Sequence[ChatMessage]) -> tuple[str | None, list]:
    instructions = None
    items: list = []
    for m in messages:
        role = m["role"]
        if role == "system":
            instructions = get_text_content_as_str(m["content"])
        elif role == "user":
            items.append({"role": "user", "content": get_text_content_as_str(m["content"])})
        elif role == "assistant":
            raw = m.get("vz_raw_output")
            if raw:  # replay exactly what the model produced (reasoning + calls + text)
                items.extend(raw)
            else:
                text = get_text_content_as_str(m["content"]) if m.get("content") else ""
                if text:
                    items.append({"role": "assistant", "content": text})
                for c in m.get("tool_calls") or []:
                    items.append({"type": "function_call", "call_id": c.id, "name": c.function,
                                  "arguments": json.dumps(c.args)})
        elif role == "tool":
            out = m.get("error") or get_text_content_as_str(m["content"])
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": out or ""})
    return instructions, items


class OpenAIResponsesLLM(BasePipelineElement):
    def __init__(self, client: openai.OpenAI, model: str, reasoning_effort: str = "medium") -> None:
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort

    def _create(self, **kw):
        for attempt in range(4):
            try:
                return self.client.responses.create(**kw)
            except (openai.BadRequestError, openai.UnprocessableEntityError):
                raise
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        instructions, items = _messages_to_input(messages)
        tools = [_tool_to_responses(f) for f in runtime.functions.values()]
        kw = dict(
            model=self.model,
            input=items,
            tools=tools,
            reasoning={"effort": self.reasoning_effort},
            store=False,
            include=["reasoning.encrypted_content"],
        )
        if instructions:
            kw["instructions"] = instructions
        resp = self._create(**kw)

        calls, texts, raw = [], [], []
        for item in resp.output:
            d = item.model_dump(exclude_none=True)
            d.pop("status", None)
            raw.append(d)
            if item.type == "function_call":
                calls.append(FunctionCall(function=item.name, args=json.loads(item.arguments or "{}"),
                                          id=item.call_id))
            elif item.type == "message":
                for part in item.content:
                    if getattr(part, "type", "") == "output_text":
                        texts.append(part.text)
        msg = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string("\n".join(texts))] if texts else None,
            tool_calls=calls or None,
        )
        msg["vz_raw_output"] = raw
        return query, runtime, env, [*messages, msg], extra_args
