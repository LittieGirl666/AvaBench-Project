from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .agent_environment import EnvironmentCall, EnvironmentSession
from .types import JsonObject, V2QueryCase


FINISH_TOOL = "harness_finish"
UNAVAILABLE_TOOL = "harness_unavailable"
RESTART_TOOL = "harness_restart"
CONTROL_TOOL_IDS = frozenset({FINISH_TOOL, UNAVAILABLE_TOOL, RESTART_TOOL})


def control_tools() -> tuple[JsonObject, ...]:
    return (
        {
            "type": "function",
            "function": {
                "name": FINISH_TOOL,
                "description": "Finish with evidence from one successful environment call in the current restart epoch.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "evidence_call_id": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["evidence_call_id", "summary"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": UNAVAILABLE_TOOL,
                "description": "Stop because no complete path through the visible environment operations can satisfy the request.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": RESTART_TOOL,
                "description": "Restore the case environment to its initial state and begin a new execution epoch.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        },
    )


@dataclass(frozen=True)
class HarnessConfig:
    max_model_turns: int = 8
    max_environment_calls: int = 6
    max_restarts: int = 1
    provider_retries: int = 2
    temperature: float = 0.0
    model_seed: int | None = 0


@dataclass(frozen=True)
class ModelTurn:
    response_id: str | None
    assistant_text: str | None
    reasoning_items: tuple[Any, ...]
    tool_calls: tuple[JsonObject, ...]
    usage: JsonObject | None
    assistant_message: JsonObject


class ModelDriver(Protocol):
    model: str

    def generate(
        self,
        *,
        messages: list[JsonObject],
        tools: tuple[JsonObject, ...],
        config: HarnessConfig,
    ) -> ModelTurn: ...


class ChatCompletionsDriver:
    """OpenAI-compatible Chat Completions adapter."""

    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        api_key: str,
        timeout: float = 180.0,
        opener: urllib.request.OpenerDirector | None = None,
        reasoning_effort: str | None = None,
        extra_body: JsonObject | None = None,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.opener = opener or urllib.request.build_opener()
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body or {}

    def generate(
        self,
        *,
        messages: list[JsonObject],
        tools: tuple[JsonObject, ...],
        config: HarnessConfig,
    ) -> ModelTurn:
        system_parts = [item["content"] for item in messages if item.get("role") in {"system", "developer"}]
        chat_messages = [item for item in messages if item.get("role") not in {"system", "developer"}]
        if system_parts:
            chat_messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        body: JsonObject = {
            "model": self.model,
            "messages": chat_messages,
            "tools": list(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": config.temperature,
            "stream": False,
        }
        if config.model_seed is not None:
            body["seed"] = config.model_seed
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        body.update(self.extra_body)
        payload = json.dumps(body, ensure_ascii=False).encode()
        provider: JsonObject | None = None
        for attempt in range(config.provider_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                payload,
                {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    provider = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:2000]
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                if attempt == config.provider_retries or not retryable:
                    raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == config.provider_retries:
                    raise RuntimeError(f"network error: {exc}") from exc
            time.sleep(min(2**attempt, 30))
        if provider is None:
            raise RuntimeError("provider returned no response")
        choices = provider.get("choices", [])
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise RuntimeError("provider returned no assistant message")
        message = choices[0]["message"]
        calls = tuple(
            {
                "id": item.get("id"),
                "type": "function",
                "function": {
                    "name": item.get("function", {}).get("name"),
                    "arguments": item.get("function", {}).get("arguments"),
                },
            }
            for item in message.get("tool_calls", [])
            if item.get("type") == "function" and isinstance(item.get("function"), dict)
        )
        reasoning: list[Any] = []
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            if message.get(key) is not None:
                reasoning.append({"field": key, "value": copy.deepcopy(message[key])})
        if provider.get("reasoning") is not None:
            reasoning.append({"field": "response.reasoning", "value": copy.deepcopy(provider["reasoning"])})
        content = message.get("content")
        history_message: JsonObject = {"role": "assistant", "content": content, "tool_calls": list(calls)}
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            if message.get(key) is not None:
                history_message[key] = copy.deepcopy(message[key])
        return ModelTurn(
            response_id=provider.get("id"),
            assistant_text=content if isinstance(content, str) else None,
            reasoning_items=tuple(reasoning),
            tool_calls=calls,
            usage=provider.get("usage") if isinstance(provider.get("usage"), dict) else None,
            assistant_message=history_message,
        )


@dataclass
class HarnessResult:
    termination_reason: str
    events: list[JsonObject]
    model_turns: int
    environment_call_count: int
    restart_count: int
    usage: JsonObject
    evidence_call: EnvironmentCall | None
    unavailable_reason: str | None
    infrastructure_error: str | None
    initial_state: Any
    final_state: Any
    initial_digest: str
    final_digest: str
    final_runtime_digest: str
    rollback: JsonObject
    calls: tuple[EnvironmentCall, ...]


def _parse_object(value: Any) -> JsonObject:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("arguments must be a JSON object")
    return value


def _remaining(config: HarnessConfig, *, turns: int, calls: int, restarts: int) -> JsonObject:
    return {
        "model_turns": max(config.max_model_turns - turns, 0),
        "environment_calls": max(config.max_environment_calls - calls, 0),
        "restarts": max(config.max_restarts - restarts, 0),
    }


def _merge_usage(total: JsonObject, usage: JsonObject | None) -> None:
    for key, value in (usage or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value


class AgentHarness:
    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or HarnessConfig()

    def run(
        self,
        *,
        case: V2QueryCase,
        driver: ModelDriver,
        messages: list[JsonObject],
        environment_tools: tuple[JsonObject, ...],
        session: EnvironmentSession,
    ) -> HarnessResult:
        conversation = copy.deepcopy(messages)
        tools = tuple(environment_tools) + control_tools()
        events: list[JsonObject] = []
        usage: JsonObject = {}
        model_turns = 0
        environment_calls = 0
        restarts = 0
        evidence: EnvironmentCall | None = None
        unavailable_reason = None
        infrastructure_error = None
        termination_reason = "budget_exhausted"

        try:
            while model_turns < self.config.max_model_turns:
                try:
                    turn = driver.generate(messages=conversation, tools=tools, config=self.config)
                except Exception as exc:
                    infrastructure_error = str(exc)
                    termination_reason = "infrastructure_error"
                    events.append({"type": "infrastructure_error", "error": str(exc)})
                    break
                model_turns += 1
                _merge_usage(usage, turn.usage)
                event: JsonObject = {
                    "type": "model_turn",
                    "turn": model_turns,
                    "provider_response_id": turn.response_id,
                    "assistant_text": turn.assistant_text,
                    "reasoning_items": list(turn.reasoning_items),
                    "tool_calls": copy.deepcopy(list(turn.tool_calls)),
                    "usage": copy.deepcopy(turn.usage),
                }
                events.append(event)

                calls = list(turn.tool_calls)
                if len(calls) != 1 or not calls[0].get("id"):
                    category = "no_tool_call" if not calls else "multiple_tool_calls" if len(calls) > 1 else "missing_tool_call_id"
                    observation = {
                        "status": "protocol_error",
                        "error_category": category,
                        "message": "Return exactly one native environment or harness control tool call.",
                        "remaining_budget": _remaining(
                            self.config, turns=model_turns, calls=environment_calls, restarts=restarts
                        ),
                    }
                    events.append({"type": "protocol_error", "turn": model_turns, **observation})
                    if calls and all(call.get("id") for call in calls):
                        conversation.append(turn.assistant_message)
                        for call in calls:
                            conversation.append({
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":")),
                            })
                    else:
                        conversation.append({"role": "assistant", "content": turn.assistant_text})
                        conversation.append({
                            "role": "user",
                            "content": "[harness_observation] " + json.dumps(observation, ensure_ascii=False, separators=(",", ":")),
                        })
                    continue

                call = calls[0]
                function = call.get("function", {})
                name = function.get("name")
                raw_arguments = function.get("arguments")
                conversation.append(turn.assistant_message)

                if name not in session.visible_operation_ids and name not in CONTROL_TOOL_IDS:
                    observation = {
                        "status": "protocol_error",
                        "error_category": "unavailable_tool_call",
                        "message": f"Tool {name!r} is not available.",
                        "remaining_budget": _remaining(
                            self.config, turns=model_turns, calls=environment_calls, restarts=restarts
                        ),
                    }
                    events.append({"type": "protocol_error", "turn": model_turns, **observation})
                    conversation.append({
                        "role": "tool", "tool_call_id": call["id"],
                        "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":")),
                    })
                    continue

                if name == FINISH_TOOL:
                    try:
                        arguments = _parse_object(raw_arguments)
                        evidence_id = arguments.get("evidence_call_id")
                        if set(arguments) != {"evidence_call_id", "summary"} or not isinstance(evidence_id, str) or not isinstance(arguments.get("summary"), str):
                            raise ValueError("finish requires string evidence_call_id and summary")
                        evidence = session.evidence(evidence_id)
                        if evidence is None:
                            raise ValueError("evidence_call_id must reference a successful call in the current epoch")
                    except Exception as exc:
                        observation = {
                            "status": "protocol_error", "error_category": "invalid_finish", "message": str(exc),
                            "remaining_budget": _remaining(self.config, turns=model_turns, calls=environment_calls, restarts=restarts),
                        }
                        events.append({"type": "control_error", "turn": model_turns, "control": FINISH_TOOL, **observation})
                        conversation.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":"))})
                        continue
                    termination_reason = "finish"
                    events.append({"type": "control", "turn": model_turns, "control": FINISH_TOOL, "arguments": arguments})
                    break

                if name == UNAVAILABLE_TOOL:
                    try:
                        arguments = _parse_object(raw_arguments)
                        reason = arguments.get("reason")
                        if set(arguments) != {"reason"} or not isinstance(reason, str) or not reason.strip():
                            raise ValueError("unavailable requires a non-empty reason")
                    except Exception as exc:
                        observation = {
                            "status": "protocol_error", "error_category": "invalid_unavailable", "message": str(exc),
                            "remaining_budget": _remaining(self.config, turns=model_turns, calls=environment_calls, restarts=restarts),
                        }
                        events.append({"type": "control_error", "turn": model_turns, "control": UNAVAILABLE_TOOL, **observation})
                        conversation.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":"))})
                        continue
                    unavailable_reason = reason
                    termination_reason = "unavailable"
                    events.append({"type": "control", "turn": model_turns, "control": UNAVAILABLE_TOOL, "arguments": arguments})
                    break

                if name == RESTART_TOOL:
                    try:
                        arguments = _parse_object(raw_arguments)
                        if set(arguments) != {"reason"} or not isinstance(arguments.get("reason"), str) or not arguments["reason"].strip():
                            raise ValueError("restart requires a non-empty reason")
                        if restarts >= self.config.max_restarts:
                            raise ValueError("restart budget exhausted")
                    except Exception as exc:
                        observation = {
                            "status": "protocol_error", "error_category": "invalid_restart", "message": str(exc),
                            "remaining_budget": _remaining(self.config, turns=model_turns, calls=environment_calls, restarts=restarts),
                        }
                        events.append({"type": "control_error", "turn": model_turns, "control": RESTART_TOOL, **observation})
                        conversation.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":"))})
                        continue
                    restarts += 1
                    restart = session.restart()
                    observation = {
                        "status": "restarted", "epoch": restart["epoch"],
                        "remaining_budget": _remaining(self.config, turns=model_turns, calls=environment_calls, restarts=restarts),
                    }
                    events.append({"type": "control", "turn": model_turns, "control": RESTART_TOOL, "arguments": arguments, "result": restart})
                    conversation.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":"))})
                    continue

                if environment_calls >= self.config.max_environment_calls:
                    observation = {
                        "status": "budget_error", "error_category": "environment_call_budget_exhausted",
                        "remaining_budget": _remaining(self.config, turns=model_turns, calls=environment_calls, restarts=restarts),
                    }
                    events.append({"type": "budget_error", "turn": model_turns, "operation_id": name, **observation})
                    conversation.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":"))})
                    continue

                environment_calls += 1
                executed = session.execute(call_id=call["id"], operation_id=name, arguments=raw_arguments)
                observation = {
                    "status": "success" if executed.success else "error",
                    "call_id": executed.call_id,
                    "operation_id": executed.operation_id,
                    "output": copy.deepcopy(executed.output) if executed.success else None,
                    "error_category": executed.error_category,
                    "error": executed.error,
                    "remaining_budget": _remaining(
                        self.config, turns=model_turns, calls=environment_calls, restarts=restarts
                    ),
                }
                if executed.success:
                    observation["payload_ref"] = {
                        "ref": executed.payload_transport["issued_ref"],
                    }
                    if executed.changed_resources:
                        observation["state_delta"] = {
                            "changed_resources": list(executed.changed_resources),
                            "state_digest_before": executed.state_digest_before,
                            "state_digest_after": executed.state_digest_after,
                        }
                events.append({"type": "environment_call", "turn": model_turns, **executed.to_dict(), "observation": observation})
                conversation.append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":")),
                })
        finally:
            finalized = session.finalize()

        return HarnessResult(
            termination_reason=termination_reason,
            events=events,
            model_turns=model_turns,
            environment_call_count=environment_calls,
            restart_count=restarts,
            usage=usage,
            evidence_call=evidence,
            unavailable_reason=unavailable_reason,
            infrastructure_error=infrastructure_error,
            initial_state=finalized["initial_state"],
            final_state=finalized["final_state"],
            initial_digest=finalized["initial_digest"],
            final_digest=finalized["final_digest"],
            final_runtime_digest=finalized["final_runtime_digest"],
            rollback=finalized["rollback"],
            calls=finalized["calls"],
        )
