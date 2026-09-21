from __future__ import annotations

import copy
import json
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from .agent_harness import HarnessConfig, ModelTurn
from .deepinfer import (
    _merge_platform_messages,
    _reasoning_locations,
    _request_json,
    split_embedded_thinking,
)
from .types import JsonObject


SILICONFLOW_ADAPTER_VERSION = "siliconflow-chat.v3"
SILICONFLOW_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_DEFAULT_MAX_TOKENS = 16_384
SILICONFLOW_DEFAULT_THINKING_BUDGET = 16_384
SILICONFLOW_DEFAULT_PROVIDER_RETRIES = 5
SILICONFLOW_DEFAULT_RETRY_BACKOFF_BASE = 5.0
SILICONFLOW_DEFAULT_RETRY_BACKOFF_MAX = 60.0
SILICONFLOW_THINKING_MODES = ("disabled", "enabled")


class SiliconFlowCapabilityError(RuntimeError):
    """The route responded but did not satisfy the harness protocol."""


@dataclass(frozen=True)
class SiliconFlowModelProfile:
    model_id: str
    route: str
    disabled_temperature: float
    enabled_temperature: float
    thinking_modes: tuple[str, ...] = SILICONFLOW_THINKING_MODES
    max_tokens: int = SILICONFLOW_DEFAULT_MAX_TOKENS
    thinking_budget: int = SILICONFLOW_DEFAULT_THINKING_BUDGET

    def temperature(self, thinking_mode: str) -> float:
        if thinking_mode not in self.thinking_modes:
            raise ValueError(
                f"{self.model_id} does not declare SiliconFlow thinking mode "
                f"{thinking_mode!r}"
            )
        return (
            self.enabled_temperature
            if thinking_mode == "enabled"
            else self.disabled_temperature
        )

    def to_dict(self) -> JsonObject:
        result = asdict(self)
        result["thinking_modes"] = list(self.thinking_modes)
        return result


SILICONFLOW_MODEL_PROFILES: dict[str, SiliconFlowModelProfile] = {
    profile.model_id: profile
    for profile in (
        SiliconFlowModelProfile(
            "kimi-k2.6",
            "Pro/moonshotai/Kimi-K2.6",
            0.6,
            1.0,
        ),
        SiliconFlowModelProfile(
            "qwen3.5-27b",
            "Qwen/Qwen3.5-27B",
            0.7,
            0.7,
        ),
        SiliconFlowModelProfile(
            "minimax-m2.5",
            "MiniMaxAI/MiniMax-M2.5",
            1.0,
            1.0,
        ),
    )
}
# MiniMax remains resolvable so historical cells can still be reproduced, but
# new campaigns target Kimi and Qwen 3.5 by default.
SILICONFLOW_DEFAULT_MODELS = ("kimi-k2.6", "qwen3.5-27b")


def get_siliconflow_profile(model: str) -> SiliconFlowModelProfile:
    for profile in SILICONFLOW_MODEL_PROFILES.values():
        if model in {profile.model_id, profile.route}:
            return profile
    supported = ", ".join(
        f"{profile.model_id} ({profile.route})"
        for profile in SILICONFLOW_MODEL_PROFILES.values()
    )
    raise ValueError(
        f"unknown SiliconFlow model {model!r}; supported models: {supported}"
    )


def siliconflow_profile_manifest() -> JsonObject:
    return {
        "adapter_version": SILICONFLOW_ADAPTER_VERSION,
        "default_models": list(SILICONFLOW_DEFAULT_MODELS),
        "models": [
            profile.to_dict()
            for profile in SILICONFLOW_MODEL_PROFILES.values()
        ],
    }


def siliconflow_chat_completions_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if not url.endswith("/v1"):
        url += "/v1"
    return url + "/chat/completions"


class SiliconFlowChatDriver:
    """SiliconFlow's documented OpenAI-compatible Chat Completions protocol."""

    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        api_key: str,
        thinking_mode: str,
        max_tokens: int = SILICONFLOW_DEFAULT_MAX_TOKENS,
        thinking_budget: int = SILICONFLOW_DEFAULT_THINKING_BUDGET,
        timeout: float = 180.0,
        opener: urllib.request.OpenerDirector | None = None,
        provider_retries: int = SILICONFLOW_DEFAULT_PROVIDER_RETRIES,
        retry_backoff_base: float = SILICONFLOW_DEFAULT_RETRY_BACKOFF_BASE,
        retry_backoff_max: float = SILICONFLOW_DEFAULT_RETRY_BACKOFF_MAX,
    ) -> None:
        profile = get_siliconflow_profile(model)
        if thinking_mode not in SILICONFLOW_THINKING_MODES:
            raise ValueError(
                "SiliconFlow thinking mode must be 'disabled' or 'enabled'"
            )
        if max_tokens < 1:
            raise ValueError("SiliconFlow max_tokens must be positive")
        if not 128 <= thinking_budget <= 32_768:
            raise ValueError(
                "SiliconFlow thinking_budget must be between 128 and 32768"
            )
        if provider_retries < 0:
            raise ValueError("SiliconFlow provider_retries must be non-negative")
        if retry_backoff_base < 0 or retry_backoff_max < retry_backoff_base:
            raise ValueError(
                "SiliconFlow retry backoff requires 0 <= base <= max"
            )
        self.profile = profile
        self.model = profile.route
        self.endpoint = endpoint
        self.api_key = api_key
        self.thinking_mode = thinking_mode
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.timeout = timeout
        self.opener = opener or urllib.request.build_opener()
        self.provider_retries = provider_retries
        self.retry_backoff_base = retry_backoff_base
        self.retry_backoff_max = retry_backoff_max

    def request_body(
        self,
        *,
        messages: list[JsonObject],
        tools: tuple[JsonObject, ...],
        config: HarnessConfig,
    ) -> JsonObject:
        return {
            "model": self.model,
            "messages": _merge_platform_messages(messages),
            "tools": copy.deepcopy(list(tools)),
            "tool_choice": "auto",
            "enable_thinking": self.thinking_mode == "enabled",
            "thinking_budget": self.thinking_budget,
            "temperature": config.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

    def generate(
        self,
        *,
        messages: list[JsonObject],
        tools: tuple[JsonObject, ...],
        config: HarnessConfig,
    ) -> ModelTurn:
        body = self.request_body(messages=messages, tools=tools, config=config)
        request = urllib.request.Request(
            self.endpoint,
            json.dumps(body, ensure_ascii=False).encode(),
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        provider = _request_json(
            opener=self.opener,
            request=request,
            timeout=self.timeout,
            retries=self.provider_retries,
            retry_backoff_base=self.retry_backoff_base,
            retry_backoff_max=self.retry_backoff_max,
            provider_name="SiliconFlow",
        )
        choices = provider.get("choices", [])
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise RuntimeError("SiliconFlow returned no assistant message")
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
            if item.get("type") == "function"
            and isinstance(item.get("function"), dict)
        )
        reasoning: list[Any] = []
        for key in ("reasoning_content", "reasoning", "reasoning_details"):
            if message.get(key) is not None:
                reasoning.append(
                    {"field": key, "value": copy.deepcopy(message[key])}
                )
        content = message.get("content")
        visible_content = content if isinstance(content, str) else None
        if isinstance(content, str):
            visible_content, embedded = split_embedded_thinking(content)
            if embedded:
                reasoning.append(
                    {"field": "content.thinking_tags", "value": embedded}
                )
        history_message = copy.deepcopy(message)
        history_message.setdefault("role", "assistant")
        return ModelTurn(
            response_id=provider.get("id"),
            assistant_text=visible_content,
            reasoning_items=tuple(reasoning),
            tool_calls=calls,
            usage=(
                provider.get("usage")
                if isinstance(provider.get("usage"), dict)
                else None
            ),
            assistant_message=history_message,
        )


def probe_siliconflow_capability(
    *,
    model: str,
    base_url: str,
    api_key: str,
    thinking_mode: str,
    opener: urllib.request.OpenerDirector,
    timeout: float = 60.0,
    retries: int = 1,
    retry_backoff_base: float = SILICONFLOW_DEFAULT_RETRY_BACKOFF_BASE,
    retry_backoff_max: float = SILICONFLOW_DEFAULT_RETRY_BACKOFF_MAX,
) -> JsonObject:
    """Verify one tool call and one tool-history continuation before a campaign."""

    profile = get_siliconflow_profile(model)
    tool = {
        "type": "function",
        "function": {
            "name": "siliconflow_capability_probe",
            "description": "Return the supplied probe token.",
            "parameters": {
                "type": "object",
                "properties": {"token": {"type": "string"}},
                "required": ["token"],
                "additionalProperties": False,
            },
        },
    }
    first_body: JsonObject = {
        "model": profile.route,
        "messages": [
            {
                "role": "system",
                "content": "Call the available probe tool exactly once.",
            },
            {
                "role": "user",
                "content": (
                    "Call siliconflow_capability_probe with token 'ok'."
                ),
            },
        ],
        "tools": [tool],
        "tool_choice": "auto",
        "enable_thinking": thinking_mode == "enabled",
        "thinking_budget": profile.thinking_budget,
        "temperature": profile.temperature(thinking_mode),
        "max_tokens": profile.max_tokens,
        "stream": False,
    }
    endpoint = siliconflow_chat_completions_url(base_url)

    def post(body: JsonObject) -> JsonObject:
        request = urllib.request.Request(
            endpoint,
            json.dumps(body, ensure_ascii=False).encode(),
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        return _request_json(
            opener=opener,
            request=request,
            timeout=timeout,
            retries=retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
            provider_name="SiliconFlow",
        )

    first = post(first_body)
    first_choices = first.get("choices", [])
    if not first_choices or not isinstance(
        first_choices[0].get("message"), dict
    ):
        raise SiliconFlowCapabilityError(
            "SiliconFlow preflight returned no first assistant message"
        )
    assistant = first_choices[0]["message"]
    calls = assistant.get("tool_calls", [])
    if (
        len(calls) != 1
        or not calls[0].get("id")
        or calls[0].get("function", {}).get("name")
        != "siliconflow_capability_probe"
    ):
        raise SiliconFlowCapabilityError(
            "SiliconFlow preflight did not return exactly one valid tool call"
        )
    arguments = calls[0].get("function", {}).get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise SiliconFlowCapabilityError(
                "SiliconFlow preflight returned invalid JSON tool arguments"
            ) from exc
    if not isinstance(arguments, dict):
        raise SiliconFlowCapabilityError(
            "SiliconFlow preflight returned non-object tool arguments"
        )

    second_body = {
        **first_body,
        "messages": [
            *first_body["messages"],
            copy.deepcopy(assistant),
            {
                "role": "tool",
                "tool_call_id": calls[0]["id"],
                "content": json.dumps(
                    {"status": "success", "token": "ok"},
                    separators=(",", ":"),
                ),
            },
        ],
        "tool_choice": "none",
    }
    second = post(second_body)
    second_choices = second.get("choices", [])
    if not second_choices or not isinstance(
        second_choices[0].get("message"), dict
    ):
        raise SiliconFlowCapabilityError(
            "SiliconFlow preflight returned no follow-up message"
        )

    first_locations = _reasoning_locations(assistant)
    second_locations = _reasoning_locations(second_choices[0]["message"])
    reasoning_observed = bool(first_locations or second_locations)
    mode_respected = not (
        thinking_mode == "disabled" and reasoning_observed
    )
    return {
        "status": "passed" if mode_respected else "incompatible",
        "adapter_version": SILICONFLOW_ADAPTER_VERSION,
        "model": profile.model_id,
        "requested_route": profile.route,
        "thinking_mode": thinking_mode,
        "tool_call": True,
        "multi_turn_tool_history": True,
        "reasoning_observed": reasoning_observed,
        "reasoning_locations": {
            "tool_call_turn": first_locations,
            "follow_up_turn": second_locations,
        },
        "mode_respected": mode_respected,
        "effective_models": [
            value
            for value in (first.get("model"), second.get("model"))
            if isinstance(value, str) and value
        ],
        "temperature": profile.temperature(thinking_mode),
        "max_tokens": profile.max_tokens,
        "thinking_budget": profile.thinking_budget,
        "reason": (
            None
            if mode_respected
            else "reasoning was observed while thinking was disabled"
        ),
    }
