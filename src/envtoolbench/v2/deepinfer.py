from __future__ import annotations

import copy
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from .agent_harness import HarnessConfig, ModelTurn
from .types import JsonObject


DEEPINFER_ADAPTER_VERSION = "deepinfer-chat.v1"
DEEPINFER_DEFAULT_MAX_TOKENS = 16_384
DEEPINFER_DEFAULT_MODELS = (
    "deepseek-v4-flash",
    "glm-5.2",
    "minimax-m2.7",
    "minimax-m3",
    "kimi-k2.6",
)
DEEPINFER_THINKING_MODES = ("disabled", "enabled")


class DeepInferCapabilityError(RuntimeError):
    """The route responded but did not satisfy the harness protocol."""


@dataclass(frozen=True)
class DeepInferModelProfile:
    model_id: str
    disabled_temperature: float
    enabled_temperature: float
    thinking_modes: tuple[str, ...] = DEEPINFER_THINKING_MODES
    max_tokens: int = DEEPINFER_DEFAULT_MAX_TOKENS

    def temperature(self, thinking_mode: str) -> float:
        if thinking_mode not in self.thinking_modes:
            raise ValueError(
                f"{self.model_id} does not declare DeepInfer thinking mode "
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


DEEPINFER_MODEL_PROFILES: dict[str, DeepInferModelProfile] = {
    profile.model_id: profile
    for profile in (
        DeepInferModelProfile("deepseek-v4-flash", 0.0, 0.0),
        DeepInferModelProfile("glm-5.2", 0.0, 0.0),
        DeepInferModelProfile("minimax-m2.7", 1.0, 1.0),
        DeepInferModelProfile("minimax-m3", 1.0, 1.0),
        DeepInferModelProfile("kimi-k2.6", 0.6, 1.0),
    )
}


def get_deepinfer_profile(model_id: str) -> DeepInferModelProfile:
    try:
        return DEEPINFER_MODEL_PROFILES[model_id]
    except KeyError as exc:
        supported = ", ".join(DEEPINFER_MODEL_PROFILES)
        raise ValueError(
            f"unknown DeepInfer model {model_id!r}; supported models: {supported}"
        ) from exc


def deepinfer_profile_manifest() -> JsonObject:
    return {
        "adapter_version": DEEPINFER_ADAPTER_VERSION,
        "models": [
            DEEPINFER_MODEL_PROFILES[model_id].to_dict()
            for model_id in DEEPINFER_DEFAULT_MODELS
        ],
    }


def deepinfer_chat_completions_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if not url.endswith("/v1"):
        url += "/v1"
    return url + "/chat/completions"


def deepinfer_models_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    if not url.endswith("/v1"):
        url += "/v1"
    return url + "/models"


def _merge_platform_messages(messages: list[JsonObject]) -> list[JsonObject]:
    system_parts = [
        str(item["content"])
        for item in messages
        if item.get("role") in {"system", "developer"}
    ]
    chat_messages = [
        copy.deepcopy(item)
        for item in messages
        if item.get("role") not in {"system", "developer"}
    ]
    if system_parts:
        chat_messages.insert(
            0,
            {"role": "system", "content": "\n\n".join(system_parts)},
        )
    return chat_messages


_THINKING_TAGS = (
    ("<mm:think>", "</mm:think>"),
    ("<think>", "</think>"),
)


def split_embedded_thinking(content: str) -> tuple[str, str]:
    if not any(start in content or end in content for start, end in _THINKING_TAGS):
        return content, ""
    visible: list[str] = []
    reasoning: list[str] = []
    cursor = 0
    while cursor < len(content):
        match: tuple[int, str, str] | None = None
        for start, end in _THINKING_TAGS:
            index = content.find(start, cursor)
            if index >= 0 and (match is None or index < match[0]):
                match = (index, start, end)
        if match is None:
            visible.append(content[cursor:])
            break
        index, start, end = match
        visible.append(content[cursor:index])
        reasoning_start = index + len(start)
        reasoning_end = content.find(end, reasoning_start)
        if reasoning_end < 0:
            reasoning.append(content[reasoning_start:])
            cursor = len(content)
            break
        reasoning.append(content[reasoning_start:reasoning_end])
        cursor = reasoning_end + len(end)
    return "".join(visible).strip(), "\n".join(
        item.strip() for item in reasoning if item.strip()
    )


def _request_json(
    *,
    opener: urllib.request.OpenerDirector,
    request: urllib.request.Request,
    timeout: float,
    retries: int,
    retry_backoff_base: float = 1.0,
    retry_backoff_max: float = 30.0,
    provider_name: str = "DeepInfer",
) -> JsonObject:
    if retries < 0:
        raise ValueError("retries must be non-negative")
    if retry_backoff_base < 0 or retry_backoff_max < 0:
        raise ValueError("retry backoff values must be non-negative")
    if retry_backoff_max < retry_backoff_base:
        raise ValueError("retry_backoff_max must be >= retry_backoff_base")
    for attempt in range(retries + 1):
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise RuntimeError(f"{provider_name} returned a non-object response")
            return payload
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
            if attempt == retries or not retryable:
                raise RuntimeError(f"{provider_name} HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"{provider_name} network error: {exc}") from exc
        time.sleep(min(retry_backoff_base * (2**attempt), retry_backoff_max))
    raise RuntimeError(f"{provider_name} returned no response")


class DeepInferChatDriver:
    """DeepInfer's documented OpenAI-compatible Chat Completions protocol."""

    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        api_key: str,
        thinking_mode: str,
        max_tokens: int = DEEPINFER_DEFAULT_MAX_TOKENS,
        user_id: str = "deepinfer-unbound",
        timeout: float = 180.0,
        opener: urllib.request.OpenerDirector | None = None,
        provider_retries: int = 2,
    ) -> None:
        get_deepinfer_profile(model)
        if thinking_mode not in DEEPINFER_THINKING_MODES:
            raise ValueError(
                "DeepInfer thinking mode must be 'disabled' or 'enabled'"
            )
        if max_tokens < 1:
            raise ValueError("DeepInfer max_tokens must be positive")
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.thinking_mode = thinking_mode
        self.max_tokens = max_tokens
        self.user_id = self._normalize_user_id(user_id)
        self.timeout = timeout
        self.opener = opener or urllib.request.build_opener()
        self.provider_retries = provider_retries

    @staticmethod
    def _normalize_user_id(value: str) -> str:
        if len(value) <= 512:
            return value
        suffix = hashlib.sha256(value.encode()).hexdigest()
        return value[: 512 - len(suffix) - 1] + ":" + suffix

    def with_user_id(self, user_id: str) -> DeepInferChatDriver:
        return DeepInferChatDriver(
            model=self.model,
            endpoint=self.endpoint,
            api_key=self.api_key,
            thinking_mode=self.thinking_mode,
            max_tokens=self.max_tokens,
            user_id=self._normalize_user_id(user_id),
            timeout=self.timeout,
            opener=self.opener,
            provider_retries=self.provider_retries,
        )

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
            "thinking": {"type": self.thinking_mode},
            "temperature": config.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "user_id": self.user_id,
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
        )
        choices = provider.get("choices", [])
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise RuntimeError("DeepInfer returned no assistant message")
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
        if provider.get("reasoning") is not None:
            reasoning.append(
                {
                    "field": "response.reasoning",
                    "value": copy.deepcopy(provider["reasoning"]),
                }
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


def _reasoning_locations(message: JsonObject) -> list[str]:
    locations: list[str] = []
    for key in ("reasoning_content", "reasoning", "reasoning_details"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            locations.append(key)
        elif isinstance(value, (list, dict)) and value:
            locations.append(key)
    content = message.get("content")
    if isinstance(content, str):
        _, embedded = split_embedded_thinking(content)
        if embedded:
            locations.append("content.thinking_tags")
    return locations


def probe_deepinfer_capability(
    *,
    model: str,
    base_url: str,
    api_key: str,
    thinking_mode: str,
    opener: urllib.request.OpenerDirector,
    timeout: float = 60.0,
    retries: int = 1,
) -> JsonObject:
    profile = get_deepinfer_profile(model)
    models_request = urllib.request.Request(
        deepinfer_models_url(base_url),
        headers={"Authorization": f"Bearer {api_key}"},
    )
    models_payload = _request_json(
        opener=opener,
        request=models_request,
        timeout=timeout,
        retries=retries,
    )
    model_ids = {
        str(item.get("id"))
        for item in models_payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }
    if model not in model_ids:
        return {
            "status": "unsupported",
            "adapter_version": DEEPINFER_ADAPTER_VERSION,
            "model": model,
            "thinking_mode": thinking_mode,
            "route_listed": False,
            "available_model_ids": sorted(model_ids),
            "reason": "model is not listed by DeepInfer /v1/models",
        }

    tool = {
        "type": "function",
        "function": {
            "name": "deepinfer_capability_probe",
            "description": "Return the supplied probe token.",
            "parameters": {
                "type": "object",
                "properties": {"token": {"type": "string"}},
                "required": ["token"],
                "additionalProperties": False,
            },
        },
    }
    user_id = f"deepinfer-preflight:{model}:{thinking_mode}"
    first_body: JsonObject = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Call the available probe tool exactly once.",
            },
            {
                "role": "user",
                "content": (
                    "Analyze the request and call deepinfer_capability_probe "
                    "with token 'ok'."
                ),
            },
        ],
        "tools": [tool],
        "tool_choice": "auto",
        "thinking": {"type": thinking_mode},
        "temperature": profile.temperature(thinking_mode),
        "max_tokens": profile.max_tokens,
        "stream": False,
        "user_id": user_id,
    }
    endpoint = deepinfer_chat_completions_url(base_url)

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
        )

    first = post(first_body)
    first_choices = first.get("choices", [])
    if not first_choices or not isinstance(
        first_choices[0].get("message"), dict
    ):
        raise DeepInferCapabilityError(
            "DeepInfer preflight returned no first assistant message"
        )
    assistant = first_choices[0]["message"]
    calls = assistant.get("tool_calls", [])
    if (
        len(calls) != 1
        or not calls[0].get("id")
        or calls[0].get("function", {}).get("name")
        != "deepinfer_capability_probe"
    ):
        raise DeepInferCapabilityError(
            "DeepInfer preflight did not return exactly one valid tool call"
        )
    arguments = calls[0].get("function", {}).get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise DeepInferCapabilityError(
                "DeepInfer preflight returned invalid JSON tool arguments"
            ) from exc
    if not isinstance(arguments, dict):
        raise DeepInferCapabilityError(
            "DeepInfer preflight returned non-object tool arguments"
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
        raise DeepInferCapabilityError(
            "DeepInfer preflight returned no follow-up message"
        )

    first_reasoning_locations = _reasoning_locations(assistant)
    second_reasoning_locations = _reasoning_locations(
        second_choices[0]["message"]
    )
    reasoning_observed = bool(
        first_reasoning_locations or second_reasoning_locations
    )
    effective_models = [
        value
        for value in (first.get("model"), second.get("model"))
        if isinstance(value, str) and value
    ]
    model_matches = len(effective_models) == 2 and all(
        value.casefold() == model.casefold()
        for value in effective_models
    )
    mode_respected = not (
        thinking_mode == "disabled" and reasoning_observed
    )
    status = "passed" if model_matches and mode_respected else "incompatible"
    reason = None
    if not model_matches:
        reason = "DeepInfer response model does not match requested model"
    elif not mode_respected:
        reason = "reasoning was observed while thinking was disabled"
    return {
        "status": status,
        "adapter_version": DEEPINFER_ADAPTER_VERSION,
        "model": model,
        "thinking_mode": thinking_mode,
        "route_listed": True,
        "tool_call": True,
        "multi_turn_tool_history": True,
        "reasoning_observed": reasoning_observed,
        "reasoning_locations": {
            "tool_call_turn": first_reasoning_locations,
            "follow_up_turn": second_reasoning_locations,
        },
        "mode_respected": mode_respected,
        "effective_models": effective_models,
        "temperature": profile.temperature(thinking_mode),
        "max_tokens": profile.max_tokens,
        "reason": reason,
    }
