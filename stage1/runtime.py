from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from envtoolbench.v2.agent_harness import HarnessConfig, ModelTurn
from envtoolbench.v2.types import JsonObject

QWEN_PROTOCOL = "qwen3"
GRANITE_PROTOCOL = "granite-3.3"


def _reusable_cache_checkpoint(cache: Any) -> int:
    """Validate and record the mutable HF cache length used for candidate scoring.

    Recent Transformers architectures require a ``Cache`` object and mutate it
    even when ``use_cache=False``.  Cropping back to this checkpoint after every
    candidate gives each sequence the same prefix without copying the full KV
    cache or converting it to the deprecated tuple representation.
    """

    if cache is None:
        raise RuntimeError("model did not return a KV cache for sequence scoring")
    get_seq_length = getattr(cache, "get_seq_length", None)
    crop = getattr(cache, "crop", None)
    if not callable(get_seq_length) or not callable(crop):
        raise RuntimeError(
            "model returned a non-reusable KV cache; expected get_seq_length() and crop()"
        )
    return int(get_seq_length())


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def merge_instruction_roles(messages: Iterable[JsonObject]) -> list[JsonObject]:
    """Match the benchmark's Chat Completions system/developer normalization."""

    copied = [copy.deepcopy(item) for item in messages]
    system_parts = [
        str(item.get("content", ""))
        for item in copied
        if item.get("role") in {"system", "developer"}
    ]
    body = [item for item in copied if item.get("role") not in {"system", "developer"}]
    if system_parts:
        body.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return body


def _normalize_arguments(value: Any) -> str:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _call_id(name: str, arguments: str, generated: str) -> str:
    digest = hashlib.sha256(f"{name}\n{arguments}\n{generated}".encode()).hexdigest()
    return f"local-tool-{digest[:20]}"


@dataclass(frozen=True)
class ToolProtocolAdapter:
    protocol: str

    def template_kwargs(self) -> JsonObject:
        if self.protocol == QWEN_PROTOCOL:
            return {"enable_thinking": False}
        if self.protocol == GRANITE_PROTOCOL:
            return {"thinking": False}
        raise ValueError(f"unsupported local protocol: {self.protocol}")

    def candidate_text(self, function_name: str) -> str:
        if self.protocol == QWEN_PROTOCOL:
            return f'<tool_call>\n{{"name": "{function_name}"'
        if self.protocol == GRANITE_PROTOCOL:
            return f'<|tool_call|>[{{"name": "{function_name}"'
        raise ValueError(f"unsupported local protocol: {self.protocol}")

    def normalize_messages(self, messages: Iterable[JsonObject]) -> list[JsonObject]:
        normalized = merge_instruction_roles(messages)
        if self.protocol == QWEN_PROTOCOL:
            return normalized
        if self.protocol != GRANITE_PROTOCOL:
            raise ValueError(f"unsupported local protocol: {self.protocol}")

        converted: list[JsonObject] = []
        for message in normalized:
            item = copy.deepcopy(message)
            if item.get("role") == "assistant" and item.get("tool_calls"):
                calls = []
                for call in item["tool_calls"]:
                    function = call.get("function", {})
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    calls.append({"name": function.get("name"), "arguments": arguments})
                item = {
                    "role": "assistant",
                    "content": "<|tool_call|>" + json.dumps(calls, ensure_ascii=False),
                }
            else:
                item.pop("tool_calls", None)
                if item.get("content") is None:
                    item["content"] = ""
            converted.append(item)
        return converted

    def parse_generated_tool_call(
        self, generated: str
    ) -> tuple[tuple[JsonObject, ...], JsonObject]:
        if self.protocol == QWEN_PROTOCOL:
            opening = "<tool_call>"
            closing = "</tool_call>"
            if generated.count(opening) != 1 or generated.count(closing) != 1:
                return (), {"role": "assistant", "content": generated, "tool_calls": []}
            body = generated.split(opening, 1)[1].split(closing, 1)[0].strip()
            payload = json.loads(body)
            objects = [payload]
        elif self.protocol == GRANITE_PROTOCOL:
            marker = "<|tool_call|>"
            if marker not in generated:
                return (), {"role": "assistant", "content": generated}
            tail = generated.split(marker, 1)[1].lstrip()
            payload, _ = json.JSONDecoder().raw_decode(tail)
            objects = payload if isinstance(payload, list) else [payload]
            if len(objects) != 1:
                return (), {"role": "assistant", "content": generated}
        else:
            raise ValueError(f"unsupported local protocol: {self.protocol}")

        payload = objects[0]
        if not isinstance(payload, dict):
            return (), {"role": "assistant", "content": generated}
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not name:
            return (), {"role": "assistant", "content": generated}
        arguments_text = _normalize_arguments(arguments)
        call: JsonObject = {
            "id": _call_id(name, arguments_text, generated),
            "type": "function",
            "function": {"name": name, "arguments": arguments_text},
        }
        history = {
            "role": "assistant",
            "content": "",
            "tool_calls": [copy.deepcopy(call)],
        }
        return (call,), history


class LocalTransformersRuntime:
    """One pinned local CausalLM plus its benchmark tool protocol."""

    def __init__(
        self,
        *,
        model_key: str,
        model_path: str | Path,
        protocol: str,
        max_context_tokens: int,
        max_new_tokens: int = 512,
        attn_implementation: str = "sdpa",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised by remote preflight
            raise RuntimeError("install the Stage 1 Transformers requirements") from exc

        self.torch = torch
        self.model_key = model_key
        self.model_path = Path(model_path)
        self.model = model_key
        self.protocol = ToolProtocolAdapter(protocol)
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.causal_lm = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map={"": 0},
        )
        self.causal_lm.eval()
        self.causal_lm.config.use_cache = True

    @property
    def device(self) -> Any:
        return next(self.causal_lm.parameters()).device

    def decoder_layers(self) -> tuple[Any, ...]:
        """Return decoder blocks in forward order for causal interventions.

        Stage 2 intentionally fails closed when an architecture is unknown: a
        guessed module path could make a successful-looking patch target the
        wrong stream.  Qwen3 uses ``model.layers``; the additional paths make
        the helper testable and reusable without weakening that invariant.
        """

        candidates = (
            ("model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
        )
        for path in candidates:
            value: Any = self.causal_lm
            for name in path:
                value = getattr(value, name, None)
                if value is None:
                    break
            if value is not None:
                layers = tuple(value)
                if layers:
                    return layers
        raise RuntimeError(
            f"unsupported decoder layout for {type(self.causal_lm).__name__}"
        )

    def render_inputs(
        self,
        *,
        messages: Iterable[JsonObject],
        tools: Iterable[JsonObject],
    ) -> JsonObject:
        normalized = self.protocol.normalize_messages(messages)
        encoded = self.tokenizer.apply_chat_template(
            normalized,
            tools=list(tools),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **self.protocol.template_kwargs(),
        )
        token_count = int(encoded["input_ids"].shape[-1])
        if token_count > self.max_context_tokens:
            raise ValueError(
                f"context_too_long: {token_count} > {self.max_context_tokens}"
            )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def generate(
        self,
        *,
        messages: list[JsonObject],
        tools: tuple[JsonObject, ...],
        config: HarnessConfig,
    ) -> ModelTurn:
        del config
        encoded = self.render_inputs(messages=messages, tools=tools)
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        with self.torch.inference_mode():
            generated_ids = self.causal_lm.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion_ids = generated_ids[0, prompt_tokens:]
        generated = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        try:
            calls, history = self.protocol.parse_generated_tool_call(generated)
        except (json.JSONDecodeError, TypeError, ValueError):
            calls, history = (), {"role": "assistant", "content": generated}
        return ModelTurn(
            response_id=None,
            assistant_text=generated,
            reasoning_items=(),
            tool_calls=tuple(calls),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": int(completion_ids.numel()),
                "total_tokens": prompt_tokens + int(completion_ids.numel()),
            },
            assistant_message=history,
        )

    def score_candidates(
        self,
        *,
        messages: Iterable[JsonObject],
        tools: Iterable[JsonObject],
        candidate_names: Iterable[str],
    ) -> tuple[int, list[JsonObject]]:
        encoded = self.render_inputs(messages=messages, tools=tools)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        with self.torch.inference_mode():
            base = self.causal_lm(
                **encoded,
                use_cache=True,
                return_dict=True,
            )
        cache = base.past_key_values
        cache_checkpoint = _reusable_cache_checkpoint(cache)
        first_logits = base.logits[:, -1, :].float().log_softmax(dim=-1)

        results: list[JsonObject] = []
        for name in candidate_names:
            candidate_text = self.protocol.candidate_text(name)
            token_ids = self.tokenizer.encode(candidate_text, add_special_tokens=False)
            if not token_ids:
                raise ValueError(f"empty candidate tokenization: {name}")
            pieces = [float(first_logits[0, token_ids[0]].item())]
            if len(token_ids) > 1:
                continuation = self.torch.tensor(
                    [token_ids[:-1]], dtype=self.torch.long, device=self.device
                )
                if attention_mask is None:
                    candidate_mask = None
                else:
                    tail_mask = self.torch.ones(
                        (1, len(token_ids) - 1),
                        dtype=attention_mask.dtype,
                        device=self.device,
                    )
                    candidate_mask = self.torch.cat((attention_mask, tail_mask), dim=-1)
                try:
                    with self.torch.inference_mode():
                        continued = self.causal_lm(
                            input_ids=continuation,
                            attention_mask=candidate_mask,
                            past_key_values=cache,
                            use_cache=False,
                            return_dict=True,
                        )
                finally:
                    cache.crop(cache_checkpoint)
                log_probs = continued.logits.float().log_softmax(dim=-1)
                for index, target in enumerate(token_ids[1:]):
                    pieces.append(float(log_probs[0, index, target].item()))
            total = sum(pieces)
            results.append(
                {
                    "candidate_name": name,
                    "candidate_text": candidate_text,
                    "candidate_token_ids": token_ids,
                    "token_logprobs": pieces,
                    "sequence_logprob": total,
                    "mean_logprob": total / len(pieces),
                }
            )
        del base
        self.torch.cuda.empty_cache()
        return int(input_ids.shape[-1]), results
