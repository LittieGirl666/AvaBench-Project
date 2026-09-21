"""Residual replacement at the first differing action token.

Numerical operations retain the original readout-patching implementation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from envtoolbench.v2.types import JsonObject
from stage1.runtime import LocalTransformersRuntime


def _context(pair, label):
    return next(c for c in pair["contexts"] if c["label"] == label)


def longest_common_prefix(sequences: Sequence[Sequence[int]]) -> list[int]:
    """Return the exact token LCP of non-empty sequences."""

    if len(sequences) < 2 or any(not sequence for sequence in sequences):
        raise ValueError("LCP requires at least two non-empty token sequences")
    limit = min(len(sequence) for sequence in sequences)
    end = 0
    while end < limit and len({int(sequence[end]) for sequence in sequences}) == 1:
        end += 1
    return [int(value) for value in sequences[0][:end]]


def append_token_prefix(
    encoded: JsonObject, prefix_ids: Sequence[int], torch: Any
) -> JsonObject:
    """Append an LCP to a rendered prompt without re-tokenizing its boundary."""

    if not prefix_ids:
        raise ValueError("decision readout prefix must not be empty")
    input_ids = encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"expected input_ids [1, sequence], got {tuple(input_ids.shape)}"
        )
    prefix = torch.tensor(
        [list(prefix_ids)], dtype=input_ids.dtype, device=input_ids.device
    )
    result = dict(encoded)
    result["input_ids"] = torch.cat((input_ids, prefix), dim=-1)
    if "attention_mask" in encoded:
        mask = encoded["attention_mask"]
        tail = torch.ones((1, len(prefix_ids)), dtype=mask.dtype, device=mask.device)
        result["attention_mask"] = torch.cat((mask, tail), dim=-1)
    if "position_ids" in encoded:
        position_ids = encoded["position_ids"]
        start = int(position_ids[0, -1].item()) + 1
        tail = torch.arange(
            start,
            start + len(prefix_ids),
            dtype=position_ids.dtype,
            device=position_ids.device,
        ).unsqueeze(0)
        result["position_ids"] = torch.cat((position_ids, tail), dim=-1)
    return result


def _hidden_tensor(output: Any) -> Any:
    if hasattr(output, "ndim"):
        return output
    if isinstance(output, (tuple, list)) and output and hasattr(output[0], "ndim"):
        return output[0]
    raise TypeError(f"decoder output does not expose a hidden tensor: {type(output)!r}")


def replace_position(output: Any, donor: Any, alpha: float, position_index: int) -> Any:
    """Interpolate exactly one residual position and preserve output structure."""

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    hidden = _hidden_tensor(output)
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(f"expected [1, sequence, hidden], got {tuple(hidden.shape)}")
    index = position_index if position_index >= 0 else hidden.shape[1] + position_index
    if not 0 <= index < hidden.shape[1]:
        raise IndexError(
            f"patch position {position_index} outside sequence length {hidden.shape[1]}"
        )
    if donor.ndim == 2 and donor.shape[0] == 1:
        donor = donor[0]
    if donor.ndim != 1 or donor.shape[0] != hidden.shape[-1]:
        raise ValueError(
            f"donor shape {tuple(donor.shape)} does not match hidden size {hidden.shape[-1]}"
        )
    patched = hidden.clone()
    source = donor.to(device=hidden.device, dtype=hidden.dtype)
    patched[:, index, :] = hidden[:, index, :] + float(alpha) * (
        source - hidden[:, index, :]
    )
    if hasattr(output, "ndim"):
        return patched
    tail = list(output[1:])
    return (patched, *tail) if isinstance(output, tuple) else [patched, *tail]


class PositionResidualPatch:
    """One-shot hook for a named absolute residual position."""

    def __init__(self, donor: Any, *, alpha: float, position_index: int) -> None:
        self.donor = donor
        self.alpha = float(alpha)
        self.position_index = int(position_index)
        self.calls = 0
        self.patches = 0

    def __call__(self, module: Any, inputs: Any, output: Any) -> Any:
        del module, inputs
        self.calls += 1
        if self.patches:
            return output
        self.patches += 1
        return replace_position(output, self.donor, self.alpha, self.position_index)


class DualPositionCapture:
    """Capture prompt-end and decision-readout residuals from the same pass."""

    def __init__(
        self, torch_module: Any, layer_count: int, prompt_end_index: int
    ) -> None:
        self.torch = torch_module
        self.prompt_end_index = int(prompt_end_index)
        self.values: list[Any | None] = [None] * layer_count

    def hook(self, index: int) -> Any:
        def capture(module: Any, inputs: Any, output: Any) -> Any:
            del module, inputs
            hidden = _hidden_tensor(output)
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError(
                    f"expected [1, sequence, hidden], got {tuple(hidden.shape)}"
                )
            if not 0 <= self.prompt_end_index < hidden.shape[1]:
                raise IndexError("prompt-end capture index is outside sequence")
            self.values[index] = (
                self.torch.stack(
                    (hidden[0, self.prompt_end_index, :], hidden[0, -1, :])
                )
                .detach()
                .to("cpu")
                .clone()
            )
            return output

        return capture

    def stacked(self) -> Any:
        if any(value is None for value in self.values):
            missing = [i for i, value in enumerate(self.values) if value is None]
            raise RuntimeError(f"decoder capture missed layers: {missing}")
        # [position=2, layer, hidden], registered cache layout.
        return self.torch.stack(self.values, dim=1)


@dataclass(frozen=True)
class EndpointDefinition:
    family: str
    high_action: str
    low_action: str
    high_candidate_ids: tuple[int, ...]
    low_candidate_ids: tuple[int, ...]
    lcp_ids: tuple[int, ...]
    high_token_id: int
    low_token_id: int
    structural_visible_control: str | None = None


def _candidate_ids(runtime: LocalTransformersRuntime, action: str) -> list[int]:
    text = runtime.protocol.candidate_text(action)
    ids = runtime.tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"empty candidate tokenization for {action}")
    return [int(value) for value in ids]


def _structural_visible_control(
    runtime: LocalTransformersRuntime, target: str, context: JsonObject
) -> str | None:
    """Choose the visible environment action with the longest target LCP."""

    target_ids = _candidate_ids(runtime, target)
    choices = []
    for name in sorted(str(value) for value in context["visible_operation_ids"]):
        if name == target:
            continue
        ids = _candidate_ids(runtime, name)
        choices.append((len(longest_common_prefix((target_ids, ids))), name))
    return min(choices, key=lambda item: (-item[0], item[1]))[1] if choices else None


def define_endpoint(
    runtime: LocalTransformersRuntime, pair: JsonObject
) -> EndpointDefinition:
    """Resolve the registered high/low actions before looking at model outputs."""

    family = str(pair["family"])
    if family == "semantic":
        high_action, low_action = "harness_finish", "harness_unavailable"
        structural_control = None
    elif family == "structural":
        targets = [str(value) for value in pair.get("target_step_ids") or ()]
        if not targets:
            raise ValueError(f"structural pair has no target steps: {pair['pair_id']}")
        high_action, low_action = targets[-1], "harness_unavailable"
        structural_control = _structural_visible_control(
            runtime, high_action, _context(pair, "x_H")
        )
    else:
        raise ValueError(f"unsupported pair family: {family}")
    universe = {str(value) for value in pair["candidate_universe"]}
    if high_action not in universe or low_action not in universe:
        raise ValueError(
            f"endpoint actions absent from candidate universe: {high_action}, {low_action}"
        )
    high_ids = _candidate_ids(runtime, high_action)
    low_ids = _candidate_ids(runtime, low_action)
    lcp = longest_common_prefix((high_ids, low_ids))
    if len(lcp) >= min(len(high_ids), len(low_ids)):
        raise ValueError(f"endpoint does not diverge after an LCP: {pair['pair_id']}")
    return EndpointDefinition(
        family=family,
        high_action=high_action,
        low_action=low_action,
        high_candidate_ids=tuple(high_ids),
        low_candidate_ids=tuple(low_ids),
        lcp_ids=tuple(lcp),
        high_token_id=high_ids[len(lcp)],
        low_token_id=low_ids[len(lcp)],
        structural_visible_control=structural_control,
    )


def _selected_logits(logits: Any, high_id: int, low_id: int) -> JsonObject:
    vector = logits[0, -1, :].float()
    high = float(vector[high_id].item())
    low = float(vector[low_id].item())
    pair_prob = vector[[high_id, low_id]].softmax(dim=-1)
    full_prob = vector.log_softmax(dim=-1).exp()
    return {
        "high_logit": high,
        "low_logit": low,
        "margin": high - low,
        "high_pair_probability": float(pair_prob[0].item()),
        "low_pair_probability": float(pair_prob[1].item()),
        "high_vocab_probability": float(full_prob[high_id].item()),
        "low_vocab_probability": float(full_prob[low_id].item()),
        "high_vocab_rank": int((vector > vector[high_id]).sum().item()) + 1,
        "low_vocab_rank": int((vector > vector[low_id]).sum().item()) + 1,
    }


def capture_readout(
    *,
    runtime: LocalTransformersRuntime,
    messages: Iterable[JsonObject],
    tools: Iterable[JsonObject],
    lcp_ids: Sequence[int],
    high_token_id: int,
    low_token_id: int,
) -> tuple[int, Any, JsonObject]:
    """Capture both positions and score the divergent tokens in one forward."""

    encoded = runtime.render_inputs(messages=messages, tools=tools)
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    augmented = append_token_prefix(encoded, lcp_ids, runtime.torch)
    prompt_end_index = prompt_tokens - 1
    layers = runtime.decoder_layers()
    capture = DualPositionCapture(runtime.torch, len(layers), prompt_end_index)
    handles = [
        layer.register_forward_hook(capture.hook(i)) for i, layer in enumerate(layers)
    ]
    try:
        with runtime.torch.inference_mode():
            output = runtime.causal_lm(**augmented, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return (
        prompt_tokens,
        capture.stacked(),
        _selected_logits(output.logits, high_token_id, low_token_id),
    )


def score_readout_patched(
    *,
    runtime: LocalTransformersRuntime,
    encoded_inputs: JsonObject,
    layer_index: int,
    donor: Any,
    alpha: float,
    position_index: int,
    high_token_id: int,
    low_token_id: int,
) -> tuple[JsonObject, PositionResidualPatch]:
    layers = runtime.decoder_layers()
    if not 0 <= layer_index < len(layers):
        raise IndexError(f"invalid layer {layer_index} for {len(layers)} layers")
    hook = PositionResidualPatch(
        donor, alpha=float(alpha), position_index=int(position_index)
    )
    handle = layers[layer_index].register_forward_hook(hook)
    try:
        with runtime.torch.inference_mode():
            output = runtime.causal_lm(
                **encoded_inputs, use_cache=False, return_dict=True
            )
    finally:
        handle.remove()
    if hook.patches != 1:
        raise RuntimeError(f"patch hook fired {hook.patches} times")
    return _selected_logits(output.logits, high_token_id, low_token_id), hook


def donor_aligned_effect(
    *, donor_margin: float, recipient_margin: float, patched_margin: float
) -> float:
    delta = float(donor_margin) - float(recipient_margin)
    sign = 0.0 if delta == 0.0 else (1.0 if delta > 0.0 else -1.0)
    return sign * (float(patched_margin) - float(recipient_margin))
