"""Original same-condition donor assignment, with bounded reuse per donor."""

from __future__ import annotations
import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Sequence
import numpy as np
from scipy.optimize import linear_sum_assignment
from envtoolbench.v2.types import JsonObject

CONDITIONS = ("x_H", "x_L")


def sha256_json(value):
    # Preserve the experiment's deterministic tie-break order.
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def endpoint_token_signature(endpoint: JsonObject) -> JsonObject:
    value = {
        "lcp_ids": [int(item) for item in endpoint["lcp_ids"]],
        "high_divergence_token_id": int(endpoint["high_divergence_token_id"]),
        "low_divergence_token_id": int(endpoint["low_divergence_token_id"]),
    }
    value["token_signature_sha256"] = sha256_json(value)
    return value


def _same_token_axis(left: JsonObject, right: JsonObject) -> bool:
    return (
        endpoint_token_signature(left)["token_signature_sha256"]
        == endpoint_token_signature(right)["token_signature_sha256"]
    )


def _matched_edge_cost(
    recipient_id: str,
    donor_id: str,
    metadata: dict[tuple[str, str], JsonObject],
    endpoints: dict[str, JsonObject],
    seed: int,
) -> tuple[Any, ...]:
    augmented = []
    tools = []
    candidates = []
    for label in CONDITIONS:
        recipient = metadata[(recipient_id, label)]
        donor = metadata[(donor_id, label)]
        augmented.append(
            abs(
                int(donor["augmented_token_count"])
                - int(recipient["augmented_token_count"])
            )
        )
        tools.append(abs(int(donor["tool_count"]) - int(recipient["tool_count"])))
        candidates.append(
            abs(int(donor["candidate_count"]) - int(recipient["candidate_count"]))
        )
    recipient_endpoint = endpoints[recipient_id]
    donor_endpoint = endpoints[donor_id]
    endpoint_length_delta = abs(
        len(donor_endpoint["high_candidate_ids"])
        + len(donor_endpoint["low_candidate_ids"])
        - len(recipient_endpoint["high_candidate_ids"])
        - len(recipient_endpoint["low_candidate_ids"])
    )
    return (
        max(augmented),
        sum(augmented),
        max(tools),
        sum(tools),
        max(candidates),
        sum(candidates),
        endpoint_length_delta,
        sha256_json({"seed": seed, "recipient": recipient_id, "donor": donor_id}),
    )


def _assign_group(
    *,
    pair_ids: Sequence[str],
    clusters: dict[str, str],
    metadata: dict[tuple[str, str], JsonObject],
    endpoints: dict[str, JsonObject],
    control: str,
    seed: int,
    donor_capacity: int,
) -> dict[str, str]:
    """Assign one cross-cluster donor per recipient with bounded donor reuse."""

    recipients = sorted(pair_ids)
    slots = [
        (donor, slot) for donor in sorted(pair_ids) for slot in range(donor_capacity)
    ]
    valid: dict[tuple[str, str], tuple[Any, ...]] = {}
    for recipient in recipients:
        for donor in sorted(pair_ids):
            if clusters[recipient] == clusters[donor]:
                continue
            if control == "same_condition_token_matched":
                cost = _matched_edge_cost(recipient, donor, metadata, endpoints, seed)
            elif control == "token_stratified_random":
                cost = (
                    sha256_json(
                        {
                            "seed": seed,
                            "control": control,
                            "recipient": recipient,
                            "donor": donor,
                        }
                    ),
                )
            else:
                raise ValueError(f"unsupported token-matched control: {control}")
            valid[(recipient, donor)] = cost
    if any(not any(key[0] == recipient for key in valid) for recipient in recipients):
        raise RuntimeError("assignment group contains a recipient without a donor")
    ranked = {
        edge: rank
        for rank, edge in enumerate(
            sorted(valid, key=lambda edge: (valid[edge], edge[0], edge[1]))
        )
    }
    invalid_cost = float((len(ranked) + 1) * (len(slots) + 1) * 100)
    costs = np.full((len(recipients), len(slots)), invalid_cost, dtype=float)
    for row, recipient in enumerate(recipients):
        for column, (donor, slot) in enumerate(slots):
            edge = (recipient, donor)
            if edge in ranked:
                costs[row, column] = float(ranked[edge] * (donor_capacity + 1) + slot)
    rows, columns = linear_sum_assignment(costs)
    result = {}
    for row, column in zip(rows.tolist(), columns.tolist()):
        if costs[row, column] >= invalid_cost:
            raise RuntimeError("bounded donor assignment has no feasible solution")
        result[recipients[row]] = slots[column][0]
    if len(result) != len(recipients):
        raise RuntimeError("bounded donor assignment is incomplete")
    reuse = Counter(result.values())
    if max(reuse.values(), default=0) > donor_capacity:
        raise RuntimeError("bounded donor assignment exceeded donor capacity")
    return result


def assign_control_donors(
    *,
    pairs: Sequence[JsonObject],
    metadata: dict[tuple[str, str], JsonObject],
    endpoints: dict[str, JsonObject],
    control: str,
    seed: int,
    donor_capacity: int,
) -> tuple[list[JsonObject], list[JsonObject]]:
    """Return frozen assignments and explicit ineligible-recipient records."""

    pair_lookup = {str(pair["pair_id"]): pair for pair in pairs}
    clusters = {
        pair_id: str(pair["cluster_id"]) for pair_id, pair in pair_lookup.items()
    }
    signatures = {
        pair_id: endpoint_token_signature(endpoints[pair_id]) for pair_id in pair_lookup
    }
    groups: dict[str, list[str]] = defaultdict(list)
    for pair_id, signature in signatures.items():
        groups[str(signature["token_signature_sha256"])].append(pair_id)
    eligible_groups: dict[str, list[str]] = {}
    exclusions = []
    for signature_hash, pair_ids in sorted(groups.items()):
        for pair_id in sorted(pair_ids):
            if not any(clusters[other] != clusters[pair_id] for other in pair_ids):
                exclusions.append(
                    {
                        "pair_id": pair_id,
                        "cluster_id": clusters[pair_id],
                        "token_signature_sha256": signature_hash,
                        "reason": "no_cross_cluster_exact_token_donor",
                    }
                )
        eligible = [
            pair_id
            for pair_id in pair_ids
            if any(clusters[other] != clusters[pair_id] for other in pair_ids)
        ]
        if eligible:
            eligible_groups[signature_hash] = sorted(eligible)
    assignments = []
    for signature_hash, pair_ids in sorted(eligible_groups.items()):
        chosen = _assign_group(
            pair_ids=pair_ids,
            clusters=clusters,
            metadata=metadata,
            endpoints=endpoints,
            control=control,
            seed=seed,
            donor_capacity=donor_capacity,
        )
        for recipient_id, donor_id in sorted(chosen.items()):
            recipient_endpoint = endpoints[recipient_id]
            donor_endpoint = endpoints[donor_id]
            if not _same_token_axis(recipient_endpoint, donor_endpoint):
                raise RuntimeError("assignment crossed an exact token-axis stratum")
            assignments.append(
                {
                    "control": control,
                    "recipient_pair_id": recipient_id,
                    "recipient_cluster_id": clusters[recipient_id],
                    "donor_pair_id": donor_id,
                    "donor_cluster_id": clusters[donor_id],
                    "token_signature": signatures[recipient_id],
                    "high_action_equal": (
                        recipient_endpoint["high_action"]
                        == donor_endpoint["high_action"]
                    ),
                    "matching_cost": list(
                        _matched_edge_cost(
                            recipient_id, donor_id, metadata, endpoints, seed
                        )[:-1]
                    ),
                }
            )
    return assignments, sorted(exclusions, key=lambda item: item["pair_id"])
