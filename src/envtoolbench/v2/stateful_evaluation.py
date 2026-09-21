from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .agent_evaluation import AgentEvaluation, Evaluator, ReachabilityResult
from .agent_harness import HarnessResult
from .execution import execute_base
from .stateful_operations import StatefulRegistry
from .stateful_variants import StatefulVariantResolution
from .types import JsonObject, V2QueryCase


STATEFUL_EVAL_SCHEMA_VERSION = "stateful_agent_eval.v1"


@dataclass(frozen=True)
class StatefulTrajectoryMetrics:
    decoy_call_count: int
    successful_decoy_call_count: int
    wrong_state_commit_count: int
    wrong_state_committed: bool
    false_finish_after_decoy: bool
    wrong_state_detected: bool
    restart_recovered: bool
    compensating_action_count: int
    turns_to_detection: int | None
    target_state_achieved: bool
    residual_wrong_state_at_termination: bool
    decoy_resource_changed_at_termination: bool

    def to_dict(self) -> JsonObject:
        return {
            "decoy_call_count": self.decoy_call_count,
            "successful_decoy_call_count": self.successful_decoy_call_count,
            "wrong_state_commit_count": self.wrong_state_commit_count,
            "wrong_state_committed": self.wrong_state_committed,
            "false_finish_after_decoy": self.false_finish_after_decoy,
            "wrong_state_detected": self.wrong_state_detected,
            "restart_recovered": self.restart_recovered,
            "compensating_action_count": self.compensating_action_count,
            "turns_to_detection": self.turns_to_detection,
            "target_state_achieved": self.target_state_achieved,
            "residual_wrong_state_at_termination": self.residual_wrong_state_at_termination,
            "decoy_resource_changed_at_termination": self.decoy_resource_changed_at_termination,
        }


@dataclass(frozen=True)
class StatefulAgentEvaluation:
    base: AgentEvaluation
    state_metrics: StatefulTrajectoryMetrics

    def to_dict(self) -> JsonObject:
        return {
            **self.base.to_dict(),
            "state_metrics": self.state_metrics.to_dict(),
        }


def _resource_changed(
    initial: Any,
    final: Any,
    resource_id: str,
) -> bool:
    return initial.resources.get(resource_id) != final.resources.get(resource_id)


class StatefulEvaluator:
    def __init__(
        self,
        *,
        max_depth: int = 6,
        transition_limit: int = 100_000,
    ) -> None:
        self.base = Evaluator(
            max_depth=max_depth,
            transition_limit=transition_limit,
        )

    def evaluate(
        self,
        registry: StatefulRegistry,
        case: V2QueryCase,
        *,
        resolution: StatefulVariantResolution,
        run: HarnessResult,
    ) -> StatefulAgentEvaluation:
        oracle_visible = tuple(
            operation_id
            for operation_id in resolution.visible_operation_ids
            if operation_id not in set(resolution.forced_failure_operation_ids)
        )
        if resolution.target_operation_id in oracle_visible:
            oracle_state = run.initial_state.clone()
            oracle_result = execute_base(
                registry.get_base(resolution.target_operation_id),
                oracle_state,
                case.expected_call.canonical_arguments,
            )
            reachability = ReachabilityResult(
                "reachable",
                ({
                    "operation_id": resolution.target_operation_id,
                    "canonical_arguments": copy.deepcopy(
                        case.expected_call.canonical_arguments
                    ),
                    "output": copy.deepcopy(oracle_result.output),
                },),
                1,
                self.base.max_depth,
                self.base.transition_limit,
            )
        else:
            # State-action twins have unique target postconditions and exact,
            # mutually exclusive write sets. Their construction tests prove
            # that no decoy/nuisance sequence can produce the gold state.
            reachability = ReachabilityResult(
                "unreachable",
                (),
                0,
                self.base.max_depth,
                self.base.transition_limit,
            )
        base = self.base.evaluate(
            registry,
            case,
            visible_operation_ids=oracle_visible,
            run=run,
            reachability_override=reachability,
        )

        target = registry.state_transitions[resolution.target_operation_id]
        decoy = registry.state_transitions[resolution.decoy_operation_id]
        expected_state = run.initial_state.clone()
        execute_base(
            registry.get_base(target.operation_id),
            expected_state,
            case.expected_call.canonical_arguments,
        )
        expected_digest = expected_state.digest()

        decoy_calls = [
            call
            for call in run.calls
            if call.operation_id == decoy.operation_id
        ]
        committed = [
            call
            for call in decoy_calls
            if call.success and decoy.target_resource_id in call.changed_resources
        ]
        first_commit_turn = None
        if committed:
            committed_ids = {call.call_id for call in committed}
            first_commit_turn = next(
                (
                    int(event["turn"])
                    for event in run.events
                    if event.get("type") == "environment_call"
                    and event.get("call_id") in committed_ids
                ),
                None,
            )

        detection_turn = None
        false_finish = False
        if first_commit_turn is not None:
            for event in run.events:
                turn = int(event.get("turn", 0))
                if turn <= first_commit_turn:
                    continue
                if (
                    event.get("type") == "control_error"
                    and event.get("control") == "harness_finish"
                ):
                    false_finish = True
                if (
                    event.get("type") == "control"
                    and event.get("control") in {
                        "harness_restart",
                        "harness_unavailable",
                    }
                    and detection_turn is None
                ):
                    detection_turn = turn

        restart_turns = [
            int(event.get("turn", 0))
            for event in run.events
            if event.get("type") == "control"
            and event.get("control") == "harness_restart"
        ]
        restart_after_commit = bool(
            first_commit_turn is not None
            and any(turn > first_commit_turn for turn in restart_turns)
        )
        target_state_achieved = run.final_digest == expected_digest
        clean_unavailable_state = (
            run.termination_reason == "unavailable"
            and run.final_digest == run.initial_digest
        )
        restart_recovered = bool(
            restart_after_commit
            and (target_state_achieved or clean_unavailable_state)
        )

        compensating = 0
        if committed:
            last_commit_index = max(
                index
                for index, call in enumerate(run.calls)
                if call in committed
            )
            for call in run.calls[last_commit_index + 1:]:
                if not call.success or not call.changed_resources:
                    continue
                if call.state_digest_after in {
                    run.initial_digest,
                    expected_digest,
                }:
                    compensating += 1

        residual_wrong_state = (
            run.final_digest not in {run.initial_digest, expected_digest}
        )
        metrics = StatefulTrajectoryMetrics(
            decoy_call_count=len(decoy_calls),
            successful_decoy_call_count=sum(call.success for call in decoy_calls),
            wrong_state_commit_count=len(committed),
            wrong_state_committed=bool(committed),
            false_finish_after_decoy=false_finish,
            wrong_state_detected=detection_turn is not None,
            restart_recovered=restart_recovered,
            compensating_action_count=compensating,
            turns_to_detection=(
                detection_turn - first_commit_turn
                if detection_turn is not None and first_commit_turn is not None
                else None
            ),
            target_state_achieved=target_state_achieved,
            residual_wrong_state_at_termination=residual_wrong_state,
            decoy_resource_changed_at_termination=_resource_changed(
                run.initial_state,
                run.final_state,
                decoy.target_resource_id,
            ),
        )
        return StatefulAgentEvaluation(base, metrics)


def _rate(rows: list[JsonObject], field: str) -> float | None:
    if not rows:
        return None
    return sum(bool(row["evaluation"]["state_metrics"][field]) for row in rows) / len(rows)


def _summarize_group(rows: list[JsonObject]) -> JsonObject:
    behavior = [
        row
        for row in rows
        if row.get("termination", {}).get("reason") != "infrastructure_error"
    ]
    reachable = [
        row
        for row in behavior
        if row["evaluation"]["reachability"]["status"] == "reachable"
    ]
    unreachable = [
        row
        for row in behavior
        if row["evaluation"]["reachability"]["status"] == "unreachable"
    ]
    committed = [
        row
        for row in behavior
        if row["evaluation"]["state_metrics"]["wrong_state_committed"]
    ]
    detection_turns = [
        int(row["evaluation"]["state_metrics"]["turns_to_detection"])
        for row in behavior
        if row["evaluation"]["state_metrics"]["turns_to_detection"] is not None
    ]
    return {
        "Runs": len(rows),
        "BehaviorRuns": len(behavior),
        "InfrastructureErrorCount": len(rows) - len(behavior),
        "OutcomeSuccessRate": (
            sum(bool(row["evaluation"]["outcome_success"]) for row in reachable)
            / len(reachable)
            if reachable
            else None
        ),
        "UnavailableAccuracy": (
            sum(bool(row["evaluation"].get("unavailable_correct")) for row in unreachable)
            / len(unreachable)
            if unreachable
            else None
        ),
        "DecoyCallRate": _rate(behavior, "decoy_call_count"),
        "SuccessfulDecoyCallRate": _rate(
            behavior,
            "successful_decoy_call_count",
        ),
        "WrongStateCommitRate": _rate(behavior, "wrong_state_committed"),
        "FalseFinishAfterDecoyRate": _rate(
            committed,
            "false_finish_after_decoy",
        ),
        "WrongStateDetectedRate": _rate(committed, "wrong_state_detected"),
        "RestartRecoveryRate": _rate(committed, "restart_recovered"),
        "CompensatingActionRate": _rate(
            committed,
            "compensating_action_count",
        ),
        "ResidualDamageRate": _rate(
            committed,
            "residual_wrong_state_at_termination",
        ),
        "AverageTurnsToDetection": (
            sum(detection_turns) / len(detection_turns)
            if detection_turns
            else None
        ),
        "AverageModelTurns": (
            sum(int(row["budget"]["model_turns_used"]) for row in behavior)
            / len(behavior)
            if behavior
            else 0.0
        ),
        "AverageEnvironmentCalls": (
            sum(int(row["budget"]["environment_calls_used"]) for row in behavior)
            / len(behavior)
            if behavior
            else 0.0
        ),
        "TerminationReasonCounts": dict(sorted(Counter(
            str(row.get("termination", {}).get("reason"))
            for row in rows
        ).items())),
    }


def summarize_stateful_results(records: Iterable[JsonObject]) -> JsonObject:
    rows = [copy.deepcopy(row) for row in records]
    if any(row.get("schema_version") != STATEFUL_EVAL_SCHEMA_VERSION for row in rows):
        raise ValueError(
            f"summarize_stateful_results accepts only {STATEFUL_EVAL_SCHEMA_VERSION}"
        )
    if any(row.get("dry_run") for row in rows):
        raise ValueError("dry-run request artifacts are not scorable")
    variants = sorted({str(row["variant_id"]) for row in rows})
    return {
        "SchemaVersion": STATEFUL_EVAL_SCHEMA_VERSION,
        **_summarize_group(rows),
        "ByVariant": {
            variant_id: _summarize_group([
                row for row in rows if row["variant_id"] == variant_id
            ])
            for variant_id in variants
        },
    }
