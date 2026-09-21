# Stage 1: stopping evidence

Structural pairs replay the same successful prefix with complete or incomplete tools. No chain and No first are compared before any action; First only follows the first success; No last follows the penultimate success. Semantic pairs replay two observers and the decoy, then compare wrong-state success with explicit failure. Both semantic contexts lack the goal-matching action.

## Computation

For a complete action name in tool-call format:

$$S(a\mid x)=\sum_j\log P(y_j\mid x,y_{<j}).$$

Let $u$, $f$, and $r$ denote unavailable, finish, and restart, and let $E(x)$ contain visible environment actions. The structural margin is

$$M_{str}(x)=S(u)-\log\left(e^{S(f)}+e^{S(r)}+\frac{1}{|E(x)|}\sum_{a\in E(x)}e^{S(a)}\right).$$

For semantic candidate set $A$:

$$M_{sem}(x)=S(u)-\log\sum_{a\in A\setminus\{u\}}e^{S(a)}.$$

Positive margins favor declaring unavailability over the combined alternatives. The reported change is missing minus complete for structure, and success minus failure for semantics. Scoring ends at the closing quote after the function name, before argument generation.

[`example_scores.json`](../results/stage1/example_scores.json) gives complete candidate scores and the resulting margins for two tasks under both conditions and both models.

## Run

Follow the installation and model-download steps in [repository README](../README.md#stage-1-action-scores-and-stopping-evidence), then:

```bash
python -m stage1.run prepare
python -m stage1.run score --model qwen3-8b
python -m stage1.run score --model qwen3-4b
```

`prepare` runs on CPU. It reconstructs contexts from the supplied task lists and the paper's cohort membership. The structural cohorts contain 329 eligible tasks for 8B and 211 for 4B. The resulting independent comparison counts are 1,306 and 837, plus 350 semantic pairs for each model.

The scorer writes `context_scores.jsonl` with candidate sequence scores, and `pair_effects.csv` with paired margins and changes. Add `--resume` to continue scoring. A small semantic run can use `--family semantic --limit 2`; use a separate `--output` directory after preparing pairs there with the same option.

## Recompute structural qualification

Qualification runs each structural task in the complete chain and all four incomplete-chain conditions. A task is eligible when the complete chain succeeds and all incomplete conditions end with unavailability or budget exhaustion.

```bash
python -m stage1.run qualify --model qwen3-8b --output outputs/fresh_stage1
python -m stage1.run qualify --model qwen3-4b --output outputs/fresh_stage1
python -m stage1.run prepare --cohorts outputs/fresh_stage1/cohorts \
  --output outputs/fresh_stage1
python -m stage1.run score --model qwen3-8b --output outputs/fresh_stage1
python -m stage1.run score --model qwen3-4b --output outputs/fresh_stage1
```

This generates new compact qualification outcomes and cohort files before constructing and scoring the replays. The default path uses the included cohorts for direct comparison with the manuscript.

`runtime.py`, `replay.py`, and `scoring.py` retain the numerical and replay operations from the original Stage 1 implementation. `pairs.py` and `run.py` provide the portable data flow.
