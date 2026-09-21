# Controlled environments

The shared implementation in [`src/envtoolbench/v2/`](../src/envtoolbench/v2/) defines deterministic enterprise tools, executable resource tables, state transitions, and a task evaluator. The fixed datasets contain 400 structural tasks and 350 semantic tasks across eight domains.

## Conditions

| Family | Paper condition | Runner ID | Available capability |
|---|---|---|---|
| Structural | Direct | `baseline` | The parent operation solves the task directly |
| Structural | No parent | `no-gold` | The parent operation is absent |
| Structural | Full chain | `sub-operations` | All necessary chain steps are available |
| Structural | No chain | `sub-operations-no-gold` | The required chain is absent |
| Structural | No first | `sub-operations-no-first` | The first required step is absent |
| Structural | First only | `sub-operations-first-only` | Only the first step of the required chain remains |
| Structural | No last | `sub-operations-no-last` | The final required step is absent |
| Semantic | Gold only | `state-gold-only` | The requested state change is available |
| Semantic | Gold + decoy | `state-gold-decoy` | Both the requested and a different state change are available |
| Semantic | Decoy fails | `state-decoy-failure` | Gold is absent; the decoy returns an execution error |
| Semantic | Decoy succeeds | `state-decoy-success` | Gold is absent; the decoy changes the wrong field |

Every condition retains distractor tools. `harness_finish` submits a successful call as evidence, `harness_unavailable` declares the goal unreachable, and `harness_restart` restores the initial state. Finish succeeds only when its evidence satisfies the requested goal. The evaluator uses executable task semantics to score the result.

## Commands

```bash
python -m environments.build
python -m environments.run --model qwen3.5-27b --family both
```

Set `AVABENCH_API_BASE` and `AVABENCH_API_KEY` first. The API settings are listed in [`api_settings.json`](api_settings.json). Credentials are read from the environment.

For a selected condition or short run:

```bash
python -m environments.run --model glm-5.2 --family structural \
  --variants sub-operations-no-last --limit 2 --output outputs/example_api
```

To add a different hosted model, use `--provider OpenAI-compatible --model <model-id>` and set its endpoint. The provider can also be selected explicitly as `DeepInfer` or `SiliconFlow`. Use `--temperature` to change the generation temperature.

The runner appends one compact row per completed interaction. `--resume` continues the same model/configuration in the same output directory. `--save-trajectories` additionally writes the new interaction histories. Rebuild resources with `build.py`; the packaged queries retain the paper's exact request text.

## Implementation map

| Module under `src/envtoolbench/v2/` | Role |
|---|---|
| `schemes/enterprise_operations/` | Domain resources and structural operation chains |
| `registry.py`, `variants.py` | Structural registry and tool availability |
| `stateful_operations.py`, `stateful_variants.py` | Paired state-changing actions and feedback conditions |
| `agent_environment.py`, `agent_harness.py` | Tool execution, call references, interaction budgets, and restart |
| `agent_evaluation.py`, `stateful_evaluation.py` | Reachability, final outcomes, and remaining wrong state |
| `rendering.py` | Visible messages and tool definitions |

The environment implementation is taken from the original AvaBench experiment source. The repository-level runner replaces machine-specific launch scripts with explicit command-line options.
