# Tasks and executable resources

`structural/` and `semantic/` contain the fixed `queries.json` and `registry.json` used in the experiments. Query records include the user request, canonical arguments, target postcondition, and task identifiers. Registries describe tool signatures, resource schemas, operation chains, and semantic action pairs.

The resource constructors and executable handlers live in [`src/envtoolbench/v2/`](../src/envtoolbench/v2/). Run `python -m environments.build` to export full seed-zero resource rows and example visible inputs. This uses the supplied query wording directly.

`cohorts/qwen3-{8b,4b}.json` stores structural eligibility and the natural-run outcome for each incomplete condition. `cohorts/stage2.json` lists the shared intervention pairs and the subset used for layer sweeps. `cohorts/control_donors.csv` maps each recipient to the paper's matched and random same-condition donors.

These inputs reconstruct the paired replay contexts by executing environment steps; full replay histories and activations are created under `outputs/` when running the experiments.
