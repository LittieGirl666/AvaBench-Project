# Stage 2: residual replacement

For each pair, tokenize the two comparison actions and append their longest shared token prefix to the replay context. The final position predicts the first differing token. The structural action pair is the final required operation versus unavailable; the semantic pair is finish versus unavailable.

The clean preference is the next-token logit difference $m(x)=z_{t_H}(x+c)-z_{t_L}(x+c)$. Replace the recipient's residual vector at that position after a selected decoder block with the donor's vector, then let the remaining blocks compute the patched preference.

With clean donor and recipient preferences $m_D,m_R$ and patched preference $m_P$:

$$e=\operatorname{sign}(m_D-m_R)(m_P-m_R),\qquad g=|m_D-m_R|.$$

Positive $e$ means movement toward the donor. Recovery is $100\,\overline e/\overline g$. Average directions and blocks within a pair, then pairs within a task group, then groups equally. The layer curves keep direction and block separate during aggregation.

## Run each step

```bash
python -m stage1.run prepare
python -m stage2.run capture --model qwen3-8b
python -m stage2.run patch --model qwen3-8b
python -m stage2.run controls --model qwen3-8b
```

Repeat with `qwen3-4b`, or use `python -m stage2.run all --model <model>` for all three steps. `--resume` continues existing outputs. `--model-dir` selects existing local weights. `--pairs` selects the prepared replay directory.

The default schedule in [`settings.json`](settings.json) runs L35 on all 186 structural and 350 semantic pairs per model, plus the selected late-block sweep on 87 structural and 175 semantic pairs. Pair membership and the layer-sweep subset are listed explicitly in [`data/cohorts/stage2.json`](../data/cohorts/stage2.json).

For a short paired intervention run, prepare a separate output directory and omit donor comparisons:

```bash
python -m stage2.run capture --model qwen3-4b --family semantic \
  --limit 2 --output outputs/example_stage2
python -m stage2.run patch --model qwen3-4b --family semantic \
  --limit 2 --output outputs/example_stage2
```

## Same-condition donors

The controls replace a recipient's state with a state from another task group under the **recipient's own condition**. Donors share the full action prefix and both differing token IDs. Matched donors minimize differences in prompt length and tool/candidate counts; random donors use the same eligible token groups. Each donor is used at most twice. Both controls use L35, and their effects are oriented by the original paired condition difference.

The paper's assignments are included in [`data/cohorts/control_donors.csv`](../data/cohorts/control_donors.csv). After capturing both models, recreate them from the new capture metadata:

```bash
python -m stage2.donors
python -m stage2.run controls --model qwen3-8b \
  --donors outputs/stage2/control_donors.csv
python -m stage2.run controls --model qwen3-4b \
  --donors outputs/stage2/control_donors.csv
```

Run these control commands after `capture` and `patch`, before running controls with any other donor file. The output retains the activation donor ID, recipient preference, paired reference preference, patched preference, and donor-oriented effect.

`intervention.py` retains the original endpoint, capture, and replacement operations. `assignment.py` retains the original donor-matching algorithm. The compact runner writes experimental values directly and stores generated activations only under `outputs/`.
