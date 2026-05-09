# Reproducibility Code

Empirical demonstration of the soundness-completeness-tractability trilemma (Section 5, Figure 3).

## Requirements

```
torch
```

Install with:
```bash
pip install torch
```

## Running the experiment

```bash
python pareto_experiment.py
```

This trains a `Linear(2→32) → ReLU → Linear(32→1)` network on the safety property `f(x₁,x₂) ≥ 0.5` for all `(x₁,x₂) ∈ [0.5,1]²`, then runs Interval Bound Propagation (IBP) over ten nested sub-boxes (4% → 100% of the safe region) and prints the coverage/certification table reported in the paper.

Results are saved to `pareto_results.csv`.

## Generating Figure 3

```bash
python plot_pareto.py
```

Reads `pareto_results.csv` and produces the Pareto frontier figure.

## Details

- Optimiser: Adam, lr=1e-3, 5000 steps, batch 128 safe + 64 unsafe
- Verifier: IBP (Gowal et al. 2018) — sound by construction, runs in <1ms per box
- Seed: 42
