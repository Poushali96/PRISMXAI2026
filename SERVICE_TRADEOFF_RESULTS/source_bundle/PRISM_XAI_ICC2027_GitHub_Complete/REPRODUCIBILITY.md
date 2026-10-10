# Reproducing PRISM-XAI v19

The current manuscript evidence uses the v19 selective-release script:

`code/prism_xai_icc2027_v1_9_selective_release.py`

The v18 script is retained for provenance and legacy sensitivity experiments.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts download the public OPSD/UCI data when needed. Dataset caches are intentionally not committed.

## Main v19 M8 experiment (720-run matrix)

```bash
python code/prism_xai_icc2027_v1_9_selective_release.py   --real-suite full   --models tree   --xai-methods treeshap feature_ablation   --epsilon-budgets 8 16 32   --max-private-releases-per-endpoint 8   --public-release-score-threshold 3.0   --no-run-ablations   --output-dir reproduced_results/prism_v19_selective_release_M8
```

The 720 runs are:
2 datasets × 2 XAI implementations × 3 privacy budgets × 6 schedulers × 10 paired seeds.

## M4 sensitivity experiment

```bash
python code/prism_xai_icc2027_v1_9_selective_release.py   --real-suite full   --models tree   --xai-methods treeshap feature_ablation   --epsilon-budgets 8 16 32   --max-private-releases-per-endpoint 4   --public-release-score-threshold 3.0   --no-run-ablations   --output-dir reproduced_results/prism_v19_selective_release_M4
```

## Verification run

```bash
python code/prism_xai_icc2027_v1_9_selective_release.py   --real-suite verification   --models tree   --xai-methods feature_ablation   --epsilon-budgets 8 16 32   --max-private-releases-per-endpoint 4   --public-release-score-threshold 3.0   --no-run-ablations   --output-dir reproduced_results/prism_v19_selective_release_verification
```

## Important evidence rule

Use `results/current_v19/M8_main_720_runs/` as the primary experiment.
`results/current_v19/M4_sensitivity/` is a sensitivity check.
`results/current_v19/feasibility_ladder/` is a diagnostic study.
Do not substitute legacy v18 ablation/results into the v19 main evidence.
