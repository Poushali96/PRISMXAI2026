# PRISM-XAI — ICC 2027 complete research artifact

**MAP-XAI privacy-accounted explanation packetization · PRISM-XAI wireless explanation scheduling · experimental public-score controller**

This repository is the **complete archived research package**, not a minimal demonstration. It includes the paper sources, original v1.9 and v1.8 code, three original evaluation matrices, nine additional public-score threshold/release-cap experiments, exported raw metrics, diagnostics, plots, validation files, and provenance. The README is a **map to the existing artifact**. It does not claim that the entire experiment has been independently rerun on another machine.

> **Start here:** New readers should use [Quick navigation](#quick-navigation), then [Reproducing experiments](#reproducing-experiments) and [Paper-to-evidence roadmap](#paper-to-evidence-roadmap).
>
> **Review/privacy caveat:** The public-score controller is an **experimental variant**, not an assertion of full-transcript differential privacy. Packet-content accounting requires the stated adjacency/sensitivity assumptions. Transmission identity, timing and silence are not automatically protected. The source includes earlier residual-score variants that should not be described as public-state schedulers.

## Quick navigation

| I want to… | Start here |
|---|---|
| Understand the proposed algorithms | [`code/prism_xai_icc2027_v1_9_selective_release.py`](code/prism_xai_icc2027_v1_9_selective_release.py) and [`code/prism_xai_icc2027_v2_0_public_score_experimental.py`](code/prism_xai_icc2027_v2_0_public_score_experimental.py) |
| See the submission manuscript | [`MAIN_PAPER/`](MAIN_PAPER/) |
| Read technical derivations and extended evaluation | [`supplementary/SUPPLEMENTARY/`](supplementary/SUPPLEMENTARY/) |
| Find original residual-score M=8 results | [`A_ORIGINAL_V19_M8_WITH_ABLATIONS/`](A_ORIGINAL_V19_M8_WITH_ABLATIONS/) |
| Find original residual-score M=4 results | [`B_ORIGINAL_V19_M4_WITH_ABLATIONS/`](B_ORIGINAL_V19_M4_WITH_ABLATIONS/) |
| Find public-score M=8 main comparison | [`C_EXPERIMENTAL_PUBLIC_SCORE_M8/`](C_EXPERIMENTAL_PUBLIC_SCORE_M8/) |
| Check the 9-point sensitivity sweep | [`SERVICE_TRADEOFF_RESULTS/paper_evidence/`](SERVICE_TRADEOFF_RESULTS/paper_evidence/) |
| Read final analysis summary and statistical comparisons | [`paper_evidence/`](paper_evidence/) |
| Inspect run manifests and original patch | [`provenance/`](provenance/) and [`SERVICE_TRADEOFF_RESULTS/provenance/`](SERVICE_TRADEOFF_RESULTS/provenance/) |
| Find archived data source descriptions | [`A_ORIGINAL_V19_M8_WITH_ABLATIONS/DATA_SOURCES.md`](A_ORIGINAL_V19_M8_WITH_ABLATIONS/DATA_SOURCES.md) |

## Repository structure

```text
PRISMXAI2026-main/
├── README.md                       # This navigation and reproduction guide
├── .gitignore / .gitattributes
├── code/
│   ├── prism_xai_icc2027_v1_8_matched_fallback.py
│   ├── prism_xai_icc2027_v1_9_selective_release.py
│   ├── prism_xai_icc2027_v2_0_public_score_experimental.py
│   └── requirements.txt
├── A_ORIGINAL_V19_M8_WITH_ABLATIONS/
│   ├── combined_raw_run_metrics.csv
│   ├── combined_validation_report.json
│   ├── combined_run_manifest.json
│   ├── combined_ablation_metrics.csv
│   ├── opsd/                        # OPSD data-specific outputs/config
│   └── uci/                         # UCI data-specific outputs/config
├── B_ORIGINAL_V19_M4_WITH_ABLATIONS/ # Same output organization; cap M=4
├── C_EXPERIMENTAL_PUBLIC_SCORE_M8/   # Same output organization; public score M=8
├── SERVICE_TRADEOFF_RESULTS/
│   ├── runs/                         # 9 settings: thr_*_cap_*
│   ├── paper_evidence/               # Pooled sweep means, paired tests, figures
│   ├── provenance/                   # Patch, completed runs, environment
│   └── source_bundle/                # Original uploaded complete bundle snapshot
├── paper_evidence/                  # Cross-configuration statistics, macros
├── provenance/                      # Input hashes, environment and patch
├── MAIN_PAPER/                      # Submission TeX/PDF and IEEE class
├── supplementary/SUPPLEMENTARY/     # Technical appendix source/assets
├── README_UPDATED_PACKAGE.md       # Historical package notes
└── *SHA256SUMS*.txt                 # Archived integrity manifests
```

**Duplicate copies are intentional archival snapshots:** `SERVICE_TRADEOFF_RESULTS/source_bundle/` is a record of the original source supplied to Colab; it is not another independent implementation to choose at random. Large result folders contain both print-ready PDFs and PNGs, plus separate OPSD/UCI detail; do not sum nested and combined CSVs together.

## Research and experiment overview

MAP-XAI forms private, compressed explanation packets by clipping attribution changes, applying a Gaussian mechanism, then Top-k selection and quantization. PRISM-XAI chooses the endpoint and packet mode (or abstains), under uplink, deadline, freshness, fairness, and accounted content-privacy constraints. The final paper evaluates OPSD and UCI load data with TreeSHAP and feature ablation.

Three archived core matrices distinguish **score type** and **maximum private releases**:

| Archive | Controller | Max releases | Use |
|---|---|---:|---|
| `A_ORIGINAL_V19_M8_WITH_ABLATIONS` | original residual-score | 8 | Scheduler sensitivity / ablations |
| `B_ORIGINAL_V19_M4_WITH_ABLATIONS` | original residual-score | 4 | Stricter cap sensitivity / ablations |
| `C_EXPERIMENTAL_PUBLIC_SCORE_M8` | experimental public-score | 8 | Primary component comparison |

The three core matrices contain their own raw data, summaries, paired analysis, and validation reports. **Do not mix code versions when reproducing a named matrix.** Confirm code hashes and arguments in manifests before comparing with archived values.

The additional nine-setting sweep varies (threshold, cap): `(3,8)`, `(2,8)`, `(1,8)`, `(0.5,8)`, `(3,16)`, `(2,16)`, `(1,16)`, `(3,32)`, `(1,32)`. These are **observed operating points**, not target-coverage guarantees; cap-16 and cap-32 results should not automatically be called matched-service comparisons.

## Data access and preparation

Data are **not assumed to be embedded in the repository**. Follow each experiment's `DATA_SOURCES.md`, `opsd/data_provenance.json`, and `uci/data_provenance.json` for exact data provenance; original local Colab cache paths in manifests are **historical** and may not exist on your machine.

- **OPSD:** Open Power System Data household data, cited dataset version 2020-04-15: https://data.open-power-system-data.org/household_data/2020-04-15/
- **UCI:** ElectricityLoadDiagrams20112014: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

Download sources under the original usage/licensing terms. Prefer the implementation's own downloading/preprocessing workflow where available; check data provenance and resulting endpoint counts before interpreting any numbers. A fresh run may need internet access and enough disk/cache space.

**Consistency check before submission:** The archived OPSD `run_config.json` files inspected here specify `n_endpoints: 11`, whereas the previously supplied final-paper text reports **nine OPSD endpoints**. Verify whether the value is an allocation setting or the actual selected number of records, and reconcile the description with the exported `data_provenance.json` / implementation before stating exact endpoints. This README does not alter the original run configurations or results.

## Installation

Recommended: Python 3 on Linux or Google Colab, with sufficient CPU RAM/disk for complete training and explanation runs. GPU runtime alone will not accelerate NumPy, SciPy or CPU-bound SHAP.

```bash
# From the repository root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
python code/prism_xai_icc2027_v1_9_selective_release.py --help
python code/prism_xai_icc2027_v2_0_public_score_experimental.py --help
```

**Reproducibility environment:** Consult `provenance/pip_freeze.txt`, `provenance/python_version.txt` and `provenance/platform.txt`, plus the sweep-specific `SERVICE_TRADEOFF_RESULTS/provenance/pip_freeze.txt`. These are recorded environments, not a guarantee that installation from the unpinned requirements file yields bit-identical outputs. Always record package versions and dataset checksums in a fresh rerun.

## Reproducing experiments

### Stage 0 — Verify archived outputs without retraining

Start by reading the machine-readable run status and config for **each** matrix:

```bash
python - <<'PY'
import json, pathlib
for name in [
    'A_ORIGINAL_V19_M8_WITH_ABLATIONS',
    'B_ORIGINAL_V19_M4_WITH_ABLATIONS',
    'C_EXPERIMENTAL_PUBLIC_SCORE_M8']:
    d=pathlib.Path(name)
    print('\n',name)
    for dataset in ['opsd','uci']:
        p=d/dataset/'validation_report.json'
        print(dataset, json.loads(p.read_text()).get('passed') if p.exists() else 'MISSING')
    print('combined CSV:', (d/'combined_raw_run_metrics.csv').exists())
PY
```

For the nine-point sweep inspect `SERVICE_TRADEOFF_RESULTS/provenance/completed_runs.json` and `SERVICE_TRADEOFF_RESULTS/paper_evidence/all_method_means.csv`. These are already aggregated; do not append per-dataset files to combined CSVs.

### Stage 1 — Test the installation

```bash
python code/prism_xai_icc2027_v1_9_selective_release.py --real-suite verification --models tree --xai-methods feature_ablation --epsilon-budgets 8 16 32 --output-dir ./reproduced/verification
```

This smoke test is **not** equivalent to the full paper experiment.

### Stage 2 — Rerun the complete M8 and M4 matrices

The commands below specify primary settings; consult archived `opsd/run_config.json` and `uci/run_config.json` for complete parameter equivalence, especially seeds, datasets, deadlines and calibration windows.

```bash
# A: original residual-score, maximum 8 releases
python code/prism_xai_icc2027_v1_9_selective_release.py --real-suite full --models tree --xai-methods treeshap feature_ablation --epsilon-budgets 8 16 32 --max-private-releases-per-endpoint 8 --run-ablations --output-dir ./reproduced/A_M8

# B: original residual-score, maximum 4 releases
python code/prism_xai_icc2027_v1_9_selective_release.py --real-suite full --models tree --xai-methods treeshap feature_ablation --epsilon-budgets 8 16 32 --max-private-releases-per-endpoint 4 --run-ablations --output-dir ./reproduced/B_M4

# C: experimental public-score, maximum 8 releases
python code/prism_xai_icc2027_v2_0_public_score_experimental.py --real-suite full --models tree --xai-methods treeshap feature_ablation --epsilon-budgets 8 16 32 --max-private-releases-per-endpoint 8 --public-release-score-threshold 3.0 --run-ablations --output-dir ./reproduced/C_PUBLIC_M8
```

**Implementation identity matters.** The nine-point sweep used a specifically patched v1.9 script, recorded in `SERVICE_TRADEOFF_RESULTS/provenance/public_score_variant.patch`. The standalone v2.0 script is not asserted to be byte-equivalent to that patch. Compare source hashes before treating any rerun as exact reproduction.

### Stage 3 — Reproduce the nine-setting sweep

The authoritative commands and source hash are recorded in `SERVICE_TRADEOFF_RESULTS/provenance/completed_runs.json` and `source_manifest.json`. The command pattern is:

```bash
python path/to/prism_xai_public_score_tradeoff_variant.py --real-suite full --models tree --xai-methods treeshap feature_ablation --epsilon-budgets 8 16 32 --max-private-releases-per-endpoint 8 --public-release-score-threshold 3.0 --output-dir ./reproduced/sweep/thr_3_cap_8
```

Run all nine listed threshold/cap pairs; update the two CLI values and the output directory. Use the **patched source from the archived source bundle** and check its SHA-256 in the manifest. The sweep is computationally expensive; checkpoint after each setting.

## Paper-to-evidence roadmap

| Paper claim or result | Primary artifact | Checks |
|---|---|---|
| 720-run public-score main matrix | `C_EXPERIMENTAL_PUBLIC_SCORE_M8/combined_raw_run_metrics.csv` | `combined_validation_report.json`, per-dataset manifests |
| Residual-score M8 matrix | `A_ORIGINAL_V19_M8_WITH_ABLATIONS/combined_raw_run_metrics.csv` | `paper_evidence/A_*_paired_prism_vs_map.csv` |
| Residual-score M4 matrix | `B_ORIGINAL_V19_M4_WITH_ABLATIONS/combined_raw_run_metrics.csv` | `paper_evidence/B_*_paired_prism_vs_map.csv` |
| 15.72% traffic reduction, matched MAP-XAI comparison | `paper_evidence/C_EXPERIMENTAL_PUBLIC_SCORE_M8_paired_prism_vs_map.csv` | `paper_evidence/C_EXPERIMENTAL_PUBLIC_SCORE_M8_matched_region_audit.csv` |
| External baseline method means | `paper_evidence/C_EXPERIMENTAL_PUBLIC_SCORE_M8_method_means.csv` | `C_EXPERIMENTAL_PUBLIC_SCORE_M8/combined_summary_ci.csv` |
| OPSD and UCI separate results | `C_EXPERIMENTAL_PUBLIC_SCORE_M8/{opsd,uci}/raw_run_metrics.csv` | `summary_ci.csv` and `data_provenance.json` |
| Fidelity and Top-k/random comparisons | `paper_evidence/*_topk_vs_random.csv` | `appendix_condition_means.csv` |
| Ablations | `A_*/combined_ablation_metrics.csv`, `B_*/combined_ablation_metrics.csv`, `C_*/combined_ablation_metrics.csv` | corresponding per-dataset `ablation_metrics.csv` |
| Nine-setting release cap / threshold sweep | `SERVICE_TRADEOFF_RESULTS/paper_evidence/all_method_means.csv` | `runs/thr_*_cap_*/combined_raw_run_metrics.csv` |
| Paired sweep statistics | `SERVICE_TRADEOFF_RESULTS/paper_evidence/within_setting_paired_comparisons.csv` | `sweep_vs_original_paired.csv`, `service_overlap_audit.csv` |
| Traffic-coverage and traffic-recall plots | `SERVICE_TRADEOFF_RESULTS/paper_evidence/traffic_vs_coverage_prism_xai.pdf`, `traffic_vs_recall_prism.pdf` | `all_method_means.csv` |
| Validation and provenance | `provenance/`, `SERVICE_TRADEOFF_RESULTS/provenance/` | source hashes, environment, completed runs |
| Tables and paper source | `MAIN_PAPER/`, `supplementary/SUPPLEMENTARY/` | generated `*.tex` and figure paths |

## Understanding each result file

- `raw_run_metrics.csv`: per-dataset seed-level records; use for statistical comparisons.
- `combined_raw_run_metrics.csv`: pooled experiment records for one configuration; **do not double-count** with OPSD/UCI files.
- `summary_ci.csv` and `combined_summary_ci.csv`: exported aggregates and uncertainty summaries.
- `paired_prism_minus_baselines.csv`: paired PRISM-to-baseline contrasts.
- `run_config.json` / `run_manifest.json`: settings and execution provenance.
- `data_provenance.json`: input dataset and preprocessing provenance.
- `validation_report.json`: internal checks; **not** independent proof of full-transcript privacy.
- `representative_trace.csv`: illustrative trajectory, not an independent run.
- `.pdf` and `.png`: print/raster renditions of the same graph; not separate measurements.

## Interpreting the findings responsibly

**Matched-service comparisons:** The primary incremental comparison is PRISM-XAI vs MAP-XAI-only. Fixed-DP baselines operate at other service levels. Failure to detect a difference does not prove service equivalence; apply the specified coverage/recall/privacy matching criteria.

**Privacy boundary:** The content accounting relies on sensitivity, noise and admission assumptions. The communication transcript includes timing, silence and endpoint selection, which are not automatically covered.

**Service limits:** Absolute critical-event coverage and recall are low even after raising the release cap. Lower traffic is not a validated operational grid-control benefit.

**Simulation setting:** Wireless evaluation is a scheduler-level abstraction; it is not link-level or operator-in-the-loop validation.

**Archive caveat:** These are recorded outputs; their completeness and source files can be inspected, but the complete computation has not been rerun in a clean environment for this README.

## Compiling the manuscript

Main source: `MAIN_PAPER/PRISM_XAI_ICC2027_MAIN_SUBMISSION_READY.tex`.

Supplement source: `supplementary/SUPPLEMENTARY/PRISM_XAI_ICC2027_SUPPLEMENTARY.tex`.

Compile from the relevant directories using an IEEEtran-compatible LaTeX environment. The supplement in this repository may be an expanded archival version; confirm that it matches the appendix accompanying your final submission.

## Troubleshooting and publication checklist

- Paths beginning with `/content/` in archived manifests refer to the original Colab machine, not the user's machine.
- If a new run yields different results, first check source SHA-256, run config, dataset version, selected endpoints and seeds.
- GPU selection does not automatically accelerate NumPy/scikit-learn/CPU-based SHAP code.
- Save each completed sweep run to persistent storage; Colab ephemeral files may disappear.
- Before public release check for private data, copyright/licensing, review anonymity and manuscript links. No software license is implied by this README.
- Historical `SHA256SUMS` manifests reflect the previously archived files. An edited README will naturally have a new hash; do not treat that as evidence of changed experimental outputs.
