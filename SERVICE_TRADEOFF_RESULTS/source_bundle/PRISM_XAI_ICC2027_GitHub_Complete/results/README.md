# Results index

## Current v19 evidence

- `M8_main_720_runs/` — primary v19 operating point and the main 720-run experiment.
- `M4_sensitivity/` — lower private-release-cap sensitivity analysis.
- `verification/` — smaller engineering verification run.
- `feasibility_ladder/` — mechanism diagnostics, privacy-release sweep, and scheduler A–F diagnostics.

Each main result directory contains machine-readable CSV/JSON files, plots, LaTeX tables,
run manifests/configuration, data provenance, and validation reports when generated.

## Legacy v18 evidence

The `legacy_v18/` folder is included only for reproducibility/history:
- tree XAI sensitivity,
- neural XAI sensitivity,
- v18 sensitivity verification.

The current v19 manuscript should be supported with v19 evidence, not by silently mixing older v18 outputs.

## External datasets

Raw downloaded OPSD/UCI caches are intentionally omitted. The code downloads public data and records provenance.
