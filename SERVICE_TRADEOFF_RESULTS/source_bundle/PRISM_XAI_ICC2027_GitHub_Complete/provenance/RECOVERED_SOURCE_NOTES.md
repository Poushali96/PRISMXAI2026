# Provenance notes

This repository was assembled from three grounded sources:

1. The latest PRISM-XAI manuscript archive uploaded on 2026-09-01.
2. The uploaded Colab notebook `run_prism_xai_v18_matched_fallback_colab (1).ipynb`.
3. The recovered `PRISM_XAI_ALL_RESULTS_CLEAN_NO_DATACACHE` archive created by that notebook.

The clean results archive contained 364 files and the following top-level experiment artifacts:
- v18 tree XAI sensitivity results,
- v18 neural XAI sensitivity results,
- v18 sensitivity verification,
- v19 selective-release M4,
- v19 selective-release M8,
- v19 selective-release verification,
- feasibility-ladder diagnostics,
- v1.8 matched-fallback script,
- v1.9 selective-release script.

The repository reorganizes these into current-v19 and legacy-v18 directories but does not alter the result-file contents.
Downloaded dataset caches were excluded in the original clean archive and remain excluded here.
