# PRISM-XAI ICC 2027 updated submission package

This package contains the six-page main manuscript, the complete supplementary
material, source code, previous validated result matrices, and the full
public-score service--communication sweep.

## Canonical submission files

- `MAIN_PAPER/PRISM_XAI_ICC2027_MAIN_SUBMISSION_READY.tex`
- `MAIN_PAPER/PRISM_XAI_ICC2027_MAIN_SUBMISSION_READY.pdf` (6 pages)
- `supplementary/SUPPLEMENTARY/PRISM_XAI_ICC2027_SUPPLEMENTARY.tex`
- `supplementary/SUPPLEMENTARY/PRISM_XAI_ICC2027_SUPPLEMENTARY.pdf` (309 pages)

The main paper adds one concise paragraph that summarizes the service sweep and
directs the reader to Supplementary Sec. VIII. The supplementary material
contains the full numerical interpretation, proofs, paired statistics,
condition tables, and aggregate/OPSD/UCI figures.

## New Service-Tradeoff Evidence

`SERVICE_TRADEOFF_RESULTS/` preserves the uploaded archive in full:

- nine completed public-score threshold/release-cap settings;
- 6,480 policy runs (720 per setting; 120 per method);
- raw CSV metrics, representative traces, validation JSON, run manifests,
  source hashes, execution logs, generated LaTeX tables, and PDF/PNG figures;
- cross-setting evidence tables and traffic--coverage/recall frontier figures.

The supplement embeds every distinct vector result figure and every generated
LaTeX table. PNG files are retained as duplicate raster exports for
reproducibility but are not embedded in the PDF because the vector versions are
print quality.

## Interpretation boundary

Across the nine observed settings, PRISM-XAI uses less mean traffic than
MAP-XAI-only. The declared all-cell matched-region condition holds for the four
settings with release cap M=8; larger-cap settings are reported as descriptive
service-frontier sensitivity results. Fixed-DP baselines remain unmatched
operating points. Privacy claims concern the accounted packet contents under
the stated mechanism and assumptions.

## Build

Compile each canonical `.tex` file from its own directory with Tectonic or an
IEEEtran-compatible LaTeX installation. `IEEEtran.cls` is included alongside
the main and supplementary sources. The supplement reads figures and generated
tables from relative paths inside this package.

`SHA256SUMS_UPDATED.txt` records package file hashes.
