# PRISM-XAI — SmartGridComm 2027 submission package

## Contents
- `SGC2027_paper/PRISM_SGC2027.pdf`: the rewritten paper. It is five pages in standard IEEE two-column conference format (IEEEtran `[conference]`); SmartGridComm has historically allowed up to six pages.
- `SGC2027_paper/PRISM_SGC2027.tex`: LaTeX source. Every number comes from `results_macros.tex` and `table_*.tex`; none is typed by hand.
- `SGC2027_paper/fig_concept.tex`: Fig. 1, the pipeline/concept figure, drawn in TikZ.
- `analysis/make_figures.py`: regenerates every figure, macro and table from the result files in `data/`.
- `analysis/prop_sim.py`: mechanism-level check of Proposition 1. It uses the simulator's own release function and does not need a dataset.
- `analysis/frontier.py`: frontier-margin analysis on the held-out seeds.
- `colab/cell21_frontier_confirm.py`: OPTIONAL pre-registered confirmation of the frontier result on fresh seeds 3001-3010.
- `colab/cell19_*` and `colab/cell20_*`: the diagnosis and redesign cells that produced `data/driver_diagnosis_runs.csv` and `data/rc_results_v3_mech`.

## Rebuild
    python analysis/prop_sim.py        # needs the simulator path set inside the script
    python analysis/frontier.py
    python analysis/make_figures.py
    cd SGC2027_paper && pdflatex PRISM_SGC2027 && pdflatex PRISM_SGC2027

## Before submission (author to check)
1. The SmartGridComm 2027 CFP is not yet published. Confirm the page limit, the template and whether review is double-blind once it appears.
2. Repository link in the paper points to https://github.com/Poushali96/PRISMXAI2026.
3. Check the author list and affiliations.
4. Optionally run cell 21 to upgrade the frontier claim from "descriptive" to "confirmatory".
