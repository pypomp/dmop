# Automatic Differentiation for Inference on Partially Observed Markov Processes Models

This repository contains the manuscript, supplementary material, and numerical
experiments for the DMOP paper ([arXiv.2407.03085](https://doi.org/10.48550/arXiv.2407.03085)).
Development for arXiv version v2 takes place here. The older
`pypomp/quant/dacca_resubmission` directory is deprecated.

## Where to start

| Topic | Source and instructions | Manuscript output |
|---|---|---|
| Manuscript and supplement | [ms.tex](ms.tex), [si.tex](si.tex) | Main text and SI |
| MOP likelihood illustration | [code/fig1a/](code/fig1a/README.md) | Fig. 1A, `imgs/095/mop.png` |
| Dhaka IF2/IFAD searches | [code/global_search/](code/global_search/) | Main Dhaka comparison |
| DS19 comparison | [ditlevsen/](ditlevsen/README.md) | SI Sections S11, S12 |
| CTDD21 comparison | [corenflos/](corenflos/README.md) | SI Sections S11, S12 |
| Combined comparison tables and plots | [imgs/competitors/](imgs/competitors/README.md) | SI Tables S7-S11 and Figs. S7–S16 |
| Daphnia experiment | [code/daphnia/](code/daphnia/README.md) | SI Section S10 |
| Other analysis code and older notebooks | [code/](code/README.md) | Code directory guide |
| Bulk experiment outputs | [artifacts/](artifacts/README.md) | Archive inventory and restoration |

## Redraw Fig. 1A or rebuild the comparison tables

From the repository root, with NumPy, pandas, and Matplotlib installed:

```sh
python code/fig1a/plot.py
python artifacts/restore_results.py --prefix corenflos/results --prefix ditlevsen/results
python code/competitor_results.py
```

Both commands use files tracked in this repository. They do not rerun fitting
experiments. Full Fig. 1A recomputation is documented in its directory;
comparison setup, fitting, evaluation, and plotting are documented in
[code/competitor_reproduction.md](code/competitor_reproduction.md).

A fresh clone includes the figures and tables needed to compile the manuscript
and supplement. Saved result files are bundled under `artifacts/packed/` to
keep the repository below Overleaf's file limit. Before regenerating figures
or auditing a run, restore its result directory with
[the archive helper](artifacts/COMPACT_HISTORY.md). Older bulk outputs that
were already untracked remain in the separate local October 6 archive.

## Earlier versions

The [original arXiv submission archive](https://zenodo.org/doi/10.5281/zenodo.13356896)
contains an earlier version of the project. It is not an archive of the current
revision. The recovered source of Fig. 1A is documented separately in
[code/fig1a/README.md](code/fig1a/README.md).
