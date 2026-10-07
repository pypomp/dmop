# Code for the DMOP manuscript

Start with the directory for the experiment or figure you want to reproduce.
Ditlevsen and Corenflos have their own directories at the repository root;
the small script here combines their saved results into SI tables.

| File or directory | Purpose | Related files |
|---|---|---|
| [fig1a/](fig1a/README.md) | Recompute or redraw the MOP likelihood illustration | [Fig. 1A PNG](../imgs/095/mop.png) |
| [competitor_results.py](competitor_results.py) | Build SI Tables S4–S5 and CSV summaries from saved evaluations | [Comparison outputs](../imgs/competitors/README.md) |
| [competitor_reproduction.md](competitor_reproduction.md) | Shared environment and reproduction guide for S11 | [Ditlevsen](../ditlevsen/README.md), [Corenflos](../corenflos/README.md) |
| [global_search/](global_search/) | Dhaka IF2/IFAD fitting, evaluation, export, and reports | `Makefile`, `precise_report.qmd`, and `exports/` in that directory |
| [daphnia/](daphnia/README.md) | Daphnia model, searches, evaluation, and report | SI Section S10 |
| [r_benchmark/](r_benchmark/) | R-side performance comparison | Main-text computational efficiency experiment |
| [dacca.ipynb](dacca.ipynb), [dacca_analysis.ipynb](dacca_analysis.ipynb) | Earlier Dhaka notebook analyses | Historical material; use the experiment guides above for the revision |

The loose `mif*.csv`, `pre*.csv`, `runs*.csv`, and
`cholera-mif1-mif2.rda` belong to the earlier analyses. `matplotlibrc` supplies
plotting defaults. Fig. 1A has its own saved curves, input data, and style in
`fig1a/`; its redraw does not depend on these older notebooks or CSVs.
The Fig. 1B image is `../imgs/095/biasvar.png`; the `fig1a` scripts generate
panel A only.

## Quick commands

Run from the repository root:

```sh
python code/fig1a/plot.py
python code/competitor_results.py
```

These use tracked inputs. The comparison guide distinguishes these short
commands from GPU fitting and operations requiring the
[bulk-output archive](../artifacts/README.md).

## Historical README

The following text records the original location of the code. Development
has moved from `quant/dacca_resubmission` to this repository.


This code is taken from <https://github.com/pypomp/quant/tree/main/dacca_resubmission>.

There is other code at <https://github.com/hetankevin/diffpomp>

The Daphnia experiment in Section S10 is in [daphnia/](daphnia/).
