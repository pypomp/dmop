# Dhaka global search: IF2 versus IFAD

This directory runs the Dhaka cholera comparison of IF2 with IFAD-0,
IFAD-0.97, and IFAD-1 in the main text. It produces Table 1 (`table:mle`),
Figures 3–5 (`fig:boxplot`, `fig:density`, `fig:optim`), and the settings
in SI Table S-1 (`table:algparams`). Each method runs 100 searches from
starting values drawn uniformly from the box of Ionides et al. (2015). We
run each search twice: once with comparable effort (about 15 minutes) and
once with extended effort, which uses four times as many IF2-only or Adam
iterations.

## Files used in the paper

| File | Role |
|---|---|
| [prep.py](prep.py) | Shared setup: model, seed, random-walk SDs, starting box, and run options from environment variables |
| [mif_search.py](mif_search.py), [mif_search.sbat](mif_search.sbat) | IF2-only searches |
| [dmop_search.py](dmop_search.py), [dmop_search.sbat](dmop_search.sbat) | IFAD searches: IF2 warm start followed by Adam |
| [Makefile](Makefile) | Submits the Slurm jobs and renders the report |
| [precise_report.qmd](precise_report.qmd), [precise_report.sbat](precise_report.sbat) | Writes `imgs/precise_raincloud{,_long}.pdf`, `precise_density_long.pdf`, and `precise_traces_long.pdf` |
| [make_table.py](make_table.py) | Writes `imgs/precise_table.tex` |
| [export_results.py](export_results.py), [exports/](exports/) | CSV copies of final estimates, log-likelihood traces, and timings |
| [requirements.txt](requirements.txt) | Frozen environment for the reported runs |

We run every configuration twice. Runs with `N_MONITORS=1` evaluate the
log-likelihood at each iteration, and they supply all log-likelihoods and
estimates in the paper. Runs with `N_MONITORS=0` skip those extra particle
filter calls, so they supply the reported times. `RUN_LEVEL=4` is the
paper's size; levels 1–3 are smaller test runs.

The result pickles (`dmop_results/`, `mif_results/`) are not in the
repository. They are large, and only the Pypomp version that wrote them can
read them. `exports/` holds the reported values in plain CSV files instead.
The plots and table themselves are built from the pickles.

## Reproducing the results

The reported runs used Pypomp commit `17f8798` (package code identical to
v1.0.5) and JAX 0.11.2 on a CUDA GPU. The batch scripts activate
`../../.venv`, which should be built from `requirements.txt`. From this
directory on a Slurm cluster:

```sh
make dmop mif                    # comparable effort
make dmop_long mif_long          # extended effort
make precise_report precise_report_long
python make_table.py
python export_results.py
```

Exact log-likelihoods can vary with hardware and JAX version.

## Historical files

[report.qmd](report.qmd) and [report.sbat](report.sbat), together with the
Makefile's `report` and `report_job` targets, are a per-configuration report
from when we were choosing optimal search settings. Nothing in the paper comes from
them. They were last updated in May 2026, before the Pypomp 1.0.5 rerun. The
only figure they write, `imgs/ll_boxplot_global.pdf`, is not used.
`precise_report.qmd` also writes comparable-effort versions of the density
and trace plots (`imgs/precise_density.pdf`, `imgs/precise_traces.pdf`),
which the paper does not include.
