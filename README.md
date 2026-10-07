# Automatic Differentiation Accelerates Inference for Partially Observed Markov Processes

* This repository provides source code for a new differentiable particle filter.

* It develops the approach described in an arXiv article <https://doi.org/10.48550/arXiv.2407.03085>. 

* The source for the original arXiv submission is on Zenodo <https://zenodo.org/doi/10.5281/zenodo.13356896>.

Experiment archives and restoration instructions are in
[artifacts/README.md](artifacts/README.md). Compact configurations, summaries,
and figures remain tracked; bulk results stay at their existing local paths.

Figure 1A (`imgs/095/mop.png`) can be redrawn with
`python code/fig1a/plot.py`. Its recovered notebook provenance, self-contained
model/filter code, input data, and full recomputation instructions are in
[code/fig1a/README.md](code/fig1a/README.md).

The Ditlevsen-style and Corenflos comparisons are in SI Section S11.
Regenerate their tables with `python code/competitor_results.py`;
see [imgs/competitors/README.md](imgs/competitors/README.md) for the data sources.
