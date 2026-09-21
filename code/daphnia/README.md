# Daphnia

The SIRJPF2 model and data are from the [advanced Daphnia tutorial](https://github.com/pypomp/Daphnia-tutorial/blob/f3183301273a409e5b079bb0ae3a22246f69ba7a/Python-code/daphnia_tut_pypomp_advanced.qmd).
The comparison uses 50 shared starts for the 38-parameter model: IFAD with 200 MPIF iterations followed by 200 Adam updates, and MPIF with 680 iterations. Settings and results are in [report.html](report.html).

Install `requirements.txt` with Python 3.12 and install Quarto. Run `make report table` to reproduce the figures and SI table from the three files in `exports/`.

For a new experiment, set `SBATCH_ACCOUNT` and run `make dmop` and `make mif`. Then run `make evaluate export report table` to evaluate the saved checkpoints and best estimates and generate the report.
