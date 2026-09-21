PYTHON ?= python
ALPHAS ?= 0 0.97 1
ALPHA ?= 0.97
TARGET_ALPHAS = $(if $(filter command line,$(origin ALPHA)),$(ALPHA),$(ALPHAS))

.PHONY: report table dmop mif evaluate export

report:
	quarto render report.qmd

table:
	$(PYTHON) make_table.py

evaluate:
	$(PYTHON) evaluate.py --source results
	$(PYTHON) evaluate.py --source results --best

export:
	$(PYTHON) export_results.py --source results --checkpoints checkpoint_results --reevaluation reevaluation

dmop:
	mkdir -p results/logs
	@for alpha in $(TARGET_ALPHAS); do \
		sbatch --job-name="daphnia_ifad_$$alpha" --export=ALL,ALPHA="$$alpha" dmop_search.sbat; \
	done

mif:
	mkdir -p results/logs
	sbatch mif_search.sbat
