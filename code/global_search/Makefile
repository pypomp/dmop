# Default parameters
ALPHAS ?= 0 0.97 1
ALPHA ?= 0.97
RUN_LEVEL ?= 4
N_MONITORS ?= 1
N_MONITORS_LIST ?= 0 1

# Use ALPHA if it was explicitly provided on the command line (e.g. make dmop ALPHA=0.5)
# Otherwise, use the full list of ALPHAS
TARGET_ALPHAS = $(if $(filter command line,$(origin ALPHA)),$(ALPHA),$(ALPHAS))
TARGET_N_MONITORS = $(if $(filter command line,$(origin N_MONITORS)),$(N_MONITORS),$(N_MONITORS_LIST))

.PHONY: help dmop mif report all clean

help:
	@echo "Available targets:"
	@echo "  make dmop     - Submit Slurm jobs for DMOP search"
	@echo "                  Default ALPHAS: $(ALPHAS)"
	@echo "                  Default N_MONITORS: $(N_MONITORS_LIST)"
	@echo "                  Overrides: 'make dmop ALPHA=0.5' or 'make dmop N_MONITORS=0'"
	@echo "  make mif      - Submit Slurm jobs for MIF search"
	@echo "                  Default N_MONITORS: $(N_MONITORS_LIST)"
	@echo "  make report   - Render report.qmd"
	@echo "                  Params: ALPHA=$(ALPHA), N_MONITORS=$(N_MONITORS), RUN_LEVEL=$(RUN_LEVEL)"
	@echo "  make all      - Run both dmop and mif targets"
	@echo "  make clean    - Remove slurm log files"

all: dmop mif

dmop:
	@for alpha in $(TARGET_ALPHAS); do \
		for nm in $(TARGET_N_MONITORS); do \
			echo "Submitting dmop_search.sbat with ALPHA=$$alpha N_MONITORS=$$nm"; \
			sbatch --job-name="dmop_a$${alpha}_nm$${nm}" --export=ALL,ALPHA="$$alpha",N_MONITORS="$$nm" dmop_search.sbat; \
		done \
	done

mif:
	@for nm in $(TARGET_N_MONITORS); do \
		echo "Submitting mif_search.sbat with N_MONITORS=$$nm"; \
		sbatch --job-name="mif_nm$${nm}" --export=ALL,N_MONITORS="$$nm" mif_search.sbat; \
	done

report:
	@echo "Rendering report.qmd (ALPHA=$(ALPHA), N_MONITORS=$(N_MONITORS), RUN_LEVEL=$(RUN_LEVEL))"
	ALPHA=$(ALPHA) N_MONITORS=$(N_MONITORS) RUN_LEVEL=$(RUN_LEVEL) quarto render report.qmd

clean:
	rm -f dmop_results/logs/slurm-*.out
	rm -f mif_results/logs/slurm-*.out
