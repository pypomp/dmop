# Default parameters
ALPHAS ?= 0 0.97 1
ALPHA ?= 0.97
RUN_LEVEL ?= 4
N_MONITORS ?= 1
N_MONITORS_LIST ?= 0 1

# GPU choice: 'rtx' (default, gpu-rtx6000 with rtx_pro_6000_blackwell:1) or 'v100' (gpu partition with v100:1)
GPU ?= rtx

ifeq ($(GPU),v100)
GPU_PARTITION = gpu
GPU_SPEC = v100:1
GPU_TIME = 0-2:00:00
else
GPU_PARTITION = gpu-rtx6000
GPU_SPEC = rtx_pro_6000_blackwell:1
GPU_TIME = 0-0:30:00
endif

# Use ALPHA if it was explicitly provided on the command line (e.g. make dmop ALPHA=0.5)
# Otherwise, use the full list of ALPHAS
TARGET_ALPHAS = $(if $(filter command line,$(origin ALPHA)),$(ALPHA),$(ALPHAS))
TARGET_N_MONITORS = $(if $(filter command line,$(origin N_MONITORS)),$(N_MONITORS),$(N_MONITORS_LIST))

.PHONY: help dmop mif report precise_report all clean

help:
	@echo "Available targets:"
	@echo "  make dmop     - Submit Slurm jobs for DMOP search"
	@echo "                  Default ALPHAS: $(ALPHAS)"
	@echo "                  Default N_MONITORS: $(N_MONITORS_LIST)"
	@echo "                  Default GPU: $(GPU) (options: rtx, v100)"
	@echo "                  Overrides: 'make dmop ALPHA=0.5' or 'make dmop N_MONITORS=0' or 'make dmop GPU=v100'"
	@echo "  make mif      - Submit Slurm jobs for MIF search"
	@echo "                  Default N_MONITORS: $(N_MONITORS_LIST)"
	@echo "                  Default GPU: $(GPU) (options: rtx, v100)"
	@echo "                  Overrides: 'make mif N_MONITORS=0' or 'make mif GPU=v100'"
	@echo "  make report   - Render report.qmd"
	@echo "                  Params: ALPHA=$(ALPHA), N_MONITORS=$(N_MONITORS), RUN_LEVEL=$(RUN_LEVEL)"
	@echo "  make precise_report - Render precise_report.qmd"
	@echo "  make all      - Run both dmop and mif targets"
	@echo "  make clean    - Remove slurm log files"

all: dmop mif

dmop:
	@for alpha in $(TARGET_ALPHAS); do \
		for nm in $(TARGET_N_MONITORS); do \
			echo "Submitting dmop_search.sbat with ALPHA=$$alpha N_MONITORS=$$nm on GPU=$(GPU) ($(GPU_PARTITION):$(GPU_SPEC), time $(GPU_TIME))"; \
			sbatch --job-name="dmop_a$${alpha}_nm$${nm}" --partition=$(GPU_PARTITION) --gpus=$(GPU_SPEC) --time=$(GPU_TIME) --export=ALL,ALPHA="$$alpha",N_MONITORS="$$nm" dmop_search.sbat; \
		done \
	done

mif:
	@for nm in $(TARGET_N_MONITORS); do \
		echo "Submitting mif_search.sbat with N_MONITORS=$$nm on GPU=$(GPU) ($(GPU_PARTITION):$(GPU_SPEC), time $(GPU_TIME))"; \
		sbatch --job-name="mif_nm$${nm}" --partition=$(GPU_PARTITION) --gpus=$(GPU_SPEC) --time=$(GPU_TIME) --export=ALL,N_MONITORS="$$nm" mif_search.sbat; \
	done

report:
	@echo "Rendering report.qmd (ALPHA=$(ALPHA), N_MONITORS=$(N_MONITORS), RUN_LEVEL=$(RUN_LEVEL))"
	ALPHA=$(ALPHA) N_MONITORS=$(N_MONITORS) RUN_LEVEL=$(RUN_LEVEL) quarto render report.qmd

precise_report:
	@echo "Rendering precise_report.qmd"
	quarto render precise_report.qmd

report_job:
	@echo "Submitting report.sbat (ALPHA=$(ALPHA), N_MONITORS=$(N_MONITORS), RUN_LEVEL=$(RUN_LEVEL))"
	sbatch --export=ALL,ALPHA=$(ALPHA),N_MONITORS=$(N_MONITORS),RUN_LEVEL=$(RUN_LEVEL) report.sbat

clean:
	rm -f dmop_results/logs/slurm-*.out
	rm -f mif_results/logs/slurm-*.out
	rm -f slurm-report-*.out
