# Default parameters
ALPHAS ?= 0 0.97 1
ALPHA ?= 0.97
RUN_LEVEL ?= 4
N_MONITORS ?= 1
N_MONITORS_LIST ?= 0 1
LONG ?= false

# GPU choice: 'rtx' (default, gpu-rtx6000 with rtx_pro_6000_blackwell:1) or 'v100' (gpu partition with v100:1)
GPU ?= rtx

ifeq ($(GPU),v100)
GPU_PARTITION = gpu
GPU_SPEC = v100:1
ifeq ($(LONG),true)
GPU_TIME = 0-8:00:00
else
GPU_TIME = 0-2:00:00
endif
else
GPU_PARTITION = gpu-rtx6000
GPU_SPEC = rtx_pro_6000_blackwell:1
ifeq ($(LONG),true)
GPU_TIME = 0-2:00:00
else
GPU_TIME = 0-0:30:00
endif
endif

# Use ALPHA if it was explicitly provided on the command line (e.g. make dmop ALPHA=0.5)
# Otherwise, use the full list of ALPHAS
TARGET_ALPHAS = $(if $(filter command line,$(origin ALPHA)),$(ALPHA),$(ALPHAS))
TARGET_N_MONITORS = $(if $(filter command line,$(origin N_MONITORS)),$(N_MONITORS),$(N_MONITORS_LIST))

.PHONY: help dmop mif report precise_report all clean dmop_long mif_long precise_report_long

help:
	@echo "Available targets:"
	@echo "  make dmop     - Submit Slurm jobs for DMOP search"
	@echo "                  Default ALPHAS: $(ALPHAS)"
	@echo "                  Default N_MONITORS: $(N_MONITORS_LIST)"
	@echo "                  Default GPU: $(GPU) (options: rtx, v100)"
	@echo "                  Overrides: 'make dmop ALPHA=0.5' or 'make dmop N_MONITORS=0' or 'make dmop GPU=v100'"
	@echo "  make dmop_long - Submit Slurm jobs for long DMOP search (quadrupled training iterations)"
	@echo "  make mif      - Submit Slurm jobs for MIF search"
	@echo "                  Default N_MONITORS: $(N_MONITORS_LIST)"
	@echo "                  Default GPU: $(GPU) (options: rtx, v100)"
	@echo "                  Overrides: 'make mif N_MONITORS=0' or 'make mif GPU=v100'"
	@echo "  make mif_long - Submit Slurm jobs for long MIF search (quadrupled IF2 iterations)"
	@echo "  make report   - Render report.qmd"
	@echo "                  Params: ALPHA=$(ALPHA), N_MONITORS=$(N_MONITORS), RUN_LEVEL=$(RUN_LEVEL), LONG=$(LONG)"
	@echo "  make precise_report - Render precise_report.qmd"
	@echo "  make precise_report_long - Render precise_report.qmd with LONG=true"
	@echo "  make all      - Run both dmop and mif targets"
	@echo "  make clean    - Remove slurm log files"

all: dmop mif

dmop:
	@for alpha in $(TARGET_ALPHAS); do \
		for nm in $(TARGET_N_MONITORS); do \
			echo "Submitting dmop_search.sbat with ALPHA=$$alpha N_MONITORS=$$nm LONG=$(LONG) on GPU=$(GPU) ($(GPU_PARTITION):$(GPU_SPEC), time $(GPU_TIME))"; \
			sbatch --job-name="dmop_a$${alpha}_nm$${nm}$(if $(filter true,$(LONG)),_l,)" --partition=$(GPU_PARTITION) --gpus=$(GPU_SPEC) --time=$(GPU_TIME) --export=ALL,ALPHA="$$alpha",N_MONITORS="$$nm",LONG="$(LONG)" dmop_search.sbat; \
		done \
	done

dmop_long:
	$(MAKE) dmop LONG=true

mif:
	@for nm in $(TARGET_N_MONITORS); do \
		echo "Submitting mif_search.sbat with N_MONITORS=$$nm LONG=$(LONG) on GPU=$(GPU) ($(GPU_PARTITION):$(GPU_SPEC), time $(GPU_TIME))"; \
		sbatch --job-name="mif_nm$${nm}$(if $(filter true,$(LONG)),_l,)" --partition=$(GPU_PARTITION) --gpus=$(GPU_SPEC) --time=$(GPU_TIME) --export=ALL,N_MONITORS="$$nm",LONG="$(LONG)" mif_search.sbat; \
	done

mif_long:
	$(MAKE) mif LONG=true

report:
	@echo "Rendering report.qmd (ALPHA=$(ALPHA), N_MONITORS=$(N_MONITORS), RUN_LEVEL=$(RUN_LEVEL), LONG=$(LONG))"
	ALPHA=$(ALPHA) N_MONITORS=$(N_MONITORS) RUN_LEVEL=$(RUN_LEVEL) LONG=$(LONG) quarto render report.qmd

precise_report:
	@echo "Rendering precise_report.qmd (LONG=$(LONG))"
	LONG=$(LONG) quarto render precise_report.qmd --output $(if $(filter true,$(LONG)),precise_report_long.html,precise_report.html)

precise_report_long:
	$(MAKE) precise_report LONG=true

report_job:
	@echo "Submitting report.sbat (ALPHA=$(ALPHA), N_MONITORS=$(N_MONITORS), RUN_LEVEL=$(RUN_LEVEL), LONG=$(LONG))"
	sbatch --export=ALL,ALPHA=$(ALPHA),N_MONITORS=$(N_MONITORS),RUN_LEVEL=$(RUN_LEVEL),LONG=$(LONG) report.sbat

clean:
	rm -f dmop_results/logs/slurm-*.out
	rm -f mif_results/logs/slurm-*.out
	rm -f slurm-report-*.out
