PYTHON ?= /home/kevin/anaconda3/envs/pypomp/bin/python
PYTHONPATH := .:../ditlevsen:../../pypomp
RESULTS ?= results/corenflos_j100_eps025_final_100

.PHONY: test plots

test:
	PYTHONPATH=$(PYTHONPATH) XLA_PYTHON_CLIENT_PREALLOCATE=false $(PYTHON) -m pytest -q tests

plots:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m corenflos.plots --data $(RESULTS)
