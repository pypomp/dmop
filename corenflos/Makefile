PYTHON ?= /home/kevin/anaconda3/envs/pypomp/bin/python
PYTHONPATH := .:../ditlevsen:../../pypomp

.PHONY: test plots

test:
	PYTHONPATH=$(PYTHONPATH) XLA_PYTHON_CLIENT_PREALLOCATE=false $(PYTHON) -m pytest -q tests

plots:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m corenflos.plots --data results/final_100
