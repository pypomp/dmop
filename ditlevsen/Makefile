PYTHON ?= /home/kevin/anaconda3/envs/pypomp/bin/python

.PHONY: all test validate withdrawn-qml-figures clean

all: method.pdf invertibility_note.pdf

test:
	PYTHONPATH=. $(PYTHON) -m pytest -q tests

validate:
	PYTHONPATH=.:../.. $(PYTHON) -m ditlevsen.validate

withdrawn-qml-figures:
	PYTHONPATH=.:../.. $(PYTHON) -m ditlevsen.reference_results --output results/reference
	PYTHONPATH=.:../.. $(PYTHON) -m ditlevsen.plot_results --data results/withdrawn_qml/data --krylov-data results/withdrawn_qml/data_krylov --reference results/reference --timed-data results/withdrawn_qml/data_timed --output results/withdrawn_qml

method.pdf: method.tex
	pdflatex -interaction=nonstopmode -halt-on-error method.tex
	pdflatex -interaction=nonstopmode -halt-on-error method.tex

invertibility_note.pdf: invertibility_note.tex
	pdflatex -interaction=nonstopmode -halt-on-error invertibility_note.tex
	pdflatex -interaction=nonstopmode -halt-on-error invertibility_note.tex

clean:
	rm -f method.aux method.log method.out method.pdf
	rm -f invertibility_note.aux invertibility_note.log invertibility_note.out invertibility_note.pdf
	rm -f results/report.aux results/report.log results/report.out results/report.pdf
