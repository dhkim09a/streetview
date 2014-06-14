
title = report
reports = $(title).tex
bibs = $(title).bib
latex = pdflatex

all: pdf

pdf: $(reports)
	$(latex) $(reports)
	bibtex $(title)
	$(latex) $(reports)
	$(latex) $(reports)

clean:
	rm -rf $(title).log $(title).bbl $(title).blg $(title).dvi
	rm -rf $(title).pdf $(title).ps $(title).aux
	rm -rf *~

