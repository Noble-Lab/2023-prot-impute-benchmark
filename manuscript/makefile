all: 2023benchmark.pdf \
	2023benchmark-supporting.pdf

FIGURES=figures/literature-survey-figure.pdf \
	figures/traditional-evaluation-figure.pdf \
	figures/differential-expression-DIA-peptide-figure.pdf \
	figures/differential-expression-DIA-protein-figure.pdf \
	figures/n-quant-peptides-figure.pdf \
	figures/lloqs-figure.pdf \
	figures/runtimes-figure.pdf \
	figures/mean-variance-peptide-figure.pdf

%.pdf: %.tex
	pdflatex $*.tex
	if ( grep -q citation $*.aux ) ; then \
		bibtex $* ; \
		pdflatex $*.tex ; \
	fi
	pdflatex $*.tex

2023benchmark.pdf: 2023benchmark.tex $(FIGURES)

%.pdf: %.fig
	fig2dev -L pdftex -p dummyarg $*.fig $*.pdf

clean:
	rm -f *.dvi *.bbl *.aux *.blg *.log *.ps *.out *.pdf

