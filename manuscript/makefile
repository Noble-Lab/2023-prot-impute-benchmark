all: 2023benchmark.pdf

%.pdf: %.tex
	pdflatex $*.tex
	if ( grep -q citation $*.aux ) ; then \
		bibtex $* ; \
		pdflatex $*.tex ; \
	fi
	pdflatex $*.tex

%.pdf: %.fig
	fig2dev -L pdftex -p dummyarg $*.fig $*.pdf

clean:
	rm -f *.dvi *.bbl *.aux *.blg *.log *.ps *.out

