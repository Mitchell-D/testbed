#!/bin/bash
pdflatex -output-directory out dodson_masters-thesis.tex ;
#bibtex out/dodson_masters-thesis ;
biber --input-directory out --output-directory out dodson_masters-thesis ;
pdflatex -output-directory out dodson_masters-thesis.tex ;
pdflatex -output-directory out dodson_masters-thesis.tex
