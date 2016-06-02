BASEDIR = $(CURDIR)
CLASSES = $(BASEDIR)/base/classes.xml
OUTPUTDIR = $(BASEDIR)/_build
TOOLSDIR = $(BASEDIR)/tools

.ONESHELL:

clean:
	rm -rf $(OUTPUTDIR)

doku:
	rm -rf $(OUTPUTDIR)/doku
	mkdir -p $(OUTPUTDIR)/doku
	pushd $(OUTPUTDIR)/doku
	python2 $(TOOLSDIR)/makedoku.py $(CLASSES)
	popd

doxygen:
	rm -rf $(OUTPUTDIR)/doxygen
	mkdir -p $(OUTPUTDIR)/doxygen
	doxygen Doxyfile

html:
	rm -rf $(OUTPUTDIR)/html
	mkdir -p  $(OUTPUTDIR)/html
	pushd $(OUTPUTDIR)/html
	python2 $(TOOLSDIR)/makehtml.py -multipage $(CLASSES)
	popd

markdown:
	rm -rf $(OUTPUTDIR)/markdown
	mkdir -p $(OUTPUTDIR)/markdown
	pushd $(OUTPUTDIR)/markdown
	python2 $(TOOLSDIR)/makemd.py $(CLASSES)
	popd

rst:
	rm -rf $(OUTPUTDIR)/rst
	mkdir -p $(OUTPUTDIR)/rst
	pushd $(OUTPUTDIR)/rst
	python2 $(TOOLSDIR)/makerst.py $(CLASSES)
	popd

textile:
	rm -rf $(OUTPUTDIR)/textile
	mkdir -p $(OUTPUTDIR)/textile
	python3 $(TOOLSDIR)/makedocs.py --input $(CLASSES) --output $(OUTPUTDIR)/textile
