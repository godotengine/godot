BASEDIR = $(CURDIR)
CLASSES = $(BASEDIR)/classes/ $(BASEDIR)/../modules/
OUTPUTDIR = $(BASEDIR)/_build
TOOLSDIR = $(BASEDIR)/tools

.ONESHELL:

clean:
	rm -rf $(OUTPUTDIR)

doxygen:
	rm -rf $(OUTPUTDIR)/doxygen
	mkdir -p $(OUTPUTDIR)/doxygen
	doxygen Doxyfile

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
	python $(TOOLSDIR)/makerst.py $(CLASSES)
	popd
