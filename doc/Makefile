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

rst:
	rm -rf $(OUTPUTDIR)/rst
	mkdir -p $(OUTPUTDIR)/rst
	pushd $(OUTPUTDIR)/rst
	python3 $(TOOLSDIR)/makerst.py $(CLASSES)
	popd
