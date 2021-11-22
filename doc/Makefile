BASEDIR = $(CURDIR)
CLASSES = $(BASEDIR)/classes/ $(BASEDIR)/../modules/
OUTPUTDIR = $(BASEDIR)/_build
TOOLSDIR = $(BASEDIR)/tools
JSDIR = $(BASEDIR)/../platform/javascript

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
	python3 $(TOOLSDIR)/make_rst.py -o $(OUTPUTDIR)/rst $(CLASSES)

rstjs:
	rm -rf $(OUTPUTDIR)/rstjs
	mkdir -p $(OUTPUTDIR)/rstjs
	npm --prefix $(JSDIR) ci
	npm --prefix $(JSDIR) run docs -- --destination $(OUTPUTDIR)/rstjs/html5_shell_classref.rst
