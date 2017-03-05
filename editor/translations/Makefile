# Makefile providing various facilities to manage translations

TEMPLATE = editor.pot
POFILES = $(wildcard *.po)
LANGS = $(POFILES:%.po=%)

all: update merge

update:
	@cd ../..; python2 editor/translations/extract.py

merge:
	@for po in $(POFILES); do \
	  echo -e "\nMerging $$po..."; \
	  msgmerge -w 79 -C $$po $$po $(TEMPLATE) > "$$po".new; \
	  mv -f "$$po".new $$po; \
	done

check:
	@for po in $(POFILES); do msgfmt -c $$po -o /dev/null; done
