# Makefile providing various facilities to manage translations

TEMPLATE = classes.pot
POFILES = $(wildcard *.po)
LANGS = $(POFILES:%.po=%)

all: update merge

update:
	@cd ../..; \
	python3 doc/translations/extract.py \
	  --path doc/classes modules/*/doc_classes \
	  --output doc/translations/$(TEMPLATE)

merge:
	@for po in $(POFILES); do \
	  echo -e "\nMerging $$po..."; \
	  msgmerge -w 79 -C $$po $$po $(TEMPLATE) > "$$po".new; \
	  mv -f "$$po".new $$po; \
	done

check:
	@for po in $(POFILES); do msgfmt -c $$po -o /dev/null; done
