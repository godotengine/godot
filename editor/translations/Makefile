# Makefile providing various facilities to manage translations

TEMPLATE = editor.pot
POFILES = $(wildcard *.po)
LANGS = $(POFILES:%.po=%)

all: update merge

update:
	@cd ../..; python3 editor/translations/extract.py

merge:
	@for po in $(POFILES); do \
	  echo -e "\nMerging $$po..."; \
	  msgmerge -w 79 -C $$po $$po $(TEMPLATE) > "$$po".new; \
	  mv -f "$$po".new $$po; \
	done

check:
	@for po in $(POFILES); do msgfmt -c $$po -o /dev/null; done

# Generate completion ratio from statistics string such as:
# 2775 translated messages, 272 fuzzy translations, 151 untranslated messages.
# First number can be 0, second and third numbers are only present if non-zero.
include-list:
	@list=""; \
	threshold=0.30; \
	for po in $(POFILES); do \
	  res=`msgfmt --statistics $$po -o /dev/null 2>&1 | sed 's/[^0-9,]*//g'`; \
	  complete=`cut -d',' -f1 <<< $$res`; \
	  fuzzy_or_untranslated=`cut -d',' -f2 <<< $$res`; \
	  untranslated_maybe=`cut -d',' -f3 <<< $$res`; \
	  if [ -z "$$fuzzy_or_untranslated" ]; then fuzzy_or_untranslated=0; fi; \
	  if [ -z "$$untranslated_maybe" ]; then untranslated_maybe=0; fi; \
	  incomplete=`expr $$fuzzy_or_untranslated + $$untranslated_maybe`; \
	  if `awk "BEGIN {exit !($$complete / ($$complete + $$incomplete) > $$threshold)}"`; then \
	    lang=`basename $$po .po`; \
	    list+="$$lang,"; \
	  fi; \
	done; \
	echo $$list;
