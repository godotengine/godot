#### TO MENTORS:
#### I'm not sure what is the right way to write a Python script in Godot? I saw other Python files all have def func_name(target, source, env).
#### Also I'm not sure of the location.
#### Right now I call this script using the "python" command.

# This script is used to generate core/translation_plural_rules.h
# The plural data used in this Python script is from POEdit Github repository - https://github.com/vslavik/poedit/raw/master/src/language_impl_plurals.h
# POEdit in turn parsed and extracted the plural data from Unicode Consortium (see http://www.unicode.org/cldr/charts/supplemental/language_plural_rules.html).

#!/bin/python

import sys
import urllib.request

target_file = "translation_plural_rules.h"

plural_data_url = "https://github.com/vslavik/poedit/raw/master/src/language_impl_plurals.h"
generate_text = "// The rest of the file is generated via plural_rules_builder.py (DO NOT MODIFY THIS LINE).\n"

# Get plural data from POEdit github repository. See above for the link.
text = ""
with urllib.request.urlopen(plural_data_url) as response:
    text = response.read().decode("utf-8")

# Build dictionary mapping plural rule to a list of locales from plural data. Also store the number of plural forms each local has.
rules = text.split("\n")
rule_to_locales = {}
locale_nplural = {}
for rule in rules:
    if rule.find("plural=") == -1:
        continue

    first_quote = rule.find('"')
    second_quote = rule.find('"', first_quote + 1)
    locale = rule[first_quote + 1 : second_quote]

    plural_start = rule.find("plural=") + len("plural=")
    plural_end = rule.find(";", plural_start)
    plural_rule = rule[plural_start:plural_end]

    if plural_rule in rule_to_locales:
        rule_to_locales[plural_rule].append(locale)
    else:
        rule_to_locales[plural_rule] = [locale]

    nplural_start = rule.find("nplurals=") + len("nplurals=")
    locale_nplural[locale] = rule[nplural_start : nplural_start + 1]

# Find the position in translation_plural_rules.h for code generation.
with open(target_file, "r") as f:
    text = f.read()
gen_start_pos = text.find(generate_text) + len(generate_text)
pretext = text[:gen_start_pos]

# Generate mapping in build_mapping().
gen_content = "\n"
rule_number = 1
for rule in rule_to_locales:
    for locale in rule_to_locales[rule]:
        gen_content += (
            '\t\ttemp.set("' + locale + '", { ' + locale_nplural[locale] + ", &rule" + str(rule_number) + " });\n"
        )
    rule_number += 1
gen_content += "\t\treturn temp;\n\t}\n"

# Generate rule functions.
rule_number = 1
for rule in rule_to_locales:
    gen_content += "\n\tstatic int rule" + str(rule_number) + "(const int p_n) {\n\t\t// "
    gen_content += ", ".join(rule_to_locales[rule]) + ".\n\t\t"
    gen_content += "return " + rule.replace("n", "p_n") + ";\n\t}\n"
    rule_number += 1

gen_content += "};\n\n#endif // TRANSLATION_PLURAL_RULES_H\n"

# Write to file.
with open(target_file, "w") as f:
    f.write(pretext + gen_content)
