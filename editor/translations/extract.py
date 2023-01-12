#!/bin/python

import enum
import fnmatch
import os
import os.path
import re
import shutil
import subprocess
import sys
from typing import Dict, Tuple


class Message:
    __slots__ = ("msgid", "msgid_plural", "msgctxt", "comments", "locations")

    def format(self):
        lines = []

        if self.comments:
            for i, content in enumerate(self.comments):
                prefix = "#. TRANSLATORS:" if i == 0 else "#."
                lines.append(prefix + content)

        lines.append("#: " + " ".join(self.locations))

        if self.msgctxt:
            lines.append('msgctxt "{}"'.format(self.msgctxt))

        if self.msgid_plural:
            lines += [
                'msgid "{}"'.format(self.msgid),
                'msgid_plural "{}"'.format(self.msgid_plural),
                'msgstr[0] ""',
                'msgstr[1] ""',
            ]
        else:
            lines += [
                'msgid "{}"'.format(self.msgid),
                'msgstr ""',
            ]

        return "\n".join(lines)


messages_map: Dict[Tuple[str, str], Message] = {}  # (id, context) -> Message.

line_nb = False

for arg in sys.argv[1:]:
    if arg == "--with-line-nb":
        print("Enabling line numbers in the context locations.")
        line_nb = True
    else:
        sys.exit("Non supported argument '" + arg + "'. Aborting.")


if not os.path.exists("editor"):
    sys.exit("ERROR: This script should be started from the root of the git repo.")


matches = []
for root, dirnames, filenames in os.walk("."):
    dirnames[:] = [d for d in dirnames if d not in ["thirdparty"]]
    for filename in fnmatch.filter(filenames, "*.cpp"):
        matches.append(os.path.join(root, filename))
    for filename in fnmatch.filter(filenames, "*.h"):
        matches.append(os.path.join(root, filename))
matches.sort()


remaps = {}
remap_re = re.compile(r'^\t*capitalize_string_remaps\["(?P<from>.+)"\] = (String::utf8\()?"(?P<to>.+)"')
stop_words = set()
stop_words_re = re.compile(r'^\t*"(?P<word>.+)",')
is_inside_stop_words = False
with open("editor/editor_property_name_processor.cpp") as f:
    for line in f:
        if is_inside_stop_words:
            m = stop_words_re.search(line)
            if m:
                stop_words.add(m.group("word"))
            else:
                is_inside_stop_words = False
        else:
            m = remap_re.search(line)
            if m:
                remaps[m.group("from")] = m.group("to")

        if not is_inside_stop_words and not stop_words:
            is_inside_stop_words = "stop_words = " in line


main_po = """
# LANGUAGE translation of the Godot Engine editor.
# Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md).
# Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.
# This file is distributed under the same license as the Godot source code.
#
# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.
#
#, fuzzy
msgid ""
msgstr ""
"Project-Id-Version: Godot Engine editor\\n"
"Report-Msgid-Bugs-To: https://github.com/godotengine/godot\\n"
"MIME-Version: 1.0\\n"
"Content-Type: text/plain; charset=UTF-8\\n"
"Content-Transfer-Encoding: 8-bit\\n"\n
"""


class ExtractType(enum.IntEnum):
    TEXT = 1
    PROPERTY_PATH = 2
    GROUP = 3
    SUBGROUP = 4


# Regex "(?P<name>([^"\\]|\\.)*)" creates a group named `name` that matches a string.
message_patterns = {
    re.compile(r'RTR\("(?P<message>([^"\\]|\\.)*)"(, "(?P<context>([^"\\]|\\.)*)")?\)'): ExtractType.TEXT,
    re.compile(r'TTR\("(?P<message>([^"\\]|\\.)*)"(, "(?P<context>([^"\\]|\\.)*)")?\)'): ExtractType.TEXT,
    re.compile(r'TTRC\("(?P<message>([^"\\]|\\.)*)"\)'): ExtractType.TEXT,
    re.compile(
        r'TTRN\("(?P<message>([^"\\]|\\.)*)", "(?P<plural_message>([^"\\]|\\.)*)",[^,)]+?(, "(?P<context>([^"\\]|\\.)*)")?\)'
    ): ExtractType.TEXT,
    re.compile(
        r'RTRN\("(?P<message>([^"\\]|\\.)*)", "(?P<plural_message>([^"\\]|\\.)*)",[^,)]+?(, "(?P<context>([^"\\]|\\.)*)")?\)'
    ): ExtractType.TEXT,
    re.compile(r'_initial_set\("(?P<message>[^"]+?)",'): ExtractType.PROPERTY_PATH,
    re.compile(r'GLOBAL_DEF(_RST)?(_NOVAL)?(_BASIC)?\("(?P<message>[^"]+?)",'): ExtractType.PROPERTY_PATH,
    re.compile(r'EDITOR_DEF(_RST)?\("(?P<message>[^"]+?)",'): ExtractType.PROPERTY_PATH,
    re.compile(
        r'EDITOR_SETTING(_USAGE)?\(Variant::[_A-Z0-9]+, [_A-Z0-9]+, "(?P<message>[^"]+?)",'
    ): ExtractType.PROPERTY_PATH,
    re.compile(
        r"(ADD_PROPERTYI?|ImportOption|ExportOption)\(PropertyInfo\("
        + r"Variant::[_A-Z0-9]+"  # Name
        + r', "(?P<message>[^"]+)"'  # Type
        + r'(, [_A-Z0-9]+(, "([^"\\]|\\.)*"(, (?P<usage>[_A-Z0-9]+))?)?|\))'  # [, hint[, hint string[, usage]]].
    ): ExtractType.PROPERTY_PATH,
    re.compile(r'ADD_ARRAY\("(?P<message>[^"]+)", '): ExtractType.PROPERTY_PATH,
    re.compile(r'ADD_ARRAY_COUNT(_WITH_USAGE_FLAGS)?\("(?P<message>[^"]+)", '): ExtractType.TEXT,
    re.compile(r'(ADD_GROUP|GNAME)\("(?P<message>[^"]+)", "(?P<prefix>[^"]*)"\)'): ExtractType.GROUP,
    re.compile(r'ADD_GROUP_INDENT\("(?P<message>[^"]+)", "(?P<prefix>[^"]*)", '): ExtractType.GROUP,
    re.compile(r'ADD_SUBGROUP\("(?P<message>[^"]+)", "(?P<prefix>[^"]*)"\)'): ExtractType.SUBGROUP,
    re.compile(r'ADD_SUBGROUP_INDENT\("(?P<message>[^"]+)", "(?P<prefix>[^"]*)", '): ExtractType.GROUP,
    re.compile(r'PNAME\("(?P<message>[^"]+)"\)'): ExtractType.PROPERTY_PATH,
}
theme_property_patterns = {
    re.compile(r'set_(constant|font|font_size|stylebox|color|icon)\("(?P<message>[^"]+)", '): ExtractType.PROPERTY_PATH,
}


# See String::_camelcase_to_underscore().
capitalize_re = re.compile(r"(?<=\D)(?=\d)|(?<=\d)(?=\D([a-z]|\d))")


def _process_editor_string(name):
    # See EditorPropertyNameProcessor::process_string().
    capitalized_parts = []
    parts = list(filter(bool, name.split("_")))  # Non-empty only.
    for i, segment in enumerate(parts):
        if i > 0 and i + 1 < len(parts) and segment in stop_words:
            capitalized_parts.append(segment)
            continue

        remapped = remaps.get(segment)
        if remapped:
            capitalized_parts.append(remapped)
        else:
            # See String::capitalize().
            # fmt: off
            capitalized_parts.append(" ".join(
                part.title()
                for part in capitalize_re.sub("_", segment).replace("_", " ").split()
            ))
            # fmt: on

    return " ".join(capitalized_parts)


def _is_block_translator_comment(translator_line):
    line = translator_line.strip()
    if line.find("//") == 0:
        return False
    else:
        return True


def _extract_translator_comment(line, is_block_translator_comment):
    line = line.strip()
    reached_end = False
    extracted_comment = ""

    start = line.find("TRANSLATORS:")
    if start == -1:
        start = 0
    else:
        start += len("TRANSLATORS:")

    if is_block_translator_comment:
        # If '*/' is found, then it's the end.
        if line.rfind("*/") != -1:
            extracted_comment = line[start : line.rfind("*/")]
            reached_end = True
        else:
            extracted_comment = line[start:]
    else:
        # If beginning is not '//', then it's the end.
        if line.find("//") != 0:
            reached_end = True
        else:
            start = 2 if start == 0 else start
            extracted_comment = line[start:]

    return (not reached_end, extracted_comment)


def process_file(f, fname):
    l = f.readline()
    lc = 1
    reading_translator_comment = False
    is_block_translator_comment = False
    translator_comment = ""
    current_group = ""
    current_subgroup = ""

    patterns = message_patterns
    if os.path.basename(fname) == "default_theme.cpp":
        patterns = {**message_patterns, **theme_property_patterns}

    while l:

        # Detect translator comments.
        if not reading_translator_comment and l.find("TRANSLATORS:") != -1:
            reading_translator_comment = True
            is_block_translator_comment = _is_block_translator_comment(l)
            translator_comment = ""

        # Gather translator comments. It will be gathered for the next translation function.
        if reading_translator_comment:
            reading_translator_comment, extracted_comment = _extract_translator_comment(l, is_block_translator_comment)
            if extracted_comment != "":
                translator_comment += extracted_comment + "\n"
            if not reading_translator_comment:
                translator_comment = translator_comment[:-1]  # Remove extra \n at the end.

        if not reading_translator_comment:
            for pattern, extract_type in patterns.items():
                for m in pattern.finditer(l):
                    location = os.path.relpath(fname).replace("\\", "/")
                    if line_nb:
                        location += ":" + str(lc)

                    captures = m.groupdict("")
                    msg = captures.get("message", "")
                    msg_plural = captures.get("plural_message", "")
                    msgctx = captures.get("context", "")

                    if extract_type == ExtractType.TEXT:
                        _add_message(msg, msg_plural, msgctx, location, translator_comment)
                    elif extract_type == ExtractType.PROPERTY_PATH:
                        if captures.get("usage") == "PROPERTY_USAGE_NO_EDITOR":
                            continue

                        if current_subgroup:
                            if msg.startswith(current_subgroup):
                                msg = msg[len(current_subgroup) :]
                            elif current_subgroup.startswith(msg):
                                pass  # Keep this as-is. See EditorInspector::update_tree().
                            else:
                                current_subgroup = ""
                        elif current_group:
                            if msg.startswith(current_group):
                                msg = msg[len(current_group) :]
                            elif current_group.startswith(msg):
                                pass  # Keep this as-is. See EditorInspector::update_tree().
                            else:
                                current_group = ""
                                current_subgroup = ""

                        if "." in msg:  # Strip feature tag.
                            msg = msg.split(".", 1)[0]
                        for part in msg.split("/"):
                            _add_message(_process_editor_string(part), msg_plural, msgctx, location, translator_comment)
                    elif extract_type == ExtractType.GROUP:
                        _add_message(msg, msg_plural, msgctx, location, translator_comment)
                        current_group = captures["prefix"]
                        current_subgroup = ""
                    elif extract_type == ExtractType.SUBGROUP:
                        _add_message(msg, msg_plural, msgctx, location, translator_comment)
                        current_subgroup = captures["prefix"]
            translator_comment = ""

        l = f.readline()
        lc += 1


def _add_message(msg, msg_plural, msgctx, location, translator_comment):
    key = (msg, msgctx)
    message = messages_map.get(key)
    if not message:
        message = Message()
        message.msgid = msg
        message.msgid_plural = msg_plural
        message.msgctxt = msgctx
        message.locations = []
        message.comments = []
        messages_map[key] = message
    if location not in message.locations:
        message.locations.append(location)
    if translator_comment and translator_comment not in message.comments:
        message.comments.append(translator_comment)


print("Updating the editor.pot template...")

for fname in matches:
    with open(fname, "r", encoding="utf8") as f:
        process_file(f, fname)

main_po += "\n\n".join(message.format() for message in messages_map.values())

with open("editor.pot", "w") as f:
    f.write(main_po)

if os.name == "posix":
    print("Wrapping template at 79 characters for compatibility with Weblate.")
    os.system("msgmerge -w79 editor.pot editor.pot > editor.pot.wrap")
    shutil.move("editor.pot.wrap", "editor.pot")

shutil.move("editor.pot", "editor/translations/editor.pot")

# TODO: Make that in a portable way, if we care; if not, kudos to Unix users
if os.name == "posix":
    added = subprocess.check_output(r"git diff editor/translations/editor.pot | grep \+msgid | wc -l", shell=True)
    removed = subprocess.check_output(r"git diff editor/translations/editor.pot | grep \\\-msgid | wc -l", shell=True)
    print("\n# Template changes compared to the staged status:")
    print("#   Additions: %s msgids.\n#   Deletions: %s msgids." % (int(added), int(removed)))
