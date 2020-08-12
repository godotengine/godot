#!/bin/python

import fnmatch
import os
import shutil
import subprocess
import sys


line_nb = False

for arg in sys.argv[1:]:
    if arg == "--with-line-nb":
        print("Enabling line numbers in the context locations.")
        line_nb = True
    else:
        os.sys.exit("Non supported argument '" + arg + "'. Aborting.")


if not os.path.exists("editor"):
    os.sys.exit("ERROR: This script should be started from the root of the git repo.")


matches = []
for root, dirnames, filenames in os.walk("."):
    dirnames[:] = [d for d in dirnames if d not in ["thirdparty"]]
    for filename in fnmatch.filter(filenames, "*.cpp"):
        matches.append(os.path.join(root, filename))
    for filename in fnmatch.filter(filenames, "*.h"):
        matches.append(os.path.join(root, filename))
matches.sort()


unique_str = []
unique_loc = {}
ctx_group = {}  # Store msgctx, msg, and locations.
main_po = """
# LANGUAGE translation of the Godot Engine editor.
# Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.
# Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).
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
"Content-Transfer-Encoding: 8-bit\\n"
"""


def _write_message(msgctx, msg, msg_plural, location):
    global main_po
    main_po += "\n#: " + location + "\n"
    if msgctx != "":
        main_po += 'msgctxt "' + msgctx + '"\n'
    main_po += 'msgid "' + msg + '"\n'
    if msg_plural != "":
        main_po += 'msgid_plural "' + msg_plural + '"\n'
        main_po += 'msgstr[0] ""\n'
        main_po += 'msgstr[1] ""\n'
    else:
        main_po += 'msgstr ""\n'


def _add_additional_location(msgctx, msg, location):
    global main_po
    # Add additional location to previous occurrence
    msg_pos = -1
    if msgctx != "":
        msg_pos = main_po.find('\nmsgctxt "' + msgctx + '"\nmsgid "' + msg + '"')
    else:
        msg_pos = main_po.find('\nmsgid "' + msg + '"')

    if msg_pos == -1:
        print("Someone apparently thought writing Python was as easy as GDScript. Ping Akien.")
    main_po = main_po[:msg_pos] + " " + location + main_po[msg_pos:]


def process_file(f, fname):

    global main_po, unique_str, unique_loc

    l = f.readline()
    lc = 1
    while l:

        patterns = ['RTR("', 'TTR("', 'TTRC("', 'TTRN("', 'RTRN("']
        idx = 0
        pos = 0
        while pos >= 0:
            # Loop until a pattern is found. If not, next line.
            pos = l.find(patterns[idx], pos)
            if pos == -1:
                if idx < len(patterns) - 1:
                    idx += 1
                    pos = 0
                continue
            pos += len(patterns[idx])

            # Read msg until "
            msg = ""
            while pos < len(l) and (l[pos] != '"' or l[pos - 1] == "\\"):
                msg += l[pos]
                pos += 1

            # Read plural.
            msg_plural = ""
            if patterns[idx] in ['TTRN("', 'RTRN("']:
                pos = l.find('"', pos + 1)
                pos += 1
                while pos < len(l) and (l[pos] != '"' or l[pos - 1] == "\\"):
                    msg_plural += l[pos]
                    pos += 1

            # Read context.
            msgctx = ""
            pos += 1
            read_ctx = False
            while pos < len(l):
                if l[pos] == ")":
                    break
                elif l[pos] == '"':
                    read_ctx = True
                    break
                pos += 1

            pos += 1
            if read_ctx:
                while pos < len(l) and (l[pos] != '"' or l[pos - 1] == "\\"):
                    msgctx += l[pos]
                    pos += 1

            # File location.
            location = os.path.relpath(fname).replace("\\", "/")
            if line_nb:
                location += ":" + str(lc)

            if msgctx != "":
                # If it's a new context or a new message within an existing context, then write new msgid.
                # Else add location to existing msgid.
                if not msgctx in ctx_group:
                    _write_message(msgctx, msg, msg_plural, location)
                    ctx_group[msgctx] = {msg: [location]}
                elif not msg in ctx_group[msgctx]:
                    _write_message(msgctx, msg, msg_plural, location)
                    ctx_group[msgctx][msg] = [location]
                elif not location in ctx_group[msgctx][msg]:
                    _add_additional_location(msgctx, msg, location)
                    ctx_group[msgctx][msg].append(location)
            else:
                if not msg in unique_str:
                    _write_message(msgctx, msg, msg_plural, location)
                    unique_str.append(msg)
                    unique_loc[msg] = [location]
                elif not location in unique_loc[msg]:
                    _add_additional_location(msgctx, msg, location)
                    unique_loc[msg].append(location)
        l = f.readline()
        lc += 1


print("Updating the editor.pot template...")

for fname in matches:
    with open(fname, "r", encoding="utf8") as f:
        process_file(f, fname)

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
