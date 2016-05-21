#!/bin/python

import fnmatch
import os
import re
import shutil
import subprocess

if (not os.path.exists("tools")):
	os.sys.exit("ERROR: This script should be started from the root of the git repo.")

matches = []
for root, dirnames, filenames in os.walk('.'):
	for filename in fnmatch.filter(filenames, '*.cpp'):
		if (filename.find("collada") != -1):
			continue
		matches.append(os.path.join(root, filename))
	for filename in fnmatch.filter(filenames, '*.h'):
		if (filename.find("collada") != -1):
			continue
		matches.append(os.path.join(root, filename))


unique_str = []
unique_loc = {}
main_po = ""

print("Updating the tools.pot template...")

for fname in matches:

	f = open(fname, "rb")

	l = f.readline()
	lc = 1
	while (l):

		pos = 0
		while (pos >= 0):
			pos = l.find('TTR(\"', pos)
			if (pos == -1):
				break
			pos += 5

			msg = ""
			while (pos < len(l) and (l[pos] != '"' or l[pos - 1] == '\\')):
				msg += l[pos]
				pos += 1

			location = os.path.relpath(fname).replace('\\','/') + ":" + str(lc)

			if (not msg in unique_str):
				main_po += "\n#: " + location + "\n"
				main_po += 'msgid "' + msg + '"\n'
				main_po += 'msgstr ""\n'
				unique_str.append(msg)
				unique_loc[msg] = [location]
			elif (not location in unique_loc[msg]):
				# Add additional location to previous occurence too
				msg_pos = main_po.find('\nmsgid "' + msg)
				main_po = main_po[:msg_pos] + ' ' + location + main_po[msg_pos:]
				unique_loc[msg].append(location)

		l = f.readline()
		lc += 1

	f.close()


f = open("tools.pot", "wb")
f.write(main_po)
f.close()

shutil.move("tools.pot", "tools/translations/tools.pot")

# TODO: Make that in a portable way, if we care; if not, kudos to Unix users
if (os.name == "posix"):
	added = subprocess.check_output("git diff tools/translations/tools.pot | grep \+msgid | wc -l", shell = True)
	removed = subprocess.check_output("git diff tools/translations/tools.pot | grep \\\-msgid | wc -l", shell = True)
	print("Template changes compared to the staged status:")
	print("  Additions: %s msgids.\n  Deletions: %s msgids." % (int(added), int(removed)))
