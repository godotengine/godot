#!/bin/python

import fnmatch
import os
import re

matches = []
for root, dirnames, filenames in os.walk('.'):
	for filename in fnmatch.filter(filenames, '*.cpp'):
		if (filename.find("collada")!=-1):
			continue
		matches.append(os.path.join(root, filename))
	for filename in fnmatch.filter(filenames, '*.h'):
		if (filename.find("collada")!=-1):
			continue
		matches.append(os.path.join(root, filename))


unique_str=[]
main_po=""

for fname in matches:

	f = open(fname,"rb")

	new_f = ""

	l = f.readline()
	lc=1
	while(l):

		pos = 0
		while(pos>=0):
			pos = l.find('TTR(\"',pos)
			if (pos==-1):
				break
			pos+=5

			msg=""
			while (pos < len(l) and (l[pos]!='"' or l[pos-1]=='\\') ):
				msg+=l[pos]
				pos+=1

			if (not msg in unique_str):
				main_po+="\n#:"+fname+":"+str(lc)+"\n"
				main_po+='msgid "'+msg+'"\n'
				main_po+='msgstr ""\n'
				unique_str.append(msg)

		l = f.readline()
		lc+=1

	f.close()


f = open("tools.pot","wb")
f.write(main_po)
f.close()
