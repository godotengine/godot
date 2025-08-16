#!/usr/bin/env python3

import os
import sys

files_set = set()

for root, dir, files in os.walk("src/catch2"):
    for file in files:
        if file not in files_set:
            files_set.add(file)
        else:
            print("File %s is duplicate" % file)
            sys.exit(1)
