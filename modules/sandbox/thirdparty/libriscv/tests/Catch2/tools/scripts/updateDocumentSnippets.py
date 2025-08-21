#!/usr/bin/env python3

from scriptCommon import catchPath
import os
import subprocess

# ---------------------------------------------------
# Update code examples
# ---------------------------------------------------
# For info on mdsnippets, see https://github.com/SimonCropp/MarkdownSnippets

# install dotnet SDK from http://go.microsoft.com/fwlink/?LinkID=798306&clcid=0x409
# Then install MarkdownSnippets.Tool with
#   dotnet tool install -g MarkdownSnippets.Tool
# To update:
#   dotnet tool update  -g MarkdownSnippets.Tool
# To uninstall (e.g. to downgrade to a lower version)
# dotnet tool uninstall -g MarkdownSnippets.Tool

os.chdir(catchPath)

subprocess.run('dotnet tool update  -g MarkdownSnippets.Tool --version 21.2.0', shell=True, check=True)
subprocess.run('mdsnippets', shell=True, check=True)
