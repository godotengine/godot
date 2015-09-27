#!/usr/bin/python3
# -*- coding: utf-8 -*-

#
# makedocs.py: Generate documentation for Open Project Wiki
#
# Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.
#
# Contributor: Jorge Araya Navarro <elcorreo@deshackra.com>
#

import re
import argparse
import logging
from os import path
from itertools import izip_longest
from xml.etree import ElementTree

logging.basicConfig(level=logging.INFO)


def getxmlfloc():
    """ Returns the supposed location of the XML file
    """
    filepath = path.dirname(path.abspath(__file__))
    return path.join(filepath, "class_list.xml")


def sortkey(c):
    """ Symbols are first, letters second
    """
    if "_" == c.attrib["name"][0]:
        return True
    else:
        return c.attrib["name"]


def toOP(text):
    """ Convert commands in text to Open Project commands
    """
    # We are going to do something very complicated with all HTML commands
    # sadly, some commands are embedded inside other commands, so some steps
    # are needed before converting some commands to Textile markup
    groups = re.finditer((r'\[html (?P<command>/?\w+/?)(\]| |=)?(\]| |=)?(?P<a'
                          'rg>\w+)?(\]| |=)?(?P<value>"[^"]+")?/?\]'), text)
    alignstr = ""
    for group in groups:
        gd = group.groupdict()
        if gd["command"] == "br/":
            text = text.replace(group.group(0), "\n\n", 1)
        elif gd["command"] == "div":
            if gd["value"] == '"center"':
                alignstr = "{display:block; margin-left:auto; margin-right:auto;}"
            elif gd["value"] == '"left"':
                alignstr = "<"
            elif gd["value"] == '"right"':
                alignstr = ">"
            text = text.replace(group.group(0), "\n\n", 1)
        elif gd["command"] == "/div":
            alignstr = ""
            text = text.replace(group.group(0), "\n\n", 1)
        elif gd["command"] == "img":
            text = text.replace(group.group(0), "!{align}{src}!".format(
                align=alignstr, src=gd["value"].strip('"')), 1)
        elif gd["command"] == "b" or gd["command"] == "/b":
            text = text.replace(group.group(0), "*", 1)
        elif gd["command"] == "i" or gd["command"] == "/i":
            text = text.replace(group.group(0), "_", 1)
        elif gd["command"] == "u" or gd["command"] == "/u":
            text = text.replace(group.group(0), "+", 1)
    # TODO: Process other non-html commands
    return text + "\n\n"

desc = "Generates documentation from a XML file to different markup languages"

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("--input", dest="xmlfp", default=getxmlfloc(),
                    help="Input XML file, default: {}".format(getxmlfloc()))
parser.add_argument("--output-dir", dest="outputdir", required=True,
                    help="Output directory for generated files")
# TODO: add an option for outputting different markup formats

args = parser.parse_args()
# Let's check if the file and output directory exists
if not path.isfile(args.xmlfp):
    logging.critical("File not found: {}".format(args.xmlfp))
    exit(1)
elif not path.isdir(args.outputdir):
    logging.critical("Path does not exist: {}".format(args.outputdir))
    exit(1)

# Let's begin
tree = ElementTree.parse(args.xmlfp)
root = tree.getroot()

# Check version attribute exists in <doc>
if "version" not in root.attrib:
    logging.critical("<doc>'s version attribute missing")
    exit(1)

version = root.attrib["version"]
classes = sorted(root, key=sortkey)
# first column is always longer, second column of classes should be shorter
zclasses = izip_longest(classes[:len(classes) / 2 + 1],
                        classes[len(classes) / 2 + 1:],
                        fillvalue="")

# We write the class_list file and also each class file at once
with open(path.join(args.outputdir, "class_list.txt"), "wb") as fcl:
    # Write header of table
    fcl.write("|^.\n")
    fcl.write("|_. Index symbol |_. Class name "
              "|_. Index symbol |_. Class name |\n")
    fcl.write("|-.\n")

    indexletterl = ""
    indexletterr = ""
    for gdclassl, gdclassr in zclasses:
        # write a row #
        # write the index symbol column, left
        if indexletterl != gdclassl.attrib["name"][0]:
            indexletterl = gdclassl.attrib["name"][0]
            fcl.write("| *{}* |".format(indexletterl.upper()))
        else:
            # empty cell
            fcl.write("| |")
        # write the class name column, left
        fcl.write("\"{name}({title})\":/class_{link}".format(
            name=gdclassl.attrib["name"],
            title="Go to page of class " + gdclassl.attrib["name"],
            link="class_" + gdclassl.attrib["name"].lower()))

        # write the index symbol column, right
        if isinstance(gdclassr, ElementTree.Element):
            if indexletterr != gdclassr.attrib["name"][0]:
                indexletterr = gdclassr.attrib["name"][0]
                fcl.write("| *{}* |".format(indexletterr.upper()))
            else:
                # empty cell
                fcl.write("| |")
        # We are dealing with an empty string
        else:
            # two empty cell
            fcl.write("| | |\n")
            # We won't get the name of the class since there is no ElementTree
            # object for the right side of the tuple, so we iterate the next
            # tuple instead
            continue

        # write the class name column (if any), right
        fcl.write("\"{name}({title})\":/{link} |\n".format(
            name=gdclassr.attrib["name"],
            title="Go to page of class " + gdclassr.attrib["name"],
            link=gdclassr.attrib["name"].lower()))

        # row written #
        # now, let's write each class page for each class
        for gdclass in [gdclassl, gdclassr]:
            if not isinstance(ElementTree.Element, gdclass):
                continue

            classname = gdclass.attrib["name"]
            with open(path.join(args.outputdir, "{}.txt".format(
                    classname.lower())), "wb") as clsf:
                # First level header with the name of the class
                clsf.write("h1. {}\n".format(classname))
                # lay the attributes
                if "inherits" in gdclass.attrib:
                    inh = gdclass.attrib["inherits"].strip()
                    clsf.write(
                        "*Inherits:* \"{name}({title})\":/class_{link}\n".
                        format(
                            name=classname,
                            title="Go to page of class " + classname,
                            link=classname.lower()))
                if "category" in gdclass.attrib:
                    clsf.write("*Category:* {}".
                               format(gdclass.attrib["category"].strip()))
                # lay child nodes
                for gdchild in gdclass.iter():
                    if gdchild.tag == "brief_description":
                        clsf.write("h2. Brief Description\n")
                        clsf.write(toOP(gdchild.text.strip()))
                        # convert commands in text
