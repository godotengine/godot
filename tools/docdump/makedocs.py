#!/usr/bin/python3
# -*- coding: utf-8 -*-

#
# makedocs.py: Generate documentation for Open Project Wiki
#
# Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.
#
# Contributor: Jorge Araya Navarro <elcorreo@deshackra.com>
#

# IMPORTANT NOTICE:
# If you are going to modify anything from this file, please be sure to follow
# the Style Guide for Python Code or often called "PEP8". To do this
# automagically just install autopep8:
#
#     $ sudo pip3 install autopep8
#
# and run:
#
#     $ autopep8 makedocs.py
#
# Before committing your changes. Also be sure to delete any trailing
# whitespace you may left.
#
# TODO:
#  * Refactor code. Refactor strings.
#  * Adapt this script for generating content in other markup formats like
#    DokuWiki, Markdown, etc.
#  * Because people will translate class_list.xml, we should implement
#    internalization.
#
# Also check other TODO entries in this script for more information on what is
# left to do.

import re
import argparse
import logging
from os import path
from itertools import zip_longest
from xml.etree import ElementTree

# add an option to change the verbosity
logging.basicConfig(level=logging.INFO)


def tb(string):
    """ Return a byte representation of a string
    """
    return bytes(string, "UTF-8")


def getxmlfloc():
    """ Returns the supposed location of the XML file
    """
    filepath = path.dirname(path.abspath(__file__))
    return path.join(filepath, "class_list.xml")


def sortkey(c):
    """ Symbols are first, letters second
    """
    if "_" == c.attrib["name"][0]:
        return "A"
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
                alignstr = ("{display:block; margin-left:auto;"
                            " margin-right:auto;}")
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
    # Process other non-html commands
    groups = re.finditer((r'\[method ((?P<class>[aA0-zZ9_]+)(?:\.))'
                          r'?(?P<method>[aA0-zZ9_]+)\]'), text)
    for group in groups:
        gd = group.groupdict()
        if gd["class"]:
            replacewith = ("\"<code>{gclass}.{method}</code>(Go "
                           "to page {gclass}, section {method})\""
                           ":/class_{lkclass}#{lkmethod}".
                           format(gclass=gd["class"],
                                  method=gd["method"],
                                  lkclass=gd["class"].lower(),
                                  lkmethod=gd["method"].lower()))
        else:
            # The method is located in the same wiki page
            replacewith = ("\"<code>{method}</code>(Jump to method"
                           " {method})\":#{lkmethod}".
                           format(method=gd["method"],
                                  lkmethod=gd["method"].lower()))

        text = text.replace(group.group(0), replacewith, 1)
    # Finally, [Classes] are around brackets, make them direct links
    groups = re.finditer(r'\[(?P<class>[az0-AZ0_]+)\]', text)
    for group in groups:
        gd = group.groupdict()
        replacewith = ("\"<code>{gclass}</code>(Go to page of class"
                       " {gclass})\":/class_{lkclass}".
                       format(gclass=gd["class"],
                              lkclass=gd["class"].lower()))
        text = text.replace(group.group(0), replacewith, 1)

    return text + "\n\n"


def mkfn(node, is_signal=False):
    """ Return a string containing a unsorted item for a function
    """
    finalstr = ""
    name = node.attrib["name"]
    rtype = node.find("return")
    if rtype:
        rtype = rtype.attrib["type"]
    else:
        rtype = "void"
    # write the return type and the function name first
    finalstr += "* "
    # return type
    if not is_signal:
        if rtype != "void":
            finalstr += " \"{rtype}({title})\":/class_{link} ".format(
                rtype=rtype,
                title="Go to page of class " + rtype,
                link=rtype.lower())
        else:
            finalstr += " void "

    # function name
    if not is_signal:
        finalstr += "\"*{funcname}*({title})\":#{link} <b>(</b> ".format(
            funcname=name,
            title="Jump to description for node " + name,
            link=name.lower())
    else:
        # Signals have no description
        finalstr += "*{funcname}* <b>(</b>".format(funcname=name)
    # loop for the arguments of the function, if any
    args = []
    for arg in sorted(
            node.iter(tag="argument"),
            key=lambda a: int(a.attrib["index"])):

        twd = "{type} {name}={default}"
        tnd = "{type} {name}"
        tlk = ("\"*{cls}*(Go to page of class {cls})"
               "\":/class_{lcls}")
        ntype = arg.attrib["type"]
        nname = arg.attrib["name"]

        if "default" in arg.attrib:
            args.insert(
                -1,
                twd.format(
                    type=tlk.format(
                        cls=ntype,
                        lcls=ntype.lower()),
                    name=nname,
                    default=arg.attrib["default"]))
        else:
            # No default value present
            args.insert(
                -1,
                tnd.format(
                    type=tlk.format(
                        cls=ntype,
                        lcls=ntype.lower()),
                    name=nname))
    # join the arguments together
    finalstr += ", ".join(args)
    # and, close the function with a )
    finalstr += " <b>)</b>"
    # write the qualifier, if any
    if "qualifiers" in node.attrib:
        qualifier = node.attrib["qualifiers"]
        finalstr += " " + qualifier

    finalstr += "\n"

    return finalstr

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
logging.debug("Number of classes: {}".format(len(classes)))
logging.debug("len(classes) / 2 + 1: {}".format(int(len(classes) / 2 + 1)))
# first column is always longer, second column of classes should be shorter
zclasses = zip_longest(classes[:int(len(classes) / 2 + 1)],
                       classes[int(len(classes) / 2 + 1):],
                       fillvalue="")

# We write the class_list file and also each class file at once
with open(path.join(args.outputdir, "class_list.txt"), "wb") as fcl:
    # Write header of table
    fcl.write(tb("|^.\n"))
    fcl.write(tb("|_. Index symbol |_. Class name "
                 "|_. Index symbol |_. Class name |\n"))
    fcl.write(tb("|-.\n"))

    indexletterl = ""
    indexletterr = ""
    for gdclassl, gdclassr in zclasses:
        # write a row #
        # write the index symbol column, left
        if indexletterl != gdclassl.attrib["name"][0]:
            indexletterl = gdclassl.attrib["name"][0]
            fcl.write(tb("| *{}* |".format(indexletterl.upper())))
        else:
            # empty cell
            fcl.write(tb("| |"))
        # write the class name column, left
        fcl.write(tb("\"{name}({title})\":/class_{link}".format(
            name=gdclassl.attrib["name"],
            title="Go to page of class " + gdclassl.attrib["name"],
            link="class_" + gdclassl.attrib["name"].lower())))

        # write the index symbol column, right
        if isinstance(gdclassr, ElementTree.Element):
            if indexletterr != gdclassr.attrib["name"][0]:
                indexletterr = gdclassr.attrib["name"][0]
                fcl.write(tb("| *{}* |".format(indexletterr.upper())))
            else:
                # empty cell
                fcl.write(tb("| |"))
        # We are dealing with an empty string
        else:
            # two empty cell
            fcl.write(tb("| | |\n"))
            # We won't get the name of the class since there is no ElementTree
            # object for the right side of the tuple, so we iterate the next
            # tuple instead
            continue

        # write the class name column (if any), right
        fcl.write(tb("\"{name}({title})\":/{link} |\n".format(
            name=gdclassr.attrib["name"],
            title="Go to page of class " + gdclassr.attrib["name"],
            link=gdclassr.attrib["name"].lower())))

        # row written #
        # now, let's write each class page for each class
        for gdclass in [gdclassl, gdclassr]:
            if not isinstance(gdclass, ElementTree.Element):
                continue

            classname = gdclass.attrib["name"]
            with open(path.join(args.outputdir, "{}.txt".format(
                    classname.lower())), "wb") as clsf:
                # First level header with the name of the class
                clsf.write(tb("h1. {}\n\n".format(classname)))
                # lay the attributes
                if "inherits" in gdclass.attrib:
                    inh = gdclass.attrib["inherits"].strip()
                    clsf.write(tb(("h4. Inherits: \"<code>{name}"
                                   "</code>({title})\":/class_{link}\n\n").
                                  format(
                        name=inh,
                        title="Go to page of class " + inh,
                        link=classname.lower())))
                if "category" in gdclass.attrib:
                    clsf.write(tb("h4. Category: {}\n\n".
                                  format(gdclass.attrib["category"].strip())))
                # lay child nodes
                briefd = gdclass.find("brief_description")
                logging.debug(
                    "Brief description was found?: {}".format(briefd))
                if briefd.text.strip():
                    logging.debug(
                        "Text: {}".format(briefd.text.strip()))
                    clsf.write(tb("h2. Brief Description\n\n"))
                    clsf.write(tb(toOP(briefd.text.strip()) +
                                  "\"read more\":#more\n\n"))

                # Write the list of member functions of this class
                methods = gdclass.find("methods")
                if methods and len(methods) > 0:
                    clsf.write(tb("\nh3. Member Functions\n\n"))
                    for method in methods.iter(tag='method'):
                        clsf.write(tb(mkfn(method)))

                signals = gdclass.find("signals")
                if signals and len(signals) > 0:
                    clsf.write(tb("\nh3. Signals\n\n"))
                    for signal in signals.iter(tag='signal'):
                        clsf.write(tb(mkfn(signal, True)))
                # TODO: <members> tag is necessary to process? it does not
                # exists in class_list.xml file.

                consts = gdclass.find("constants")
                if consts and len(consts) > 0:
                    clsf.write(tb("\nh3. Numeric Constants\n\n"))
                    for const in sorted(consts, key=lambda k:
                                        k.attrib["name"]):
                        if const.text.strip():
                            clsf.write(tb("* *{name}* = *{value}* - {desc}\n".
                                          format(
                                              name=const.attrib["name"],
                                              value=const.attrib["value"],
                                              desc=const.text.strip())))
                        else:
                            # Constant have no description
                            clsf.write(tb("* *{name}* = *{value}*\n".
                                          format(
                                              name=const.attrib["name"],
                                              value=const.attrib["value"])))
                descrip = gdclass.find("description")
                clsf.write(tb("\nh3(#more). Description\n\n"))
                if descrip.text:
                    clsf.write(tb(descrip.text.strip() + "\n"))
                else:
                    clsf.write(tb("_Nothing here, yet..._\n"))

                # and finally, the description for each method
                if methods and len(methods) > 0:
                    clsf.write(tb("\nh3. Member Function Description\n\n"))
                    for method in methods.iter(tag='method'):
                        clsf.write(tb("h4(#{n}). {name}\n\n".format(
                            n=method.attrib["name"].lower(),
                            name=method.attrib["name"])))
                        clsf.write(tb(mkfn(method) + "\n"))
                        clsf.write(tb(toOP(method.find(
                            "description").text.strip())))
