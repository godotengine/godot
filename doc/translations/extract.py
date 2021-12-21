#!/usr/bin/env python3

import argparse
import os
import shutil
from collections import OrderedDict

EXTRACT_TAGS = ["description", "brief_description", "member", "constant", "theme_item", "link"]
HEADER = """\
# LANGUAGE translation of the Godot Engine class reference.
# Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.
# Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).
# This file is distributed under the same license as the Godot source code.
#
# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.
#
#, fuzzy
msgid ""
msgstr ""
"Project-Id-Version: Godot Engine class reference\\n"
"Report-Msgid-Bugs-To: https://github.com/godotengine/godot\\n"
"MIME-Version: 1.0\\n"
"Content-Type: text/plain; charset=UTF-8\\n"
"Content-Transfer-Encoding: 8-bit\\n"

"""
# Some strings used by make_rst.py are normally part of the editor translations,
# so we need to include them manually here for the online docs.
HEADINGS = [
    "Description",
    "Tutorials",
    "Properties",
    "Constructors",
    "Methods",
    "Operators",
    "Theme Properties",
    "Signals",
    "Enumerations",
    "Constants",
    "Property Descriptions",
    "Constructor Descriptions",
    "Method Descriptions",
    "Operator Descriptions",
    "Theme Property Descriptions",
]

## <xml-line-number-hack from="https://stackoverflow.com/a/36430270/10846399">
import sys

sys.modules["_elementtree"] = None
import xml.etree.ElementTree as ET

## override the parser to get the line number
class LineNumberingParser(ET.XMLParser):
    def _start(self, *args, **kwargs):
        ## Here we assume the default XML parser which is expat
        ## and copy its element position attributes into output Elements
        element = super(self.__class__, self)._start(*args, **kwargs)
        element._start_line_number = self.parser.CurrentLineNumber
        element._start_column_number = self.parser.CurrentColumnNumber
        element._start_byte_index = self.parser.CurrentByteIndex
        return element

    def _end(self, *args, **kwargs):
        element = super(self.__class__, self)._end(*args, **kwargs)
        element._end_line_number = self.parser.CurrentLineNumber
        element._end_column_number = self.parser.CurrentColumnNumber
        element._end_byte_index = self.parser.CurrentByteIndex
        return element


## </xml-line-number-hack>


class Desc:
    def __init__(self, line_no, msg, desc_list=None):
        ## line_no   : the line number where the desc is
        ## msg       : the description string
        ## desc_list : the DescList it belongs to
        self.line_no = line_no
        self.msg = msg
        self.desc_list = desc_list


class DescList:
    def __init__(self, doc, path):
        ## doc  : root xml element of the document
        ## path : file path of the xml document
        ## list : list of Desc objects for this document
        self.doc = doc
        self.path = path
        self.list = []


def print_error(error):
    print("ERROR: {}".format(error))


## build classes with xml elements recursively
def _collect_classes_dir(path, classes):
    if not os.path.isdir(path):
        print_error("Invalid directory path: {}".format(path))
        exit(1)
    for _dir in map(lambda dir: os.path.join(path, dir), os.listdir(path)):
        if os.path.isdir(_dir):
            _collect_classes_dir(_dir, classes)
        elif os.path.isfile(_dir):
            if not _dir.endswith(".xml"):
                # print("Got non-.xml file '{}', skipping.".format(path))
                continue
            _collect_classes_file(_dir, classes)


## opens a file and parse xml add to classes
def _collect_classes_file(path, classes):
    if not os.path.isfile(path) or not path.endswith(".xml"):
        print_error("Invalid xml file path: {}".format(path))
        exit(1)
    print("Collecting file: {}".format(os.path.basename(path)))

    try:
        tree = ET.parse(path, parser=LineNumberingParser())
    except ET.ParseError as e:
        print_error("Parse error reading file '{}': {}".format(path, e))
        exit(1)

    doc = tree.getroot()

    if "name" in doc.attrib:
        if "version" not in doc.attrib:
            print_error("Version missing from 'doc', file: {}".format(path))

        name = doc.attrib["name"]
        if name in classes:
            print_error("Duplicate class {} at path {}".format(name, path))
            exit(1)
        classes[name] = DescList(doc, path)
    else:
        print_error("Unknown XML file {}, skipping".format(path))


## regions are list of tuples with size 3 (start_index, end_index, indent)
## indication in string where the codeblock starts, ends, and it's indent
## if i inside the region returns the indent, else returns -1
def _get_xml_indent(i, regions):
    for region in regions:
        if region[0] < i < region[1]:
            return region[2]
    return -1


## find and build all regions of codeblock which we need later
def _make_codeblock_regions(desc, path=""):
    code_block_end = False
    code_block_index = 0
    code_block_regions = []
    while not code_block_end:
        code_block_index = desc.find("[codeblock]", code_block_index)
        if code_block_index < 0:
            break
        xml_indent = 0
        while True:
            ## [codeblock] always have a trailing new line and some tabs
            ## those tabs are belongs to xml indentations not code indent
            if desc[code_block_index + len("[codeblock]\n") + xml_indent] == "\t":
                xml_indent += 1
            else:
                break
        end_index = desc.find("[/codeblock]", code_block_index)
        if end_index < 0:
            print_error("Non terminating codeblock: {}".format(path))
            exit(1)
        code_block_regions.append((code_block_index, end_index, xml_indent))
        code_block_index += 1
    return code_block_regions


def _strip_and_split_desc(desc, code_block_regions):
    desc_strip = ""  ## a stripped desc msg
    total_indent = 0  ## code indent = total indent - xml indent
    for i in range(len(desc)):
        c = desc[i]
        if c == "\n":
            c = "\\n"
        if c == '"':
            c = '\\"'
        if c == "\\":
            c = "\\\\"  ## <element \> is invalid for msgmerge
        if c == "\t":
            xml_indent = _get_xml_indent(i, code_block_regions)
            if xml_indent >= 0:
                total_indent += 1
                if xml_indent < total_indent:
                    c = "\\t"
                else:
                    continue
            else:
                continue
        desc_strip += c
        if c == "\\n":
            total_indent = 0
    return desc_strip


## make catalog strings from xml elements
def _make_translation_catalog(classes):
    unique_msgs = OrderedDict()
    for class_name in classes:
        desc_list = classes[class_name]
        for elem in desc_list.doc.iter():
            if elem.tag in EXTRACT_TAGS:
                if not elem.text or len(elem.text) == 0:
                    continue
                line_no = elem._start_line_number if elem.text[0] != "\n" else elem._start_line_number + 1
                desc_str = elem.text.strip()
                code_block_regions = _make_codeblock_regions(desc_str, desc_list.path)
                desc_msg = _strip_and_split_desc(desc_str, code_block_regions)
                desc_obj = Desc(line_no, desc_msg, desc_list)
                desc_list.list.append(desc_obj)

                if desc_msg not in unique_msgs:
                    unique_msgs[desc_msg] = [desc_obj]
                else:
                    unique_msgs[desc_msg].append(desc_obj)
    return unique_msgs


## generate the catalog file
def _generate_translation_catalog_file(unique_msgs, output):
    with open(output, "w", encoding="utf8") as f:
        f.write(HEADER)
        for msg in HEADINGS:
            f.write("#: doc/tools/make_rst.py\n")
            f.write('msgid "{}"\n'.format(msg))
            f.write('msgstr ""\n\n')
        for msg in unique_msgs:
            if len(msg) == 0 or msg in HEADINGS:
                continue

            f.write("#:")
            desc_list = unique_msgs[msg]
            for desc in desc_list:
                path = desc.desc_list.path.replace("\\", "/")
                if path.startswith("./"):
                    path = path[2:]
                f.write(" {}:{}".format(path, desc.line_no))
            f.write("\n")

            f.write('msgid "{}"\n'.format(msg))
            f.write('msgstr ""\n\n')

    ## TODO: what if 'nt'?
    if os.name == "posix":
        print("Wrapping template at 79 characters for compatibility with Weblate.")
        os.system("msgmerge -w79 {0} {0} > {0}.wrap".format(output))
        shutil.move("{}.wrap".format(output), output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", nargs="+", default=".", help="The directory or directories containing XML files to collect."
    )
    parser.add_argument("--output", "-o", default="translation_catalog.pot", help="The path to the output file.")
    args = parser.parse_args()

    output = os.path.abspath(args.output)
    if not os.path.isdir(os.path.dirname(output)) or not output.endswith(".pot"):
        print_error("Invalid output path: {}".format(output))
        exit(1)

    classes = OrderedDict()
    for path in args.path:
        if not os.path.isdir(path):
            print_error("Invalid working directory path: {}".format(path))
            exit(1)

        print("\nCurrent working dir: {}".format(path))

        path_classes = OrderedDict()  ## dictionary of key=class_name, value=DescList objects
        _collect_classes_dir(path, path_classes)
        classes.update(path_classes)

    classes = OrderedDict(sorted(classes.items(), key=lambda kv: kv[0].lower()))
    unique_msgs = _make_translation_catalog(classes)
    _generate_translation_catalog_file(unique_msgs, output)


if __name__ == "__main__":
    main()
