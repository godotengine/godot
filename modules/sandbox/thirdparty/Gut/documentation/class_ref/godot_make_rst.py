#!/usr/bin/env python3
# ##############################################################################
#
# Adapted from Godot's make_rst.py as of ~8/25/24.
# https://github.com/godotengine/godot/blob/master/doc/tools/make_rst.py
#
# This aims to generate class reference rst files for plugins.  It is meant to
# operate on a directory of xml files generated using --no-docbase.  It only
# generates rst files for scripts (their doctool xml representation) that
# that have a description (a ## comment at the top), so that only user relevant
# files are included.
#
# Changes
#   - This script has been broken up into multiple scripts.  It's a rough break up.
#     Should get better with time.
#   - Does not generate an index.rst (commented out in main)
#   - Does not include scripts that do not have any description.
#   - Does not include private methods or properties unless they have a
#     description.
#   - Sorts methods by name
#   - Class names for scripts without a class_name will be
#        path/to/file/filename.gd
#        path/to/file/filename.gd.InnerClassName
#     You can use [path/to/file/filename.gd] inside your doc comments to link to
#     pages.  This is also how they appear in the TOC.
#   - Since this is intended to be used with the --no-docbase doctool option,
#     any reference to a class it can't find is assumed to be a Godot class and
#     links to the Godot docs (latest).  A message is printed on the first
#     occurance of each unknown class.
#   - The DO NOT EDIT THIS warnings are altered to indicate this is GUT stuff.
#   - Changed how missing description messages are generated in output and
#     translation entries.  See no_description method.  Pretty simple.
#   - Deprecated methods are grouped together at the bottom of the list of
#     methods.
#   - Does not list Variant datatype for method parameters.  I barely use types
#     in GUT, so it is just noise.  Change marked in make_method_signature.
#
#
# Additional Doc Comment BBCode tags:
# NOTE:  These do not work with the in-engine documentation and will appear
#        in the comments unaltered.
#
#   Unordered Lists
#   - [li][/li] support for list items.  You need a [br] before the first [li].
#     I could add a TON more code so you don't have to...anyway...[li] turns
#     into "* " and [/li] turns into "\n".  It's hacked together, but it works.
#   - [wiki][/wiki] Creates a link to a wiki page.  Very specific to this repo's
#     structure, but you should be able to read it and adapt.
#
#
# Additional Doc Comment Anotations:
# NOTE:  These do not apply to in-engine documentation and the annotations
#        will appear in the in-engine method descriptions.
#
#   Classes (use these in the description, NOT the brief_description):
#   - @ignore-uncommented:  Public members that are not commented will not be
#     included unless that have a doc comment.  Currently works for:
#       - Methods
#       - Properties
#       - Constants
#       - Signals
#
#   Methods:
#   - @ignore - The method will not appear in generated documentation.
#   - @internal -  This will cause the method to be listed seperately (like
#     deprecated methods).  This is for methods that are public but aren't
#     really supposed to be consumed by the general public.
#
#   Properties
#   - @ignore - The property will not appear in generated documentation.
#
#   Signals
#   - @ignore - The signal will not appear in the generated documenration.
# ##############################################################################


# This script makes RST files from the XML class reference for use with the online docs.

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

root_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(root_directory)  # Include the root directory

# Import hardcoded version information from version.py
import godot_version as version # noqa: E402
import doc_bbcode_to_rst as bb2rst
from godot_classes import *
from godot_consts import *
import logger as lgr
import bitwes

# $DOCS_URL/path/to/page.html(#fragment-tag)


# Based on reStructuredText inline markup recognition rules
# https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#inline-markup-recognition-rules




# ------------------
# -------- START bitwes methods/vars --------
# ------------------
# Used this method to change the message in one place and make it easier to be
# sure translation entries and calls to translate match.
def no_description(name):
    return "No description"
    # return f"There is currently no description for this {name}. Please help us by :ref:`contributing one <doc_updating_the_class_reference>`!"


def no_description_container(f, name):
    f.write(".. container:: contribute\n\n\t")
    f.write(
        translate(no_description(name))
        + "\n\n"
    )


def make_inheritance_tree(f, class_def, state):
    # Ascendants
    if class_def.inherits:
        inherits = class_def.inherits.strip()
        f.write(f'**{translate("Inherits:")}** ')
        first = True
        while inherits in state.classes:
            if not first:
                f.write(" **<** ")
            else:
                first = False

            f.write(make_type(inherits, state))
            inode = state.classes[inherits].inherits
            if inode:
                inherits = inode.strip()
            else:
                break
        # If we didn't ever print anything then the class wasn't found so
        # we just use it and assume it is a Godot class (bitwes).
        if(first):
            f.write(bitwes.make_type_link(class_def.inherits.strip()))
        f.write("\n\n")

    # Descendants
    inherited: List[str] = []
    for c in state.classes.values():
        if c.inherits and c.inherits.strip() == class_def.name:
            inherited.append(c.name)

    if len(inherited):
        f.write(f'**{translate("Inherited By:")}** ')
        for i, child in enumerate(inherited):
            if i > 0:
                f.write(", ")
            f.write(make_type(child, state))
        f.write("\n\n")


def make_class_description(f, class_def, state):
    has_any_description = False

    if class_def.brief_description is not None and class_def.brief_description.strip() != "":
        has_any_description = True

        f.write(f"{bb2rst.format_text_block(class_def.brief_description.strip(), class_def, state)}\n\n")

    if class_def.description is not None and class_def.description.strip() != "":
        has_any_description = True

        f.write(".. rst-class:: classref-introduction-group\n\n")
        f.write(make_heading("Description", "-"))

        f.write(f"{bb2rst.format_text_block(class_def.description.strip(), class_def, state)}\n\n")

    if not has_any_description:
        no_description_container(f, "class")

    if class_def.name in CLASSES_WITH_CSHARP_DIFFERENCES:
        f.write(".. note::\n\n\t")
        f.write(
            translate(
                "There are notable differences when using this API with C#. See :ref:`doc_c_sharp_differences` for more information."
            )
            + "\n\n"
        )


def make_method_table(f, class_def, state):
    ml = []
    dep = []
    internal = []
    for key in sorted(class_def.methods.keys()): # list by name
        for m in class_def.methods[key]:
            if(class_def.ignore_uncommented and m.is_description_empty()):
                continue

            to_append = make_method_signature(class_def, m, "method", state)
            if(m.deprecated is not None):
                dep.append(("Deprecated", to_append[0], to_append[1]))
            elif(m.internal):
                internal.append(("Internal Use", to_append[0], to_append[1]))
            elif(not m.ignore):
                ml.append(to_append)

    f.write(".. rst-class:: classref-reftable-group\n\n")
    f.write(make_heading("Methods", "-"))

    format_table(f, ml)
    format_table(f, dep)
    format_table(f, internal)


def make_method_descriptions(f, class_def, state):
    f.write(make_separator(True))
    f.write(".. rst-class:: classref-descriptions-group\n\n")
    f.write(make_heading("Method Descriptions", "-"))

    index = 0

    for method_list in class_def.methods.values():
        if(class_def.ignore_uncommented and method_list[0].is_description_empty()):
            continue

        for i, m in enumerate(method_list):
            if(m.ignore):
                continue

            if index != 0:
                f.write(make_separator())

            # Create method signature and anchor point.

            self_link = ""

            if i == 0:
                method_qualifier = ""
                if m.name.startswith("_"):
                    method_qualifier = "private_"
                method_anchor = f"class_{class_def.name}_{method_qualifier}method_{m.name}"
                f.write(f".. _{method_anchor}:\n\n")
                self_link = f" :ref:`🔗<{method_anchor}>`"

            f.write(".. rst-class:: classref-method\n\n")

            ret_type, signature = make_method_signature(class_def, m, "", state)

            f.write(f"{ret_type} {signature}{self_link}\n\n")

            # Add method description, or a call to action if it's missing.

            f.write(make_deprecated_experimental(m, state))

            if m.description is not None and m.description.strip() != "":
                f.write(f"{bb2rst.format_text_block(m.description.strip(), m, state)}\n\n")
            elif m.deprecated is None and m.experimental is None:
                lgr.vprint(f'Missing method description {class_def.name}.{m.name}')
                no_description_container(f, "method")

            index += 1


def make_property_table(f, class_def, state):
    f.write(".. rst-class:: classref-reftable-group\n\n")
    f.write(make_heading("Properties", "-"))

    ml = []
    for property_def in class_def.properties.values():
        if(property_def.ignore or class_def.ignore_uncommented and property_def.is_description_empty()):
            continue

        type_rst = property_def.type_name.to_rst(state)
        default = property_def.default_value
        if default is not None and property_def.overrides:
            ref = (
                f":ref:`{property_def.overrides}<class_{property_def.overrides}_property_{property_def.name}>`"
            )
            # Not using translate() for now as it breaks table formatting.
            ml.append((type_rst, property_def.name, f"{default} (overrides {ref})"))
        else:
            ref = f":ref:`{property_def.name}<class_{class_def.name}_property_{property_def.name}>`"
            ml.append((type_rst, ref, default))

    format_table(f, ml, True)


def make_property_descriptions(f, class_def, state):
    f.write(make_separator(True))
    f.write(".. rst-class:: classref-descriptions-group\n\n")
    f.write(make_heading("Property Descriptions", "-"))

    index = 0

    for property_def in class_def.properties.values():
        if property_def.overrides or property_def.ignore or \
            (class_def.ignore_uncommented and property_def.is_description_empty()):
            continue

        if index != 0:
            f.write(make_separator())

        # Create property signature and anchor point.

        property_anchor = f"class_{class_def.name}_property_{property_def.name}"
        f.write(f".. _{property_anchor}:\n\n")
        self_link = f":ref:`🔗<{property_anchor}>`"
        f.write(".. rst-class:: classref-property\n\n")

        property_default = ""
        if property_def.default_value is not None:
            property_default = f" = {property_def.default_value}"
        f.write(
            f"{property_def.type_name.to_rst(state)} **{property_def.name}**{property_default} {self_link}\n\n"
        )

        # Create property setter and getter records.

        property_setget = ""

        if property_def.setter is not None and not property_def.setter.startswith("_"):
            property_setter = make_setter_signature(class_def, property_def, state)
            property_setget += f"- {property_setter}\n"

        if property_def.getter is not None and not property_def.getter.startswith("_"):
            property_getter = make_getter_signature(class_def, property_def, state)
            property_setget += f"- {property_getter}\n"

        if property_setget != "":
            f.write(".. rst-class:: classref-property-setget\n\n")
            f.write(property_setget)
            f.write("\n")

        # Add property description, or a call to action if it's missing.

        f.write(make_deprecated_experimental(property_def, state))

        if not property_def.is_description_empty():
            f.write(f"{bb2rst.format_text_block(property_def.description.strip(), property_def, state)}\n\n")
            if property_def.type_name.type_name in PACKED_ARRAY_TYPES:
                tmp = f"[b]Note:[/b] The returned array is [i]copied[/i] and any changes to it will not update the original property value. See [{property_def.type_name.type_name}] for more details."
                f.write(f"{bb2rst.format_text_block(tmp, property_def, state)}\n\n")
        elif property_def.deprecated is None and property_def.experimental is None:
            no_description_container(f, "property")

        index += 1


def make_constructor_table(f, class_def, state):
    f.write(".. rst-class:: classref-reftable-group\n\n")
    f.write(make_heading("Constructors", "-"))

    ml = []
    for method_list in class_def.constructors.values():
        for m in method_list:
            ml.append(make_method_signature(class_def, m, "constructor", state))

    format_table(f, ml)


def make_constructor_descriptions(f, class_def, state):
    f.write(make_separator(True))
    f.write(".. rst-class:: classref-descriptions-group\n\n")
    f.write(make_heading("Constructor Descriptions", "-"))

    index = 0

    for method_list in class_def.constructors.values():
        for i, m in enumerate(method_list):

            if index != 0:
                f.write(make_separator())

            # Create constructor signature and anchor point.

            self_link = ""
            if i == 0:
                constructor_anchor = f"class_{class_def.name}_constructor_{m.name}"
                f.write(f".. _{constructor_anchor}:\n\n")
                self_link = f" :ref:`🔗<{constructor_anchor}>`"

            f.write(".. rst-class:: classref-constructor\n\n")

            ret_type, signature = make_method_signature(class_def, m, "", state)
            f.write(f"{ret_type} {signature}{self_link}\n\n")

            # Add constructor description, or a call to action if it's missing.

            f.write(make_deprecated_experimental(m, state))

            if not m.is_description_empty():
                f.write(f"{bb2rst.format_text_block(m.description.strip(), m, state)}\n\n")
            elif m.deprecated is None and m.experimental is None:
                f.write(".. container:: contribute\n\n\t")
                f.write(
                    translate(no_description("constructor"))
                    + "\n\n"
                )

            index += 1


def make_operator_table(f, class_def, state):
    f.write(".. rst-class:: classref-reftable-group\n\n")
    f.write(make_heading("Operators", "-"))

    ml = []
    for method_list in class_def.operators.values():
        for m in method_list:
            ml.append(make_method_signature(class_def, m, "operator", state))

    format_table(f, ml)


def make_operator_descriptions(f, class_def, state):
    f.write(make_separator(True))
    f.write(".. rst-class:: classref-descriptions-group\n\n")
    f.write(make_heading("Operator Descriptions", "-"))

    index = 0

    for method_list in class_def.operators.values():
        for i, m in enumerate(method_list):
            if index != 0:
                f.write(make_separator())

            # Create operator signature and anchor point.

            operator_anchor = f"class_{class_def.name}_operator_{sanitize_operator_name(m.name, state)}"
            for parameter in m.parameters:
                operator_anchor += f"_{parameter.type_name.type_name}"
            f.write(f".. _{operator_anchor}:\n\n")
            self_link = f":ref:`🔗<{operator_anchor}>`"

            f.write(".. rst-class:: classref-operator\n\n")

            ret_type, signature = make_method_signature(class_def, m, "", state)
            f.write(f"{ret_type} {signature} {self_link}\n\n")

            # Add operator description, or a call to action if it's missing.

            f.write(make_deprecated_experimental(m, state))

            if m.description is not None and m.description.strip() != "":
                f.write(f"{bb2rst.format_text_block(m.description.strip(), m, state)}\n\n")
            elif m.deprecated is None and m.experimental is None:
                f.write(".. container:: contribute\n\n\t")
                f.write(
                    translate(no_description("operator"))
                    + "\n\n"
                )

            index += 1


def make_theme_properties_table(f, class_def, state):
    f.write(".. rst-class:: classref-reftable-group\n\n")
    f.write(make_heading("Theme Properties", "-"))

    ml = []
    for theme_item_def in class_def.theme_items.values():
        ref = f":ref:`{theme_item_def.name}<class_{class_def.name}_theme_{theme_item_def.data_name}_{theme_item_def.name}>`"
        ml.append((theme_item_def.type_name.to_rst(state), ref, theme_item_def.default_value))

    format_table(f, ml, True)


def make_theme_property_descriptions(f, class_def, state):
    f.write(make_separator(True))
    f.write(".. rst-class:: classref-descriptions-group\n\n")
    f.write(make_heading("Theme Property Descriptions", "-"))

    index = 0

    for theme_item_def in class_def.theme_items.values():
        if index != 0:
            f.write(make_separator())

        # Create theme property signature and anchor point.

        theme_item_anchor = f"class_{class_def.name}_theme_{theme_item_def.data_name}_{theme_item_def.name}"
        f.write(f".. _{theme_item_anchor}:\n\n")
        self_link = f":ref:`🔗<{theme_item_anchor}>`"
        f.write(".. rst-class:: classref-themeproperty\n\n")

        theme_item_default = ""
        if theme_item_def.default_value is not None:
            theme_item_default = f" = {theme_item_def.default_value}"
        f.write(
            f"{theme_item_def.type_name.to_rst(state)} **{theme_item_def.name}**{theme_item_default} {self_link}\n\n"
        )

        # Add theme property description, or a call to action if it's missing.

        f.write(make_deprecated_experimental(theme_item_def, state))

        if theme_item_def.text is not None and theme_item_def.text.strip() != "":
            f.write(f"{bb2rst.format_text_block(theme_item_def.text.strip(), theme_item_def, state)}\n\n")
        elif theme_item_def.deprecated is None and theme_item_def.experimental is None:
            f.write(".. container:: contribute\n\n\t")
            f.write(
                translate(no_description("property"))
                + "\n\n"
            )

        index += 1


def make_constant_descriptions(f, class_def, state):
    num_printed = 0
    for constant in class_def.constants.values():
        if(class_def.ignore_uncommented and constant.text.strip() == ""):
            continue

        num_printed += 1
        if(num_printed == 1):
            f.write(make_separator(True))
            f.write(".. rst-class:: classref-descriptions-group\n\n")
            f.write(make_heading("Constants", "-"))

        # Create constant signature and anchor point.

        constant_anchor = f"class_{class_def.name}_constant_{constant.name}"
        f.write(f".. _{constant_anchor}:\n\n")
        self_link = f":ref:`🔗<{constant_anchor}>`"
        f.write(".. rst-class:: classref-constant\n\n")

        f.write(f"**{constant.name}** = ``{constant.value}`` {self_link}\n\n")

        # Add constant description.

        f.write(make_deprecated_experimental(constant, state))

        if constant.text is not None and constant.text.strip() != "":
            f.write(f"{bb2rst.format_text_block(constant.text.strip(), constant, state)}")
        elif constant.deprecated is None and constant.experimental is None:
            no_description_container(f, "constant")

        f.write("\n\n")


def make_signal_descriptions(f, class_def, state):
    index = 0
    for signal in class_def.signals.values():
        if(signal.ignore or (class_def.ignore_uncommented and signal.is_description_empty())):
            continue

        if(index == 0):
            f.write(make_separator(True))
            f.write(".. rst-class:: classref-descriptions-group\n\n")
            f.write(make_heading("Signals", "-"))

        # if index != 0:
        #     f.write(make_separator())

        # Create signal signature and anchor point.

        signal_anchor = f"class_{class_def.name}_signal_{signal.name}"
        f.write(f".. _{signal_anchor}:\n\n")
        self_link = f":ref:`🔗<{signal_anchor}>`"
        f.write(".. rst-class:: classref-signal\n\n")

        _, signature = make_method_signature(class_def, signal, "", state)
        f.write(f"{signature} {self_link}\n\n")

        # Add signal description, or a call to action if it's missing.

        f.write(make_deprecated_experimental(signal, state))

        if signal.description is not None and signal.description.strip() != "":
            f.write(f"{bb2rst.format_text_block(signal.description.strip(), signal, state)}\n\n")
        elif signal.deprecated is None and signal.experimental is None:
            pass# no_description_container(f, "signal")

        index += 1


def make_enum_descriptions(f, class_def, state):
    f.write(make_separator(True))
    f.write(".. rst-class:: classref-descriptions-group\n\n")
    f.write(make_heading("Enumerations", "-"))

    index = 0

    for e in class_def.enums.values():
        if index != 0:
            f.write(make_separator())

        # Create enumeration signature and anchor point.

        enum_anchor = f"enum_{class_def.name}_{e.name}"
        f.write(f".. _{enum_anchor}:\n\n")
        self_link = f":ref:`🔗<{enum_anchor}>`"
        f.write(".. rst-class:: classref-enumeration\n\n")

        if e.is_bitfield:
            f.write(f"flags **{e.name}**: {self_link}\n\n")
        else:
            f.write(f"enum **{e.name}**: {self_link}\n\n")

        for value in e.values.values():
            # Also create signature and anchor point for each enum constant.

            f.write(f".. _class_{class_def.name}_constant_{value.name}:\n\n")
            f.write(".. rst-class:: classref-enumeration-constant\n\n")

            f.write(f"{e.type_name.to_rst(state)} **{value.name}** = ``{value.value}``\n\n")

            # Add enum constant description.

            f.write(make_deprecated_experimental(value, state))

            if value.text is not None and value.text.strip() != "":
                f.write(f"{bb2rst.format_text_block(value.text.strip(), value, state)}")
            elif value.deprecated is None and value.experimental is None:
                f.write(".. container:: contribute\n\n\t")
                f.write(
                    translate(no_description("enum"))
                    + "\n\n"
                )

            f.write("\n\n")

        index += 1


def make_annotation_descriptions(f, class_def, state):
    f.write(make_separator(True))
    f.write(make_heading("Annotations", "-"))

    index = 0

    for method_list in class_def.annotations.values():  # type: ignore
        for i, m in enumerate(method_list):
            if index != 0:
                f.write(make_separator())

            # Create annotation signature and anchor point.

            self_link = ""
            if i == 0:
                annotation_anchor = f"class_{class_def.name}_annotation_{m.name}"
                f.write(f".. _{annotation_anchor}:\n\n")
                self_link = f" :ref:`🔗<{annotation_anchor}>`"

            f.write(".. rst-class:: classref-annotation\n\n")

            _, signature = make_method_signature(class_def, m, "", state)
            f.write(f"{signature}{self_link}\n\n")

            # Add annotation description, or a call to action if it's missing.

            if m.description is not None and m.description.strip() != "":
                f.write(f"{bb2rst.format_text_block(m.description.strip(), m, state)}\n\n")
            else:
                f.write(".. container:: contribute\n\n\t")
                f.write(
                    translate(no_description("annotation"))
                    + "\n\n"
                )

            index += 1
# ------------------
# -------- END bitwes methods/vars --------
# ------------------




# Used to translate section headings and other hardcoded strings when required with
# the --lang argument. The BASE_STRINGS list should be synced with what we actually
# write in this script (check `translate()` uses), and also hardcoded in
# `scripts/extract_classes.py` (godotengine/godot-editor-l10n repo) to include them in the source POT file.
BASE_STRINGS = [
    "All classes",
    "Globals",
    "Nodes",
    "Resources",
    "Editor-only",
    "Other objects",
    "Variant types",
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
    "Annotations",
    "Property Descriptions",
    "Constructor Descriptions",
    "Method Descriptions",
    "Operator Descriptions",
    "Theme Property Descriptions",
    "Inherits:",
    "Inherited By:",
    "(overrides %s)",
    "Default",
    "Setter",
    "value",
    "Getter",
    "This method should typically be overridden by the user to have any effect.",
    "This method has no side effects. It doesn't modify any of the instance's member variables.",
    "This method accepts any number of arguments after the ones described here.",
    "This method is used to construct a type.",
    "This method doesn't need an instance to be called, so it can be called directly using the class name.",
    "This method describes a valid operator to use with this type as left-hand operand.",
    "This value is an integer composed as a bitmask of the following flags.",
    "No return value.",
    no_description("class"),
    no_description("signal"),
    no_description("enum"),
    no_description("constant"),
    no_description("annotation"),
    no_description("property"),
    no_description("constructor"),
    no_description("method"),
    no_description("operator"),
    no_description("theme property"),
    "There are notable differences when using this API with C#. See :ref:`doc_c_sharp_differences` for more information.",
    "Deprecated:",
    "Experimental:",
    "This signal may be changed or removed in future versions.",
    "This constant may be changed or removed in future versions.",
    "This property may be changed or removed in future versions.",
    "This constructor may be changed or removed in future versions.",
    "This method may be changed or removed in future versions.",
    "This operator may be changed or removed in future versions.",
    "This theme property may be changed or removed in future versions.",
    "[b]Note:[/b] The returned array is [i]copied[/i] and any changes to it will not update the original property value. See [%s] for more details.",
]

def translate(string: str) -> str:
    """Translate a string based on translations sourced from `doc/translations/*.po`
    for a language if defined via the --lang command line argument.
    Returns the original string if no translation exists.
    """
    return strings_l10n.get(string, string)


# Removed uses of this, the uses of this were not using the result of this.
# def get_git_branch() -> str:
#     if hasattr(version, "docs") and version.docs != "latest":
#         return version.docs
#     return "master"


def make_rst_class(class_def: ClassDef, state: State, dry_run: bool, output_dir: str) -> None:
    class_name = class_def.name
    filename = os.path.join(output_dir, f"class_{class_name.lower()}.rst")

    # bitwes --
    # shortcircuit when class has no description.
    if((class_def.description is None or class_def.description.strip() == "") and
        (class_def.brief_description is None or class_def.brief_description.strip() == "")):
        lgr.vprint("SKIP", class_name, ".  No description.")
        return

    adjusted_class_name = class_name

    # converts '"path/to/script.gd"' to 'path_to_script'
    if('.gd' in class_name.strip()):
        adjusted_class_name = class_name.lower()\
            .replace('.gd', "")\
            .replace(os.sep, '_')
        # filename will be <output_dir>/class_path_to_script.rst
        filename = os.path.join(output_dir, f"class_{adjusted_class_name}.rst")

    lgr.print_style("green", "Writing ", f'{class_name} -> {filename}')
    # -- bitwes

    with open(
        os.devnull if dry_run else filename,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        # Remove the "Edit on Github" button from the online docs page.
        f.write(":github_url: hide\n\n")

        # Add keywords metadata.
        if class_def.keywords is not None and class_def.keywords != "":
            f.write(f".. meta::\n\t:keywords: {class_def.keywords}\n\n")

        # Warn contributors not to edit this file directly.
        # Also provide links to the source files for reference.

        # git_branch = get_git_branch()
        # source_xml_path = os.path.relpath(class_def.filepath, root_directory).replace("\\", "/")
        # source_github_url = f"https://github.com/godotengine/godot/tree/{git_branch}/{source_xml_path}"
        # generator_github_url = f"https://github.com/godotengine/godot/tree/{git_branch}/doc/tools/make_rst.py"

        f.write(".. DO NOT EDIT THIS FILE!!!\n")
        f.write(".. Generated automatically from GUT Plugin sources.\n")
        f.write(f".. Generator: documentation/godot_make_rst.py.\n")

        # Document reference id and header.
        f.write(f".. _class_{class_name}:\n\n")
        f.write(make_heading(class_name.replace('"', ''), "=", False))

        f.write(make_deprecated_experimental(class_def, state))

        make_inheritance_tree(f, class_def, state)

        make_class_description(f, class_def, state)

        # Online tutorials
        if len(class_def.tutorials) > 0:
            f.write(".. rst-class:: classref-introduction-group\n\n")
            f.write(make_heading("Tutorials", "-"))

            for url, title in class_def.tutorials:
                f.write(f"- {bb2rst.make_link(url, title)}\n\n")

        ### REFERENCE TABLES ###

        if len(class_def.properties) > 0:
            make_property_table(f, class_def, state)

        if len(class_def.constructors) > 0:
            make_constructor_table(f, class_def, state)

        if len(class_def.methods) > 0:
            make_method_table(f, class_def, state)

        if len(class_def.operators) > 0:
            make_operator_table(f, class_def, state)


        if len(class_def.theme_items) > 0:
            make_theme_properties_table(f, class_def, state)


        ### DETAILED DESCRIPTIONS ###


        # Signal descriptions
        if len(class_def.signals) > 0:
            make_signal_descriptions(f, class_def, state)

        # Enumeration descriptions
        if len(class_def.enums) > 0:
            make_enum_descriptions(f, class_def, state)

        # Constant descriptions
        if len(class_def.constants) > 0:
            make_constant_descriptions(f, class_def, state)

        # Annotation descriptions
        if len(class_def.annotations) > 0:
            make_annotation_descriptions(f, class_def, state)

        # Property descriptions
        if any(not p.overrides for p in class_def.properties.values()) > 0:
            make_property_descriptions(f, class_def, state)

        # Constructor, Method, Operator descriptions
        if len(class_def.constructors) > 0:
            make_constructor_descriptions(f, class_def, state)

        # Method descrptions
        if len(class_def.methods) > 0:
            make_method_descriptions(f, class_def, state)

        if len(class_def.operators) > 0:
            make_operator_descriptions(f, class_def, state)

        # Theme property descriptions
        if len(class_def.theme_items) > 0:
            make_theme_property_descriptions(f, class_def, state)

        f.write(make_footer())


def make_method_signature(
    class_def: ClassDef, definition: Union[AnnotationDef, MethodDef, SignalDef], ref_type: str, state: State
) -> Tuple[str, str]:
    ret_type = ""

    if isinstance(definition, MethodDef):
        ret_type = definition.return_type.to_rst(state)

    qualifiers = None
    if isinstance(definition, (MethodDef, AnnotationDef)):
        qualifiers = definition.qualifiers

    out = ""
    if isinstance(definition, MethodDef) and ref_type != "":
        if ref_type == "operator":
            op_name = definition.name.replace("<", "\\<")  # So operator "<" gets correctly displayed.
            out += f":ref:`{op_name}<class_{class_def.name}_{ref_type}_{sanitize_operator_name(definition.name, state)}"
            for parameter in definition.parameters:
                out += f"_{parameter.type_name.type_name}"
            out += ">`"
        elif ref_type == "method":
            ref_type_qualifier = ""
            if definition.name.startswith("_"):
                ref_type_qualifier = "private_"
            out += f":ref:`{definition.name}<class_{class_def.name}_{ref_type_qualifier}{ref_type}_{definition.name}>`"
        else:
            out += f":ref:`{definition.name}<class_{class_def.name}_{ref_type}_{definition.name}>`"
    else:
        out += f"**{definition.name}**"

    out += "\\ ("
    for i, arg in enumerate(definition.parameters):
        if i > 0:
            out += ", "
        else:
            out += "\\ "

        # hide variant datatype, too noisey for me (bitwes)
        if(arg.type_name.type_name == "Variant"):
            out += f"{arg.name}"
        else:
            out += f"{arg.name}\\: {arg.type_name.to_rst(state)}"

        if arg.default_value is not None:
            out += f" = {arg.default_value}"

    if qualifiers is not None and "vararg" in qualifiers:
        if len(definition.parameters) > 0:
            out += ", ..."
        else:
            out += "\\ ..."

    out += "\\ )"

    if qualifiers is not None:
        # Use substitutions for abbreviations. This is used to display tooltips on hover.
        # See `make_footer()` for descriptions.
        for qualifier in qualifiers.split():
            out += f" |{qualifier}|"

    return ret_type, out


def make_setter_signature(class_def: ClassDef, property_def: PropertyDef, state: State) -> str:
    if property_def.setter is None:
        return ""

    # If setter is a method available as a method definition, we use that.
    if property_def.setter in class_def.methods:
        setter = class_def.methods[property_def.setter][0]
    # Otherwise we fake it with the information we have available.
    else:
        setter_params: List[ParameterDef] = []
        setter_params.append(ParameterDef("value", property_def.type_name, None))
        setter = MethodDef(property_def.setter, TypeName("void"), setter_params, None, None)

    ret_type, signature = make_method_signature(class_def, setter, "", state)
    return f"{ret_type} {signature}"


def make_getter_signature(class_def: ClassDef, property_def: PropertyDef, state: State) -> str:
    if property_def.getter is None:
        return ""

    # If getter is a method available as a method definition, we use that.
    if property_def.getter in class_def.methods:
        getter = class_def.methods[property_def.getter][0]
    # Otherwise we fake it with the information we have available.
    else:
        getter_params: List[ParameterDef] = []
        getter = MethodDef(property_def.getter, property_def.type_name, getter_params, None, None)

    ret_type, signature = make_method_signature(class_def, getter, "", state)
    return f"{ret_type} {signature}"


def make_deprecated_experimental(item: DefinitionBase, state: State) -> str:
    result = ""

    if item.deprecated is not None:
        deprecated_prefix = translate("Deprecated:")
        if item.deprecated.strip() == "":
            default_message = translate(f"This {item.definition_name} may be changed or removed in future versions.")
            result += f"**{deprecated_prefix}** {default_message}\n\n"
        else:
            result += f"**{deprecated_prefix}** {bb2rst.format_text_block(item.deprecated.strip(), item, state)}\n\n"

    if item.experimental is not None:
        experimental_prefix = translate("Experimental:")
        if item.experimental.strip() == "":
            default_message = translate(f"This {item.definition_name} may be changed or removed in future versions.")
            result += f"**{experimental_prefix}** {default_message}\n\n"
        else:
            result += f"**{experimental_prefix}** {bb2rst.format_text_block(item.experimental.strip(), item, state)}\n\n"

    return result


def make_heading(title: str, underline: str, l10n: bool = True) -> str:
    if l10n:
        new_title = translate(title)
        if new_title != title:
            title = new_title
            underline *= 2  # Double length to handle wide chars.
    return f"{title}\n{(underline * len(title))}\n\n"


def make_footer() -> str:
    # Generate reusable abbreviation substitutions.
    # This way, we avoid bloating the generated rST with duplicate abbreviations.
    virtual_msg = translate("This method should typically be overridden by the user to have any effect.")
    const_msg = translate("This method has no side effects. It doesn't modify any of the instance's member variables.")
    vararg_msg = translate("This method accepts any number of arguments after the ones described here.")
    constructor_msg = translate("This method is used to construct a type.")
    static_msg = translate(
        "This method doesn't need an instance to be called, so it can be called directly using the class name."
    )
    operator_msg = translate("This method describes a valid operator to use with this type as left-hand operand.")
    bitfield_msg = translate("This value is an integer composed as a bitmask of the following flags.")
    void_msg = translate("No return value.")

    return (
        f".. |virtual| replace:: :abbr:`virtual ({virtual_msg})`\n"
        f".. |const| replace:: :abbr:`const ({const_msg})`\n"
        f".. |vararg| replace:: :abbr:`vararg ({vararg_msg})`\n"
        f".. |constructor| replace:: :abbr:`constructor ({constructor_msg})`\n"
        f".. |static| replace:: :abbr:`static ({static_msg})`\n"
        f".. |operator| replace:: :abbr:`operator ({operator_msg})`\n"
        f".. |bitfield| replace:: :abbr:`BitField ({bitfield_msg})`\n"
        f".. |void| replace:: :abbr:`void ({void_msg})`\n"
    )


def make_separator(section_level: bool = False) -> str:
    separator_class = "item"
    if section_level:
        separator_class = "section"

    return f".. rst-class:: classref-{separator_class}-separator\n\n----\n\n"


def make_rst_index(grouped_classes: Dict[str, List[str]], dry_run: bool, output_dir: str) -> None:
    with open(
        os.devnull if dry_run else os.path.join(output_dir, "index.rst"), "w", encoding="utf-8", newline="\n"
    ) as f:
        # Remove the "Edit on Github" button from the online docs page, and disallow user-contributed notes
        # on the index page. User-contributed notes are allowed on individual class pages.
        f.write(":github_url: hide\n:allow_comments: False\n\n")

        # Warn contributors not to edit this file directly.
        # Also provide links to the source files for reference.

        # git_branch = get_git_branch()
        # generator_github_url = f"https://github.com/godotengine/godot/tree/{git_branch}/doc/tools/make_rst.py"

        f.write(".. DO NOT EDIT THIS FILE!!!\n")
        f.write(".. Generated automatically from GUT Plugin sources.\n")
        # f.write(f".. Generator: {generator_github_url}.\n\n")

        f.write(".. _doc_class_reference:\n\n")

        f.write(make_heading("All classes", "="))

        for group_name in CLASS_GROUPS:
            if group_name in grouped_classes:
                f.write(make_heading(CLASS_GROUPS[group_name], "="))

                f.write(".. toctree::\n")
                f.write("    :maxdepth: 1\n")
                f.write(f"    :name: toc-class-ref-{group_name}s\n")
                f.write("\n")

                if group_name in CLASS_GROUPS_BASE:
                    f.write(f"    class_{CLASS_GROUPS_BASE[group_name].lower()}\n")

                for class_name in grouped_classes[group_name]:
                    if group_name in CLASS_GROUPS_BASE and CLASS_GROUPS_BASE[group_name].lower() == class_name.lower():
                        continue

                    adjusted_class_name = f"class_{class_name.lower()}"
                    if('.gd"' in adjusted_class_name.strip()):
                        adjusted_class_name = adjusted_class_name.replace('.gd"', "")\
                            .replace('"', "")\
                            .replace(os.sep, '_')

                    f.write(f"    {adjusted_class_name}\n")

                f.write("\n")


# Formatting helpers.

def format_table(f: TextIO, data: List[Tuple[Optional[str], ...]], remove_empty_columns: bool = False) -> None:
    if len(data) == 0:
        return

    f.write(".. table::\n")
    f.write("   :widths: auto\n\n")

    # Calculate the width of each column first, we will use this information
    # to properly format RST-style tables.
    column_sizes = [0] * len(data[0])
    for row in data:
        for i, text in enumerate(row):
            text_length = len(text or "")
            if text_length > column_sizes[i]:
                column_sizes[i] = text_length

    # Each table row is wrapped in two separators, consecutive rows share the same separator.
    # All separators, or rather borders, have the same shape and content. We compose it once,
    # then reuse it.

    sep = ""
    for size in column_sizes:
        if size == 0 and remove_empty_columns:
            continue
        sep += "+" + "-" * (size + 2)  # Content of each cell is padded by 1 on each side.
    sep += "+\n"

    # Draw the first separator.
    f.write(f"   {sep}")

    # Draw each row and close it with a separator.
    for row in data:
        row_text = "|"
        for i, text in enumerate(row):
            if column_sizes[i] == 0 and remove_empty_columns:
                continue
            row_text += f' {(text or "").ljust(column_sizes[i])} |'
        row_text += "\n"

        f.write(f"   {row_text}")
        f.write(f"   {sep}")

    f.write("\n")


def sanitize_operator_name(dirty_name: str, state: State) -> str:
    clear_name = dirty_name.replace("operator ", "")

    if clear_name == "!=":
        clear_name = "neq"
    elif clear_name == "==":
        clear_name = "eq"

    elif clear_name == "<":
        clear_name = "lt"
    elif clear_name == "<=":
        clear_name = "lte"
    elif clear_name == ">":
        clear_name = "gt"
    elif clear_name == ">=":
        clear_name = "gte"

    elif clear_name == "+":
        clear_name = "sum"
    elif clear_name == "-":
        clear_name = "dif"
    elif clear_name == "*":
        clear_name = "mul"
    elif clear_name == "/":
        clear_name = "div"
    elif clear_name == "%":
        clear_name = "mod"
    elif clear_name == "**":
        clear_name = "pow"

    elif clear_name == "unary+":
        clear_name = "unplus"
    elif clear_name == "unary-":
        clear_name = "unminus"

    elif clear_name == "<<":
        clear_name = "bwsl"
    elif clear_name == ">>":
        clear_name = "bwsr"
    elif clear_name == "&":
        clear_name = "bwand"
    elif clear_name == "|":
        clear_name = "bwor"
    elif clear_name == "^":
        clear_name = "bwxor"
    elif clear_name == "~":
        clear_name = "bwnot"

    elif clear_name == "[]":
        clear_name = "idx"

    else:
        clear_name = "xxx"
        lgr.print_error(f'Unsupported operator type "{dirty_name}", please add the missing rule.', state)

    return clear_name


def load_translations(lang):
    lang_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "translations", "{}.po".format(lang)
    )
    if os.path.exists(lang_file):
        try:
            import polib  # type: ignore
        except ImportError:
            print("Base template strings localization requires `polib`.")
            exit(1)

        pofile = polib.pofile(lang_file)
        for entry in pofile.translated_entries():
            if entry.msgid in BASE_STRINGS:
                strings_l10n[entry.msgid] = entry.msgstr
    else:
        print(f'No PO file at "{lang_file}" for language "{lang}".')


def get_xml_file_list(xml_path):
    file_list: List[str] = []

    for path in xml_path:
        # Cut off trailing slashes so os.path.basename doesn't choke.
        if path.endswith("/") or path.endswith("\\"):
            path = path[:-1]

        if os.path.basename(path) in ["modules", "platform"]:
            for subdir, dirs, _ in os.walk(path):
                if "doc_classes" in dirs:
                    doc_dir = os.path.join(subdir, "doc_classes")
                    class_file_names = (f for f in os.listdir(doc_dir) if f.endswith(".xml"))
                    file_list += (os.path.join(doc_dir, f) for f in class_file_names)
        elif os.path.isdir(path):
            file_list += (os.path.join(path, f) for f in os.listdir(path) if f.endswith(".xml"))
        elif os.path.isfile(path):
            if not path.endswith(".xml"):
                print(f'Got non-.xml file "{path}" in input, skipping.')
                continue

            file_list.append(path)

    return file_list


def new_state_from_xml_files(file_list):
    state = State()
    classes: Dict[str, Tuple[ET.Element, str]] = {}
    for cur_file in file_list:
        try:
            tree = ET.parse(cur_file)
        except ET.ParseError as e:
            lgr.print_error(f"{cur_file}: Parse error while reading the file: {e}", state)
            continue
        doc = tree.getroot()

        name = doc.attrib["name"]
        if name in classes:
            lgr.print_error(f'{cur_file}: Duplicate class "{name}".', state)
            continue

        classes[name] = (doc, cur_file)

    for name, data in classes.items():
        try:
            state.parse_class(data[0], data[1])
        except Exception as e:
            lgr.print_error(f"{name}.xml: Exception while parsing class: {e}", state)
    return state



def get_cmdline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="+", help="A path to an XML file or a directory containing XML files to parse.")
    parser.add_argument("--filter", default="", help="The filepath pattern for XML files to filter.")
    parser.add_argument("--lang", "-l", default="en", help="Language to use for section headings.")
    parser.add_argument(
        "--color",
        action="store_true",
        help="If passed, force colored output even if stdout is not a TTY (useful for continuous integration).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", "-o", default=".", help="The directory to save output .rst files in.")
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="If passed, no output will be generated and XML files are only checked for errors.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If passed, enables verbose printing.",
    )
    return parser.parse_args()


# Entry point for the RST generator.
def main() -> None:
    args = get_cmdline_args()
    lgr.verbose_enabled = args.verbose
    should_color = bool(args.color or sys.stdout.isatty() or os.environ.get("CI"))
    lgr.set_should_color(should_color)

    # Retrieve heading translations for the given language.
    if not args.dry_run and args.lang != "en":
        load_translations(args.lang)

    print("Checking for errors in the XML class reference...")
    file_list = get_xml_file_list(args.path)
    state = new_state_from_xml_files(file_list)
    state.sort_classes()

    # Create the output folder recursively if it doesn't already exist.
    os.makedirs(args.output, exist_ok=True)

    print("Generating the RST class reference...")

    grouped_classes: Dict[str, List[str]] = {}
    pattern = re.compile(args.filter)
    for class_name, class_def in state.classes.items():
        if args.filter and not pattern.search(class_def.filepath):
            continue

        state.current_class = class_name
        class_def.update_class_group(state)
        class_def.strip_privates()

        make_rst_class(class_def, state, args.dry_run, args.output)

        if class_def.class_group not in grouped_classes:
            grouped_classes[class_def.class_group] = []
        grouped_classes[class_def.class_group].append(class_name)

        if class_def.editor_class:
            if "editor" not in grouped_classes:
                grouped_classes["editor"] = []
            grouped_classes["editor"].append(class_name)

    # print("")
    # print("Generating the index file...")

    # make_rst_index(grouped_classes, args.dry_run, args.output)

    # print("")

    # Print out checks.

    if state.script_language_parity_check.hit_count > 0:
        if not args.verbose:
            lgr.print_style("yellow", f'{state.script_language_parity_check.hit_count} code samples failed parity check. Use --verbose to get more information.')
        else:
            lgr.print_style("yellow", f'{state.script_language_parity_check.hit_count} code samples failed parity check:')

            for class_name in state.script_language_parity_check.hit_map.keys():
                class_hits = state.script_language_parity_check.hit_map[class_name]
                lgr.print_style("yellow", f'- {len(class_hits)} hits in class "{class_name}"')

                for context, error in class_hits:
                    print(f"  - {error} in {bb2rst.format_context_name(context)}")
        print("")

    # Print out warnings and errors, or lack thereof, and exit with an appropriate code.

    if state.num_warnings >= 2:
        lgr.print_style("yellow", f'{state.num_warnings} warnings were found in the class reference XML. Please check the messages above.')
    elif state.num_warnings == 1:
        lgr.print_style("yellow", f'1 warning was found in the class reference XML. Please check the messages above.')

    if state.num_errors >= 2:
        lgr.print_style("red", f'{state.num_errors} errors were found in the class reference XML. Please check the messages above.')
    elif state.num_errors == 1:
        lgr.print_style("red", f'1 error was found in the class reference XML. Please check the messages above.')

    if state.num_warnings == 0 and state.num_errors == 0:
        lgr.print_style("green", f'No warnings or errors found in the class reference XML.')
        if not args.dry_run:
            print(f"Wrote reStructuredText files for each class to: {args.output}")
    else:
        exit(1)




if __name__ == "__main__":
    main()
