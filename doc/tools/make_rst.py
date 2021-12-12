#!/usr/bin/env python3

# This script makes RST files from the XML class reference for use with the online docs.

import argparse
import os
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict

# Uncomment to do type checks. I have it commented out so it works below Python 3.5
# from typing import List, Dict, TextIO, Tuple, Iterable, Optional, DefaultDict, Any, Union

# $DOCS_URL/path/to/page.html(#fragment-tag)
GODOT_DOCS_PATTERN = re.compile(r"^\$DOCS_URL/(.*)\.html(#.*)?$")


def print_error(error, state):  # type: (str, State) -> None
    print("ERROR: {}".format(error))
    state.errored = True


class TypeName:
    def __init__(self, type_name, enum=None):  # type: (str, Optional[str]) -> None
        self.type_name = type_name
        self.enum = enum

    def to_rst(self, state):  # type: ("State") -> str
        if self.enum is not None:
            return make_enum(self.enum, state)
        elif self.type_name == "void":
            return "void"
        else:
            return make_type(self.type_name, state)

    @classmethod
    def from_element(cls, element):  # type: (ET.Element) -> "TypeName"
        return cls(element.attrib["type"], element.get("enum"))


class PropertyDef:
    def __init__(
        self, name, type_name, setter, getter, text, default_value, overrides
    ):  # type: (str, TypeName, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]) -> None
        self.name = name
        self.type_name = type_name
        self.setter = setter
        self.getter = getter
        self.text = text
        self.default_value = default_value
        self.overrides = overrides


class ParameterDef:
    def __init__(self, name, type_name, default_value):  # type: (str, TypeName, Optional[str]) -> None
        self.name = name
        self.type_name = type_name
        self.default_value = default_value


class SignalDef:
    def __init__(self, name, parameters, description):  # type: (str, List[ParameterDef], Optional[str]) -> None
        self.name = name
        self.parameters = parameters
        self.description = description


class MethodDef:
    def __init__(
        self, name, return_type, parameters, description, qualifiers
    ):  # type: (str, TypeName, List[ParameterDef], Optional[str], Optional[str]) -> None
        self.name = name
        self.return_type = return_type
        self.parameters = parameters
        self.description = description
        self.qualifiers = qualifiers


class ConstantDef:
    def __init__(self, name, value, text):  # type: (str, str, Optional[str]) -> None
        self.name = name
        self.value = value
        self.text = text


class EnumDef:
    def __init__(self, name):  # type: (str) -> None
        self.name = name
        self.values = OrderedDict()  # type: OrderedDict[str, ConstantDef]


class ThemeItemDef:
    def __init__(
        self, name, type_name, data_name, text, default_value
    ):  # type: (str, TypeName, str, Optional[str], Optional[str]) -> None
        self.name = name
        self.type_name = type_name
        self.data_name = data_name
        self.text = text
        self.default_value = default_value


class ClassDef:
    def __init__(self, name):  # type: (str) -> None
        self.name = name
        self.constants = OrderedDict()  # type: OrderedDict[str, ConstantDef]
        self.enums = OrderedDict()  # type: OrderedDict[str, EnumDef]
        self.properties = OrderedDict()  # type: OrderedDict[str, PropertyDef]
        self.constructors = OrderedDict()  # type: OrderedDict[str, List[MethodDef]]
        self.methods = OrderedDict()  # type: OrderedDict[str, List[MethodDef]]
        self.operators = OrderedDict()  # type: OrderedDict[str, List[MethodDef]]
        self.signals = OrderedDict()  # type: OrderedDict[str, SignalDef]
        self.theme_items = OrderedDict()  # type: OrderedDict[str, ThemeItemDef]
        self.inherits = None  # type: Optional[str]
        self.brief_description = None  # type: Optional[str]
        self.description = None  # type: Optional[str]
        self.tutorials = []  # type: List[Tuple[str, str]]

        # Used to match the class with XML source for output filtering purposes.
        self.filepath = ""  # type: str


class State:
    def __init__(self):  # type: () -> None
        # Has any error been reported?
        self.errored = False
        self.classes = OrderedDict()  # type: OrderedDict[str, ClassDef]
        self.current_class = ""  # type: str

    def parse_class(self, class_root, filepath):  # type: (ET.Element, str) -> None
        class_name = class_root.attrib["name"]

        class_def = ClassDef(class_name)
        self.classes[class_name] = class_def
        class_def.filepath = filepath

        inherits = class_root.get("inherits")
        if inherits is not None:
            class_def.inherits = inherits

        brief_desc = class_root.find("brief_description")
        if brief_desc is not None and brief_desc.text:
            class_def.brief_description = brief_desc.text

        desc = class_root.find("description")
        if desc is not None and desc.text:
            class_def.description = desc.text

        properties = class_root.find("members")
        if properties is not None:
            for property in properties:
                assert property.tag == "member"

                property_name = property.attrib["name"]
                if property_name in class_def.properties:
                    print_error("Duplicate property '{}', file: {}".format(property_name, class_name), self)
                    continue

                type_name = TypeName.from_element(property)
                setter = property.get("setter") or None  # Use or None so '' gets turned into None.
                getter = property.get("getter") or None
                default_value = property.get("default") or None
                if default_value is not None:
                    default_value = "``{}``".format(default_value)
                overrides = property.get("overrides") or None

                property_def = PropertyDef(
                    property_name, type_name, setter, getter, property.text, default_value, overrides
                )
                class_def.properties[property_name] = property_def

        constructors = class_root.find("constructors")
        if constructors is not None:
            for constructor in constructors:
                assert constructor.tag == "constructor"

                method_name = constructor.attrib["name"]
                qualifiers = constructor.get("qualifiers")

                return_element = constructor.find("return")
                if return_element is not None:
                    return_type = TypeName.from_element(return_element)

                else:
                    return_type = TypeName("void")

                params = parse_arguments(constructor)

                desc_element = constructor.find("description")
                method_desc = None
                if desc_element is not None:
                    method_desc = desc_element.text

                method_def = MethodDef(method_name, return_type, params, method_desc, qualifiers)
                if method_name not in class_def.constructors:
                    class_def.constructors[method_name] = []

                class_def.constructors[method_name].append(method_def)

        methods = class_root.find("methods")
        if methods is not None:
            for method in methods:
                assert method.tag == "method"

                method_name = method.attrib["name"]
                qualifiers = method.get("qualifiers")

                return_element = method.find("return")
                if return_element is not None:
                    return_type = TypeName.from_element(return_element)

                else:
                    return_type = TypeName("void")

                params = parse_arguments(method)

                desc_element = method.find("description")
                method_desc = None
                if desc_element is not None:
                    method_desc = desc_element.text

                method_def = MethodDef(method_name, return_type, params, method_desc, qualifiers)
                if method_name not in class_def.methods:
                    class_def.methods[method_name] = []

                class_def.methods[method_name].append(method_def)

        operators = class_root.find("operators")
        if operators is not None:
            for operator in operators:
                assert operator.tag == "operator"

                method_name = operator.attrib["name"]
                qualifiers = operator.get("qualifiers")

                return_element = operator.find("return")
                if return_element is not None:
                    return_type = TypeName.from_element(return_element)

                else:
                    return_type = TypeName("void")

                params = parse_arguments(operator)

                desc_element = operator.find("description")
                method_desc = None
                if desc_element is not None:
                    method_desc = desc_element.text

                method_def = MethodDef(method_name, return_type, params, method_desc, qualifiers)
                if method_name not in class_def.operators:
                    class_def.operators[method_name] = []

                class_def.operators[method_name].append(method_def)

        constants = class_root.find("constants")
        if constants is not None:
            for constant in constants:
                assert constant.tag == "constant"

                constant_name = constant.attrib["name"]
                value = constant.attrib["value"]
                enum = constant.get("enum")
                constant_def = ConstantDef(constant_name, value, constant.text)
                if enum is None:
                    if constant_name in class_def.constants:
                        print_error("Duplicate constant '{}', file: {}".format(constant_name, class_name), self)
                        continue

                    class_def.constants[constant_name] = constant_def

                else:
                    if enum in class_def.enums:
                        enum_def = class_def.enums[enum]

                    else:
                        enum_def = EnumDef(enum)
                        class_def.enums[enum] = enum_def

                    enum_def.values[constant_name] = constant_def

        signals = class_root.find("signals")
        if signals is not None:
            for signal in signals:
                assert signal.tag == "signal"

                signal_name = signal.attrib["name"]

                if signal_name in class_def.signals:
                    print_error("Duplicate signal '{}', file: {}".format(signal_name, class_name), self)
                    continue

                params = parse_arguments(signal)

                desc_element = signal.find("description")
                signal_desc = None
                if desc_element is not None:
                    signal_desc = desc_element.text

                signal_def = SignalDef(signal_name, params, signal_desc)
                class_def.signals[signal_name] = signal_def

        theme_items = class_root.find("theme_items")
        if theme_items is not None:
            for theme_item in theme_items:
                assert theme_item.tag == "theme_item"

                theme_item_name = theme_item.attrib["name"]
                theme_item_data_name = theme_item.attrib["data_type"]
                theme_item_id = "{}_{}".format(theme_item_data_name, theme_item_name)
                if theme_item_id in class_def.theme_items:
                    print_error(
                        "Duplicate theme property '{}' of type '{}', file: {}".format(
                            theme_item_name, theme_item_data_name, class_name
                        ),
                        self,
                    )
                    continue

                default_value = theme_item.get("default") or None
                if default_value is not None:
                    default_value = "``{}``".format(default_value)

                theme_item_def = ThemeItemDef(
                    theme_item_name,
                    TypeName.from_element(theme_item),
                    theme_item_data_name,
                    theme_item.text,
                    default_value,
                )
                class_def.theme_items[theme_item_name] = theme_item_def

        tutorials = class_root.find("tutorials")
        if tutorials is not None:
            for link in tutorials:
                assert link.tag == "link"

                if link.text is not None:
                    class_def.tutorials.append((link.text.strip(), link.get("title", "")))

    def sort_classes(self):  # type: () -> None
        self.classes = OrderedDict(sorted(self.classes.items(), key=lambda t: t[0]))


def parse_arguments(root):  # type: (ET.Element) -> List[ParameterDef]
    param_elements = root.findall("argument")
    params = [None] * len(param_elements)  # type: Any
    for param_element in param_elements:
        param_name = param_element.attrib["name"]
        index = int(param_element.attrib["index"])
        type_name = TypeName.from_element(param_element)
        default = param_element.get("default")

        params[index] = ParameterDef(param_name, type_name, default)

    cast = params  # type: List[ParameterDef]

    return cast


def main():  # type: () -> None
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="+", help="A path to an XML file or a directory containing XML files to parse.")
    parser.add_argument("--filter", default="", help="The filepath pattern for XML files to filter.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", "-o", default=".", help="The directory to save output .rst files in.")
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="If passed, no output will be generated and XML files are only checked for errors.",
    )
    args = parser.parse_args()

    print("Checking for errors in the XML class reference...")

    file_list = []  # type: List[str]

    for path in args.path:
        # Cut off trailing slashes so os.path.basename doesn't choke.
        if path.endswith(os.sep):
            path = path[:-1]

        if os.path.basename(path) == "modules":
            for subdir, dirs, _ in os.walk(path):
                if "doc_classes" in dirs:
                    doc_dir = os.path.join(subdir, "doc_classes")
                    class_file_names = (f for f in os.listdir(doc_dir) if f.endswith(".xml"))
                    file_list += (os.path.join(doc_dir, f) for f in class_file_names)

        elif os.path.isdir(path):
            file_list += (os.path.join(path, f) for f in os.listdir(path) if f.endswith(".xml"))

        elif os.path.isfile(path):
            if not path.endswith(".xml"):
                print("Got non-.xml file '{}' in input, skipping.".format(path))
                continue

            file_list.append(path)

    classes = {}  # type: Dict[str, ET.Element]
    state = State()

    for cur_file in file_list:
        try:
            tree = ET.parse(cur_file)
        except ET.ParseError as e:
            print_error("Parse error reading file '{}': {}".format(cur_file, e), state)
            continue
        doc = tree.getroot()

        if "version" not in doc.attrib:
            print_error("Version missing from 'doc', file: {}".format(cur_file), state)
            continue

        name = doc.attrib["name"]
        if name in classes:
            print_error("Duplicate class '{}'".format(name), state)
            continue

        classes[name] = (doc, cur_file)

    for name, data in classes.items():
        try:
            state.parse_class(data[0], data[1])
        except Exception as e:
            print_error("Exception while parsing class '{}': {}".format(name, e), state)

    state.sort_classes()

    pattern = re.compile(args.filter)

    # Create the output folder recursively if it doesn't already exist.
    os.makedirs(args.output, exist_ok=True)

    for class_name, class_def in state.classes.items():
        if args.filter and not pattern.search(class_def.filepath):
            continue
        state.current_class = class_name
        make_rst_class(class_def, state, args.dry_run, args.output)

    if not state.errored:
        print("No errors found.")
        if not args.dry_run:
            print("Wrote reStructuredText files for each class to: %s" % args.output)
    else:
        print("Errors were found in the class reference XML. Please check the messages above.")
        exit(1)


def make_rst_class(class_def, state, dry_run, output_dir):  # type: (ClassDef, State, bool, str) -> None
    class_name = class_def.name

    if dry_run:
        f = open(os.devnull, "w", encoding="utf-8")
    else:
        f = open(os.path.join(output_dir, "class_" + class_name.lower() + ".rst"), "w", encoding="utf-8")

    # Warn contributors not to edit this file directly
    f.write(":github_url: hide\n\n")
    f.write(".. Generated automatically by doc/tools/make_rst.py in Godot's source tree.\n")
    f.write(".. DO NOT EDIT THIS FILE, but the " + class_name + ".xml source instead.\n")
    f.write(".. The source is found in doc/classes or modules/<name>/doc_classes.\n\n")

    f.write(".. _class_" + class_name + ":\n\n")
    f.write(make_heading(class_name, "="))

    # Inheritance tree
    # Ascendants
    if class_def.inherits:
        inh = class_def.inherits.strip()
        f.write("**Inherits:** ")
        first = True
        while inh in state.classes:
            if not first:
                f.write(" **<** ")
            else:
                first = False

            f.write(make_type(inh, state))
            inode = state.classes[inh].inherits
            if inode:
                inh = inode.strip()
            else:
                break
        f.write("\n\n")

    # Descendents
    inherited = []
    for c in state.classes.values():
        if c.inherits and c.inherits.strip() == class_name:
            inherited.append(c.name)

    if len(inherited):
        f.write("**Inherited By:** ")
        for i, child in enumerate(inherited):
            if i > 0:
                f.write(", ")
            f.write(make_type(child, state))
        f.write("\n\n")

    # Brief description
    if class_def.brief_description is not None:
        f.write(rstize_text(class_def.brief_description.strip(), state) + "\n\n")

    # Class description
    if class_def.description is not None and class_def.description.strip() != "":
        f.write(make_heading("Description", "-"))
        f.write(rstize_text(class_def.description.strip(), state) + "\n\n")

    # Online tutorials
    if len(class_def.tutorials) > 0:
        f.write(make_heading("Tutorials", "-"))
        for url, title in class_def.tutorials:
            f.write("- " + make_link(url, title) + "\n\n")

    # Properties overview
    if len(class_def.properties) > 0:
        f.write(make_heading("Properties", "-"))
        ml = []  # type: List[Tuple[str, str, str]]
        for property_def in class_def.properties.values():
            type_rst = property_def.type_name.to_rst(state)
            default = property_def.default_value
            if default is not None and property_def.overrides:
                ref = ":ref:`{1}<class_{1}_property_{0}>`".format(property_def.name, property_def.overrides)
                ml.append((type_rst, property_def.name, default + " (overrides " + ref + ")"))
            else:
                ref = ":ref:`{0}<class_{1}_property_{0}>`".format(property_def.name, class_name)
                ml.append((type_rst, ref, default))
        format_table(f, ml, True)

    # Constructors, Methods, Operators overview
    if len(class_def.constructors) > 0:
        f.write(make_heading("Constructors", "-"))
        ml = []
        for method_list in class_def.constructors.values():
            for m in method_list:
                ml.append(make_method_signature(class_def, m, "constructor", state))
        format_table(f, ml)

    if len(class_def.methods) > 0:
        f.write(make_heading("Methods", "-"))
        ml = []
        for method_list in class_def.methods.values():
            for m in method_list:
                ml.append(make_method_signature(class_def, m, "method", state))
        format_table(f, ml)

    if len(class_def.operators) > 0:
        f.write(make_heading("Operators", "-"))
        ml = []
        for method_list in class_def.operators.values():
            for m in method_list:
                ml.append(make_method_signature(class_def, m, "operator", state))
        format_table(f, ml)

    # Theme properties
    if len(class_def.theme_items) > 0:
        f.write(make_heading("Theme Properties", "-"))
        pl = []
        for theme_item_def in class_def.theme_items.values():
            ref = ":ref:`{0}<class_{2}_theme_{1}_{0}>`".format(
                theme_item_def.name, theme_item_def.data_name, class_name
            )
            pl.append((theme_item_def.type_name.to_rst(state), ref, theme_item_def.default_value))
        format_table(f, pl, True)

    # Signals
    if len(class_def.signals) > 0:
        f.write(make_heading("Signals", "-"))
        index = 0

        for signal in class_def.signals.values():
            if index != 0:
                f.write("----\n\n")

            f.write(".. _class_{}_signal_{}:\n\n".format(class_name, signal.name))
            _, signature = make_method_signature(class_def, signal, "", state)
            f.write("- {}\n\n".format(signature))

            if signal.description is not None and signal.description.strip() != "":
                f.write(rstize_text(signal.description.strip(), state) + "\n\n")

            index += 1

    # Enums
    if len(class_def.enums) > 0:
        f.write(make_heading("Enumerations", "-"))
        index = 0

        for e in class_def.enums.values():
            if index != 0:
                f.write("----\n\n")

            f.write(".. _enum_{}_{}:\n\n".format(class_name, e.name))
            # Sphinx seems to divide the bullet list into individual <ul> tags if we weave the labels into it.
            # As such I'll put them all above the list. Won't be perfect but better than making the list visually broken.
            # As to why I'm not modifying the reference parser to directly link to the _enum label:
            # If somebody gets annoyed enough to fix it, all existing references will magically improve.
            for value in e.values.values():
                f.write(".. _class_{}_constant_{}:\n\n".format(class_name, value.name))

            f.write("enum **{}**:\n\n".format(e.name))
            for value in e.values.values():
                f.write("- **{}** = **{}**".format(value.name, value.value))
                if value.text is not None and value.text.strip() != "":
                    f.write(" --- " + rstize_text(value.text.strip(), state))

                f.write("\n\n")

            index += 1

    # Constants
    if len(class_def.constants) > 0:
        f.write(make_heading("Constants", "-"))
        # Sphinx seems to divide the bullet list into individual <ul> tags if we weave the labels into it.
        # As such I'll put them all above the list. Won't be perfect but better than making the list visually broken.
        for constant in class_def.constants.values():
            f.write(".. _class_{}_constant_{}:\n\n".format(class_name, constant.name))

        for constant in class_def.constants.values():
            f.write("- **{}** = **{}**".format(constant.name, constant.value))
            if constant.text is not None and constant.text.strip() != "":
                f.write(" --- " + rstize_text(constant.text.strip(), state))

            f.write("\n\n")

    # Property descriptions
    if any(not p.overrides for p in class_def.properties.values()) > 0:
        f.write(make_heading("Property Descriptions", "-"))
        index = 0

        for property_def in class_def.properties.values():
            if property_def.overrides:
                continue

            if index != 0:
                f.write("----\n\n")

            f.write(".. _class_{}_property_{}:\n\n".format(class_name, property_def.name))
            f.write("- {} **{}**\n\n".format(property_def.type_name.to_rst(state), property_def.name))

            info = []
            if property_def.default_value is not None:
                info.append(("*Default*", property_def.default_value))
            if property_def.setter is not None and not property_def.setter.startswith("_"):
                info.append(("*Setter*", property_def.setter + "(value)"))
            if property_def.getter is not None and not property_def.getter.startswith("_"):
                info.append(("*Getter*", property_def.getter + "()"))

            if len(info) > 0:
                format_table(f, info)

            if property_def.text is not None and property_def.text.strip() != "":
                f.write(rstize_text(property_def.text.strip(), state) + "\n\n")

            index += 1

    # Constructor, Method, Operator descriptions
    if len(class_def.constructors) > 0:
        f.write(make_heading("Constructor Descriptions", "-"))
        index = 0

        for method_list in class_def.constructors.values():
            for i, m in enumerate(method_list):
                if index != 0:
                    f.write("----\n\n")

                if i == 0:
                    f.write(".. _class_{}_constructor_{}:\n\n".format(class_name, m.name))

                ret_type, signature = make_method_signature(class_def, m, "", state)
                f.write("- {} {}\n\n".format(ret_type, signature))

                if m.description is not None and m.description.strip() != "":
                    f.write(rstize_text(m.description.strip(), state) + "\n\n")

                index += 1

    if len(class_def.methods) > 0:
        f.write(make_heading("Method Descriptions", "-"))
        index = 0

        for method_list in class_def.methods.values():
            for i, m in enumerate(method_list):
                if index != 0:
                    f.write("----\n\n")

                if i == 0:
                    f.write(".. _class_{}_method_{}:\n\n".format(class_name, m.name))

                ret_type, signature = make_method_signature(class_def, m, "", state)
                f.write("- {} {}\n\n".format(ret_type, signature))

                if m.description is not None and m.description.strip() != "":
                    f.write(rstize_text(m.description.strip(), state) + "\n\n")

                index += 1

    if len(class_def.operators) > 0:
        f.write(make_heading("Operator Descriptions", "-"))
        index = 0

        for method_list in class_def.operators.values():
            for i, m in enumerate(method_list):
                if index != 0:
                    f.write("----\n\n")

                if i == 0:
                    f.write(
                        ".. _class_{}_operator_{}_{}:\n\n".format(
                            class_name, sanitize_operator_name(m.name, state), m.return_type.type_name
                        )
                    )

                ret_type, signature = make_method_signature(class_def, m, "", state)
                f.write("- {} {}\n\n".format(ret_type, signature))

                if m.description is not None and m.description.strip() != "":
                    f.write(rstize_text(m.description.strip(), state) + "\n\n")

                index += 1

    # Theme property descriptions
    if len(class_def.theme_items) > 0:
        f.write(make_heading("Theme Property Descriptions", "-"))
        index = 0

        for theme_item_def in class_def.theme_items.values():
            if index != 0:
                f.write("----\n\n")

            f.write(".. _class_{}_theme_{}_{}:\n\n".format(class_name, theme_item_def.data_name, theme_item_def.name))
            f.write("- {} **{}**\n\n".format(theme_item_def.type_name.to_rst(state), theme_item_def.name))

            info = []
            if theme_item_def.default_value is not None:
                info.append(("*Default*", theme_item_def.default_value))

            if len(info) > 0:
                format_table(f, info)

            if theme_item_def.text is not None and theme_item_def.text.strip() != "":
                f.write(rstize_text(theme_item_def.text.strip(), state) + "\n\n")

            index += 1

    f.write(make_footer())


def escape_rst(text, until_pos=-1):  # type: (str) -> str
    # Escape \ character, otherwise it ends up as an escape character in rst
    pos = 0
    while True:
        pos = text.find("\\", pos, until_pos)
        if pos == -1:
            break
        text = text[:pos] + "\\\\" + text[pos + 1 :]
        pos += 2

    # Escape * character to avoid interpreting it as emphasis
    pos = 0
    while True:
        pos = text.find("*", pos, until_pos)
        if pos == -1:
            break
        text = text[:pos] + "\*" + text[pos + 1 :]
        pos += 2

    # Escape _ character at the end of a word to avoid interpreting it as an inline hyperlink
    pos = 0
    while True:
        pos = text.find("_", pos, until_pos)
        if pos == -1:
            break
        if not text[pos + 1].isalnum():  # don't escape within a snake_case word
            text = text[:pos] + "\_" + text[pos + 1 :]
            pos += 2
        else:
            pos += 1

    return text


def format_codeblock(code_type, post_text, indent_level, state):  # types: str, str, int, state
    end_pos = post_text.find("[/" + code_type + "]")
    if end_pos == -1:
        print_error("[" + code_type + "] without a closing tag, file: {}".format(state.current_class), state)
        return None

    code_text = post_text[len("[" + code_type + "]") : end_pos]
    post_text = post_text[end_pos:]

    # Remove extraneous tabs
    code_pos = 0
    while True:
        code_pos = code_text.find("\n", code_pos)
        if code_pos == -1:
            break

        to_skip = 0
        while code_pos + to_skip + 1 < len(code_text) and code_text[code_pos + to_skip + 1] == "\t":
            to_skip += 1

        if to_skip > indent_level:
            print_error(
                "Four spaces should be used for indentation within ["
                + code_type
                + "], file: {}".format(state.current_class),
                state,
            )

        if len(code_text[code_pos + to_skip + 1 :]) == 0:
            code_text = code_text[:code_pos] + "\n"
            code_pos += 1
        else:
            code_text = code_text[:code_pos] + "\n    " + code_text[code_pos + to_skip + 1 :]
            code_pos += 5 - to_skip
    return ["\n[" + code_type + "]" + code_text + post_text, len("\n[" + code_type + "]" + code_text)]


def rstize_text(text, state):  # type: (str, State) -> str
    # Linebreak + tabs in the XML should become two line breaks unless in a "codeblock"
    pos = 0
    while True:
        pos = text.find("\n", pos)
        if pos == -1:
            break

        pre_text = text[:pos]
        indent_level = 0
        while text[pos + 1] == "\t":
            pos += 1
            indent_level += 1
        post_text = text[pos + 1 :]

        # Handle codeblocks
        if (
            post_text.startswith("[codeblock]")
            or post_text.startswith("[gdscript]")
            or post_text.startswith("[csharp]")
        ):
            block_type = post_text[1:].split("]")[0]
            result = format_codeblock(block_type, post_text, indent_level, state)
            if result is None:
                return ""
            text = pre_text + result[0]
            pos += result[1]

        # Handle normal text
        else:
            text = pre_text + "\n\n" + post_text
            pos += 2

    next_brac_pos = text.find("[")
    text = escape_rst(text, next_brac_pos)

    # Handle [tags]
    inside_code = False
    pos = 0
    tag_depth = 0
    previous_pos = 0
    while True:
        pos = text.find("[", pos)
        if pos == -1:
            break

        endq_pos = text.find("]", pos + 1)
        if endq_pos == -1:
            break

        pre_text = text[:pos]
        post_text = text[endq_pos + 1 :]
        tag_text = text[pos + 1 : endq_pos]

        escape_post = False

        if tag_text in state.classes:
            if tag_text == state.current_class:
                # We don't want references to the same class
                tag_text = "``{}``".format(tag_text)
            else:
                tag_text = make_type(tag_text, state)
            escape_post = True
        else:  # command
            cmd = tag_text
            space_pos = tag_text.find(" ")
            if cmd == "/codeblock" or cmd == "/gdscript" or cmd == "/csharp":
                tag_text = ""
                tag_depth -= 1
                inside_code = False
                # Strip newline if the tag was alone on one
                if pre_text[-1] == "\n":
                    pre_text = pre_text[:-1]
            elif cmd == "/code":
                tag_text = "``"
                tag_depth -= 1
                inside_code = False
                escape_post = True
            elif inside_code:
                tag_text = "[" + tag_text + "]"
            elif cmd.find("html") == 0:
                param = tag_text[space_pos + 1 :]
                tag_text = param
            elif (
                cmd.startswith("method")
                or cmd.startswith("member")
                or cmd.startswith("signal")
                or cmd.startswith("constant")
                or cmd.startswith("theme_item")
            ):
                param = tag_text[space_pos + 1 :]

                if param.find(".") != -1:
                    ss = param.split(".")
                    if len(ss) > 2:
                        print_error("Bad reference: '{}', file: {}".format(param, state.current_class), state)
                    class_param, method_param = ss

                else:
                    class_param = state.current_class
                    method_param = param

                ref_type = ""
                if class_param in state.classes:
                    class_def = state.classes[class_param]
                    if cmd.startswith("constructor"):
                        if method_param not in class_def.constructors:
                            print_error(
                                "Unresolved constructor '{}', file: {}".format(param, state.current_class), state
                            )
                        ref_type = "_constructor"
                    if cmd.startswith("method"):
                        if method_param not in class_def.methods:
                            print_error("Unresolved method '{}', file: {}".format(param, state.current_class), state)
                        ref_type = "_method"
                    if cmd.startswith("operator"):
                        if method_param not in class_def.operators:
                            print_error("Unresolved operator '{}', file: {}".format(param, state.current_class), state)
                        ref_type = "_operator"

                    elif cmd.startswith("member"):
                        if method_param not in class_def.properties:
                            print_error("Unresolved member '{}', file: {}".format(param, state.current_class), state)
                        ref_type = "_property"

                    elif cmd.startswith("theme_item"):
                        if method_param not in class_def.theme_items:
                            print_error(
                                "Unresolved theme item '{}', file: {}".format(param, state.current_class), state
                            )
                        ref_type = "_theme_{}".format(class_def.theme_items[method_param].data_name)

                    elif cmd.startswith("signal"):
                        if method_param not in class_def.signals:
                            print_error("Unresolved signal '{}', file: {}".format(param, state.current_class), state)
                        ref_type = "_signal"

                    elif cmd.startswith("constant"):
                        found = False

                        # Search in the current class
                        search_class_defs = [class_def]

                        if param.find(".") == -1:
                            # Also search in @GlobalScope as a last resort if no class was specified
                            search_class_defs.append(state.classes["@GlobalScope"])

                        for search_class_def in search_class_defs:
                            if method_param in search_class_def.constants:
                                class_param = search_class_def.name
                                found = True

                            else:
                                for enum in search_class_def.enums.values():
                                    if method_param in enum.values:
                                        class_param = search_class_def.name
                                        found = True
                                        break

                        if not found:
                            print_error("Unresolved constant '{}', file: {}".format(param, state.current_class), state)
                        ref_type = "_constant"

                else:
                    print_error(
                        "Unresolved type reference '{}' in method reference '{}', file: {}".format(
                            class_param, param, state.current_class
                        ),
                        state,
                    )

                repl_text = method_param
                if class_param != state.current_class:
                    repl_text = "{}.{}".format(class_param, method_param)
                tag_text = ":ref:`{}<class_{}{}_{}>`".format(repl_text, class_param, ref_type, method_param)
                escape_post = True
            elif cmd.find("image=") == 0:
                tag_text = ""  # '![](' + cmd[6:] + ')'
            elif cmd.find("url=") == 0:
                # URLs are handled in full here as we need to extract the optional link
                # title to use `make_link`.
                link_url = cmd[4:]
                endurl_pos = text.find("[/url]", endq_pos + 1)
                if endurl_pos == -1:
                    print_error(
                        "Tag depth mismatch for [url]: no closing [/url], file: {}".format(state.current_class), state
                    )
                    break
                link_title = text[endq_pos + 1 : endurl_pos]
                tag_text = make_link(link_url, link_title)

                pre_text = text[:pos]
                text = pre_text + tag_text + text[endurl_pos + 6 :]
                pos = len(pre_text) + len(tag_text)
                previous_pos = pos
                continue
            elif cmd == "center":
                tag_depth += 1
                tag_text = ""
            elif cmd == "/center":
                tag_depth -= 1
                tag_text = ""
            elif cmd == "codeblock":
                tag_depth += 1
                tag_text = "\n::\n"
                inside_code = True
            elif cmd == "gdscript":
                tag_depth += 1
                tag_text = "\n .. code-tab:: gdscript\n"
                inside_code = True
            elif cmd == "csharp":
                tag_depth += 1
                tag_text = "\n .. code-tab:: csharp\n"
                inside_code = True
            elif cmd == "codeblocks":
                tag_depth += 1
                tag_text = "\n.. tabs::"
            elif cmd == "/codeblocks":
                tag_depth -= 1
                tag_text = ""
            elif cmd == "br":
                # Make a new paragraph instead of a linebreak, rst is not so linebreak friendly
                tag_text = "\n\n"
                # Strip potential leading spaces
                while post_text[0] == " ":
                    post_text = post_text[1:]
            elif cmd == "i" or cmd == "/i":
                if cmd == "/i":
                    tag_depth -= 1
                else:
                    tag_depth += 1
                tag_text = "*"
            elif cmd == "b" or cmd == "/b":
                if cmd == "/b":
                    tag_depth -= 1
                else:
                    tag_depth += 1
                tag_text = "**"
            elif cmd == "u" or cmd == "/u":
                if cmd == "/u":
                    tag_depth -= 1
                else:
                    tag_depth += 1
                tag_text = ""
            elif cmd == "code":
                tag_text = "``"
                tag_depth += 1
                inside_code = True
            elif cmd == "kbd":
                tag_text = ":kbd:`"
                tag_depth += 1
            elif cmd == "/kbd":
                tag_text = "`"
                tag_depth -= 1
            elif cmd.startswith("enum "):
                tag_text = make_enum(cmd[5:], state)
                escape_post = True
            else:
                tag_text = make_type(tag_text, state)
                escape_post = True

        # Properly escape things like `[Node]s`
        if escape_post and post_text and (post_text[0].isalnum() or post_text[0] == "("):  # not punctuation, escape
            post_text = "\ " + post_text

        next_brac_pos = post_text.find("[", 0)
        iter_pos = 0
        while not inside_code:
            iter_pos = post_text.find("*", iter_pos, next_brac_pos)
            if iter_pos == -1:
                break
            post_text = post_text[:iter_pos] + "\*" + post_text[iter_pos + 1 :]
            iter_pos += 2

        iter_pos = 0
        while not inside_code:
            iter_pos = post_text.find("_", iter_pos, next_brac_pos)
            if iter_pos == -1:
                break
            if not post_text[iter_pos + 1].isalnum():  # don't escape within a snake_case word
                post_text = post_text[:iter_pos] + "\_" + post_text[iter_pos + 1 :]
                iter_pos += 2
            else:
                iter_pos += 1

        text = pre_text + tag_text + post_text
        pos = len(pre_text) + len(tag_text)
        previous_pos = pos

    if tag_depth > 0:
        print_error("Tag depth mismatch: too many/little open/close tags, file: {}".format(state.current_class), state)

    return text


def format_table(f, data, remove_empty_columns=False):  # type: (TextIO, Iterable[Tuple[str, ...]]) -> None
    if len(data) == 0:
        return

    column_sizes = [0] * len(data[0])
    for row in data:
        for i, text in enumerate(row):
            text_length = len(text or "")
            if text_length > column_sizes[i]:
                column_sizes[i] = text_length

    sep = ""
    for size in column_sizes:
        if size == 0 and remove_empty_columns:
            continue
        sep += "+" + "-" * (size + 2)
    sep += "+\n"
    f.write(sep)

    for row in data:
        row_text = "|"
        for i, text in enumerate(row):
            if column_sizes[i] == 0 and remove_empty_columns:
                continue
            row_text += " " + (text or "").ljust(column_sizes[i]) + " |"
        row_text += "\n"
        f.write(row_text)
        f.write(sep)
    f.write("\n")


def make_type(klass, state):  # type: (str, State) -> str
    if klass.find("*") != -1:  # Pointer, ignore
        return klass
    link_type = klass
    if link_type.endswith("[]"):  # Typed array, strip [] to link to contained type.
        link_type = link_type[:-2]
    if link_type in state.classes:
        return ":ref:`{}<class_{}>`".format(klass, link_type)
    print_error("Unresolved type '{}', file: {}".format(klass, state.current_class), state)
    return klass


def make_enum(t, state):  # type: (str, State) -> str
    p = t.find(".")
    if p >= 0:
        c = t[0:p]
        e = t[p + 1 :]
        # Variant enums live in GlobalScope but still use periods.
        if c == "Variant":
            c = "@GlobalScope"
            e = "Variant." + e
    else:
        c = state.current_class
        e = t
        if c in state.classes and e not in state.classes[c].enums:
            c = "@GlobalScope"

    if c in state.classes and e in state.classes[c].enums:
        return ":ref:`{0}<enum_{1}_{0}>`".format(e, c)

    # Don't fail for `Vector3.Axis`, as this enum is a special case which is expected not to be resolved.
    if "{}.{}".format(c, e) != "Vector3.Axis":
        print_error("Unresolved enum '{}', file: {}".format(t, state.current_class), state)

    return t


def make_method_signature(
    class_def, method_def, ref_type, state
):  # type: (ClassDef, Union[MethodDef, SignalDef], str, State) -> Tuple[str, str]
    ret_type = " "

    if isinstance(method_def, MethodDef):
        ret_type = method_def.return_type.to_rst(state)

    out = ""

    if ref_type != "":
        if ref_type == "operator":
            out += ":ref:`{0}<class_{1}_{2}_{3}_{4}>` ".format(
                method_def.name,
                class_def.name,
                ref_type,
                sanitize_operator_name(method_def.name, state),
                method_def.return_type.type_name,
            )
        else:
            out += ":ref:`{0}<class_{1}_{2}_{0}>` ".format(method_def.name, class_def.name, ref_type)
    else:
        out += "**{}** ".format(method_def.name)

    out += "**(**"
    for i, arg in enumerate(method_def.parameters):
        if i > 0:
            out += ", "
        else:
            out += " "

        out += "{} {}".format(arg.type_name.to_rst(state), arg.name)

        if arg.default_value is not None:
            out += "=" + arg.default_value

    if isinstance(method_def, MethodDef) and method_def.qualifiers is not None and "vararg" in method_def.qualifiers:
        if len(method_def.parameters) > 0:
            out += ", ..."
        else:
            out += " ..."

    out += " **)**"

    if isinstance(method_def, MethodDef) and method_def.qualifiers is not None:
        # Use substitutions for abbreviations. This is used to display tooltips on hover.
        # See `make_footer()` for descriptions.
        for qualifier in method_def.qualifiers.split():
            out += " |" + qualifier + "|"

    return ret_type, out


def make_heading(title, underline):  # type: (str, str) -> str
    return title + "\n" + (underline * len(title)) + "\n\n"


def make_footer():  # type: () -> str
    # Generate reusable abbreviation substitutions.
    # This way, we avoid bloating the generated rST with duplicate abbreviations.
    # fmt: off
    return (
        ".. |virtual| replace:: :abbr:`virtual (This method should typically be overridden by the user to have any effect.)`\n"
        ".. |const| replace:: :abbr:`const (This method has no side effects. It doesn't modify any of the instance's member variables.)`\n"
        ".. |vararg| replace:: :abbr:`vararg (This method accepts any number of arguments after the ones described here.)`\n"
        ".. |constructor| replace:: :abbr:`constructor (This method is used to construct a type.)`\n"
        ".. |static| replace:: :abbr:`static (This method doesn't need an instance to be called, so it can be called directly using the class name.)`\n"
        ".. |operator| replace:: :abbr:`operator (This method describes a valid operator to use with this type as left-hand operand.)`\n"
    )
    # fmt: on


def make_link(url, title):  # type: (str, str) -> str
    match = GODOT_DOCS_PATTERN.search(url)
    if match:
        groups = match.groups()
        if match.lastindex == 2:
            # Doc reference with fragment identifier: emit direct link to section with reference to page, for example:
            # `#calling-javascript-from-script in Exporting For Web`
            # Or use the title if provided.
            if title != "":
                return "`" + title + " <../" + groups[0] + ".html" + groups[1] + ">`__"
            return "`" + groups[1] + " <../" + groups[0] + ".html" + groups[1] + ">`__ in :doc:`../" + groups[0] + "`"
        elif match.lastindex == 1:
            # Doc reference, for example:
            # `Math`
            if title != "":
                return ":doc:`" + title + " <../" + groups[0] + ">`"
            return ":doc:`../" + groups[0] + "`"
    else:
        # External link, for example:
        # `http://enet.bespin.org/usergroup0.html`
        if title != "":
            return "`" + title + " <" + url + ">`__"
        return "`" + url + " <" + url + ">`__"


def sanitize_operator_name(dirty_name, state):  # type: (str, State) -> str
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
        print_error("Unsupported operator type '{}', please add the missing rule.".format(dirty_name), state)

    return clear_name


if __name__ == "__main__":
    main()
