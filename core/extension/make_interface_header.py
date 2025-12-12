import difflib
import json
from collections import OrderedDict

import methods

BASE_TYPES = [
    "void",
    "int8_t",
    "uint8_t",
    "int16_t",
    "uint16_t",
    "int32_t",
    "uint32_t",
    "int64_t",
    "uint64_t",
    "size_t",
    "char",
    "char16_t",
    "char32_t",
    "wchar_t",
    "float",
    "double",
]


def run(target, source, env):
    filename = str(source[0])
    buffer = methods.get_buffer(filename)
    data = json.loads(buffer, object_pairs_hook=OrderedDict)
    check_formatting(buffer.decode("utf-8"), data, filename)
    check_allowed_keys(data, ["_copyright", "$schema", "format_version", "types", "interface"])

    valid_data_types = {}
    for type in BASE_TYPES:
        valid_data_types[type] = True

    with methods.generated_wrapper(str(target[0])) as file:
        file.write("""\
#ifndef __cplusplus
#include <stddef.h>
#include <stdint.h>

typedef uint32_t char32_t;
typedef uint16_t char16_t;
#else
#include <cstddef>
#include <cstdint>

extern "C" {
#endif

""")

        handles = []
        type_replacements = []
        for type in data["types"]:
            kind = type["kind"]

            check_type(kind, type, valid_data_types)
            valid_data_types[type["name"]] = type

            if "deprecated" in type:
                check_allowed_keys(type["deprecated"], ["since"], ["message", "replace_with"])
                if "replace_with" in type["deprecated"]:
                    type_replacements.append((type["name"], type["deprecated"]["replace_with"]))

            if "description" in type:
                write_doc(file, type["description"])

            if kind == "handle":
                check_allowed_keys(
                    type, ["name", "kind"], ["is_const", "is_uninitialized", "parent", "description", "deprecated"]
                )
                if "parent" in type and type["parent"] not in handles:
                    raise UnknownTypeError(type["parent"], type["name"])
                # @todo In the future, let's write these as `struct *` so the compiler can help us with type checking.
                type["type"] = "void*" if not type.get("is_const", False) else "const void*"
                write_simple_type(file, type)
                handles.append(type["name"])
            elif kind == "alias":
                check_allowed_keys(type, ["name", "kind", "type"], ["description", "deprecated"])
                write_simple_type(file, type)
            elif kind == "enum":
                check_allowed_keys(type, ["name", "kind", "values"], ["is_bitfield", "description", "deprecated"])
                write_enum_type(file, type)
            elif kind == "function":
                check_allowed_keys(type, ["name", "kind", "return_value", "arguments"], ["description", "deprecated"])
                write_function_type(file, type)
            elif kind == "struct":
                check_allowed_keys(type, ["name", "kind", "members"], ["description", "deprecated"])
                write_struct_type(file, type)
            else:
                raise Exception(f"Unknown kind of type: {kind}")

        for type_name, replace_with in type_replacements:
            if replace_with not in valid_data_types:
                raise Exception(f"Unknown type '{replace_with}' used as replacement for '{type_name}'")
            replacement = valid_data_types[replace_with]
            if isinstance(replacement, dict) and "deprecated" in replacement:
                raise Exception(
                    f"Cannot use '{replace_with}' as replacement for '{type_name}' because it's deprecated too"
                )

        interface_replacements = []
        valid_interfaces = {}
        for interface in data["interface"]:
            check_type("function", interface, valid_data_types)
            check_allowed_keys(
                interface,
                ["name", "return_value", "arguments", "since", "description"],
                ["see", "legacy_type_name", "deprecated"],
            )
            valid_interfaces[interface["name"]] = interface
            if "deprecated" in interface:
                check_allowed_keys(interface["deprecated"], ["since"], ["message", "replace_with"])
                if "replace_with" in interface["deprecated"]:
                    interface_replacements.append((interface["name"], interface["deprecated"]["replace_with"]))
            write_interface(file, interface)

        for function_name, replace_with in interface_replacements:
            if replace_with not in valid_interfaces:
                raise Exception(
                    f"Unknown interface function '{replace_with}' used as replacement for '{function_name}'"
                )
            replacement = valid_interfaces[replace_with]
            if "deprecated" in replacement:
                raise Exception(
                    f"Cannot use '{replace_with}' as replacement for '{function_name}' because it's deprecated too"
                )

        file.write("""\
#ifdef __cplusplus
}
#endif
""")


# Serialize back into JSON in order to see if the formatting remains the same.
def check_formatting(buffer, data, filename):
    buffer2 = json.dumps(data, indent=4)

    lines1 = buffer.splitlines()
    lines2 = buffer2.splitlines()

    diff = difflib.unified_diff(
        lines1,
        lines2,
        fromfile="a/" + filename,
        tofile="b/" + filename,
        lineterm="",
    )

    diff = list(diff)
    if len(diff) > 0:
        print(" *** Apply this patch to fix: ***\n")
        print("\n".join(diff))
        raise Exception(f"Formatting issues in {filename}")


def check_allowed_keys(data, required, optional=[]):
    keys = data.keys()
    allowed = required + optional
    for k in keys:
        if k not in allowed:
            raise Exception(f"Found unknown key '{k}'")
    for r in required:
        if r not in keys:
            raise Exception(f"Missing required key '{r}'")


class UnknownTypeError(Exception):
    def __init__(self, unknown, parent, item=None):
        self.unknown = unknown
        self.parent = parent
        if item:
            msg = f"Unknown type '{unknown}' for '{item}' used in '{parent}'"
        else:
            msg = f"Unknown type '{unknown}' used in '{parent}'"
        super().__init__(msg)


def base_type_name(type_name):
    if type_name.startswith("const "):
        type_name = type_name[6:]
    if type_name.endswith("*"):
        type_name = type_name[:-1]
    return type_name


def format_type_and_name(type, name=None):
    ret = type
    if ret[-1] == "*":
        ret = ret[:-1] + " *"
    if name:
        if ret[-1] == "*":
            ret = ret + name
        else:
            ret = ret + " " + name
    return ret


def check_type(kind, type, valid_data_types):
    if kind == "alias":
        if base_type_name(type["type"]) not in valid_data_types:
            raise UnknownTypeError(type["type"], type["name"])
    elif kind == "struct":
        for member in type["members"]:
            if base_type_name(member["type"]) not in valid_data_types:
                raise UnknownTypeError(member["type"], type["name"], member["name"])
    elif kind == "function":
        for arg in type["arguments"]:
            if base_type_name(arg["type"]) not in valid_data_types:
                raise UnknownTypeError(arg["type"], type["name"], arg.get("name"))
        if "return_value" in type:
            if base_type_name(type["return_value"]["type"]) not in valid_data_types:
                raise UnknownTypeError(type["return_value"]["type"], type["name"])


def write_doc(file, doc, indent=""):
    if len(doc) == 1:
        file.write(f"{indent}/* {doc[0]} */\n")
        return

    first = True
    for line in doc:
        if first:
            file.write(indent + "/*")
            first = False
        else:
            file.write(indent + " *")

        if line != "":
            file.write(" " + line)
        file.write("\n")
    file.write(indent + " */\n")


def make_deprecated_message(data):
    parts = [
        f"Deprecated in Godot {data['since']}.",
        data["message"] if "message" in data else "",
        f"Use `{data['replace_with']}` instead." if "replace_with" in data else "",
    ]
    return " ".join([x for x in parts if x.strip() != ""])


def make_deprecated_comment_for_type(type):
    if "deprecated" not in type:
        return ""
    message = make_deprecated_message(type["deprecated"])
    return f" /* {message} */"


def write_simple_type(file, type):
    file.write(f"typedef {format_type_and_name(type['type'], type['name'])};{make_deprecated_comment_for_type(type)}\n")


def write_enum_type(file, enum):
    file.write("typedef enum {\n")
    for value in enum["values"]:
        check_allowed_keys(value, ["name", "value"], ["description", "deprecated"])
        if "description" in value:
            write_doc(file, value["description"], "\t")
        file.write(f"\t{value['name']} = {value['value']},\n")
    file.write(f"}} {enum['name']};{make_deprecated_comment_for_type(enum)}\n\n")


def make_args_text(args):
    combined = []
    for arg in args:
        check_allowed_keys(arg, ["type"], ["name", "description"])
        combined.append(format_type_and_name(arg["type"], arg.get("name")))
    return ", ".join(combined)


def write_function_type(file, fn):
    args_text = make_args_text(fn["arguments"]) if ("arguments" in fn) else ""
    name_and_args = f"(*{fn['name']})({args_text})"
    file.write(
        f"typedef {format_type_and_name(fn['return_value']['type'], name_and_args)};{make_deprecated_comment_for_type(fn)}\n"
    )


def write_struct_type(file, struct):
    file.write("typedef struct {\n")
    for member in struct["members"]:
        check_allowed_keys(member, ["name", "type"], ["description"])
        if "description" in member:
            write_doc(file, member["description"], "\t")
        file.write(f"\t{format_type_and_name(member['type'], member['name'])};\n")
    file.write(f"}} {struct['name']};{make_deprecated_comment_for_type(struct)}\n\n")


def write_interface(file, interface):
    doc = [
        f"@name {interface['name']}",
        f"@since {interface['since']}",
    ]

    if "deprecated" in interface:
        doc.append(f"@deprecated {make_deprecated_message(interface['deprecated'])}")

    doc += [
        "",
        interface["description"][0],
    ]

    if len(interface["description"]) > 1:
        doc.append("")
        doc += interface["description"][1:]

    if "arguments" in interface:
        doc.append("")
        for arg in interface["arguments"]:
            if "description" not in arg:
                raise Exception(f"Interface function {interface['name']} is missing docs for {arg['name']} argument")
            arg_doc = " ".join(arg["description"])
            doc.append(f"@param {arg['name']} {arg_doc}")

    if "return_value" in interface and interface["return_value"]["type"] != "void":
        if "description" not in interface["return_value"]:
            raise Exception(f"Interface function {interface['name']} is missing docs for return value")
        ret_doc = " ".join(interface["return_value"]["description"])
        doc.append("")
        doc.append(f"@return {ret_doc}")

    if "see" in interface:
        doc.append("")
        for see in interface["see"]:
            doc.append(f"@see {see}")

    file.write("/**\n")
    for d in doc:
        if d != "":
            file.write(f" * {d}\n")
        else:
            file.write(" *\n")
    file.write(" */\n")

    fn = interface.copy()
    if "deprecated" in fn:
        del fn["deprecated"]
    fn["name"] = "GDExtensionInterface" + "".join(word.capitalize() for word in interface["name"].split("_"))
    write_function_type(file, fn)

    file.write("\n")
