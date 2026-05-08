#utilities
import json
import re

from metadata import (
    argument_types,
    cpp_keywords,
    cpp_local_reserved,
    cs_keywords,
    cs_local_reserved,
    forced_return_type_by_method,
)


# takes a raw identiier and normalizes it. the language determines keyword rules
def sanitize_identifier(name, language="cpp"):
    text = re.sub(r"[^0-9A-Za-z_]", "_", str(name))
    if not text:
        text = "_"
    if text[0].isdigit():
        text = "_" + text
    if language == "cpp" and text in cpp_keywords:
        text += "_"
    if language == "cs" and text in cs_keywords:
        text += "_"
    return text


# ensures uniqueness within a given used set
def unique_identifier(name, used, language="cpp", prefix="arg"):
    base = sanitize_identifier(name, language)
    if language == "cpp" and base in cpp_local_reserved:
        base = f"{prefix}_{base}"
    if language == "cs" and base in cs_local_reserved:
        base = f"{prefix}_{base}"

    candidate = base
    idx = 2
    while candidate in used:
        candidate = f"{base}_{idx}"
        idx += 1
    used.add(candidate)
    return candidate


# loads data from api.json
def load_entries(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("classes", "methods", "api", "data", "entries"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError("Unsupported JSON structure: expected a list of entries or a top-level list field.")

    if not isinstance(data, list):
        raise ValueError("Unsupported JSON structure: expected a list of entries.")

    return data

#validates and standardizes the arguments
def normalize_args(arguments):
    if not isinstance(arguments, list):
        return []

    out = []
    for argument in arguments:
        if not isinstance(argument, dict):
            continue
        argument_type = argument.get("type")
        argument_name = argument.get("name")
        if argument_type not in argument_types:
            return None
        if not argument_name:
            return None
        out.append({"type": int(argument_type), "name": str(argument_name)})
    return out


# creates a unique signature tuple
def method_signature(class_name, method_name, arguments, return_type):
    return (class_name, method_name, tuple(argument["type"] for argument in arguments), return_type)


#builds command identifiers
def make_command_name(class_name, method_name, arguments, return_type):
    command_identifier = [str(argument["type"]) for argument in arguments]
    command_identifier.append(f"r{return_type}")
    suffix = "__".join(command_identifier) if command_identifier else "noargs"
    return sanitize_identifier(f"CMD_{class_name}_{method_name}__{suffix}", "cpp")


# sort classes based on inheritance depth
def order_classes(class_methods, class_parent):
    depth_cache = {}
    visiting = set()

    def depth(class_name):
        if class_name in depth_cache:
            return depth_cache[class_name]
        if class_name in visiting:
            return 0
        visiting.add(class_name)
        parent = class_parent.get(class_name, "")
        if not parent or parent == class_name:
            value = 0
        elif parent not in class_methods:
            value = 1
        else:
            value = 1 + depth(parent)
        visiting.remove(class_name)
        depth_cache[class_name] = value
        return value

    return sorted(class_methods.keys(), key=lambda class_name: (depth(class_name), class_name))


# class file name helpers to generate unique  class file names
def cpp_class_file_name(class_name, used):
    return unique_identifier(class_name, used, "cpp", prefix="Class")


def cs_class_file_name(class_name, used):
    return unique_identifier(class_name, used, "cs", prefix="Class")


# this one ensures that if a method has a forced type, we use it
def force_return_type(class_name, method_name, return_type):
    return forced_return_type_by_method.get(method_name, return_type)

# writes texts to a selected file
def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
