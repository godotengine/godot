import json
import sys


def parse_build_profile(profile_filepath, api):
    if profile_filepath == "":
        return {}

    with open(profile_filepath, encoding="utf-8") as profile_file:
        profile = json.load(profile_file)

    api_dict = {}
    parents = {}
    children = {}
    for engine_class in api["classes"]:
        api_dict[engine_class["name"]] = engine_class
        parent = engine_class.get("inherits", "")
        child = engine_class["name"]
        parents[child] = parent
        if parent == "":
            continue
        children[parent] = children.get(parent, [])
        children[parent].append(child)

    included = []
    front = list(profile.get("enabled_classes", []))
    if front:
        # These must always be included
        front.append("WorkerThreadPool")
        front.append("ClassDB")
        front.append("ClassDBSingleton")
        # In src/classes/low_level.cpp
        front.append("FileAccess")
        front.append("Image")
        front.append("XMLParser")
        # In include/godot_cpp/templates/thread_work_pool.hpp
        front.append("Semaphore")
    while front:
        cls = front.pop()
        if cls in included:
            continue
        included.append(cls)
        parent = parents.get(cls, "")
        if parent:
            front.append(parent)

    excluded = []
    front = list(profile.get("disabled_classes", []))
    while front:
        cls = front.pop()
        if cls in excluded:
            continue
        excluded.append(cls)
        front += children.get(cls, [])

    if included and excluded:
        print(
            "WARNING: Cannot specify both 'enabled_classes' and 'disabled_classes' in build profile. 'disabled_classes' will be ignored."
        )

    return {
        "enabled_classes": included,
        "disabled_classes": excluded,
    }


def generate_trimmed_api(source_api_filepath, profile_filepath):
    with open(source_api_filepath, encoding="utf-8") as api_file:
        api = json.load(api_file)

    if profile_filepath == "":
        return api

    build_profile = parse_build_profile(profile_filepath, api)

    engine_classes = {}
    for class_api in api["classes"]:
        engine_classes[class_api["name"]] = class_api["is_refcounted"]
    for native_struct in api["native_structures"]:
        if native_struct["name"] == "ObjectID":
            continue
        engine_classes[native_struct["name"]] = False

    classes = []
    for class_api in api["classes"]:
        if not is_class_included(class_api["name"], build_profile):
            continue
        if "methods" in class_api:
            methods = []
            for method in class_api["methods"]:
                if not is_method_included(method, build_profile, engine_classes):
                    continue
                methods.append(method)
            class_api["methods"] = methods
        classes.append(class_api)
    api["classes"] = classes

    return api


def is_class_included(class_name, build_profile):
    """
    Check if an engine class should be included.
    This removes classes according to a build profile of enabled or disabled classes.
    """
    included = build_profile.get("enabled_classes", [])
    excluded = build_profile.get("disabled_classes", [])
    if included:
        return class_name in included
    if excluded:
        return class_name not in excluded
    return True


def is_method_included(method, build_profile, engine_classes):
    """
    Check if an engine class method should be included.
    This removes methods according to a build profile of enabled or disabled classes.
    """
    included = build_profile.get("enabled_classes", [])
    excluded = build_profile.get("disabled_classes", [])
    ref_cls = set()
    rtype = get_base_type(method.get("return_value", {}).get("type", ""))
    args = [get_base_type(a["type"]) for a in method.get("arguments", [])]
    if rtype in engine_classes:
        ref_cls.add(rtype)
    elif is_enum(rtype) and get_enum_class(rtype) in engine_classes:
        ref_cls.add(get_enum_class(rtype))
    for arg in args:
        if arg in engine_classes:
            ref_cls.add(arg)
        elif is_enum(arg) and get_enum_class(arg) in engine_classes:
            ref_cls.add(get_enum_class(arg))
    for acls in ref_cls:
        if len(included) > 0 and acls not in included:
            return False
        elif len(excluded) > 0 and acls in excluded:
            return False
    return True


def is_enum(type_name):
    return type_name.startswith("enum::") or type_name.startswith("bitfield::")


def get_enum_class(enum_name: str):
    if "." in enum_name:
        if is_bitfield(enum_name):
            return enum_name.replace("bitfield::", "").split(".")[0]
        else:
            return enum_name.replace("enum::", "").split(".")[0]
    else:
        return "GlobalConstants"


def get_base_type(type_name):
    if type_name.startswith("const "):
        type_name = type_name[6:]
    if type_name.endswith("*"):
        type_name = type_name[:-1]
    if type_name.startswith("typedarray::"):
        type_name = type_name.replace("typedarray::", "")
    return type_name


def is_bitfield(type_name):
    return type_name.startswith("bitfield::")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: %s BUILD_PROFILE INPUT_JSON [OUTPUT_JSON]" % (sys.argv[0]))
        sys.exit(1)
    profile = sys.argv[1]
    infile = sys.argv[2]
    outfile = sys.argv[3] if len(sys.argv) > 3 else ""
    api = generate_trimmed_api(infile, profile)

    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(api, f)
    else:
        json.dump(api, sys.stdout)
