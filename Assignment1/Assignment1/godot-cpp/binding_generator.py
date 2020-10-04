#!/usr/bin/env python

import json

# comment.

# Convenience function for using template get_node
def correct_method_name(method_list):
    for method in method_list:
        if method["name"] == "get_node":
            method["name"] = "get_node_internal"


classes = []

def generate_bindings(path, use_template_get_node):

    global classes
    classes = json.load(open(path))

    icalls = set()

    for c in classes:
        # print c['name']
        used_classes = get_used_classes(c)
        if use_template_get_node and c["name"] == "Node":
            correct_method_name(c["methods"])

        header = generate_class_header(used_classes, c, use_template_get_node)

        impl = generate_class_implementation(icalls, used_classes, c, use_template_get_node)

        header_file = open("include/gen/" + strip_name(c["name"]) + ".hpp", "w+")
        header_file.write(header)

        source_file = open("src/gen/" + strip_name(c["name"]) + ".cpp", "w+")
        source_file.write(impl)


    icall_header_file = open("include/gen/__icalls.hpp", "w+")
    icall_header_file.write(generate_icall_header(icalls))

    register_types_file = open("src/gen/__register_types.cpp", "w+")
    register_types_file.write(generate_type_registry(classes))

    init_method_bindings_file = open("src/gen/__init_method_bindings.cpp", "w+")
    init_method_bindings_file.write(generate_init_method_bindings(classes))


def is_reference_type(t):
    for c in classes:
        if c['name'] != t:
            continue
        if c['is_reference']:
            return True
    return False

def make_gdnative_type(t, ref_allowed):
    if is_enum(t):
        return remove_enum_prefix(t) + " "
    elif is_class_type(t):
        if is_reference_type(t) and ref_allowed:
            return "Ref<" + strip_name(t) + "> "
        else:
            return strip_name(t) + " *"
    else:
        if t == "int":
            return "int64_t "
        if t == "float" or t == "real":
            return "real_t "
        return strip_name(t) + " "


def generate_class_header(used_classes, c, use_template_get_node):

    source = []
    source.append("#ifndef GODOT_CPP_" + strip_name(c["name"]).upper() + "_HPP")
    source.append("#define GODOT_CPP_" + strip_name(c["name"]).upper() + "_HPP")
    source.append("")
    source.append("")

    source.append("#include <gdnative_api_struct.gen.h>")
    source.append("#include <stdint.h>")
    source.append("")


    source.append("#include <core/CoreTypes.hpp>")

    class_name = strip_name(c["name"])

    # Ref<T> is not included in object.h in Godot either,
    # so don't include it here because it's not needed
    if class_name != "Object" and class_name != "Reference":
        source.append("#include <core/Ref.hpp>")
        ref_allowed = True
    else:
        source.append("#include <core/TagDB.hpp>")
        ref_allowed = False


    included = []

    for used_class in used_classes:
        if is_enum(used_class) and is_nested_type(used_class):
            used_class_name = remove_enum_prefix(extract_nested_type(used_class))
            if used_class_name not in included:
                included.append(used_class_name)
                source.append("#include \"" + used_class_name + ".hpp\"")
        elif is_enum(used_class) and is_nested_type(used_class) and not is_nested_type(used_class, class_name):
            used_class_name = remove_enum_prefix(used_class)
            if used_class_name not in included:
                included.append(used_class_name)
                source.append("#include \"" + used_class_name + ".hpp\"")

    source.append("")

    if c["base_class"] != "":
        source.append("#include \"" + strip_name(c["base_class"]) + ".hpp\"")


    source.append("namespace godot {")
    source.append("")


    for used_type in used_classes:
        if is_enum(used_type) or is_nested_type(used_type, class_name):
            continue
        else:
            source.append("class " + strip_name(used_type) + ";")


    source.append("")

    vararg_templates = ""

    # generate the class definition here
    source.append("class " + class_name + (" : public _Wrapped" if c["base_class"] == "" else (" : public " + strip_name(c["base_class"])) ) + " {")

    if c["base_class"] == "":
        source.append("public: enum { ___CLASS_IS_SCRIPT = 0, };")
        source.append("")
        source.append("private:")

    if c["singleton"]:
        source.append("\tstatic " + class_name + " *_singleton;")
        source.append("")
        source.append("\t" + class_name + "();")
        source.append("")

    # Generate method table
    source.append("\tstruct ___method_bindings {")

    for method in c["methods"]:
        source.append("\t\tgodot_method_bind *mb_" + method["name"] + ";")

    source.append("\t};")
    source.append("\tstatic ___method_bindings ___mb;")
    source.append("\tstatic void *_detail_class_tag;")
    source.append("")
    source.append("public:")
    source.append("\tstatic void ___init_method_bindings();")

    # class id from core engine for casting
    source.append("\tinline static size_t ___get_id() { return (size_t)_detail_class_tag; }")

    source.append("")


    if c["singleton"]:
        source.append("\tstatic inline " + class_name + " *get_singleton()")
        source.append("\t{")
        source.append("\t\tif (!" + class_name + "::_singleton) {")
        source.append("\t\t\t" + class_name + "::_singleton = new " + class_name + ";")
        source.append("\t\t}")
        source.append("\t\treturn " + class_name + "::_singleton;")
        source.append("\t}")
        source.append("")

        # godot::api->godot_global_get_singleton((char *) \"" + strip_name(c["name"]) + "\");"

    # ___get_class_name
    source.append("\tstatic inline const char *___get_class_name() { return (const char *) \"" + strip_name(c["name"]) + "\"; }")

    source.append("\tstatic inline Object *___get_from_variant(Variant a) { godot_object *o = (godot_object*) a; return (o) ? (Object *) godot::nativescript_1_1_api->godot_nativescript_get_instance_binding_data(godot::_RegisterState::language_index, o) : nullptr; }")

    enum_values = []

    source.append("\n\t// enums")
    for enum in c["enums"]:
        source.append("\tenum " + strip_name(enum["name"]) + " {")
        for value in enum["values"]:
            source.append("\t\t" + remove_nested_type_prefix(value) + " = " + str(enum["values"][value]) + ",")
            enum_values.append(value)
        source.append("\t};")

    source.append("\n\t// constants")

    for name in c["constants"]:
        if name not in enum_values:
            source.append("\tconst static int " + name + " = " + str(c["constants"][name]) + ";")


    if c["instanciable"]:
        source.append("")
        source.append("")
        source.append("\tstatic " + class_name + " *_new();")

    source.append("\n\t// methods")


    if class_name == "Object":
        source.append("#ifndef GODOT_CPP_NO_OBJECT_CAST")
        source.append("\ttemplate<class T>")
        source.append("\tstatic T *cast_to(const Object *obj);")
        source.append("#endif")
        source.append("")

    for method in c["methods"]:
        method_signature = ""

        # TODO decide what to do about virtual methods
        # method_signature += "virtual " if method["is_virtual"] else ""
        method_signature += make_gdnative_type(method["return_type"], ref_allowed)
        method_name = escape_cpp(method["name"])
        method_signature +=  method_name + "("


        has_default_argument = False
        method_arguments = ""

        for i, argument in enumerate(method["arguments"]):
            method_signature += "const " + make_gdnative_type(argument["type"], ref_allowed)
            argument_name = escape_cpp(argument["name"])
            method_signature += argument_name
            method_arguments += argument_name


            # default arguments
            def escape_default_arg(_type, default_value):
                if _type == "Color":
                    return "Color(" + default_value + ")"
                if _type == "bool" or _type == "int":
                    return default_value.lower()
                if _type == "Array":
                    return "Array()"
                if _type in ["PoolVector2Array", "PoolStringArray", "PoolVector3Array", "PoolColorArray", "PoolIntArray", "PoolRealArray"]:
                    return _type + "()"
                if _type == "Vector2":
                    return "Vector2" + default_value
                if _type == "Vector3":
                    return "Vector3" + default_value
                if _type == "Transform":
                    return "Transform()"
                if _type == "Transform2D":
                    return "Transform2D()"
                if _type == "Rect2":
                    return "Rect2" + default_value
                if _type == "Variant":
                    return "Variant()" if default_value == "Null" else default_value
                if _type == "String":
                    return "\"" + default_value + "\""
                if _type == "RID":
                    return "RID()"

                if default_value == "Null" or default_value == "[Object:null]":
                    return "nullptr"

                return default_value




            if argument["has_default_value"] or has_default_argument:
                method_signature += " = " + escape_default_arg(argument["type"], argument["default_value"])
                has_default_argument = True



            if i != len(method["arguments"]) - 1:
                method_signature += ", "
                method_arguments += ","

        if method["has_varargs"]:
            if len(method["arguments"]) > 0:
                method_signature += ", "
                method_arguments += ", "
            vararg_templates += "\ttemplate <class... Args> " + method_signature + "Args... args){\n\t\treturn " + method_name + "(" + method_arguments + "Array::make(args...));\n\t}\n"""
            method_signature += "const Array& __var_args = Array()"

        method_signature += ")" + (" const" if method["is_const"] else "")


        source.append("\t" + method_signature + ";")

    source.append(vararg_templates)

    if use_template_get_node and class_name == "Node":
        # Extra definition for template get_node that calls the renamed get_node_internal; has a default template parameter for backwards compatibility.
        source.append("\ttemplate <class T = Node>")
        source.append("\tT *get_node(const NodePath path) const {")
        source.append("\t\treturn Object::cast_to<T>(get_node_internal(path));")
        source.append("\t}")

        source.append("};")
        source.append("")

        # ...And a specialized version so we don't unnecessarily cast when using the default.
        source.append("template <>")
        source.append("inline Node *Node::get_node<Node>(const NodePath path) const {")
        source.append("\treturn get_node_internal(path);")
        source.append("}")
        source.append("")
    else:
        source.append("};")
        source.append("")

    source.append("}")
    source.append("")

    source.append("#endif")


    return "\n".join(source)





def generate_class_implementation(icalls, used_classes, c, use_template_get_node):
    class_name = strip_name(c["name"])

    ref_allowed = class_name != "Object" and class_name != "Reference"

    source = []
    source.append("#include \"" + class_name + ".hpp\"")
    source.append("")
    source.append("")

    source.append("#include <core/GodotGlobal.hpp>")
    source.append("#include <core/CoreTypes.hpp>")
    source.append("#include <core/Ref.hpp>")

    source.append("#include <core/Godot.hpp>")
    source.append("")


    source.append("#include \"__icalls.hpp\"")
    source.append("")
    source.append("")

    for used_class in used_classes:
        if is_enum(used_class):
            continue
        else:
            source.append("#include \"" + strip_name(used_class) + ".hpp\"")

    source.append("")
    source.append("")

    source.append("namespace godot {")


    core_object_name = "this"


    source.append("")
    source.append("")

    if c["singleton"]:
        source.append("" + class_name + " *" + class_name + "::_singleton = NULL;")
        source.append("")
        source.append("")

        # FIXME Test if inlining has a huge impact on binary size
        source.append(class_name + "::" + class_name + "() {")
        source.append("\t_owner = godot::api->godot_global_get_singleton((char *) \"" + strip_name(c["name"]) + "\");")
        source.append("}")

        source.append("")
        source.append("")

    # Method table initialization
    source.append(class_name + "::___method_bindings " + class_name + "::___mb = {};")
    source.append("")

    source.append("void *" + class_name + "::_detail_class_tag = nullptr;")
    source.append("")

    source.append("void " + class_name + "::___init_method_bindings() {")

    for method in c["methods"]:
        source.append("\t___mb.mb_" + method["name"] + " = godot::api->godot_method_bind_get_method(\"" + c["name"] + "\", \"" + ("get_node" if use_template_get_node and method["name"] == "get_node_internal" else method["name"]) + "\");")

    source.append("\tgodot_string_name class_name;")
    source.append("\tgodot::api->godot_string_name_new_data(&class_name, \"" + c["name"] + "\");")
    source.append("\t_detail_class_tag = godot::core_1_2_api->godot_get_class_tag(&class_name);")
    source.append("\tgodot::api->godot_string_name_destroy(&class_name);")

    source.append("}")
    source.append("")

    if c["instanciable"]:
        source.append(class_name + " *" + strip_name(c["name"]) + "::_new()")
        source.append("{")
        source.append("\treturn (" + class_name + " *) godot::nativescript_1_1_api->godot_nativescript_get_instance_binding_data(godot::_RegisterState::language_index, godot::api->godot_get_class_constructor((char *)\"" + c["name"] + "\")());")
        source.append("}")

    for method in c["methods"]:
        method_signature = ""

        method_signature += make_gdnative_type(method["return_type"], ref_allowed)
        method_signature += strip_name(c["name"]) + "::" + escape_cpp(method["name"]) + "("

        for i, argument in enumerate(method["arguments"]):
            method_signature += "const " + make_gdnative_type(argument["type"], ref_allowed)
            method_signature += escape_cpp(argument["name"])

            if i != len(method["arguments"]) - 1:
                method_signature += ", "

        if method["has_varargs"]:
            if len(method["arguments"]) > 0:
                method_signature += ", "
            method_signature += "const Array& __var_args"

        method_signature += ")" + (" const" if method["is_const"] else "")

        source.append(method_signature + " {")


        if method["name"] == "free":
            # dirty hack because Object::free is marked virtual but doesn't actually exist...
            source.append("\tgodot::api->godot_object_destroy(_owner);")
            source.append("}")
            source.append("")
            continue

        return_statement = ""
        return_type_is_ref = is_reference_type(method["return_type"]) and ref_allowed

        if method["return_type"] != "void":
            if is_class_type(method["return_type"]):
                if is_enum(method["return_type"]):
                    return_statement += "return (" + remove_enum_prefix(method["return_type"]) + ") "
                elif return_type_is_ref:
                    return_statement += "return Ref<" + strip_name(method["return_type"]) + ">::__internal_constructor(";
                else:
                    return_statement += "return " + ("(" + strip_name(method["return_type"]) + " *) " if is_class_type(method["return_type"]) else "")
            else:
                return_statement += "return "

        def get_icall_type_name(name):
            if is_enum(name):
                return "int"
            if is_class_type(name):
                return "Object"
            return name



        if method["has_varargs"]:

            if len(method["arguments"]) != 0:
                source.append("\tVariant __given_args[" + str(len(method["arguments"])) + "];")

            for i, argument in enumerate(method["arguments"]):
                source.append("\tgodot::api->godot_variant_new_nil((godot_variant *) &__given_args[" + str(i) + "]);")

            source.append("")


            for i, argument in enumerate(method["arguments"]):
                source.append("\t__given_args[" + str(i) + "] = " + escape_cpp(argument["name"]) + ";")

            source.append("")

            size = ""
            if method["has_varargs"]:
                size = "(__var_args.size() + " + str(len(method["arguments"])) + ")"
            else:
                size = "(" + str(len(method["arguments"])) + ")"

            source.append("\tgodot_variant **__args = (godot_variant **) alloca(sizeof(godot_variant *) * " + size + ");")

            source.append("")

            for i, argument in enumerate(method["arguments"]):
                source.append("\t__args[" + str(i) + "] = (godot_variant *) &__given_args[" + str(i) + "];")

            source.append("")

            if method["has_varargs"]:
                source.append("\tfor (int i = 0; i < __var_args.size(); i++) {")
                source.append("\t\t__args[i + " + str(len(method["arguments"])) + "] = (godot_variant *) &((Array &) __var_args)[i];")
                source.append("\t}")

            source.append("")

            source.append("\tVariant __result;")
            source.append("\t*(godot_variant *) &__result = godot::api->godot_method_bind_call(___mb.mb_" + method["name"] + ", ((const Object *) " + core_object_name + ")->_owner, (const godot_variant **) __args, " + size + ", nullptr);")

            source.append("")

            if is_class_type(method["return_type"]):
                source.append("\tObject *obj = Object::___get_from_variant(__result);")
                source.append("\tif (obj->has_method(\"reference\"))")
                source.append("\t\tobj->callv(\"reference\", Array());")

                source.append("")


            for i, argument in enumerate(method["arguments"]):
                source.append("\tgodot::api->godot_variant_destroy((godot_variant *) &__given_args[" + str(i) + "]);")

            source.append("")

            if method["return_type"] != "void":
                cast = ""
                if is_class_type(method["return_type"]):
                    if return_type_is_ref:
                        cast += "Ref<" + strip_name(method["return_type"]) + ">::__internal_constructor(__result);"
                    else:
                        cast += "(" + strip_name(method["return_type"]) + " *) " + strip_name(method["return_type"] + "::___get_from_variant(") + "__result);"
                else:
                    cast += "__result;"
                source.append("\treturn " + cast)


        else:

            args = []
            for arg in method["arguments"]:
                args.append(get_icall_type_name(arg["type"]))

            icall_ret_type = get_icall_type_name(method["return_type"])

            icall_sig = tuple((icall_ret_type, tuple(args)))

            icalls.add(icall_sig)

            icall_name = get_icall_name(icall_sig)

            return_statement += icall_name + "(___mb.mb_" + method["name"] + ", (const Object *) " + core_object_name

            for arg in method["arguments"]:
                arg_is_ref = is_reference_type(arg["type"]) and ref_allowed
                return_statement += ", " + escape_cpp(arg["name"]) + (".ptr()" if arg_is_ref else "")

            return_statement += ")"

            if return_type_is_ref:
                return_statement += ")"

            source.append("\t" + return_statement + ";")

        source.append("}")
        source.append("")



    source.append("}")


    return "\n".join(source)





def generate_icall_header(icalls):

    source = []
    source.append("#ifndef GODOT_CPP__ICALLS_HPP")
    source.append("#define GODOT_CPP__ICALLS_HPP")

    source.append("")

    source.append("#include <gdnative_api_struct.gen.h>")
    source.append("#include <stdint.h>")
    source.append("")

    source.append("#include <core/GodotGlobal.hpp>")
    source.append("#include <core/CoreTypes.hpp>")
    source.append("#include \"Object.hpp\"")
    source.append("")
    source.append("")

    source.append("namespace godot {")
    source.append("")

    for icall in icalls:
        ret_type = icall[0]
        args = icall[1]

        method_signature = "static inline "

        method_signature += get_icall_return_type(ret_type) + get_icall_name(icall) + "(godot_method_bind *mb, const Object *inst"

        for i, arg in enumerate(args):
            method_signature += ", const "

            if is_core_type(arg):
                method_signature += arg + "&"
            elif arg == "int":
                method_signature += "int64_t "
            elif arg == "float":
                method_signature += "double "
            elif is_primitive(arg):
                method_signature += arg + " "
            else:
                method_signature += "Object *"

            method_signature += "arg" + str(i)

        method_signature += ")"

        source.append(method_signature + " {")

        if ret_type != "void":
            source.append("\t" + ("godot_object *" if is_class_type(ret_type) else get_icall_return_type(ret_type)) + "ret;")
            if is_class_type(ret_type):
                source.append("\tret = nullptr;")


        source.append("\tconst void *args[" + ("1" if len(args) == 0 else "") + "] = {")

        for i, arg in enumerate(args):

            wrapped_argument = "\t\t"
            if is_primitive(arg) or is_core_type(arg):
                wrapped_argument += "(void *) &arg" + str(i)
            else:
                wrapped_argument += "(void *) (arg" + str(i) + ") ? arg" + str(i) + "->_owner : nullptr"

            wrapped_argument += ","
            source.append(wrapped_argument)

        source.append("\t};")
        source.append("")

        source.append("\tgodot::api->godot_method_bind_ptrcall(mb, inst->_owner, args, " + ("nullptr" if ret_type == "void" else "&ret") + ");")

        if ret_type != "void":
            if is_class_type(ret_type):
                source.append("\tif (ret) {")
                source.append("\t\treturn (Object *) godot::nativescript_1_1_api->godot_nativescript_get_instance_binding_data(godot::_RegisterState::language_index, ret);")
                source.append("\t}")
                source.append("")
                source.append("\treturn (Object *) ret;")
            else:
                source.append("\treturn ret;")

        source.append("}")
        source.append("")

    source.append("}")
    source.append("")

    source.append("#endif")

    return "\n".join(source)


def generate_type_registry(classes):
    source = []

    source.append("#include \"TagDB.hpp\"")
    source.append("#include <typeinfo>")
    source.append("\n")

    for c in classes:
        source.append("#include <" + strip_name(c["name"]) + ".hpp>")

    source.append("")
    source.append("")

    source.append("namespace godot {")

    source.append("void ___register_types()")
    source.append("{")

    for c in classes:
        class_name = strip_name(c["name"])
        base_class_name = strip_name(c["base_class"])

        class_type_hash = "typeid(" + class_name + ").hash_code()"

        base_class_type_hash = "typeid(" + base_class_name + ").hash_code()"

        if base_class_name == "":
            base_class_type_hash = "0"

        source.append("\tgodot::_TagDB::register_global_type(\"" + c["name"] + "\", " + class_type_hash + ", " + base_class_type_hash + ");")

    source.append("}")

    source.append("")
    source.append("}")


    return "\n".join(source)


def generate_init_method_bindings(classes):
    source = []

    for c in classes:
        source.append("#include <" + strip_name(c["name"]) + ".hpp>")

    source.append("")
    source.append("")

    source.append("namespace godot {")

    source.append("void ___init_method_bindings()")
    source.append("{")

    for c in classes:
        class_name = strip_name(c["name"])

        source.append("\t" + strip_name(c["name"]) + "::___init_method_bindings();")

    source.append("}")

    source.append("")
    source.append("}")

    return "\n".join(source)


def get_icall_return_type(t):
    if is_class_type(t):
        return "Object *"
    if t == "int":
        return "int64_t "
    if t == "float" or t == "real":
        return "double "
    return t + " "


def get_icall_name(sig):
    ret_type = sig[0]
    args = sig[1]

    name = "___godot_icall_"
    name += strip_name(ret_type)
    for arg in args:
        name += "_" + strip_name(arg)

    return name





def get_used_classes(c):
    classes = []
    for method in c["methods"]:
        if is_class_type(method["return_type"]) and not (method["return_type"] in classes):
            classes.append(method["return_type"])

        for arg in method["arguments"]:
            if is_class_type(arg["type"]) and not (arg["type"] in classes):
                classes.append(arg["type"])
    return classes





def strip_name(name):
    if len(name) == 0:
        return name
    if name[0] == '_':
        return name[1:]
    return name

def extract_nested_type(nested_type):
    return strip_name(nested_type[:nested_type.find("::")])

def remove_nested_type_prefix(name):
    return name if name.find("::") == -1 else strip_name(name[name.find("::") + 2:])

def remove_enum_prefix(name):
    return strip_name(name[name.find("enum.") + 5:])

def is_nested_type(name, type = ""):
    return name.find(type + "::") != -1

def is_enum(name):
    return name.find("enum.") == 0

def is_class_type(name):
    return not is_core_type(name) and not is_primitive(name)

def is_core_type(name):
    core_types = ["Array",
                  "Basis",
                  "Color",
                  "Dictionary",
                  "Error",
                  "NodePath",
                  "Plane",
                  "PoolByteArray",
                  "PoolIntArray",
                  "PoolRealArray",
                  "PoolStringArray",
                  "PoolVector2Array",
                  "PoolVector3Array",
                  "PoolColorArray",
                  "PoolIntArray",
                  "PoolRealArray",
                  "Quat",
                  "Rect2",
                  "AABB",
                  "RID",
                  "String",
                  "Transform",
                  "Transform2D",
                  "Variant",
                  "Vector2",
                  "Vector3"]
    return name in core_types




def is_primitive(name):
    core_types = ["int", "bool", "real", "float", "void"]
    return name in core_types

def escape_cpp(name):
    escapes = {
        "class":    "_class",
        "char":     "_char",
        "short":    "_short",
        "bool":     "_bool",
        "int":      "_int",
        "default":  "_default",
        "case":     "_case",
        "switch":   "_switch",
        "export":   "_export",
        "template": "_template",
        "new":      "new_",
        "operator": "_operator",
        "typename": "_typename"
    }
    if name in escapes:
        return escapes[name]
    return name
