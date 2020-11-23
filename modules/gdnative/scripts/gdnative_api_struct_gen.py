#!/usr/bin/python3

import json
import argparse


def _spaced(e):
    return e if e[-1] == "*" else e + " "


def _build_gdnative_api_struct_header(api):
    out = [
        "/* THIS FILE IS GENERATED DO NOT EDIT */",
        "#ifndef GODOT_GDNATIVE_API_STRUCT_H",
        "#define GODOT_GDNATIVE_API_STRUCT_H",
        "",
        "#include <gdnative/gdnative.h>",
        "#include <android/godot_android.h>",
        "#include <xr/godot_xr.h>",
        "#include <nativescript/godot_nativescript.h>",
        "#include <net/godot_net.h>",
        "#include <pluginscript/godot_pluginscript.h>",
        "#include <videodecoder/godot_videodecoder.h>",
        "#include <text/godot_text.h>",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "enum GDNATIVE_API_TYPES {",
        "\tGDNATIVE_" + api["core"]["type"] + ",",
    ]

    for ext in api["extensions"]:
        out += ["\tGDNATIVE_EXT_" + ext["type"] + ","]

    out += ["};", ""]

    def generate_extension_struct(name, ext, include_version=True):
        ret_val = []
        if ext["next"]:
            ret_val += generate_extension_struct(name, ext["next"])

        ret_val += [
            "typedef struct godot_gdnative_ext_"
            + name
            + ("" if not include_version else ("_{0}_{1}".format(
                ext["version"]["major"], ext["version"]["minor"])))
            + "_api_struct {",
            "\tunsigned int type;",
            "\tgodot_gdnative_api_version version;",
            "\tconst godot_gdnative_api_struct *next;",
        ]

        for funcdef in ext["api"]:
            args = ", ".join(["%s%s" % (_spaced(t), n)
                              for t, n in funcdef["arguments"]])
            ret_val.append("\t%s(*%s)(%s);" %
                           (_spaced(funcdef["return_type"]), funcdef["name"], args))

        ret_val += [
            "} godot_gdnative_ext_"
            + name
            + ("" if not include_version else ("_{0}_{1}".format(
                ext["version"]["major"], ext["version"]["minor"])))
            + "_api_struct;",
            "",
        ]

        return ret_val

    def generate_core_extension_struct(core):
        ret_val = []
        if core["next"]:
            ret_val += generate_core_extension_struct(core["next"])

        ret_val += [
            "typedef struct godot_gdnative_core_"
            + "{0}_{1}".format(core["version"]["major"],
                               core["version"]["minor"])
            + "_api_struct {",
            "\tunsigned int type;",
            "\tgodot_gdnative_api_version version;",
            "\tconst godot_gdnative_api_struct *next;",
        ]

        for funcdef in core["api"]:
            args = ", ".join(["%s%s" % (_spaced(t), n)
                              for t, n in funcdef["arguments"]])
            ret_val.append("\t%s(*%s)(%s);" %
                           (_spaced(funcdef["return_type"]), funcdef["name"], args))

        ret_val += [
            "} godot_gdnative_core_"
            + "{0}_{1}".format(core["version"]["major"],
                               core["version"]["minor"])
            + "_api_struct;",
            "",
        ]

        return ret_val

    for ext in api["extensions"]:
        name = ext["name"]
        out += generate_extension_struct(name, ext, False)

    if api["core"]["next"]:
        out += generate_core_extension_struct(api["core"]["next"])

    out += [
        "typedef struct godot_gdnative_core_api_struct {",
        "\tunsigned int type;",
        "\tgodot_gdnative_api_version version;",
        "\tconst godot_gdnative_api_struct *next;",
        "\tunsigned int num_extensions;",
        "\tconst godot_gdnative_api_struct **extensions;",
    ]

    for funcdef in api["core"]["api"]:
        args = ", ".join(["%s%s" % (_spaced(t), n)
                          for t, n in funcdef["arguments"]])
        out.append("\t%s(*%s)(%s);" %
                   (_spaced(funcdef["return_type"]), funcdef["name"], args))

    out += [
        "} godot_gdnative_core_api_struct;",
        "",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        "#endif // GODOT_GDNATIVE_API_STRUCT_H",
        "",
    ]
    return "\n".join(out)


def _build_gdnative_api_struct_source(api):
    out = ["/* THIS FILE IS GENERATED DO NOT EDIT */",
           "", "#include <gdnative_api_struct.gen.h>", ""]

    def get_extension_struct_name(name, ext, include_version=True):
        return (
            "godot_gdnative_ext_"
            + name
            + ("" if not include_version else ("_{0}_{1}".format(
                ext["version"]["major"], ext["version"]["minor"])))
            + "_api_struct"
        )

    def get_extension_struct_instance_name(name, ext, include_version=True):
        return (
            "api_extension_"
            + name
            + ("" if not include_version else ("_{0}_{1}".format(
                ext["version"]["major"], ext["version"]["minor"])))
            + "_struct"
        )

    def get_extension_struct_definition(name, ext, include_version=True):

        ret_val = []

        if ext["next"]:
            ret_val += get_extension_struct_definition(name, ext["next"])

        ret_val += [
            "extern const "
            + get_extension_struct_name(name, ext, include_version)
            + " "
            + get_extension_struct_instance_name(name, ext, include_version)
            + " = {",
            "\tGDNATIVE_EXT_" + ext["type"] + ",",
            "\t{" + str(ext["version"]["major"]) + ", " +
            str(ext["version"]["minor"]) + "},",
            "\t"
            + (
                "nullptr"
                if not ext["next"]
                else ("(const godot_gdnative_api_struct *)&" + get_extension_struct_instance_name(name, ext["next"]))
            )
            + ",",
        ]

        for funcdef in ext["api"]:
            ret_val.append("\t%s," % funcdef["name"])

        ret_val += ["};\n"]

        return ret_val

    def get_core_struct_definition(core):
        ret_val = []

        if core["next"]:
            ret_val += get_core_struct_definition(core["next"])

        ret_val += [
            "extern const godot_gdnative_core_"
            + "{0}_{1}_api_struct api_{0}_{1}".format(
                core["version"]["major"], core["version"]["minor"])
            + " = {",
            "\tGDNATIVE_" + core["type"] + ",",
            "\t{" + str(core["version"]["major"]) + ", " +
            str(core["version"]["minor"]) + "},",
            "\t"
            + (
                "nullptr"
                if not core["next"]
                else (
                    "(const godot_gdnative_api_struct *)& api_{0}_{1}".format(
                        core["next"]["version"]["major"], core["next"]["version"]["minor"]
                    )
                )
            )
            + ",",
        ]

        for funcdef in core["api"]:
            ret_val.append("\t%s," % funcdef["name"])

        ret_val += ["};\n"]

        return ret_val

    for ext in api["extensions"]:
        name = ext["name"]
        out += get_extension_struct_definition(name, ext, False)

    out += ["",
            "const godot_gdnative_api_struct *gdnative_extensions_pointers[] = {"]

    for ext in api["extensions"]:
        name = ext["name"]
        out += ["\t(godot_gdnative_api_struct *)&api_extension_" +
                name + "_struct,"]

    out += ["};\n"]

    if api["core"]["next"]:
        out += get_core_struct_definition(api["core"]["next"])

    out += [
        "extern const godot_gdnative_core_api_struct api_struct = {",
        "\tGDNATIVE_" + api["core"]["type"] + ",",
        "\t{" + str(api["core"]["version"]["major"]) + ", " +
        str(api["core"]["version"]["minor"]) + "},",
        "\t"
        + (
            "nullptr, "
            if not api["core"]["next"]
            else (
                "(const godot_gdnative_api_struct *)& api_{0}_{1},".format(
                    api["core"]["next"]["version"]["major"], api["core"]["next"]["version"]["minor"]
                )
            )
        ),
        "\t" + str(len(api["extensions"])) + ",",
        "\tgdnative_extensions_pointers,",
    ]

    for funcdef in api["core"]["api"]:
        out.append("\t%s," % funcdef["name"])
    out.append("};\n")

    return "\n".join(out)


def build_gdnative_api_struct_header(input_json_file: str, output_struct_header: str):

    api: dict() = dict()
    with open(input_json_file, "r") as fd:
        api = json.load(fd)

    with open(output_struct_header, "w") as fd:
        fd.write(_build_gdnative_api_struct_header(api))


def build_gdnative_api_struct_source(input_json_file: str, output_struct_source: str):
    api: dict() = dict()
    with open(input_json_file, "r") as fd:
        api = json.load(fd)

    with open(output_struct_source, "w") as fd:
        fd.write(_build_gdnative_api_struct_source(api))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate GDNative API Structs')
    parser.add_argument('api_json', type=str,
                        help='The GDNative JSON API file')

    subparsers = parser.add_subparsers(title='actions', dest='action')

    parser_header = subparsers.add_parser(
        'gen_header', description='Generate the API Struct Header')
    parser_header.add_argument('out_gen_h', type=str,
                               help='The generated API Struct Header')

    parser_source = subparsers.add_parser(
        'gen_source', description='Generate the API Struct Source')
    parser_source.add_argument('out_gen_cpp', type=str,
                               help='The generated API Struct Source')

    args = parser.parse_args()

    if args.action == 'gen_header':
        build_gdnative_api_struct_header(args.api_json, args.out_gen_h)
    else:
        build_gdnative_api_struct_source(args.api_json, args.out_gen_cpp)
