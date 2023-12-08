#!/usr/bin/env python3

"""Functions used to generate source files during build time
All such functions are invoked in a subprocess on Windows to prevent build flakiness.
"""

import os, sys
from io import StringIO
#from platform_methods import subprocess_main

def replace_if_different(output_path_str, new_content_path_str):
    import pathlib

    output_path = pathlib.Path(output_path_str)
    new_content_path = pathlib.Path(new_content_path_str)
    if not output_path.exists():
        new_content_path.replace(output_path)
        return
    if output_path.read_bytes() == new_content_path.read_bytes():
        new_content_path.unlink()
    else:
        new_content_path.replace(output_path)


def parse_template(inherits, source, delimiter):
    script_template = {
        "inherits": inherits,
        "name": "",
        "description": "",
        "version": "",
        "script": "",
        "space-indent": "4",
    }
    meta_prefix = delimiter + " meta-"
    meta = ["name", "description", "version", "space-indent"]

    with open(source) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(meta_prefix):
                line = line[len(meta_prefix) :]
                for m in meta:
                    if line.startswith(m):
                        strip_lenght = len(m) + 1
                        script_template[m] = line[strip_lenght:].strip()
            else:
                script_template["script"] += line
        if script_template["space-indent"] != "":
            indent = " " * int(script_template["space-indent"])
            script_template["script"] = script_template["script"].replace(indent, "_TS_")
        if script_template["name"] == "":
            script_template["name"] = os.path.splitext(os.path.basename(source))[0].replace("_", " ").title()
        script_template["script"] = (
            script_template["script"].replace('"', '\\"').lstrip().replace("\n", "\\n").replace("\t", "_TS_")
        )
        return (
            '{ String("'
            + script_template["inherits"]
            + '"), String("'
            + script_template["name"]
            + '"),  String("'
            + script_template["description"]
            + '"),  String("'
            + script_template["script"]
            + '")'
            + " },\n"
        )


def make_templates(target, source, env):
    dst = target[0]
    s = StringIO()
    s.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n\n")
    s.write("#ifndef _CODE_TEMPLATES_H\n")
    s.write("#define _CODE_TEMPLATES_H\n\n")
    s.write('#include "core/object/object.h"\n')
    s.write('#include "core/object/script_language.h"\n')

    delimiter = "#"  # GDScript single line comment delimiter by default.
    if source:
        ext = os.path.splitext(source[0])[1]
        if ext == ".cs":
            delimiter = "//"

    parsed_template_string = ""
    number_of_templates = 0

    for filepath in source:
        node_name = os.path.basename(os.path.dirname(filepath))
        parsed_template = parse_template(node_name, filepath, delimiter)
        parsed_template_string += "\t" + parsed_template
        number_of_templates += 1

    s.write("\nstatic const int TEMPLATES_ARRAY_SIZE = " + str(number_of_templates) + ";\n")
    s.write("\nstatic const struct ScriptLanguage::ScriptTemplate TEMPLATES[" + str(number_of_templates) + "] = {\n")

    s.write(parsed_template_string)

    s.write("};\n")

    s.write("\n#endif\n")

    tmpfile = dst + '~'
    with open(tmpfile, "w") as f:
        f.write(s.getvalue())
    replace_if_different(dst, tmpfile)

    s.close()


if __name__ == "__main__":
    #subprocess_main(globals())
    make_templates([sys.argv[1]], sys.argv[2:], None)
