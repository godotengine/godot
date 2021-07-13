#!/usr/bin/python

import sys
import os

if len(sys.argv) > 0:
    if "create_module.py" in sys.argv[0]:
        sys.argv.pop(0)
    if not os.getcwd().endswith("modules"):
        if os.path.exists("modules"):
            os.chdir("modules")
        else:
            print("Could not find modules folder.")
            exit()
    custom_module_name = sys.argv[0].replace(" ", "_")
    custom_module_name = custom_module_name.lower()

    # Make directory
    os.mkdir(custom_module_name)

    # Make register_types, config.py, SCsub, and module class
    # register_types Header
    register_types_h = open(custom_module_name + "/register_types.h", "w+")

    lines_to_write = [
        '#include "core/object/class_db.h"\n\n',
        "void register_" + custom_module_name + "_types();\n",
        "void unregister_" + custom_module_name + "_types();",
    ]

    register_types_h.writelines(lines_to_write)
    register_types_h.close()

    # register_types CPP
    register_types_cpp = open(custom_module_name + "/register_types.cpp", "w+")
    lines_to_write = [
        '#include "register_types.h"\n\n',
        "void register_" + custom_module_name + "_types() {\n}\n",
        "\nvoid unregister_" + custom_module_name + "_types() {\n}",
    ]
    register_types_cpp.writelines(lines_to_write)
    register_types_cpp.close()

    # config.py
    config_py = open(custom_module_name + "/config.py", "w+")
    lines_to_write = [
        "def can_build(env, platform):\n",
        "    return True\n\n",
        "def configure(env):\n",
        "    pass\n\n",
    ]
    config_py.writelines(lines_to_write)
    config_py.close()

    # SCsub
    scsub = open(custom_module_name + "/SCsub", "w+")
    lines_to_write = [
        "#!/usr/bin/env python\n",
        "Import('env')\n\n",
        'env.add_source_files(env.modules_sources, "*.cpp")\n',
    ]
    scsub.writelines(lines_to_write)
    scsub.close()

    # Create module class
    custom_module_header = open(custom_module_name + "/" + custom_module_name + ".h", "w+")
    custom_module_cpp = open(custom_module_name + "/" + custom_module_name + ".cpp", "w+")

    # Convert module name to CamelCase
    custom_module_name_parts = custom_module_name.split("_")
    camel_case_name = ""
    for i in custom_module_name_parts:
        camel_case_name += i.title()

    # Write lines for header file
    lines_to_write = [
        "#ifndef GODOT_" + custom_module_name.upper() + "_H\n",
        "#define GODOT_" + custom_module_name.upper() + "_H\n\n",
        '#include "core/object/ref_counted.h"\n\n',
        "class " + camel_case_name + " : public RefCounted {\n",
        "    GDCLASS(" + camel_case_name + ", RefCounted);\n\n",
        "protected:\n    static void _bind_methods();\n\n",
        "public:\n    " + camel_case_name + "();\n",
        "};\n\n#endif",
    ]
    custom_module_header.writelines(lines_to_write)
    custom_module_header.close()

    lines_to_write = [
        '#include "' + custom_module_name + '.h"\n\n',
        "void " + camel_case_name + "::_bind_methods() {\n}\n",
        camel_case_name + "::" + camel_case_name + "() {}\n",
    ]
    custom_module_cpp.writelines(lines_to_write)
    custom_module_cpp.close()

    print("Created module with name " + custom_module_name)
else:
    print("You must provide the name of the module you would like to create.")
