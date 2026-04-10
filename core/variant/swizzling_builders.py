"""Functions used to generate swizzling member bindings during build time"""

import methods


def swizzling_setget_builder(target, source, env):
    with methods.generated_wrapper(str(target[0])) as file:
        file.write("#define swizzled_set(...) swizzled_set<__VA_ARGS__>\n")
        file.write("#define swizzled_get(...) swizzled_get<__VA_ARGS__>\n")
        file.write("#define SETGET_SWIZZLING_STRUCTS() \\\n")
        write_setget_template("Vector2", ["x", "y", "o"], file)
        write_setget_template("Vector3", ["x", "y", "z", "o"], file)
        write_setget_template("Vector4", ["x", "y", "z", "w", "o"], file)
        write_setget_template("Vector2i", ["x", "y", "o"], file)
        write_setget_template("Vector3i", ["x", "y", "z", "o"], file)
        write_setget_template("Vector4i", ["x", "y", "z", "w", "o"], file)
        write_setget_template("Color", ["r", "g", "b", "a", "o"], file)
        file.write("// end\n\n")
        file.write("#define SETGET_SWIZZLING_MEMBERS() \\\n")
        write_members_template("Vector2", ["x", "y", "o"], file)
        write_members_template("Vector3", ["x", "y", "z", "o"], file)
        write_members_template("Vector4", ["x", "y", "z", "w", "o"], file)
        write_members_template("Vector2i", ["x", "y", "o"], file)
        write_members_template("Vector3i", ["x", "y", "z", "o"], file)
        write_members_template("Vector4i", ["x", "y", "z", "w", "o"], file)
        write_members_template("Color", ["r", "g", "b", "a", "o"], file)
        file.write("\t// end\n\n")


_COMP_NAMES = {
    "x": 0,
    "y": 1,
    "z": 2,
    "w": 3,
    "r": 0,
    "g": 1,
    "b": 2,
    "a": 3,
    "o": 10,
}


def write_setget_template(type: str, valid_components: list[str], out_file):
    other_types = ["Vector2", "Vector3", "Vector4"]
    if type.endswith("i"):
        other_types = ["Vector2i", "Vector3i", "Vector4i"]
    valid = list(filter(lambda comp: comp in valid_components, _COMP_NAMES))
    for c1 in valid:
        for c2 in valid:
            if not (c1 == "o" and c2 == "o"):
                out_file.write(
                    f"SETGET_STRUCT_FUNC_VAR_ARGS({type}, {other_types[0]}, {c1}{c2}, swizzled_set({other_types[0]}), swizzled_get({other_types[0]}),{_COMP_NAMES[c1]}, {_COMP_NAMES[c2]}) \\\n"
                )
            for c3 in valid:
                if not (c1 == "o" and c2 == "o" and c3 == "o"):
                    out_file.write(
                        f"SETGET_STRUCT_FUNC_VAR_ARGS({type}, {other_types[1]}, {c1}{c2}{c3}, swizzled_set({other_types[1]}), swizzled_get({other_types[1]}),{_COMP_NAMES[c1]}, {_COMP_NAMES[c2]},{_COMP_NAMES[c3]}) \\\n"
                    )
                for c4 in valid:
                    if not (c1 == "o" and c2 == "o" and c3 == "o" and c4 == "o"):
                        out_file.write(
                            f"SETGET_STRUCT_FUNC_VAR_ARGS({type}, {other_types[2]}, {c1}{c2}{c3}{c4}, swizzled_set({other_types[2]}), swizzled_get({other_types[2]}),{_COMP_NAMES[c1]}, {_COMP_NAMES[c2]},{_COMP_NAMES[c3]},{_COMP_NAMES[c4]})\\\n"
                        )


def write_members_template(type: str, valid_components: list[str], out_file):
    valid = list(filter(lambda comp: comp in valid_components, _COMP_NAMES))
    for c1 in valid:
        for c2 in valid:
            if not (c1 == "o" and c2 == "o"):
                out_file.write(f"\tREGISTER_MEMBER({type}, {c1}{c2}); \\\n")
            for c3 in valid:
                if not (c1 == "o" and c2 == "o" and c3 == "o"):
                    out_file.write(f"\tREGISTER_MEMBER({type}, {c1}{c2}{c3}); \\\n")
                for c4 in valid:
                    if not (c1 == "o" and c2 == "o" and c3 == "o" and c4 == "o"):
                        out_file.write(f"\tREGISTER_MEMBER({type}, {c1}{c2}{c3}{c4}); \\\n")
