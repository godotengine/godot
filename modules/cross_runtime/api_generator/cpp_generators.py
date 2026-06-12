# cpp_generators.py

import shutil

from utilities import (
    argument_types,
    unique_identifier,
    write_text,
)


# this one is specifically to take note of unnamed arguments which introduced duplicates that are not able to build. The problem specifically exists in Csharp, since C# side was corrected, we have to correct this one so that it matches the csharp one for correct calls
def _has_unnamed_args(method_arguments):
    for argument in method_arguments or []:
        if str(argument.get("name", "")).startswith("_unnamed_arg"):
            return True
    return False


# produces bridge_api.h
def generate_bridge_api_header(folder_path, command_names):
    lines = []
    lines.append("#pragma once")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("// Command bridge ")
    lines.append("static volatile uint32_t *const CMD_OFFSET = reinterpret_cast<volatile uint32_t *>(0x5000u);")
    lines.append("static volatile uint8_t *const STATUS_OFFSET = reinterpret_cast<volatile uint8_t *>(0x5004u);")
    lines.append("static const uint8_t *const CMD_DATA = reinterpret_cast<const uint8_t *>(0x5016u);")
    lines.append("")
    lines.append("const std::uint32_t CMD_NONE = 0;")
    lines.append("const std::uint8_t STATUS_PENDING = 0;")
    lines.append("const std::uint8_t STATUS_DONE = 1;")
    lines.append("")

    max_cmd_id = 0
    for d in command_names:
        if d["const_name"].startswith("CMD_Engine_get_singleton"):
            continue
        # there were some duplicate commands with unnamed arguments, this specifically avoids them as the correct commands are typically present
        if _has_unnamed_args(d.get("method_arguments", [])):
            continue
        if d["cmd_id"] > max_cmd_id:
            max_cmd_id = d["cmd_id"]

    for d in command_names:
        if d["const_name"].startswith("CMD_Engine_get_singleton"):
            continue
        if _has_unnamed_args(d.get("method_arguments", [])):
            continue
        lines.append(f"const std::uint32_t {d['const_name']} = {d['cmd_id']};")
    # this one is specifically added as a custom
    lines.append(f"const std::uint32_t CMD_Engine_get_singleton__r0 = {max_cmd_id + 1};")

    write_text(folder_path / "cpp" / "headers" / "bridge_api.h", "\n".join(lines) + "\n")


# generate CPP class files
def generate_cpp_class_file(folder_path, file_base, class_, methods):
    path = folder_path / "cpp" / f"{file_base}.cpp"

    # defines the strieds for the packed types, its noot a universal truth but it does help estimate how far the next argument should start
    def _packed_stride(argument_type):
        if argument_type == 29:  # PackedByteArray
            return 1
        if argument_type == 30:  # PackedInt32Array
            return 4
        if argument_type == 31:  # PackedInt64Array
            return 8
        if argument_type == 32:  # PackedFloat32Array
            return 4
        if argument_type == 33:  # PackedFloat64Array
            return 8
        if argument_type == 34:  # PackedStringArray
            return 1028
        if argument_type == 35:  # PackedVector2Array
            return 8
        if argument_type == 36:  # PackedVector3Array
            return 12
        if argument_type == 37:  # PackedColorArray
            return 16
        if argument_type == 38:  # PackedVector4Array
            return 16
        return 0

    # computes where each next argument begins
    def _format_offset(constant: int, var_terms: list[str]) -> str:
        if not var_terms:
            return str(constant)
        if constant == 0:
            return " + ".join(var_terms)
        return f"{constant} + " + " + ".join(var_terms)

    with open(path, "w", encoding="utf-8") as cpp:
        cpp.write('#include "headers/bridge_helpers.h"\n')

        if file_base == "Engine":
            cpp.write('#include "scene/main/scene_tree.h"\n')
        cpp.write("")
        cpp.write(
            f"void handle_{file_base}(uint32_t cmd,  volatile uint32_t *cmd_ptr, volatile uint8_t *status_ptr) {{\n"
        )
        cpp.write("    switch (cmd) {\n")

        if file_base == "Engine":
            cpp.write("        case CMD_Engine_get_singleton__r0: {\n")
            cpp.write("            SceneTree *tree = SceneTree::get_singleton();\n")
            cpp.write("            ObjectID id = tree ? tree->get_instance_id() : ObjectID();\n")
            cpp.write("            write_object_id(CMD_DATA, id);\n")
            cpp.write("            update_status(STATUS_OFFSET, 1);\n")
            cpp.write("            *cmd_ptr = CMD_NONE;\n")
            cpp.write("        } break;\n")

        for method_name, method_arguments, method_return_type, constant_name, cmd_id in methods:
            if constant_name.startswith("CMD_Engine_get_singleton"):
                continue
            if _has_unnamed_args(method_arguments):
                continue

            cpp.write(f"        case {constant_name}: {{\n")
            cpp.write("            ObjectID target_id = read_object_id(CMD_DATA);\n")
            cpp.write("            Object *target_obj = ObjectDB::get_instance(target_id);\n")
            cpp.write("            if (!target_obj) {\n")
            cpp.write("                *status_ptr = STATUS_DONE;\n")
            cpp.write("                *cmd_ptr = CMD_NONE;\n")
            cpp.write("                break;\n")
            cpp.write("            }\n\n")

            used_cpp_names = {
                "target_id",
                "target_obj",
                "cmd_ptr",
                "status_ptr",
                "ret_value",
                "bind",
                "error",
                "args",
                "argptrs",
            }

            arg_names = []
            constant_offset = 8
            var_terms = []

            for i, arg in enumerate(method_arguments):
                argument_type = arg["type"]
                spec = argument_types[argument_type]
                argument_name = unique_identifier(arg["name"], used_cpp_names, "cpp", prefix="arg")
                used_cpp_names.add(argument_name)

                current_offset = _format_offset(constant_offset, var_terms)

                cpp.write("            " + spec["cpp_read"].format(name=argument_name, offset=current_offset) + "\n")
                arg_names.append(argument_name)

                # Only advance offset if there is a next argument
                if i == len(method_arguments) - 1:
                    break  # no need to compute offset for the non-existent next arg

                # Fixed offsets for the ones with variable size 'False'
                if not spec.get("variable_size", False):
                    constant_offset += spec["size"]
                    continue

                # variable‑size, not last argument → may need to emit size variable
                if 29 <= argument_type <= 38:  # packed arrays
                    stride = _packed_stride(argument_type)
                    constant_offset += 4
                    if stride == 1:
                        var_terms.append(f"{argument_name}.size()")
                    else:
                        var_terms.append(f"({argument_name}.size() * {stride})")
                else:
                    # other variable‑size (Variant, Dictionary, etc.)
                    size_name = unique_identifier(f"{argument_name}_size", used_cpp_names, "cpp", prefix="size")
                    used_cpp_names.add(size_name)

                    cpp.write(f"            const uint32_t {size_name} = read_int32(CMD_DATA + {current_offset});\n")
                    constant_offset += 4
                    var_terms.append(size_name)

            cpp.write(f'            MethodBind *bind = ClassDB::get_method("{class_}", "{method_name}");\n')
            cpp.write("            if (bind) {\n")

            if arg_names:
                cpp.write(f"                Variant args[{len(arg_names)}];\n")
                for i, argument_name in enumerate(arg_names):
                    cpp.write(f"                args[{i}] = {argument_name};\n")
                cpp.write(f"                const Variant *argptrs[{len(arg_names)}];\n")
                for i in range(len(arg_names)):
                    cpp.write(f"                argptrs[{i}] = &args[{i}];\n")
            else:
                cpp.write("                const Variant **argptrs = nullptr;\n")

            cpp.write("                Callable::CallError error;\n")
            if method_return_type == 0:
                cpp.write(f"                bind->call(target_obj, argptrs, {len(arg_names)}, error);\n")
            else:
                cpp.write(
                    f"                Variant ret_value = bind->call(target_obj, argptrs, {len(arg_names)}, error);\n"
                )
                if method_return_type == 1:
                    cpp.write("                write_int32(CMD_DATA, ret_value.operator bool() ? 1 : 0);\n")
                elif method_return_type == 2:
                    cpp.write("                write_int64(CMD_DATA, ret_value.operator int64_t());\n")
                elif method_return_type == 3:
                    cpp.write("                write_double(CMD_DATA, ret_value.operator double());\n")
                elif method_return_type == 5:
                    cpp.write("                Vector2 v2 = ret_value.operator Vector2();\n")
                    cpp.write("                write_float(CMD_DATA, v2.x);\n")
                    cpp.write("                write_float(CMD_DATA + 4, v2.y);\n")
                elif method_return_type == 24:
                    cpp.write("                write_object_id(CMD_DATA, ret_value.operator ObjectID());\n")

            cpp.write("            }\n")
            cpp.write("            *status_ptr = STATUS_DONE;\n")
            cpp.write("            *cmd_ptr = CMD_NONE;\n")
            cpp.write("        } break;\n\n")

        cpp.write("        default: {\n")
        cpp.write("            *status_ptr = STATUS_DONE;\n")
        cpp.write("            *cmd_ptr = CMD_NONE;\n")
        cpp.write("        } break;\n")
        cpp.write("    }\n")
        cpp.write("}\n")


def generate_command_dispatcher(folder_path, ordered_classes, cpp_class_map, command_defs):
    lines = []
    lines.append('#include "cpp/headers/bridge_helpers.h"')
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("using CommandHandler = void (*)(uint32_t, volatile uint32_t *, volatile uint8_t *);")
    lines.append("")

    for class_ in ordered_classes:
        handler_name = f"handle_{cpp_class_map[class_]}"
        lines.append(f"void {handler_name}(uint32_t cmd, volatile uint32_t *cmd_ptr, volatile uint8_t *status_ptr);")
    lines.append("")

    handler_by_id = {
        d["cmd_id"]: f"handle_{cpp_class_map[d['class']]}"
        for d in command_defs
        if not d["const_name"].startswith("CMD_Engine_get_singleton")
        if not _has_unnamed_args(d.get("method_arguments", []))  # skips the unnamed args instances
    }

    engine_cmd_id = max(handler_by_id.keys(), default=0) + 1
    handler_by_id[engine_cmd_id] = "handle_Engine"

    max_cmd = max(handler_by_id.keys(), default=1)

    lines.append("static const CommandHandler command_handlers[] = {")
    for idx in range(max_cmd + 1):
        handler = handler_by_id.get(idx, "nullptr")
        lines.append(f"    {handler},")
    lines.append("};")
    lines.append("")

    lines.append("void process_api_commands() {")
    lines.append("    volatile uint32_t *cmd_ptr = reinterpret_cast<volatile uint32_t *>(CMD_OFFSET);")
    lines.append("    volatile uint8_t *status_ptr = reinterpret_cast<volatile uint8_t *>(STATUS_OFFSET);")
    lines.append("")
    lines.append("    if (*status_ptr != STATUS_PENDING || *cmd_ptr == CMD_NONE) {")
    lines.append("        return;")
    lines.append("    }")
    lines.append("")
    lines.append("    uint32_t cmd = *cmd_ptr;")
    lines.append(
        "    const uint32_t handler_count = static_cast<uint32_t>(sizeof(command_handlers) / sizeof(command_handlers[0]));"
    )
    lines.append("    if (cmd >= handler_count || command_handlers[cmd] == nullptr) {")
    lines.append("        update_status(STATUS_OFFSET, STATUS_DONE);")
    lines.append("        *cmd_ptr = CMD_NONE;")
    lines.append("        return;")
    lines.append("    }")
    lines.append("")
    lines.append("    command_handlers[cmd](cmd, cmd_ptr, status_ptr);")
    lines.append("}")
    write_text(folder_path / "command_dispatcher.cpp", "\n".join(lines) + "\n")


def generate_cpp_helpers_copy(folder_path, script_path):
    src = folder_path / "bridge_helpers.h"
    destination = script_path / "cpp" / "headers" / "bridge_helpers.h"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, destination)
    else:
        write_text(destination, "#pragma once\n\n// bridge_helpers.h is expected to be supplied by the project.\n")
