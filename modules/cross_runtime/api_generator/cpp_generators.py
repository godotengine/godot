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
    lines.append("const std::uint32_t CMD_OFFSET   = 0x5000u;")
    lines.append("const std::uint32_t STATUS_OFFSET = 0x5004u;")
    lines.append("const std::uint32_t CMD_DATA     = 0x5008u;")
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


# creates the api cpp files
def generate_cpp_class_file(folder_path, file_base, class_, methods):
    path = folder_path / "cpp" / f"{file_base}.cpp"
    with open(path, "w", encoding="utf-8") as cpp:
        cpp.write('#include "headers/bridge_helpers.h"\n')
        # specifically for the custom case we have created
        if file_base == "Engine":
            cpp.write('#include "scene/main/scene_tree.h"\n')

        cpp.write(
            f"void handle_{file_base}(uint32_t cmd, volatile uint8_t *payload, volatile uint32_t *cmd_ptr, volatile uint8_t *status_ptr) {{\n"
        )
        cpp.write("    switch (cmd) {\n")
        if file_base == "Engine":
            cpp.write("        case CMD_Engine_get_singleton__r0: {\n")
            cpp.write("            SceneTree *tree = SceneTree::get_singleton();\n")
            cpp.write("            ObjectID id = tree ? tree->get_instance_id() : ObjectID();\n")
            cpp.write("            write_object_id(payload, 0, id);\n")
            cpp.write("            update_status(STATUS_OFFSET, 1);\n")
            cpp.write("            *cmd_ptr = CMD_NONE;\n")
            cpp.write("        } break;\n")

        for method_name, method_arguments, method_return_type, constant_name, cmd_id in methods:
            # during the build, the engine.get_singleton methods were bringing errors so this basically skips writing their cases
            if constant_name.startswith("CMD_Engine_get_singleton"):
                continue
            if _has_unnamed_args(method_arguments):
                continue

            cpp.write(f"        case {constant_name}: {{\n")
            cpp.write("            ObjectID target_id = read_object_id(payload, 0);\n")
            cpp.write("            Object *target_obj = ObjectDB::get_instance(target_id);\n")
            cpp.write("            if (!target_obj) {\n")
            cpp.write("                *status_ptr = STATUS_DONE;\n")
            cpp.write("                *cmd_ptr = CMD_NONE;\n")
            cpp.write("                break;\n")
            cpp.write("            }\n\n")

            used_cpp_names = {
                "payload",
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
            # works with both fixed offsets and when more follow each other - we have to ensure no overwriting
            arg_offset_expr = "8"
            arg_names = []
            for arg in method_arguments:
                argument_type = arg["type"]
                argument_name = unique_identifier(arg["name"], used_cpp_names, "cpp", prefix="arg")
                spec = argument_types[argument_type]

                cpp.write(
                    "            "
                    + spec["cpp_read"].format(name=argument_name, offset=arg_offset_expr, buf="payload")
                    + "\n"
                )
                arg_names.append(argument_name)
                # the generator had an issue where it was assigning the wrong offsets due to its fixed mechanism, this case specifically ensures that no overwriting takes place
                if argument_type == 29:  # PackedByteArray
                    arg_offset_expr += f" + 4 + {argument_name}.size()"
                elif argument_type == 32:  # PackedFloat32Array
                    arg_offset_expr += f" + 4 + ({argument_name}.size() * 4)"
                else:
                    if arg_offset_expr.isdigit():
                        arg_offset_expr = str(int(arg_offset_expr) + spec["size"])
                    else:
                        arg_offset_expr += f" + {spec['size']}"

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
                # This determines the kinds of reads and writes that will go into the specific case. The functions originally exist in bridge_helpers.h
                # Its best you confirm whether this aligns with the one in cpp generator, otherwise it will introduce mismatches
                if method_return_type == 1:
                    cpp.write("                write_int32(payload, 0, ret_value.operator bool() ? 1 : 0);\n")
                elif method_return_type == 2:
                    cpp.write("                write_int64(payload, 0, ret_value.operator int64_t());\n")
                elif method_return_type == 3:
                    cpp.write("                write_double(payload, 0, ret_value.operator double());\n")
                elif method_return_type == 5:
                    cpp.write("                Vector2 v2 = ret_value.operator Vector2();\n")
                    cpp.write("                write_float(payload, 0, v2.x);\n")
                    cpp.write("                write_float(payload, 4, v2.y);\n")
                elif method_return_type == 24:
                    cpp.write("                write_object_id(payload, 0, ret_value.operator ObjectID());\n")

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
    lines.append(
        "using CommandHandler = void (*)(uint32_t, volatile uint8_t *, volatile uint32_t *, volatile uint8_t *);"
    )
    lines.append("")

    for class_ in ordered_classes:
        handler_name = f"handle_{cpp_class_map[class_]}"
        lines.append(
            f"void {handler_name}(uint32_t cmd, volatile uint8_t *payload, volatile uint32_t *cmd_ptr, volatile uint8_t *status_ptr);"
        )
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
    lines.append("    volatile uint8_t *payload = reinterpret_cast<volatile uint8_t *>(CMD_DATA);")
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
    lines.append("    command_handlers[cmd](cmd, payload, cmd_ptr, status_ptr);")
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
