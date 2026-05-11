"""
Similar implementations to CPP generator
"""

from metadata import return_types
from utilities import (
    argument_types,
    cs_local_reserved,
    sanitize_identifier,
    unique_identifier,
    write_text,
)


def _has_unnamed_args(method_arguments):
    for argument in method_arguments or []:
        if str(argument.get("name", "")).startswith("_unnamed_arg"):
            return True
    return False


def generate_commands_cs(folder_path, command_definitions):
    max_tracker = 0
    lines = []
    lines.append("public static partial class Commands")
    lines.append("{")
    lines.append("    public const int CMD_OFFSET = 0x5000;")
    lines.append("    public const int STATUS_OFFSET = 0x5004;")
    lines.append("    public const int CMD_DATA = 0x5008;")
    lines.append("")
    lines.append("    public const int CMD_NONE = 0;")
    lines.append("    public const byte STATUS_PENDING = 0;")
    lines.append("    public const byte STATUS_DONE = 1;")
    lines.append("")
    for command_definition in command_definitions:
        if command_definition["const_name"].startswith("CMD_Engine_get_singleton"):
            continue
        if _has_unnamed_args(command_definition.get("method_arguments", [])):
            continue
        lines.append(f"    public const int {command_definition['const_name']} = {command_definition['cmd_id']};")
        if command_definition["cmd_id"] > max_tracker:
            max_tracker = command_definition["cmd_id"]
    lines.append(f"    public const int CMD_Engine_get_singleton__r0 = {max_tracker + 1};")
    lines.append("}")
    write_text(folder_path / "cs" / "Commands.cs", "\n".join(lines) + "\n")


def generate_godot_object_cs(folder_path):
    lines = []
    lines.append("using System;")
    lines.append("")
    lines.append("namespace GodotWeb")
    lines.append("{")
    lines.append("    public class GodotObject")
    lines.append("    {")
    lines.append("        public ulong Id { get; protected set; }")
    lines.append("")
    lines.append("        public GodotObject(ulong id)")
    lines.append("        {")
    lines.append("            Id = id;")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    write_text(folder_path / "cs" / "GodotApi" / "GodotObject.cs", "\n".join(lines) + "\n")


def generate_cs_class_file(folder_path, file_base, cs_class_name, cs_base_class_name, methods):
    lines = []
    lines.append("using System;")
    lines.append("")
    lines.append("namespace GodotWeb")
    lines.append("{")
    lines.append(f"    public class {cs_class_name} : {cs_base_class_name}")
    lines.append("    {")
    lines.append(f"        public {cs_class_name}(ulong id) : base(id) {{ }}")
    lines.append("")

    for method_name, method_arguments, method_return_type, constant_name, command_id in methods:
        if constant_name.startswith("CMD_Engine_get_singleton"):
            continue
        if _has_unnamed_args(method_arguments):
            continue

        used_cs_arg_names = set(cs_local_reserved)
        argument_parts = []
        argument_names = []
        for argument in method_arguments:
            argument_name = unique_identifier(argument["name"], used_cs_arg_names, "cs", prefix="arg")
            argument_names.append(argument_name)
            argument_parts.append(f"{argument_types[argument['type']]['cs_type']} {argument_name}")

        arguments_cs = ", ".join(argument_parts)
        return_cs = return_types[method_return_type]["cs"]
        cs_method_name = sanitize_identifier(method_name, "cs")

        lines.append(f"        public {return_cs} {cs_method_name}({arguments_cs})")
        lines.append("        {")
        lines.append("            Helpers.WriteUInt64(Commands.CMD_DATA, Id);")

        arg_offset_expr = "8"
        for argument, argument_name in zip(method_arguments, argument_names):
            spec = argument_types[argument["type"]]
            lines.append(
                f"            {spec['cs_write'].format(pos=f'Commands.CMD_DATA + {arg_offset_expr}', name=argument_name)}"
            )

            if argument["type"] == 29:  # PackedByteArray / byte[]
                arg_offset_expr += f" + 4 + {argument_name}.Length"
            elif argument["type"] == 32:  # PackedFloat32Array / float[]
                arg_offset_expr += f" + 4 + ({argument_name}.Length * 4)"
            else:
                if arg_offset_expr.isdigit():
                    arg_offset_expr = str(int(arg_offset_expr) + spec["size"])
                else:
                    arg_offset_expr += f" + {spec['size']}"

        lines.append(f"            Helpers.SendCommand(Commands.{constant_name});")
        lines.append("            Helpers.WaitForCompletion();")
        if return_cs != "void":
            lines.append(
                f"            {return_cs} result = {return_types[method_return_type]['cs_read'].format(pos='Commands.CMD_DATA')};"
            )
            lines.append("            return result;")
        lines.append("        }")
        lines.append("")

    if file_base == "Engine":
        lines.append("        public static ulong get_singleton()")
        lines.append("        {")
        lines.append("            Helpers.SendCommand(Commands.CMD_Engine_get_singleton__r0);")
        lines.append("            Helpers.WaitForCompletion();")
        lines.append("            ulong result = Helpers.ReadUInt64(Commands.CMD_DATA);")
        lines.append("            return result;")
        lines.append("        }")
        lines.append("")

    lines.append("    }")
    lines.append("}")
    write_text(folder_path / "cs" / "GodotApi" / f"{file_base}.cs", "\n".join(lines) + "\n")
