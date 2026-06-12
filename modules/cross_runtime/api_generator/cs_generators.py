# cs_generators.py
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

    def _packed_stride(argument_type: int) -> int:
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

    def _format_offset(constant: int, var_terms: list[str]) -> str:
        if not var_terms:
            return str(constant)
        if constant == 0:
            return " + ".join(var_terms)
        return f"{constant} + " + " + ".join(var_terms)

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

        constant_offset = 8
        var_terms = []

        for i, (argument, argument_name) in enumerate(zip(method_arguments, argument_names)):
            argument_type = argument["type"]
            spec = argument_types[argument_type]

            current_offset = _format_offset(constant_offset, var_terms)

            lines.append(
                f"            {spec['cs_write'].format(pos=f'Commands.CMD_DATA + {current_offset}', name=argument_name)}"
            )

            # Only advance offset if not the last argument
            if i == len(method_arguments) - 1:
                break

            if not spec.get("variable_size", False):
                constant_offset += spec["size"]
                continue

            if 29 <= argument_type <= 38:  # packed arrays
                stride = _packed_stride(argument_type)
                constant_offset += 4
                if stride == 1:
                    var_terms.append(f"{argument_name}.Length")
                else:
                    var_terms.append(f"({argument_name}.Length * {stride})")
            else:
                # other variable‑size → read length into a temp variable
                size_name = unique_identifier(f"{argument_name}_size", used_cs_arg_names, "cs", prefix="size")
                used_cs_arg_names.add(size_name)

                lines.append(f"            int {size_name} = Helpers.ReadInt32(Commands.CMD_DATA + {current_offset});")
                constant_offset += 4
                var_terms.append(size_name)

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
