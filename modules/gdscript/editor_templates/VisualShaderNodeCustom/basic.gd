# meta-description: Visual shader's node plugin template

@tool
extends _BASE_
class_name VisualShaderNode_CLASS_

func _get_name() -> String:
	return "_CLASS_"

func _get_category() -> String:
	return ""

func _get_description() -> String:
	return ""

func _get_return_icon_type() -> int:
	return PORT_TYPE_SCALAR

func _get_input_port_count() -> int:
	return 0

func _get_input_port_name(port: int) -> String:
	return ""

func _get_input_port_type(port: int) -> int:
	return PORT_TYPE_SCALAR

func _get_output_port_count() -> int:
	return 1

func _get_output_port_name(port: int) -> String:
	return "result"

func _get_output_port_type(port: int) -> int:
	return PORT_TYPE_SCALAR

func _get_global_code(mode: Shader.Mode) -> String:
	return ""

func _get_code(input_vars: Array[String], output_vars: Array[String], mode: Shader.Mode, type: VisualShader.Type) -> String:
	return output_vars[0] + " = 0.0;"
