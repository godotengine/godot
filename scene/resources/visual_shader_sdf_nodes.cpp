/*************************************************************************/
/*  visual_shader_sdf_nodes.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "visual_shader_sdf_nodes.h"

// VisualShaderNodeSDFToScreenUV

String VisualShaderNodeSDFToScreenUV::get_caption() const {
	return "SDFToScreenUV";
}

int VisualShaderNodeSDFToScreenUV::get_input_port_count() const {
	return 1;
}

VisualShaderNodeSDFToScreenUV::PortType VisualShaderNodeSDFToScreenUV::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeSDFToScreenUV::get_input_port_name(int p_port) const {
	return "sdf_pos";
}

int VisualShaderNodeSDFToScreenUV::get_output_port_count() const {
	return 1;
}

VisualShaderNodeSDFToScreenUV::PortType VisualShaderNodeSDFToScreenUV::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeSDFToScreenUV::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeSDFToScreenUV::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = vec3(sdf_to_screen_uv(" + (p_input_vars[0] == String() ? "vec2(0.0)" : p_input_vars[0] + ".xy") + "), 0.0f);\n";
}

VisualShaderNodeSDFToScreenUV::VisualShaderNodeSDFToScreenUV() {
}

// VisualShaderNodeScreenUVToSDF

String VisualShaderNodeScreenUVToSDF::get_caption() const {
	return "ScreenUVToSDF";
}

int VisualShaderNodeScreenUVToSDF::get_input_port_count() const {
	return 1;
}

VisualShaderNodeScreenUVToSDF::PortType VisualShaderNodeScreenUVToSDF::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeScreenUVToSDF::get_input_port_name(int p_port) const {
	return "uv";
}

int VisualShaderNodeScreenUVToSDF::get_output_port_count() const {
	return 1;
}

VisualShaderNodeScreenUVToSDF::PortType VisualShaderNodeScreenUVToSDF::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeScreenUVToSDF::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeScreenUVToSDF::get_input_port_default_hint(int p_port) const {
	if (p_port == 0) {
		return "default";
	}
	return "";
}

String VisualShaderNodeScreenUVToSDF::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = vec3(screen_uv_to_sdf(" + (p_input_vars[0] == String() ? "SCREEN_UV" : p_input_vars[0] + ".xy") + "), 0.0f);\n";
}

VisualShaderNodeScreenUVToSDF::VisualShaderNodeScreenUVToSDF() {
}

// VisualShaderNodeTextureSDF

String VisualShaderNodeTextureSDF::get_caption() const {
	return "TextureSDF";
}

int VisualShaderNodeTextureSDF::get_input_port_count() const {
	return 1;
}

VisualShaderNodeTextureSDF::PortType VisualShaderNodeTextureSDF::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeTextureSDF::get_input_port_name(int p_port) const {
	return "sdf_pos";
}

int VisualShaderNodeTextureSDF::get_output_port_count() const {
	return 1;
}

VisualShaderNodeTextureSDF::PortType VisualShaderNodeTextureSDF::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeTextureSDF::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeTextureSDF::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = texture_sdf(" + (p_input_vars[0] == String() ? "vec2(0.0)" : p_input_vars[0] + ".xy") + ");\n";
}

VisualShaderNodeTextureSDF::VisualShaderNodeTextureSDF() {
}

// VisualShaderNodeTextureSDFNormal

String VisualShaderNodeTextureSDFNormal::get_caption() const {
	return "TextureSDFNormal";
}

int VisualShaderNodeTextureSDFNormal::get_input_port_count() const {
	return 1;
}

VisualShaderNodeTextureSDFNormal::PortType VisualShaderNodeTextureSDFNormal::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeTextureSDFNormal::get_input_port_name(int p_port) const {
	return "sdf_pos";
}

int VisualShaderNodeTextureSDFNormal::get_output_port_count() const {
	return 1;
}

VisualShaderNodeTextureSDFNormal::PortType VisualShaderNodeTextureSDFNormal::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeTextureSDFNormal::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeTextureSDFNormal::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = vec3(texture_sdf_normal(" + (p_input_vars[0] == String() ? "vec2(0.0)" : p_input_vars[0] + ".xy") + "), 0.0f);\n";
}

VisualShaderNodeTextureSDFNormal::VisualShaderNodeTextureSDFNormal() {
}

// VisualShaderNodeSDFRaymarch

String VisualShaderNodeSDFRaymarch::get_caption() const {
	return "SDFRaymarch";
}

int VisualShaderNodeSDFRaymarch::get_input_port_count() const {
	return 2;
}

VisualShaderNodeSDFRaymarch::PortType VisualShaderNodeSDFRaymarch::get_input_port_type(int p_port) const {
	if (p_port == 0 || p_port == 1) {
		return PORT_TYPE_VECTOR;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeSDFRaymarch::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "from_pos";
	} else if (p_port == 1) {
		return "to_pos";
	}
	return String();
}

int VisualShaderNodeSDFRaymarch::get_output_port_count() const {
	return 3;
}

VisualShaderNodeSDFRaymarch::PortType VisualShaderNodeSDFRaymarch::get_output_port_type(int p_port) const {
	if (p_port == 0) {
		return PORT_TYPE_SCALAR;
	} else if (p_port == 1) {
		return PORT_TYPE_BOOLEAN;
	} else if (p_port == 2) {
		return PORT_TYPE_VECTOR;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeSDFRaymarch::get_output_port_name(int p_port) const {
	if (p_port == 0) {
		return "distance";
	} else if (p_port == 1) {
		return "hit";
	} else if (p_port == 2) {
		return "end_pos";
	}
	return String();
}

String VisualShaderNodeSDFRaymarch::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;

	code += "\t{\n";

	if (p_input_vars[0] == String()) {
		code += "\t\tvec2 __from_pos = vec2(0.0f);\n";
	} else {
		code += "\t\tvec2 __from_pos = " + p_input_vars[0] + ".xy;\n";
	}

	if (p_input_vars[1] == String()) {
		code += "\t\tvec2 __to_pos = vec2(0.0f);\n";
	} else {
		code += "\t\tvec2 __to_pos = " + p_input_vars[1] + ".xy;\n";
	}

	code += "\n\t\tvec2 __at = __from_pos;\n";
	code += "\t\tfloat __max_dist = distance(__from_pos, __to_pos);\n";
	code += "\t\tvec2 __dir = normalize(__to_pos - __from_pos);\n\n";

	code += "\t\tfloat __accum = 0.0f;\n";
	code += "\t\twhile(__accum < __max_dist) {\n";
	code += "\t\t\tfloat __d = texture_sdf(__at);\n";
	code += "\t\t\t__accum += __d;\n";
	code += "\t\t\tif (__d < 0.01f) {\n";
	code += "\t\t\t\tbreak;\n";
	code += "\t\t\t}\n";
	code += "\t\t\t__at += __d * __dir;\n";
	code += "\t\t}\n";

	code += "\t\tfloat __dist = min(__max_dist, __accum);\n";
	code += "\t\t" + p_output_vars[0] + " = __dist;\n";
	code += "\t\t" + p_output_vars[1] + " = __accum < __max_dist;\n";
	code += "\t\t" + p_output_vars[2] + " = vec3(__from_pos + __dir * __dist, 0.0f);\n";

	code += "\t}\n";

	return code;
}

VisualShaderNodeSDFRaymarch::VisualShaderNodeSDFRaymarch() {
	simple_decl = false;
}
