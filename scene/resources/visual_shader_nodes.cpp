/*************************************************************************/
/*  visual_shader_nodes.cpp                                              */
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

#include "visual_shader_nodes.h"

////////////// Constants Base

VisualShaderNodeConstant::VisualShaderNodeConstant() {
}

////////////// Scalar(Float)

String VisualShaderNodeFloatConstant::get_caption() const {
	return "FloatConstant";
}

int VisualShaderNodeFloatConstant::get_input_port_count() const {
	return 0;
}

VisualShaderNodeFloatConstant::PortType VisualShaderNodeFloatConstant::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeFloatConstant::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeFloatConstant::get_output_port_count() const {
	return 1;
}

VisualShaderNodeFloatConstant::PortType VisualShaderNodeFloatConstant::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeFloatConstant::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeFloatConstant::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = " + vformat("%.6f", constant) + ";\n";
}

void VisualShaderNodeFloatConstant::set_constant(float p_constant) {
	if (Math::is_equal_approx(constant, p_constant)) {
		return;
	}
	constant = p_constant;
	emit_changed();
}

float VisualShaderNodeFloatConstant::get_constant() const {
	return constant;
}

Vector<StringName> VisualShaderNodeFloatConstant::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("constant");
	return props;
}

void VisualShaderNodeFloatConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant", "constant"), &VisualShaderNodeFloatConstant::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant"), &VisualShaderNodeFloatConstant::get_constant);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "constant"), "set_constant", "get_constant");
}

VisualShaderNodeFloatConstant::VisualShaderNodeFloatConstant() {
}

////////////// Scalar(Int)

String VisualShaderNodeIntConstant::get_caption() const {
	return "IntConstant";
}

int VisualShaderNodeIntConstant::get_input_port_count() const {
	return 0;
}

VisualShaderNodeIntConstant::PortType VisualShaderNodeIntConstant::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR_INT;
}

String VisualShaderNodeIntConstant::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeIntConstant::get_output_port_count() const {
	return 1;
}

VisualShaderNodeIntConstant::PortType VisualShaderNodeIntConstant::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR_INT;
}

String VisualShaderNodeIntConstant::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeIntConstant::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = " + itos(constant) + ";\n";
}

void VisualShaderNodeIntConstant::set_constant(int p_constant) {
	if (constant == p_constant) {
		return;
	}
	constant = p_constant;
	emit_changed();
}

int VisualShaderNodeIntConstant::get_constant() const {
	return constant;
}

Vector<StringName> VisualShaderNodeIntConstant::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("constant");
	return props;
}

void VisualShaderNodeIntConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant", "constant"), &VisualShaderNodeIntConstant::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant"), &VisualShaderNodeIntConstant::get_constant);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "constant"), "set_constant", "get_constant");
}

VisualShaderNodeIntConstant::VisualShaderNodeIntConstant() {
}

////////////// Boolean

String VisualShaderNodeBooleanConstant::get_caption() const {
	return "BooleanConstant";
}

int VisualShaderNodeBooleanConstant::get_input_port_count() const {
	return 0;
}

VisualShaderNodeBooleanConstant::PortType VisualShaderNodeBooleanConstant::get_input_port_type(int p_port) const {
	return PORT_TYPE_BOOLEAN;
}

String VisualShaderNodeBooleanConstant::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeBooleanConstant::get_output_port_count() const {
	return 1;
}

VisualShaderNodeBooleanConstant::PortType VisualShaderNodeBooleanConstant::get_output_port_type(int p_port) const {
	return PORT_TYPE_BOOLEAN;
}

String VisualShaderNodeBooleanConstant::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeBooleanConstant::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = " + (constant ? "true" : "false") + ";\n";
}

void VisualShaderNodeBooleanConstant::set_constant(bool p_constant) {
	if (constant == p_constant) {
		return;
	}
	constant = p_constant;
	emit_changed();
}

bool VisualShaderNodeBooleanConstant::get_constant() const {
	return constant;
}

Vector<StringName> VisualShaderNodeBooleanConstant::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("constant");
	return props;
}

void VisualShaderNodeBooleanConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant", "constant"), &VisualShaderNodeBooleanConstant::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant"), &VisualShaderNodeBooleanConstant::get_constant);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "constant"), "set_constant", "get_constant");
}

VisualShaderNodeBooleanConstant::VisualShaderNodeBooleanConstant() {
}

////////////// Color

String VisualShaderNodeColorConstant::get_caption() const {
	return "ColorConstant";
}

int VisualShaderNodeColorConstant::get_input_port_count() const {
	return 0;
}

VisualShaderNodeColorConstant::PortType VisualShaderNodeColorConstant::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeColorConstant::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeColorConstant::get_output_port_count() const {
	return 2;
}

VisualShaderNodeColorConstant::PortType VisualShaderNodeColorConstant::get_output_port_type(int p_port) const {
	return p_port == 0 ? PORT_TYPE_VECTOR : PORT_TYPE_SCALAR;
}

String VisualShaderNodeColorConstant::get_output_port_name(int p_port) const {
	return p_port == 0 ? "" : "alpha"; //no output port means the editor will be used as port
}

bool VisualShaderNodeColorConstant::is_output_port_expandable(int p_port) const {
	if (p_port == 0) {
		return true;
	}
	return false;
}

String VisualShaderNodeColorConstant::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	code += "	" + p_output_vars[0] + " = " + vformat("vec3(%.6f, %.6f, %.6f)", constant.r, constant.g, constant.b) + ";\n";
	code += "	" + p_output_vars[1] + " = " + vformat("%.6f", constant.a) + ";\n";

	return code;
}

void VisualShaderNodeColorConstant::set_constant(const Color &p_constant) {
	if (constant.is_equal_approx(p_constant)) {
		return;
	}
	constant = p_constant;
	emit_changed();
}

Color VisualShaderNodeColorConstant::get_constant() const {
	return constant;
}

Vector<StringName> VisualShaderNodeColorConstant::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("constant");
	return props;
}

void VisualShaderNodeColorConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant", "constant"), &VisualShaderNodeColorConstant::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant"), &VisualShaderNodeColorConstant::get_constant);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "constant"), "set_constant", "get_constant");
}

VisualShaderNodeColorConstant::VisualShaderNodeColorConstant() {
}

////////////// Vector

String VisualShaderNodeVec3Constant::get_caption() const {
	return "VectorConstant";
}

int VisualShaderNodeVec3Constant::get_input_port_count() const {
	return 0;
}

VisualShaderNodeVec3Constant::PortType VisualShaderNodeVec3Constant::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVec3Constant::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeVec3Constant::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVec3Constant::PortType VisualShaderNodeVec3Constant::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVec3Constant::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeVec3Constant::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = " + vformat("vec3(%.6f, %.6f, %.6f)", constant.x, constant.y, constant.z) + ";\n";
}

void VisualShaderNodeVec3Constant::set_constant(const Vector3 &p_constant) {
	if (constant.is_equal_approx(p_constant)) {
		return;
	}
	constant = p_constant;
	emit_changed();
}

Vector3 VisualShaderNodeVec3Constant::get_constant() const {
	return constant;
}

Vector<StringName> VisualShaderNodeVec3Constant::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("constant");
	return props;
}

void VisualShaderNodeVec3Constant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant", "constant"), &VisualShaderNodeVec3Constant::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant"), &VisualShaderNodeVec3Constant::get_constant);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "constant"), "set_constant", "get_constant");
}

VisualShaderNodeVec3Constant::VisualShaderNodeVec3Constant() {
}

////////////// Transform3D

String VisualShaderNodeTransformConstant::get_caption() const {
	return "TransformConstant";
}

int VisualShaderNodeTransformConstant::get_input_port_count() const {
	return 0;
}

VisualShaderNodeTransformConstant::PortType VisualShaderNodeTransformConstant::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeTransformConstant::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeTransformConstant::get_output_port_count() const {
	return 1;
}

VisualShaderNodeTransformConstant::PortType VisualShaderNodeTransformConstant::get_output_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeTransformConstant::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeTransformConstant::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	Transform3D t = constant;
	t.basis.transpose();

	String code = "	" + p_output_vars[0] + " = mat4(";
	code += vformat("vec4(%.6f, %.6f, %.6f, 0.0), ", t.basis[0].x, t.basis[0].y, t.basis[0].z);
	code += vformat("vec4(%.6f, %.6f, %.6f, 0.0), ", t.basis[1].x, t.basis[1].y, t.basis[1].z);
	code += vformat("vec4(%.6f, %.6f, %.6f, 0.0), ", t.basis[2].x, t.basis[2].y, t.basis[2].z);
	code += vformat("vec4(%.6f, %.6f, %.6f, 1.0));\n", t.origin.x, t.origin.y, t.origin.z);
	return code;
}

void VisualShaderNodeTransformConstant::set_constant(const Transform3D &p_constant) {
	if (constant.is_equal_approx(p_constant)) {
		return;
	}
	constant = p_constant;
	emit_changed();
}

Transform3D VisualShaderNodeTransformConstant::get_constant() const {
	return constant;
}

Vector<StringName> VisualShaderNodeTransformConstant::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("constant");
	return props;
}

void VisualShaderNodeTransformConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant", "constant"), &VisualShaderNodeTransformConstant::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant"), &VisualShaderNodeTransformConstant::get_constant);

	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "constant"), "set_constant", "get_constant");
}

VisualShaderNodeTransformConstant::VisualShaderNodeTransformConstant() {
}

////////////// Texture

String VisualShaderNodeTexture::get_caption() const {
	return "Texture2D";
}

int VisualShaderNodeTexture::get_input_port_count() const {
	return 3;
}

VisualShaderNodeTexture::PortType VisualShaderNodeTexture::get_input_port_type(int p_port) const {
	switch (p_port) {
		case 0:
			return PORT_TYPE_VECTOR;
		case 1:
			return PORT_TYPE_SCALAR;
		case 2:
			return PORT_TYPE_SAMPLER;
		default:
			return PORT_TYPE_SCALAR;
	}
}

String VisualShaderNodeTexture::get_input_port_name(int p_port) const {
	switch (p_port) {
		case 0:
			return "uv";
		case 1:
			return "lod";
		case 2:
			return "sampler2D";
		default:
			return "";
	}
}

int VisualShaderNodeTexture::get_output_port_count() const {
	return 2;
}

VisualShaderNodeTexture::PortType VisualShaderNodeTexture::get_output_port_type(int p_port) const {
	if (p_port == 0 && source == SOURCE_DEPTH) {
		return PORT_TYPE_SCALAR;
	}
	return p_port == 0 ? PORT_TYPE_VECTOR : PORT_TYPE_SCALAR;
}

String VisualShaderNodeTexture::get_output_port_name(int p_port) const {
	if (p_port == 0 && source == SOURCE_DEPTH) {
		return "depth";
	}
	return p_port == 0 ? "rgb" : "alpha";
}

bool VisualShaderNodeTexture::is_output_port_expandable(int p_port) const {
	if (p_port == 0) {
		return true;
	}
	return false;
}

String VisualShaderNodeTexture::get_input_port_default_hint(int p_port) const {
	if (p_port == 0) {
		return "default";
	}
	return "";
}

Vector<VisualShader::DefaultTextureParam> VisualShaderNodeTexture::get_default_texture_parameters(VisualShader::Type p_type, int p_id) const {
	VisualShader::DefaultTextureParam dtp;
	dtp.name = make_unique_id(p_type, p_id, "tex");
	dtp.params.push_back(texture);
	Vector<VisualShader::DefaultTextureParam> ret;
	ret.push_back(dtp);
	return ret;
}

String VisualShaderNodeTexture::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	if (source == SOURCE_TEXTURE) {
		String u = "uniform sampler2D " + make_unique_id(p_type, p_id, "tex");
		switch (texture_type) {
			case TYPE_DATA:
				break;
			case TYPE_COLOR:
				u += " : hint_albedo";
				break;
			case TYPE_NORMAL_MAP:
				u += " : hint_normal";
				break;
			default:
				break;
		}
		return u + ";\n";
	}

	return String();
}

String VisualShaderNodeTexture::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String default_uv;
	if (p_mode != Shader::MODE_PARTICLES && p_mode != Shader::MODE_SKY) {
		default_uv = "UV.xy";
	} else {
		default_uv = "vec2(0.0)";
	}

	if (source == SOURCE_TEXTURE) {
		String id = make_unique_id(p_type, p_id, "tex");
		String code;
		if (p_input_vars[0].is_empty()) { // Use UV by default.

			if (p_input_vars[1].is_empty()) {
				code += "	vec4 " + id + "_read = texture(" + id + ", " + default_uv + ");\n";
			} else {
				code += "	vec4 " + id + "_read = textureLod(" + id + ", " + default_uv + ", " + p_input_vars[1] + ");\n";
			}

		} else if (p_input_vars[1].is_empty()) {
			//no lod
			code += "	vec4 " + id + "_read = texture(" + id + ", " + p_input_vars[0] + ".xy);\n";
		} else {
			code += "	vec4 " + id + "_read = textureLod(" + id + ", " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ");\n";
		}

		code += "	" + p_output_vars[0] + " = " + id + "_read.rgb;\n";
		code += "	" + p_output_vars[1] + " = " + id + "_read.a;\n";
		return code;
	}

	if (source == SOURCE_PORT) {
		String id = p_input_vars[2];

		String code;
		code += "	{\n";
		if (id.is_empty()) {
			code += "		vec4 " + id + "_tex_read = vec4(0.0);\n";
		} else {
			if (p_input_vars[0].is_empty()) { // Use UV by default.

				if (p_input_vars[1].is_empty()) {
					code += "		vec4 " + id + "_tex_read = texture(" + id + ", " + default_uv + ");\n";
				} else {
					code += "		vec4 " + id + "_tex_read = textureLod(" + id + ", " + default_uv + ", " + p_input_vars[1] + ");\n";
				}

			} else if (p_input_vars[1].is_empty()) {
				//no lod
				code += "		vec4 " + id + "_tex_read = texture(" + id + ", " + p_input_vars[0] + ".xy);\n";
			} else {
				code += "		vec4 " + id + "_tex_read = textureLod(" + id + ", " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ");\n";
			}

			code += "		" + p_output_vars[0] + " = " + id + "_tex_read.rgb;\n";
			code += "		" + p_output_vars[1] + " = " + id + "_tex_read.a;\n";
		}
		code += "	}\n";
		return code;
	}

	if (source == SOURCE_SCREEN && (p_mode == Shader::MODE_SPATIAL || p_mode == Shader::MODE_CANVAS_ITEM) && p_type == VisualShader::TYPE_FRAGMENT) {
		String code = "	{\n";
		if (p_input_vars[0].is_empty() || p_for_preview) { // Use UV by default.

			if (p_input_vars[1].is_empty()) {
				code += "		vec4 _tex_read = textureLod(SCREEN_TEXTURE, " + default_uv + ", 0.0 );\n";
			} else {
				code += "		vec4 _tex_read = textureLod(SCREEN_TEXTURE, " + default_uv + ", " + p_input_vars[1] + ");\n";
			}

		} else if (p_input_vars[1].is_empty()) {
			//no lod
			code += "		vec4 _tex_read = textureLod(SCREEN_TEXTURE, " + p_input_vars[0] + ".xy, 0.0);\n";
		} else {
			code += "		vec4 _tex_read = textureLod(SCREEN_TEXTURE, " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ");\n";
		}

		code += "		" + p_output_vars[0] + " = _tex_read.rgb;\n";
		code += "		" + p_output_vars[1] + " = _tex_read.a;\n";
		code += "	}\n";
		return code;
	}

	if (source == SOURCE_2D_TEXTURE && p_mode == Shader::MODE_CANVAS_ITEM && p_type == VisualShader::TYPE_FRAGMENT) {
		String code = "	{\n";
		if (p_input_vars[0].is_empty()) { // Use UV by default.

			if (p_input_vars[1].is_empty()) {
				code += "		vec4 _tex_read = texture(TEXTURE, " + default_uv + ");\n";
			} else {
				code += "		vec4 _tex_read = textureLod(TEXTURE, " + default_uv + ", " + p_input_vars[1] + ");\n";
			}

		} else if (p_input_vars[1].is_empty()) {
			//no lod
			code += "		vec4 _tex_read = texture(TEXTURE, " + p_input_vars[0] + ".xy);\n";
		} else {
			code += "		vec4 _tex_read = textureLod(TEXTURE, " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ");\n";
		}

		code += "		" + p_output_vars[0] + " = _tex_read.rgb;\n";
		code += "		" + p_output_vars[1] + " = _tex_read.a;\n";
		code += "	}\n";
		return code;
	}

	if (source == SOURCE_2D_NORMAL && p_mode == Shader::MODE_CANVAS_ITEM && p_type == VisualShader::TYPE_FRAGMENT) {
		String code = "	{\n";
		if (p_input_vars[0].is_empty()) { // Use UV by default.

			if (p_input_vars[1].is_empty()) {
				code += "		vec4 _tex_read = texture(NORMAL_TEXTURE, " + default_uv + ");\n";
			} else {
				code += "		vec4 _tex_read = textureLod(NORMAL_TEXTURE, " + default_uv + ", " + p_input_vars[1] + ");\n";
			}

		} else if (p_input_vars[1].is_empty()) {
			//no lod
			code += "		vec4 _tex_read = texture(NORMAL_TEXTURE, " + p_input_vars[0] + ".xy);\n";
		} else {
			code += "		vec4 _tex_read = textureLod(NORMAL_TEXTURE, " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ");\n";
		}

		code += "		" + p_output_vars[0] + " = _tex_read.rgb;\n";
		code += "		" + p_output_vars[1] + " = _tex_read.a;\n";
		code += "	}\n";
		return code;
	}

	if (p_for_preview) // DEPTH_TEXTURE is not supported in preview(canvas_item) shader
	{
		if (source == SOURCE_DEPTH) {
			String code;
			code += "	" + p_output_vars[0] + " = 0.0;\n";
			code += "	" + p_output_vars[1] + " = 1.0;\n";
			return code;
		}
	}

	if (source == SOURCE_DEPTH && p_mode == Shader::MODE_SPATIAL && p_type == VisualShader::TYPE_FRAGMENT) {
		String code = "	{\n";
		if (p_input_vars[0].is_empty()) { // Use UV by default.

			if (p_input_vars[1].is_empty()) {
				code += "		float _depth = texture(DEPTH_TEXTURE, " + default_uv + ").r;\n";
			} else {
				code += "		float _depth = textureLod(DEPTH_TEXTURE, " + default_uv + ", " + p_input_vars[1] + ").r;\n";
			}

		} else if (p_input_vars[1].is_empty()) {
			//no lod
			code += "		float _depth = texture(DEPTH_TEXTURE, " + p_input_vars[0] + ".xy).r;\n";
		} else {
			code += "		float _depth = textureLod(DEPTH_TEXTURE, " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ").r;\n";
		}

		code += "		" + p_output_vars[0] + " = _depth;\n";
		code += "		" + p_output_vars[1] + " = 1.0;\n";
		code += "	}\n";
		return code;
	} else if (source == SOURCE_DEPTH) {
		String code;
		code += "	" + p_output_vars[0] + " = 0.0;\n";
		code += "	" + p_output_vars[1] + " = 1.0;\n";
		return code;
	}

	//none
	String code;
	code += "	" + p_output_vars[0] + " = vec3(0.0);\n";
	code += "	" + p_output_vars[1] + " = 1.0;\n";
	return code;
}

void VisualShaderNodeTexture::set_source(Source p_source) {
	ERR_FAIL_INDEX(int(p_source), int(SOURCE_MAX));
	if (source == p_source) {
		return;
	}
	switch (p_source) {
		case SOURCE_TEXTURE:
			simple_decl = true;
			break;
		case SOURCE_SCREEN:
			simple_decl = false;
			break;
		case SOURCE_2D_TEXTURE:
			simple_decl = false;
			break;
		case SOURCE_2D_NORMAL:
			simple_decl = false;
			break;
		case SOURCE_DEPTH:
			simple_decl = false;
			break;
		case SOURCE_PORT:
			simple_decl = false;
			break;
		default:
			break;
	}
	source = p_source;
	emit_changed();
	emit_signal(SNAME("editor_refresh_request"));
}

VisualShaderNodeTexture::Source VisualShaderNodeTexture::get_source() const {
	return source;
}

void VisualShaderNodeTexture::set_texture(Ref<Texture2D> p_texture) {
	texture = p_texture;
	emit_changed();
}

Ref<Texture2D> VisualShaderNodeTexture::get_texture() const {
	return texture;
}

void VisualShaderNodeTexture::set_texture_type(TextureType p_texture_type) {
	ERR_FAIL_INDEX(int(p_texture_type), int(TYPE_MAX));
	if (texture_type == p_texture_type) {
		return;
	}
	texture_type = p_texture_type;
	emit_changed();
}

VisualShaderNodeTexture::TextureType VisualShaderNodeTexture::get_texture_type() const {
	return texture_type;
}

Vector<StringName> VisualShaderNodeTexture::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("source");
	if (source == SOURCE_TEXTURE) {
		props.push_back("texture");
		props.push_back("texture_type");
	}
	return props;
}

String VisualShaderNodeTexture::get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const {
	if (is_input_port_connected(2) && source != SOURCE_PORT) {
		return TTR("The sampler port is connected but not used. Consider changing the source to 'SamplerPort'.");
	}

	if (source == SOURCE_TEXTURE) {
		return String(); // all good
	}

	if (source == SOURCE_PORT) {
		return String(); // all good
	}

	if (source == SOURCE_SCREEN && (p_mode == Shader::MODE_SPATIAL || p_mode == Shader::MODE_CANVAS_ITEM) && p_type == VisualShader::TYPE_FRAGMENT) {
		return String(); // all good
	}

	if (source == SOURCE_2D_TEXTURE && p_mode == Shader::MODE_CANVAS_ITEM && p_type == VisualShader::TYPE_FRAGMENT) {
		return String(); // all good
	}

	if (source == SOURCE_2D_NORMAL && p_mode == Shader::MODE_CANVAS_ITEM) {
		return String(); // all good
	}

	if (source == SOURCE_DEPTH && p_mode == Shader::MODE_SPATIAL && p_type == VisualShader::TYPE_FRAGMENT) {
		if (get_output_port_for_preview() == 0) { // DEPTH_TEXTURE is not supported in preview(canvas_item) shader
			return TTR("Invalid source for preview.");
		}
		return String(); // all good
	}

	return TTR("Invalid source for shader.");
}

void VisualShaderNodeTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_source", "value"), &VisualShaderNodeTexture::set_source);
	ClassDB::bind_method(D_METHOD("get_source"), &VisualShaderNodeTexture::get_source);

	ClassDB::bind_method(D_METHOD("set_texture", "value"), &VisualShaderNodeTexture::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &VisualShaderNodeTexture::get_texture);

	ClassDB::bind_method(D_METHOD("set_texture_type", "value"), &VisualShaderNodeTexture::set_texture_type);
	ClassDB::bind_method(D_METHOD("get_texture_type"), &VisualShaderNodeTexture::get_texture_type);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "source", PROPERTY_HINT_ENUM, "Texture,Screen,Texture2D,NormalMap2D,Depth,SamplerPort"), "set_source", "get_source");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_type", PROPERTY_HINT_ENUM, "Data,Color,Normal Map"), "set_texture_type", "get_texture_type");

	BIND_ENUM_CONSTANT(SOURCE_TEXTURE);
	BIND_ENUM_CONSTANT(SOURCE_SCREEN);
	BIND_ENUM_CONSTANT(SOURCE_2D_TEXTURE);
	BIND_ENUM_CONSTANT(SOURCE_2D_NORMAL);
	BIND_ENUM_CONSTANT(SOURCE_DEPTH);
	BIND_ENUM_CONSTANT(SOURCE_PORT);
	BIND_ENUM_CONSTANT(SOURCE_MAX);

	BIND_ENUM_CONSTANT(TYPE_DATA);
	BIND_ENUM_CONSTANT(TYPE_COLOR);
	BIND_ENUM_CONSTANT(TYPE_NORMAL_MAP);
	BIND_ENUM_CONSTANT(TYPE_MAX);
}

VisualShaderNodeTexture::VisualShaderNodeTexture() {
}

////////////// CurveTexture

String VisualShaderNodeCurveTexture::get_caption() const {
	return "CurveTexture";
}

int VisualShaderNodeCurveTexture::get_input_port_count() const {
	return 1;
}

VisualShaderNodeCurveTexture::PortType VisualShaderNodeCurveTexture::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeCurveTexture::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeCurveTexture::get_output_port_count() const {
	return 1;
}

VisualShaderNodeCurveTexture::PortType VisualShaderNodeCurveTexture::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeCurveTexture::get_output_port_name(int p_port) const {
	return String();
}

void VisualShaderNodeCurveTexture::set_texture(Ref<CurveTexture> p_texture) {
	texture = p_texture;
	emit_changed();
}

Ref<CurveTexture> VisualShaderNodeCurveTexture::get_texture() const {
	return texture;
}

Vector<StringName> VisualShaderNodeCurveTexture::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("texture");
	return props;
}

String VisualShaderNodeCurveTexture::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	return "uniform sampler2D " + make_unique_id(p_type, p_id, "curve") + ";\n";
}

String VisualShaderNodeCurveTexture::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	if (p_input_vars[0].is_empty()) {
		return "	" + p_output_vars[0] + " = 0.0;\n";
	}
	String id = make_unique_id(p_type, p_id, "curve");
	String code;
	code += "	" + p_output_vars[0] + " = texture(" + id + ", vec2(" + p_input_vars[0] + ")).r;\n";
	return code;
}

Vector<VisualShader::DefaultTextureParam> VisualShaderNodeCurveTexture::get_default_texture_parameters(VisualShader::Type p_type, int p_id) const {
	VisualShader::DefaultTextureParam dtp;
	dtp.name = make_unique_id(p_type, p_id, "curve");
	dtp.params.push_back(texture);
	Vector<VisualShader::DefaultTextureParam> ret;
	ret.push_back(dtp);
	return ret;
}

void VisualShaderNodeCurveTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &VisualShaderNodeCurveTexture::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &VisualShaderNodeCurveTexture::get_texture);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "CurveTexture"), "set_texture", "get_texture");
}

bool VisualShaderNodeCurveTexture::is_use_prop_slots() const {
	return true;
}

VisualShaderNodeCurveTexture::VisualShaderNodeCurveTexture() {
	set_input_port_default_value(0, 0.0);
	simple_decl = true;
	allow_v_resize = false;
}

////////////// CurveXYZTexture

String VisualShaderNodeCurveXYZTexture::get_caption() const {
	return "CurveXYZTexture";
}

int VisualShaderNodeCurveXYZTexture::get_input_port_count() const {
	return 1;
}

VisualShaderNodeCurveXYZTexture::PortType VisualShaderNodeCurveXYZTexture::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeCurveXYZTexture::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeCurveXYZTexture::get_output_port_count() const {
	return 1;
}

VisualShaderNodeCurveXYZTexture::PortType VisualShaderNodeCurveXYZTexture::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeCurveXYZTexture::get_output_port_name(int p_port) const {
	return String();
}

void VisualShaderNodeCurveXYZTexture::set_texture(Ref<CurveXYZTexture> p_texture) {
	texture = p_texture;
	emit_changed();
}

Ref<CurveXYZTexture> VisualShaderNodeCurveXYZTexture::get_texture() const {
	return texture;
}

Vector<StringName> VisualShaderNodeCurveXYZTexture::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("texture");
	return props;
}

String VisualShaderNodeCurveXYZTexture::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	return "uniform sampler2D " + make_unique_id(p_type, p_id, "curve3d") + ";\n";
}

String VisualShaderNodeCurveXYZTexture::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	if (p_input_vars[0].is_empty()) {
		return "	" + p_output_vars[0] + " = vec3(0.0);\n";
	}
	String id = make_unique_id(p_type, p_id, "curve3d");
	String code;
	code += "	" + p_output_vars[0] + " = texture(" + id + ", vec2(" + p_input_vars[0] + ")).rgb;\n";
	return code;
}

Vector<VisualShader::DefaultTextureParam> VisualShaderNodeCurveXYZTexture::get_default_texture_parameters(VisualShader::Type p_type, int p_id) const {
	VisualShader::DefaultTextureParam dtp;
	dtp.name = make_unique_id(p_type, p_id, "curve3d");
	dtp.params.push_back(texture);
	Vector<VisualShader::DefaultTextureParam> ret;
	ret.push_back(dtp);
	return ret;
}

void VisualShaderNodeCurveXYZTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &VisualShaderNodeCurveXYZTexture::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &VisualShaderNodeCurveXYZTexture::get_texture);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "CurveXYZTexture"), "set_texture", "get_texture");
}

bool VisualShaderNodeCurveXYZTexture::is_use_prop_slots() const {
	return true;
}

VisualShaderNodeCurveXYZTexture::VisualShaderNodeCurveXYZTexture() {
	set_input_port_default_value(0, 0.0);
	simple_decl = true;
	allow_v_resize = false;
}

////////////// Sample3D

int VisualShaderNodeSample3D::get_input_port_count() const {
	return 3;
}

VisualShaderNodeSample3D::PortType VisualShaderNodeSample3D::get_input_port_type(int p_port) const {
	switch (p_port) {
		case 0:
			return PORT_TYPE_VECTOR;
		case 1:
			return PORT_TYPE_SCALAR;
		case 2:
			return PORT_TYPE_SAMPLER;
		default:
			return PORT_TYPE_SCALAR;
	}
}

String VisualShaderNodeSample3D::get_input_port_name(int p_port) const {
	switch (p_port) {
		case 0:
			return "uvw";
		case 1:
			return "lod";
		default:
			return "";
	}
}

int VisualShaderNodeSample3D::get_output_port_count() const {
	return 2;
}

VisualShaderNodeSample3D::PortType VisualShaderNodeSample3D::get_output_port_type(int p_port) const {
	return p_port == 0 ? PORT_TYPE_VECTOR : PORT_TYPE_SCALAR;
}

String VisualShaderNodeSample3D::get_output_port_name(int p_port) const {
	return p_port == 0 ? "rgb" : "alpha";
}

bool VisualShaderNodeSample3D::is_output_port_expandable(int p_port) const {
	if (p_port == 0) {
		return true;
	}
	return false;
}

String VisualShaderNodeSample3D::get_input_port_default_hint(int p_port) const {
	if (p_port == 0) {
		return "default";
	}
	return "";
}

String VisualShaderNodeSample3D::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String default_uv;
	if (p_mode != Shader::MODE_PARTICLES && p_mode != Shader::MODE_SKY) {
		default_uv = "vec3(UV, 0.0)";
	} else {
		default_uv = "vec3(0.0)";
	}

	String code;
	if (source == SOURCE_TEXTURE || source == SOURCE_PORT) {
		String id;
		code += "	{\n";
		if (source == SOURCE_TEXTURE) {
			id = make_unique_id(p_type, p_id, "tex3d");
		} else {
			id = p_input_vars[2];
		}
		if (!id.is_empty()) {
			if (p_input_vars[0].is_empty()) { // Use UV by default.
				if (p_input_vars[1].is_empty()) {
					code += "		vec4 " + id + "_tex_read = texture(" + id + ", " + default_uv + ");\n";
				} else {
					code += "		vec4 " + id + "_tex_read = textureLod(" + id + ", " + default_uv + ", " + p_input_vars[1] + ");\n";
				}
			} else if (p_input_vars[1].is_empty()) {
				//no lod
				code += "		vec4 " + id + "_tex_read = texture(" + id + ", " + p_input_vars[0] + ");\n";
			} else {
				code += "		vec4 " + id + "_tex_read = textureLod(" + id + ", " + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			}
		} else {
			code += "		vec4 " + id + "_tex_read = vec4(0.0);\n";
		}

		code += "		" + p_output_vars[0] + " = " + id + "_tex_read.rgb;\n";
		code += "		" + p_output_vars[1] + " = " + id + "_tex_read.a;\n";
		code += "	}\n";
		return code;
	}
	code += "	" + p_output_vars[0] + " = vec3(0.0);\n";
	code += "	" + p_output_vars[1] + " = 1.0;\n";
	return code;
}

void VisualShaderNodeSample3D::set_source(Source p_source) {
	ERR_FAIL_INDEX(int(p_source), int(SOURCE_MAX));
	if (source == p_source) {
		return;
	}
	source = p_source;
	emit_changed();
	emit_signal(SNAME("editor_refresh_request"));
}

VisualShaderNodeSample3D::Source VisualShaderNodeSample3D::get_source() const {
	return source;
}

void VisualShaderNodeSample3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_source", "value"), &VisualShaderNodeSample3D::set_source);
	ClassDB::bind_method(D_METHOD("get_source"), &VisualShaderNodeSample3D::get_source);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "source", PROPERTY_HINT_ENUM, "Texture,SamplerPort"), "set_source", "get_source");

	BIND_ENUM_CONSTANT(SOURCE_TEXTURE);
	BIND_ENUM_CONSTANT(SOURCE_PORT);
	BIND_ENUM_CONSTANT(SOURCE_MAX);
}

String VisualShaderNodeSample3D::get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const {
	if (is_input_port_connected(2) && source != SOURCE_PORT) {
		return TTR("The sampler port is connected but not used. Consider changing the source to 'SamplerPort'.");
	}

	if (source == SOURCE_TEXTURE) {
		return String(); // all good
	}
	if (source == SOURCE_PORT) {
		return String(); // all good
	}
	return TTR("Invalid source for shader.");
}

VisualShaderNodeSample3D::VisualShaderNodeSample3D() {
	simple_decl = false;
}

////////////// Texture2DArray

String VisualShaderNodeTexture2DArray::get_caption() const {
	return "Texture2DArray";
}

String VisualShaderNodeTexture2DArray::get_input_port_name(int p_port) const {
	if (p_port == 2) {
		return "sampler2DArray";
	}
	return VisualShaderNodeSample3D::get_input_port_name(p_port);
}

Vector<VisualShader::DefaultTextureParam> VisualShaderNodeTexture2DArray::get_default_texture_parameters(VisualShader::Type p_type, int p_id) const {
	VisualShader::DefaultTextureParam dtp;
	dtp.name = make_unique_id(p_type, p_id, "tex3d");
	dtp.params.push_back(texture_array);
	Vector<VisualShader::DefaultTextureParam> ret;
	ret.push_back(dtp);
	return ret;
}

String VisualShaderNodeTexture2DArray::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	if (source == SOURCE_TEXTURE) {
		return "uniform sampler2DArray " + make_unique_id(p_type, p_id, "tex3d") + ";\n";
	}
	return String();
}

void VisualShaderNodeTexture2DArray::set_texture_array(Ref<Texture2DArray> p_texture_array) {
	texture_array = p_texture_array;
	emit_changed();
}

Ref<Texture2DArray> VisualShaderNodeTexture2DArray::get_texture_array() const {
	return texture_array;
}

Vector<StringName> VisualShaderNodeTexture2DArray::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("source");
	if (source == SOURCE_TEXTURE) {
		props.push_back("texture_array");
	}
	return props;
}

void VisualShaderNodeTexture2DArray::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture_array", "value"), &VisualShaderNodeTexture2DArray::set_texture_array);
	ClassDB::bind_method(D_METHOD("get_texture_array"), &VisualShaderNodeTexture2DArray::get_texture_array);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture_array", PROPERTY_HINT_RESOURCE_TYPE, "Texture2DArray"), "set_texture_array", "get_texture_array");
}

VisualShaderNodeTexture2DArray::VisualShaderNodeTexture2DArray() {
}

////////////// Texture3D

String VisualShaderNodeTexture3D::get_caption() const {
	return "Texture3D";
}

String VisualShaderNodeTexture3D::get_input_port_name(int p_port) const {
	if (p_port == 2) {
		return "sampler3D";
	}
	return VisualShaderNodeSample3D::get_input_port_name(p_port);
}

Vector<VisualShader::DefaultTextureParam> VisualShaderNodeTexture3D::get_default_texture_parameters(VisualShader::Type p_type, int p_id) const {
	VisualShader::DefaultTextureParam dtp;
	dtp.name = make_unique_id(p_type, p_id, "tex3d");
	dtp.params.push_back(texture);
	Vector<VisualShader::DefaultTextureParam> ret;
	ret.push_back(dtp);
	return ret;
}

String VisualShaderNodeTexture3D::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	if (source == SOURCE_TEXTURE) {
		return "uniform sampler3D " + make_unique_id(p_type, p_id, "tex3d") + ";\n";
	}
	return String();
}

void VisualShaderNodeTexture3D::set_texture(Ref<Texture3D> p_texture) {
	texture = p_texture;
	emit_changed();
}

Ref<Texture3D> VisualShaderNodeTexture3D::get_texture() const {
	return texture;
}

Vector<StringName> VisualShaderNodeTexture3D::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("source");
	if (source == SOURCE_TEXTURE) {
		props.push_back("texture");
	}
	return props;
}

void VisualShaderNodeTexture3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture", "value"), &VisualShaderNodeTexture3D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &VisualShaderNodeTexture3D::get_texture);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture3D"), "set_texture", "get_texture");
}

VisualShaderNodeTexture3D::VisualShaderNodeTexture3D() {
}

////////////// Cubemap

String VisualShaderNodeCubemap::get_caption() const {
	return "Cubemap";
}

int VisualShaderNodeCubemap::get_input_port_count() const {
	return 3;
}

VisualShaderNodeCubemap::PortType VisualShaderNodeCubemap::get_input_port_type(int p_port) const {
	switch (p_port) {
		case 0:
			return PORT_TYPE_VECTOR;
		case 1:
			return PORT_TYPE_SCALAR;
		case 2:
			return PORT_TYPE_SAMPLER;
		default:
			return PORT_TYPE_SCALAR;
	}
}

String VisualShaderNodeCubemap::get_input_port_name(int p_port) const {
	switch (p_port) {
		case 0:
			return "uv";
		case 1:
			return "lod";
		case 2:
			return "samplerCube";
		default:
			return "";
	}
}

int VisualShaderNodeCubemap::get_output_port_count() const {
	return 2;
}

VisualShaderNodeCubemap::PortType VisualShaderNodeCubemap::get_output_port_type(int p_port) const {
	return p_port == 0 ? PORT_TYPE_VECTOR : PORT_TYPE_SCALAR;
}

String VisualShaderNodeCubemap::get_output_port_name(int p_port) const {
	return p_port == 0 ? "rgb" : "alpha";
}

bool VisualShaderNodeCubemap::is_output_port_expandable(int p_port) const {
	if (p_port == 0) {
		return true;
	}
	return false;
}

Vector<VisualShader::DefaultTextureParam> VisualShaderNodeCubemap::get_default_texture_parameters(VisualShader::Type p_type, int p_id) const {
	VisualShader::DefaultTextureParam dtp;
	dtp.name = make_unique_id(p_type, p_id, "cube");
	dtp.params.push_back(cube_map);
	Vector<VisualShader::DefaultTextureParam> ret;
	ret.push_back(dtp);
	return ret;
}

String VisualShaderNodeCubemap::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	if (source == SOURCE_TEXTURE) {
		String u = "uniform samplerCube " + make_unique_id(p_type, p_id, "cube");
		switch (texture_type) {
			case TYPE_DATA:
				break;
			case TYPE_COLOR:
				u += " : hint_albedo";
				break;
			case TYPE_NORMAL_MAP:
				u += " : hint_normal";
				break;
			default:
				break;
		}
		return u + ";\n";
	}
	return String();
}

String VisualShaderNodeCubemap::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String default_uv;
	if (p_mode != Shader::MODE_PARTICLES && p_mode != Shader::MODE_SKY) {
		default_uv = "vec3(UV, 0.0)";
	} else {
		default_uv = "vec3(0.0)";
	}

	String code;
	String id;
	if (source == SOURCE_TEXTURE) {
		id = make_unique_id(p_type, p_id, "cube");
	} else if (source == SOURCE_PORT) {
		id = p_input_vars[2];
	} else {
		return String();
	}

	code += "	{\n";

	if (id.is_empty()) {
		code += "		vec4 " + id + "_read = vec4(0.0);\n";
		code += "		" + p_output_vars[0] + " = " + id + "_read.rgb;\n";
		code += "		" + p_output_vars[1] + " = " + id + "_read.a;\n";
		code += "	}\n";
		return code;
	}

	if (p_input_vars[0].is_empty()) { // Use UV by default.

		if (p_input_vars[1].is_empty()) {
			code += "		vec4 " + id + "_read = texture(" + id + ", " + default_uv + ");\n";
		} else {
			code += "		vec4 " + id + "_read = textureLod(" + id + ", " + default_uv + ", " + p_input_vars[1] + " );\n";
		}

	} else if (p_input_vars[1].is_empty()) {
		//no lod
		code += "		vec4 " + id + "_read = texture(" + id + ", " + p_input_vars[0] + ");\n";
	} else {
		code += "		vec4 " + id + "_read = textureLod(" + id + ", " + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
	}
	code += "		" + p_output_vars[0] + " = " + id + "_read.rgb;\n";
	code += "		" + p_output_vars[1] + " = " + id + "_read.a;\n";
	code += "	}\n";

	return code;
}

String VisualShaderNodeCubemap::get_input_port_default_hint(int p_port) const {
	if (p_port == 0) {
		return "default";
	}
	return "";
}

void VisualShaderNodeCubemap::set_source(Source p_source) {
	ERR_FAIL_INDEX(int(p_source), int(SOURCE_MAX));
	if (source == p_source) {
		return;
	}
	source = p_source;
	emit_changed();
	emit_signal(SNAME("editor_refresh_request"));
}

VisualShaderNodeCubemap::Source VisualShaderNodeCubemap::get_source() const {
	return source;
}

void VisualShaderNodeCubemap::set_cube_map(Ref<Cubemap> p_cube_map) {
	cube_map = p_cube_map;
	emit_changed();
}

Ref<Cubemap> VisualShaderNodeCubemap::get_cube_map() const {
	return cube_map;
}

void VisualShaderNodeCubemap::set_texture_type(TextureType p_texture_type) {
	ERR_FAIL_INDEX(int(p_texture_type), int(TYPE_MAX));
	if (texture_type == p_texture_type) {
		return;
	}
	texture_type = p_texture_type;
	emit_changed();
}

VisualShaderNodeCubemap::TextureType VisualShaderNodeCubemap::get_texture_type() const {
	return texture_type;
}

Vector<StringName> VisualShaderNodeCubemap::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("source");
	if (source == SOURCE_TEXTURE) {
		props.push_back("cube_map");
		props.push_back("texture_type");
	}
	return props;
}

String VisualShaderNodeCubemap::get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const {
	if (is_input_port_connected(2) && source != SOURCE_PORT) {
		return TTR("The sampler port is connected but not used. Consider changing the source to 'SamplerPort'.");
	}
	return String();
}

void VisualShaderNodeCubemap::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_source", "value"), &VisualShaderNodeCubemap::set_source);
	ClassDB::bind_method(D_METHOD("get_source"), &VisualShaderNodeCubemap::get_source);

	ClassDB::bind_method(D_METHOD("set_cube_map", "value"), &VisualShaderNodeCubemap::set_cube_map);
	ClassDB::bind_method(D_METHOD("get_cube_map"), &VisualShaderNodeCubemap::get_cube_map);

	ClassDB::bind_method(D_METHOD("set_texture_type", "value"), &VisualShaderNodeCubemap::set_texture_type);
	ClassDB::bind_method(D_METHOD("get_texture_type"), &VisualShaderNodeCubemap::get_texture_type);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "source", PROPERTY_HINT_ENUM, "Texture,SamplerPort"), "set_source", "get_source");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "cube_map", PROPERTY_HINT_RESOURCE_TYPE, "Cubemap"), "set_cube_map", "get_cube_map");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_type", PROPERTY_HINT_ENUM, "Data,Color,Normal Map"), "set_texture_type", "get_texture_type");

	BIND_ENUM_CONSTANT(SOURCE_TEXTURE);
	BIND_ENUM_CONSTANT(SOURCE_PORT);
	BIND_ENUM_CONSTANT(SOURCE_MAX);

	BIND_ENUM_CONSTANT(TYPE_DATA);
	BIND_ENUM_CONSTANT(TYPE_COLOR);
	BIND_ENUM_CONSTANT(TYPE_NORMAL_MAP);
	BIND_ENUM_CONSTANT(TYPE_MAX);
}

VisualShaderNodeCubemap::VisualShaderNodeCubemap() {
	simple_decl = false;
}

////////////// Float Op

String VisualShaderNodeFloatOp::get_caption() const {
	return "FloatOp";
}

int VisualShaderNodeFloatOp::get_input_port_count() const {
	return 2;
}

VisualShaderNodeFloatOp::PortType VisualShaderNodeFloatOp::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeFloatOp::get_input_port_name(int p_port) const {
	return p_port == 0 ? "a" : "b";
}

int VisualShaderNodeFloatOp::get_output_port_count() const {
	return 1;
}

VisualShaderNodeFloatOp::PortType VisualShaderNodeFloatOp::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeFloatOp::get_output_port_name(int p_port) const {
	return "op"; //no output port means the editor will be used as port
}

String VisualShaderNodeFloatOp::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code = "	" + p_output_vars[0] + " = ";
	switch (op) {
		case OP_ADD:
			code += p_input_vars[0] + " + " + p_input_vars[1] + ";\n";
			break;
		case OP_SUB:
			code += p_input_vars[0] + " - " + p_input_vars[1] + ";\n";
			break;
		case OP_MUL:
			code += p_input_vars[0] + " * " + p_input_vars[1] + ";\n";
			break;
		case OP_DIV:
			code += p_input_vars[0] + " / " + p_input_vars[1] + ";\n";
			break;
		case OP_MOD:
			code += "mod(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_POW:
			code += "pow(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_MAX:
			code += "max(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_MIN:
			code += "min(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_ATAN2:
			code += "atan(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_STEP:
			code += "step(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		default:
			break;
	}
	return code;
}

void VisualShaderNodeFloatOp::set_operator(Operator p_op) {
	ERR_FAIL_INDEX(int(p_op), int(OP_ENUM_SIZE));
	if (op == p_op) {
		return;
	}
	op = p_op;
	emit_changed();
}

VisualShaderNodeFloatOp::Operator VisualShaderNodeFloatOp::get_operator() const {
	return op;
}

Vector<StringName> VisualShaderNodeFloatOp::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("operator");
	return props;
}

void VisualShaderNodeFloatOp::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_operator", "op"), &VisualShaderNodeFloatOp::set_operator);
	ClassDB::bind_method(D_METHOD("get_operator"), &VisualShaderNodeFloatOp::get_operator);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operator", PROPERTY_HINT_ENUM, "Add,Subtract,Multiply,Divide,Remainder,Power,Max,Min,ATan2,Step"), "set_operator", "get_operator");

	BIND_ENUM_CONSTANT(OP_ADD);
	BIND_ENUM_CONSTANT(OP_SUB);
	BIND_ENUM_CONSTANT(OP_MUL);
	BIND_ENUM_CONSTANT(OP_DIV);
	BIND_ENUM_CONSTANT(OP_MOD);
	BIND_ENUM_CONSTANT(OP_POW);
	BIND_ENUM_CONSTANT(OP_MAX);
	BIND_ENUM_CONSTANT(OP_MIN);
	BIND_ENUM_CONSTANT(OP_ATAN2);
	BIND_ENUM_CONSTANT(OP_STEP);
	BIND_ENUM_CONSTANT(OP_ENUM_SIZE);
}

VisualShaderNodeFloatOp::VisualShaderNodeFloatOp() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
}

////////////// Integer Op

String VisualShaderNodeIntOp::get_caption() const {
	return "IntOp";
}

int VisualShaderNodeIntOp::get_input_port_count() const {
	return 2;
}

VisualShaderNodeIntOp::PortType VisualShaderNodeIntOp::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR_INT;
}

String VisualShaderNodeIntOp::get_input_port_name(int p_port) const {
	return p_port == 0 ? "a" : "b";
}

int VisualShaderNodeIntOp::get_output_port_count() const {
	return 1;
}

VisualShaderNodeIntOp::PortType VisualShaderNodeIntOp::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR_INT;
}

String VisualShaderNodeIntOp::get_output_port_name(int p_port) const {
	return "op"; //no output port means the editor will be used as port
}

String VisualShaderNodeIntOp::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code = "	" + p_output_vars[0] + " = ";
	switch (op) {
		case OP_ADD:
			code += p_input_vars[0] + " + " + p_input_vars[1] + ";\n";
			break;
		case OP_SUB:
			code += p_input_vars[0] + " - " + p_input_vars[1] + ";\n";
			break;
		case OP_MUL:
			code += p_input_vars[0] + " * " + p_input_vars[1] + ";\n";
			break;
		case OP_DIV:
			code += p_input_vars[0] + " / " + p_input_vars[1] + ";\n";
			break;
		case OP_MOD:
			code += p_input_vars[0] + " % " + p_input_vars[1] + ";\n";
			break;
		case OP_MAX:
			code += "max(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_MIN:
			code += "min(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_BITWISE_AND:
			code += p_input_vars[0] + " & " + p_input_vars[1] + ";\n";
			break;
		case OP_BITWISE_OR:
			code += p_input_vars[0] + " | " + p_input_vars[1] + ";\n";
			break;
		case OP_BITWISE_XOR:
			code += p_input_vars[0] + " ^ " + p_input_vars[1] + ";\n";
			break;
		case OP_BITWISE_LEFT_SHIFT:
			code += p_input_vars[0] + " << " + p_input_vars[1] + ";\n";
			break;
		case OP_BITWISE_RIGHT_SHIFT:
			code += p_input_vars[0] + " >> " + p_input_vars[1] + ";\n";
			break;
		default:
			break;
	}

	return code;
}

void VisualShaderNodeIntOp::set_operator(Operator p_op) {
	ERR_FAIL_INDEX(int(p_op), OP_ENUM_SIZE);
	if (op == p_op) {
		return;
	}
	op = p_op;
	emit_changed();
}

VisualShaderNodeIntOp::Operator VisualShaderNodeIntOp::get_operator() const {
	return op;
}

Vector<StringName> VisualShaderNodeIntOp::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("operator");
	return props;
}

void VisualShaderNodeIntOp::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_operator", "op"), &VisualShaderNodeIntOp::set_operator);
	ClassDB::bind_method(D_METHOD("get_operator"), &VisualShaderNodeIntOp::get_operator);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operator", PROPERTY_HINT_ENUM, "Add,Subtract,Multiply,Divide,Remainder,Max,Min,Bitwise AND,Bitwise OR,Bitwise XOR,Bitwise Left Shift,Bitwise Right Shift"), "set_operator", "get_operator");

	BIND_ENUM_CONSTANT(OP_ADD);
	BIND_ENUM_CONSTANT(OP_SUB);
	BIND_ENUM_CONSTANT(OP_MUL);
	BIND_ENUM_CONSTANT(OP_DIV);
	BIND_ENUM_CONSTANT(OP_MOD);
	BIND_ENUM_CONSTANT(OP_MAX);
	BIND_ENUM_CONSTANT(OP_MIN);
	BIND_ENUM_CONSTANT(OP_BITWISE_AND);
	BIND_ENUM_CONSTANT(OP_BITWISE_OR);
	BIND_ENUM_CONSTANT(OP_BITWISE_XOR);
	BIND_ENUM_CONSTANT(OP_BITWISE_LEFT_SHIFT);
	BIND_ENUM_CONSTANT(OP_BITWISE_RIGHT_SHIFT);
	BIND_ENUM_CONSTANT(OP_ENUM_SIZE);
}

VisualShaderNodeIntOp::VisualShaderNodeIntOp() {
	set_input_port_default_value(0, 0);
	set_input_port_default_value(1, 0);
}

////////////// Vector Op

String VisualShaderNodeVectorOp::get_caption() const {
	return "VectorOp";
}

int VisualShaderNodeVectorOp::get_input_port_count() const {
	return 2;
}

VisualShaderNodeVectorOp::PortType VisualShaderNodeVectorOp::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorOp::get_input_port_name(int p_port) const {
	return p_port == 0 ? "a" : "b";
}

int VisualShaderNodeVectorOp::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorOp::PortType VisualShaderNodeVectorOp::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorOp::get_output_port_name(int p_port) const {
	return "op"; //no output port means the editor will be used as port
}

String VisualShaderNodeVectorOp::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code = "	" + p_output_vars[0] + " = ";
	switch (op) {
		case OP_ADD:
			code += p_input_vars[0] + " + " + p_input_vars[1] + ";\n";
			break;
		case OP_SUB:
			code += p_input_vars[0] + " - " + p_input_vars[1] + ";\n";
			break;
		case OP_MUL:
			code += p_input_vars[0] + " * " + p_input_vars[1] + ";\n";
			break;
		case OP_DIV:
			code += p_input_vars[0] + " / " + p_input_vars[1] + ";\n";
			break;
		case OP_MOD:
			code += "mod(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_POW:
			code += "pow(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_MAX:
			code += "max(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_MIN:
			code += "min(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_CROSS:
			code += "cross(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_ATAN2:
			code += "atan(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_REFLECT:
			code += "reflect(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		case OP_STEP:
			code += "step(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
			break;
		default:
			break;
	}

	return code;
}

void VisualShaderNodeVectorOp::set_operator(Operator p_op) {
	ERR_FAIL_INDEX(int(p_op), int(OP_ENUM_SIZE));
	if (op == p_op) {
		return;
	}
	op = p_op;
	emit_changed();
}

VisualShaderNodeVectorOp::Operator VisualShaderNodeVectorOp::get_operator() const {
	return op;
}

Vector<StringName> VisualShaderNodeVectorOp::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("operator");
	return props;
}

void VisualShaderNodeVectorOp::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_operator", "op"), &VisualShaderNodeVectorOp::set_operator);
	ClassDB::bind_method(D_METHOD("get_operator"), &VisualShaderNodeVectorOp::get_operator);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operator", PROPERTY_HINT_ENUM, "Add,Subtract,Multiply,Divide,Remainder,Power,Max,Min,Cross,ATan2,Reflect,Step"), "set_operator", "get_operator");

	BIND_ENUM_CONSTANT(OP_ADD);
	BIND_ENUM_CONSTANT(OP_SUB);
	BIND_ENUM_CONSTANT(OP_MUL);
	BIND_ENUM_CONSTANT(OP_DIV);
	BIND_ENUM_CONSTANT(OP_MOD);
	BIND_ENUM_CONSTANT(OP_POW);
	BIND_ENUM_CONSTANT(OP_MAX);
	BIND_ENUM_CONSTANT(OP_MIN);
	BIND_ENUM_CONSTANT(OP_CROSS);
	BIND_ENUM_CONSTANT(OP_ATAN2);
	BIND_ENUM_CONSTANT(OP_REFLECT);
	BIND_ENUM_CONSTANT(OP_STEP);
	BIND_ENUM_CONSTANT(OP_ENUM_SIZE);
}

VisualShaderNodeVectorOp::VisualShaderNodeVectorOp() {
	set_input_port_default_value(0, Vector3());
	set_input_port_default_value(1, Vector3());
}

////////////// Color Op

String VisualShaderNodeColorOp::get_caption() const {
	return "ColorOp";
}

int VisualShaderNodeColorOp::get_input_port_count() const {
	return 2;
}

VisualShaderNodeColorOp::PortType VisualShaderNodeColorOp::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeColorOp::get_input_port_name(int p_port) const {
	return p_port == 0 ? "a" : "b";
}

int VisualShaderNodeColorOp::get_output_port_count() const {
	return 1;
}

VisualShaderNodeColorOp::PortType VisualShaderNodeColorOp::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeColorOp::get_output_port_name(int p_port) const {
	return "op"; //no output port means the editor will be used as port
}

String VisualShaderNodeColorOp::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	static const char *axisn[3] = { "x", "y", "z" };
	switch (op) {
		case OP_SCREEN: {
			code += "	" + p_output_vars[0] + " = vec3(1.0) - (vec3(1.0) - " + p_input_vars[0] + ") * (vec3(1.0) - " + p_input_vars[1] + ");\n";
		} break;
		case OP_DIFFERENCE: {
			code += "	" + p_output_vars[0] + " = abs(" + p_input_vars[0] + " - " + p_input_vars[1] + ");\n";
		} break;
		case OP_DARKEN: {
			code += "	" + p_output_vars[0] + " = min(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
		} break;
		case OP_LIGHTEN: {
			code += "	" + p_output_vars[0] + " = max(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";

		} break;
		case OP_OVERLAY: {
			for (int i = 0; i < 3; i++) {
				code += "	{\n";
				code += "		float base = " + p_input_vars[0] + "." + axisn[i] + ";\n";
				code += "		float blend = " + p_input_vars[1] + "." + axisn[i] + ";\n";
				code += "		if (base < 0.5) {\n";
				code += "			" + p_output_vars[0] + "." + axisn[i] + " = 2.0 * base * blend;\n";
				code += "		} else {\n";
				code += "			" + p_output_vars[0] + "." + axisn[i] + " = 1.0 - 2.0 * (1.0 - blend) * (1.0 - base);\n";
				code += "		}\n";
				code += "	}\n";
			}

		} break;
		case OP_DODGE: {
			code += "	" + p_output_vars[0] + " = (" + p_input_vars[0] + ") / (vec3(1.0) - " + p_input_vars[1] + ");\n";

		} break;
		case OP_BURN: {
			code += "	" + p_output_vars[0] + " = vec3(1.0) - (vec3(1.0) - " + p_input_vars[0] + ") / (" + p_input_vars[1] + ");\n";
		} break;
		case OP_SOFT_LIGHT: {
			for (int i = 0; i < 3; i++) {
				code += "	{\n";
				code += "		float base = " + p_input_vars[0] + "." + axisn[i] + ";\n";
				code += "		float blend = " + p_input_vars[1] + "." + axisn[i] + ";\n";
				code += "		if (base < 0.5) {\n";
				code += "			" + p_output_vars[0] + "." + axisn[i] + " = (base * (blend + 0.5));\n";
				code += "		} else {\n";
				code += "			" + p_output_vars[0] + "." + axisn[i] + " = (1.0 - (1.0 - base) * (1.0 - (blend - 0.5)));\n";
				code += "		}\n";
				code += "	}\n";
			}

		} break;
		case OP_HARD_LIGHT: {
			for (int i = 0; i < 3; i++) {
				code += "	{\n";
				code += "		float base = " + p_input_vars[0] + "." + axisn[i] + ";\n";
				code += "		float blend = " + p_input_vars[1] + "." + axisn[i] + ";\n";
				code += "		if (base < 0.5) {\n";
				code += "			" + p_output_vars[0] + "." + axisn[i] + " = (base * (2.0 * blend));\n";
				code += "		} else {\n";
				code += "			" + p_output_vars[0] + "." + axisn[i] + " = (1.0 - (1.0 - base) * (1.0 - 2.0 * (blend - 0.5)));\n";
				code += "		}\n";
				code += "	}\n";
			}

		} break;
		default:
			break;
	}

	return code;
}

void VisualShaderNodeColorOp::set_operator(Operator p_op) {
	ERR_FAIL_INDEX(int(p_op), int(OP_MAX));
	if (op == p_op) {
		return;
	}
	switch (p_op) {
		case OP_SCREEN:
			simple_decl = true;
			break;
		case OP_DIFFERENCE:
			simple_decl = true;
			break;
		case OP_DARKEN:
			simple_decl = true;
			break;
		case OP_LIGHTEN:
			simple_decl = true;
			break;
		case OP_OVERLAY:
			simple_decl = false;
			break;
		case OP_DODGE:
			simple_decl = true;
			break;
		case OP_BURN:
			simple_decl = true;
			break;
		case OP_SOFT_LIGHT:
			simple_decl = false;
			break;
		case OP_HARD_LIGHT:
			simple_decl = false;
			break;
		default:
			break;
	}
	op = p_op;
	emit_changed();
}

VisualShaderNodeColorOp::Operator VisualShaderNodeColorOp::get_operator() const {
	return op;
}

Vector<StringName> VisualShaderNodeColorOp::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("operator");
	return props;
}

void VisualShaderNodeColorOp::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_operator", "op"), &VisualShaderNodeColorOp::set_operator);
	ClassDB::bind_method(D_METHOD("get_operator"), &VisualShaderNodeColorOp::get_operator);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operator", PROPERTY_HINT_ENUM, "Screen,Difference,Darken,Lighten,Overlay,Dodge,Burn,Soft Light,Hard Light"), "set_operator", "get_operator");

	BIND_ENUM_CONSTANT(OP_SCREEN);
	BIND_ENUM_CONSTANT(OP_DIFFERENCE);
	BIND_ENUM_CONSTANT(OP_DARKEN);
	BIND_ENUM_CONSTANT(OP_LIGHTEN);
	BIND_ENUM_CONSTANT(OP_OVERLAY);
	BIND_ENUM_CONSTANT(OP_DODGE);
	BIND_ENUM_CONSTANT(OP_BURN);
	BIND_ENUM_CONSTANT(OP_SOFT_LIGHT);
	BIND_ENUM_CONSTANT(OP_HARD_LIGHT);
	BIND_ENUM_CONSTANT(OP_MAX);
}

VisualShaderNodeColorOp::VisualShaderNodeColorOp() {
	set_input_port_default_value(0, Vector3());
	set_input_port_default_value(1, Vector3());
}

////////////// Transform Op

String VisualShaderNodeTransformOp::get_caption() const {
	return "TransformOp";
}

int VisualShaderNodeTransformOp::get_input_port_count() const {
	return 2;
}

VisualShaderNodeTransformOp::PortType VisualShaderNodeTransformOp::get_input_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeTransformOp::get_input_port_name(int p_port) const {
	return p_port == 0 ? "a" : "b";
}

int VisualShaderNodeTransformOp::get_output_port_count() const {
	return 1;
}

VisualShaderNodeTransformOp::PortType VisualShaderNodeTransformOp::get_output_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeTransformOp::get_output_port_name(int p_port) const {
	return "mult"; //no output port means the editor will be used as port
}

String VisualShaderNodeTransformOp::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	switch (op) {
		case OP_AxB:
			return "	" + p_output_vars[0] + " = " + p_input_vars[0] + " * " + p_input_vars[1] + ";\n";
		case OP_BxA:
			return "	" + p_output_vars[0] + " = " + p_input_vars[1] + " * " + p_input_vars[0] + ";\n";
		case OP_AxB_COMP:
			return "	" + p_output_vars[0] + " = matrixCompMult(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
		case OP_BxA_COMP:
			return "	" + p_output_vars[0] + " = matrixCompMult(" + p_input_vars[1] + ", " + p_input_vars[0] + ");\n";
		case OP_ADD:
			return "	" + p_output_vars[0] + " = " + p_input_vars[0] + " + " + p_input_vars[1] + ";\n";
		case OP_A_MINUS_B:
			return "	" + p_output_vars[0] + " = " + p_input_vars[0] + " - " + p_input_vars[1] + ";\n";
		case OP_B_MINUS_A:
			return "	" + p_output_vars[0] + " = " + p_input_vars[1] + " - " + p_input_vars[0] + ";\n";
		case OP_A_DIV_B:
			return "	" + p_output_vars[0] + " = " + p_input_vars[0] + " / " + p_input_vars[1] + ";\n";
		case OP_B_DIV_A:
			return "	" + p_output_vars[0] + " = " + p_input_vars[1] + " / " + p_input_vars[0] + ";\n";
		default:
			return "";
	}
}

void VisualShaderNodeTransformOp::set_operator(Operator p_op) {
	ERR_FAIL_INDEX(int(p_op), int(OP_MAX));
	if (op == p_op) {
		return;
	}
	op = p_op;
	emit_changed();
}

VisualShaderNodeTransformOp::Operator VisualShaderNodeTransformOp::get_operator() const {
	return op;
}

Vector<StringName> VisualShaderNodeTransformOp::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("operator");
	return props;
}

void VisualShaderNodeTransformOp::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_operator", "op"), &VisualShaderNodeTransformOp::set_operator);
	ClassDB::bind_method(D_METHOD("get_operator"), &VisualShaderNodeTransformOp::get_operator);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operator", PROPERTY_HINT_ENUM, "A x B,B x A,A x B(per component),B x A(per component),A + B,A - B,B - A,A / B,B / A"), "set_operator", "get_operator");

	BIND_ENUM_CONSTANT(OP_AxB);
	BIND_ENUM_CONSTANT(OP_BxA);
	BIND_ENUM_CONSTANT(OP_AxB_COMP);
	BIND_ENUM_CONSTANT(OP_BxA_COMP);
	BIND_ENUM_CONSTANT(OP_ADD);
	BIND_ENUM_CONSTANT(OP_A_MINUS_B);
	BIND_ENUM_CONSTANT(OP_B_MINUS_A);
	BIND_ENUM_CONSTANT(OP_A_DIV_B);
	BIND_ENUM_CONSTANT(OP_B_DIV_A);
	BIND_ENUM_CONSTANT(OP_MAX);
}

VisualShaderNodeTransformOp::VisualShaderNodeTransformOp() {
	set_input_port_default_value(0, Transform3D());
	set_input_port_default_value(1, Transform3D());
}

////////////// TransformVec Mult

String VisualShaderNodeTransformVecMult::get_caption() const {
	return "TransformVectorMult";
}

int VisualShaderNodeTransformVecMult::get_input_port_count() const {
	return 2;
}

VisualShaderNodeTransformVecMult::PortType VisualShaderNodeTransformVecMult::get_input_port_type(int p_port) const {
	return p_port == 0 ? PORT_TYPE_TRANSFORM : PORT_TYPE_VECTOR;
}

String VisualShaderNodeTransformVecMult::get_input_port_name(int p_port) const {
	return p_port == 0 ? "a" : "b";
}

int VisualShaderNodeTransformVecMult::get_output_port_count() const {
	return 1;
}

VisualShaderNodeTransformVecMult::PortType VisualShaderNodeTransformVecMult::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeTransformVecMult::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeTransformVecMult::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	if (op == OP_AxB) {
		return "	" + p_output_vars[0] + " = (" + p_input_vars[0] + " * vec4(" + p_input_vars[1] + ", 1.0)).xyz;\n";
	} else if (op == OP_BxA) {
		return "	" + p_output_vars[0] + " = (vec4(" + p_input_vars[1] + ", 1.0) * " + p_input_vars[0] + ").xyz;\n";
	} else if (op == OP_3x3_AxB) {
		return "	" + p_output_vars[0] + " = (" + p_input_vars[0] + " * vec4(" + p_input_vars[1] + ", 0.0)).xyz;\n";
	} else {
		return "	" + p_output_vars[0] + " = (vec4(" + p_input_vars[1] + ", 0.0) * " + p_input_vars[0] + ").xyz;\n";
	}
}

void VisualShaderNodeTransformVecMult::set_operator(Operator p_op) {
	ERR_FAIL_INDEX(int(p_op), int(OP_MAX));
	if (op == p_op) {
		return;
	}
	op = p_op;
	emit_changed();
}

VisualShaderNodeTransformVecMult::Operator VisualShaderNodeTransformVecMult::get_operator() const {
	return op;
}

Vector<StringName> VisualShaderNodeTransformVecMult::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("operator");
	return props;
}

void VisualShaderNodeTransformVecMult::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_operator", "op"), &VisualShaderNodeTransformVecMult::set_operator);
	ClassDB::bind_method(D_METHOD("get_operator"), &VisualShaderNodeTransformVecMult::get_operator);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operator", PROPERTY_HINT_ENUM, "A x B,B x A,A x B (3x3),B x A (3x3)"), "set_operator", "get_operator");

	BIND_ENUM_CONSTANT(OP_AxB);
	BIND_ENUM_CONSTANT(OP_BxA);
	BIND_ENUM_CONSTANT(OP_3x3_AxB);
	BIND_ENUM_CONSTANT(OP_3x3_BxA);
	BIND_ENUM_CONSTANT(OP_MAX);
}

VisualShaderNodeTransformVecMult::VisualShaderNodeTransformVecMult() {
	set_input_port_default_value(0, Transform3D());
	set_input_port_default_value(1, Vector3());
}

////////////// Float Func

String VisualShaderNodeFloatFunc::get_caption() const {
	return "FloatFunc";
}

int VisualShaderNodeFloatFunc::get_input_port_count() const {
	return 1;
}

VisualShaderNodeFloatFunc::PortType VisualShaderNodeFloatFunc::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeFloatFunc::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeFloatFunc::get_output_port_count() const {
	return 1;
}

VisualShaderNodeFloatFunc::PortType VisualShaderNodeFloatFunc::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeFloatFunc::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeFloatFunc::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	static const char *functions[FUNC_MAX] = {
		"sin($)",
		"cos($)",
		"tan($)",
		"asin($)",
		"acos($)",
		"atan($)",
		"sinh($)",
		"cosh($)",
		"tanh($)",
		"log($)",
		"exp($)",
		"sqrt($)",
		"abs($)",
		"sign($)",
		"floor($)",
		"round($)",
		"ceil($)",
		"fract($)",
		"min(max($, 0.0), 1.0)",
		"-($)",
		"acosh($)",
		"asinh($)",
		"atanh($)",
		"degrees($)",
		"exp2($)",
		"inversesqrt($)",
		"log2($)",
		"radians($)",
		"1.0 / ($)",
		"roundEven($)",
		"trunc($)",
		"1.0 - $"
	};
	return "	" + p_output_vars[0] + " = " + String(functions[func]).replace("$", p_input_vars[0]) + ";\n";
}

void VisualShaderNodeFloatFunc::set_function(Function p_func) {
	ERR_FAIL_INDEX(int(p_func), int(FUNC_MAX));
	if (func == p_func) {
		return;
	}
	func = p_func;
	emit_changed();
}

VisualShaderNodeFloatFunc::Function VisualShaderNodeFloatFunc::get_function() const {
	return func;
}

Vector<StringName> VisualShaderNodeFloatFunc::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("function");
	return props;
}

void VisualShaderNodeFloatFunc::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_function", "func"), &VisualShaderNodeFloatFunc::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &VisualShaderNodeFloatFunc::get_function);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "function", PROPERTY_HINT_ENUM, "Sin,Cos,Tan,ASin,ACos,ATan,SinH,CosH,TanH,Log,Exp,Sqrt,Abs,Sign,Floor,Round,Ceil,Frac,Saturate,Negate,ACosH,ASinH,ATanH,Degrees,Exp2,InverseSqrt,Log2,Radians,Reciprocal,RoundEven,Trunc,OneMinus"), "set_function", "get_function");

	BIND_ENUM_CONSTANT(FUNC_SIN);
	BIND_ENUM_CONSTANT(FUNC_COS);
	BIND_ENUM_CONSTANT(FUNC_TAN);
	BIND_ENUM_CONSTANT(FUNC_ASIN);
	BIND_ENUM_CONSTANT(FUNC_ACOS);
	BIND_ENUM_CONSTANT(FUNC_ATAN);
	BIND_ENUM_CONSTANT(FUNC_SINH);
	BIND_ENUM_CONSTANT(FUNC_COSH);
	BIND_ENUM_CONSTANT(FUNC_TANH);
	BIND_ENUM_CONSTANT(FUNC_LOG);
	BIND_ENUM_CONSTANT(FUNC_EXP);
	BIND_ENUM_CONSTANT(FUNC_SQRT);
	BIND_ENUM_CONSTANT(FUNC_ABS);
	BIND_ENUM_CONSTANT(FUNC_SIGN);
	BIND_ENUM_CONSTANT(FUNC_FLOOR);
	BIND_ENUM_CONSTANT(FUNC_ROUND);
	BIND_ENUM_CONSTANT(FUNC_CEIL);
	BIND_ENUM_CONSTANT(FUNC_FRAC);
	BIND_ENUM_CONSTANT(FUNC_SATURATE);
	BIND_ENUM_CONSTANT(FUNC_NEGATE);
	BIND_ENUM_CONSTANT(FUNC_ACOSH);
	BIND_ENUM_CONSTANT(FUNC_ASINH);
	BIND_ENUM_CONSTANT(FUNC_ATANH);
	BIND_ENUM_CONSTANT(FUNC_DEGREES);
	BIND_ENUM_CONSTANT(FUNC_EXP2);
	BIND_ENUM_CONSTANT(FUNC_INVERSE_SQRT);
	BIND_ENUM_CONSTANT(FUNC_LOG2);
	BIND_ENUM_CONSTANT(FUNC_RADIANS);
	BIND_ENUM_CONSTANT(FUNC_RECIPROCAL);
	BIND_ENUM_CONSTANT(FUNC_ROUNDEVEN);
	BIND_ENUM_CONSTANT(FUNC_TRUNC);
	BIND_ENUM_CONSTANT(FUNC_ONEMINUS);
	BIND_ENUM_CONSTANT(FUNC_MAX);
}

VisualShaderNodeFloatFunc::VisualShaderNodeFloatFunc() {
	set_input_port_default_value(0, 0.0);
}

////////////// Int Func

String VisualShaderNodeIntFunc::get_caption() const {
	return "IntFunc";
}

int VisualShaderNodeIntFunc::get_input_port_count() const {
	return 1;
}

VisualShaderNodeIntFunc::PortType VisualShaderNodeIntFunc::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR_INT;
}

String VisualShaderNodeIntFunc::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeIntFunc::get_output_port_count() const {
	return 1;
}

VisualShaderNodeIntFunc::PortType VisualShaderNodeIntFunc::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR_INT;
}

String VisualShaderNodeIntFunc::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeIntFunc::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	static const char *functions[FUNC_MAX] = {
		"abs($)",
		"-($)",
		"sign($)",
		"~($)"
	};

	return "	" + p_output_vars[0] + " = " + String(functions[func]).replace("$", p_input_vars[0]) + ";\n";
}

void VisualShaderNodeIntFunc::set_function(Function p_func) {
	ERR_FAIL_INDEX(int(p_func), int(FUNC_MAX));
	if (func == p_func) {
		return;
	}
	func = p_func;
	emit_changed();
}

VisualShaderNodeIntFunc::Function VisualShaderNodeIntFunc::get_function() const {
	return func;
}

Vector<StringName> VisualShaderNodeIntFunc::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("function");
	return props;
}

void VisualShaderNodeIntFunc::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_function", "func"), &VisualShaderNodeIntFunc::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &VisualShaderNodeIntFunc::get_function);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "function", PROPERTY_HINT_ENUM, "Abs,Negate,Sign,Bitwise NOT"), "set_function", "get_function");

	BIND_ENUM_CONSTANT(FUNC_ABS);
	BIND_ENUM_CONSTANT(FUNC_NEGATE);
	BIND_ENUM_CONSTANT(FUNC_SIGN);
	BIND_ENUM_CONSTANT(FUNC_BITWISE_NOT);
	BIND_ENUM_CONSTANT(FUNC_MAX);
}

VisualShaderNodeIntFunc::VisualShaderNodeIntFunc() {
	set_input_port_default_value(0, 0);
}

////////////// Vector Func

String VisualShaderNodeVectorFunc::get_caption() const {
	return "VectorFunc";
}

int VisualShaderNodeVectorFunc::get_input_port_count() const {
	return 1;
}

VisualShaderNodeVectorFunc::PortType VisualShaderNodeVectorFunc::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorFunc::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeVectorFunc::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorFunc::PortType VisualShaderNodeVectorFunc::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorFunc::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeVectorFunc::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	static const char *vec_func_id[FUNC_MAX] = {
		"normalize($)",
		"max(min($, vec3(1.0)), vec3(0.0))",
		"-($)",
		"1.0 / ($)",
		"",
		"",
		"abs($)",
		"acos($)",
		"acosh($)",
		"asin($)",
		"asinh($)",
		"atan($)",
		"atanh($)",
		"ceil($)",
		"cos($)",
		"cosh($)",
		"degrees($)",
		"exp($)",
		"exp2($)",
		"floor($)",
		"fract($)",
		"inversesqrt($)",
		"log($)",
		"log2($)",
		"radians($)",
		"round($)",
		"roundEven($)",
		"sign($)",
		"sin($)",
		"sinh($)",
		"sqrt($)",
		"tan($)",
		"tanh($)",
		"trunc($)",
		"vec3(1.0, 1.0, 1.0) - $"
	};

	String code;

	if (func == FUNC_RGB2HSV) {
		code += "	{\n";
		code += "		vec3 c = " + p_input_vars[0] + ";\n";
		code += "		vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);\n";
		code += "		vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));\n";
		code += "		vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));\n";
		code += "		float d = q.x - min(q.w, q.y);\n";
		code += "		float e = 1.0e-10;\n";
		code += "		" + p_output_vars[0] + " = vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);\n";
		code += "	}\n";
	} else if (func == FUNC_HSV2RGB) {
		code += "	{\n";
		code += "		vec3 c = " + p_input_vars[0] + ";\n";
		code += "		vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);\n";
		code += "		vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);\n";
		code += "		" + p_output_vars[0] + " = c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);\n";
		code += "	}\n";

	} else {
		code += "	" + p_output_vars[0] + " = " + String(vec_func_id[func]).replace("$", p_input_vars[0]) + ";\n";
	}

	return code;
}

void VisualShaderNodeVectorFunc::set_function(Function p_func) {
	ERR_FAIL_INDEX(int(p_func), int(FUNC_MAX));
	if (func == p_func) {
		return;
	}
	if (p_func == FUNC_RGB2HSV) {
		simple_decl = false;
	} else if (p_func == FUNC_HSV2RGB) {
		simple_decl = false;
	} else {
		simple_decl = true;
	}
	func = p_func;
	emit_changed();
}

VisualShaderNodeVectorFunc::Function VisualShaderNodeVectorFunc::get_function() const {
	return func;
}

Vector<StringName> VisualShaderNodeVectorFunc::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("function");
	return props;
}

void VisualShaderNodeVectorFunc::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_function", "func"), &VisualShaderNodeVectorFunc::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &VisualShaderNodeVectorFunc::get_function);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "function", PROPERTY_HINT_ENUM, "Normalize,Saturate,Negate,Reciprocal,RGB2HSV,HSV2RGB,Abs,ACos,ACosH,ASin,ASinH,ATan,ATanH,Ceil,Cos,CosH,Degrees,Exp,Exp2,Floor,Frac,InverseSqrt,Log,Log2,Radians,Round,RoundEven,Sign,Sin,SinH,Sqrt,Tan,TanH,Trunc,OneMinus"), "set_function", "get_function");

	BIND_ENUM_CONSTANT(FUNC_NORMALIZE);
	BIND_ENUM_CONSTANT(FUNC_SATURATE);
	BIND_ENUM_CONSTANT(FUNC_NEGATE);
	BIND_ENUM_CONSTANT(FUNC_RECIPROCAL);
	BIND_ENUM_CONSTANT(FUNC_RGB2HSV);
	BIND_ENUM_CONSTANT(FUNC_HSV2RGB);
	BIND_ENUM_CONSTANT(FUNC_ABS);
	BIND_ENUM_CONSTANT(FUNC_ACOS);
	BIND_ENUM_CONSTANT(FUNC_ACOSH);
	BIND_ENUM_CONSTANT(FUNC_ASIN);
	BIND_ENUM_CONSTANT(FUNC_ASINH);
	BIND_ENUM_CONSTANT(FUNC_ATAN);
	BIND_ENUM_CONSTANT(FUNC_ATANH);
	BIND_ENUM_CONSTANT(FUNC_CEIL);
	BIND_ENUM_CONSTANT(FUNC_COS);
	BIND_ENUM_CONSTANT(FUNC_COSH);
	BIND_ENUM_CONSTANT(FUNC_DEGREES);
	BIND_ENUM_CONSTANT(FUNC_EXP);
	BIND_ENUM_CONSTANT(FUNC_EXP2);
	BIND_ENUM_CONSTANT(FUNC_FLOOR);
	BIND_ENUM_CONSTANT(FUNC_FRAC);
	BIND_ENUM_CONSTANT(FUNC_INVERSE_SQRT);
	BIND_ENUM_CONSTANT(FUNC_LOG);
	BIND_ENUM_CONSTANT(FUNC_LOG2);
	BIND_ENUM_CONSTANT(FUNC_RADIANS);
	BIND_ENUM_CONSTANT(FUNC_ROUND);
	BIND_ENUM_CONSTANT(FUNC_ROUNDEVEN);
	BIND_ENUM_CONSTANT(FUNC_SIGN);
	BIND_ENUM_CONSTANT(FUNC_SIN);
	BIND_ENUM_CONSTANT(FUNC_SINH);
	BIND_ENUM_CONSTANT(FUNC_SQRT);
	BIND_ENUM_CONSTANT(FUNC_TAN);
	BIND_ENUM_CONSTANT(FUNC_TANH);
	BIND_ENUM_CONSTANT(FUNC_TRUNC);
	BIND_ENUM_CONSTANT(FUNC_ONEMINUS);
	BIND_ENUM_CONSTANT(FUNC_MAX);
}

VisualShaderNodeVectorFunc::VisualShaderNodeVectorFunc() {
	set_input_port_default_value(0, Vector3());
}

////////////// ColorFunc

String VisualShaderNodeColorFunc::get_caption() const {
	return "ColorFunc";
}

int VisualShaderNodeColorFunc::get_input_port_count() const {
	return 1;
}

VisualShaderNodeColorFunc::PortType VisualShaderNodeColorFunc::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeColorFunc::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeColorFunc::get_output_port_count() const {
	return 1;
}

VisualShaderNodeColorFunc::PortType VisualShaderNodeColorFunc::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeColorFunc::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeColorFunc::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;

	switch (func) {
		case FUNC_GRAYSCALE:
			code += "	{\n";
			code += "		vec3 c = " + p_input_vars[0] + ";\n";
			code += "		float max1 = max(c.r, c.g);\n";
			code += "		float max2 = max(max1, c.b);\n";
			code += "		float max3 = max(max1, max2);\n";
			code += "		" + p_output_vars[0] + " = vec3(max3, max3, max3);\n";
			code += "	}\n";
			break;
		case FUNC_SEPIA:
			code += "	{\n";
			code += "		vec3 c = " + p_input_vars[0] + ";\n";
			code += "		float r = (c.r * .393) + (c.g *.769) + (c.b * .189);\n";
			code += "		float g = (c.r * .349) + (c.g *.686) + (c.b * .168);\n";
			code += "		float b = (c.r * .272) + (c.g *.534) + (c.b * .131);\n";
			code += "		" + p_output_vars[0] + " = vec3(r, g, b);\n";
			code += "	}\n";
			break;
		default:
			break;
	}

	return code;
}

void VisualShaderNodeColorFunc::set_function(Function p_func) {
	ERR_FAIL_INDEX(int(p_func), int(FUNC_MAX));
	if (func == p_func) {
		return;
	}
	func = p_func;
	emit_changed();
}

VisualShaderNodeColorFunc::Function VisualShaderNodeColorFunc::get_function() const {
	return func;
}

Vector<StringName> VisualShaderNodeColorFunc::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("function");
	return props;
}

void VisualShaderNodeColorFunc::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_function", "func"), &VisualShaderNodeColorFunc::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &VisualShaderNodeColorFunc::get_function);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "function", PROPERTY_HINT_ENUM, "Grayscale,Sepia"), "set_function", "get_function");

	BIND_ENUM_CONSTANT(FUNC_GRAYSCALE);
	BIND_ENUM_CONSTANT(FUNC_SEPIA);
	BIND_ENUM_CONSTANT(FUNC_MAX);
}

VisualShaderNodeColorFunc::VisualShaderNodeColorFunc() {
	simple_decl = false;
	set_input_port_default_value(0, Vector3());
}

////////////// Transform Func

String VisualShaderNodeTransformFunc::get_caption() const {
	return "TransformFunc";
}

int VisualShaderNodeTransformFunc::get_input_port_count() const {
	return 1;
}

VisualShaderNodeTransformFunc::PortType VisualShaderNodeTransformFunc::get_input_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeTransformFunc::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeTransformFunc::get_output_port_count() const {
	return 1;
}

VisualShaderNodeTransformFunc::PortType VisualShaderNodeTransformFunc::get_output_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeTransformFunc::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeTransformFunc::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	static const char *functions[FUNC_MAX] = {
		"inverse($)",
		"transpose($)"
	};

	String code;
	code += "	" + p_output_vars[0] + " = " + String(functions[func]).replace("$", p_input_vars[0]) + ";\n";
	return code;
}

void VisualShaderNodeTransformFunc::set_function(Function p_func) {
	ERR_FAIL_INDEX(int(p_func), int(FUNC_MAX));
	if (func == p_func) {
		return;
	}
	func = p_func;
	emit_changed();
}

VisualShaderNodeTransformFunc::Function VisualShaderNodeTransformFunc::get_function() const {
	return func;
}

Vector<StringName> VisualShaderNodeTransformFunc::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("function");
	return props;
}

void VisualShaderNodeTransformFunc::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_function", "func"), &VisualShaderNodeTransformFunc::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &VisualShaderNodeTransformFunc::get_function);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "function", PROPERTY_HINT_ENUM, "Inverse,Transpose"), "set_function", "get_function");

	BIND_ENUM_CONSTANT(FUNC_INVERSE);
	BIND_ENUM_CONSTANT(FUNC_TRANSPOSE);
	BIND_ENUM_CONSTANT(FUNC_MAX);
}

VisualShaderNodeTransformFunc::VisualShaderNodeTransformFunc() {
	set_input_port_default_value(0, Transform3D());
}

////////////// UV Func

String VisualShaderNodeUVFunc::get_caption() const {
	return "UVFunc";
}

int VisualShaderNodeUVFunc::get_input_port_count() const {
	return 3;
}

VisualShaderNodeUVFunc::PortType VisualShaderNodeUVFunc::get_input_port_type(int p_port) const {
	switch (p_port) {
		case 0:
			[[fallthrough]]; // uv
		case 1:
			return PORT_TYPE_VECTOR; // scale
		case 2:
			return PORT_TYPE_VECTOR; // offset & pivot
		default:
			break;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeUVFunc::get_input_port_name(int p_port) const {
	switch (p_port) {
		case 0:
			return "uv";
		case 1:
			return "scale";
		case 2:
			switch (func) {
				case FUNC_PANNING:
					return "offset";
				case FUNC_SCALING:
					return "pivot";
				default:
					break;
			}
			break;
		default:
			break;
	}
	return "";
}

String VisualShaderNodeUVFunc::get_input_port_default_hint(int p_port) const {
	if (p_port == 0) {
		return "UV";
	}
	return "";
}

int VisualShaderNodeUVFunc::get_output_port_count() const {
	return 1;
}

VisualShaderNodeUVFunc::PortType VisualShaderNodeUVFunc::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeUVFunc::get_output_port_name(int p_port) const {
	return "uv";
}

bool VisualShaderNodeUVFunc::is_show_prop_names() const {
	return true;
}

String VisualShaderNodeUVFunc::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;

	String uv;
	if (p_input_vars[0].is_empty()) {
		uv = "vec3(UV.xy, 0.0)";
	} else {
		uv = vformat("%s", p_input_vars[0]);
	}
	String scale = vformat("%s", p_input_vars[1]);
	String offset_pivot = vformat("%s", p_input_vars[2]);

	switch (func) {
		case FUNC_PANNING: {
			code += vformat("	%s = fma(%s, %s, %s);\n", p_output_vars[0], offset_pivot, scale, uv);
		} break;
		case FUNC_SCALING: {
			code += vformat("	%s = fma((%s - %s), %s, %s);\n", p_output_vars[0], uv, offset_pivot, scale, offset_pivot);
		} break;
		default:
			break;
	}
	return code;
}

void VisualShaderNodeUVFunc::set_function(VisualShaderNodeUVFunc::Function p_func) {
	ERR_FAIL_INDEX(int(p_func), int(FUNC_MAX));
	if (func == p_func) {
		return;
	}
	if (p_func == FUNC_PANNING) {
		set_input_port_default_value(2, Vector3()); // offset
	} else { // FUNC_SCALING
		set_input_port_default_value(2, Vector3(0.5, 0.5, 0.0)); // pivot
	}
	func = p_func;
	emit_changed();
}

VisualShaderNodeUVFunc::Function VisualShaderNodeUVFunc::get_function() const {
	return func;
}

Vector<StringName> VisualShaderNodeUVFunc::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("function");
	return props;
}

void VisualShaderNodeUVFunc::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_function", "func"), &VisualShaderNodeUVFunc::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &VisualShaderNodeUVFunc::get_function);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "function", PROPERTY_HINT_ENUM, "Panning,Scaling"), "set_function", "get_function");

	BIND_ENUM_CONSTANT(FUNC_PANNING);
	BIND_ENUM_CONSTANT(FUNC_SCALING);
	BIND_ENUM_CONSTANT(FUNC_MAX);
}

VisualShaderNodeUVFunc::VisualShaderNodeUVFunc() {
	set_input_port_default_value(1, Vector3(1.0, 1.0, 0.0)); // scale
	set_input_port_default_value(2, Vector3()); // offset
}

////////////// Dot Product

String VisualShaderNodeDotProduct::get_caption() const {
	return "DotProduct";
}

int VisualShaderNodeDotProduct::get_input_port_count() const {
	return 2;
}

VisualShaderNodeDotProduct::PortType VisualShaderNodeDotProduct::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeDotProduct::get_input_port_name(int p_port) const {
	return p_port == 0 ? "a" : "b";
}

int VisualShaderNodeDotProduct::get_output_port_count() const {
	return 1;
}

VisualShaderNodeDotProduct::PortType VisualShaderNodeDotProduct::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeDotProduct::get_output_port_name(int p_port) const {
	return "dot";
}

String VisualShaderNodeDotProduct::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = dot(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
}

VisualShaderNodeDotProduct::VisualShaderNodeDotProduct() {
	set_input_port_default_value(0, Vector3());
	set_input_port_default_value(1, Vector3());
}

////////////// Vector Len

String VisualShaderNodeVectorLen::get_caption() const {
	return "VectorLen";
}

int VisualShaderNodeVectorLen::get_input_port_count() const {
	return 1;
}

VisualShaderNodeVectorLen::PortType VisualShaderNodeVectorLen::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorLen::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeVectorLen::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorLen::PortType VisualShaderNodeVectorLen::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeVectorLen::get_output_port_name(int p_port) const {
	return "length";
}

String VisualShaderNodeVectorLen::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = length(" + p_input_vars[0] + ");\n";
}

VisualShaderNodeVectorLen::VisualShaderNodeVectorLen() {
	set_input_port_default_value(0, Vector3());
}

////////////// Determinant

String VisualShaderNodeDeterminant::get_caption() const {
	return "Determinant";
}

int VisualShaderNodeDeterminant::get_input_port_count() const {
	return 1;
}

VisualShaderNodeDeterminant::PortType VisualShaderNodeDeterminant::get_input_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeDeterminant::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeDeterminant::get_output_port_count() const {
	return 1;
}

VisualShaderNodeDeterminant::PortType VisualShaderNodeDeterminant::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeDeterminant::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeDeterminant::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = determinant(" + p_input_vars[0] + ");\n";
}

VisualShaderNodeDeterminant::VisualShaderNodeDeterminant() {
	set_input_port_default_value(0, Transform3D());
}

////////////// Scalar Derivative Function

String VisualShaderNodeScalarDerivativeFunc::get_caption() const {
	return "ScalarDerivativeFunc";
}

int VisualShaderNodeScalarDerivativeFunc::get_input_port_count() const {
	return 1;
}

VisualShaderNodeScalarDerivativeFunc::PortType VisualShaderNodeScalarDerivativeFunc::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarDerivativeFunc::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeScalarDerivativeFunc::get_output_port_count() const {
	return 1;
}

VisualShaderNodeScalarDerivativeFunc::PortType VisualShaderNodeScalarDerivativeFunc::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarDerivativeFunc::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeScalarDerivativeFunc::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	static const char *functions[FUNC_MAX] = {
		"fwidth($)",
		"dFdx($)",
		"dFdy($)"
	};

	String code;
	code += "	" + p_output_vars[0] + " = " + String(functions[func]).replace("$", p_input_vars[0]) + ";\n";
	return code;
}

void VisualShaderNodeScalarDerivativeFunc::set_function(Function p_func) {
	ERR_FAIL_INDEX(int(p_func), int(FUNC_MAX));
	if (func == p_func) {
		return;
	}
	func = p_func;
	emit_changed();
}

VisualShaderNodeScalarDerivativeFunc::Function VisualShaderNodeScalarDerivativeFunc::get_function() const {
	return func;
}

Vector<StringName> VisualShaderNodeScalarDerivativeFunc::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("function");
	return props;
}

void VisualShaderNodeScalarDerivativeFunc::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_function", "func"), &VisualShaderNodeScalarDerivativeFunc::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &VisualShaderNodeScalarDerivativeFunc::get_function);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "function", PROPERTY_HINT_ENUM, "Sum,X,Y"), "set_function", "get_function");

	BIND_ENUM_CONSTANT(FUNC_SUM);
	BIND_ENUM_CONSTANT(FUNC_X);
	BIND_ENUM_CONSTANT(FUNC_Y);
	BIND_ENUM_CONSTANT(FUNC_MAX);
}

VisualShaderNodeScalarDerivativeFunc::VisualShaderNodeScalarDerivativeFunc() {
	set_input_port_default_value(0, 0.0);
}

////////////// Vector Derivative Function

String VisualShaderNodeVectorDerivativeFunc::get_caption() const {
	return "VectorDerivativeFunc";
}

int VisualShaderNodeVectorDerivativeFunc::get_input_port_count() const {
	return 1;
}

VisualShaderNodeVectorDerivativeFunc::PortType VisualShaderNodeVectorDerivativeFunc::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorDerivativeFunc::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeVectorDerivativeFunc::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorDerivativeFunc::PortType VisualShaderNodeVectorDerivativeFunc::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorDerivativeFunc::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeVectorDerivativeFunc::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	static const char *functions[FUNC_MAX] = {
		"fwidth($)",
		"dFdx($)",
		"dFdy($)"
	};

	String code;
	code += "	" + p_output_vars[0] + " = " + String(functions[func]).replace("$", p_input_vars[0]) + ";\n";
	return code;
}

void VisualShaderNodeVectorDerivativeFunc::set_function(Function p_func) {
	ERR_FAIL_INDEX(int(p_func), int(FUNC_MAX));
	if (func == p_func) {
		return;
	}
	func = p_func;
	emit_changed();
}

VisualShaderNodeVectorDerivativeFunc::Function VisualShaderNodeVectorDerivativeFunc::get_function() const {
	return func;
}

Vector<StringName> VisualShaderNodeVectorDerivativeFunc::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("function");
	return props;
}

void VisualShaderNodeVectorDerivativeFunc::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_function", "func"), &VisualShaderNodeVectorDerivativeFunc::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &VisualShaderNodeVectorDerivativeFunc::get_function);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "function", PROPERTY_HINT_ENUM, "Sum,X,Y"), "set_function", "get_function");

	BIND_ENUM_CONSTANT(FUNC_SUM);
	BIND_ENUM_CONSTANT(FUNC_X);
	BIND_ENUM_CONSTANT(FUNC_Y);
	BIND_ENUM_CONSTANT(FUNC_MAX);
}

VisualShaderNodeVectorDerivativeFunc::VisualShaderNodeVectorDerivativeFunc() {
	set_input_port_default_value(0, Vector3());
}

////////////// Clamp

String VisualShaderNodeClamp::get_caption() const {
	return "Clamp";
}

int VisualShaderNodeClamp::get_input_port_count() const {
	return 3;
}

VisualShaderNodeClamp::PortType VisualShaderNodeClamp::get_input_port_type(int p_port) const {
	switch (op_type) {
		case OP_TYPE_INT:
			return PORT_TYPE_SCALAR_INT;
		case OP_TYPE_VECTOR:
			return PORT_TYPE_VECTOR;
		default:
			break;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeClamp::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "";
	} else if (p_port == 1) {
		return "min";
	} else if (p_port == 2) {
		return "max";
	}
	return "";
}

int VisualShaderNodeClamp::get_output_port_count() const {
	return 1;
}

VisualShaderNodeClamp::PortType VisualShaderNodeClamp::get_output_port_type(int p_port) const {
	switch (op_type) {
		case OP_TYPE_INT:
			return PORT_TYPE_SCALAR_INT;
		case OP_TYPE_VECTOR:
			return PORT_TYPE_VECTOR;
		default:
			break;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeClamp::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeClamp::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = clamp(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

void VisualShaderNodeClamp::set_op_type(OpType p_op_type) {
	ERR_FAIL_INDEX((int)p_op_type, int(OP_TYPE_MAX));
	if (op_type == p_op_type) {
		return;
	}
	switch (p_op_type) {
		case OP_TYPE_FLOAT:
			set_input_port_default_value(0, 0.0);
			set_input_port_default_value(1, 0.0);
			set_input_port_default_value(2, 0.0);
			break;
		case OP_TYPE_INT:
			set_input_port_default_value(0, 0);
			set_input_port_default_value(1, 0);
			set_input_port_default_value(2, 0);
			break;
		case OP_TYPE_VECTOR:
			set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0));
			set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0));
			set_input_port_default_value(2, Vector3(0.0, 0.0, 0.0));
			break;
		default:
			break;
	}
	op_type = p_op_type;
	emit_changed();
}

VisualShaderNodeClamp::OpType VisualShaderNodeClamp::get_op_type() const {
	return op_type;
}

Vector<StringName> VisualShaderNodeClamp::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("op_type");
	return props;
}

void VisualShaderNodeClamp::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_op_type", "op_type"), &VisualShaderNodeClamp::set_op_type);
	ClassDB::bind_method(D_METHOD("get_op_type"), &VisualShaderNodeClamp::get_op_type);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "op_type", PROPERTY_HINT_ENUM, "Float,Int,Vector"), "set_op_type", "get_op_type");

	BIND_ENUM_CONSTANT(OP_TYPE_FLOAT);
	BIND_ENUM_CONSTANT(OP_TYPE_INT);
	BIND_ENUM_CONSTANT(OP_TYPE_VECTOR);
	BIND_ENUM_CONSTANT(OP_TYPE_MAX);
}

VisualShaderNodeClamp::VisualShaderNodeClamp() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, 1.0);
}

////////////// FaceForward

String VisualShaderNodeFaceForward::get_caption() const {
	return "FaceForward";
}

int VisualShaderNodeFaceForward::get_input_port_count() const {
	return 3;
}

VisualShaderNodeFaceForward::PortType VisualShaderNodeFaceForward::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeFaceForward::get_input_port_name(int p_port) const {
	switch (p_port) {
		case 0:
			return "N";
		case 1:
			return "I";
		case 2:
			return "Nref";
		default:
			return "";
	}
}

int VisualShaderNodeFaceForward::get_output_port_count() const {
	return 1;
}

VisualShaderNodeFaceForward::PortType VisualShaderNodeFaceForward::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeFaceForward::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeFaceForward::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = faceforward(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

VisualShaderNodeFaceForward::VisualShaderNodeFaceForward() {
	set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(2, Vector3(0.0, 0.0, 0.0));
}

////////////// Outer Product

String VisualShaderNodeOuterProduct::get_caption() const {
	return "OuterProduct";
}

int VisualShaderNodeOuterProduct::get_input_port_count() const {
	return 2;
}

VisualShaderNodeOuterProduct::PortType VisualShaderNodeOuterProduct::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeOuterProduct::get_input_port_name(int p_port) const {
	switch (p_port) {
		case 0:
			return "c";
		case 1:
			return "r";
		default:
			return "";
	}
}

int VisualShaderNodeOuterProduct::get_output_port_count() const {
	return 1;
}

VisualShaderNodeOuterProduct::PortType VisualShaderNodeOuterProduct::get_output_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeOuterProduct::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeOuterProduct::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = outerProduct(vec4(" + p_input_vars[0] + ", 0.0), vec4(" + p_input_vars[1] + ", 0.0));\n";
}

VisualShaderNodeOuterProduct::VisualShaderNodeOuterProduct() {
	set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0));
}

////////////// Step

String VisualShaderNodeStep::get_caption() const {
	return "Step";
}

int VisualShaderNodeStep::get_input_port_count() const {
	return 2;
}

VisualShaderNodeStep::PortType VisualShaderNodeStep::get_input_port_type(int p_port) const {
	switch (op_type) {
		case OP_TYPE_VECTOR:
			return PORT_TYPE_VECTOR;
		case OP_TYPE_VECTOR_SCALAR:
			if (p_port == 1) {
				return PORT_TYPE_VECTOR;
			}
			break;
		default:
			break;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeStep::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "edge";
	} else if (p_port == 1) {
		return "x";
	}
	return "";
}

int VisualShaderNodeStep::get_output_port_count() const {
	return 1;
}

VisualShaderNodeStep::PortType VisualShaderNodeStep::get_output_port_type(int p_port) const {
	switch (op_type) {
		case OP_TYPE_VECTOR:
			return PORT_TYPE_VECTOR;
		case OP_TYPE_VECTOR_SCALAR:
			return PORT_TYPE_VECTOR;
		default:
			break;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeStep::get_output_port_name(int p_port) const {
	return "";
}

void VisualShaderNodeStep::set_op_type(OpType p_op_type) {
	ERR_FAIL_INDEX(int(p_op_type), int(OP_TYPE_MAX));
	if (op_type == p_op_type) {
		return;
	}
	switch (p_op_type) {
		case OP_TYPE_SCALAR:
			if (op_type == OP_TYPE_VECTOR) {
				set_input_port_default_value(0, 0.0); // edge
			}
			if (op_type == OP_TYPE_VECTOR || op_type == OP_TYPE_VECTOR_SCALAR) {
				set_input_port_default_value(1, 0.0); // x
			}
			break;
		case OP_TYPE_VECTOR:
			if (op_type == OP_TYPE_SCALAR || op_type == OP_TYPE_VECTOR_SCALAR) {
				set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0)); // edge
			}
			if (op_type == OP_TYPE_SCALAR) {
				set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0)); // x
			}
			break;
		case OP_TYPE_VECTOR_SCALAR:
			if (op_type == OP_TYPE_VECTOR) {
				set_input_port_default_value(0, 0.0); // edge
			}
			if (op_type == OP_TYPE_SCALAR) {
				set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0)); // x
			}
			break;
		default:
			break;
	}
	op_type = p_op_type;
	emit_changed();
}

VisualShaderNodeStep::OpType VisualShaderNodeStep::get_op_type() const {
	return op_type;
}

String VisualShaderNodeStep::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = step(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
}

Vector<StringName> VisualShaderNodeStep::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("op_type");
	return props;
}

void VisualShaderNodeStep::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_op_type", "op_type"), &VisualShaderNodeStep::set_op_type);
	ClassDB::bind_method(D_METHOD("get_op_type"), &VisualShaderNodeStep::get_op_type);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "op_type", PROPERTY_HINT_ENUM, "Scalar,Vector,VectorScalar"), "set_op_type", "get_op_type");

	BIND_ENUM_CONSTANT(OP_TYPE_SCALAR);
	BIND_ENUM_CONSTANT(OP_TYPE_VECTOR);
	BIND_ENUM_CONSTANT(OP_TYPE_VECTOR_SCALAR);
	BIND_ENUM_CONSTANT(OP_TYPE_MAX);
}

VisualShaderNodeStep::VisualShaderNodeStep() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
}

////////////// SmoothStep

String VisualShaderNodeSmoothStep::get_caption() const {
	return "SmoothStep";
}

int VisualShaderNodeSmoothStep::get_input_port_count() const {
	return 3;
}

VisualShaderNodeSmoothStep::PortType VisualShaderNodeSmoothStep::get_input_port_type(int p_port) const {
	switch (op_type) {
		case OP_TYPE_VECTOR:
			return PORT_TYPE_VECTOR;
		case OP_TYPE_VECTOR_SCALAR:
			if (p_port == 2) {
				return PORT_TYPE_VECTOR; // x
			}
			break;
		default:
			break;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeSmoothStep::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "edge0";
	} else if (p_port == 1) {
		return "edge1";
	} else if (p_port == 2) {
		return "x";
	}
	return "";
}

int VisualShaderNodeSmoothStep::get_output_port_count() const {
	return 1;
}

VisualShaderNodeSmoothStep::PortType VisualShaderNodeSmoothStep::get_output_port_type(int p_port) const {
	switch (op_type) {
		case OP_TYPE_VECTOR:
			return PORT_TYPE_VECTOR;
		case OP_TYPE_VECTOR_SCALAR:
			return PORT_TYPE_VECTOR;
		default:
			break;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeSmoothStep::get_output_port_name(int p_port) const {
	return "";
}

void VisualShaderNodeSmoothStep::set_op_type(OpType p_op_type) {
	ERR_FAIL_INDEX(int(p_op_type), int(OP_TYPE_MAX));
	if (op_type == p_op_type) {
		return;
	}
	switch (p_op_type) {
		case OP_TYPE_SCALAR:
			if (op_type == OP_TYPE_VECTOR) {
				set_input_port_default_value(0, 0.0); // edge0
				set_input_port_default_value(1, 0.0); // edge1
			}
			if (op_type == OP_TYPE_VECTOR || op_type == OP_TYPE_VECTOR_SCALAR) {
				set_input_port_default_value(2, 0.0); // x
			}
			break;
		case OP_TYPE_VECTOR:
			if (op_type == OP_TYPE_SCALAR || op_type == OP_TYPE_VECTOR_SCALAR) {
				set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0)); // edge0
				set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0)); // edge1
			}
			if (op_type == OP_TYPE_SCALAR) {
				set_input_port_default_value(2, Vector3(0.0, 0.0, 0.0)); // x
			}
			break;
		case OP_TYPE_VECTOR_SCALAR:
			if (op_type == OP_TYPE_VECTOR) {
				set_input_port_default_value(0, 0.0); // edge0
				set_input_port_default_value(1, 0.0); // edge1
			}
			if (op_type == OP_TYPE_SCALAR) {
				set_input_port_default_value(2, Vector3(0.0, 0.0, 0.0)); // x
			}
			break;
		default:
			break;
	}
	op_type = p_op_type;
	emit_changed();
}

VisualShaderNodeSmoothStep::OpType VisualShaderNodeSmoothStep::get_op_type() const {
	return op_type;
}

String VisualShaderNodeSmoothStep::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = smoothstep(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

Vector<StringName> VisualShaderNodeSmoothStep::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("op_type");
	return props;
}

void VisualShaderNodeSmoothStep::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_op_type", "op_type"), &VisualShaderNodeSmoothStep::set_op_type);
	ClassDB::bind_method(D_METHOD("get_op_type"), &VisualShaderNodeSmoothStep::get_op_type);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "op_type", PROPERTY_HINT_ENUM, "Scalar,Vector,VectorScalar"), "set_op_type", "get_op_type");

	BIND_ENUM_CONSTANT(OP_TYPE_SCALAR);
	BIND_ENUM_CONSTANT(OP_TYPE_VECTOR);
	BIND_ENUM_CONSTANT(OP_TYPE_VECTOR_SCALAR);
	BIND_ENUM_CONSTANT(OP_TYPE_MAX);
}

VisualShaderNodeSmoothStep::VisualShaderNodeSmoothStep() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, 0.0);
}

////////////// Distance

String VisualShaderNodeVectorDistance::get_caption() const {
	return "Distance";
}

int VisualShaderNodeVectorDistance::get_input_port_count() const {
	return 2;
}

VisualShaderNodeVectorDistance::PortType VisualShaderNodeVectorDistance::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorDistance::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "p0";
	} else if (p_port == 1) {
		return "p1";
	}
	return "";
}

int VisualShaderNodeVectorDistance::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorDistance::PortType VisualShaderNodeVectorDistance::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeVectorDistance::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeVectorDistance::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = distance(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
}

VisualShaderNodeVectorDistance::VisualShaderNodeVectorDistance() {
	set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0));
}

////////////// Refract Vector

String VisualShaderNodeVectorRefract::get_caption() const {
	return "Refract";
}

int VisualShaderNodeVectorRefract::get_input_port_count() const {
	return 3;
}

VisualShaderNodeVectorRefract::PortType VisualShaderNodeVectorRefract::get_input_port_type(int p_port) const {
	if (p_port == 2) {
		return PORT_TYPE_SCALAR;
	}

	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorRefract::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "I";
	} else if (p_port == 1) {
		return "N";
	} else if (p_port == 2) {
		return "eta";
	}
	return "";
}

int VisualShaderNodeVectorRefract::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorRefract::PortType VisualShaderNodeVectorRefract::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorRefract::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeVectorRefract::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = refract(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

VisualShaderNodeVectorRefract::VisualShaderNodeVectorRefract() {
	set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(2, 0.0);
}

////////////// Mix

String VisualShaderNodeMix::get_caption() const {
	return "Mix";
}

int VisualShaderNodeMix::get_input_port_count() const {
	return 3;
}

VisualShaderNodeMix::PortType VisualShaderNodeMix::get_input_port_type(int p_port) const {
	switch (op_type) {
		case OP_TYPE_VECTOR:
			return PORT_TYPE_VECTOR;
		case OP_TYPE_VECTOR_SCALAR:
			if (p_port == 2) {
				break;
			}
			return PORT_TYPE_VECTOR;
		default:
			break;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeMix::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "a";
	} else if (p_port == 1) {
		return "b";
	} else {
		return "weight";
	}
}

int VisualShaderNodeMix::get_output_port_count() const {
	return 1;
}

VisualShaderNodeMix::PortType VisualShaderNodeMix::get_output_port_type(int p_port) const {
	switch (op_type) {
		case OP_TYPE_VECTOR:
			return PORT_TYPE_VECTOR;
		case OP_TYPE_VECTOR_SCALAR:
			return PORT_TYPE_VECTOR;
		default:
			break;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeMix::get_output_port_name(int p_port) const {
	return "mix";
}

void VisualShaderNodeMix::set_op_type(OpType p_op_type) {
	ERR_FAIL_INDEX(int(p_op_type), int(OP_TYPE_MAX));
	if (op_type == p_op_type) {
		return;
	}
	switch (p_op_type) {
		case OP_TYPE_SCALAR:
			set_input_port_default_value(0, 0.0); // a
			set_input_port_default_value(1, 1.0); // b
			if (op_type == OP_TYPE_VECTOR) {
				set_input_port_default_value(2, 0.5); // weight
			}
			break;
		case OP_TYPE_VECTOR:
			set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0)); // a
			set_input_port_default_value(1, Vector3(1.0, 1.0, 1.0)); // b
			if (op_type == OP_TYPE_SCALAR || op_type == OP_TYPE_VECTOR_SCALAR) {
				set_input_port_default_value(2, Vector3(0.5, 0.5, 0.5)); // weight
			}
			break;
		case OP_TYPE_VECTOR_SCALAR:
			set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0)); // a
			set_input_port_default_value(1, Vector3(1.0, 1.0, 1.0)); // b
			if (op_type == OP_TYPE_VECTOR) {
				set_input_port_default_value(2, 0.5); // weight
			}
			break;
		default:
			break;
	}
	op_type = p_op_type;
	emit_changed();
}

VisualShaderNodeMix::OpType VisualShaderNodeMix::get_op_type() const {
	return op_type;
}

String VisualShaderNodeMix::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = mix(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

Vector<StringName> VisualShaderNodeMix::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("op_type");
	return props;
}

void VisualShaderNodeMix::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_op_type", "op_type"), &VisualShaderNodeMix::set_op_type);
	ClassDB::bind_method(D_METHOD("get_op_type"), &VisualShaderNodeMix::get_op_type);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "op_type", PROPERTY_HINT_ENUM, "Scalar,Vector,VectorScalar"), "set_op_type", "get_op_type");

	BIND_ENUM_CONSTANT(OP_TYPE_SCALAR);
	BIND_ENUM_CONSTANT(OP_TYPE_VECTOR);
	BIND_ENUM_CONSTANT(OP_TYPE_VECTOR_SCALAR);
	BIND_ENUM_CONSTANT(OP_TYPE_MAX);
}

VisualShaderNodeMix::VisualShaderNodeMix() {
	set_input_port_default_value(0, 0.0); // a
	set_input_port_default_value(1, 1.0); // b
	set_input_port_default_value(2, 0.5); // weight
}

////////////// Vector Compose

String VisualShaderNodeVectorCompose::get_caption() const {
	return "VectorCompose";
}

int VisualShaderNodeVectorCompose::get_input_port_count() const {
	return 3;
}

VisualShaderNodeVectorCompose::PortType VisualShaderNodeVectorCompose::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeVectorCompose::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "x";
	} else if (p_port == 1) {
		return "y";
	} else {
		return "z";
	}
}

int VisualShaderNodeVectorCompose::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorCompose::PortType VisualShaderNodeVectorCompose::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorCompose::get_output_port_name(int p_port) const {
	return "vec";
}

String VisualShaderNodeVectorCompose::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = vec3(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

VisualShaderNodeVectorCompose::VisualShaderNodeVectorCompose() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, 0.0);
}

////////////// Transform Compose

String VisualShaderNodeTransformCompose::get_caption() const {
	return "TransformCompose";
}

int VisualShaderNodeTransformCompose::get_input_port_count() const {
	return 4;
}

VisualShaderNodeTransformCompose::PortType VisualShaderNodeTransformCompose::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeTransformCompose::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "x";
	} else if (p_port == 1) {
		return "y";
	} else if (p_port == 2) {
		return "z";
	} else {
		return "origin";
	}
}

int VisualShaderNodeTransformCompose::get_output_port_count() const {
	return 1;
}

VisualShaderNodeTransformCompose::PortType VisualShaderNodeTransformCompose::get_output_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeTransformCompose::get_output_port_name(int p_port) const {
	return "xform";
}

String VisualShaderNodeTransformCompose::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = mat4(vec4(" + p_input_vars[0] + ", 0.0), vec4(" + p_input_vars[1] + ", 0.0), vec4(" + p_input_vars[2] + ", 0.0), vec4(" + p_input_vars[3] + ", 1.0));\n";
}

VisualShaderNodeTransformCompose::VisualShaderNodeTransformCompose() {
	set_input_port_default_value(0, Vector3());
	set_input_port_default_value(1, Vector3());
	set_input_port_default_value(2, Vector3());
	set_input_port_default_value(3, Vector3());
}

////////////// Vector Decompose
String VisualShaderNodeVectorDecompose::get_caption() const {
	return "VectorDecompose";
}

int VisualShaderNodeVectorDecompose::get_input_port_count() const {
	return 1;
}

VisualShaderNodeVectorDecompose::PortType VisualShaderNodeVectorDecompose::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorDecompose::get_input_port_name(int p_port) const {
	return "vec";
}

int VisualShaderNodeVectorDecompose::get_output_port_count() const {
	return 3;
}

VisualShaderNodeVectorDecompose::PortType VisualShaderNodeVectorDecompose::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeVectorDecompose::get_output_port_name(int p_port) const {
	if (p_port == 0) {
		return "x";
	} else if (p_port == 1) {
		return "y";
	} else {
		return "z";
	}
}

String VisualShaderNodeVectorDecompose::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	code += "	" + p_output_vars[0] + " = " + p_input_vars[0] + ".x;\n";
	code += "	" + p_output_vars[1] + " = " + p_input_vars[0] + ".y;\n";
	code += "	" + p_output_vars[2] + " = " + p_input_vars[0] + ".z;\n";
	return code;
}

VisualShaderNodeVectorDecompose::VisualShaderNodeVectorDecompose() {
	set_input_port_default_value(0, Vector3());
}

////////////// Transform Decompose

String VisualShaderNodeTransformDecompose::get_caption() const {
	return "TransformDecompose";
}

int VisualShaderNodeTransformDecompose::get_input_port_count() const {
	return 1;
}

VisualShaderNodeTransformDecompose::PortType VisualShaderNodeTransformDecompose::get_input_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeTransformDecompose::get_input_port_name(int p_port) const {
	return "xform";
}

int VisualShaderNodeTransformDecompose::get_output_port_count() const {
	return 4;
}

VisualShaderNodeTransformDecompose::PortType VisualShaderNodeTransformDecompose::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeTransformDecompose::get_output_port_name(int p_port) const {
	if (p_port == 0) {
		return "x";
	} else if (p_port == 1) {
		return "y";
	} else if (p_port == 2) {
		return "z";
	} else {
		return "origin";
	}
}

String VisualShaderNodeTransformDecompose::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	code += "	" + p_output_vars[0] + " = " + p_input_vars[0] + "[0].xyz;\n";
	code += "	" + p_output_vars[1] + " = " + p_input_vars[0] + "[1].xyz;\n";
	code += "	" + p_output_vars[2] + " = " + p_input_vars[0] + "[2].xyz;\n";
	code += "	" + p_output_vars[3] + " = " + p_input_vars[0] + "[3].xyz;\n";
	return code;
}

VisualShaderNodeTransformDecompose::VisualShaderNodeTransformDecompose() {
	set_input_port_default_value(0, Transform3D());
}

////////////// Float Uniform

String VisualShaderNodeFloatUniform::get_caption() const {
	return "FloatUniform";
}

int VisualShaderNodeFloatUniform::get_input_port_count() const {
	return 0;
}

VisualShaderNodeFloatUniform::PortType VisualShaderNodeFloatUniform::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeFloatUniform::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeFloatUniform::get_output_port_count() const {
	return 1;
}

VisualShaderNodeFloatUniform::PortType VisualShaderNodeFloatUniform::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeFloatUniform::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeFloatUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = "";
	if (hint == HINT_RANGE) {
		code += _get_qual_str() + "uniform float " + get_uniform_name() + " : hint_range(" + rtos(hint_range_min) + ", " + rtos(hint_range_max) + ")";
	} else if (hint == HINT_RANGE_STEP) {
		code += _get_qual_str() + "uniform float " + get_uniform_name() + " : hint_range(" + rtos(hint_range_min) + ", " + rtos(hint_range_max) + ", " + rtos(hint_range_step) + ")";
	} else {
		code += _get_qual_str() + "uniform float " + get_uniform_name();
	}
	if (default_value_enabled) {
		code += " = " + rtos(default_value);
	}
	code += ";\n";
	return code;
}

String VisualShaderNodeFloatUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
}

bool VisualShaderNodeFloatUniform::is_show_prop_names() const {
	return true;
}

bool VisualShaderNodeFloatUniform::is_use_prop_slots() const {
	return true;
}

void VisualShaderNodeFloatUniform::set_hint(Hint p_hint) {
	ERR_FAIL_INDEX(int(p_hint), int(HINT_MAX));
	if (hint == p_hint) {
		return;
	}
	hint = p_hint;
	emit_changed();
}

VisualShaderNodeFloatUniform::Hint VisualShaderNodeFloatUniform::get_hint() const {
	return hint;
}

void VisualShaderNodeFloatUniform::set_min(float p_value) {
	if (Math::is_equal_approx(hint_range_min, p_value)) {
		return;
	}
	hint_range_min = p_value;
	emit_changed();
}

float VisualShaderNodeFloatUniform::get_min() const {
	return hint_range_min;
}

void VisualShaderNodeFloatUniform::set_max(float p_value) {
	if (Math::is_equal_approx(hint_range_max, p_value)) {
		return;
	}
	hint_range_max = p_value;
	emit_changed();
}

float VisualShaderNodeFloatUniform::get_max() const {
	return hint_range_max;
}

void VisualShaderNodeFloatUniform::set_step(float p_value) {
	if (Math::is_equal_approx(hint_range_step, p_value)) {
		return;
	}
	hint_range_step = p_value;
	emit_changed();
}

float VisualShaderNodeFloatUniform::get_step() const {
	return hint_range_step;
}

void VisualShaderNodeFloatUniform::set_default_value_enabled(bool p_enabled) {
	if (default_value_enabled == p_enabled) {
		return;
	}
	default_value_enabled = p_enabled;
	emit_changed();
}

bool VisualShaderNodeFloatUniform::is_default_value_enabled() const {
	return default_value_enabled;
}

void VisualShaderNodeFloatUniform::set_default_value(float p_value) {
	if (Math::is_equal_approx(default_value, p_value)) {
		return;
	}
	default_value = p_value;
	emit_changed();
}

float VisualShaderNodeFloatUniform::get_default_value() const {
	return default_value;
}

void VisualShaderNodeFloatUniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_hint", "hint"), &VisualShaderNodeFloatUniform::set_hint);
	ClassDB::bind_method(D_METHOD("get_hint"), &VisualShaderNodeFloatUniform::get_hint);

	ClassDB::bind_method(D_METHOD("set_min", "value"), &VisualShaderNodeFloatUniform::set_min);
	ClassDB::bind_method(D_METHOD("get_min"), &VisualShaderNodeFloatUniform::get_min);

	ClassDB::bind_method(D_METHOD("set_max", "value"), &VisualShaderNodeFloatUniform::set_max);
	ClassDB::bind_method(D_METHOD("get_max"), &VisualShaderNodeFloatUniform::get_max);

	ClassDB::bind_method(D_METHOD("set_step", "value"), &VisualShaderNodeFloatUniform::set_step);
	ClassDB::bind_method(D_METHOD("get_step"), &VisualShaderNodeFloatUniform::get_step);

	ClassDB::bind_method(D_METHOD("set_default_value_enabled", "enabled"), &VisualShaderNodeFloatUniform::set_default_value_enabled);
	ClassDB::bind_method(D_METHOD("is_default_value_enabled"), &VisualShaderNodeFloatUniform::is_default_value_enabled);

	ClassDB::bind_method(D_METHOD("set_default_value", "value"), &VisualShaderNodeFloatUniform::set_default_value);
	ClassDB::bind_method(D_METHOD("get_default_value"), &VisualShaderNodeFloatUniform::get_default_value);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "hint", PROPERTY_HINT_ENUM, "None,Range,Range+Step"), "set_hint", "get_hint");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min"), "set_min", "get_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max"), "set_max", "get_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "step"), "set_step", "get_step");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value_enabled"), "set_default_value_enabled", "is_default_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_value"), "set_default_value", "get_default_value");

	BIND_ENUM_CONSTANT(HINT_NONE);
	BIND_ENUM_CONSTANT(HINT_RANGE);
	BIND_ENUM_CONSTANT(HINT_RANGE_STEP);
	BIND_ENUM_CONSTANT(HINT_MAX);
}

bool VisualShaderNodeFloatUniform::is_qualifier_supported(Qualifier p_qual) const {
	return true; // all qualifiers are supported
}

bool VisualShaderNodeFloatUniform::is_convertible_to_constant() const {
	return true; // conversion is allowed
}

Vector<StringName> VisualShaderNodeFloatUniform::get_editable_properties() const {
	Vector<StringName> props = VisualShaderNodeUniform::get_editable_properties();
	props.push_back("hint");
	if (hint == HINT_RANGE || hint == HINT_RANGE_STEP) {
		props.push_back("min");
		props.push_back("max");
	}
	if (hint == HINT_RANGE_STEP) {
		props.push_back("step");
	}
	props.push_back("default_value_enabled");
	if (default_value_enabled) {
		props.push_back("default_value");
	}
	return props;
}

VisualShaderNodeFloatUniform::VisualShaderNodeFloatUniform() {
}

////////////// Integer Uniform

String VisualShaderNodeIntUniform::get_caption() const {
	return "IntUniform";
}

int VisualShaderNodeIntUniform::get_input_port_count() const {
	return 0;
}

VisualShaderNodeIntUniform::PortType VisualShaderNodeIntUniform::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR_INT;
}

String VisualShaderNodeIntUniform::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeIntUniform::get_output_port_count() const {
	return 1;
}

VisualShaderNodeIntUniform::PortType VisualShaderNodeIntUniform::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR_INT;
}

String VisualShaderNodeIntUniform::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeIntUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = "";
	if (hint == HINT_RANGE) {
		code += _get_qual_str() + "uniform int " + get_uniform_name() + " : hint_range(" + itos(hint_range_min) + ", " + itos(hint_range_max) + ")";
	} else if (hint == HINT_RANGE_STEP) {
		code += _get_qual_str() + "uniform int " + get_uniform_name() + " : hint_range(" + itos(hint_range_min) + ", " + itos(hint_range_max) + ", " + itos(hint_range_step) + ")";
	} else {
		code += _get_qual_str() + "uniform int " + get_uniform_name();
	}
	if (default_value_enabled) {
		code += " = " + itos(default_value);
	}
	code += ";\n";
	return code;
}

String VisualShaderNodeIntUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
}

bool VisualShaderNodeIntUniform::is_show_prop_names() const {
	return true;
}

bool VisualShaderNodeIntUniform::is_use_prop_slots() const {
	return true;
}

void VisualShaderNodeIntUniform::set_hint(Hint p_hint) {
	ERR_FAIL_INDEX(int(p_hint), int(HINT_MAX));
	if (hint == p_hint) {
		return;
	}
	hint = p_hint;
	emit_changed();
}

VisualShaderNodeIntUniform::Hint VisualShaderNodeIntUniform::get_hint() const {
	return hint;
}

void VisualShaderNodeIntUniform::set_min(int p_value) {
	if (hint_range_min == p_value) {
		return;
	}
	hint_range_min = p_value;
	emit_changed();
}

int VisualShaderNodeIntUniform::get_min() const {
	return hint_range_min;
}

void VisualShaderNodeIntUniform::set_max(int p_value) {
	if (hint_range_max == p_value) {
		return;
	}
	hint_range_max = p_value;
	emit_changed();
}

int VisualShaderNodeIntUniform::get_max() const {
	return hint_range_max;
}

void VisualShaderNodeIntUniform::set_step(int p_value) {
	if (hint_range_step == p_value) {
		return;
	}
	hint_range_step = p_value;
	emit_changed();
}

int VisualShaderNodeIntUniform::get_step() const {
	return hint_range_step;
}

void VisualShaderNodeIntUniform::set_default_value_enabled(bool p_default_value_enabled) {
	if (default_value_enabled == p_default_value_enabled) {
		return;
	}
	default_value_enabled = p_default_value_enabled;
	emit_changed();
}

bool VisualShaderNodeIntUniform::is_default_value_enabled() const {
	return default_value_enabled;
}

void VisualShaderNodeIntUniform::set_default_value(int p_default_value) {
	if (default_value == p_default_value) {
		return;
	}
	default_value = p_default_value;
	emit_changed();
}

int VisualShaderNodeIntUniform::get_default_value() const {
	return default_value;
}

void VisualShaderNodeIntUniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_hint", "hint"), &VisualShaderNodeIntUniform::set_hint);
	ClassDB::bind_method(D_METHOD("get_hint"), &VisualShaderNodeIntUniform::get_hint);

	ClassDB::bind_method(D_METHOD("set_min", "value"), &VisualShaderNodeIntUniform::set_min);
	ClassDB::bind_method(D_METHOD("get_min"), &VisualShaderNodeIntUniform::get_min);

	ClassDB::bind_method(D_METHOD("set_max", "value"), &VisualShaderNodeIntUniform::set_max);
	ClassDB::bind_method(D_METHOD("get_max"), &VisualShaderNodeIntUniform::get_max);

	ClassDB::bind_method(D_METHOD("set_step", "value"), &VisualShaderNodeIntUniform::set_step);
	ClassDB::bind_method(D_METHOD("get_step"), &VisualShaderNodeIntUniform::get_step);

	ClassDB::bind_method(D_METHOD("set_default_value_enabled", "enabled"), &VisualShaderNodeIntUniform::set_default_value_enabled);
	ClassDB::bind_method(D_METHOD("is_default_value_enabled"), &VisualShaderNodeIntUniform::is_default_value_enabled);

	ClassDB::bind_method(D_METHOD("set_default_value", "value"), &VisualShaderNodeIntUniform::set_default_value);
	ClassDB::bind_method(D_METHOD("get_default_value"), &VisualShaderNodeIntUniform::get_default_value);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "hint", PROPERTY_HINT_ENUM, "None,Range,Range + Step"), "set_hint", "get_hint");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "min"), "set_min", "get_min");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max"), "set_max", "get_max");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "step"), "set_step", "get_step");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value_enabled"), "set_default_value_enabled", "is_default_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "default_value"), "set_default_value", "get_default_value");

	BIND_ENUM_CONSTANT(HINT_NONE);
	BIND_ENUM_CONSTANT(HINT_RANGE);
	BIND_ENUM_CONSTANT(HINT_RANGE_STEP);
	BIND_ENUM_CONSTANT(HINT_MAX);
}

bool VisualShaderNodeIntUniform::is_qualifier_supported(Qualifier p_qual) const {
	return true; // all qualifiers are supported
}

bool VisualShaderNodeIntUniform::is_convertible_to_constant() const {
	return true; // conversion is allowed
}

Vector<StringName> VisualShaderNodeIntUniform::get_editable_properties() const {
	Vector<StringName> props = VisualShaderNodeUniform::get_editable_properties();
	props.push_back("hint");
	if (hint == HINT_RANGE || hint == HINT_RANGE_STEP) {
		props.push_back("min");
		props.push_back("max");
	}
	if (hint == HINT_RANGE_STEP) {
		props.push_back("step");
	}
	props.push_back("default_value_enabled");
	if (default_value_enabled) {
		props.push_back("default_value");
	}
	return props;
}

VisualShaderNodeIntUniform::VisualShaderNodeIntUniform() {
}

////////////// Boolean Uniform

String VisualShaderNodeBooleanUniform::get_caption() const {
	return "BooleanUniform";
}

int VisualShaderNodeBooleanUniform::get_input_port_count() const {
	return 0;
}

VisualShaderNodeBooleanUniform::PortType VisualShaderNodeBooleanUniform::get_input_port_type(int p_port) const {
	return PORT_TYPE_BOOLEAN;
}

String VisualShaderNodeBooleanUniform::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeBooleanUniform::get_output_port_count() const {
	return 1;
}

VisualShaderNodeBooleanUniform::PortType VisualShaderNodeBooleanUniform::get_output_port_type(int p_port) const {
	return PORT_TYPE_BOOLEAN;
}

String VisualShaderNodeBooleanUniform::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

void VisualShaderNodeBooleanUniform::set_default_value_enabled(bool p_default_value_enabled) {
	if (default_value_enabled == p_default_value_enabled) {
		return;
	}
	default_value_enabled = p_default_value_enabled;
	emit_changed();
}

bool VisualShaderNodeBooleanUniform::is_default_value_enabled() const {
	return default_value_enabled;
}

void VisualShaderNodeBooleanUniform::set_default_value(bool p_default_value) {
	if (default_value == p_default_value) {
		return;
	}
	default_value = p_default_value;
	emit_changed();
}

bool VisualShaderNodeBooleanUniform::get_default_value() const {
	return default_value;
}

String VisualShaderNodeBooleanUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = _get_qual_str() + "uniform bool " + get_uniform_name();
	if (default_value_enabled) {
		if (default_value) {
			code += " = true";
		} else {
			code += " = false";
		}
	}
	code += ";\n";
	return code;
}

String VisualShaderNodeBooleanUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
}

bool VisualShaderNodeBooleanUniform::is_show_prop_names() const {
	return true;
}

bool VisualShaderNodeBooleanUniform::is_use_prop_slots() const {
	return true;
}

void VisualShaderNodeBooleanUniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_default_value_enabled", "enabled"), &VisualShaderNodeBooleanUniform::set_default_value_enabled);
	ClassDB::bind_method(D_METHOD("is_default_value_enabled"), &VisualShaderNodeBooleanUniform::is_default_value_enabled);

	ClassDB::bind_method(D_METHOD("set_default_value", "value"), &VisualShaderNodeBooleanUniform::set_default_value);
	ClassDB::bind_method(D_METHOD("get_default_value"), &VisualShaderNodeBooleanUniform::get_default_value);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value_enabled"), "set_default_value_enabled", "is_default_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value"), "set_default_value", "get_default_value");
}

bool VisualShaderNodeBooleanUniform::is_qualifier_supported(Qualifier p_qual) const {
	return true; // all qualifiers are supported
}

bool VisualShaderNodeBooleanUniform::is_convertible_to_constant() const {
	return true; // conversion is allowed
}

Vector<StringName> VisualShaderNodeBooleanUniform::get_editable_properties() const {
	Vector<StringName> props = VisualShaderNodeUniform::get_editable_properties();
	props.push_back("default_value_enabled");
	if (default_value_enabled) {
		props.push_back("default_value");
	}
	return props;
}

VisualShaderNodeBooleanUniform::VisualShaderNodeBooleanUniform() {
}

////////////// Color Uniform

String VisualShaderNodeColorUniform::get_caption() const {
	return "ColorUniform";
}

int VisualShaderNodeColorUniform::get_input_port_count() const {
	return 0;
}

VisualShaderNodeColorUniform::PortType VisualShaderNodeColorUniform::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeColorUniform::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeColorUniform::get_output_port_count() const {
	return 2;
}

VisualShaderNodeColorUniform::PortType VisualShaderNodeColorUniform::get_output_port_type(int p_port) const {
	return p_port == 0 ? PORT_TYPE_VECTOR : PORT_TYPE_SCALAR;
}

String VisualShaderNodeColorUniform::get_output_port_name(int p_port) const {
	return p_port == 0 ? "color" : "alpha"; //no output port means the editor will be used as port
}

void VisualShaderNodeColorUniform::set_default_value_enabled(bool p_enabled) {
	if (default_value_enabled == p_enabled) {
		return;
	}
	default_value_enabled = p_enabled;
	emit_changed();
}

bool VisualShaderNodeColorUniform::is_default_value_enabled() const {
	return default_value_enabled;
}

void VisualShaderNodeColorUniform::set_default_value(const Color &p_value) {
	if (default_value.is_equal_approx(p_value)) {
		return;
	}
	default_value = p_value;
	emit_changed();
}

Color VisualShaderNodeColorUniform::get_default_value() const {
	return default_value;
}

String VisualShaderNodeColorUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = _get_qual_str() + "uniform vec4 " + get_uniform_name() + " : hint_color";
	if (default_value_enabled) {
		code += vformat(" = vec4(%.6f, %.6f, %.6f, %.6f)", default_value.r, default_value.g, default_value.b, default_value.a);
	}
	code += ";\n";
	return code;
}

String VisualShaderNodeColorUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code = "	" + p_output_vars[0] + " = " + get_uniform_name() + ".rgb;\n";
	code += "	" + p_output_vars[1] + " = " + get_uniform_name() + ".a;\n";
	return code;
}

bool VisualShaderNodeColorUniform::is_show_prop_names() const {
	return true;
}

void VisualShaderNodeColorUniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_default_value_enabled", "enabled"), &VisualShaderNodeColorUniform::set_default_value_enabled);
	ClassDB::bind_method(D_METHOD("is_default_value_enabled"), &VisualShaderNodeColorUniform::is_default_value_enabled);

	ClassDB::bind_method(D_METHOD("set_default_value", "value"), &VisualShaderNodeColorUniform::set_default_value);
	ClassDB::bind_method(D_METHOD("get_default_value"), &VisualShaderNodeColorUniform::get_default_value);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value_enabled"), "set_default_value_enabled", "is_default_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "default_value"), "set_default_value", "get_default_value");
}

bool VisualShaderNodeColorUniform::is_qualifier_supported(Qualifier p_qual) const {
	return true; // all qualifiers are supported
}

bool VisualShaderNodeColorUniform::is_convertible_to_constant() const {
	return true; // conversion is allowed
}

Vector<StringName> VisualShaderNodeColorUniform::get_editable_properties() const {
	Vector<StringName> props = VisualShaderNodeUniform::get_editable_properties();
	props.push_back("default_value_enabled");
	if (default_value_enabled) {
		props.push_back("default_value");
	}
	return props;
}

VisualShaderNodeColorUniform::VisualShaderNodeColorUniform() {
}

////////////// Vector Uniform

String VisualShaderNodeVec3Uniform::get_caption() const {
	return "VectorUniform";
}

int VisualShaderNodeVec3Uniform::get_input_port_count() const {
	return 0;
}

VisualShaderNodeVec3Uniform::PortType VisualShaderNodeVec3Uniform::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVec3Uniform::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeVec3Uniform::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVec3Uniform::PortType VisualShaderNodeVec3Uniform::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVec3Uniform::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

void VisualShaderNodeVec3Uniform::set_default_value_enabled(bool p_enabled) {
	default_value_enabled = p_enabled;
	emit_changed();
}

bool VisualShaderNodeVec3Uniform::is_default_value_enabled() const {
	return default_value_enabled;
}

void VisualShaderNodeVec3Uniform::set_default_value(const Vector3 &p_value) {
	default_value = p_value;
	emit_changed();
}

Vector3 VisualShaderNodeVec3Uniform::get_default_value() const {
	return default_value;
}

String VisualShaderNodeVec3Uniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = _get_qual_str() + "uniform vec3 " + get_uniform_name();
	if (default_value_enabled) {
		code += vformat(" = vec3(%.6f, %.6f, %.6f)", default_value.x, default_value.y, default_value.z);
	}
	code += ";\n";
	return code;
}

String VisualShaderNodeVec3Uniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
}

void VisualShaderNodeVec3Uniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_default_value_enabled", "enabled"), &VisualShaderNodeVec3Uniform::set_default_value_enabled);
	ClassDB::bind_method(D_METHOD("is_default_value_enabled"), &VisualShaderNodeVec3Uniform::is_default_value_enabled);

	ClassDB::bind_method(D_METHOD("set_default_value", "value"), &VisualShaderNodeVec3Uniform::set_default_value);
	ClassDB::bind_method(D_METHOD("get_default_value"), &VisualShaderNodeVec3Uniform::get_default_value);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value_enabled"), "set_default_value_enabled", "is_default_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "default_value"), "set_default_value", "get_default_value");
}

bool VisualShaderNodeVec3Uniform::is_show_prop_names() const {
	return true;
}

bool VisualShaderNodeVec3Uniform::is_use_prop_slots() const {
	return true;
}

bool VisualShaderNodeVec3Uniform::is_qualifier_supported(Qualifier p_qual) const {
	return true; // all qualifiers are supported
}

bool VisualShaderNodeVec3Uniform::is_convertible_to_constant() const {
	return true; // conversion is allowed
}

Vector<StringName> VisualShaderNodeVec3Uniform::get_editable_properties() const {
	Vector<StringName> props = VisualShaderNodeUniform::get_editable_properties();
	props.push_back("default_value_enabled");
	if (default_value_enabled) {
		props.push_back("default_value");
	}
	return props;
}

VisualShaderNodeVec3Uniform::VisualShaderNodeVec3Uniform() {
}

////////////// Transform Uniform

String VisualShaderNodeTransformUniform::get_caption() const {
	return "TransformUniform";
}

int VisualShaderNodeTransformUniform::get_input_port_count() const {
	return 0;
}

VisualShaderNodeTransformUniform::PortType VisualShaderNodeTransformUniform::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeTransformUniform::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeTransformUniform::get_output_port_count() const {
	return 1;
}

VisualShaderNodeTransformUniform::PortType VisualShaderNodeTransformUniform::get_output_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeTransformUniform::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

void VisualShaderNodeTransformUniform::set_default_value_enabled(bool p_enabled) {
	default_value_enabled = p_enabled;
	emit_changed();
}

bool VisualShaderNodeTransformUniform::is_default_value_enabled() const {
	return default_value_enabled;
}

void VisualShaderNodeTransformUniform::set_default_value(const Transform3D &p_value) {
	default_value = p_value;
	emit_changed();
}

Transform3D VisualShaderNodeTransformUniform::get_default_value() const {
	return default_value;
}

String VisualShaderNodeTransformUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = _get_qual_str() + "uniform mat4 " + get_uniform_name();
	if (default_value_enabled) {
		Vector3 row0 = default_value.basis.get_row(0);
		Vector3 row1 = default_value.basis.get_row(1);
		Vector3 row2 = default_value.basis.get_row(2);
		Vector3 origin = default_value.origin;
		code += " = mat4(" + vformat("vec4(%.6f, %.6f, %.6f, 0.0)", row0.x, row0.y, row0.z) + vformat(", vec4(%.6f, %.6f, %.6f, 0.0)", row1.x, row1.y, row1.z) + vformat(", vec4(%.6f, %.6f, %.6f, 0.0)", row2.x, row2.y, row2.z) + vformat(", vec4(%.6f, %.6f, %.6f, 1.0)", origin.x, origin.y, origin.z) + ")";
	}
	code += ";\n";
	return code;
}

String VisualShaderNodeTransformUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
}

void VisualShaderNodeTransformUniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_default_value_enabled", "enabled"), &VisualShaderNodeTransformUniform::set_default_value_enabled);
	ClassDB::bind_method(D_METHOD("is_default_value_enabled"), &VisualShaderNodeTransformUniform::is_default_value_enabled);

	ClassDB::bind_method(D_METHOD("set_default_value", "value"), &VisualShaderNodeTransformUniform::set_default_value);
	ClassDB::bind_method(D_METHOD("get_default_value"), &VisualShaderNodeTransformUniform::get_default_value);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value_enabled"), "set_default_value_enabled", "is_default_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "default_value"), "set_default_value", "get_default_value");
}

bool VisualShaderNodeTransformUniform::is_show_prop_names() const {
	return true;
}

bool VisualShaderNodeTransformUniform::is_use_prop_slots() const {
	return true;
}

bool VisualShaderNodeTransformUniform::is_qualifier_supported(Qualifier p_qual) const {
	if (p_qual == Qualifier::QUAL_INSTANCE) {
		return false;
	}
	return true;
}

bool VisualShaderNodeTransformUniform::is_convertible_to_constant() const {
	return true; // conversion is allowed
}

Vector<StringName> VisualShaderNodeTransformUniform::get_editable_properties() const {
	Vector<StringName> props = VisualShaderNodeUniform::get_editable_properties();
	props.push_back("default_value_enabled");
	if (default_value_enabled) {
		props.push_back("default_value");
	}
	return props;
}

VisualShaderNodeTransformUniform::VisualShaderNodeTransformUniform() {
}

////////////// Texture Uniform

String VisualShaderNodeTextureUniform::get_caption() const {
	return "TextureUniform";
}

int VisualShaderNodeTextureUniform::get_input_port_count() const {
	return 2;
}

VisualShaderNodeTextureUniform::PortType VisualShaderNodeTextureUniform::get_input_port_type(int p_port) const {
	return p_port == 0 ? PORT_TYPE_VECTOR : PORT_TYPE_SCALAR;
}

String VisualShaderNodeTextureUniform::get_input_port_name(int p_port) const {
	return p_port == 0 ? "uv" : "lod";
}

int VisualShaderNodeTextureUniform::get_output_port_count() const {
	return 3;
}

VisualShaderNodeTextureUniform::PortType VisualShaderNodeTextureUniform::get_output_port_type(int p_port) const {
	switch (p_port) {
		case 0:
			return PORT_TYPE_VECTOR;
		case 1:
			return PORT_TYPE_SCALAR;
		case 2:
			return PORT_TYPE_SAMPLER;
		default:
			return PORT_TYPE_SCALAR;
	}
}

String VisualShaderNodeTextureUniform::get_output_port_name(int p_port) const {
	switch (p_port) {
		case 0:
			return "rgb";
		case 1:
			return "alpha";
		case 2:
			return "sampler2D";
		default:
			return "";
	}
}

String VisualShaderNodeTextureUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = _get_qual_str() + "uniform sampler2D " + get_uniform_name();

	switch (texture_type) {
		case TYPE_DATA:
			if (color_default == COLOR_DEFAULT_BLACK) {
				code += " : hint_black;\n";
			} else {
				code += ";\n";
			}
			break;
		case TYPE_COLOR:
			if (color_default == COLOR_DEFAULT_BLACK) {
				code += " : hint_black_albedo;\n";
			} else {
				code += " : hint_albedo;\n";
			}
			break;
		case TYPE_NORMAL_MAP:
			code += " : hint_normal;\n";
			break;
		case TYPE_ANISOTROPY:
			code += " : hint_anisotropy;\n";
			break;
		default:
			code += ";\n";
			break;
	}

	return code;
}

bool VisualShaderNodeTextureUniform::is_code_generated() const {
	return is_output_port_connected(0) || is_output_port_connected(1); // rgb or alpha
}

String VisualShaderNodeTextureUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String default_uv;
	if (p_mode != Shader::MODE_PARTICLES && p_mode != Shader::MODE_SKY) {
		default_uv = "UV.xy";
	} else {
		default_uv = "vec2(0.0)";
	}

	String id = get_uniform_name();
	String code = "	{\n";
	if (p_input_vars[0].is_empty()) { // Use UV by default.
		if (p_input_vars[1].is_empty()) {
			code += "		vec4 n_tex_read = texture(" + id + ", " + default_uv + ");\n";
		} else {
			code += "		vec4 n_tex_read = textureLod(" + id + ", " + default_uv + ", " + p_input_vars[1] + ");\n";
		}
	} else if (p_input_vars[1].is_empty()) {
		//no lod
		code += "		vec4 n_tex_read = texture(" + id + ", " + p_input_vars[0] + ".xy);\n";
	} else {
		code += "		vec4 n_tex_read = textureLod(" + id + ", " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ");\n";
	}

	code += "		" + p_output_vars[0] + " = n_tex_read.rgb;\n";
	code += "		" + p_output_vars[1] + " = n_tex_read.a;\n";
	code += "	}\n";
	return code;
}

void VisualShaderNodeTextureUniform::set_texture_type(TextureType p_texture_type) {
	ERR_FAIL_INDEX(int(p_texture_type), int(TYPE_MAX));
	if (texture_type == p_texture_type) {
		return;
	}
	texture_type = p_texture_type;
	emit_changed();
}

VisualShaderNodeTextureUniform::TextureType VisualShaderNodeTextureUniform::get_texture_type() const {
	return texture_type;
}

void VisualShaderNodeTextureUniform::set_color_default(ColorDefault p_color_default) {
	ERR_FAIL_INDEX(int(p_color_default), int(COLOR_DEFAULT_MAX));
	if (color_default == p_color_default) {
		return;
	}
	color_default = p_color_default;
	emit_changed();
}

VisualShaderNodeTextureUniform::ColorDefault VisualShaderNodeTextureUniform::get_color_default() const {
	return color_default;
}

Vector<StringName> VisualShaderNodeTextureUniform::get_editable_properties() const {
	Vector<StringName> props = VisualShaderNodeUniform::get_editable_properties();
	props.push_back("texture_type");
	props.push_back("color_default");
	return props;
}

void VisualShaderNodeTextureUniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture_type", "type"), &VisualShaderNodeTextureUniform::set_texture_type);
	ClassDB::bind_method(D_METHOD("get_texture_type"), &VisualShaderNodeTextureUniform::get_texture_type);

	ClassDB::bind_method(D_METHOD("set_color_default", "type"), &VisualShaderNodeTextureUniform::set_color_default);
	ClassDB::bind_method(D_METHOD("get_color_default"), &VisualShaderNodeTextureUniform::get_color_default);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_type", PROPERTY_HINT_ENUM, "Data,Color,Normal Map,Anisotropic"), "set_texture_type", "get_texture_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "color_default", PROPERTY_HINT_ENUM, "White Default,Black Default"), "set_color_default", "get_color_default");

	BIND_ENUM_CONSTANT(TYPE_DATA);
	BIND_ENUM_CONSTANT(TYPE_COLOR);
	BIND_ENUM_CONSTANT(TYPE_NORMAL_MAP);
	BIND_ENUM_CONSTANT(TYPE_ANISOTROPY);
	BIND_ENUM_CONSTANT(TYPE_MAX);

	BIND_ENUM_CONSTANT(COLOR_DEFAULT_WHITE);
	BIND_ENUM_CONSTANT(COLOR_DEFAULT_BLACK);
	BIND_ENUM_CONSTANT(COLOR_DEFAULT_MAX);
}

String VisualShaderNodeTextureUniform::get_input_port_default_hint(int p_port) const {
	if (p_port == 0) {
		return "default";
	}
	return "";
}

bool VisualShaderNodeTextureUniform::is_qualifier_supported(Qualifier p_qual) const {
	switch (p_qual) {
		case Qualifier::QUAL_NONE:
			return true;
		case Qualifier::QUAL_GLOBAL:
			return true;
		case Qualifier::QUAL_INSTANCE:
			return false;
		default:
			break;
	}
	return false;
}

bool VisualShaderNodeTextureUniform::is_convertible_to_constant() const {
	return false; // conversion is not allowed
}

VisualShaderNodeTextureUniform::VisualShaderNodeTextureUniform() {
	simple_decl = false;
}

////////////// Texture Uniform (Triplanar)

String VisualShaderNodeTextureUniformTriplanar::get_caption() const {
	return "TextureUniformTriplanar";
}

int VisualShaderNodeTextureUniformTriplanar::get_input_port_count() const {
	return 2;
}

VisualShaderNodeTextureUniform::PortType VisualShaderNodeTextureUniformTriplanar::get_input_port_type(int p_port) const {
	if (p_port == 0 || p_port == 1) {
		return PORT_TYPE_VECTOR;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeTextureUniformTriplanar::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "weights";
	} else if (p_port == 1) {
		return "pos";
	}
	return "";
}

String VisualShaderNodeTextureUniformTriplanar::generate_global_per_node(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code;

	code += "// TRIPLANAR FUNCTION GLOBAL CODE\n";
	code += "	vec4 triplanar_texture(sampler2D p_sampler, vec3 p_weights, vec3 p_triplanar_pos) {\n";
	code += "		vec4 samp = vec4(0.0);\n";
	code += "		samp += texture(p_sampler, p_triplanar_pos.xy) * p_weights.z;\n";
	code += "		samp += texture(p_sampler, p_triplanar_pos.xz) * p_weights.y;\n";
	code += "		samp += texture(p_sampler, p_triplanar_pos.zy * vec2(-1.0, 1.0)) * p_weights.x;\n";
	code += "		return samp;\n";
	code += "	}\n";
	code += "\n";
	code += "	uniform vec3 triplanar_scale = vec3(1.0, 1.0, 1.0);\n";
	code += "	uniform vec3 triplanar_offset;\n";
	code += "	uniform float triplanar_sharpness = 0.5;\n";
	code += "\n";
	code += "	varying vec3 triplanar_power_normal;\n";
	code += "	varying vec3 triplanar_pos;\n";

	return code;
}

String VisualShaderNodeTextureUniformTriplanar::generate_global_per_func(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code;

	if (p_type == VisualShader::TYPE_VERTEX) {
		code += "	// TRIPLANAR FUNCTION VERTEX CODE\n";
		code += "		triplanar_power_normal = pow(abs(NORMAL), vec3(triplanar_sharpness));\n";
		code += "		triplanar_power_normal /= dot(triplanar_power_normal, vec3(1.0));\n";
		code += "		triplanar_pos = VERTEX * triplanar_scale + triplanar_offset;\n";
		code += "		triplanar_pos *= vec3(1.0, -1.0, 1.0);\n";
	}

	return code;
}

String VisualShaderNodeTextureUniformTriplanar::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String id = get_uniform_name();
	String code = "	{\n";

	if (p_input_vars[0].is_empty() && p_input_vars[1].is_empty()) {
		code += "		vec4 n_tex_read = triplanar_texture(" + id + ", triplanar_power_normal, triplanar_pos);\n";
	} else if (!p_input_vars[0].is_empty() && p_input_vars[1].is_empty()) {
		code += "		vec4 n_tex_read = triplanar_texture(" + id + ", " + p_input_vars[0] + ", triplanar_pos);\n";
	} else if (p_input_vars[0].is_empty() && !p_input_vars[1].is_empty()) {
		code += "		vec4 n_tex_read = triplanar_texture(" + id + ", triplanar_power_normal, " + p_input_vars[1] + ");\n";
	} else {
		code += "		vec4 n_tex_read = triplanar_texture(" + id + ", " + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
	}

	code += "		" + p_output_vars[0] + " = n_tex_read.rgb;\n";
	code += "		" + p_output_vars[1] + " = n_tex_read.a;\n";
	code += "	}\n";

	return code;
}

String VisualShaderNodeTextureUniformTriplanar::get_input_port_default_hint(int p_port) const {
	if (p_port == 0) {
		return "default";
	} else if (p_port == 1) {
		return "default";
	}
	return "";
}

VisualShaderNodeTextureUniformTriplanar::VisualShaderNodeTextureUniformTriplanar() {
}

////////////// Texture2DArray Uniform

String VisualShaderNodeTexture2DArrayUniform::get_caption() const {
	return "Texture2DArrayUniform";
}

int VisualShaderNodeTexture2DArrayUniform::get_output_port_count() const {
	return 1;
}

VisualShaderNodeTexture2DArrayUniform::PortType VisualShaderNodeTexture2DArrayUniform::get_output_port_type(int p_port) const {
	return PORT_TYPE_SAMPLER;
}

String VisualShaderNodeTexture2DArrayUniform::get_output_port_name(int p_port) const {
	return "sampler2DArray";
}

int VisualShaderNodeTexture2DArrayUniform::get_input_port_count() const {
	return 0;
}

VisualShaderNodeTexture2DArrayUniform::PortType VisualShaderNodeTexture2DArrayUniform::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeTexture2DArrayUniform::get_input_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeTexture2DArrayUniform::get_input_port_default_hint(int p_port) const {
	return "";
}

String VisualShaderNodeTexture2DArrayUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = _get_qual_str() + "uniform sampler2DArray " + get_uniform_name();

	switch (texture_type) {
		case TYPE_DATA:
			if (color_default == COLOR_DEFAULT_BLACK) {
				code += " : hint_black;\n";
			} else {
				code += ";\n";
			}
			break;
		case TYPE_COLOR:
			if (color_default == COLOR_DEFAULT_BLACK) {
				code += " : hint_black_albedo;\n";
			} else {
				code += " : hint_albedo;\n";
			}
			break;
		case TYPE_NORMAL_MAP:
			code += " : hint_normal;\n";
			break;
		case TYPE_ANISOTROPY:
			code += " : hint_anisotropy;\n";
			break;
		default:
			code += ";\n";
			break;
	}

	return code;
}

String VisualShaderNodeTexture2DArrayUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return String();
}

VisualShaderNodeTexture2DArrayUniform::VisualShaderNodeTexture2DArrayUniform() {
}

////////////// Texture3D Uniform

String VisualShaderNodeTexture3DUniform::get_caption() const {
	return "Texture3DUniform";
}

int VisualShaderNodeTexture3DUniform::get_output_port_count() const {
	return 1;
}

VisualShaderNodeTexture3DUniform::PortType VisualShaderNodeTexture3DUniform::get_output_port_type(int p_port) const {
	return PORT_TYPE_SAMPLER;
}

String VisualShaderNodeTexture3DUniform::get_output_port_name(int p_port) const {
	return "sampler3D";
}

int VisualShaderNodeTexture3DUniform::get_input_port_count() const {
	return 0;
}

VisualShaderNodeTexture3DUniform::PortType VisualShaderNodeTexture3DUniform::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeTexture3DUniform::get_input_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeTexture3DUniform::get_input_port_default_hint(int p_port) const {
	return "";
}

String VisualShaderNodeTexture3DUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = _get_qual_str() + "uniform sampler3D " + get_uniform_name();

	switch (texture_type) {
		case TYPE_DATA:
			if (color_default == COLOR_DEFAULT_BLACK) {
				code += " : hint_black;\n";
			} else {
				code += ";\n";
			}
			break;
		case TYPE_COLOR:
			if (color_default == COLOR_DEFAULT_BLACK) {
				code += " : hint_black_albedo;\n";
			} else {
				code += " : hint_albedo;\n";
			}
			break;
		case TYPE_NORMAL_MAP:
			code += " : hint_normal;\n";
			break;
		case TYPE_ANISOTROPY:
			code += " : hint_anisotropy;\n";
			break;
		default:
			code += ";\n";
			break;
	}

	return code;
}

String VisualShaderNodeTexture3DUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return String();
}

VisualShaderNodeTexture3DUniform::VisualShaderNodeTexture3DUniform() {
}

////////////// Cubemap Uniform

String VisualShaderNodeCubemapUniform::get_caption() const {
	return "CubemapUniform";
}

int VisualShaderNodeCubemapUniform::get_output_port_count() const {
	return 1;
}

VisualShaderNodeCubemapUniform::PortType VisualShaderNodeCubemapUniform::get_output_port_type(int p_port) const {
	return PORT_TYPE_SAMPLER;
}

String VisualShaderNodeCubemapUniform::get_output_port_name(int p_port) const {
	return "samplerCube";
}

int VisualShaderNodeCubemapUniform::get_input_port_count() const {
	return 0;
}

VisualShaderNodeCubemapUniform::PortType VisualShaderNodeCubemapUniform::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeCubemapUniform::get_input_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeCubemapUniform::get_input_port_default_hint(int p_port) const {
	return "";
}

String VisualShaderNodeCubemapUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = _get_qual_str() + "uniform samplerCube " + get_uniform_name();

	switch (texture_type) {
		case TYPE_DATA:
			if (color_default == COLOR_DEFAULT_BLACK) {
				code += " : hint_black;\n";
			} else {
				code += ";\n";
			}
			break;
		case TYPE_COLOR:
			if (color_default == COLOR_DEFAULT_BLACK) {
				code += " : hint_black_albedo;\n";
			} else {
				code += " : hint_albedo;\n";
			}
			break;
		case TYPE_NORMAL_MAP:
			code += " : hint_normal;\n";
			break;
		case TYPE_ANISOTROPY:
			code += " : hint_anisotropy;\n";
			break;
		default:
			code += ";\n";
			break;
	}

	return code;
}

String VisualShaderNodeCubemapUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return String();
}

VisualShaderNodeCubemapUniform::VisualShaderNodeCubemapUniform() {
}

////////////// If

String VisualShaderNodeIf::get_caption() const {
	return "If";
}

int VisualShaderNodeIf::get_input_port_count() const {
	return 6;
}

VisualShaderNodeIf::PortType VisualShaderNodeIf::get_input_port_type(int p_port) const {
	if (p_port == 0 || p_port == 1 || p_port == 2) {
		return PORT_TYPE_SCALAR;
	}
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeIf::get_input_port_name(int p_port) const {
	switch (p_port) {
		case 0:
			return "a";
		case 1:
			return "b";
		case 2:
			return "tolerance";
		case 3:
			return "a == b";
		case 4:
			return "a > b";
		case 5:
			return "a < b";
		default:
			return "";
	}
}

int VisualShaderNodeIf::get_output_port_count() const {
	return 1;
}

VisualShaderNodeIf::PortType VisualShaderNodeIf::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeIf::get_output_port_name(int p_port) const {
	return "result";
}

String VisualShaderNodeIf::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	code += "	if(abs(" + p_input_vars[0] + " - " + p_input_vars[1] + ") < " + p_input_vars[2] + ")\n"; // abs(a - b) < tolerance eg. a == b
	code += "	{\n";
	code += "		" + p_output_vars[0] + " = " + p_input_vars[3] + ";\n";
	code += "	}\n";
	code += "	else if(" + p_input_vars[0] + " < " + p_input_vars[1] + ")\n"; // a < b
	code += "	{\n";
	code += "		" + p_output_vars[0] + " = " + p_input_vars[5] + ";\n";
	code += "	}\n";
	code += "	else\n"; // a > b (or a >= b if abs(a - b) < tolerance is false)
	code += "	{\n";
	code += "		" + p_output_vars[0] + " = " + p_input_vars[4] + ";\n";
	code += "	}\n";
	return code;
}

VisualShaderNodeIf::VisualShaderNodeIf() {
	simple_decl = false;
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, CMP_EPSILON);
	set_input_port_default_value(3, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(4, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(5, Vector3(0.0, 0.0, 0.0));
}

////////////// Switch

String VisualShaderNodeSwitch::get_caption() const {
	return "Switch";
}

int VisualShaderNodeSwitch::get_input_port_count() const {
	return 3;
}

VisualShaderNodeSwitch::PortType VisualShaderNodeSwitch::get_input_port_type(int p_port) const {
	if (p_port == 0) {
		return PORT_TYPE_BOOLEAN;
	}
	if (p_port == 1 || p_port == 2) {
		switch (op_type) {
			case OP_TYPE_INT:
				return PORT_TYPE_SCALAR_INT;
			case OP_TYPE_VECTOR:
				return PORT_TYPE_VECTOR;
			case OP_TYPE_BOOLEAN:
				return PORT_TYPE_BOOLEAN;
			case OP_TYPE_TRANSFORM:
				return PORT_TYPE_TRANSFORM;
			default:
				break;
		}
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeSwitch::get_input_port_name(int p_port) const {
	switch (p_port) {
		case 0:
			return "value";
		case 1:
			return "true";
		case 2:
			return "false";
		default:
			return "";
	}
}

int VisualShaderNodeSwitch::get_output_port_count() const {
	return 1;
}

VisualShaderNodeSwitch::PortType VisualShaderNodeSwitch::get_output_port_type(int p_port) const {
	switch (op_type) {
		case OP_TYPE_INT:
			return PORT_TYPE_SCALAR_INT;
		case OP_TYPE_VECTOR:
			return PORT_TYPE_VECTOR;
		case OP_TYPE_BOOLEAN:
			return PORT_TYPE_BOOLEAN;
		case OP_TYPE_TRANSFORM:
			return PORT_TYPE_TRANSFORM;
		default:
			break;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeSwitch::get_output_port_name(int p_port) const {
	return "result";
}

void VisualShaderNodeSwitch::set_op_type(OpType p_op_type) {
	ERR_FAIL_INDEX(int(p_op_type), int(OP_TYPE_MAX));
	if (op_type == p_op_type) {
		return;
	}
	switch (p_op_type) {
		case OP_TYPE_FLOAT:
			set_input_port_default_value(1, 1.0);
			set_input_port_default_value(2, 0.0);
			break;
		case OP_TYPE_INT:
			set_input_port_default_value(1, 1);
			set_input_port_default_value(2, 0);
			break;
		case OP_TYPE_VECTOR:
			set_input_port_default_value(1, Vector3(1.0, 1.0, 1.0));
			set_input_port_default_value(2, Vector3(0.0, 0.0, 0.0));
			break;
		case OP_TYPE_BOOLEAN:
			set_input_port_default_value(1, true);
			set_input_port_default_value(2, false);
			break;
		case OP_TYPE_TRANSFORM:
			set_input_port_default_value(1, Transform3D());
			set_input_port_default_value(2, Transform3D());
			break;
		default:
			break;
	}
	op_type = p_op_type;
	emit_changed();
}

VisualShaderNodeSwitch::OpType VisualShaderNodeSwitch::get_op_type() const {
	return op_type;
}

Vector<StringName> VisualShaderNodeSwitch::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("op_type");
	return props;
}

void VisualShaderNodeSwitch::_bind_methods() { // static
	ClassDB::bind_method(D_METHOD("set_op_type", "type"), &VisualShaderNodeSwitch::set_op_type);
	ClassDB::bind_method(D_METHOD("get_op_type"), &VisualShaderNodeSwitch::get_op_type);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "op_type", PROPERTY_HINT_ENUM, "Float,Int,Vector,Boolean,Transform"), "set_op_type", "get_op_type");

	BIND_ENUM_CONSTANT(OP_TYPE_FLOAT);
	BIND_ENUM_CONSTANT(OP_TYPE_INT);
	BIND_ENUM_CONSTANT(OP_TYPE_VECTOR);
	BIND_ENUM_CONSTANT(OP_TYPE_BOOLEAN);
	BIND_ENUM_CONSTANT(OP_TYPE_TRANSFORM);
	BIND_ENUM_CONSTANT(OP_TYPE_MAX);
}

String VisualShaderNodeSwitch::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	code += "	if(" + p_input_vars[0] + ")\n";
	code += "	{\n";
	code += "		" + p_output_vars[0] + " = " + p_input_vars[1] + ";\n";
	code += "	}\n";
	code += "	else\n";
	code += "	{\n";
	code += "		" + p_output_vars[0] + " = " + p_input_vars[2] + ";\n";
	code += "	}\n";
	return code;
}

VisualShaderNodeSwitch::VisualShaderNodeSwitch() {
	simple_decl = false;
	set_input_port_default_value(0, false);
	set_input_port_default_value(1, 1.0);
	set_input_port_default_value(2, 0.0);
}

////////////// Fresnel

String VisualShaderNodeFresnel::get_caption() const {
	return "Fresnel";
}

int VisualShaderNodeFresnel::get_input_port_count() const {
	return 4;
}

VisualShaderNodeFresnel::PortType VisualShaderNodeFresnel::get_input_port_type(int p_port) const {
	switch (p_port) {
		case 0:
			return PORT_TYPE_VECTOR;
		case 1:
			return PORT_TYPE_VECTOR;
		case 2:
			return PORT_TYPE_BOOLEAN;
		case 3:
			return PORT_TYPE_SCALAR;
		default:
			return PORT_TYPE_VECTOR;
	}
}

String VisualShaderNodeFresnel::get_input_port_name(int p_port) const {
	switch (p_port) {
		case 0:
			return "normal";
		case 1:
			return "view";
		case 2:
			return "invert";
		case 3:
			return "power";
		default:
			return "";
	}
}

int VisualShaderNodeFresnel::get_output_port_count() const {
	return 1;
}

VisualShaderNodeFresnel::PortType VisualShaderNodeFresnel::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeFresnel::get_output_port_name(int p_port) const {
	return "result";
}

bool VisualShaderNodeFresnel::is_generate_input_var(int p_port) const {
	if (p_port == 2) {
		return false;
	}
	return true;
}

String VisualShaderNodeFresnel::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String normal;
	String view;
	if (p_input_vars[0].is_empty()) {
		normal = "NORMAL";
	} else {
		normal = p_input_vars[0];
	}
	if (p_input_vars[1].is_empty()) {
		view = "VIEW";
	} else {
		view = p_input_vars[1];
	}

	if (is_input_port_connected(2)) {
		return "	" + p_output_vars[0] + " = " + p_input_vars[2] + " ? (pow(clamp(dot(" + normal + ", " + view + "), 0.0, 1.0), " + p_input_vars[3] + ")) : (pow(1.0 - clamp(dot(" + normal + ", " + view + "), 0.0, 1.0), " + p_input_vars[3] + "));\n";
	} else {
		if (get_input_port_default_value(2)) {
			return "	" + p_output_vars[0] + " = pow(clamp(dot(" + normal + ", " + view + "), 0.0, 1.0), " + p_input_vars[3] + ");\n";
		} else {
			return "	" + p_output_vars[0] + " = pow(1.0 - clamp(dot(" + normal + ", " + view + "), 0.0, 1.0), " + p_input_vars[3] + ");\n";
		}
	}
}

String VisualShaderNodeFresnel::get_input_port_default_hint(int p_port) const {
	if (p_port == 0) {
		return "default";
	} else if (p_port == 1) {
		return "default";
	}
	return "";
}

VisualShaderNodeFresnel::VisualShaderNodeFresnel() {
	set_input_port_default_value(2, false);
	set_input_port_default_value(3, 1.0);
}

////////////// Is

String VisualShaderNodeIs::get_caption() const {
	return "Is";
}

int VisualShaderNodeIs::get_input_port_count() const {
	return 1;
}

VisualShaderNodeIs::PortType VisualShaderNodeIs::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeIs::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeIs::get_output_port_count() const {
	return 1;
}

VisualShaderNodeIs::PortType VisualShaderNodeIs::get_output_port_type(int p_port) const {
	return PORT_TYPE_BOOLEAN;
}

String VisualShaderNodeIs::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeIs::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	static const char *functions[FUNC_MAX] = {
		"isinf($)",
		"isnan($)"
	};

	String code;
	code += "	" + p_output_vars[0] + " = " + String(functions[func]).replace("$", p_input_vars[0]) + ";\n";
	return code;
}

void VisualShaderNodeIs::set_function(Function p_func) {
	ERR_FAIL_INDEX(int(p_func), int(FUNC_MAX));
	if (func == p_func) {
		return;
	}
	func = p_func;
	emit_changed();
}

VisualShaderNodeIs::Function VisualShaderNodeIs::get_function() const {
	return func;
}

Vector<StringName> VisualShaderNodeIs::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("function");
	return props;
}

void VisualShaderNodeIs::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_function", "func"), &VisualShaderNodeIs::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &VisualShaderNodeIs::get_function);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "function", PROPERTY_HINT_ENUM, "Inf,NaN"), "set_function", "get_function");

	BIND_ENUM_CONSTANT(FUNC_IS_INF);
	BIND_ENUM_CONSTANT(FUNC_IS_NAN);
	BIND_ENUM_CONSTANT(FUNC_MAX);
}

VisualShaderNodeIs::VisualShaderNodeIs() {
	set_input_port_default_value(0, 0.0);
}

////////////// Compare

String VisualShaderNodeCompare::get_caption() const {
	return "Compare";
}

int VisualShaderNodeCompare::get_input_port_count() const {
	if (comparison_type == CTYPE_SCALAR && (func == FUNC_EQUAL || func == FUNC_NOT_EQUAL)) {
		return 3;
	}
	return 2;
}

VisualShaderNodeCompare::PortType VisualShaderNodeCompare::get_input_port_type(int p_port) const {
	switch (comparison_type) {
		case CTYPE_SCALAR:
			return PORT_TYPE_SCALAR;
		case CTYPE_SCALAR_INT:
			return PORT_TYPE_SCALAR_INT;
		case CTYPE_VECTOR:
			return PORT_TYPE_VECTOR;
		case CTYPE_BOOLEAN:
			return PORT_TYPE_BOOLEAN;
		case CTYPE_TRANSFORM:
			return PORT_TYPE_TRANSFORM;
		default:
			return PORT_TYPE_SCALAR;
	}
}

String VisualShaderNodeCompare::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "a";
	} else if (p_port == 1) {
		return "b";
	} else if (p_port == 2) {
		return "tolerance";
	}
	return "";
}

int VisualShaderNodeCompare::get_output_port_count() const {
	return 1;
}

VisualShaderNodeCompare::PortType VisualShaderNodeCompare::get_output_port_type(int p_port) const {
	return PORT_TYPE_BOOLEAN;
}

String VisualShaderNodeCompare::get_output_port_name(int p_port) const {
	if (p_port == 0) {
		return "result";
	}
	return "";
}

String VisualShaderNodeCompare::get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const {
	if (comparison_type == CTYPE_BOOLEAN || comparison_type == CTYPE_TRANSFORM) {
		if (func > FUNC_NOT_EQUAL) {
			return TTR("Invalid comparison function for that type.");
		}
	}
	return "";
}

String VisualShaderNodeCompare::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	static const char *operators[FUNC_MAX] = {
		"==",
		"!=",
		">",
		">=",
		"<",
		"<=",
	};

	static const char *functions[FUNC_MAX] = {
		"equal($)",
		"notEqual($)",
		"greaterThan($)",
		"greaterThanEqual($)",
		"lessThan($)",
		"lessThanEqual($)",
	};

	static const char *conditions[COND_MAX] = {
		"all($)",
		"any($)",
	};

	String code;
	switch (comparison_type) {
		case CTYPE_SCALAR:
			if (func == FUNC_EQUAL) {
				code += "	" + p_output_vars[0] + " = (abs(" + p_input_vars[0] + " - " + p_input_vars[1] + ") < " + p_input_vars[2] + ");";
			} else if (func == FUNC_NOT_EQUAL) {
				code += "	" + p_output_vars[0] + " = !(abs(" + p_input_vars[0] + " - " + p_input_vars[1] + ") < " + p_input_vars[2] + ");";
			} else {
				code += "	" + p_output_vars[0] + " = " + (p_input_vars[0] + " $ " + p_input_vars[1]).replace("$", operators[func]) + ";\n";
			}
			break;

		case CTYPE_SCALAR_INT:
			code += "	" + p_output_vars[0] + " = " + (p_input_vars[0] + " $ " + p_input_vars[1]).replace("$", operators[func]) + ";\n";
			break;

		case CTYPE_VECTOR:
			code += "	{\n";
			code += "		bvec3 _bv = " + String(functions[func]).replace("$", p_input_vars[0] + ", " + p_input_vars[1]) + ";\n";
			code += "		" + p_output_vars[0] + " = " + String(conditions[condition]).replace("$", "_bv") + ";\n";
			code += "	}\n";
			break;

		case CTYPE_BOOLEAN:
			if (func > FUNC_NOT_EQUAL) {
				return "	" + p_output_vars[0] + " = false;\n";
			}
			code += "	" + p_output_vars[0] + " = " + (p_input_vars[0] + " $ " + p_input_vars[1]).replace("$", operators[func]) + ";\n";
			break;

		case CTYPE_TRANSFORM:
			if (func > FUNC_NOT_EQUAL) {
				return "	" + p_output_vars[0] + " = false;\n";
			}
			code += "	" + p_output_vars[0] + " = " + (p_input_vars[0] + " $ " + p_input_vars[1]).replace("$", operators[func]) + ";\n";
			break;

		default:
			break;
	}
	return code;
}

void VisualShaderNodeCompare::set_comparison_type(ComparisonType p_comparison_type) {
	ERR_FAIL_INDEX(int(p_comparison_type), int(CTYPE_MAX));
	if (comparison_type == p_comparison_type) {
		return;
	}
	switch (p_comparison_type) {
		case CTYPE_SCALAR:
			set_input_port_default_value(0, 0.0);
			set_input_port_default_value(1, 0.0);
			simple_decl = true;
			break;
		case CTYPE_SCALAR_INT:
			set_input_port_default_value(0, 0);
			set_input_port_default_value(1, 0);
			simple_decl = true;
			break;
		case CTYPE_VECTOR:
			set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0));
			set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0));
			simple_decl = false;
			break;
		case CTYPE_BOOLEAN:
			set_input_port_default_value(0, false);
			set_input_port_default_value(1, false);
			simple_decl = true;
			break;
		case CTYPE_TRANSFORM:
			set_input_port_default_value(0, Transform3D());
			set_input_port_default_value(1, Transform3D());
			simple_decl = true;
			break;
		default:
			break;
	}
	comparison_type = p_comparison_type;
	emit_changed();
}

VisualShaderNodeCompare::ComparisonType VisualShaderNodeCompare::get_comparison_type() const {
	return comparison_type;
}

void VisualShaderNodeCompare::set_function(Function p_func) {
	ERR_FAIL_INDEX(int(p_func), int(FUNC_MAX));
	if (func == p_func) {
		return;
	}
	func = p_func;
	emit_changed();
}

VisualShaderNodeCompare::Function VisualShaderNodeCompare::get_function() const {
	return func;
}

void VisualShaderNodeCompare::set_condition(Condition p_condition) {
	ERR_FAIL_INDEX(int(p_condition), int(COND_MAX));
	if (condition == p_condition) {
		return;
	}
	condition = p_condition;
	emit_changed();
}

VisualShaderNodeCompare::Condition VisualShaderNodeCompare::get_condition() const {
	return condition;
}

Vector<StringName> VisualShaderNodeCompare::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("type");
	props.push_back("function");
	if (comparison_type == CTYPE_VECTOR) {
		props.push_back("condition");
	}
	return props;
}

void VisualShaderNodeCompare::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_comparison_type", "type"), &VisualShaderNodeCompare::set_comparison_type);
	ClassDB::bind_method(D_METHOD("get_comparison_type"), &VisualShaderNodeCompare::get_comparison_type);

	ClassDB::bind_method(D_METHOD("set_function", "func"), &VisualShaderNodeCompare::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &VisualShaderNodeCompare::get_function);

	ClassDB::bind_method(D_METHOD("set_condition", "condition"), &VisualShaderNodeCompare::set_condition);
	ClassDB::bind_method(D_METHOD("get_condition"), &VisualShaderNodeCompare::get_condition);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, "Float,Int,Vector,Boolean,Transform"), "set_comparison_type", "get_comparison_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "function", PROPERTY_HINT_ENUM, "a == b,a != b,a > b,a >= b,a < b,a <= b"), "set_function", "get_function");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "condition", PROPERTY_HINT_ENUM, "All,Any"), "set_condition", "get_condition");

	BIND_ENUM_CONSTANT(CTYPE_SCALAR);
	BIND_ENUM_CONSTANT(CTYPE_SCALAR_INT);
	BIND_ENUM_CONSTANT(CTYPE_VECTOR);
	BIND_ENUM_CONSTANT(CTYPE_BOOLEAN);
	BIND_ENUM_CONSTANT(CTYPE_TRANSFORM);
	BIND_ENUM_CONSTANT(CTYPE_MAX);

	BIND_ENUM_CONSTANT(FUNC_EQUAL);
	BIND_ENUM_CONSTANT(FUNC_NOT_EQUAL);
	BIND_ENUM_CONSTANT(FUNC_GREATER_THAN);
	BIND_ENUM_CONSTANT(FUNC_GREATER_THAN_EQUAL);
	BIND_ENUM_CONSTANT(FUNC_LESS_THAN);
	BIND_ENUM_CONSTANT(FUNC_LESS_THAN_EQUAL);
	BIND_ENUM_CONSTANT(FUNC_MAX);

	BIND_ENUM_CONSTANT(COND_ALL);
	BIND_ENUM_CONSTANT(COND_ANY);
	BIND_ENUM_CONSTANT(COND_MAX);
}

VisualShaderNodeCompare::VisualShaderNodeCompare() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, CMP_EPSILON);
}

////////////// Fma

String VisualShaderNodeMultiplyAdd::get_caption() const {
	return "MultiplyAdd";
}

int VisualShaderNodeMultiplyAdd::get_input_port_count() const {
	return 3;
}

VisualShaderNodeMultiplyAdd::PortType VisualShaderNodeMultiplyAdd::get_input_port_type(int p_port) const {
	if (op_type == OP_TYPE_VECTOR) {
		return PORT_TYPE_VECTOR;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeMultiplyAdd::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "a";
	} else if (p_port == 1) {
		return "b(*)";
	} else if (p_port == 2) {
		return "c(+)";
	}
	return "";
}

int VisualShaderNodeMultiplyAdd::get_output_port_count() const {
	return 1;
}

VisualShaderNodeMultiplyAdd::PortType VisualShaderNodeMultiplyAdd::get_output_port_type(int p_port) const {
	if (op_type == OP_TYPE_SCALAR) {
		return PORT_TYPE_SCALAR;
	} else {
		return PORT_TYPE_VECTOR;
	}
}

String VisualShaderNodeMultiplyAdd::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeMultiplyAdd::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "	" + p_output_vars[0] + " = fma(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

void VisualShaderNodeMultiplyAdd::set_op_type(OpType p_op_type) {
	ERR_FAIL_INDEX((int)p_op_type, int(OP_TYPE_MAX));
	if (op_type == p_op_type) {
		return;
	}
	switch (p_op_type) {
		case OP_TYPE_SCALAR:
			set_input_port_default_value(0, 0.0);
			set_input_port_default_value(1, 0.0);
			set_input_port_default_value(2, 0.0);
			break;
		case OP_TYPE_VECTOR:
			set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0));
			set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0));
			set_input_port_default_value(2, Vector3(0.0, 0.0, 0.0));
			break;
		default:
			break;
	}
	op_type = p_op_type;
	emit_changed();
}

VisualShaderNodeMultiplyAdd::OpType VisualShaderNodeMultiplyAdd::get_op_type() const {
	return op_type;
}

Vector<StringName> VisualShaderNodeMultiplyAdd::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("op_type");
	return props;
}

void VisualShaderNodeMultiplyAdd::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_op_type", "type"), &VisualShaderNodeMultiplyAdd::set_op_type);
	ClassDB::bind_method(D_METHOD("get_op_type"), &VisualShaderNodeMultiplyAdd::get_op_type);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "op_type", PROPERTY_HINT_ENUM, "Scalar,Vector"), "set_op_type", "get_op_type");

	BIND_ENUM_CONSTANT(OP_TYPE_SCALAR);
	BIND_ENUM_CONSTANT(OP_TYPE_VECTOR);
	BIND_ENUM_CONSTANT(OP_TYPE_MAX);
}

VisualShaderNodeMultiplyAdd::VisualShaderNodeMultiplyAdd() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, 0.0);
}

////////////// Billboard

String VisualShaderNodeBillboard::get_caption() const {
	return "GetBillboardMatrix";
}

int VisualShaderNodeBillboard::get_input_port_count() const {
	return 0;
}

VisualShaderNodeBillboard::PortType VisualShaderNodeBillboard::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeBillboard::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeBillboard::get_output_port_count() const {
	return 1;
}

VisualShaderNodeBillboard::PortType VisualShaderNodeBillboard::get_output_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeBillboard::get_output_port_name(int p_port) const {
	return "model_view_matrix";
}

String VisualShaderNodeBillboard::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;

	switch (billboard_type) {
		case BILLBOARD_TYPE_ENABLED:
			code += "	{\n";
			code += "		mat4 __mvm = INV_CAMERA_MATRIX * mat4(CAMERA_MATRIX[0], CAMERA_MATRIX[1], CAMERA_MATRIX[2], WORLD_MATRIX[3]);\n";
			if (keep_scale) {
				code += "		__mvm = __mvm * mat4(vec4(length(WORLD_MATRIX[0].xyz), 0.0, 0.0, 0.0), vec4(0.0, length(WORLD_MATRIX[1].xyz), 0.0, 0.0), vec4(0.0, 0.0, length(WORLD_MATRIX[2].xyz), 0.0), vec4(0.0, 0.0, 0.0, 1.0));\n";
			}
			code += "		" + p_output_vars[0] + " = __mvm;\n";
			code += "	}\n";
			break;
		case BILLBOARD_TYPE_FIXED_Y:
			code += "	{\n";
			code += "		mat4 __mvm = INV_CAMERA_MATRIX * mat4(CAMERA_MATRIX[0], WORLD_MATRIX[1], vec4(normalize(cross(CAMERA_MATRIX[0].xyz, WORLD_MATRIX[1].xyz)), 0.0), WORLD_MATRIX[3]);\n";
			if (keep_scale) {
				code += "		__mvm = __mvm * mat4(vec4(length(WORLD_MATRIX[0].xyz), 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.0, 0.0, length(WORLD_MATRIX[2].xyz), 0.0), vec4(0.0, 0.0, 0.0, 1.0));\n";
			} else {
				code += "		__mvm = __mvm * mat4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0 / length(WORLD_MATRIX[1].xyz), 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));\n";
			}
			code += "		" + p_output_vars[0] + " = __mvm;\n";
			code += "	}\n";
			break;
		case BILLBOARD_TYPE_PARTICLES:
			code += "	{\n";
			code += "		mat4 __wm = mat4(normalize(CAMERA_MATRIX[0]) * length(WORLD_MATRIX[0]), normalize(CAMERA_MATRIX[1]) * length(WORLD_MATRIX[0]), normalize(CAMERA_MATRIX[2]) * length(WORLD_MATRIX[2]), WORLD_MATRIX[3]);\n";
			code += "		__wm = __wm * mat4(vec4(cos(INSTANCE_CUSTOM.x), -sin(INSTANCE_CUSTOM.x), 0.0, 0.0), vec4(sin(INSTANCE_CUSTOM.x), cos(INSTANCE_CUSTOM.x), 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));\n";
			code += "		" + p_output_vars[0] + " = INV_CAMERA_MATRIX * __wm;\n";
			code += "	}\n";
			break;
		default:
			code += "	" + p_output_vars[0] + " = mat4(1.0);\n";
			break;
	}

	return code;
}

bool VisualShaderNodeBillboard::is_show_prop_names() const {
	return true;
}

void VisualShaderNodeBillboard::set_billboard_type(BillboardType p_billboard_type) {
	ERR_FAIL_INDEX(int(p_billboard_type), int(BILLBOARD_TYPE_MAX));
	if (billboard_type == p_billboard_type) {
		return;
	}
	billboard_type = p_billboard_type;
	simple_decl = bool(billboard_type == BILLBOARD_TYPE_DISABLED);
	set_disabled(simple_decl);
	emit_changed();
}

VisualShaderNodeBillboard::BillboardType VisualShaderNodeBillboard::get_billboard_type() const {
	return billboard_type;
}

void VisualShaderNodeBillboard::set_keep_scale_enabled(bool p_enabled) {
	keep_scale = p_enabled;
	emit_changed();
}

bool VisualShaderNodeBillboard::is_keep_scale_enabled() const {
	return keep_scale;
}

Vector<StringName> VisualShaderNodeBillboard::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("billboard_type");
	if (billboard_type == BILLBOARD_TYPE_ENABLED || billboard_type == BILLBOARD_TYPE_FIXED_Y) {
		props.push_back("keep_scale");
	}
	return props;
}

void VisualShaderNodeBillboard::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_billboard_type", "billboard_type"), &VisualShaderNodeBillboard::set_billboard_type);
	ClassDB::bind_method(D_METHOD("get_billboard_type"), &VisualShaderNodeBillboard::get_billboard_type);

	ClassDB::bind_method(D_METHOD("set_keep_scale_enabled", "enabled"), &VisualShaderNodeBillboard::set_keep_scale_enabled);
	ClassDB::bind_method(D_METHOD("is_keep_scale_enabled"), &VisualShaderNodeBillboard::is_keep_scale_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "billboard_type", PROPERTY_HINT_ENUM, "Disabled,Enabled,Y-Billboard,Particles"), "set_billboard_type", "get_billboard_type");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "keep_scale"), "set_keep_scale_enabled", "is_keep_scale_enabled");

	BIND_ENUM_CONSTANT(BILLBOARD_TYPE_DISABLED);
	BIND_ENUM_CONSTANT(BILLBOARD_TYPE_ENABLED);
	BIND_ENUM_CONSTANT(BILLBOARD_TYPE_FIXED_Y);
	BIND_ENUM_CONSTANT(BILLBOARD_TYPE_PARTICLES);
	BIND_ENUM_CONSTANT(BILLBOARD_TYPE_MAX);
}

VisualShaderNodeBillboard::VisualShaderNodeBillboard() {
	simple_decl = false;
}
