/**************************************************************************/
/*  visual_shader_nodes.cpp                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "visual_shader_nodes.h"

////////////// Scalar

String VisualShaderNodeScalarConstant::get_caption() const {
	return "Scalar";
}

int VisualShaderNodeScalarConstant::get_input_port_count() const {
	return 0;
}

VisualShaderNodeScalarConstant::PortType VisualShaderNodeScalarConstant::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarConstant::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeScalarConstant::get_output_port_count() const {
	return 1;
}

VisualShaderNodeScalarConstant::PortType VisualShaderNodeScalarConstant::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarConstant::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeScalarConstant::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = " + vformat("%.6f", constant) + ";\n";
}

void VisualShaderNodeScalarConstant::set_constant(float p_value) {
	constant = p_value;
	emit_changed();
}

float VisualShaderNodeScalarConstant::get_constant() const {
	return constant;
}

Vector<StringName> VisualShaderNodeScalarConstant::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("constant");
	return props;
}

void VisualShaderNodeScalarConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant", "value"), &VisualShaderNodeScalarConstant::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant"), &VisualShaderNodeScalarConstant::get_constant);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "constant"), "set_constant", "get_constant");
}

VisualShaderNodeScalarConstant::VisualShaderNodeScalarConstant() {
	constant = 0;
}

////////////// Boolean

String VisualShaderNodeBooleanConstant::get_caption() const {
	return "Boolean";
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
	return "\t" + p_output_vars[0] + " = " + (constant ? "true" : "false") + ";\n";
}

void VisualShaderNodeBooleanConstant::set_constant(bool p_value) {
	constant = p_value;
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
	ClassDB::bind_method(D_METHOD("set_constant", "value"), &VisualShaderNodeBooleanConstant::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant"), &VisualShaderNodeBooleanConstant::get_constant);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "constant"), "set_constant", "get_constant");
}

VisualShaderNodeBooleanConstant::VisualShaderNodeBooleanConstant() {
	constant = false;
}

////////////// Color

String VisualShaderNodeColorConstant::get_caption() const {
	return "Color";
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

String VisualShaderNodeColorConstant::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	code += "\t" + p_output_vars[0] + " = " + vformat("vec3(%.6f, %.6f, %.6f)", constant.r, constant.g, constant.b) + ";\n";
	code += "\t" + p_output_vars[1] + " = " + vformat("%.6f", constant.a) + ";\n";

	return code;
}

void VisualShaderNodeColorConstant::set_constant(Color p_value) {
	constant = p_value;
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
	ClassDB::bind_method(D_METHOD("set_constant", "value"), &VisualShaderNodeColorConstant::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant"), &VisualShaderNodeColorConstant::get_constant);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "constant"), "set_constant", "get_constant");
}

VisualShaderNodeColorConstant::VisualShaderNodeColorConstant() {
	constant = Color(1, 1, 1, 1);
}

////////////// Vector

String VisualShaderNodeVec3Constant::get_caption() const {
	return "Vector";
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
	return "\t" + p_output_vars[0] + " = " + vformat("vec3(%.6f, %.6f, %.6f)", constant.x, constant.y, constant.z) + ";\n";
}

void VisualShaderNodeVec3Constant::set_constant(Vector3 p_value) {
	constant = p_value;
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
	ClassDB::bind_method(D_METHOD("set_constant", "value"), &VisualShaderNodeVec3Constant::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant"), &VisualShaderNodeVec3Constant::get_constant);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "constant"), "set_constant", "get_constant");
}

VisualShaderNodeVec3Constant::VisualShaderNodeVec3Constant() {
}

////////////// Transform

String VisualShaderNodeTransformConstant::get_caption() const {
	return "Transform";
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
	Transform t = constant;
	t.basis.transpose();

	String code = "\t" + p_output_vars[0] + " = mat4(";
	code += vformat("vec4(%.6f, %.6f, %.6f, 0.0), ", t.basis[0].x, t.basis[0].y, t.basis[0].z);
	code += vformat("vec4(%.6f, %.6f, %.6f, 0.0), ", t.basis[1].x, t.basis[1].y, t.basis[1].z);
	code += vformat("vec4(%.6f, %.6f, %.6f, 0.0), ", t.basis[2].x, t.basis[2].y, t.basis[2].z);
	code += vformat("vec4(%.6f, %.6f, %.6f, 1.0));\n", t.origin.x, t.origin.y, t.origin.z);
	return code;
}

void VisualShaderNodeTransformConstant::set_constant(Transform p_value) {
	constant = p_value;
	emit_changed();
}

Transform VisualShaderNodeTransformConstant::get_constant() const {
	return constant;
}

Vector<StringName> VisualShaderNodeTransformConstant::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("constant");
	return props;
}

void VisualShaderNodeTransformConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant", "value"), &VisualShaderNodeTransformConstant::set_constant);
	ClassDB::bind_method(D_METHOD("get_constant"), &VisualShaderNodeTransformConstant::get_constant);

	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "constant"), "set_constant", "get_constant");
}

VisualShaderNodeTransformConstant::VisualShaderNodeTransformConstant() {
}

////////////// Texture

String VisualShaderNodeTexture::get_caption() const {
	return "Texture";
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

String VisualShaderNodeTexture::get_input_port_default_hint(int p_port) const {
	if (p_port == 0) {
		return "UV.xy";
	}
	return "";
}

static String make_unique_id(VisualShader::Type p_type, int p_id, const String &p_name) {
	static const char *typepf[VisualShader::TYPE_MAX] = { "vtx", "frg", "lgt" };
	return p_name + "_" + String(typepf[p_type]) + "_" + itos(p_id);
}

Vector<VisualShader::DefaultTextureParam> VisualShaderNodeTexture::get_default_texture_parameters(VisualShader::Type p_type, int p_id) const {
	VisualShader::DefaultTextureParam dtp;
	dtp.name = make_unique_id(p_type, p_id, "tex");
	dtp.param = texture;
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
			case TYPE_NORMALMAP:
				u += " : hint_normal";
				break;
		}
		return u + ";\n";
	}

	return String();
}

String VisualShaderNodeTexture::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	if (source == SOURCE_TEXTURE) {
		String id = make_unique_id(p_type, p_id, "tex");
		String code;
		if (p_input_vars[0] == String()) { // Use UV by default.

			if (p_input_vars[1] == String()) {
				code += "\tvec4 " + id + "_read = texture(" + id + ", UV.xy);\n";
			} else {
				code += "\tvec4 " + id + "_read = textureLod(" + id + ", UV.xy, " + p_input_vars[1] + ");\n";
			}

		} else if (p_input_vars[1] == String()) {
			//no lod
			code += "\tvec4 " + id + "_read = texture(" + id + ", " + p_input_vars[0] + ".xy);\n";
		} else {
			code += "\tvec4 " + id + "_read = textureLod(" + id + ", " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ");\n";
		}

		code += "\t" + p_output_vars[0] + " = " + id + "_read.rgb;\n";
		code += "\t" + p_output_vars[1] + " = " + id + "_read.a;\n";
		return code;
	}

	if (source == SOURCE_PORT) {
		String id = p_input_vars[2];

		String code;
		code += "\t{\n";
		if (id == String()) {
			code += "\t\tvec4 " + id + "_tex_read = vec4(0.0);\n";
		} else {
			if (p_input_vars[0] == String()) { // Use UV by default.

				if (p_input_vars[1] == String()) {
					code += "\t\tvec4 " + id + "_tex_read = texture(" + id + ", UV.xy);\n";
				} else {
					code += "\t\tvec4 " + id + "_tex_read = textureLod(" + id + ", UV.xy, " + p_input_vars[1] + ");\n";
				}

			} else if (p_input_vars[1] == String()) {
				//no lod
				code += "\t\tvec4 " + id + "_tex_read = texture(" + id + ", " + p_input_vars[0] + ".xy);\n";
			} else {
				code += "\t\tvec4 " + id + "_tex_read = textureLod(" + id + ", " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ");\n";
			}

			code += "\t\t" + p_output_vars[0] + " = " + id + "_tex_read.rgb;\n";
			code += "\t\t" + p_output_vars[1] + " = " + id + "_tex_read.a;\n";
		}
		code += "\t}\n";
		return code;
	}

	if (source == SOURCE_SCREEN && (p_mode == Shader::MODE_SPATIAL || p_mode == Shader::MODE_CANVAS_ITEM) && p_type == VisualShader::TYPE_FRAGMENT) {
		String code = "\t{\n";
		if (p_input_vars[0] == String() || p_for_preview) { // Use UV by default.

			if (p_input_vars[1] == String()) {
				code += "\t\tvec4 _tex_read = textureLod(SCREEN_TEXTURE, UV.xy, 0.0 );\n";
			} else {
				code += "\t\tvec4 _tex_read = textureLod(SCREEN_TEXTURE, UV.xy, " + p_input_vars[1] + ");\n";
			}

		} else if (p_input_vars[1] == String()) {
			//no lod
			code += "\t\tvec4 _tex_read = textureLod(SCREEN_TEXTURE, " + p_input_vars[0] + ".xy, 0.0);\n";
		} else {
			code += "\t\tvec4 _tex_read = textureLod(SCREEN_TEXTURE, " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ");\n";
		}

		code += "\t\t" + p_output_vars[0] + " = _tex_read.rgb;\n";
		code += "\t\t" + p_output_vars[1] + " = _tex_read.a;\n";
		code += "\t}\n";
		return code;
	}

	if (source == SOURCE_2D_TEXTURE && p_mode == Shader::MODE_CANVAS_ITEM && p_type == VisualShader::TYPE_FRAGMENT) {
		String code = "\t{\n";
		if (p_input_vars[0] == String()) { // Use UV by default.

			if (p_input_vars[1] == String()) {
				code += "\t\tvec4 _tex_read = texture(TEXTURE , UV.xy);\n";
			} else {
				code += "\t\tvec4 _tex_read = textureLod(TEXTURE, UV.xy, " + p_input_vars[1] + ");\n";
			}

		} else if (p_input_vars[1] == String()) {
			//no lod
			code += "\t\tvec4 _tex_read = texture(TEXTURE, " + p_input_vars[0] + ".xy);\n";
		} else {
			code += "\t\tvec4 _tex_read = textureLod(TEXTURE, " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ");\n";
		}

		code += "\t\t" + p_output_vars[0] + " = _tex_read.rgb;\n";
		code += "\t\t" + p_output_vars[1] + " = _tex_read.a;\n";
		code += "\t}\n";
		return code;
	}

	if (source == SOURCE_2D_NORMAL && p_mode == Shader::MODE_CANVAS_ITEM && p_type == VisualShader::TYPE_FRAGMENT) {
		String code = "\t{\n";
		if (p_input_vars[0] == String()) { // Use UV by default.

			if (p_input_vars[1] == String()) {
				code += "\t\tvec4 _tex_read = texture(NORMAL_TEXTURE, UV.xy);\n";
			} else {
				code += "\t\tvec4 _tex_read = textureLod(NORMAL_TEXTURE, UV.xy, " + p_input_vars[1] + ");\n";
			}

		} else if (p_input_vars[1] == String()) {
			//no lod
			code += "\t\tvec4 _tex_read = texture(NORMAL_TEXTURE, " + p_input_vars[0] + ".xy);\n";
		} else {
			code += "\t\tvec4 _tex_read = textureLod(NORMAL_TEXTURE, " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ");\n";
		}

		code += "\t\t" + p_output_vars[0] + " = _tex_read.rgb;\n";
		code += "\t\t" + p_output_vars[1] + " = _tex_read.a;\n";
		code += "\t}\n";
		return code;
	}

	if (p_for_preview) // DEPTH_TEXTURE is not supported in preview(canvas_item) shader
	{
		if (source == SOURCE_DEPTH) {
			String code;
			code += "\t" + p_output_vars[0] + " = 0.0;\n";
			code += "\t" + p_output_vars[1] + " = 1.0;\n";
			return code;
		}
	}

	if (source == SOURCE_DEPTH && p_mode == Shader::MODE_SPATIAL && p_type == VisualShader::TYPE_FRAGMENT) {
		String code = "\t{\n";
		if (p_input_vars[0] == String()) { // Use UV by default.

			if (p_input_vars[1] == String()) {
				code += "\t\tfloat _depth = texture(DEPTH_TEXTURE, UV.xy).r;\n";
			} else {
				code += "\t\tfloat _depth = textureLod(DEPTH_TEXTURE, UV.xy, " + p_input_vars[1] + ").r;\n";
			}

		} else if (p_input_vars[1] == String()) {
			//no lod
			code += "\t\tfloat _depth = texture(DEPTH_TEXTURE, " + p_input_vars[0] + ".xy).r;\n";
		} else {
			code += "\t\tfloat _depth = textureLod(DEPTH_TEXTURE, " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ").r;\n";
		}

		code += "\t\t" + p_output_vars[0] + " = _depth;\n";
		code += "\t\t" + p_output_vars[1] + " = 1.0;\n";
		code += "\t}\n";
		return code;
	} else if (source == SOURCE_DEPTH) {
		String code;
		code += "\t" + p_output_vars[0] + " = 0.0;\n";
		code += "\t" + p_output_vars[1] + " = 1.0;\n";
		return code;
	}

	//none
	String code;
	code += "\t" + p_output_vars[0] + " = vec3(0.0);\n";
	code += "\t" + p_output_vars[1] + " = 1.0;\n";
	return code;
}

void VisualShaderNodeTexture::set_source(Source p_source) {
	source = p_source;
	switch (source) {
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
	}
	emit_changed();
	emit_signal("editor_refresh_request");
}

VisualShaderNodeTexture::Source VisualShaderNodeTexture::get_source() const {
	return source;
}

void VisualShaderNodeTexture::set_texture(Ref<Texture> p_value) {
	texture = p_value;
	emit_changed();
}

Ref<Texture> VisualShaderNodeTexture::get_texture() const {
	return texture;
}

void VisualShaderNodeTexture::set_texture_type(TextureType p_type) {
	texture_type = p_type;
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
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_type", PROPERTY_HINT_ENUM, "Data,Color,Normalmap"), "set_texture_type", "get_texture_type");

	BIND_ENUM_CONSTANT(SOURCE_TEXTURE);
	BIND_ENUM_CONSTANT(SOURCE_SCREEN);
	BIND_ENUM_CONSTANT(SOURCE_2D_TEXTURE);
	BIND_ENUM_CONSTANT(SOURCE_2D_NORMAL);
	BIND_ENUM_CONSTANT(SOURCE_DEPTH);
	BIND_ENUM_CONSTANT(SOURCE_PORT);
	BIND_ENUM_CONSTANT(TYPE_DATA);
	BIND_ENUM_CONSTANT(TYPE_COLOR);
	BIND_ENUM_CONSTANT(TYPE_NORMALMAP);
}

VisualShaderNodeTexture::VisualShaderNodeTexture() {
	texture_type = TYPE_DATA;
	source = SOURCE_TEXTURE;
}

////////////// CubeMap

String VisualShaderNodeCubeMap::get_caption() const {
	return "CubeMap";
}

int VisualShaderNodeCubeMap::get_input_port_count() const {
	return 3;
}

VisualShaderNodeCubeMap::PortType VisualShaderNodeCubeMap::get_input_port_type(int p_port) const {
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

String VisualShaderNodeCubeMap::get_input_port_name(int p_port) const {
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

int VisualShaderNodeCubeMap::get_output_port_count() const {
	return 2;
}

VisualShaderNodeCubeMap::PortType VisualShaderNodeCubeMap::get_output_port_type(int p_port) const {
	return p_port == 0 ? PORT_TYPE_VECTOR : PORT_TYPE_SCALAR;
}

String VisualShaderNodeCubeMap::get_output_port_name(int p_port) const {
	return p_port == 0 ? "rgb" : "alpha";
}

Vector<VisualShader::DefaultTextureParam> VisualShaderNodeCubeMap::get_default_texture_parameters(VisualShader::Type p_type, int p_id) const {
	VisualShader::DefaultTextureParam dtp;
	dtp.name = make_unique_id(p_type, p_id, "cube");
	dtp.param = cube_map;
	Vector<VisualShader::DefaultTextureParam> ret;
	ret.push_back(dtp);
	return ret;
}

String VisualShaderNodeCubeMap::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	if (source == SOURCE_TEXTURE) {
		String u = "uniform samplerCube " + make_unique_id(p_type, p_id, "cube");
		switch (texture_type) {
			case TYPE_DATA:
				break;
			case TYPE_COLOR:
				u += " : hint_albedo";
				break;
			case TYPE_NORMALMAP:
				u += " : hint_normal";
				break;
		}
		return u + ";\n";
	}
	return String();
}

String VisualShaderNodeCubeMap::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	String id;
	if (source == SOURCE_TEXTURE) {
		id = make_unique_id(p_type, p_id, "cube");
	} else if (source == SOURCE_PORT) {
		id = p_input_vars[2];
	} else {
		return String();
	}

	code += "\t{\n";

	if (id == String()) {
		code += "\t\tvec4 " + id + "_read = vec4(0.0);\n";
		code += "\t\t" + p_output_vars[0] + " = " + id + "_read.rgb;\n";
		code += "\t\t" + p_output_vars[1] + " = " + id + "_read.a;\n";
		code += "\t}\n";
		return code;
	}

	if (p_input_vars[0] == String()) { // Use UV by default.

		if (p_input_vars[1] == String()) {
			code += "\t\tvec4 " + id + "_read = texture(" + id + " , vec3(UV, 0.0));\n";
		} else {
			code += "\t\tvec4 " + id + "_read = textureLod(" + id + " , vec3(UV, 0.0)" + " , " + p_input_vars[1] + " );\n";
		}

	} else if (p_input_vars[1] == String()) {
		//no lod
		code += "\t\tvec4 " + id + "_read = texture(" + id + ", " + p_input_vars[0] + ");\n";
	} else {
		code += "\t\tvec4 " + id + "_read = textureLod(" + id + ", " + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
	}
	code += "\t\t" + p_output_vars[0] + " = " + id + "_read.rgb;\n";
	code += "\t\t" + p_output_vars[1] + " = " + id + "_read.a;\n";
	code += "\t}\n";

	return code;
}

String VisualShaderNodeCubeMap::get_input_port_default_hint(int p_port) const {
	if (p_port == 0) {
		return "vec3(UV, 0.0)";
	}
	return "";
}

void VisualShaderNodeCubeMap::set_source(Source p_source) {
	source = p_source;
	emit_changed();
	emit_signal("editor_refresh_request");
}

VisualShaderNodeCubeMap::Source VisualShaderNodeCubeMap::get_source() const {
	return source;
}

void VisualShaderNodeCubeMap::set_cube_map(Ref<CubeMap> p_value) {
	cube_map = p_value;
	emit_changed();
}

Ref<CubeMap> VisualShaderNodeCubeMap::get_cube_map() const {
	return cube_map;
}

void VisualShaderNodeCubeMap::set_texture_type(TextureType p_type) {
	texture_type = p_type;
	emit_changed();
}

VisualShaderNodeCubeMap::TextureType VisualShaderNodeCubeMap::get_texture_type() const {
	return texture_type;
}

Vector<StringName> VisualShaderNodeCubeMap::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("source");
	if (source == SOURCE_TEXTURE) {
		props.push_back("cube_map");
		props.push_back("texture_type");
	}
	return props;
}

String VisualShaderNodeCubeMap::get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const {
	if (is_input_port_connected(2) && source != SOURCE_PORT) {
		return TTR("The sampler port is connected but not used. Consider changing the source to 'SamplerPort'.");
	}
	return String();
}

void VisualShaderNodeCubeMap::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_source", "value"), &VisualShaderNodeCubeMap::set_source);
	ClassDB::bind_method(D_METHOD("get_source"), &VisualShaderNodeCubeMap::get_source);

	ClassDB::bind_method(D_METHOD("set_cube_map", "value"), &VisualShaderNodeCubeMap::set_cube_map);
	ClassDB::bind_method(D_METHOD("get_cube_map"), &VisualShaderNodeCubeMap::get_cube_map);

	ClassDB::bind_method(D_METHOD("set_texture_type", "value"), &VisualShaderNodeCubeMap::set_texture_type);
	ClassDB::bind_method(D_METHOD("get_texture_type"), &VisualShaderNodeCubeMap::get_texture_type);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "source", PROPERTY_HINT_ENUM, "Texture,SamplerPort"), "set_source", "get_source");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "cube_map", PROPERTY_HINT_RESOURCE_TYPE, "CubeMap"), "set_cube_map", "get_cube_map");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_type", PROPERTY_HINT_ENUM, "Data,Color,Normalmap"), "set_texture_type", "get_texture_type");

	BIND_ENUM_CONSTANT(SOURCE_TEXTURE);
	BIND_ENUM_CONSTANT(SOURCE_PORT);

	BIND_ENUM_CONSTANT(TYPE_DATA);
	BIND_ENUM_CONSTANT(TYPE_COLOR);
	BIND_ENUM_CONSTANT(TYPE_NORMALMAP);
}

VisualShaderNodeCubeMap::VisualShaderNodeCubeMap() {
	texture_type = TYPE_DATA;
	source = SOURCE_TEXTURE;
	simple_decl = false;
}

////////////// Scalar Op

String VisualShaderNodeScalarOp::get_caption() const {
	return "ScalarOp";
}

int VisualShaderNodeScalarOp::get_input_port_count() const {
	return 2;
}

VisualShaderNodeScalarOp::PortType VisualShaderNodeScalarOp::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarOp::get_input_port_name(int p_port) const {
	return p_port == 0 ? "a" : "b";
}

int VisualShaderNodeScalarOp::get_output_port_count() const {
	return 1;
}

VisualShaderNodeScalarOp::PortType VisualShaderNodeScalarOp::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarOp::get_output_port_name(int p_port) const {
	return "op"; //no output port means the editor will be used as port
}

String VisualShaderNodeScalarOp::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code = "\t" + p_output_vars[0] + " = ";
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
	}

	return code;
}

void VisualShaderNodeScalarOp::set_operator(Operator p_op) {
	op = p_op;
	emit_changed();
}

VisualShaderNodeScalarOp::Operator VisualShaderNodeScalarOp::get_operator() const {
	return op;
}

Vector<StringName> VisualShaderNodeScalarOp::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("operator");
	return props;
}

void VisualShaderNodeScalarOp::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_operator", "op"), &VisualShaderNodeScalarOp::set_operator);
	ClassDB::bind_method(D_METHOD("get_operator"), &VisualShaderNodeScalarOp::get_operator);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operator", PROPERTY_HINT_ENUM, "Add,Sub,Multiply,Divide,Remainder,Power,Max,Min,Atan2,Step"), "set_operator", "get_operator");

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
}

VisualShaderNodeScalarOp::VisualShaderNodeScalarOp() {
	op = OP_ADD;
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
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
	String code = "\t" + p_output_vars[0] + " = ";
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
	}

	return code;
}

void VisualShaderNodeVectorOp::set_operator(Operator p_op) {
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

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operator", PROPERTY_HINT_ENUM, "Add,Sub,Multiply,Divide,Remainder,Power,Max,Min,Cross,Atan2,Reflect,Step"), "set_operator", "get_operator");

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
}

VisualShaderNodeVectorOp::VisualShaderNodeVectorOp() {
	op = OP_ADD;
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
			code += "\t" + p_output_vars[0] + " = vec3(1.0) - (vec3(1.0) - " + p_input_vars[0] + ") * (vec3(1.0) - " + p_input_vars[1] + ");\n";
		} break;
		case OP_DIFFERENCE: {
			code += "\t" + p_output_vars[0] + " = abs(" + p_input_vars[0] + " - " + p_input_vars[1] + ");\n";
		} break;
		case OP_DARKEN: {
			code += "\t" + p_output_vars[0] + " = min(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
		} break;
		case OP_LIGHTEN: {
			code += "\t" + p_output_vars[0] + " = max(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";

		} break;
		case OP_OVERLAY: {
			for (int i = 0; i < 3; i++) {
				code += "\t{\n";
				code += "\t\tfloat base = " + p_input_vars[0] + "." + axisn[i] + ";\n";
				code += "\t\tfloat blend = " + p_input_vars[1] + "." + axisn[i] + ";\n";
				code += "\t\tif (base < 0.5) {\n";
				code += "\t\t\t" + p_output_vars[0] + "." + axisn[i] + " = 2.0 * base * blend;\n";
				code += "\t\t} else {\n";
				code += "\t\t\t" + p_output_vars[0] + "." + axisn[i] + " = 1.0 - 2.0 * (1.0 - blend) * (1.0 - base);\n";
				code += "\t\t}\n";
				code += "\t}\n";
			}

		} break;
		case OP_DODGE: {
			code += "\t" + p_output_vars[0] + " = (" + p_input_vars[0] + ") / (vec3(1.0) - " + p_input_vars[1] + ");\n";

		} break;
		case OP_BURN: {
			code += "\t" + p_output_vars[0] + " = vec3(1.0) - (vec3(1.0) - " + p_input_vars[0] + ") / (" + p_input_vars[1] + ");\n";
		} break;
		case OP_SOFT_LIGHT: {
			for (int i = 0; i < 3; i++) {
				code += "\t{\n";
				code += "\t\tfloat base = " + p_input_vars[0] + "." + axisn[i] + ";\n";
				code += "\t\tfloat blend = " + p_input_vars[1] + "." + axisn[i] + ";\n";
				code += "\t\tif (base < 0.5) {\n";
				code += "\t\t\t" + p_output_vars[0] + "." + axisn[i] + " = (base * (blend + 0.5));\n";
				code += "\t\t} else {\n";
				code += "\t\t\t" + p_output_vars[0] + "." + axisn[i] + " = (1.0 - (1.0 - base) * (1.0 - (blend - 0.5)));\n";
				code += "\t\t}\n";
				code += "\t}\n";
			}

		} break;
		case OP_HARD_LIGHT: {
			for (int i = 0; i < 3; i++) {
				code += "\t{\n";
				code += "\t\tfloat base = " + p_input_vars[0] + "." + axisn[i] + ";\n";
				code += "\t\tfloat blend = " + p_input_vars[1] + "." + axisn[i] + ";\n";
				code += "\t\tif (base < 0.5) {\n";
				code += "\t\t\t" + p_output_vars[0] + "." + axisn[i] + " = (base * (2.0 * blend));\n";
				code += "\t\t} else {\n";
				code += "\t\t\t" + p_output_vars[0] + "." + axisn[i] + " = (1.0 - (1.0 - base) * (1.0 - 2.0 * (blend - 0.5)));\n";
				code += "\t\t}\n";
				code += "\t}\n";
			}

		} break;
	}

	return code;
}

void VisualShaderNodeColorOp::set_operator(Operator p_op) {
	op = p_op;
	switch (op) {
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
	}
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

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operator", PROPERTY_HINT_ENUM, "Screen,Difference,Darken,Lighten,Overlay,Dodge,Burn,SoftLight,HardLight"), "set_operator", "get_operator");

	BIND_ENUM_CONSTANT(OP_SCREEN);
	BIND_ENUM_CONSTANT(OP_DIFFERENCE);
	BIND_ENUM_CONSTANT(OP_DARKEN);
	BIND_ENUM_CONSTANT(OP_LIGHTEN);
	BIND_ENUM_CONSTANT(OP_OVERLAY);
	BIND_ENUM_CONSTANT(OP_DODGE);
	BIND_ENUM_CONSTANT(OP_BURN);
	BIND_ENUM_CONSTANT(OP_SOFT_LIGHT);
	BIND_ENUM_CONSTANT(OP_HARD_LIGHT);
}

VisualShaderNodeColorOp::VisualShaderNodeColorOp() {
	op = OP_SCREEN;
	set_input_port_default_value(0, Vector3());
	set_input_port_default_value(1, Vector3());
}

////////////// Transform Mult

String VisualShaderNodeTransformMult::get_caption() const {
	return "TransformMult";
}

int VisualShaderNodeTransformMult::get_input_port_count() const {
	return 2;
}

VisualShaderNodeTransformMult::PortType VisualShaderNodeTransformMult::get_input_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeTransformMult::get_input_port_name(int p_port) const {
	return p_port == 0 ? "a" : "b";
}

int VisualShaderNodeTransformMult::get_output_port_count() const {
	return 1;
}

VisualShaderNodeTransformMult::PortType VisualShaderNodeTransformMult::get_output_port_type(int p_port) const {
	return PORT_TYPE_TRANSFORM;
}

String VisualShaderNodeTransformMult::get_output_port_name(int p_port) const {
	return "mult"; //no output port means the editor will be used as port
}

String VisualShaderNodeTransformMult::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	if (op == OP_AxB) {
		return "\t" + p_output_vars[0] + " = " + p_input_vars[0] + " * " + p_input_vars[1] + ";\n";
	} else if (op == OP_BxA) {
		return "\t" + p_output_vars[0] + " = " + p_input_vars[1] + " * " + p_input_vars[0] + ";\n";
	} else if (op == OP_AxB_COMP) {
		return "\t" + p_output_vars[0] + " = matrixCompMult(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
	} else {
		return "\t" + p_output_vars[0] + " = matrixCompMult(" + p_input_vars[1] + ", " + p_input_vars[0] + ");\n";
	}
}

void VisualShaderNodeTransformMult::set_operator(Operator p_op) {
	op = p_op;
	emit_changed();
}

VisualShaderNodeTransformMult::Operator VisualShaderNodeTransformMult::get_operator() const {
	return op;
}

Vector<StringName> VisualShaderNodeTransformMult::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("operator");
	return props;
}

void VisualShaderNodeTransformMult::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_operator", "op"), &VisualShaderNodeTransformMult::set_operator);
	ClassDB::bind_method(D_METHOD("get_operator"), &VisualShaderNodeTransformMult::get_operator);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operator", PROPERTY_HINT_ENUM, "A x B,B x A,A x B(per component),B x A(per component)"), "set_operator", "get_operator");

	BIND_ENUM_CONSTANT(OP_AxB);
	BIND_ENUM_CONSTANT(OP_BxA);
	BIND_ENUM_CONSTANT(OP_AxB_COMP);
	BIND_ENUM_CONSTANT(OP_BxA_COMP);
}

VisualShaderNodeTransformMult::VisualShaderNodeTransformMult() {
	op = OP_AxB;
	set_input_port_default_value(0, Transform());
	set_input_port_default_value(1, Transform());
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
		return "\t" + p_output_vars[0] + " = (" + p_input_vars[0] + " * vec4(" + p_input_vars[1] + ", 1.0)).xyz;\n";
	} else if (op == OP_BxA) {
		return "\t" + p_output_vars[0] + " = (vec4(" + p_input_vars[1] + ", 1.0) * " + p_input_vars[0] + ").xyz;\n";
	} else if (op == OP_3x3_AxB) {
		return "\t" + p_output_vars[0] + " = (" + p_input_vars[0] + " * vec4(" + p_input_vars[1] + ", 0.0)).xyz;\n";
	} else {
		return "\t" + p_output_vars[0] + " = (vec4(" + p_input_vars[1] + ", 0.0) * " + p_input_vars[0] + ").xyz;\n";
	}
}

void VisualShaderNodeTransformVecMult::set_operator(Operator p_op) {
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
}

VisualShaderNodeTransformVecMult::VisualShaderNodeTransformVecMult() {
	op = OP_AxB;
	set_input_port_default_value(0, Transform());
	set_input_port_default_value(1, Vector3());
}

////////////// Scalar Func

String VisualShaderNodeScalarFunc::get_caption() const {
	return "ScalarFunc";
}

int VisualShaderNodeScalarFunc::get_input_port_count() const {
	return 1;
}

VisualShaderNodeScalarFunc::PortType VisualShaderNodeScalarFunc::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarFunc::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeScalarFunc::get_output_port_count() const {
	return 1;
}

VisualShaderNodeScalarFunc::PortType VisualShaderNodeScalarFunc::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarFunc::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeScalarFunc::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	static const char *scalar_func_id[FUNC_ONEMINUS + 1] = {
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

	return "\t" + p_output_vars[0] + " = " + String(scalar_func_id[func]).replace("$", p_input_vars[0]) + ";\n";
}

void VisualShaderNodeScalarFunc::set_function(Function p_func) {
	func = p_func;
	emit_changed();
}

VisualShaderNodeScalarFunc::Function VisualShaderNodeScalarFunc::get_function() const {
	return func;
}

Vector<StringName> VisualShaderNodeScalarFunc::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("function");
	return props;
}

void VisualShaderNodeScalarFunc::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_function", "func"), &VisualShaderNodeScalarFunc::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &VisualShaderNodeScalarFunc::get_function);

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
}

VisualShaderNodeScalarFunc::VisualShaderNodeScalarFunc() {
	func = FUNC_SIGN;
	set_input_port_default_value(0, 0.0);
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
	static const char *vec_func_id[FUNC_ONEMINUS + 1] = {
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
		code += "\t{\n";
		code += "\t\tvec3 c = " + p_input_vars[0] + ";\n";
		code += "\t\tvec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);\n";
		code += "\t\tvec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));\n";
		code += "\t\tvec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));\n";
		code += "\t\tfloat d = q.x - min(q.w, q.y);\n";
		code += "\t\tfloat e = 1.0e-10;\n";
		code += "\t\t" + p_output_vars[0] + " = vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);\n";
		code += "\t}\n";
	} else if (func == FUNC_HSV2RGB) {
		code += "\t{\n";
		code += "\t\tvec3 c = " + p_input_vars[0] + ";\n";
		code += "\t\tvec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);\n";
		code += "\t\tvec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);\n";
		code += "\t\t" + p_output_vars[0] + " = c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);\n";
		code += "\t}\n";

	} else {
		code += "\t" + p_output_vars[0] + " = " + String(vec_func_id[func]).replace("$", p_input_vars[0]) + ";\n";
	}

	return code;
}

void VisualShaderNodeVectorFunc::set_function(Function p_func) {
	func = p_func;
	if (func == FUNC_RGB2HSV) {
		simple_decl = false;
	} else if (func == FUNC_HSV2RGB) {
		simple_decl = false;
	} else {
		simple_decl = true;
	}
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
}

VisualShaderNodeVectorFunc::VisualShaderNodeVectorFunc() {
	func = FUNC_NORMALIZE;
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
			code += "\t{\n";
			code += "\t\tvec3 c = " + p_input_vars[0] + ";\n";
			code += "\t\tfloat max1 = max(c.r, c.g);\n";
			code += "\t\tfloat max2 = max(max1, c.b);\n";
			code += "\t\t" + p_output_vars[0] + " = vec3(max2, max2, max2);\n";
			code += "\t}\n";
			break;
		case FUNC_SEPIA:
			code += "\t{\n";
			code += "\t\tvec3 c = " + p_input_vars[0] + ";\n";
			code += "\t\tfloat r = (c.r * .393) + (c.g *.769) + (c.b * .189);\n";
			code += "\t\tfloat g = (c.r * .349) + (c.g *.686) + (c.b * .168);\n";
			code += "\t\tfloat b = (c.r * .272) + (c.g *.534) + (c.b * .131);\n";
			code += "\t\t" + p_output_vars[0] + " = vec3(r, g, b);\n";
			code += "\t}\n";
			break;
	}

	return code;
}

void VisualShaderNodeColorFunc::set_function(Function p_func) {
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
}

VisualShaderNodeColorFunc::VisualShaderNodeColorFunc() {
	func = FUNC_GRAYSCALE;
	set_input_port_default_value(0, Vector3());
	simple_decl = false;
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
	static const char *funcs[FUNC_TRANSPOSE + 1] = {
		"inverse($)",
		"transpose($)"
	};

	String code;
	code += "\t" + p_output_vars[0] + " = " + String(funcs[func]).replace("$", p_input_vars[0]) + ";\n";
	return code;
}

void VisualShaderNodeTransformFunc::set_function(Function p_func) {
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
}

VisualShaderNodeTransformFunc::VisualShaderNodeTransformFunc() {
	func = FUNC_INVERSE;
	set_input_port_default_value(0, Transform());
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
	return "\t" + p_output_vars[0] + " = dot(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
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
	return "\t" + p_output_vars[0] + " = length(" + p_input_vars[0] + ");\n";
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
	return "\t" + p_output_vars[0] + " = determinant(" + p_input_vars[0] + ");\n";
}

VisualShaderNodeDeterminant::VisualShaderNodeDeterminant() {
	set_input_port_default_value(0, Transform());
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
	static const char *funcs[FUNC_Y + 1] = {
		"fwidth($)",
		"dFdx($)",
		"dFdy($)"
	};

	String code;
	code += "\t" + p_output_vars[0] + " = " + String(funcs[func]).replace("$", p_input_vars[0]) + ";\n";
	return code;
}

void VisualShaderNodeScalarDerivativeFunc::set_function(Function p_func) {
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
}

VisualShaderNodeScalarDerivativeFunc::VisualShaderNodeScalarDerivativeFunc() {
	func = FUNC_SUM;
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
	static const char *funcs[FUNC_Y + 1] = {
		"fwidth($)",
		"dFdx($)",
		"dFdy($)"
	};

	String code;
	code += "\t" + p_output_vars[0] + " = " + String(funcs[func]).replace("$", p_input_vars[0]) + ";\n";
	return code;
}

void VisualShaderNodeVectorDerivativeFunc::set_function(Function p_func) {
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
}

VisualShaderNodeVectorDerivativeFunc::VisualShaderNodeVectorDerivativeFunc() {
	func = FUNC_SUM;
	set_input_port_default_value(0, Vector3());
}

////////////// Scalar Clamp

String VisualShaderNodeScalarClamp::get_caption() const {
	return "ScalarClamp";
}

int VisualShaderNodeScalarClamp::get_input_port_count() const {
	return 3;
}

VisualShaderNodeScalarClamp::PortType VisualShaderNodeScalarClamp::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarClamp::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "";
	} else if (p_port == 1) {
		return "min";
	} else if (p_port == 2) {
		return "max";
	}
	return "";
}

int VisualShaderNodeScalarClamp::get_output_port_count() const {
	return 1;
}

VisualShaderNodeScalarClamp::PortType VisualShaderNodeScalarClamp::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarClamp::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeScalarClamp::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = clamp(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

VisualShaderNodeScalarClamp::VisualShaderNodeScalarClamp() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, 1.0);
}

////////////// Vector Clamp

String VisualShaderNodeVectorClamp::get_caption() const {
	return "VectorClamp";
}

int VisualShaderNodeVectorClamp::get_input_port_count() const {
	return 3;
}

VisualShaderNodeVectorClamp::PortType VisualShaderNodeVectorClamp::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorClamp::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "";
	} else if (p_port == 1) {
		return "min";
	} else if (p_port == 2) {
		return "max";
	}
	return "";
}

int VisualShaderNodeVectorClamp::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorClamp::PortType VisualShaderNodeVectorClamp::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorClamp::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeVectorClamp::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = clamp(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

VisualShaderNodeVectorClamp::VisualShaderNodeVectorClamp() {
	set_input_port_default_value(0, Vector3(0, 0, 0));
	set_input_port_default_value(1, Vector3(0, 0, 0));
	set_input_port_default_value(2, Vector3(1, 1, 1));
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
	return "\t" + p_output_vars[0] + " = faceforward(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
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
	return "\t" + p_output_vars[0] + " = outerProduct(vec4(" + p_input_vars[0] + ", 0.0), vec4(" + p_input_vars[1] + ", 0.0));\n";
}

VisualShaderNodeOuterProduct::VisualShaderNodeOuterProduct() {
	set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0));
}

////////////// Vector-Scalar Step

String VisualShaderNodeVectorScalarStep::get_caption() const {
	return "VectorScalarStep";
}

int VisualShaderNodeVectorScalarStep::get_input_port_count() const {
	return 2;
}

VisualShaderNodeVectorScalarStep::PortType VisualShaderNodeVectorScalarStep::get_input_port_type(int p_port) const {
	if (p_port == 0) {
		return PORT_TYPE_SCALAR;
	}
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorScalarStep::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "edge";
	} else if (p_port == 1) {
		return "x";
	}
	return "";
}

int VisualShaderNodeVectorScalarStep::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorScalarStep::PortType VisualShaderNodeVectorScalarStep::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorScalarStep::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeVectorScalarStep::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = step(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
}

VisualShaderNodeVectorScalarStep::VisualShaderNodeVectorScalarStep() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0));
}

////////////// Scalar SmoothStep

String VisualShaderNodeScalarSmoothStep::get_caption() const {
	return "ScalarSmoothStep";
}

int VisualShaderNodeScalarSmoothStep::get_input_port_count() const {
	return 3;
}

VisualShaderNodeScalarSmoothStep::PortType VisualShaderNodeScalarSmoothStep::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarSmoothStep::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "edge0";
	} else if (p_port == 1) {
		return "edge1";
	} else if (p_port == 2) {
		return "x";
	}
	return "";
}

int VisualShaderNodeScalarSmoothStep::get_output_port_count() const {
	return 1;
}

VisualShaderNodeScalarSmoothStep::PortType VisualShaderNodeScalarSmoothStep::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarSmoothStep::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeScalarSmoothStep::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = smoothstep(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

VisualShaderNodeScalarSmoothStep::VisualShaderNodeScalarSmoothStep() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, 0.0);
}

////////////// Vector SmoothStep

String VisualShaderNodeVectorSmoothStep::get_caption() const {
	return "VectorSmoothStep";
}

int VisualShaderNodeVectorSmoothStep::get_input_port_count() const {
	return 3;
}

VisualShaderNodeVectorSmoothStep::PortType VisualShaderNodeVectorSmoothStep::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorSmoothStep::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "edge0";
	} else if (p_port == 1) {
		return "edge1";
	} else if (p_port == 2) {
		return "x";
	}
	return "";
}

int VisualShaderNodeVectorSmoothStep::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorSmoothStep::PortType VisualShaderNodeVectorSmoothStep::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorSmoothStep::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeVectorSmoothStep::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = smoothstep(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

VisualShaderNodeVectorSmoothStep::VisualShaderNodeVectorSmoothStep() {
	set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(2, Vector3(0.0, 0.0, 0.0));
}

////////////// Vector-Scalar SmoothStep

String VisualShaderNodeVectorScalarSmoothStep::get_caption() const {
	return "VectorScalarSmoothStep";
}

int VisualShaderNodeVectorScalarSmoothStep::get_input_port_count() const {
	return 3;
}

VisualShaderNodeVectorScalarSmoothStep::PortType VisualShaderNodeVectorScalarSmoothStep::get_input_port_type(int p_port) const {
	if (p_port == 0) {
		return PORT_TYPE_SCALAR;
	} else if (p_port == 1) {
		return PORT_TYPE_SCALAR;
	}
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorScalarSmoothStep::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "edge0";
	} else if (p_port == 1) {
		return "edge1";
	} else if (p_port == 2) {
		return "x";
	}
	return "";
}

int VisualShaderNodeVectorScalarSmoothStep::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorScalarSmoothStep::PortType VisualShaderNodeVectorScalarSmoothStep::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorScalarSmoothStep::get_output_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeVectorScalarSmoothStep::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = smoothstep(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

VisualShaderNodeVectorScalarSmoothStep::VisualShaderNodeVectorScalarSmoothStep() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, Vector3(0.0, 0.0, 0.0));
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
	return "\t" + p_output_vars[0] + " = distance(" + p_input_vars[0] + ", " + p_input_vars[1] + ");\n";
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
	return "\t" + p_output_vars[0] + " = refract(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

VisualShaderNodeVectorRefract::VisualShaderNodeVectorRefract() {
	set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(1, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(2, 0.0);
}

////////////// Scalar Mix

String VisualShaderNodeScalarInterp::get_caption() const {
	return "ScalarMix";
}

int VisualShaderNodeScalarInterp::get_input_port_count() const {
	return 3;
}

VisualShaderNodeScalarInterp::PortType VisualShaderNodeScalarInterp::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarInterp::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "a";
	} else if (p_port == 1) {
		return "b";
	} else {
		return "weight";
	}
}

int VisualShaderNodeScalarInterp::get_output_port_count() const {
	return 1;
}

VisualShaderNodeScalarInterp::PortType VisualShaderNodeScalarInterp::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarInterp::get_output_port_name(int p_port) const {
	return "mix";
}

String VisualShaderNodeScalarInterp::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = mix(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

VisualShaderNodeScalarInterp::VisualShaderNodeScalarInterp() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 1.0);
	set_input_port_default_value(2, 0.5);
}

////////////// Vector Mix

String VisualShaderNodeVectorInterp::get_caption() const {
	return "VectorMix";
}

int VisualShaderNodeVectorInterp::get_input_port_count() const {
	return 3;
}

VisualShaderNodeVectorInterp::PortType VisualShaderNodeVectorInterp::get_input_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorInterp::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "a";
	} else if (p_port == 1) {
		return "b";
	} else {
		return "weight";
	}
}

int VisualShaderNodeVectorInterp::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorInterp::PortType VisualShaderNodeVectorInterp::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorInterp::get_output_port_name(int p_port) const {
	return "mix";
}

String VisualShaderNodeVectorInterp::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = mix(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

VisualShaderNodeVectorInterp::VisualShaderNodeVectorInterp() {
	set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(1, Vector3(1.0, 1.0, 1.0));
	set_input_port_default_value(2, Vector3(0.5, 0.5, 0.5));
}

////////////// Vector Mix (by scalar)

String VisualShaderNodeVectorScalarMix::get_caption() const {
	return "VectorScalarMix";
}

int VisualShaderNodeVectorScalarMix::get_input_port_count() const {
	return 3;
}

VisualShaderNodeVectorScalarMix::PortType VisualShaderNodeVectorScalarMix::get_input_port_type(int p_port) const {
	if (p_port == 2) {
		return PORT_TYPE_SCALAR;
	}
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorScalarMix::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "a";
	} else if (p_port == 1) {
		return "b";
	} else {
		return "weight";
	}
}

int VisualShaderNodeVectorScalarMix::get_output_port_count() const {
	return 1;
}

VisualShaderNodeVectorScalarMix::PortType VisualShaderNodeVectorScalarMix::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeVectorScalarMix::get_output_port_name(int p_port) const {
	return "mix";
}

String VisualShaderNodeVectorScalarMix::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = mix(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
}

VisualShaderNodeVectorScalarMix::VisualShaderNodeVectorScalarMix() {
	set_input_port_default_value(0, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(1, Vector3(1.0, 1.0, 1.0));
	set_input_port_default_value(2, 0.5);
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
	return "\t" + p_output_vars[0] + " = vec3(" + p_input_vars[0] + ", " + p_input_vars[1] + ", " + p_input_vars[2] + ");\n";
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
	return "\t" + p_output_vars[0] + " = mat4(vec4(" + p_input_vars[0] + ", 0.0), vec4(" + p_input_vars[1] + ", 0.0), vec4(" + p_input_vars[2] + ", 0.0), vec4(" + p_input_vars[3] + ", 1.0));\n";
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
	code += "\t" + p_output_vars[0] + " = " + p_input_vars[0] + ".x;\n";
	code += "\t" + p_output_vars[1] + " = " + p_input_vars[0] + ".y;\n";
	code += "\t" + p_output_vars[2] + " = " + p_input_vars[0] + ".z;\n";
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
	code += "\t" + p_output_vars[0] + " = " + p_input_vars[0] + "[0].xyz;\n";
	code += "\t" + p_output_vars[1] + " = " + p_input_vars[0] + "[1].xyz;\n";
	code += "\t" + p_output_vars[2] + " = " + p_input_vars[0] + "[2].xyz;\n";
	code += "\t" + p_output_vars[3] + " = " + p_input_vars[0] + "[3].xyz;\n";
	return code;
}

VisualShaderNodeTransformDecompose::VisualShaderNodeTransformDecompose() {
	set_input_port_default_value(0, Transform());
}

////////////// Scalar Uniform

String VisualShaderNodeScalarUniform::get_caption() const {
	return "ScalarUniform";
}

int VisualShaderNodeScalarUniform::get_input_port_count() const {
	return 0;
}

VisualShaderNodeScalarUniform::PortType VisualShaderNodeScalarUniform::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarUniform::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeScalarUniform::get_output_port_count() const {
	return 1;
}

VisualShaderNodeScalarUniform::PortType VisualShaderNodeScalarUniform::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeScalarUniform::get_output_port_name(int p_port) const {
	return ""; //no output port means the editor will be used as port
}

String VisualShaderNodeScalarUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = "uniform float " + get_uniform_name();
	if (hint == HINT_RANGE) {
		code += " : hint_range(" + rtos(hint_range_min) + ", " + rtos(hint_range_max) + ")";
	} else if (hint == HINT_RANGE_STEP) {
		code += " : hint_range(" + rtos(hint_range_min) + ", " + rtos(hint_range_max) + ", " + rtos(hint_range_step) + ")";
	}
	if (default_value_enabled) {
		code += " = " + rtos(default_value);
	}
	code += ";\n";
	return code;
}

String VisualShaderNodeScalarUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
}

void VisualShaderNodeScalarUniform::set_hint(Hint p_hint) {
	ERR_FAIL_INDEX(int(p_hint), int(HINT_MAX));
	if (hint == p_hint) {
		return;
	}
	hint = p_hint;
	emit_changed();
}

VisualShaderNodeScalarUniform::Hint VisualShaderNodeScalarUniform::get_hint() const {
	return hint;
}

void VisualShaderNodeScalarUniform::set_min(float p_value) {
	if (Math::is_equal_approx(hint_range_min, p_value)) {
		return;
	}
	hint_range_min = p_value;
	emit_changed();
}

float VisualShaderNodeScalarUniform::get_min() const {
	return hint_range_min;
}

void VisualShaderNodeScalarUniform::set_max(float p_value) {
	if (Math::is_equal_approx(hint_range_max, p_value)) {
		return;
	}
	hint_range_max = p_value;
	emit_changed();
}

float VisualShaderNodeScalarUniform::get_max() const {
	return hint_range_max;
}

void VisualShaderNodeScalarUniform::set_step(float p_value) {
	if (Math::is_equal_approx(hint_range_step, p_value)) {
		return;
	}
	hint_range_step = p_value;
	emit_changed();
}

float VisualShaderNodeScalarUniform::get_step() const {
	return hint_range_step;
}

void VisualShaderNodeScalarUniform::set_default_value_enabled(bool p_default_value_enabled) {
	if (default_value_enabled == p_default_value_enabled) {
		return;
	}
	default_value_enabled = p_default_value_enabled;
	emit_changed();
}

bool VisualShaderNodeScalarUniform::is_default_value_enabled() const {
	return default_value_enabled;
}

void VisualShaderNodeScalarUniform::set_default_value(float p_default_value) {
	if (Math::is_equal_approx(default_value, p_default_value)) {
		return;
	}
	default_value = p_default_value;
	emit_changed();
}

float VisualShaderNodeScalarUniform::get_default_value() const {
	return default_value;
}

Vector<StringName> VisualShaderNodeScalarUniform::get_editable_properties() const {
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

void VisualShaderNodeScalarUniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_hint", "hint"), &VisualShaderNodeScalarUniform::set_hint);
	ClassDB::bind_method(D_METHOD("get_hint"), &VisualShaderNodeScalarUniform::get_hint);

	ClassDB::bind_method(D_METHOD("set_min", "value"), &VisualShaderNodeScalarUniform::set_min);
	ClassDB::bind_method(D_METHOD("get_min"), &VisualShaderNodeScalarUniform::get_min);

	ClassDB::bind_method(D_METHOD("set_max", "value"), &VisualShaderNodeScalarUniform::set_max);
	ClassDB::bind_method(D_METHOD("get_max"), &VisualShaderNodeScalarUniform::get_max);

	ClassDB::bind_method(D_METHOD("set_step", "value"), &VisualShaderNodeScalarUniform::set_step);
	ClassDB::bind_method(D_METHOD("get_step"), &VisualShaderNodeScalarUniform::get_step);

	ClassDB::bind_method(D_METHOD("set_default_value_enabled", "enabled"), &VisualShaderNodeScalarUniform::set_default_value_enabled);
	ClassDB::bind_method(D_METHOD("is_default_value_enabled"), &VisualShaderNodeScalarUniform::is_default_value_enabled);

	ClassDB::bind_method(D_METHOD("set_default_value", "value"), &VisualShaderNodeScalarUniform::set_default_value);
	ClassDB::bind_method(D_METHOD("get_default_value"), &VisualShaderNodeScalarUniform::get_default_value);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "hint", PROPERTY_HINT_ENUM, "None,Range,Range+Step"), "set_hint", "get_hint");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "min"), "set_min", "get_min");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max"), "set_max", "get_max");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "step"), "set_step", "get_step");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value_enabled"), "set_default_value_enabled", "is_default_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "default_value"), "set_default_value", "get_default_value");

	BIND_ENUM_CONSTANT(HINT_NONE);
	BIND_ENUM_CONSTANT(HINT_RANGE);
	BIND_ENUM_CONSTANT(HINT_RANGE_STEP);
	BIND_ENUM_CONSTANT(HINT_MAX);
}

VisualShaderNodeScalarUniform::VisualShaderNodeScalarUniform() {
	hint = HINT_NONE;
	hint_range_min = 0.0f;
	hint_range_max = 1.0f;
	hint_range_step = 0.1f;
	default_value_enabled = false;
	default_value = 0.0f;
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

String VisualShaderNodeBooleanUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = "uniform bool " + get_uniform_name();
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
	return "\t" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
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

void VisualShaderNodeBooleanUniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_default_value_enabled", "enabled"), &VisualShaderNodeBooleanUniform::set_default_value_enabled);
	ClassDB::bind_method(D_METHOD("is_default_value_enabled"), &VisualShaderNodeBooleanUniform::is_default_value_enabled);

	ClassDB::bind_method(D_METHOD("set_default_value", "value"), &VisualShaderNodeBooleanUniform::set_default_value);
	ClassDB::bind_method(D_METHOD("get_default_value"), &VisualShaderNodeBooleanUniform::get_default_value);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value_enabled"), "set_default_value_enabled", "is_default_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value"), "set_default_value", "get_default_value");
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
	default_value_enabled = false;
	default_value = false;
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

String VisualShaderNodeColorUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = "uniform vec4 " + get_uniform_name() + " : hint_color";
	if (default_value_enabled) {
		code += vformat(" = vec4(%.6f, %.6f, %.6f, %.6f)", default_value.r, default_value.g, default_value.b, default_value.a);
	}
	code += ";\n";
	return code;
}

String VisualShaderNodeColorUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code = "\t" + p_output_vars[0] + " = " + get_uniform_name() + ".rgb;\n";
	code += "\t" + p_output_vars[1] + " = " + get_uniform_name() + ".a;\n";
	return code;
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

void VisualShaderNodeColorUniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_default_value_enabled", "enabled"), &VisualShaderNodeColorUniform::set_default_value_enabled);
	ClassDB::bind_method(D_METHOD("is_default_value_enabled"), &VisualShaderNodeColorUniform::is_default_value_enabled);

	ClassDB::bind_method(D_METHOD("set_default_value", "value"), &VisualShaderNodeColorUniform::set_default_value);
	ClassDB::bind_method(D_METHOD("get_default_value"), &VisualShaderNodeColorUniform::get_default_value);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value_enabled"), "set_default_value_enabled", "is_default_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "default_value"), "set_default_value", "get_default_value");
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
	default_value_enabled = false;
	default_value = Color(1.0, 1.0, 1.0, 1.0);
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

String VisualShaderNodeVec3Uniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = "uniform vec3 " + get_uniform_name();
	if (default_value_enabled) {
		code += vformat(" = vec3(%.6f, %.6f, %.6f)", default_value.x, default_value.y, default_value.z);
	}
	code += ";\n";
	return code;
}

String VisualShaderNodeVec3Uniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "\t" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
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

void VisualShaderNodeVec3Uniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_default_value_enabled", "enabled"), &VisualShaderNodeVec3Uniform::set_default_value_enabled);
	ClassDB::bind_method(D_METHOD("is_default_value_enabled"), &VisualShaderNodeVec3Uniform::is_default_value_enabled);

	ClassDB::bind_method(D_METHOD("set_default_value", "value"), &VisualShaderNodeVec3Uniform::set_default_value);
	ClassDB::bind_method(D_METHOD("get_default_value"), &VisualShaderNodeVec3Uniform::get_default_value);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value_enabled"), "set_default_value_enabled", "is_default_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "default_value"), "set_default_value", "get_default_value");
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
	default_value_enabled = false;
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

String VisualShaderNodeTransformUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = "uniform mat4 " + get_uniform_name();
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
	return "\t" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
}

void VisualShaderNodeTransformUniform::set_default_value_enabled(bool p_enabled) {
	default_value_enabled = p_enabled;
	emit_changed();
}

bool VisualShaderNodeTransformUniform::is_default_value_enabled() const {
	return default_value_enabled;
}

void VisualShaderNodeTransformUniform::set_default_value(const Transform &p_value) {
	default_value = p_value;
	emit_changed();
}

Transform VisualShaderNodeTransformUniform::get_default_value() const {
	return default_value;
}

void VisualShaderNodeTransformUniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_default_value_enabled", "enabled"), &VisualShaderNodeTransformUniform::set_default_value_enabled);
	ClassDB::bind_method(D_METHOD("is_default_value_enabled"), &VisualShaderNodeTransformUniform::is_default_value_enabled);

	ClassDB::bind_method(D_METHOD("set_default_value", "value"), &VisualShaderNodeTransformUniform::set_default_value);
	ClassDB::bind_method(D_METHOD("get_default_value"), &VisualShaderNodeTransformUniform::get_default_value);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "default_value_enabled"), "set_default_value_enabled", "is_default_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "default_value"), "set_default_value", "get_default_value");
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
	default_value_enabled = false;
	default_value = Transform(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
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

bool VisualShaderNodeTextureUniform::is_code_generated() const {
	return is_output_port_connected(0) || is_output_port_connected(1); // rgb or alpha
}

bool VisualShaderNodeTextureUniform::is_show_prop_names() const {
	return false;
}

String VisualShaderNodeTextureUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = "uniform sampler2D " + get_uniform_name();

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
		case TYPE_NORMALMAP:
			code += " : hint_normal;\n";
			break;
		case TYPE_ANISO:
			code += " : hint_aniso;\n";
			break;
	}

	return code;
}

String VisualShaderNodeTextureUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String id = get_uniform_name();
	String code = "\t{\n";
	if (p_input_vars[0] == String()) { // Use UV by default.
		if (p_input_vars[1] == String()) {
			code += "\t\tvec4 n_tex_read = texture(" + id + ", UV.xy);\n";
		} else {
			code += "\t\tvec4 n_tex_read = textureLod(" + id + ", UV.xy, " + p_input_vars[1] + ");\n";
		}
	} else if (p_input_vars[1] == String()) {
		//no lod
		code += "\t\tvec4 n_tex_read = texture(" + id + ", " + p_input_vars[0] + ".xy);\n";
	} else {
		code += "\t\tvec4 n_tex_read = textureLod(" + id + ", " + p_input_vars[0] + ".xy, " + p_input_vars[1] + ");\n";
	}

	code += "\t\t" + p_output_vars[0] + " = n_tex_read.rgb;\n";
	code += "\t\t" + p_output_vars[1] + " = n_tex_read.a;\n";
	code += "\t}\n";
	return code;
}

void VisualShaderNodeTextureUniform::set_texture_type(TextureType p_type) {
	texture_type = p_type;
	emit_changed();
}

VisualShaderNodeTextureUniform::TextureType VisualShaderNodeTextureUniform::get_texture_type() const {
	return texture_type;
}

void VisualShaderNodeTextureUniform::set_color_default(ColorDefault p_default) {
	color_default = p_default;
	emit_changed();
}

VisualShaderNodeTextureUniform::ColorDefault VisualShaderNodeTextureUniform::get_color_default() const {
	return color_default;
}

Vector<StringName> VisualShaderNodeTextureUniform::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("texture_type");
	props.push_back("color_default");
	return props;
}

void VisualShaderNodeTextureUniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture_type", "type"), &VisualShaderNodeTextureUniform::set_texture_type);
	ClassDB::bind_method(D_METHOD("get_texture_type"), &VisualShaderNodeTextureUniform::get_texture_type);

	ClassDB::bind_method(D_METHOD("set_color_default", "type"), &VisualShaderNodeTextureUniform::set_color_default);
	ClassDB::bind_method(D_METHOD("get_color_default"), &VisualShaderNodeTextureUniform::get_color_default);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_type", PROPERTY_HINT_ENUM, "Data,Color,Normalmap,Aniso"), "set_texture_type", "get_texture_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "color_default", PROPERTY_HINT_ENUM, "White Default,Black Default"), "set_color_default", "get_color_default");

	BIND_ENUM_CONSTANT(TYPE_DATA);
	BIND_ENUM_CONSTANT(TYPE_COLOR);
	BIND_ENUM_CONSTANT(TYPE_NORMALMAP);
	BIND_ENUM_CONSTANT(TYPE_ANISO);

	BIND_ENUM_CONSTANT(COLOR_DEFAULT_WHITE);
	BIND_ENUM_CONSTANT(COLOR_DEFAULT_BLACK);
}

String VisualShaderNodeTextureUniform::get_input_port_default_hint(int p_port) const {
	if (p_port == 0) {
		return "UV.xy";
	}
	return "";
}

VisualShaderNodeTextureUniform::VisualShaderNodeTextureUniform() {
	texture_type = TYPE_DATA;
	color_default = COLOR_DEFAULT_WHITE;
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
	if (p_port == 0) {
		return PORT_TYPE_VECTOR;
	} else if (p_port == 1) {
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
	code += "\tvec4 triplanar_texture(sampler2D p_sampler, vec3 p_weights, vec3 p_triplanar_pos) {\n";
	code += "\t\tvec4 samp = vec4(0.0);\n";
	code += "\t\tsamp += texture(p_sampler, p_triplanar_pos.xy) * p_weights.z;\n";
	code += "\t\tsamp += texture(p_sampler, p_triplanar_pos.xz) * p_weights.y;\n";
	code += "\t\tsamp += texture(p_sampler, p_triplanar_pos.zy * vec2(-1.0, 1.0)) * p_weights.x;\n";
	code += "\t\treturn samp;\n";
	code += "\t}\n";
	code += "\n";
	code += "\tuniform vec3 triplanar_scale = vec3(1.0, 1.0, 1.0);\n";
	code += "\tuniform vec3 triplanar_offset;\n";
	code += "\tuniform float triplanar_sharpness = 0.5;\n";
	code += "\n";
	code += "\tvarying vec3 triplanar_power_normal;\n";
	code += "\tvarying vec3 triplanar_pos;\n";

	return code;
}

String VisualShaderNodeTextureUniformTriplanar::generate_global_per_func(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code;

	if (p_type == VisualShader::TYPE_VERTEX) {
		code += "\t// TRIPLANAR FUNCTION VERTEX CODE\n";
		code += "\t\ttriplanar_power_normal = pow(abs(NORMAL), vec3(triplanar_sharpness));\n";
		code += "\t\ttriplanar_power_normal /= dot(triplanar_power_normal, vec3(1.0));\n";
		code += "\t\ttriplanar_pos = VERTEX * triplanar_scale + triplanar_offset;\n";
		code += "\t\ttriplanar_pos *= vec3(1.0, -1.0, 1.0);\n";
	}

	return code;
}

String VisualShaderNodeTextureUniformTriplanar::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String id = get_uniform_name();
	String code = "\t{\n";

	if (p_input_vars[0] == String() && p_input_vars[1] == String()) {
		code += "\t\tvec4 n_tex_read = triplanar_texture( " + id + ", triplanar_power_normal, triplanar_pos );\n";
	} else if (p_input_vars[0] != String() && p_input_vars[1] == String()) {
		code += "\t\tvec4 n_tex_read = triplanar_texture( " + id + ", " + p_input_vars[0] + ", triplanar_pos );\n";
	} else if (p_input_vars[0] == String() && p_input_vars[1] != String()) {
		code += "\t\tvec4 n_tex_read = triplanar_texture( " + id + ", triplanar_power_normal," + p_input_vars[1] + " );\n";
	} else {
		code += "\t\tvec4 n_tex_read = triplanar_texture( " + id + ", " + p_input_vars[0] + ", " + p_input_vars[1] + " );\n";
	}

	code += "\t\t" + p_output_vars[0] + " = n_tex_read.rgb;\n";
	code += "\t\t" + p_output_vars[1] + " = n_tex_read.a;\n";
	code += "\t}\n";

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

////////////// CubeMap Uniform

String VisualShaderNodeCubeMapUniform::get_caption() const {
	return "CubeMapUniform";
}

int VisualShaderNodeCubeMapUniform::get_output_port_count() const {
	return 1;
}

VisualShaderNodeCubeMapUniform::PortType VisualShaderNodeCubeMapUniform::get_output_port_type(int p_port) const {
	return PORT_TYPE_SAMPLER;
}

String VisualShaderNodeCubeMapUniform::get_output_port_name(int p_port) const {
	return "samplerCube";
}

int VisualShaderNodeCubeMapUniform::get_input_port_count() const {
	return 0;
}

VisualShaderNodeCubeMapUniform::PortType VisualShaderNodeCubeMapUniform::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeCubeMapUniform::get_input_port_name(int p_port) const {
	return "";
}

String VisualShaderNodeCubeMapUniform::get_input_port_default_hint(int p_port) const {
	return "";
}

String VisualShaderNodeCubeMapUniform::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code = "uniform samplerCube " + get_uniform_name();

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
		case TYPE_NORMALMAP:
			code += " : hint_normal;\n";
			break;
		case TYPE_ANISO:
			code += " : hint_aniso;\n";
			break;
	}

	return code;
}

String VisualShaderNodeCubeMapUniform::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return String();
}

VisualShaderNodeCubeMapUniform::VisualShaderNodeCubeMapUniform() {
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
	code += "\tif(abs(" + p_input_vars[0] + " - " + p_input_vars[1] + ") < " + p_input_vars[2] + ")\n"; // abs(a - b) < tolerance eg. a == b
	code += "\t{\n";
	code += "\t\t" + p_output_vars[0] + " = " + p_input_vars[3] + ";\n";
	code += "\t}\n";
	code += "\telse if(" + p_input_vars[0] + " < " + p_input_vars[1] + ")\n"; // a < b
	code += "\t{\n";
	code += "\t\t" + p_output_vars[0] + " = " + p_input_vars[5] + ";\n";
	code += "\t}\n";
	code += "\telse\n"; // a > b (or a >= b if abs(a - b) < tolerance is false)
	code += "\t{\n";
	code += "\t\t" + p_output_vars[0] + " = " + p_input_vars[4] + ";\n";
	code += "\t}\n";
	return code;
}

VisualShaderNodeIf::VisualShaderNodeIf() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, CMP_EPSILON);
	set_input_port_default_value(3, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(4, Vector3(0.0, 0.0, 0.0));
	set_input_port_default_value(5, Vector3(0.0, 0.0, 0.0));
	simple_decl = false;
}

////////////// Switch

String VisualShaderNodeSwitch::get_caption() const {
	return "VectorSwitch";
}

int VisualShaderNodeSwitch::get_input_port_count() const {
	return 3;
}

VisualShaderNodeSwitch::PortType VisualShaderNodeSwitch::get_input_port_type(int p_port) const {
	if (p_port == 0) {
		return PORT_TYPE_BOOLEAN;
	}
	return PORT_TYPE_VECTOR;
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
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeSwitch::get_output_port_name(int p_port) const {
	return "result";
}

String VisualShaderNodeSwitch::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	code += "\tif(" + p_input_vars[0] + ")\n";
	code += "\t{\n";
	code += "\t\t" + p_output_vars[0] + " = " + p_input_vars[1] + ";\n";
	code += "\t}\n";
	code += "\telse\n";
	code += "\t{\n";
	code += "\t\t" + p_output_vars[0] + " = " + p_input_vars[2] + ";\n";
	code += "\t}\n";
	return code;
}

VisualShaderNodeSwitch::VisualShaderNodeSwitch() {
	set_input_port_default_value(0, false);
	set_input_port_default_value(1, Vector3(1.0, 1.0, 1.0));
	set_input_port_default_value(2, Vector3(0.0, 0.0, 0.0));
	simple_decl = false;
}

////////////// Switch(scalar)

String VisualShaderNodeScalarSwitch::get_caption() const {
	return "ScalarSwitch";
}

VisualShaderNodeScalarSwitch::PortType VisualShaderNodeScalarSwitch::get_input_port_type(int p_port) const {
	if (p_port == 0) {
		return PORT_TYPE_BOOLEAN;
	}
	return PORT_TYPE_SCALAR;
}

VisualShaderNodeScalarSwitch::PortType VisualShaderNodeScalarSwitch::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

VisualShaderNodeScalarSwitch::VisualShaderNodeScalarSwitch() {
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
	if (p_input_vars[0] == String()) {
		normal = "NORMAL";
	} else {
		normal = p_input_vars[0];
	}
	if (p_input_vars[1] == String()) {
		view = "VIEW";
	} else {
		view = p_input_vars[1];
	}

	if (is_input_port_connected(2)) {
		return "\t" + p_output_vars[0] + " = " + p_input_vars[2] + " ? (pow(clamp(dot(" + normal + ", " + view + "), 0.0, 1.0), " + p_input_vars[3] + ")) : (pow(1.0 - clamp(dot(" + normal + ", " + view + "), 0.0, 1.0), " + p_input_vars[3] + "));\n";
	} else {
		if (get_input_port_default_value(2)) {
			return "\t" + p_output_vars[0] + " = pow(clamp(dot(" + normal + ", " + view + "), 0.0, 1.0), " + p_input_vars[3] + ");\n";
		} else {
			return "\t" + p_output_vars[0] + " = pow(1.0 - clamp(dot(" + normal + ", " + view + "), 0.0, 1.0), " + p_input_vars[3] + ");\n";
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
	static const char *funcs[FUNC_IS_NAN + 1] = {
		"isinf($)",
		"isnan($)"
	};

	String code;
	code += "\t" + p_output_vars[0] + " = " + String(funcs[func]).replace("$", p_input_vars[0]) + ";\n";
	return code;
}

void VisualShaderNodeIs::set_function(Function p_func) {
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
}

VisualShaderNodeIs::VisualShaderNodeIs() {
	func = FUNC_IS_INF;
	set_input_port_default_value(0, 0.0);
}

////////////// Compare

String VisualShaderNodeCompare::get_caption() const {
	return "Compare";
}

int VisualShaderNodeCompare::get_input_port_count() const {
	if (ctype == CTYPE_SCALAR && (func == FUNC_EQUAL || func == FUNC_NOT_EQUAL)) {
		return 3;
	}
	return 2;
}

VisualShaderNodeCompare::PortType VisualShaderNodeCompare::get_input_port_type(int p_port) const {
	if (p_port == 2) {
		return PORT_TYPE_SCALAR;
	}
	switch (ctype) {
		case CTYPE_SCALAR:
			return PORT_TYPE_SCALAR;
		case CTYPE_VECTOR:
			return PORT_TYPE_VECTOR;
		case CTYPE_BOOLEAN:
			return PORT_TYPE_BOOLEAN;
		case CTYPE_TRANSFORM:
			return PORT_TYPE_TRANSFORM;
	}
	return PORT_TYPE_VECTOR;
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
	if (ctype == CTYPE_BOOLEAN || ctype == CTYPE_TRANSFORM) {
		if (func > FUNC_NOT_EQUAL) {
			return TTR("Invalid comparison function for that type.");
		}
	}

	return "";
}

String VisualShaderNodeCompare::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	static const char *ops[FUNC_LESS_THAN_EQUAL + 1] = {
		"==",
		"!=",
		">",
		">=",
		"<",
		"<=",
	};

	static const char *funcs[FUNC_LESS_THAN_EQUAL + 1] = {
		"equal($)",
		"notEqual($)",
		"greaterThan($)",
		"greaterThanEqual($)",
		"lessThan($)",
		"lessThanEqual($)",
	};

	static const char *conds[COND_ANY + 1] = {
		"all($)",
		"any($)",
	};

	String code;
	switch (ctype) {
		case CTYPE_SCALAR:
			if (func == FUNC_EQUAL) {
				code += "\t" + p_output_vars[0] + " = (abs(" + p_input_vars[0] + " - " + p_input_vars[1] + ") < " + p_input_vars[2] + ");";
			} else if (func == FUNC_NOT_EQUAL) {
				code += "\t" + p_output_vars[0] + " = !(abs(" + p_input_vars[0] + " - " + p_input_vars[1] + ") < " + p_input_vars[2] + ");";
			} else {
				code += "\t" + p_output_vars[0] + " = " + (p_input_vars[0] + "$" + p_input_vars[1]).replace("$", ops[func]) + ";\n";
			}
			break;

		case CTYPE_VECTOR:
			code += "\t{\n";
			code += "\t\tbvec3 _bv = " + String(funcs[func]).replace("$", p_input_vars[0] + ", " + p_input_vars[1]) + ";\n";
			code += "\t\t" + p_output_vars[0] + " = " + String(conds[condition]).replace("$", "_bv") + ";\n";
			code += "\t}\n";
			break;

		case CTYPE_BOOLEAN:
			if (func > FUNC_NOT_EQUAL) {
				return "\t" + p_output_vars[0] + " = false;\n";
			}
			code += "\t" + p_output_vars[0] + " = " + (p_input_vars[0] + " $ " + p_input_vars[1]).replace("$", ops[func]) + ";\n";
			break;

		case CTYPE_TRANSFORM:
			if (func > FUNC_NOT_EQUAL) {
				return "\t" + p_output_vars[0] + " = false;\n";
			}
			code += "\t" + p_output_vars[0] + " = " + (p_input_vars[0] + " $ " + p_input_vars[1]).replace("$", ops[func]) + ";\n";
			break;

		default:
			break;
	}
	return code;
}

void VisualShaderNodeCompare::set_comparison_type(ComparisonType p_type) {
	ctype = p_type;

	switch (ctype) {
		case CTYPE_SCALAR:
			set_input_port_default_value(0, 0.0);
			set_input_port_default_value(1, 0.0);
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
			set_input_port_default_value(0, Transform());
			set_input_port_default_value(1, Transform());
			simple_decl = true;
			break;
	}
	emit_changed();
}

VisualShaderNodeCompare::ComparisonType VisualShaderNodeCompare::get_comparison_type() const {
	return ctype;
}

void VisualShaderNodeCompare::set_function(Function p_func) {
	func = p_func;
	emit_changed();
}

VisualShaderNodeCompare::Function VisualShaderNodeCompare::get_function() const {
	return func;
}

void VisualShaderNodeCompare::set_condition(Condition p_cond) {
	condition = p_cond;
	emit_changed();
}

VisualShaderNodeCompare::Condition VisualShaderNodeCompare::get_condition() const {
	return condition;
}

Vector<StringName> VisualShaderNodeCompare::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("type");
	props.push_back("function");
	if (ctype == CTYPE_VECTOR) {
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

	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, "Scalar,Vector,Boolean,Transform"), "set_comparison_type", "get_comparison_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "function", PROPERTY_HINT_ENUM, "a == b,a != b,a > b,a >= b,a < b,a <= b"), "set_function", "get_function");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "condition", PROPERTY_HINT_ENUM, "All,Any"), "set_condition", "get_condition");

	BIND_ENUM_CONSTANT(CTYPE_SCALAR);
	BIND_ENUM_CONSTANT(CTYPE_VECTOR);
	BIND_ENUM_CONSTANT(CTYPE_BOOLEAN);
	BIND_ENUM_CONSTANT(CTYPE_TRANSFORM);

	BIND_ENUM_CONSTANT(FUNC_EQUAL);
	BIND_ENUM_CONSTANT(FUNC_NOT_EQUAL);
	BIND_ENUM_CONSTANT(FUNC_GREATER_THAN);
	BIND_ENUM_CONSTANT(FUNC_GREATER_THAN_EQUAL);
	BIND_ENUM_CONSTANT(FUNC_LESS_THAN);
	BIND_ENUM_CONSTANT(FUNC_LESS_THAN_EQUAL);

	BIND_ENUM_CONSTANT(COND_ALL);
	BIND_ENUM_CONSTANT(COND_ANY);
}

VisualShaderNodeCompare::VisualShaderNodeCompare() {
	ctype = CTYPE_SCALAR;
	func = FUNC_EQUAL;
	condition = COND_ALL;
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, CMP_EPSILON);
}
