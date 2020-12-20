/*************************************************************************/
/*  visual_shader_nodes.h                                                */
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

#ifndef VISUAL_SHADER_NODES_H
#define VISUAL_SHADER_NODES_H

#include "scene/resources/visual_shader.h"

///////////////////////////////////////
/// CONSTANTS
///////////////////////////////////////

class VisualShaderNodeConstant : public VisualShaderNode {
	GDCLASS(VisualShaderNodeConstant, VisualShaderNode);

public:
	virtual String get_caption() const override = 0;

	virtual int get_input_port_count() const override = 0;
	virtual PortType get_input_port_type(int p_port) const override = 0;
	virtual String get_input_port_name(int p_port) const override = 0;

	virtual int get_output_port_count() const override = 0;
	virtual PortType get_output_port_type(int p_port) const override = 0;
	virtual String get_output_port_name(int p_port) const override = 0;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override = 0;

	VisualShaderNodeConstant();
};

class VisualShaderNodeFloatConstant : public VisualShaderNodeConstant {
	GDCLASS(VisualShaderNodeFloatConstant, VisualShaderNodeConstant);
	float constant = 0.0f;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_constant(float p_value);
	float get_constant() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeFloatConstant();
};

///////////////////////////////////////

class VisualShaderNodeIntConstant : public VisualShaderNodeConstant {
	GDCLASS(VisualShaderNodeIntConstant, VisualShaderNodeConstant);
	int constant = 0;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_constant(int p_value);
	int get_constant() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeIntConstant();
};

///////////////////////////////////////

class VisualShaderNodeBooleanConstant : public VisualShaderNodeConstant {
	GDCLASS(VisualShaderNodeBooleanConstant, VisualShaderNodeConstant);
	bool constant = false;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_constant(bool p_value);
	bool get_constant() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeBooleanConstant();
};

///////////////////////////////////////

class VisualShaderNodeColorConstant : public VisualShaderNodeConstant {
	GDCLASS(VisualShaderNodeColorConstant, VisualShaderNodeConstant);
	Color constant = Color(1, 1, 1, 1);

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;
	virtual bool is_output_port_expandable(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_constant(Color p_value);
	Color get_constant() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeColorConstant();
};

///////////////////////////////////////

class VisualShaderNodeVec3Constant : public VisualShaderNodeConstant {
	GDCLASS(VisualShaderNodeVec3Constant, VisualShaderNodeConstant);
	Vector3 constant;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_constant(Vector3 p_value);
	Vector3 get_constant() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeVec3Constant();
};

///////////////////////////////////////

class VisualShaderNodeTransformConstant : public VisualShaderNodeConstant {
	GDCLASS(VisualShaderNodeTransformConstant, VisualShaderNodeConstant);
	Transform constant;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_constant(Transform p_value);
	Transform get_constant() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeTransformConstant();
};

///////////////////////////////////////
/// TEXTURES
///////////////////////////////////////

class VisualShaderNodeTexture : public VisualShaderNode {
	GDCLASS(VisualShaderNodeTexture, VisualShaderNode);
	Ref<Texture2D> texture;

public:
	enum Source {
		SOURCE_TEXTURE,
		SOURCE_SCREEN,
		SOURCE_2D_TEXTURE,
		SOURCE_2D_NORMAL,
		SOURCE_DEPTH,
		SOURCE_PORT,
	};

	enum TextureType {
		TYPE_DATA,
		TYPE_COLOR,
		TYPE_NORMAL_MAP,
	};

private:
	Source source = SOURCE_TEXTURE;
	TextureType texture_type = TYPE_DATA;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;
	virtual bool is_output_port_expandable(int p_port) const override;

	virtual String get_input_port_default_hint(int p_port) const override;

	virtual Vector<VisualShader::DefaultTextureParam> get_default_texture_parameters(VisualShader::Type p_type, int p_id) const override;
	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_source(Source p_source);
	Source get_source() const;

	void set_texture(Ref<Texture2D> p_value);
	Ref<Texture2D> get_texture() const;

	void set_texture_type(TextureType p_type);
	TextureType get_texture_type() const;

	virtual Vector<StringName> get_editable_properties() const override;

	virtual String get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const override;

	VisualShaderNodeTexture();
};

VARIANT_ENUM_CAST(VisualShaderNodeTexture::TextureType)
VARIANT_ENUM_CAST(VisualShaderNodeTexture::Source)

///////////////////////////////////////

class VisualShaderNodeCurveTexture : public VisualShaderNodeResizableBase {
	GDCLASS(VisualShaderNodeCurveTexture, VisualShaderNodeResizableBase);
	Ref<CurveTexture> texture;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual Vector<VisualShader::DefaultTextureParam> get_default_texture_parameters(VisualShader::Type p_type, int p_id) const override;
	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_texture(Ref<CurveTexture> p_value);
	Ref<CurveTexture> get_texture() const;

	virtual Vector<StringName> get_editable_properties() const override;
	virtual bool is_use_prop_slots() const override;

	VisualShaderNodeCurveTexture();
};

///////////////////////////////////////

class VisualShaderNodeSample3D : public VisualShaderNode {
	GDCLASS(VisualShaderNodeSample3D, VisualShaderNode);

public:
	enum Source {
		SOURCE_TEXTURE,
		SOURCE_PORT,
	};

protected:
	Source source = SOURCE_TEXTURE;

	static void _bind_methods();

public:
	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;
	virtual String get_input_port_default_hint(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;
	virtual bool is_output_port_expandable(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_source(Source p_source);
	Source get_source() const;

	virtual String get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const override;

	VisualShaderNodeSample3D();
};

VARIANT_ENUM_CAST(VisualShaderNodeSample3D::Source)

class VisualShaderNodeTexture2DArray : public VisualShaderNodeSample3D {
	GDCLASS(VisualShaderNodeTexture2DArray, VisualShaderNodeSample3D);
	Ref<Texture2DArray> texture;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual String get_input_port_name(int p_port) const override;

	virtual Vector<VisualShader::DefaultTextureParam> get_default_texture_parameters(VisualShader::Type p_type, int p_id) const override;
	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;

	void set_texture_array(Ref<Texture2DArray> p_value);
	Ref<Texture2DArray> get_texture_array() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeTexture2DArray();
};

class VisualShaderNodeTexture3D : public VisualShaderNodeSample3D {
	GDCLASS(VisualShaderNodeTexture3D, VisualShaderNodeSample3D);
	Ref<Texture3D> texture;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual String get_input_port_name(int p_port) const override;

	virtual Vector<VisualShader::DefaultTextureParam> get_default_texture_parameters(VisualShader::Type p_type, int p_id) const override;
	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;

	void set_texture(Ref<Texture3D> p_value);
	Ref<Texture3D> get_texture() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeTexture3D();
};

class VisualShaderNodeCubemap : public VisualShaderNode {
	GDCLASS(VisualShaderNodeCubemap, VisualShaderNode);
	Ref<Cubemap> cube_map;

public:
	enum Source {
		SOURCE_TEXTURE,
		SOURCE_PORT
	};

	enum TextureType {
		TYPE_DATA,
		TYPE_COLOR,
		TYPE_NORMAL_MAP
	};

private:
	Source source = SOURCE_TEXTURE;
	TextureType texture_type = TYPE_DATA;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;
	virtual String get_input_port_default_hint(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;
	virtual bool is_output_port_expandable(int p_port) const override;

	virtual Vector<VisualShader::DefaultTextureParam> get_default_texture_parameters(VisualShader::Type p_type, int p_id) const override;
	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_source(Source p_source);
	Source get_source() const;

	void set_cube_map(Ref<Cubemap> p_value);
	Ref<Cubemap> get_cube_map() const;

	void set_texture_type(TextureType p_type);
	TextureType get_texture_type() const;

	virtual Vector<StringName> get_editable_properties() const override;
	virtual String get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const override;

	VisualShaderNodeCubemap();
};

VARIANT_ENUM_CAST(VisualShaderNodeCubemap::TextureType)
VARIANT_ENUM_CAST(VisualShaderNodeCubemap::Source)

///////////////////////////////////////
/// OPS
///////////////////////////////////////

class VisualShaderNodeFloatOp : public VisualShaderNode {
	GDCLASS(VisualShaderNodeFloatOp, VisualShaderNode);

public:
	enum Operator {
		OP_ADD,
		OP_SUB,
		OP_MUL,
		OP_DIV,
		OP_MOD,
		OP_POW,
		OP_MAX,
		OP_MIN,
		OP_ATAN2,
		OP_STEP
	};

protected:
	Operator op = OP_ADD;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_operator(Operator p_op);
	Operator get_operator() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeFloatOp();
};

VARIANT_ENUM_CAST(VisualShaderNodeFloatOp::Operator)

class VisualShaderNodeIntOp : public VisualShaderNode {
	GDCLASS(VisualShaderNodeIntOp, VisualShaderNode);

public:
	enum Operator {
		OP_ADD,
		OP_SUB,
		OP_MUL,
		OP_DIV,
		OP_MOD,
		OP_MAX,
		OP_MIN,
	};

protected:
	Operator op = OP_ADD;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_operator(Operator p_op);
	Operator get_operator() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeIntOp();
};

VARIANT_ENUM_CAST(VisualShaderNodeIntOp::Operator)

class VisualShaderNodeVectorOp : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorOp, VisualShaderNode);

public:
	enum Operator {
		OP_ADD,
		OP_SUB,
		OP_MUL,
		OP_DIV,
		OP_MOD,
		OP_POW,
		OP_MAX,
		OP_MIN,
		OP_CROSS,
		OP_ATAN2,
		OP_REFLECT,
		OP_STEP
	};

protected:
	Operator op = OP_ADD;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_operator(Operator p_op);
	Operator get_operator() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeVectorOp();
};

VARIANT_ENUM_CAST(VisualShaderNodeVectorOp::Operator)

///////////////////////////////////////

class VisualShaderNodeColorOp : public VisualShaderNode {
	GDCLASS(VisualShaderNodeColorOp, VisualShaderNode);

public:
	enum Operator {
		OP_SCREEN,
		OP_DIFFERENCE,
		OP_DARKEN,
		OP_LIGHTEN,
		OP_OVERLAY,
		OP_DODGE,
		OP_BURN,
		OP_SOFT_LIGHT,
		OP_HARD_LIGHT
	};

protected:
	Operator op = OP_SCREEN;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_operator(Operator p_op);
	Operator get_operator() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeColorOp();
};

VARIANT_ENUM_CAST(VisualShaderNodeColorOp::Operator)

///////////////////////////////////////
/// TRANSFORM-TRANSFORM MULTIPLICATION
///////////////////////////////////////

class VisualShaderNodeTransformMult : public VisualShaderNode {
	GDCLASS(VisualShaderNodeTransformMult, VisualShaderNode);

public:
	enum Operator {
		OP_AxB,
		OP_BxA,
		OP_AxB_COMP,
		OP_BxA_COMP
	};

protected:
	Operator op = OP_AxB;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_operator(Operator p_op);
	Operator get_operator() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeTransformMult();
};

VARIANT_ENUM_CAST(VisualShaderNodeTransformMult::Operator)

///////////////////////////////////////
/// TRANSFORM-VECTOR MULTIPLICATION
///////////////////////////////////////

class VisualShaderNodeTransformVecMult : public VisualShaderNode {
	GDCLASS(VisualShaderNodeTransformVecMult, VisualShaderNode);

public:
	enum Operator {
		OP_AxB,
		OP_BxA,
		OP_3x3_AxB,
		OP_3x3_BxA,
	};

protected:
	Operator op = OP_AxB;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_operator(Operator p_op);
	Operator get_operator() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeTransformVecMult();
};

VARIANT_ENUM_CAST(VisualShaderNodeTransformVecMult::Operator)

///////////////////////////////////////
/// FLOAT FUNC
///////////////////////////////////////

class VisualShaderNodeFloatFunc : public VisualShaderNode {
	GDCLASS(VisualShaderNodeFloatFunc, VisualShaderNode);

public:
	enum Function {
		FUNC_SIN,
		FUNC_COS,
		FUNC_TAN,
		FUNC_ASIN,
		FUNC_ACOS,
		FUNC_ATAN,
		FUNC_SINH,
		FUNC_COSH,
		FUNC_TANH,
		FUNC_LOG,
		FUNC_EXP,
		FUNC_SQRT,
		FUNC_ABS,
		FUNC_SIGN,
		FUNC_FLOOR,
		FUNC_ROUND,
		FUNC_CEIL,
		FUNC_FRAC,
		FUNC_SATURATE,
		FUNC_NEGATE,
		FUNC_ACOSH,
		FUNC_ASINH,
		FUNC_ATANH,
		FUNC_DEGREES,
		FUNC_EXP2,
		FUNC_INVERSE_SQRT,
		FUNC_LOG2,
		FUNC_RADIANS,
		FUNC_RECIPROCAL,
		FUNC_ROUNDEVEN,
		FUNC_TRUNC,
		FUNC_ONEMINUS
	};

protected:
	Function func = FUNC_SIGN;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_function(Function p_func);
	Function get_function() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeFloatFunc();
};

VARIANT_ENUM_CAST(VisualShaderNodeFloatFunc::Function)

///////////////////////////////////////
/// INT FUNC
///////////////////////////////////////

class VisualShaderNodeIntFunc : public VisualShaderNode {
	GDCLASS(VisualShaderNodeIntFunc, VisualShaderNode);

public:
	enum Function {
		FUNC_ABS,
		FUNC_NEGATE,
		FUNC_SIGN,
	};

protected:
	Function func = FUNC_SIGN;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_function(Function p_func);
	Function get_function() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeIntFunc();
};

VARIANT_ENUM_CAST(VisualShaderNodeIntFunc::Function)

///////////////////////////////////////
/// VECTOR FUNC
///////////////////////////////////////

class VisualShaderNodeVectorFunc : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorFunc, VisualShaderNode);

public:
	enum Function {
		FUNC_NORMALIZE,
		FUNC_SATURATE,
		FUNC_NEGATE,
		FUNC_RECIPROCAL,
		FUNC_RGB2HSV,
		FUNC_HSV2RGB,
		FUNC_ABS,
		FUNC_ACOS,
		FUNC_ACOSH,
		FUNC_ASIN,
		FUNC_ASINH,
		FUNC_ATAN,
		FUNC_ATANH,
		FUNC_CEIL,
		FUNC_COS,
		FUNC_COSH,
		FUNC_DEGREES,
		FUNC_EXP,
		FUNC_EXP2,
		FUNC_FLOOR,
		FUNC_FRAC,
		FUNC_INVERSE_SQRT,
		FUNC_LOG,
		FUNC_LOG2,
		FUNC_RADIANS,
		FUNC_ROUND,
		FUNC_ROUNDEVEN,
		FUNC_SIGN,
		FUNC_SIN,
		FUNC_SINH,
		FUNC_SQRT,
		FUNC_TAN,
		FUNC_TANH,
		FUNC_TRUNC,
		FUNC_ONEMINUS
	};

protected:
	Function func = FUNC_NORMALIZE;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_function(Function p_func);
	Function get_function() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeVectorFunc();
};

VARIANT_ENUM_CAST(VisualShaderNodeVectorFunc::Function)

///////////////////////////////////////
/// COLOR FUNC
///////////////////////////////////////

class VisualShaderNodeColorFunc : public VisualShaderNode {
	GDCLASS(VisualShaderNodeColorFunc, VisualShaderNode);

public:
	enum Function {
		FUNC_GRAYSCALE,
		FUNC_SEPIA
	};

protected:
	Function func = FUNC_GRAYSCALE;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_function(Function p_func);
	Function get_function() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeColorFunc();
};

VARIANT_ENUM_CAST(VisualShaderNodeColorFunc::Function)

///////////////////////////////////////
/// TRANSFORM FUNC
///////////////////////////////////////

class VisualShaderNodeTransformFunc : public VisualShaderNode {
	GDCLASS(VisualShaderNodeTransformFunc, VisualShaderNode);

public:
	enum Function {
		FUNC_INVERSE,
		FUNC_TRANSPOSE
	};

protected:
	Function func = FUNC_INVERSE;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_function(Function p_func);
	Function get_function() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeTransformFunc();
};

VARIANT_ENUM_CAST(VisualShaderNodeTransformFunc::Function)

///////////////////////////////////////
/// DOT
///////////////////////////////////////

class VisualShaderNodeDotProduct : public VisualShaderNode {
	GDCLASS(VisualShaderNodeDotProduct, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeDotProduct();
};

///////////////////////////////////////
/// LENGTH
///////////////////////////////////////

class VisualShaderNodeVectorLen : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorLen, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeVectorLen();
};

///////////////////////////////////////
/// DETERMINANT
///////////////////////////////////////

class VisualShaderNodeDeterminant : public VisualShaderNode {
	GDCLASS(VisualShaderNodeDeterminant, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeDeterminant();
};

///////////////////////////////////////
/// CLAMP
///////////////////////////////////////

class VisualShaderNodeClamp : public VisualShaderNode {
	GDCLASS(VisualShaderNodeClamp, VisualShaderNode);

public:
	enum OpType {
		OP_TYPE_FLOAT,
		OP_TYPE_INT,
		OP_TYPE_VECTOR,
		OP_TYPE_MAX,
	};

protected:
	OpType op_type = OP_TYPE_FLOAT;
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	void set_op_type(OpType p_type);
	OpType get_op_type() const;

	virtual Vector<StringName> get_editable_properties() const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeClamp();
};

VARIANT_ENUM_CAST(VisualShaderNodeClamp::OpType)

///////////////////////////////////////
/// DERIVATIVE FUNCTIONS
///////////////////////////////////////

class VisualShaderNodeScalarDerivativeFunc : public VisualShaderNode {
	GDCLASS(VisualShaderNodeScalarDerivativeFunc, VisualShaderNode);

public:
	enum Function {
		FUNC_SUM,
		FUNC_X,
		FUNC_Y
	};

protected:
	Function func = FUNC_SUM;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_function(Function p_func);
	Function get_function() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeScalarDerivativeFunc();
};

VARIANT_ENUM_CAST(VisualShaderNodeScalarDerivativeFunc::Function)

///////////////////////////////////////

class VisualShaderNodeVectorDerivativeFunc : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorDerivativeFunc, VisualShaderNode);

public:
	enum Function {
		FUNC_SUM,
		FUNC_X,
		FUNC_Y
	};

protected:
	Function func = FUNC_SUM;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_function(Function p_func);
	Function get_function() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeVectorDerivativeFunc();
};

VARIANT_ENUM_CAST(VisualShaderNodeVectorDerivativeFunc::Function)

///////////////////////////////////////
/// FACEFORWARD
///////////////////////////////////////

class VisualShaderNodeFaceForward : public VisualShaderNode {
	GDCLASS(VisualShaderNodeFaceForward, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeFaceForward();
};

///////////////////////////////////////
/// OUTER PRODUCT
///////////////////////////////////////

class VisualShaderNodeOuterProduct : public VisualShaderNode {
	GDCLASS(VisualShaderNodeOuterProduct, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeOuterProduct();
};

///////////////////////////////////////
/// STEP
///////////////////////////////////////

class VisualShaderNodeStep : public VisualShaderNode {
	GDCLASS(VisualShaderNodeStep, VisualShaderNode);

public:
	enum OpType {
		OP_TYPE_SCALAR,
		OP_TYPE_VECTOR,
		OP_TYPE_VECTOR_SCALAR,
		OP_TYPE_MAX,
	};

protected:
	OpType op_type = OP_TYPE_SCALAR;
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	void set_op_type(OpType p_type);
	OpType get_op_type() const;

	virtual Vector<StringName> get_editable_properties() const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeStep();
};

VARIANT_ENUM_CAST(VisualShaderNodeStep::OpType)

///////////////////////////////////////
/// SMOOTHSTEP
///////////////////////////////////////

class VisualShaderNodeSmoothStep : public VisualShaderNode {
	GDCLASS(VisualShaderNodeSmoothStep, VisualShaderNode);

public:
	enum OpType {
		OP_TYPE_SCALAR,
		OP_TYPE_VECTOR,
		OP_TYPE_VECTOR_SCALAR,
		OP_TYPE_MAX,
	};

protected:
	OpType op_type = OP_TYPE_SCALAR;
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	void set_op_type(OpType p_type);
	OpType get_op_type() const;

	virtual Vector<StringName> get_editable_properties() const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeSmoothStep();
};

VARIANT_ENUM_CAST(VisualShaderNodeSmoothStep::OpType)

///////////////////////////////////////
/// DISTANCE
///////////////////////////////////////

class VisualShaderNodeVectorDistance : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorDistance, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeVectorDistance();
};

///////////////////////////////////////
/// REFRACT
///////////////////////////////////////

class VisualShaderNodeVectorRefract : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorRefract, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeVectorRefract();
};

///////////////////////////////////////
/// MIX
///////////////////////////////////////

class VisualShaderNodeMix : public VisualShaderNode {
	GDCLASS(VisualShaderNodeMix, VisualShaderNode);

public:
	enum OpType {
		OP_TYPE_SCALAR,
		OP_TYPE_VECTOR,
		OP_TYPE_VECTOR_SCALAR,
		OP_TYPE_MAX,
	};

protected:
	OpType op_type = OP_TYPE_SCALAR;
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	void set_op_type(OpType p_type);
	OpType get_op_type() const;

	virtual Vector<StringName> get_editable_properties() const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeMix();
};

VARIANT_ENUM_CAST(VisualShaderNodeMix::OpType)

///////////////////////////////////////
/// COMPOSE
///////////////////////////////////////

class VisualShaderNodeVectorCompose : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorCompose, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeVectorCompose();
};

///////////////////////////////////////

class VisualShaderNodeTransformCompose : public VisualShaderNode {
	GDCLASS(VisualShaderNodeTransformCompose, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeTransformCompose();
};

///////////////////////////////////////
/// DECOMPOSE
///////////////////////////////////////

class VisualShaderNodeVectorDecompose : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorDecompose, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeVectorDecompose();
};

///////////////////////////////////////

class VisualShaderNodeTransformDecompose : public VisualShaderNode {
	GDCLASS(VisualShaderNodeTransformDecompose, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeTransformDecompose();
};

///////////////////////////////////////
/// UNIFORMS
///////////////////////////////////////

class VisualShaderNodeFloatUniform : public VisualShaderNodeUniform {
	GDCLASS(VisualShaderNodeFloatUniform, VisualShaderNodeUniform);

public:
	enum Hint {
		HINT_NONE,
		HINT_RANGE,
		HINT_RANGE_STEP,
	};

private:
	Hint hint = HINT_NONE;
	float hint_range_min = 0.0f;
	float hint_range_max = 1.0f;
	float hint_range_step = 0.1f;
	bool default_value_enabled = false;
	float default_value = 0.0f;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	virtual bool is_show_prop_names() const override;
	virtual bool is_use_prop_slots() const override;

	void set_hint(Hint p_hint);
	Hint get_hint() const;

	void set_min(float p_value);
	float get_min() const;

	void set_max(float p_value);
	float get_max() const;

	void set_step(float p_value);
	float get_step() const;

	void set_default_value_enabled(bool p_enabled);
	bool is_default_value_enabled() const;

	void set_default_value(float p_value);
	float get_default_value() const;

	bool is_qualifier_supported(Qualifier p_qual) const override;
	bool is_convertible_to_constant() const override;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeFloatUniform();
};

VARIANT_ENUM_CAST(VisualShaderNodeFloatUniform::Hint)

class VisualShaderNodeIntUniform : public VisualShaderNodeUniform {
	GDCLASS(VisualShaderNodeIntUniform, VisualShaderNodeUniform);

public:
	enum Hint {
		HINT_NONE,
		HINT_RANGE,
		HINT_RANGE_STEP,
	};

private:
	Hint hint = HINT_NONE;
	int hint_range_min = 0;
	int hint_range_max = 100;
	int hint_range_step = 1;
	bool default_value_enabled = false;
	int default_value = 0;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	virtual bool is_show_prop_names() const override;
	virtual bool is_use_prop_slots() const override;

	void set_hint(Hint p_hint);
	Hint get_hint() const;

	void set_min(int p_value);
	int get_min() const;

	void set_max(int p_value);
	int get_max() const;

	void set_step(int p_value);
	int get_step() const;

	void set_default_value_enabled(bool p_enabled);
	bool is_default_value_enabled() const;

	void set_default_value(int p_value);
	int get_default_value() const;

	bool is_qualifier_supported(Qualifier p_qual) const override;
	bool is_convertible_to_constant() const override;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeIntUniform();
};

VARIANT_ENUM_CAST(VisualShaderNodeIntUniform::Hint)

///////////////////////////////////////

class VisualShaderNodeBooleanUniform : public VisualShaderNodeUniform {
	GDCLASS(VisualShaderNodeBooleanUniform, VisualShaderNodeUniform);

private:
	bool default_value_enabled = false;
	bool default_value = false;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	virtual bool is_show_prop_names() const override;
	virtual bool is_use_prop_slots() const override;

	void set_default_value_enabled(bool p_enabled);
	bool is_default_value_enabled() const;

	void set_default_value(bool p_value);
	bool get_default_value() const;

	bool is_qualifier_supported(Qualifier p_qual) const override;
	bool is_convertible_to_constant() const override;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeBooleanUniform();
};

///////////////////////////////////////

class VisualShaderNodeColorUniform : public VisualShaderNodeUniform {
	GDCLASS(VisualShaderNodeColorUniform, VisualShaderNodeUniform);

private:
	bool default_value_enabled = false;
	Color default_value = Color(1.0, 1.0, 1.0, 1.0);

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	virtual bool is_show_prop_names() const override;

	void set_default_value_enabled(bool p_enabled);
	bool is_default_value_enabled() const;

	void set_default_value(const Color &p_value);
	Color get_default_value() const;

	bool is_qualifier_supported(Qualifier p_qual) const override;
	bool is_convertible_to_constant() const override;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeColorUniform();
};

///////////////////////////////////////

class VisualShaderNodeVec3Uniform : public VisualShaderNodeUniform {
	GDCLASS(VisualShaderNodeVec3Uniform, VisualShaderNodeUniform);

private:
	bool default_value_enabled = false;
	Vector3 default_value;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	virtual bool is_show_prop_names() const override;
	virtual bool is_use_prop_slots() const override;

	void set_default_value_enabled(bool p_enabled);
	bool is_default_value_enabled() const;

	void set_default_value(const Vector3 &p_value);
	Vector3 get_default_value() const;

	bool is_qualifier_supported(Qualifier p_qual) const override;
	bool is_convertible_to_constant() const override;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeVec3Uniform();
};

///////////////////////////////////////

class VisualShaderNodeTransformUniform : public VisualShaderNodeUniform {
	GDCLASS(VisualShaderNodeTransformUniform, VisualShaderNodeUniform);

private:
	bool default_value_enabled = false;
	Transform default_value = Transform(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	virtual bool is_show_prop_names() const override;
	virtual bool is_use_prop_slots() const override;

	void set_default_value_enabled(bool p_enabled);
	bool is_default_value_enabled() const;

	void set_default_value(const Transform &p_value);
	Transform get_default_value() const;

	bool is_qualifier_supported(Qualifier p_qual) const override;
	bool is_convertible_to_constant() const override;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeTransformUniform();
};

///////////////////////////////////////

class VisualShaderNodeTextureUniform : public VisualShaderNodeUniform {
	GDCLASS(VisualShaderNodeTextureUniform, VisualShaderNodeUniform);

public:
	enum TextureType {
		TYPE_DATA,
		TYPE_COLOR,
		TYPE_NORMAL_MAP,
		TYPE_ANISO,
	};

	enum ColorDefault {
		COLOR_DEFAULT_WHITE,
		COLOR_DEFAULT_BLACK
	};

protected:
	TextureType texture_type = TYPE_DATA;
	ColorDefault color_default = COLOR_DEFAULT_WHITE;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;
	virtual String get_input_port_default_hint(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	virtual bool is_code_generated() const override;

	Vector<StringName> get_editable_properties() const override;

	void set_texture_type(TextureType p_type);
	TextureType get_texture_type() const;

	void set_color_default(ColorDefault p_default);
	ColorDefault get_color_default() const;

	bool is_qualifier_supported(Qualifier p_qual) const override;
	bool is_convertible_to_constant() const override;

	VisualShaderNodeTextureUniform();
};

VARIANT_ENUM_CAST(VisualShaderNodeTextureUniform::TextureType)
VARIANT_ENUM_CAST(VisualShaderNodeTextureUniform::ColorDefault)

///////////////////////////////////////

class VisualShaderNodeTextureUniformTriplanar : public VisualShaderNodeTextureUniform {
	GDCLASS(VisualShaderNodeTextureUniformTriplanar, VisualShaderNodeTextureUniform);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual String get_input_port_default_hint(int p_port) const override;

	virtual String generate_global_per_node(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_global_per_func(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeTextureUniformTriplanar();
};

///////////////////////////////////////

class VisualShaderNodeTexture2DArrayUniform : public VisualShaderNodeTextureUniform {
	GDCLASS(VisualShaderNodeTexture2DArrayUniform, VisualShaderNodeTextureUniform);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String get_input_port_default_hint(int p_port) const override;
	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeTexture2DArrayUniform();
};

///////////////////////////////////////

class VisualShaderNodeTexture3DUniform : public VisualShaderNodeTextureUniform {
	GDCLASS(VisualShaderNodeTexture3DUniform, VisualShaderNodeTextureUniform);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String get_input_port_default_hint(int p_port) const override;
	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeTexture3DUniform();
};

///////////////////////////////////////

class VisualShaderNodeCubemapUniform : public VisualShaderNodeTextureUniform {
	GDCLASS(VisualShaderNodeCubemapUniform, VisualShaderNodeTextureUniform);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String get_input_port_default_hint(int p_port) const override;
	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeCubemapUniform();
};

///////////////////////////////////////
/// IF
///////////////////////////////////////

class VisualShaderNodeIf : public VisualShaderNode {
	GDCLASS(VisualShaderNodeIf, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeIf();
};

///////////////////////////////////////
/// SWITCH
///////////////////////////////////////

class VisualShaderNodeSwitch : public VisualShaderNode {
	GDCLASS(VisualShaderNodeSwitch, VisualShaderNode);

public:
	enum OpType {
		OP_TYPE_FLOAT,
		OP_TYPE_INT,
		OP_TYPE_VECTOR,
		OP_TYPE_BOOLEAN,
		OP_TYPE_TRANSFORM,
		OP_TYPE_MAX,
	};

protected:
	OpType op_type = OP_TYPE_FLOAT;

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	void set_op_type(OpType p_type);
	OpType get_op_type() const;

	virtual Vector<StringName> get_editable_properties() const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeSwitch();
};

VARIANT_ENUM_CAST(VisualShaderNodeSwitch::OpType)

///////////////////////////////////////
/// FRESNEL
///////////////////////////////////////

class VisualShaderNodeFresnel : public VisualShaderNode {
	GDCLASS(VisualShaderNodeFresnel, VisualShaderNode);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String get_input_port_default_hint(int p_port) const override;
	virtual bool is_generate_input_var(int p_port) const override;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeFresnel();
};

///////////////////////////////////////
/// Is
///////////////////////////////////////

class VisualShaderNodeIs : public VisualShaderNode {
	GDCLASS(VisualShaderNodeIs, VisualShaderNode);

public:
	enum Function {
		FUNC_IS_INF,
		FUNC_IS_NAN,
	};

protected:
	Function func = FUNC_IS_INF;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_function(Function p_func);
	Function get_function() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeIs();
};

VARIANT_ENUM_CAST(VisualShaderNodeIs::Function)

///////////////////////////////////////
/// Compare
///////////////////////////////////////

class VisualShaderNodeCompare : public VisualShaderNode {
	GDCLASS(VisualShaderNodeCompare, VisualShaderNode);

public:
	enum ComparisonType {
		CTYPE_SCALAR,
		CTYPE_SCALAR_INT,
		CTYPE_VECTOR,
		CTYPE_BOOLEAN,
		CTYPE_TRANSFORM,
	};

	enum Function {
		FUNC_EQUAL,
		FUNC_NOT_EQUAL,
		FUNC_GREATER_THAN,
		FUNC_GREATER_THAN_EQUAL,
		FUNC_LESS_THAN,
		FUNC_LESS_THAN_EQUAL,
	};

	enum Condition {
		COND_ALL,
		COND_ANY,
	};

protected:
	ComparisonType ctype = CTYPE_SCALAR;
	Function func = FUNC_EQUAL;
	Condition condition = COND_ALL;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_comparison_type(ComparisonType p_type);
	ComparisonType get_comparison_type() const;

	void set_function(Function p_func);
	Function get_function() const;

	void set_condition(Condition p_cond);
	Condition get_condition() const;

	virtual Vector<StringName> get_editable_properties() const override;
	virtual String get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const override;

	VisualShaderNodeCompare();
};

VARIANT_ENUM_CAST(VisualShaderNodeCompare::ComparisonType)
VARIANT_ENUM_CAST(VisualShaderNodeCompare::Function)
VARIANT_ENUM_CAST(VisualShaderNodeCompare::Condition)

class VisualShaderNodeMultiplyAdd : public VisualShaderNode {
	GDCLASS(VisualShaderNodeMultiplyAdd, VisualShaderNode);

public:
	enum OpType {
		OP_TYPE_SCALAR,
		OP_TYPE_VECTOR,
		OP_TYPE_MAX,
	};

protected:
	OpType op_type = OP_TYPE_SCALAR;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_op_type(OpType p_type);
	OpType get_op_type() const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeMultiplyAdd();
};

VARIANT_ENUM_CAST(VisualShaderNodeMultiplyAdd::OpType)

#endif // VISUAL_SHADER_NODES_H
