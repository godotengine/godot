/*************************************************************************/
/*  visual_shader_nodes.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

/// CONSTANTS ///

class VisualShaderNodeScalarConstant : public VisualShaderNode {
	GDCLASS(VisualShaderNodeScalarConstant, VisualShaderNode)
	float constant;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_constant(float p_value);
	float get_constant() const;

	virtual Vector<StringName> get_editable_properties() const;

	VisualShaderNodeScalarConstant();
};

class VisualShaderNodeColorConstant : public VisualShaderNode {
	GDCLASS(VisualShaderNodeColorConstant, VisualShaderNode)
	Color constant;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_constant(Color p_value);
	Color get_constant() const;

	virtual Vector<StringName> get_editable_properties() const;

	VisualShaderNodeColorConstant();
};

class VisualShaderNodeVec3Constant : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVec3Constant, VisualShaderNode)
	Vector3 constant;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_constant(Vector3 p_value);
	Vector3 get_constant() const;

	virtual Vector<StringName> get_editable_properties() const;

	VisualShaderNodeVec3Constant();
};

class VisualShaderNodeTransformConstant : public VisualShaderNode {
	GDCLASS(VisualShaderNodeTransformConstant, VisualShaderNode)
	Transform constant;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_constant(Transform p_value);
	Transform get_constant() const;

	virtual Vector<StringName> get_editable_properties() const;

	VisualShaderNodeTransformConstant();
};

//////////////////////////////////

class VisualShaderNodeTexture : public VisualShaderNode {
	GDCLASS(VisualShaderNodeTexture, VisualShaderNode)
	Ref<Texture> texture;

public:
	enum Source {
		SOURCE_TEXTURE,
		SOURCE_SCREEN,
		SOURCE_2D_TEXTURE,
		SOURCE_2D_NORMAL
	};

	enum TextureType {
		TYPE_DATA,
		TYPE_COLOR,
		TYPE_NORMALMAP
	};

private:
	Source source;
	TextureType texture_type;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual Vector<VisualShader::DefaultTextureParam> get_default_texture_parameters(VisualShader::Type p_type, int p_id) const;
	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_source(Source p_source);
	Source get_source() const;

	void set_texture(Ref<Texture> p_value);
	Ref<Texture> get_texture() const;

	void set_texture_type(TextureType p_type);
	TextureType get_texture_type() const;

	virtual Vector<StringName> get_editable_properties() const;

	virtual String get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const;

	VisualShaderNodeTexture();
};

VARIANT_ENUM_CAST(VisualShaderNodeTexture::TextureType)
VARIANT_ENUM_CAST(VisualShaderNodeTexture::Source)

//////////////////////////////////

class VisualShaderNodeCubeMap : public VisualShaderNode {
	GDCLASS(VisualShaderNodeCubeMap, VisualShaderNode)
	Ref<CubeMap> cube_map;

public:
	enum TextureType {
		TYPE_DATA,
		TYPE_COLOR,
		TYPE_NORMALMAP
	};

private:
	TextureType texture_type;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual Vector<VisualShader::DefaultTextureParam> get_default_texture_parameters(VisualShader::Type p_type, int p_id) const;
	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_cube_map(Ref<CubeMap> p_value);
	Ref<CubeMap> get_cube_map() const;

	void set_texture_type(TextureType p_type);
	TextureType get_texture_type() const;

	virtual Vector<StringName> get_editable_properties() const;

	VisualShaderNodeCubeMap();
};

VARIANT_ENUM_CAST(VisualShaderNodeCubeMap::TextureType)
///////////////////////////////////////

class VisualShaderNodeScalarOp : public VisualShaderNode {
	GDCLASS(VisualShaderNodeScalarOp, VisualShaderNode)

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
		OP_ATAN2
	};

protected:
	Operator op;

	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_operator(Operator p_op);
	Operator get_operator() const;

	virtual Vector<StringName> get_editable_properties() const;

	VisualShaderNodeScalarOp();
};

VARIANT_ENUM_CAST(VisualShaderNodeScalarOp::Operator)

class VisualShaderNodeVectorOp : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorOp, VisualShaderNode)

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
		OP_CROSS

	};

protected:
	Operator op;

	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_operator(Operator p_op);
	Operator get_operator() const;

	virtual Vector<StringName> get_editable_properties() const;

	VisualShaderNodeVectorOp();
};

VARIANT_ENUM_CAST(VisualShaderNodeVectorOp::Operator)

class VisualShaderNodeColorOp : public VisualShaderNode {
	GDCLASS(VisualShaderNodeColorOp, VisualShaderNode)

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
		OP_HARD_LIGHT,
	};

protected:
	Operator op;

	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_operator(Operator p_op);
	Operator get_operator() const;

	virtual Vector<StringName> get_editable_properties() const;

	VisualShaderNodeColorOp();
};

VARIANT_ENUM_CAST(VisualShaderNodeColorOp::Operator)

class VisualShaderNodeTransformMult : public VisualShaderNode {
	GDCLASS(VisualShaderNodeTransformMult, VisualShaderNode)

public:
	enum Operator {
		OP_AxB,
		OP_BxA,
	};

protected:
	Operator op;

	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_operator(Operator p_op);
	Operator get_operator() const;

	virtual Vector<StringName> get_editable_properties() const;

	VisualShaderNodeTransformMult();
};

VARIANT_ENUM_CAST(VisualShaderNodeTransformMult::Operator)

class VisualShaderNodeTransformVecMult : public VisualShaderNode {
	GDCLASS(VisualShaderNodeTransformVecMult, VisualShaderNode)

public:
	enum Operator {
		OP_AxB,
		OP_BxA,
		OP_3x3_AxB,
		OP_3x3_BxA,
	};

protected:
	Operator op;

	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_operator(Operator p_op);
	Operator get_operator() const;

	virtual Vector<StringName> get_editable_properties() const;

	VisualShaderNodeTransformVecMult();
};

VARIANT_ENUM_CAST(VisualShaderNodeTransformVecMult::Operator)

///////////////////////////////////////

class VisualShaderNodeScalarFunc : public VisualShaderNode {
	GDCLASS(VisualShaderNodeScalarFunc, VisualShaderNode)

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
	};

protected:
	Function func;

	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_function(Function p_func);
	Function get_function() const;

	virtual Vector<StringName> get_editable_properties() const;

	VisualShaderNodeScalarFunc();
};

VARIANT_ENUM_CAST(VisualShaderNodeScalarFunc::Function)

///////////////////////////////////////

class VisualShaderNodeVectorFunc : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorFunc, VisualShaderNode)

public:
	enum Function {
		FUNC_NORMALIZE,
		FUNC_SATURATE,
		FUNC_NEGATE,
		FUNC_RECIPROCAL,
		FUNC_RGB2HSV,
		FUNC_HSV2RGB,
	};

protected:
	Function func;

	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	void set_function(Function p_func);
	Function get_function() const;

	virtual Vector<StringName> get_editable_properties() const;

	VisualShaderNodeVectorFunc();
};

VARIANT_ENUM_CAST(VisualShaderNodeVectorFunc::Function)

///////////////////////////////////////

class VisualShaderNodeDotProduct : public VisualShaderNode {
	GDCLASS(VisualShaderNodeDotProduct, VisualShaderNode)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeDotProduct();
};

///////////////////////////////////////

class VisualShaderNodeVectorLen : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorLen, VisualShaderNode)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeVectorLen();
};

///////////////////////////////////////

class VisualShaderNodeScalarInterp : public VisualShaderNode {
	GDCLASS(VisualShaderNodeScalarInterp, VisualShaderNode)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeScalarInterp();
};

///////////////////////////////////////

class VisualShaderNodeVectorInterp : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorInterp, VisualShaderNode)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeVectorInterp();
};

///////////////////////////////////////

class VisualShaderNodeVectorCompose : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorCompose, VisualShaderNode)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeVectorCompose();
};

///////////////////////////////////////

class VisualShaderNodeTransformCompose : public VisualShaderNode {
	GDCLASS(VisualShaderNodeTransformCompose, VisualShaderNode)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeTransformCompose();
};

///////////////////////////////////////

class VisualShaderNodeVectorDecompose : public VisualShaderNode {
	GDCLASS(VisualShaderNodeVectorDecompose, VisualShaderNode)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeVectorDecompose();
};

///////////////////////////////////////

class VisualShaderNodeTransformDecompose : public VisualShaderNode {
	GDCLASS(VisualShaderNodeTransformDecompose, VisualShaderNode)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeTransformDecompose();
};

///////////////////////////////////////

class VisualShaderNodeScalarUniform : public VisualShaderNodeUniform {
	GDCLASS(VisualShaderNodeScalarUniform, VisualShaderNodeUniform)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeScalarUniform();
};

class VisualShaderNodeColorUniform : public VisualShaderNodeUniform {
	GDCLASS(VisualShaderNodeColorUniform, VisualShaderNodeUniform)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeColorUniform();
};

class VisualShaderNodeVec3Uniform : public VisualShaderNodeUniform {
	GDCLASS(VisualShaderNodeVec3Uniform, VisualShaderNodeUniform)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeVec3Uniform();
};

class VisualShaderNodeTransformUniform : public VisualShaderNodeUniform {
	GDCLASS(VisualShaderNodeTransformUniform, VisualShaderNodeUniform)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeTransformUniform();
};

//////////////////////////////////

class VisualShaderNodeTextureUniform : public VisualShaderNodeUniform {
	GDCLASS(VisualShaderNodeTextureUniform, VisualShaderNodeUniform)
public:
	enum TextureType {
		TYPE_DATA,
		TYPE_COLOR,
		TYPE_NORMALMAP,
		TYPE_ANISO,
	};

	enum ColorDefault {
		COLOR_DEFAULT_WHITE,
		COLOR_DEFAULT_BLACK
	};

private:
	TextureType texture_type;
	ColorDefault color_default;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const;
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	Vector<StringName> get_editable_properties() const;

	void set_texture_type(TextureType p_type);
	TextureType get_texture_type() const;

	void set_color_default(ColorDefault p_default);
	ColorDefault get_color_default() const;

	VisualShaderNodeTextureUniform();
};

VARIANT_ENUM_CAST(VisualShaderNodeTextureUniform::TextureType)
VARIANT_ENUM_CAST(VisualShaderNodeTextureUniform::ColorDefault)

//////////////////////////////////

class VisualShaderNodeCubeMapUniform : public VisualShaderNode {
	GDCLASS(VisualShaderNodeCubeMapUniform, VisualShaderNode)

public:
	virtual String get_caption() const;

	virtual int get_input_port_count() const;
	virtual PortType get_input_port_type(int p_port) const;
	virtual String get_input_port_name(int p_port) const;

	virtual int get_output_port_count() const;
	virtual PortType get_output_port_type(int p_port) const;
	virtual String get_output_port_name(int p_port) const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars) const; //if no output is connected, the output var passed will be empty. if no input is connected and input is NIL, the input var passed will be empty

	VisualShaderNodeCubeMapUniform();
};

#endif // VISUAL_SHADER_NODES_H
