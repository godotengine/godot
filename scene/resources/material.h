/*************************************************************************/
/*  material.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef MATERIAL_H
#define MATERIAL_H

#include "servers/visual_server.h"
#include "scene/resources/texture.h"
#include "scene/resources/shader.h"
#include "resource.h"
#include "servers/visual/shader_language.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class Material : public Resource {

	OBJ_TYPE(Material,Resource);	
	RES_BASE_EXTENSION("mtl");
	OBJ_SAVE_TYPE( Material );

public:

	enum Flag {
		FLAG_VISIBLE = VS::MATERIAL_FLAG_VISIBLE,
		FLAG_DOUBLE_SIDED = VS::MATERIAL_FLAG_DOUBLE_SIDED,
		FLAG_INVERT_FACES = VS::MATERIAL_FLAG_INVERT_FACES,
		FLAG_UNSHADED = VS::MATERIAL_FLAG_UNSHADED,
		FLAG_ONTOP = VS::MATERIAL_FLAG_ONTOP,
		FLAG_LIGHTMAP_ON_UV2 = VS::MATERIAL_FLAG_LIGHTMAP_ON_UV2,
		FLAG_COLOR_ARRAY_SRGB = VS::MATERIAL_FLAG_COLOR_ARRAY_SRGB,
		FLAG_MAX = VS::MATERIAL_FLAG_MAX
	};

	enum BlendMode {
		BLEND_MODE_MIX = VS::MATERIAL_BLEND_MODE_MIX,
		BLEND_MODE_MUL = VS::MATERIAL_BLEND_MODE_MUL,
		BLEND_MODE_ADD = VS::MATERIAL_BLEND_MODE_ADD,
		BLEND_MODE_SUB = VS::MATERIAL_BLEND_MODE_SUB,
		BLEND_MODE_PREMULT_ALPHA = VS::MATERIAL_BLEND_MODE_PREMULT_ALPHA,

	};

	enum DepthDrawMode {
		DEPTH_DRAW_ALWAYS = VS::MATERIAL_DEPTH_DRAW_ALWAYS,
		DEPTH_DRAW_OPAQUE_ONLY = VS::MATERIAL_DEPTH_DRAW_OPAQUE_ONLY,
		DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA = VS::MATERIAL_DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA,
		DEPTH_DRAW_NEVER = VS::MATERIAL_DEPTH_DRAW_NEVER
	};



private:
	BlendMode blend_mode;
	bool flags[VS::MATERIAL_FLAG_MAX];
	float line_width;
	DepthDrawMode depth_draw_mode;
protected:
	RID material;

	static void _bind_methods();

public:
	void set_flag(Flag p_flag,bool p_enabled);
	bool get_flag(Flag p_flag) const;

	void set_blend_mode(BlendMode p_blend_mode);
	BlendMode get_blend_mode() const;

	void set_depth_draw_mode(DepthDrawMode p_depth_draw_mode);
	DepthDrawMode get_depth_draw_mode() const;

	void set_line_width(float p_width);
	float get_line_width() const;

	virtual RID get_rid() const;

	Material(const RID& p_rid=RID());
	virtual ~Material();
};

VARIANT_ENUM_CAST( Material::Flag );
VARIANT_ENUM_CAST( Material::DepthDrawMode );

VARIANT_ENUM_CAST( Material::BlendMode );


class FixedMaterial : public Material {

	OBJ_TYPE( FixedMaterial, Material );
	REVERSE_GET_PROPERTY_LIST
public:

	enum Parameter {
		PARAM_DIFFUSE=VS::FIXED_MATERIAL_PARAM_DIFFUSE,
		PARAM_DETAIL=VS::FIXED_MATERIAL_PARAM_DETAIL,
		PARAM_SPECULAR=VS::FIXED_MATERIAL_PARAM_SPECULAR,
		PARAM_EMISSION=VS::FIXED_MATERIAL_PARAM_EMISSION,
		PARAM_SPECULAR_EXP=VS::FIXED_MATERIAL_PARAM_SPECULAR_EXP,
		PARAM_GLOW=VS::FIXED_MATERIAL_PARAM_GLOW,
		PARAM_NORMAL=VS::FIXED_MATERIAL_PARAM_NORMAL,
		PARAM_SHADE_PARAM=VS::FIXED_MATERIAL_PARAM_SHADE_PARAM,
		PARAM_MAX=VS::FIXED_MATERIAL_PARAM_MAX
	};


	enum TexCoordMode {

		TEXCOORD_UV=VS::FIXED_MATERIAL_TEXCOORD_UV,
		TEXCOORD_UV_TRANSFORM=VS::FIXED_MATERIAL_TEXCOORD_UV_TRANSFORM,
		TEXCOORD_UV2=VS::FIXED_MATERIAL_TEXCOORD_UV2,
		TEXCOORD_SPHERE=VS::FIXED_MATERIAL_TEXCOORD_SPHERE
	};

	enum FixedFlag {
		FLAG_USE_ALPHA=VS::FIXED_MATERIAL_FLAG_USE_ALPHA,
		FLAG_USE_COLOR_ARRAY=VS::FIXED_MATERIAL_FLAG_USE_COLOR_ARRAY,
		FLAG_USE_POINT_SIZE=VS::FIXED_MATERIAL_FLAG_USE_POINT_SIZE,
		FLAG_DISCARD_ALPHA=VS::FIXED_MATERIAL_FLAG_DISCARD_ALPHA,
		FLAG_USE_XY_NORMALMAP=VS::FIXED_MATERIAL_FLAG_USE_XY_NORMALMAP,
		FLAG_MAX=VS::FIXED_MATERIAL_FLAG_MAX
	};

	enum LightShader {

		LIGHT_SHADER_LAMBERT=VS::FIXED_MATERIAL_LIGHT_SHADER_LAMBERT,
		LIGHT_SHADER_WRAP=VS::FIXED_MATERIAL_LIGHT_SHADER_WRAP,
		LIGHT_SHADER_VELVET=VS::FIXED_MATERIAL_LIGHT_SHADER_VELVET,
		LIGHT_SHADER_TOON=VS::FIXED_MATERIAL_LIGHT_SHADER_TOON
	};

private:


	struct Node {

		int param;
		int mult;
		int tex;
	};

	Variant param[PARAM_MAX];
	Ref<Texture> texture_param[PARAM_MAX];
	TexCoordMode texture_texcoord[PARAM_MAX];
	LightShader light_shader;
	bool fixed_flags[FLAG_MAX];
	float point_size;


	Transform uv_transform;

protected:


	static void _bind_methods();


public:

	void set_fixed_flag(FixedFlag p_flag, bool p_value);
	bool get_fixed_flag(FixedFlag p_flag) const;

	void set_parameter(Parameter p_parameter, const Variant& p_value);
	Variant get_parameter(Parameter p_parameter) const;

	void set_texture(Parameter p_parameter, Ref<Texture> p_texture);
	Ref<Texture> get_texture(Parameter p_parameter) const;

	void set_texcoord_mode(Parameter p_parameter, TexCoordMode p_mode);
	TexCoordMode get_texcoord_mode(Parameter p_parameter) const;

	void set_light_shader(LightShader p_shader);
	LightShader get_light_shader() const;

	void set_uv_transform(const Transform& p_transform);
	Transform get_uv_transform() const;

	void set_point_size(float p_transform);
	float get_point_size() const;

	FixedMaterial();
	~FixedMaterial();

};



VARIANT_ENUM_CAST( FixedMaterial::Parameter );
VARIANT_ENUM_CAST( FixedMaterial::TexCoordMode );
VARIANT_ENUM_CAST( FixedMaterial::FixedFlag );
VARIANT_ENUM_CAST( FixedMaterial::LightShader );

class ShaderMaterial : public Material {

	OBJ_TYPE( ShaderMaterial, Material );

	Ref<Shader> shader;



	void _shader_changed();
	static void _shader_parse(void*p_self,ShaderLanguage::ProgramNode*p_node);

protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:

	void set_shader(const Ref<Shader>& p_shader);
	Ref<Shader> get_shader() const;

	void set_shader_param(const StringName& p_param,const Variant& p_value);
	Variant get_shader_param(const StringName& p_param) const;

	void get_argument_options(const StringName& p_function,int p_idx,List<String>*r_options) const;

	ShaderMaterial();
};

//////////////////////






#endif
