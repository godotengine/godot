/*************************************************************************/
/*  shader_types.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "shader_types.h"

const Map<StringName, ShaderLanguage::FunctionInfo> &ShaderTypes::get_functions(VS::ShaderMode p_mode) {

	return shader_modes[p_mode].functions;
}

const Set<String> &ShaderTypes::get_modes(VS::ShaderMode p_mode) {

	return shader_modes[p_mode].modes;
}

const Set<String> &ShaderTypes::get_types() {
	return shader_types;
}

ShaderTypes *ShaderTypes::singleton = NULL;

ShaderTypes::ShaderTypes() {
	singleton = this;

	/*************** SPATIAL ***********************/

	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["VERTEX"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["TANGENT"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["BINORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["UV2"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["POINT_SIZE"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["INSTANCE_ID"] = ShaderLanguage::TYPE_INT;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["INSTANCE_CUSTOM"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["ROUGHNESS"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].can_discard = false;

	//builtins
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["WORLD_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["INV_CAMERA_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["CAMERA_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["PROJECTION_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["MODELVIEW_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["INV_PROJECTION_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["TIME"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["VIEWPORT_SIZE"] = ShaderLanguage::TYPE_VEC2;

	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["VERTEX"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["FRAGCOORD"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["FRONT_FACING"] = ShaderLanguage::TYPE_BOOL;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["TANGENT"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["BINORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["NORMALMAP"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["NORMALMAP_DEPTH"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["UV2"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["ALBEDO"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["ALPHA"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["METALLIC"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["SPECULAR"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["ROUGHNESS"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["RIM"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["RIM_TINT"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["CLEARCOAT"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["CLEARCOAT_GLOSS"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["ANISOTROPY"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["ANISOTROPY_FLOW"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["SSS_STRENGTH"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["TRANSMISSION"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["AO"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["AO_LIGHT_AFFECT"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["EMISSION"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["SCREEN_TEXTURE"] = ShaderLanguage::TYPE_SAMPLER2D;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["DEPTH_TEXTURE"] = ShaderLanguage::TYPE_SAMPLER2D;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["SCREEN_UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["POINT_COORD"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["SIDE"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["ALPHA_SCISSOR"] = ShaderLanguage::TYPE_FLOAT;

	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["WORLD_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["INV_CAMERA_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["PROJECTION_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["INV_PROJECTION_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["TIME"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["VIEWPORT_SIZE"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].can_discard = true;

	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["WORLD_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["INV_CAMERA_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["PROJECTION_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["INV_PROJECTION_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["TIME"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["VIEWPORT_SIZE"] = ShaderLanguage::TYPE_VEC2;

	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["VIEW"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["LIGHT"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["LIGHT_COLOR"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["ATTENUATION"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["ALBEDO"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["TRANSMISSION"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["ROUGHNESS"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["DIFFUSE_LIGHT"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["SPECULAR_LIGHT"] = ShaderLanguage::TYPE_VEC3;

	shader_modes[VS::SHADER_SPATIAL].functions["light"].can_discard = true;

	shader_modes[VS::SHADER_SPATIAL].modes.insert("blend_mix");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("blend_add");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("blend_sub");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("blend_mul");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_draw_opaque");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_draw_always");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_draw_never");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_draw_alpha_prepass");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_test_disable");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("cull_front");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("cull_back");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("cull_disabled");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("unshaded");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("diffuse_lambert");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("diffuse_lambert_wrap");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("diffuse_oren_nayar");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("diffuse_burley");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("diffuse_toon");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("specular_schlick_ggx");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("specular_blinn");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("specular_phong");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("specular_toon");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("specular_disabled");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("skip_vertex_transform");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("world_vertex_coords");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("vertex_lighting");

	/************ CANVAS ITEM **************************/

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["VERTEX"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["POINT_SIZE"] = ShaderLanguage::TYPE_FLOAT;

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["WORLD_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["PROJECTION_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["EXTRA_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["TIME"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["INSTANCE_CUSTOM"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["AT_LIGHT_PASS"] = ShaderLanguage::TYPE_BOOL;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].can_discard = false;

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["FRAGCOORD"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["NORMALMAP"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["NORMALMAP_DEPTH"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["TEXTURE"] = ShaderLanguage::TYPE_SAMPLER2D;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["TEXTURE_PIXEL_SIZE"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["NORMAL_TEXTURE"] = ShaderLanguage::TYPE_SAMPLER2D;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["SCREEN_UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["SCREEN_PIXEL_SIZE"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["POINT_COORD"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["TIME"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["AT_LIGHT_PASS"] = ShaderLanguage::TYPE_BOOL;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["SCREEN_TEXTURE"] = ShaderLanguage::TYPE_SAMPLER2D;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].can_discard = true;

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["POSITION"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["TEXTURE"] = ShaderLanguage::TYPE_SAMPLER2D;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["TEXTURE_PIXEL_SIZE"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["SCREEN_UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["LIGHT_VEC"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["LIGHT_HEIGHT"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["LIGHT_COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["LIGHT_UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["LIGHT_SHADOW"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["LIGHT"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["SHADOW"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["POINT_COORD"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["TIME"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].can_discard = true;

	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("skip_vertex_transform");

	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("blend_mix");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("blend_add");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("blend_sub");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("blend_mul");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("blend_premul_alpha");

	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("unshaded");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("light_only");

	/************ PARTICLES **************************/

	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["VELOCITY"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["MASS"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["ACTIVE"] = ShaderLanguage::TYPE_BOOL;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["RESTART"] = ShaderLanguage::TYPE_BOOL;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["CUSTOM"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["TRANSFORM"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["TIME"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["LIFETIME"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["DELTA"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["NUMBER"] = ShaderLanguage::TYPE_UINT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["INDEX"] = ShaderLanguage::TYPE_INT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["EMISSION_TRANSFORM"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["RANDOM_SEED"] = ShaderLanguage::TYPE_UINT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].can_discard = false;

	shader_modes[VS::SHADER_PARTICLES].modes.insert("billboard");
	shader_modes[VS::SHADER_PARTICLES].modes.insert("disable_force");
	shader_modes[VS::SHADER_PARTICLES].modes.insert("disable_velocity");
	shader_modes[VS::SHADER_PARTICLES].modes.insert("keep_data");

	shader_types.insert("spatial");
	shader_types.insert("canvas_item");
	shader_types.insert("particles");
}
