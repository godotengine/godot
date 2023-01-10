/**************************************************************************/
/*  shader_types.cpp                                                      */
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

#include "shader_types.h"

const Map<StringName, ShaderLanguage::FunctionInfo> &ShaderTypes::get_functions(VS::ShaderMode p_mode) {
	return shader_modes[p_mode].functions;
}

const Vector<StringName> &ShaderTypes::get_modes(VS::ShaderMode p_mode) {
	return shader_modes[p_mode].modes;
}

const Set<String> &ShaderTypes::get_types() {
	return shader_types;
}

ShaderTypes *ShaderTypes::singleton = nullptr;

static ShaderLanguage::BuiltInInfo constt(ShaderLanguage::DataType p_type) {
	return ShaderLanguage::BuiltInInfo(p_type, true);
}

ShaderTypes::ShaderTypes() {
	singleton = this;

	/*************** SPATIAL ***********************/

	shader_modes[VS::SHADER_SPATIAL].functions["global"].built_ins["TIME"] = constt(ShaderLanguage::TYPE_FLOAT);

	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["VERTEX"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["TANGENT"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["BINORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["POSITION"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["UV2"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["POINT_SIZE"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["INSTANCE_ID"] = constt(ShaderLanguage::TYPE_INT);
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["INSTANCE_CUSTOM"] = constt(ShaderLanguage::TYPE_VEC4);
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["VERTEX_ID"] = constt(ShaderLanguage::TYPE_INT);
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["ROUGHNESS"] = ShaderLanguage::TYPE_FLOAT;

	//builtins
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["WORLD_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["INV_CAMERA_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["CAMERA_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["PROJECTION_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["MODELVIEW_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["INV_PROJECTION_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["VIEW_INDEX"] = constt(ShaderLanguage::TYPE_INT);
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["VIEW_MONO_LEFT"] = constt(ShaderLanguage::TYPE_INT);
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["VIEW_RIGHT"] = constt(ShaderLanguage::TYPE_INT);
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["VIEWPORT_SIZE"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["OUTPUT_IS_SRGB"] = constt(ShaderLanguage::TYPE_BOOL);
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["NODE_POSITION_WORLD"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["CAMERA_POSITION_WORLD"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["CAMERA_DIRECTION_WORLD"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].built_ins["NODE_POSITION_VIEW"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"].main_function = true;

	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["VERTEX"] = constt(ShaderLanguage::TYPE_VEC3);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["FRAGCOORD"] = constt(ShaderLanguage::TYPE_VEC4);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["FRONT_FACING"] = constt(ShaderLanguage::TYPE_BOOL);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["TANGENT"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["BINORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["VIEW"] = constt(ShaderLanguage::TYPE_VEC3);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["NORMALMAP"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["NORMALMAP_DEPTH"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["UV"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["UV2"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["COLOR"] = constt(ShaderLanguage::TYPE_VEC4);
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
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["DEPTH"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["SCREEN_UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["POINT_COORD"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["ALPHA_SCISSOR"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["VIEW_INDEX"] = constt(ShaderLanguage::TYPE_INT);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["VIEW_MONO_LEFT"] = constt(ShaderLanguage::TYPE_INT);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["VIEW_RIGHT"] = constt(ShaderLanguage::TYPE_INT);

	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["OUTPUT_IS_SRGB"] = constt(ShaderLanguage::TYPE_BOOL);

	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["WORLD_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["INV_CAMERA_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["CAMERA_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["PROJECTION_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["INV_PROJECTION_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["VIEWPORT_SIZE"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["NODE_POSITION_WORLD"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["CAMERA_POSITION_WORLD"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["CAMERA_DIRECTION_WORLD"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].built_ins["NODE_POSITION_VIEW"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].can_discard = true;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"].main_function = true;

	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["WORLD_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["INV_CAMERA_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["CAMERA_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["PROJECTION_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["INV_PROJECTION_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["VIEWPORT_SIZE"] = constt(ShaderLanguage::TYPE_VEC2);

	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["FRAGCOORD"] = constt(ShaderLanguage::TYPE_VEC4);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["NORMAL"] = constt(ShaderLanguage::TYPE_VEC3);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["UV"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["UV2"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["VIEW"] = constt(ShaderLanguage::TYPE_VEC3);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["LIGHT"] = constt(ShaderLanguage::TYPE_VEC3);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["LIGHT_COLOR"] = constt(ShaderLanguage::TYPE_VEC3);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["ATTENUATION"] = constt(ShaderLanguage::TYPE_VEC3);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["ALBEDO"] = constt(ShaderLanguage::TYPE_VEC3);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["TRANSMISSION"] = constt(ShaderLanguage::TYPE_VEC3);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["METALLIC"] = constt(ShaderLanguage::TYPE_FLOAT);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["ROUGHNESS"] = constt(ShaderLanguage::TYPE_FLOAT);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["DIFFUSE_LIGHT"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["SPECULAR_LIGHT"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["OUTPUT_IS_SRGB"] = constt(ShaderLanguage::TYPE_BOOL);
	shader_modes[VS::SHADER_SPATIAL].functions["light"].built_ins["ALPHA"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].can_discard = true;
	shader_modes[VS::SHADER_SPATIAL].functions["light"].main_function = true;

	//order used puts first enum mode (default) first
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("blend_mix");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("blend_add");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("blend_sub");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("blend_mul");

	shader_modes[VS::SHADER_SPATIAL].modes.push_back("depth_draw_opaque");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("depth_draw_always");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("depth_draw_never");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("depth_draw_alpha_prepass");

	shader_modes[VS::SHADER_SPATIAL].modes.push_back("depth_test_disable");

	shader_modes[VS::SHADER_SPATIAL].modes.push_back("cull_back");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("cull_front");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("cull_disabled");

	shader_modes[VS::SHADER_SPATIAL].modes.push_back("unshaded");

	shader_modes[VS::SHADER_SPATIAL].modes.push_back("diffuse_lambert");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("diffuse_lambert_wrap");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("diffuse_oren_nayar");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("diffuse_burley");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("diffuse_toon");

	shader_modes[VS::SHADER_SPATIAL].modes.push_back("specular_schlick_ggx");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("specular_blinn");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("specular_phong");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("specular_toon");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("specular_disabled");

	shader_modes[VS::SHADER_SPATIAL].modes.push_back("skip_vertex_transform");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("world_vertex_coords");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("ensure_correct_normals");

	shader_modes[VS::SHADER_SPATIAL].modes.push_back("shadows_disabled");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("ambient_light_disabled");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("shadow_to_opacity");

	shader_modes[VS::SHADER_SPATIAL].modes.push_back("vertex_lighting");

	shader_modes[VS::SHADER_SPATIAL].modes.push_back("async_visible");
	shader_modes[VS::SHADER_SPATIAL].modes.push_back("async_hidden");

	/************ CANVAS ITEM **************************/

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["global"].built_ins["TIME"] = constt(ShaderLanguage::TYPE_FLOAT);

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["VERTEX"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["MODULATE"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["POINT_SIZE"] = ShaderLanguage::TYPE_FLOAT;

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["WORLD_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["PROJECTION_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["EXTRA_MATRIX"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["INSTANCE_CUSTOM"] = constt(ShaderLanguage::TYPE_VEC4);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["AT_LIGHT_PASS"] = constt(ShaderLanguage::TYPE_BOOL);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["TEXTURE_PIXEL_SIZE"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["INSTANCE_ID"] = constt(ShaderLanguage::TYPE_INT);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].built_ins["VERTEX_ID"] = constt(ShaderLanguage::TYPE_INT);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"].main_function = true;

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["FRAGCOORD"] = constt(ShaderLanguage::TYPE_VEC4);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["NORMALMAP"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["NORMALMAP_DEPTH"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["UV"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["MODULATE"] = constt(ShaderLanguage::TYPE_VEC4);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["TEXTURE"] = constt(ShaderLanguage::TYPE_SAMPLER2D);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["TEXTURE_PIXEL_SIZE"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["NORMAL_TEXTURE"] = constt(ShaderLanguage::TYPE_SAMPLER2D);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["SCREEN_UV"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["SCREEN_PIXEL_SIZE"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["POINT_COORD"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["AT_LIGHT_PASS"] = constt(ShaderLanguage::TYPE_BOOL);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].built_ins["SCREEN_TEXTURE"] = constt(ShaderLanguage::TYPE_SAMPLER2D);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].can_discard = true;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"].main_function = true;

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["FRAGCOORD"] = constt(ShaderLanguage::TYPE_VEC4);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["NORMAL"] = constt(ShaderLanguage::TYPE_VEC3);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["UV"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["COLOR"] = constt(ShaderLanguage::TYPE_VEC4);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["MODULATE"] = constt(ShaderLanguage::TYPE_VEC4);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["TEXTURE"] = constt(ShaderLanguage::TYPE_SAMPLER2D);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["TEXTURE_PIXEL_SIZE"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["SCREEN_UV"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["LIGHT_VEC"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["SHADOW_VEC"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["LIGHT_HEIGHT"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["LIGHT_COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["LIGHT_UV"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["LIGHT"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["SHADOW_COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].built_ins["POINT_COORD"] = constt(ShaderLanguage::TYPE_VEC2);
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].can_discard = true;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"].main_function = true;

	shader_modes[VS::SHADER_CANVAS_ITEM].modes.push_back("skip_vertex_transform");

	shader_modes[VS::SHADER_CANVAS_ITEM].modes.push_back("blend_mix");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.push_back("blend_add");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.push_back("blend_sub");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.push_back("blend_mul");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.push_back("blend_premul_alpha");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.push_back("blend_disabled");

	shader_modes[VS::SHADER_CANVAS_ITEM].modes.push_back("unshaded");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.push_back("light_only");

	/************ PARTICLES **************************/

	shader_modes[VS::SHADER_PARTICLES].functions["global"].built_ins["TIME"] = constt(ShaderLanguage::TYPE_FLOAT);
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["VELOCITY"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["MASS"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["ACTIVE"] = ShaderLanguage::TYPE_BOOL;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["RESTART"] = constt(ShaderLanguage::TYPE_BOOL);
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["CUSTOM"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["TRANSFORM"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["LIFETIME"] = constt(ShaderLanguage::TYPE_FLOAT);
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["DELTA"] = constt(ShaderLanguage::TYPE_FLOAT);
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["NUMBER"] = constt(ShaderLanguage::TYPE_UINT);
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["INDEX"] = constt(ShaderLanguage::TYPE_INT);
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["EMISSION_TRANSFORM"] = constt(ShaderLanguage::TYPE_MAT4);
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].built_ins["RANDOM_SEED"] = constt(ShaderLanguage::TYPE_UINT);
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"].main_function = true;

	shader_modes[VS::SHADER_PARTICLES].modes.push_back("disable_force");
	shader_modes[VS::SHADER_PARTICLES].modes.push_back("disable_velocity");
	shader_modes[VS::SHADER_PARTICLES].modes.push_back("keep_data");

	shader_types.insert("spatial");
	shader_types.insert("canvas_item");
	shader_types.insert("particles");
}
