/*************************************************************************/
/*  shader_types.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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

const Map<StringName, Map<StringName, ShaderLanguage::DataType> > &ShaderTypes::get_functions(VS::ShaderMode p_mode) {

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

	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["SRC_VERTEX"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["SRC_NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["SRC_TANGENT"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["SRC_BONES"] = ShaderLanguage::TYPE_IVEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["SRC_WEIGHTS"] = ShaderLanguage::TYPE_VEC4;

	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["POSITION"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["VERTEX"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["TANGENT"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["BINORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["UV2"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["POINT_SIZE"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["INSTANCE_ID"] = ShaderLanguage::TYPE_INT;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["INSTANCE_CUSTOM"] = ShaderLanguage::TYPE_VEC4;

	//builtins
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["WORLD_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["INV_CAMERA_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["CAMERA_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["PROJECTION_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["MODELVIEW_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["TIME"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["VIEWPORT_SIZE"] = ShaderLanguage::TYPE_VEC2;

	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["VERTEX"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["FRAGCOORD"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["FRONT_FACING"] = ShaderLanguage::TYPE_BOOL;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["TANGENT"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["BINORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["NORMALMAP"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["NORMALMAP_DEPTH"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["UV2"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["ALBEDO"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["ALPHA"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["SPECULAR"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["ROUGHNESS"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["RIM"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["RIM_TINT"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["CLEARCOAT"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["CLEARCOAT_GLOSS"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["ANISOTROPY"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["ANISOTROPY_FLOW"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["SSS_STRENGTH"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["AO"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["EMISSION"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["SPECIAL"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["DISCARD"] = ShaderLanguage::TYPE_BOOL;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["SCREEN_UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["POINT_COORD"] = ShaderLanguage::TYPE_VEC2;

	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["WORLD_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["INV_CAMERA_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["PROJECTION_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["TIME"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["VIEWPORT_SIZE"] = ShaderLanguage::TYPE_VEC2;

	shader_modes[VS::SHADER_SPATIAL].modes.insert("blend_mix");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("blend_add");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("blend_sub");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("blend_mul");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_draw_opaque");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_draw_always");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_draw_never");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_draw_alpha_prepass");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("cull_front");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("cull_back");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("cull_disabled");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("unshaded");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("ontop");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("skip_default_transform");

	/************ CANVAS ITEM **************************/

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"]["SRC_VERTEX"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"]["VERTEX"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"]["UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"]["VERTEX_COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"]["POINT_SIZE"] = ShaderLanguage::TYPE_FLOAT;

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"]["WORLD_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"]["PROJECTION_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"]["EXTRA_MATRIX"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"]["TIME"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["vertex"]["PARTICLE_CUSTOM"] = ShaderLanguage::TYPE_VEC4;

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"]["SRC_COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"]["POSITION"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"]["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"]["NORMALMAP"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"]["NORMALMAP_DEPTH"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"]["UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"]["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"]["TEXTURE"] = ShaderLanguage::TYPE_SAMPLER2D;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"]["TEXTURE_PIXEL_SIZE"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"]["SCREEN_UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"]["POINT_COORD"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["fragment"]["TIME"] = ShaderLanguage::TYPE_FLOAT;

	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["POSITION"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["NORMAL"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["TEXTURE"] = ShaderLanguage::TYPE_SAMPLER2D;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["TEXTURE_PIXEL_SIZE"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["VAR1"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["VAR2"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["SCREEN_UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["LIGHT_VEC"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["LIGHT_HEIGHT"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["LIGHT_COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["LIGHT_UV"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["LIGHT_SHADOW"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["LIGHT"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["SHADOW"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["POINT_COORD"] = ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_CANVAS_ITEM].functions["light"]["TIME"] = ShaderLanguage::TYPE_FLOAT;

	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("skip_transform");

	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("blend_mix");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("blend_add");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("blend_sub");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("blend_mul");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("blend_premul_alpha");

	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("unshaded");
	shader_modes[VS::SHADER_CANVAS_ITEM].modes.insert("light_only");

	/************ PARTICLES **************************/

	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["COLOR"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["VELOCITY"] = ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["MASS"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["ACTIVE"] = ShaderLanguage::TYPE_BOOL;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["RESTART"] = ShaderLanguage::TYPE_BOOL;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["CUSTOM"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["TRANSFORM"] = ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["TIME"] = ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["LIFETIME"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["DELTA"] = ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["NUMBER"] = ShaderLanguage::TYPE_UINT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["INDEX"] = ShaderLanguage::TYPE_INT;
	shader_modes[VS::SHADER_PARTICLES].functions["vertex"]["EMISSION_TRANSFORM"] = ShaderLanguage::TYPE_MAT4;

	shader_modes[VS::SHADER_PARTICLES].modes.insert("billboard");
	shader_modes[VS::SHADER_PARTICLES].modes.insert("disable_force");
	shader_modes[VS::SHADER_PARTICLES].modes.insert("disable_velocity");
	shader_modes[VS::SHADER_PARTICLES].modes.insert("keep_data");

	shader_types.insert("spatial");
	shader_types.insert("canvas_item");
	shader_types.insert("particles");
}
