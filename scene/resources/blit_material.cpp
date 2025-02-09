/**************************************************************************/
/*  blit_material.cpp                                                     */
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

#include "blit_material.h"

#include "core/version.h"

void BlitMaterial::_update_shader(BlendMode p_blend) {
	MutexLock shader_lock(shader_mutex);
	int index = int(p_blend);
	if (shader_cache[p_blend].is_null()) {
		shader_cache[p_blend] = RS::get_singleton()->shader_create();
		String code = "// NOTE: Shader automatically converted from " GODOT_VERSION_NAME " " GODOT_VERSION_FULL_CONFIG "'s BlitMaterial.\n\n";

		code += "shader_type texture_blit;\nrender_mode ";
		switch (p_blend) {
			case BLEND_MODE_MIX:
				code += "blend_mix";
				break;
			case BLEND_MODE_ADD:
				code += "blend_add";
				break;
			case BLEND_MODE_SUB:
				code += "blend_sub";
				break;
			case BLEND_MODE_MUL:
				code += "blend_mul";
				break;
			case BLEND_MODE_DISABLED:
				code += "blend_disabled";
				break;
			default:
				code += "blend_mix";
				break;
		}
		code += ";\n\n";

		code += "uniform sampler2D source_texture0 : hint_blit_source0;\n";
		code += "uniform sampler2D source_texture1 : hint_blit_source1;\n";
		code += "uniform sampler2D source_texture2 : hint_blit_source2;\n";
		code += "uniform sampler2D source_texture3 : hint_blit_source3;\n\n";

		code += "void blit() {\n";
		code += "	// Copies from each whole source texture to a rect on each output texture.\n";
		code += "	COLOR0 = texture(source_texture0, UV) * MODULATE;\n";
		code += "	COLOR1 = texture(source_texture1, UV) * MODULATE;\n";
		code += "	COLOR2 = texture(source_texture2, UV) * MODULATE;\n";
		code += "	COLOR3 = texture(source_texture3, UV) * MODULATE;\n}";
		RS::get_singleton()->shader_set_code(shader_cache[index], code);
	}
}

void BlitMaterial::set_blend_mode(BlendMode p_blend_mode) {
	blend_mode = p_blend_mode;
	_update_shader(blend_mode);
	if (shader_set) {
		RS::get_singleton()->material_set_shader(_get_material(), shader_cache[int(blend_mode)]);
	}
}

BlitMaterial::BlendMode BlitMaterial::get_blend_mode() const {
	return blend_mode;
}

RID BlitMaterial::get_shader_rid() const {
	_update_shader(blend_mode);
	return shader_cache[int(blend_mode)];
}

Shader::Mode BlitMaterial::get_shader_mode() const {
	return Shader::MODE_TEXTURE_BLIT;
}

RID BlitMaterial::get_rid() const {
	_update_shader(blend_mode);
	if (!shader_set) {
		RS::get_singleton()->material_set_shader(_get_material(), shader_cache[int(blend_mode)]);
		shader_set = true;
	}
	return _get_material();
}

Mutex BlitMaterial::shader_mutex;
RID BlitMaterial::shader_cache[5];

void BlitMaterial::cleanup_shader() {
	for (int i = 0; i < 5; i++) {
		if (shader_cache[i].is_valid()) {
			RS::get_singleton()->free_rid(shader_cache[i]);
		}
	}
}

void BlitMaterial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_blend_mode", "blend_mode"), &BlitMaterial::set_blend_mode);
	ClassDB::bind_method(D_METHOD("get_blend_mode"), &BlitMaterial::get_blend_mode);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "blend_mode", PROPERTY_HINT_ENUM, "Mix,Add,Subtract,Multiply,Disabled"), "set_blend_mode", "get_blend_mode");

	BIND_ENUM_CONSTANT(BLEND_MODE_MIX);
	BIND_ENUM_CONSTANT(BLEND_MODE_ADD);
	BIND_ENUM_CONSTANT(BLEND_MODE_SUB);
	BIND_ENUM_CONSTANT(BLEND_MODE_MUL);
	BIND_ENUM_CONSTANT(BLEND_MODE_DISABLED);
}

BlitMaterial::BlitMaterial() {
	_set_material(RS::get_singleton()->material_create());
	set_blend_mode(BLEND_MODE_MIX);
}

BlitMaterial::~BlitMaterial() {
}
