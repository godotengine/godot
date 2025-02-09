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

void BlitMaterial::init_shaders() {
	return;
}

void BlitMaterial::finish_shaders() {
	dirty_materials.clear();
}

void BlitMaterial::_update_shader() {
	MaterialKey mk = _compute_key();
	if (mk.key == current_key.key) {
		return; //no update required in the end
	}

	if (shader_map.has(current_key)) {
		shader_map[current_key].users--;
		if (shader_map[current_key].users == 0) {
			//deallocate shader, as it's no longer in use
			RS::get_singleton()->free(shader_map[current_key].shader);
			shader_map.erase(current_key);
		}
	}

	current_key = mk;

	if (shader_map.has(mk)) {
		RS::get_singleton()->material_set_shader(_get_material(), shader_map[mk].shader);
		shader_map[mk].users++;
		return;
	}

	//must create a shader!

	// Add a comment to describe the shader origin (useful when converting to ShaderMaterial).
	String code = "// NOTE: Shader automatically converted from " GODOT_VERSION_NAME " " GODOT_VERSION_FULL_CONFIG "'s BlitMaterial.\n\n";

	code += "shader_type texture_blit;\nrender_mode ";
	switch (blend_mode) {
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
	}
	code += ";\n\n";

	code += "uniform sampler2D source_texture : hint_blit_source;\n";
	code += "uniform sampler2D source_texture2 : hint_blit_source2;\n";
	code += "uniform sampler2D source_texture3 : hint_blit_source3;\n";
	code += "uniform sampler2D source_texture4 : hint_blit_source4;\n\n";

	code += "void blit() {\n";
	code += "	// Copies from each whole source texture to a rect on each output texture.\n";
	code += "	COLOR = texture(source_texture, UV) * MODULATE;\n";
	code += "	COLOR2 = texture(source_texture2, UV) * MODULATE;\n";
	code += "	COLOR3 = texture(source_texture3, UV) * MODULATE;\n";
	code += "	COLOR4 = texture(source_texture4, UV) * MODULATE;\n}";

	ShaderData shader_data;
	shader_data.shader = RS::get_singleton()->shader_create();
	shader_data.users = 1;

	RS::get_singleton()->shader_set_code(shader_data.shader, code);

	shader_map[mk] = shader_data;

	RS::get_singleton()->material_set_shader(_get_material(), shader_data.shader);
}

void BlitMaterial::flush_changes() {
	MutexLock lock(material_mutex);

	while (dirty_materials.first()) {
		dirty_materials.first()->self()->_update_shader();
		dirty_materials.first()->remove_from_list();
	}
}

void BlitMaterial::_queue_shader_change() {
	if (!_is_initialized()) {
		return;
	}

	MutexLock lock(material_mutex);

	if (!element.in_list()) {
		dirty_materials.add(&element);
	}
}

void BlitMaterial::set_blend_mode(BlendMode p_blend_mode) {
	blend_mode = p_blend_mode;
	_queue_shader_change();
}

BlitMaterial::BlendMode BlitMaterial::get_blend_mode() const {
	return blend_mode;
}

RID BlitMaterial::get_shader_rid() const {
	ERR_FAIL_COND_V(!shader_map.has(current_key), RID());
	return shader_map[current_key].shader;
}

Shader::Mode BlitMaterial::get_shader_mode() const {
	return Shader::MODE_TEXTURE_BLIT;
}

void BlitMaterial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_blend_mode", "blend_mode"), &BlitMaterial::set_blend_mode);
	ClassDB::bind_method(D_METHOD("get_blend_mode"), &BlitMaterial::get_blend_mode);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "blend_mode", PROPERTY_HINT_ENUM, "Mix,Add,Subtract,Multiply"), "set_blend_mode", "get_blend_mode");

	BIND_ENUM_CONSTANT(BLEND_MODE_MIX);
	BIND_ENUM_CONSTANT(BLEND_MODE_ADD);
	BIND_ENUM_CONSTANT(BLEND_MODE_SUB);
	BIND_ENUM_CONSTANT(BLEND_MODE_MUL);
}

BlitMaterial::BlitMaterial() :
		element(this) {
	_set_material(RS::get_singleton()->material_create());

	current_key.invalid_key = 1;

	_mark_initialized(callable_mp(this, &BlitMaterial::_queue_shader_change), callable_mp(this, &BlitMaterial::_update_shader));
}

BlitMaterial::~BlitMaterial() {
	MutexLock lock(material_mutex);

	ERR_FAIL_NULL(RenderingServer::get_singleton());

	if (shader_map.has(current_key)) {
		shader_map[current_key].users--;
		if (shader_map[current_key].users == 0) {
			//deallocate shader, as it's no longer in use
			RS::get_singleton()->free(shader_map[current_key].shader);
			shader_map.erase(current_key);
		}

		RS::get_singleton()->material_set_shader(_get_material(), RID());
	}
}
