/*************************************************************************/
/*  sky_material.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "sky_material.h"

Mutex *SkyMaterial::material_mutex = NULL;
SelfList<SkyMaterial>::List *SkyMaterial::dirty_materials = NULL;
Map<SkyMaterial::MaterialKey, SkyMaterial::ShaderData> SkyMaterial::shader_map;
SkyMaterial::ShaderNames *SkyMaterial::shader_names = NULL;

void SkyMaterial::init_shaders() {

#ifndef NO_THREADS
	material_mutex = Mutex::create();
#endif

	dirty_materials = memnew(SelfList<SkyMaterial>::List);

	shader_names = memnew(ShaderNames);

	shader_names->placeholder = "placeholder";
}

void SkyMaterial::finish_shaders() {

#ifndef NO_THREADS
	memdelete(material_mutex);
#endif

	memdelete(dirty_materials);
	dirty_materials = NULL;

	memdelete(shader_names);
}

void SkyMaterial::_update_shader() {

	dirty_materials->remove(&element);

	MaterialKey mk = _compute_key();
	if (mk.key == current_key.key)
		return; //no update required in the end

	if (shader_map.has(current_key)) {
		shader_map[current_key].users--;
		if (shader_map[current_key].users == 0) {
			//deallocate shader, as it's no longer in use
			VS::get_singleton()->free(shader_map[current_key].shader);
			shader_map.erase(current_key);
		}
	}

	current_key = mk;

	if (shader_map.has(mk)) {

		VS::get_singleton()->material_set_shader(_get_material(), shader_map[mk].shader);
		shader_map[mk].users++;
		return;
	}

	//must create a shader!

	String code = "shader_type sky;\n";

	ShaderData shader_data;
	shader_data.shader = VS::get_singleton()->shader_create();
	shader_data.users = 1;

	VS::get_singleton()->shader_set_code(shader_data.shader, code);

	shader_map[mk] = shader_data;

	VS::get_singleton()->material_set_shader(_get_material(), shader_data.shader);
}

void SkyMaterial::flush_changes() {

	if (material_mutex)
		material_mutex->lock();

	while (dirty_materials->first()) {

		dirty_materials->first()->self()->_update_shader();
	}

	if (material_mutex)
		material_mutex->unlock();
}

void SkyMaterial::_queue_shader_change() {

	if (material_mutex)
		material_mutex->lock();

	if (!element.in_list()) {
		dirty_materials->add(&element);
	}

	if (material_mutex)
		material_mutex->unlock();
}

bool SkyMaterial::_is_shader_dirty() const {

	bool dirty = false;

	if (material_mutex)
		material_mutex->lock();

	dirty = element.in_list();

	if (material_mutex)
		material_mutex->unlock();

	return dirty;
}


RID SkyMaterial::get_shader_rid() const {

	ERR_FAIL_COND_V(!shader_map.has(current_key), RID());
	return shader_map[current_key].shader;
}

void SkyMaterial::_validate_property(PropertyInfo &property) const {

}

Shader::Mode SkyMaterial::get_shader_mode() const {

	return Shader::MODE_SKY;
}

void SkyMaterial::_bind_methods() {

}

SkyMaterial::SkyMaterial() :
		element(this) {


	_queue_shader_change();
}

SkyMaterial::~SkyMaterial() {

	if (material_mutex)
		material_mutex->lock();

	if (shader_map.has(current_key)) {
		shader_map[current_key].users--;
		if (shader_map[current_key].users == 0) {
			//deallocate shader, as it's no longer in use
			VS::get_singleton()->free(shader_map[current_key].shader);
			shader_map.erase(current_key);
		}

		VS::get_singleton()->material_set_shader(_get_material(), RID());
	}

	if (material_mutex)
		material_mutex->unlock();
}
