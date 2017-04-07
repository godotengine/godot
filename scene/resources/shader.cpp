/*************************************************************************/
/*  shader.cpp                                                           */
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
#include "shader.h"
#include "os/file_access.h"
#include "scene/scene_string_names.h"
#include "servers/visual/shader_language.h"
#include "servers/visual_server.h"
#include "texture.h"

Shader::Mode Shader::get_mode() const {

	return mode;
}

void Shader::set_code(const String &p_code) {

	String type = ShaderLanguage::get_shader_type(p_code);

	print_line("mode: " + type);

	if (type == "canvas_item") {
		mode = MODE_CANVAS_ITEM;
	} else if (type == "particles") {
		mode = MODE_PARTICLES;
	} else {
		mode = MODE_SPATIAL;
	}

	VisualServer::get_singleton()->shader_set_code(shader, p_code);
	params_cache_dirty = true;
	emit_signal(SceneStringNames::get_singleton()->changed);
}

String Shader::get_code() const {

	return VisualServer::get_singleton()->shader_get_code(shader);
}

void Shader::get_param_list(List<PropertyInfo> *p_params) const {

	List<PropertyInfo> local;
	VisualServer::get_singleton()->shader_get_param_list(shader, &local);
	params_cache.clear();
	params_cache_dirty = false;

	for (List<PropertyInfo>::Element *E = local.front(); E; E = E->next()) {

		PropertyInfo pi = E->get();
		pi.name = "shader_param/" + pi.name;
		params_cache[pi.name] = E->get().name;
		if (p_params) {

			//small little hack
			if (pi.type == Variant::_RID)
				pi.type = Variant::OBJECT;
			p_params->push_back(pi);
		}
	}
}

RID Shader::get_rid() const {

	return shader;
}

void Shader::set_default_texture_param(const StringName &p_param, const Ref<Texture> &p_texture) {

	if (p_texture.is_valid()) {
		default_textures[p_param] = p_texture;
		VS::get_singleton()->shader_set_default_texture_param(shader, p_param, p_texture->get_rid());
	} else {
		default_textures.erase(p_param);
		VS::get_singleton()->shader_set_default_texture_param(shader, p_param, RID());
	}
}

Ref<Texture> Shader::get_default_texture_param(const StringName &p_param) const {

	if (default_textures.has(p_param))
		return default_textures[p_param];
	else
		return Ref<Texture>();
}

void Shader::get_default_texture_param_list(List<StringName> *r_textures) const {

	for (const Map<StringName, Ref<Texture> >::Element *E = default_textures.front(); E; E = E->next()) {

		r_textures->push_back(E->key());
	}
}
bool Shader::has_param(const StringName &p_param) const {

	return params_cache.has(p_param);
}

void Shader::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_mode"), &Shader::get_mode);

	ClassDB::bind_method(D_METHOD("set_code", "code"), &Shader::set_code);
	ClassDB::bind_method(D_METHOD("get_code"), &Shader::get_code);

	ClassDB::bind_method(D_METHOD("set_default_texture_param", "param", "texture:Texture"), &Shader::set_default_texture_param);
	ClassDB::bind_method(D_METHOD("get_default_texture_param:Texture", "param"), &Shader::get_default_texture_param);

	ClassDB::bind_method(D_METHOD("has_param", "name"), &Shader::has_param);

	//ClassDB::bind_method(D_METHOD("get_param_list"),&Shader::get_fragment_code);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "code", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_code", "get_code");

	BIND_CONSTANT(MODE_SPATIAL);
	BIND_CONSTANT(MODE_CANVAS_ITEM);
	BIND_CONSTANT(MODE_PARTICLES);
}

Shader::Shader() {

	mode = MODE_SPATIAL;
	shader = VisualServer::get_singleton()->shader_create();
	params_cache_dirty = true;
}

Shader::~Shader() {

	VisualServer::get_singleton()->free(shader);
}
