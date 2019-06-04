/*************************************************************************/
/*  shader.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "servers/visual_server.h"
#include "texture.h"

Shader::Mode Shader::get_mode() const {

	return mode;
}

void Shader::set_code(const String &p_vertex, const String &p_fragment, const String &p_light, int p_fragment_ofs, int p_light_ofs) {

	VisualServer::get_singleton()->shader_set_code(shader, p_vertex, p_fragment, p_light, 0, p_fragment_ofs, p_light_ofs);
	params_cache_dirty = true;
	emit_signal(SceneStringNames::get_singleton()->changed);
}

String Shader::get_vertex_code() const {

	return VisualServer::get_singleton()->shader_get_vertex_code(shader);
}

String Shader::get_fragment_code() const {

	return VisualServer::get_singleton()->shader_get_fragment_code(shader);
}

String Shader::get_light_code() const {

	return VisualServer::get_singleton()->shader_get_light_code(shader);
}

bool Shader::has_param(const StringName &p_param) const {

	if (params_cache_dirty)
		get_param_list(NULL);

	return (params_cache.has(p_param));
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

Dictionary Shader::_get_code() {

	String fs = VisualServer::get_singleton()->shader_get_fragment_code(shader);
	String vs = VisualServer::get_singleton()->shader_get_vertex_code(shader);
	String ls = VisualServer::get_singleton()->shader_get_light_code(shader);

	Dictionary c;
	c["fragment"] = fs;
	c["fragment_ofs"] = 0;
	c["vertex"] = vs;
	c["vertex_ofs"] = 0;
	c["light"] = ls;
	c["light_ofs"] = 0;
	Array arr;
	for (const Map<StringName, Ref<Texture> >::Element *E = default_textures.front(); E; E = E->next()) {
		arr.push_back(E->key());
		arr.push_back(E->get());
	}
	if (arr.size())
		c["default_tex"] = arr;
	return c;
}

void Shader::_set_code(const Dictionary &p_string) {

	ERR_FAIL_COND(!p_string.has("fragment"));
	ERR_FAIL_COND(!p_string.has("vertex"));
	String light;
	if (p_string.has("light"))
		light = p_string["light"];

	set_code(p_string["vertex"], p_string["fragment"], light);
	if (p_string.has("default_tex")) {
		Array arr = p_string["default_tex"];
		if ((arr.size() & 1) == 0) {
			for (int i = 0; i < arr.size(); i += 2) {

				set_default_texture_param(arr[i], arr[i + 1]);
			}
		}
	}
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

void Shader::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_mode"), &Shader::get_mode);

	ObjectTypeDB::bind_method(_MD("set_code", "vcode", "fcode", "lcode", "fofs", "lofs"), &Shader::set_code, DEFVAL(0), DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("get_vertex_code"), &Shader::get_vertex_code);
	ObjectTypeDB::bind_method(_MD("get_fragment_code"), &Shader::get_fragment_code);
	ObjectTypeDB::bind_method(_MD("get_light_code"), &Shader::get_light_code);

	ObjectTypeDB::bind_method(_MD("set_default_texture_param", "param", "texture:Texture"), &Shader::set_default_texture_param);
	ObjectTypeDB::bind_method(_MD("get_default_texture_param:Texture", "param"), &Shader::get_default_texture_param);

	ObjectTypeDB::bind_method(_MD("has_param", "name"), &Shader::has_param);

	ObjectTypeDB::bind_method(_MD("_set_code", "code"), &Shader::_set_code);
	ObjectTypeDB::bind_method(_MD("_get_code"), &Shader::_get_code);

	//ObjectTypeDB::bind_method(_MD("get_param_list"),&Shader::get_fragment_code);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "_code", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), _SCS("_set_code"), _SCS("_get_code"));

	BIND_CONSTANT(MODE_MATERIAL);
	BIND_CONSTANT(MODE_CANVAS_ITEM);
	BIND_CONSTANT(MODE_POST_PROCESS);
}

Shader::Shader(Mode p_mode) {

	mode = p_mode;
	shader = VisualServer::get_singleton()->shader_create(VS::ShaderMode(p_mode));
	params_cache_dirty = true;
}

Shader::~Shader() {

	VisualServer::get_singleton()->free(shader);
}

/************ Loader from text ***************/

RES ResourceFormatLoaderShader::load(const String &p_path, const String &p_original_path, Error *r_error) {

	if (r_error)
		*r_error = ERR_FILE_CANT_OPEN;

	String fragment_code;
	String vertex_code;
	String light_code;

	int mode = -1;

	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);

	ERR_EXPLAIN("Unable to open shader file: " + p_path);
	ERR_FAIL_COND_V(err, RES());
	String base_path = p_path.get_base_dir();

	if (r_error)
		*r_error = ERR_FILE_CORRUPT;

	Ref<Shader> shader; //( memnew( Shader ) );

	int line = 0;

	while (!f->eof_reached()) {

		String l = f->get_line();
		line++;

		if (mode <= 0) {
			l = l.strip_edges();
			int comment = l.find(";");
			if (comment != -1)
				l = l.substr(0, comment);
		}

		if (mode < 1)
			vertex_code += "\n";
		if (mode < 2)
			fragment_code += "\n";

		if (mode < 1 && l == "")
			continue;

		if (l.begins_with("[")) {
			l = l.strip_edges();
			if (l == "[params]") {
				if (mode >= 0) {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Misplaced [params] section.");
					ERR_FAIL_V(RES());
				}
				mode = 0;
			} else if (l == "[vertex]") {
				if (mode >= 1) {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Misplaced [vertex] section.");
					ERR_FAIL_V(RES());
				}
				mode = 1;
			} else if (l == "[fragment]") {
				if (mode >= 2) {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Misplaced [fragment] section.");
					ERR_FAIL_V(RES());
				}
				mode = 1;
			} else {
				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": Unknown section type: '" + l + "'.");
				ERR_FAIL_V(RES());
			}
			continue;
		}

		if (mode == 0) {

			int eqpos = l.find("=");
			if (eqpos == -1) {
				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": Expected '='.");
				ERR_FAIL_V(RES());
			}

			String right = l.substr(eqpos + 1, l.length()).strip_edges();
			if (right == "") {
				memdelete(f);
				ERR_EXPLAIN(p_path + ":" + itos(line) + ": Expected value after '='.");
				ERR_FAIL_V(RES());
			}

			Variant value;

			if (right == "true") {
				value = true;
			} else if (right == "false") {
				value = false;
			} else if (right.is_valid_float()) {
				//is number
				value = right.to_double();
			} else if (right.is_valid_html_color()) {
				//is html color
				value = Color::html(right);
			} else {
				//attempt to parse a constructor
				int popenpos = right.find("(");

				if (popenpos == -1) {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid constructor syntax: " + right);
					ERR_FAIL_V(RES());
				}

				int pclosepos = right.find_last(")");

				if (pclosepos == -1) {
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid constructor parameter syntax: " + right);
					ERR_FAIL_V(RES());
				}

				String type = right.substr(0, popenpos);
				String param = right.substr(popenpos + 1, pclosepos - popenpos - 1).strip_edges();

				if (type == "tex") {

					if (param == "") {

						value = RID();
					} else {

						String path;

						if (param.is_abs_path())
							path = param;
						else
							path = base_path + "/" + param;

						Ref<Texture> texture = ResourceLoader::load(path);
						if (!texture.is_valid()) {
							memdelete(f);
							ERR_EXPLAIN(p_path + ":" + itos(line) + ": Couldn't find icon at path: " + path);
							ERR_FAIL_V(RES());
						}

						value = texture;
					}

				} else if (type == "vec3") {

					if (param == "") {
						value = Vector3();
					} else {
						Vector<String> params = param.split(",");
						if (params.size() != 3) {
							memdelete(f);
							ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid param count for vec3(): '" + right + "'.");
							ERR_FAIL_V(RES());
						}

						Vector3 v;
						for (int i = 0; i < 3; i++)
							v[i] = params[i].to_double();
						value = v;
					}

				} else if (type == "xform") {

					if (param == "") {
						value = Transform();
					} else {

						Vector<String> params = param.split(",");
						if (params.size() != 12) {
							memdelete(f);
							ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid param count for xform(): '" + right + "'.");
							ERR_FAIL_V(RES());
						}

						Transform t;
						for (int i = 0; i < 9; i++)
							t.basis[i % 3][i / 3] = params[i].to_double();
						for (int i = 0; i < 3; i++)
							t.origin[i] = params[i - 9].to_double();

						value = t;
					}

				} else {
					memdelete(f);
					ERR_EXPLAIN(p_path + ":" + itos(line) + ": Invalid constructor type: '" + type + "'.");
					ERR_FAIL_V(RES());
				}
			}

			String left = l.substr(0, eqpos);

			//			shader->set_param(left,value);
		} else if (mode == 1) {

			vertex_code += l;

		} else if (mode == 2) {

			fragment_code += l;
		}
	}

	shader->set_code(vertex_code, fragment_code, light_code);

	f->close();
	memdelete(f);
	if (r_error)
		*r_error = OK;

	return shader;
}

void ResourceFormatLoaderShader::get_recognized_extensions(List<String> *p_extensions) const {

	ObjectTypeDB::get_extensions_for_type("Shader", p_extensions);
}

bool ResourceFormatLoaderShader::handles_type(const String &p_type) const {

	return ObjectTypeDB::is_type(p_type, "Shader");
}

String ResourceFormatLoaderShader::get_resource_type(const String &p_path) const {

	if (p_path.extension().to_lower() == "shd")
		return "Shader";
	return "";
}
