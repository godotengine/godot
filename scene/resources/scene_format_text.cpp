/*************************************************************************/
/*  scene_format_text.cpp                                                */
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
#include "scene_format_text.h"
#include "core/io/resource_format_binary.h"
#include "os/dir_access.h"
#include "project_settings.h"
#include "version.h"

//version 2: changed names for basis, aabb, poolvectors, etc.
#define FORMAT_VERSION 2

#include "os/dir_access.h"
#include "version.h"

#define _printerr() ERR_PRINT(String(res_path + ":" + itos(lines) + " - Parse Error: " + error_text).utf8().get_data());

///

void ResourceInteractiveLoaderText::set_local_path(const String &p_local_path) {

	res_path = p_local_path;
}

Ref<Resource> ResourceInteractiveLoaderText::get_resource() {

	return resource;
}

Error ResourceInteractiveLoaderText::_parse_sub_resource_dummy(DummyReadData *p_data, VariantParser::Stream *p_stream, Ref<Resource> &r_res, int &line, String &r_err_str) {

	VariantParser::Token token;
	VariantParser::get_token(p_stream, token, line, r_err_str);
	if (token.type != VariantParser::TK_NUMBER) {
		r_err_str = "Expected number (sub-resource index)";
		return ERR_PARSE_ERROR;
	}

	int index = token.value;

	if (!p_data->resource_map.has(index)) {
		Ref<DummyResource> dr;
		dr.instance();
		dr->set_subindex(index);
		p_data->resource_map[index] = dr;
		p_data->resource_set.insert(dr);
	}

	r_res = p_data->resource_map[index];

	VariantParser::get_token(p_stream, token, line, r_err_str);
	if (token.type != VariantParser::TK_PARENTHESIS_CLOSE) {
		r_err_str = "Expected ')'";
		return ERR_PARSE_ERROR;
	}

	return OK;
}

Error ResourceInteractiveLoaderText::_parse_ext_resource_dummy(DummyReadData *p_data, VariantParser::Stream *p_stream, Ref<Resource> &r_res, int &line, String &r_err_str) {

	VariantParser::Token token;
	VariantParser::get_token(p_stream, token, line, r_err_str);
	if (token.type != VariantParser::TK_NUMBER) {
		r_err_str = "Expected number (sub-resource index)";
		return ERR_PARSE_ERROR;
	}

	int id = token.value;

	ERR_FAIL_COND_V(!p_data->rev_external_resources.has(id), ERR_PARSE_ERROR);

	r_res = p_data->rev_external_resources[id];

	VariantParser::get_token(p_stream, token, line, r_err_str);
	if (token.type != VariantParser::TK_PARENTHESIS_CLOSE) {
		r_err_str = "Expected ')'";
		return ERR_PARSE_ERROR;
	}

	return OK;
}

Error ResourceInteractiveLoaderText::_parse_sub_resource(VariantParser::Stream *p_stream, Ref<Resource> &r_res, int &line, String &r_err_str) {

	VariantParser::Token token;
	VariantParser::get_token(p_stream, token, line, r_err_str);
	if (token.type != VariantParser::TK_NUMBER) {
		r_err_str = "Expected number (sub-resource index)";
		return ERR_PARSE_ERROR;
	}

	int index = token.value;

	String path = local_path + "::" + itos(index);

	if (!ignore_resource_parsing) {

		if (!ResourceCache::has(path)) {
			r_err_str = "Can't load cached sub-resource: " + path;
			return ERR_PARSE_ERROR;
		}

		r_res = RES(ResourceCache::get(path));
	} else {
		r_res = RES();
	}

	VariantParser::get_token(p_stream, token, line, r_err_str);
	if (token.type != VariantParser::TK_PARENTHESIS_CLOSE) {
		r_err_str = "Expected ')'";
		return ERR_PARSE_ERROR;
	}

	return OK;
}

Error ResourceInteractiveLoaderText::_parse_ext_resource(VariantParser::Stream *p_stream, Ref<Resource> &r_res, int &line, String &r_err_str) {

	VariantParser::Token token;
	VariantParser::get_token(p_stream, token, line, r_err_str);
	if (token.type != VariantParser::TK_NUMBER) {
		r_err_str = "Expected number (sub-resource index)";
		return ERR_PARSE_ERROR;
	}

	int id = token.value;

	if (!ignore_resource_parsing) {

		if (!ext_resources.has(id)) {
			r_err_str = "Can't load cached ext-resource #" + itos(id);
			return ERR_PARSE_ERROR;
		}

		String path = ext_resources[id].path;
		String type = ext_resources[id].type;

		if (path.find("://") == -1 && path.is_rel_path()) {
			// path is relative to file being loaded, so convert to a resource path
			path = ProjectSettings::get_singleton()->localize_path(res_path.get_base_dir().plus_file(path));
		}

		r_res = ResourceLoader::load(path, type);

		if (r_res.is_null()) {
			WARN_PRINT(String("Couldn't load external resource: " + path).utf8().get_data());
		}
	} else {
		r_res = RES();
	}

	VariantParser::get_token(p_stream, token, line, r_err_str);
	if (token.type != VariantParser::TK_PARENTHESIS_CLOSE) {
		r_err_str = "Expected ')'";
		return ERR_PARSE_ERROR;
	}

	return OK;
}

Ref<PackedScene> ResourceInteractiveLoaderText::_parse_node_tag(VariantParser::ResourceParser &parser) {
	Ref<PackedScene> packed_scene;
	packed_scene.instance();

	while (true) {

		if (next_tag.name == "node") {

			int parent = -1;
			int owner = -1;
			int type = -1;
			int name = -1;
			int instance = -1;
			int index = -1;
			//int base_scene=-1;

			if (next_tag.fields.has("name")) {
				name = packed_scene->get_state()->add_name(next_tag.fields["name"]);
			}

			if (next_tag.fields.has("parent")) {
				NodePath np = next_tag.fields["parent"];
				np.prepend_period(); //compatible to how it manages paths internally
				parent = packed_scene->get_state()->add_node_path(np);
			}

			if (next_tag.fields.has("type")) {
				type = packed_scene->get_state()->add_name(next_tag.fields["type"]);
			} else {
				type = SceneState::TYPE_INSTANCED; //no type? assume this was instanced
			}

			if (next_tag.fields.has("instance")) {

				instance = packed_scene->get_state()->add_value(next_tag.fields["instance"]);

				if (packed_scene->get_state()->get_node_count() == 0 && parent == -1) {
					packed_scene->get_state()->set_base_scene(instance);
					instance = -1;
				}
			}

			if (next_tag.fields.has("instance_placeholder")) {

				String path = next_tag.fields["instance_placeholder"];

				int path_v = packed_scene->get_state()->add_value(path);

				if (packed_scene->get_state()->get_node_count() == 0) {
					error = ERR_FILE_CORRUPT;
					error_text = "Instance Placeholder can't be used for inheritance.";
					_printerr();
					return Ref<PackedScene>();
				}

				instance = path_v | SceneState::FLAG_INSTANCE_IS_PLACEHOLDER;
			}

			if (next_tag.fields.has("owner")) {
				owner = packed_scene->get_state()->add_node_path(next_tag.fields["owner"]);
			} else {
				if (parent != -1 && !(type == SceneState::TYPE_INSTANCED && instance == -1))
					owner = 0; //if no owner, owner is root
			}

			if (next_tag.fields.has("index")) {
				index = next_tag.fields["index"];
			}

			int node_id = packed_scene->get_state()->add_node(parent, owner, type, name, instance, index);

			if (next_tag.fields.has("groups")) {

				Array groups = next_tag.fields["groups"];
				for (int i = 0; i < groups.size(); i++) {
					packed_scene->get_state()->add_node_group(node_id, packed_scene->get_state()->add_name(groups[i]));
				}
			}

			while (true) {

				String assign;
				Variant value;

				error = VariantParser::parse_tag_assign_eof(&stream, lines, error_text, next_tag, assign, value, &parser);

				if (error) {
					if (error != ERR_FILE_EOF) {
						_printerr();
						return Ref<PackedScene>();
					} else {
						return packed_scene;
					}
				}

				if (assign != String()) {
					int nameidx = packed_scene->get_state()->add_name(assign);
					int valueidx = packed_scene->get_state()->add_value(value);
					packed_scene->get_state()->add_node_property(node_id, nameidx, valueidx);
					//it's assignment
				} else if (next_tag.name != String()) {
					break;
				}
			}
		} else if (next_tag.name == "connection") {

			if (!next_tag.fields.has("from")) {
				error = ERR_FILE_CORRUPT;
				error_text = "missing 'from' field fron connection tag";
				return Ref<PackedScene>();
			}

			if (!next_tag.fields.has("to")) {
				error = ERR_FILE_CORRUPT;
				error_text = "missing 'to' field fron connection tag";
				return Ref<PackedScene>();
			}

			if (!next_tag.fields.has("signal")) {
				error = ERR_FILE_CORRUPT;
				error_text = "missing 'signal' field fron connection tag";
				return Ref<PackedScene>();
			}

			if (!next_tag.fields.has("method")) {
				error = ERR_FILE_CORRUPT;
				error_text = "missing 'method' field fron connection tag";
				return Ref<PackedScene>();
			}

			NodePath from = next_tag.fields["from"];
			NodePath to = next_tag.fields["to"];
			StringName method = next_tag.fields["method"];
			StringName signal = next_tag.fields["signal"];
			int flags = CONNECT_PERSIST;
			Array binds;

			if (next_tag.fields.has("flags")) {
				flags = next_tag.fields["flags"];
			}

			if (next_tag.fields.has("binds")) {
				binds = next_tag.fields["binds"];
			}

			Vector<int> bind_ints;
			for (int i = 0; i < binds.size(); i++) {
				bind_ints.push_back(packed_scene->get_state()->add_value(binds[i]));
			}

			packed_scene->get_state()->add_connection(
					packed_scene->get_state()->add_node_path(from.simplified()),
					packed_scene->get_state()->add_node_path(to.simplified()),
					packed_scene->get_state()->add_name(signal),
					packed_scene->get_state()->add_name(method),
					flags,
					bind_ints);

			error = VariantParser::parse_tag(&stream, lines, error_text, next_tag, &parser);

			if (error) {
				if (error != ERR_FILE_EOF) {
					_printerr();
					return Ref<PackedScene>();
				} else {
					return packed_scene;
				}
			}
		} else if (next_tag.name == "editable") {

			if (!next_tag.fields.has("path")) {
				error = ERR_FILE_CORRUPT;
				error_text = "missing 'path' field fron connection tag";
				_printerr();
				return Ref<PackedScene>();
			}

			NodePath path = next_tag.fields["path"];

			packed_scene->get_state()->add_editable_instance(path.simplified());

			error = VariantParser::parse_tag(&stream, lines, error_text, next_tag, &parser);

			if (error) {
				if (error != ERR_FILE_EOF) {
					_printerr();
					return Ref<PackedScene>();
				} else {
					return packed_scene;
				}
			}
		} else {

			error = ERR_FILE_CORRUPT;
			_printerr();
			return Ref<PackedScene>();
		}
	}

	return packed_scene;
}

Error ResourceInteractiveLoaderText::poll() {

	if (error != OK)
		return error;

	if (next_tag.name == "ext_resource") {

		if (!next_tag.fields.has("path")) {
			error = ERR_FILE_CORRUPT;
			error_text = "Missing 'path' in external resource tag";
			_printerr();
			return error;
		}

		if (!next_tag.fields.has("type")) {
			error = ERR_FILE_CORRUPT;
			error_text = "Missing 'type' in external resource tag";
			_printerr();
			return error;
		}

		if (!next_tag.fields.has("id")) {
			error = ERR_FILE_CORRUPT;
			error_text = "Missing 'id' in external resource tag";
			_printerr();
			return error;
		}

		String path = next_tag.fields["path"];
		String type = next_tag.fields["type"];
		int index = next_tag.fields["id"];

		if (path.find("://") == -1 && path.is_rel_path()) {
			// path is relative to file being loaded, so convert to a resource path
			path = ProjectSettings::get_singleton()->localize_path(local_path.get_base_dir().plus_file(path));
		}

		if (remaps.has(path)) {
			path = remaps[path];
		}

		RES res = ResourceLoader::load(path, type);

		if (res.is_null()) {

			if (ResourceLoader::get_abort_on_missing_resources()) {
				error = ERR_FILE_CORRUPT;
				error_text = "[ext_resource] referenced nonexistent resource at: " + path;
				_printerr();
				return error;
			} else {
				ResourceLoader::notify_dependency_error(local_path, path, type);
			}
		} else {

			resource_cache.push_back(res);
		}

		ExtResource er;
		er.path = path;
		er.type = type;
		ext_resources[index] = er;

		error = VariantParser::parse_tag(&stream, lines, error_text, next_tag, &rp);

		if (error) {
			_printerr();
		}

		resource_current++;
		return error;

	} else if (next_tag.name == "sub_resource") {

		if (!next_tag.fields.has("type")) {
			error = ERR_FILE_CORRUPT;
			error_text = "Missing 'type' in external resource tag";
			_printerr();
			return error;
		}

		if (!next_tag.fields.has("id")) {
			error = ERR_FILE_CORRUPT;
			error_text = "Missing 'index' in external resource tag";
			_printerr();
			return error;
		}

		String type = next_tag.fields["type"];
		int id = next_tag.fields["id"];

		String path = local_path + "::" + itos(id);

		//bool exists=ResourceCache::has(path);

		Ref<Resource> res;

		if (!ResourceCache::has(path)) { //only if it doesn't exist

			Object *obj = ClassDB::instance(type);
			if (!obj) {

				error_text += "Can't create sub resource of type: " + type;
				_printerr();
				error = ERR_FILE_CORRUPT;
				return error;
			}

			Resource *r = Object::cast_to<Resource>(obj);
			if (!r) {

				error_text += "Can't create sub resource of type, because not a resource: " + type;
				_printerr();
				error = ERR_FILE_CORRUPT;
				return error;
			}

			res = Ref<Resource>(r);
			resource_cache.push_back(res);
			res->set_path(path);
		}

		resource_current++;

		while (true) {

			String assign;
			Variant value;

			error = VariantParser::parse_tag_assign_eof(&stream, lines, error_text, next_tag, assign, value, &rp);

			if (error) {
				_printerr();
				return error;
			}

			if (assign != String()) {
				if (res.is_valid()) {
					res->set(assign, value);
				}
				//it's assignment
			} else if (next_tag.name != String()) {

				error = OK;
				break;
			} else {
				error = ERR_FILE_CORRUPT;
				error_text = "Premature end of file while parsing [sub_resource]";
				_printerr();
				return error;
			}
		}

		return OK;

	} else if (next_tag.name == "resource") {

		if (is_scene) {

			error_text += "found the 'resource' tag on a scene file!";
			_printerr();
			error = ERR_FILE_CORRUPT;
			return error;
		}

		Object *obj = ClassDB::instance(res_type);
		if (!obj) {

			error_text += "Can't create sub resource of type: " + res_type;
			_printerr();
			error = ERR_FILE_CORRUPT;
			return error;
		}

		Resource *r = Object::cast_to<Resource>(obj);
		if (!r) {

			error_text += "Can't create sub resource of type, because not a resource: " + res_type;
			_printerr();
			error = ERR_FILE_CORRUPT;
			return error;
		}

		resource = Ref<Resource>(r);

		resource_current++;

		while (true) {

			String assign;
			Variant value;

			error = VariantParser::parse_tag_assign_eof(&stream, lines, error_text, next_tag, assign, value, &rp);

			if (error) {
				if (error != ERR_FILE_EOF) {
					_printerr();
				} else {
					if (!ResourceCache::has(res_path)) {
						resource->set_path(res_path);
					}
					resource->set_as_translation_remapped(translation_remapped);
				}
				return error;
			}

			if (assign != String()) {
				resource->set(assign, value);
				//it's assignment
			} else if (next_tag.name != String()) {

				error = ERR_FILE_CORRUPT;
				error_text = "Extra tag found when parsing main resource file";
				_printerr();
				return error;
			} else {
				error = ERR_FILE_EOF;
				return error;
			}
		}

		return OK;

	} else if (next_tag.name == "node") {

		if (!is_scene) {

			error_text += "found the 'node' tag on a resource file!";
			_printerr();
			error = ERR_FILE_CORRUPT;
			return error;
		}

		Ref<PackedScene> packed_scene = _parse_node_tag(rp);

		if (!packed_scene.is_valid())
			return error;

		error = OK;
		//get it here
		resource = packed_scene;
		if (!ResourceCache::has(res_path)) {
			packed_scene->set_path(res_path);
		}

		return ERR_FILE_EOF;

	} else {
		error_text += "Unknown tag in file: " + next_tag.name;
		_printerr();
		error = ERR_FILE_CORRUPT;
		return error;
	}

	return OK;
}

int ResourceInteractiveLoaderText::get_stage() const {

	return resource_current;
}
int ResourceInteractiveLoaderText::get_stage_count() const {

	return resources_total; //+ext_resources;
}

void ResourceInteractiveLoaderText::set_translation_remapped(bool p_remapped) {

	translation_remapped = p_remapped;
}

ResourceInteractiveLoaderText::ResourceInteractiveLoaderText() {
	translation_remapped = false;
}

ResourceInteractiveLoaderText::~ResourceInteractiveLoaderText() {

	memdelete(f);
}

void ResourceInteractiveLoaderText::get_dependencies(FileAccess *p_f, List<String> *p_dependencies, bool p_add_types) {

	open(p_f);
	ignore_resource_parsing = true;
	ERR_FAIL_COND(error != OK);

	while (next_tag.name == "ext_resource") {

		if (!next_tag.fields.has("type")) {
			error = ERR_FILE_CORRUPT;
			error_text = "Missing 'type' in external resource tag";
			_printerr();
			return;
		}

		if (!next_tag.fields.has("id")) {
			error = ERR_FILE_CORRUPT;
			error_text = "Missing 'index' in external resource tag";
			_printerr();
			return;
		}

		String path = next_tag.fields["path"];
		String type = next_tag.fields["type"];

		if (path.find("://") == -1 && path.is_rel_path()) {
			// path is relative to file being loaded, so convert to a resource path
			path = ProjectSettings::get_singleton()->localize_path(local_path.get_base_dir().plus_file(path));
		}

		if (p_add_types) {
			path += "::" + type;
		}

		p_dependencies->push_back(path);

		Error err = VariantParser::parse_tag(&stream, lines, error_text, next_tag, &rp);

		if (err) {
			print_line(error_text + " - " + itos(lines));
			error_text = "Unexpected end of file";
			_printerr();
			error = ERR_FILE_CORRUPT;
		}
	}
}

Error ResourceInteractiveLoaderText::rename_dependencies(FileAccess *p_f, const String &p_path, const Map<String, String> &p_map) {

	open(p_f, true);
	ERR_FAIL_COND_V(error != OK, error);
	ignore_resource_parsing = true;
	//FileAccess

	FileAccess *fw = NULL;

	String base_path = local_path.get_base_dir();

	uint64_t tag_end = f->get_position();

	while (true) {

		Error err = VariantParser::parse_tag(&stream, lines, error_text, next_tag, &rp);

		if (err != OK) {
			if (fw) {
				memdelete(fw);
			}
			error = ERR_FILE_CORRUPT;
			ERR_FAIL_V(error);
		}

		if (next_tag.name != "ext_resource") {

			//nothing was done
			if (!fw)
				return OK;

			break;

		} else {

			if (!fw) {

				fw = FileAccess::open(p_path + ".depren", FileAccess::WRITE);
				if (is_scene) {
					fw->store_line("[gd_scene load_steps=" + itos(resources_total) + " format=" + itos(FORMAT_VERSION) + "]\n");
				} else {
					fw->store_line("[gd_resource type=\"" + res_type + "\" load_steps=" + itos(resources_total) + " format=" + itos(FORMAT_VERSION) + "]\n");
				}
			}

			if (!next_tag.fields.has("path") || !next_tag.fields.has("id") || !next_tag.fields.has("type")) {
				memdelete(fw);
				error = ERR_FILE_CORRUPT;
				ERR_FAIL_V(error);
			}

			String path = next_tag.fields["path"];
			int index = next_tag.fields["id"];
			String type = next_tag.fields["type"];

			bool relative = false;
			if (!path.begins_with("res://")) {
				path = base_path.plus_file(path).simplify_path();
				relative = true;
			}

			if (p_map.has(path)) {
				String np = p_map[path];
				path = np;
			}

			if (relative) {
				//restore relative
				path = base_path.path_to_file(path);
			}

			fw->store_line("[ext_resource path=\"" + path + "\" type=\"" + type + "\" id=" + itos(index) + "]");

			tag_end = f->get_position();
		}
	}

	f->seek(tag_end);

	uint8_t c = f->get_8();
	while (!f->eof_reached()) {
		fw->store_8(c);
		c = f->get_8();
	}
	f->close();

	bool all_ok = fw->get_error() == OK;

	memdelete(fw);

	if (!all_ok) {
		return ERR_CANT_CREATE;
	}

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	da->remove(p_path);
	da->rename(p_path + ".depren", p_path);
	memdelete(da);

	return OK;
}

void ResourceInteractiveLoaderText::open(FileAccess *p_f, bool p_skip_first_tag) {

	error = OK;

	lines = 1;
	f = p_f;

	stream.f = f;
	is_scene = false;
	ignore_resource_parsing = false;
	resource_current = 0;

	VariantParser::Tag tag;
	Error err = VariantParser::parse_tag(&stream, lines, error_text, tag);

	if (err) {

		error = err;
		_printerr();
		return;
	}

	if (tag.fields.has("format")) {
		int fmt = tag.fields["format"];
		if (fmt > FORMAT_VERSION) {
			error_text = "Saved with newer format version";
			_printerr();
			error = ERR_PARSE_ERROR;
			return;
		}
	}

	if (tag.name == "gd_scene") {
		is_scene = true;

	} else if (tag.name == "gd_resource") {
		if (!tag.fields.has("type")) {
			error_text = "Missing 'type' field in 'gd_resource' tag";
			_printerr();
			error = ERR_PARSE_ERROR;
			return;
		}

		res_type = tag.fields["type"];

	} else {
		error_text = "Unrecognized file type: " + tag.name;
		_printerr();
		error = ERR_PARSE_ERROR;
		return;
	}

	if (tag.fields.has("load_steps")) {
		resources_total = tag.fields["load_steps"];
	} else {
		resources_total = 0;
	}

	if (!p_skip_first_tag) {

		err = VariantParser::parse_tag(&stream, lines, error_text, next_tag, &rp);

		if (err) {
			error_text = "Unexpected end of file";
			_printerr();
			error = ERR_FILE_CORRUPT;
		}
	}

	rp.ext_func = _parse_ext_resources;
	rp.sub_func = _parse_sub_resources;
	rp.func = NULL;
	rp.userdata = this;
}

static void bs_save_unicode_string(FileAccess *f, const String &p_string, bool p_bit_on_len = false) {

	CharString utf8 = p_string.utf8();
	if (p_bit_on_len) {
		f->store_32(utf8.length() + 1 | 0x80000000);
	} else {
		f->store_32(utf8.length() + 1);
	}
	f->store_buffer((const uint8_t *)utf8.get_data(), utf8.length() + 1);
}

Error ResourceInteractiveLoaderText::save_as_binary(FileAccess *p_f, const String &p_path) {

	if (error)
		return error;

	FileAccessRef wf = FileAccess::open(p_path, FileAccess::WRITE);
	if (!wf) {
		return ERR_CANT_OPEN;
	}

	//save header compressed
	static const uint8_t header[4] = { 'R', 'S', 'R', 'C' };
	wf->store_buffer(header, 4);

	wf->store_32(0); //endianness, little endian
	wf->store_32(0); //64 bits file, false for now
	wf->store_32(VERSION_MAJOR);
	wf->store_32(VERSION_MINOR);
	static const int save_format_version = 3; //use format version 3 for saving
	wf->store_32(save_format_version);

	bs_save_unicode_string(wf.f, is_scene ? "PackedScene" : resource_type);
	wf->store_64(0); //offset to import metadata, this is no longer used
	for (int i = 0; i < 14; i++)
		wf->store_32(0); // reserved

	wf->store_32(0); //string table size, will not be in use
	size_t ext_res_count_pos = wf->get_position();

	wf->store_32(0); //zero ext resources, still parsing them

	//go with external resources

	DummyReadData dummy_read;
	VariantParser::ResourceParser rp;
	rp.ext_func = _parse_ext_resource_dummys;
	rp.sub_func = _parse_sub_resource_dummys;
	rp.userdata = &dummy_read;

	while (next_tag.name == "ext_resource") {

		if (!next_tag.fields.has("path")) {
			error = ERR_FILE_CORRUPT;
			error_text = "Missing 'path' in external resource tag";
			_printerr();
			return error;
		}

		if (!next_tag.fields.has("type")) {
			error = ERR_FILE_CORRUPT;
			error_text = "Missing 'type' in external resource tag";
			_printerr();
			return error;
		}

		if (!next_tag.fields.has("id")) {
			error = ERR_FILE_CORRUPT;
			error_text = "Missing 'id' in external resource tag";
			_printerr();
			return error;
		}

		String path = next_tag.fields["path"];
		String type = next_tag.fields["type"];
		int index = next_tag.fields["id"];

		bs_save_unicode_string(wf.f, type);
		bs_save_unicode_string(wf.f, path);

		int lindex = dummy_read.external_resources.size();
		Ref<DummyResource> dr;
		dr.instance();
		dr->set_path("res://dummy" + itos(lindex)); //anything is good to detect it for saving as external
		dummy_read.external_resources[dr] = lindex;
		dummy_read.rev_external_resources[index] = dr;

		error = VariantParser::parse_tag(&stream, lines, error_text, next_tag, &rp);

		if (error) {
			_printerr();
			return error;
		}
	}

	// save external resource table
	wf->seek(ext_res_count_pos);
	wf->store_32(dummy_read.external_resources.size());
	wf->seek_end();

	//now, save resources to a separate file, for now

	size_t sub_res_count_pos = wf->get_position();
	wf->store_32(0); //zero sub resources, still parsing them

	String temp_file = p_path + ".temp";
	FileAccessRef wf2 = FileAccess::open(temp_file, FileAccess::WRITE);
	if (!wf2) {
		return ERR_CANT_OPEN;
	}

	Vector<size_t> local_offsets;
	Vector<size_t> local_pointers_pos;

	while (next_tag.name == "sub_resource" || next_tag.name == "resource") {

		String type;
		int id = -1;
		bool main_res;

		if (next_tag.name == "sub_resource") {
			if (!next_tag.fields.has("type")) {
				error = ERR_FILE_CORRUPT;
				error_text = "Missing 'type' in external resource tag";
				_printerr();
				return error;
			}

			if (!next_tag.fields.has("id")) {
				error = ERR_FILE_CORRUPT;
				error_text = "Missing 'index' in external resource tag";
				_printerr();
				return error;
			}

			type = next_tag.fields["type"];
			id = next_tag.fields["id"];
			main_res = false;
		} else {
			type = res_type;
			id = 0; //used for last anyway
			main_res = true;
		}

		local_offsets.push_back(wf2->get_position());

		bs_save_unicode_string(wf, "local://" + itos(id));
		local_pointers_pos.push_back(wf->get_position());
		wf->store_64(0); //temp local offset

		bs_save_unicode_string(wf2, type);
		size_t propcount_ofs = wf2->get_position();
		wf2->store_32(0);

		int prop_count = 0;

		while (true) {

			String assign;
			Variant value;

			error = VariantParser::parse_tag_assign_eof(&stream, lines, error_text, next_tag, assign, value, &rp);

			if (error) {
				if (main_res && error == ERR_FILE_EOF) {
					next_tag.name = ""; //exit
					break;
				}

				_printerr();
				return error;
			}

			if (assign != String()) {

				Map<StringName, int> empty_string_map; //unused
				bs_save_unicode_string(wf2, assign, true);
				ResourceFormatSaverBinaryInstance::write_variant(wf2, value, dummy_read.resource_set, dummy_read.external_resources, empty_string_map);
				prop_count++;

			} else if (next_tag.name != String()) {

				error = OK;
				break;
			} else {
				error = ERR_FILE_CORRUPT;
				error_text = "Premature end of file while parsing [sub_resource]";
				_printerr();
				return error;
			}
		}

		wf2->seek(propcount_ofs);
		wf2->store_32(prop_count);
		wf2->seek_end();
	}

	if (next_tag.name == "node") {
		//this is a node, must save one more!

		if (!is_scene) {

			error_text += "found the 'node' tag on a resource file!";
			_printerr();
			error = ERR_FILE_CORRUPT;
			return error;
		}

		Ref<PackedScene> packed_scene = _parse_node_tag(rp);

		if (!packed_scene.is_valid())
			return error;

		error = OK;
		//get it here
		List<PropertyInfo> props;
		packed_scene->get_property_list(&props);

		bs_save_unicode_string(wf, "local://0");
		local_pointers_pos.push_back(wf->get_position());
		wf->store_64(0); //temp local offset

		local_offsets.push_back(wf2->get_position());
		bs_save_unicode_string(wf2, "PackedScene");
		size_t propcount_ofs = wf2->get_position();
		wf2->store_32(0);

		int prop_count = 0;

		for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {

			if (!(E->get().usage & PROPERTY_USAGE_STORAGE))
				continue;

			String name = E->get().name;
			Variant value = packed_scene->get(name);

			Map<StringName, int> empty_string_map; //unused
			bs_save_unicode_string(wf2, name, true);
			ResourceFormatSaverBinaryInstance::write_variant(wf2, value, dummy_read.resource_set, dummy_read.external_resources, empty_string_map);
			prop_count++;
		}

		wf2->seek(propcount_ofs);
		wf2->store_32(prop_count);
		wf2->seek_end();
	}

	wf2->close();

	size_t offset_from = wf->get_position();
	wf->seek(sub_res_count_pos); //plus one because the saved one
	wf->store_32(local_offsets.size());

	for (int i = 0; i < local_offsets.size(); i++) {
		wf->seek(local_pointers_pos[i]);
		wf->store_64(local_offsets[i] + offset_from);
	}

	wf->seek_end();

	Vector<uint8_t> data = FileAccess::get_file_as_array(temp_file);
	wf->store_buffer(data.ptr(), data.size());
	{
		DirAccessRef dar = DirAccess::open(temp_file.get_base_dir());
		dar->remove(temp_file);
	}

	wf->store_buffer((const uint8_t *)"RSRC", 4); //magic at end

	wf->close();

	return OK;
}

String ResourceInteractiveLoaderText::recognize(FileAccess *p_f) {

	error = OK;

	lines = 1;
	f = p_f;

	stream.f = f;

	ignore_resource_parsing = true;

	VariantParser::Tag tag;
	Error err = VariantParser::parse_tag(&stream, lines, error_text, tag);

	if (err) {
		_printerr();
		return "";
	}

	if (tag.fields.has("format")) {
		int fmt = tag.fields["format"];
		if (fmt > FORMAT_VERSION) {
			error_text = "Saved with newer format version";
			_printerr();
			return "";
		}
	}

	if (tag.name == "gd_scene")
		return "PackedScene";

	if (tag.name != "gd_resource")
		return "";

	if (!tag.fields.has("type")) {
		error_text = "Missing 'type' field in 'gd_resource' tag";
		_printerr();
		return "";
	}

	return tag.fields["type"];
}

/////////////////////

Ref<ResourceInteractiveLoader> ResourceFormatLoaderText::load_interactive(const String &p_path, const String &p_original_path, Error *r_error) {

	if (r_error)
		*r_error = ERR_CANT_OPEN;

	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);

	if (err != OK) {

		ERR_FAIL_COND_V(err != OK, Ref<ResourceInteractiveLoader>());
	}

	Ref<ResourceInteractiveLoaderText> ria = memnew(ResourceInteractiveLoaderText);
	String path = p_original_path != "" ? p_original_path : p_path;
	ria->local_path = ProjectSettings::get_singleton()->localize_path(path);
	ria->res_path = ria->local_path;
	//ria->set_local_path( ProjectSettings::get_singleton()->localize_path(p_path) );
	ria->open(f);

	return ria;
}

void ResourceFormatLoaderText::get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) const {

	if (p_type == "") {
		get_recognized_extensions(p_extensions);
		return;
	}

	if (p_type == "PackedScene")
		p_extensions->push_back("tscn");
	else
		p_extensions->push_back("tres");
}

void ResourceFormatLoaderText::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("tscn");
	p_extensions->push_back("tres");
}

bool ResourceFormatLoaderText::handles_type(const String &p_type) const {

	return true;
}
String ResourceFormatLoaderText::get_resource_type(const String &p_path) const {

	String ext = p_path.get_extension().to_lower();
	if (ext == "tscn")
		return "PackedScene";
	else if (ext != "tres")
		return String();

	//for anyhting else must test..

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	if (!f) {

		return ""; //could not rwead
	}

	Ref<ResourceInteractiveLoaderText> ria = memnew(ResourceInteractiveLoaderText);
	ria->local_path = ProjectSettings::get_singleton()->localize_path(p_path);
	ria->res_path = ria->local_path;
	//ria->set_local_path( ProjectSettings::get_singleton()->localize_path(p_path) );
	String r = ria->recognize(f);
	return r;
}

void ResourceFormatLoaderText::get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types) {

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	if (!f) {

		ERR_FAIL();
	}

	Ref<ResourceInteractiveLoaderText> ria = memnew(ResourceInteractiveLoaderText);
	ria->local_path = ProjectSettings::get_singleton()->localize_path(p_path);
	ria->res_path = ria->local_path;
	//ria->set_local_path( ProjectSettings::get_singleton()->localize_path(p_path) );
	ria->get_dependencies(f, p_dependencies, p_add_types);
}

Error ResourceFormatLoaderText::rename_dependencies(const String &p_path, const Map<String, String> &p_map) {

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	if (!f) {

		ERR_FAIL_V(ERR_CANT_OPEN);
	}

	Ref<ResourceInteractiveLoaderText> ria = memnew(ResourceInteractiveLoaderText);
	ria->local_path = ProjectSettings::get_singleton()->localize_path(p_path);
	ria->res_path = ria->local_path;
	//ria->set_local_path( ProjectSettings::get_singleton()->localize_path(p_path) );
	return ria->rename_dependencies(f, p_path, p_map);
}

Error ResourceFormatLoaderText::convert_file_to_binary(const String &p_src_path, const String &p_dst_path) {

	Error err;
	FileAccess *f = FileAccess::open(p_src_path, FileAccess::READ, &err);

	if (err != OK) {

		ERR_FAIL_COND_V(err != OK, ERR_CANT_OPEN);
	}

	Ref<ResourceInteractiveLoaderText> ria = memnew(ResourceInteractiveLoaderText);
	String path = p_src_path;
	ria->local_path = ProjectSettings::get_singleton()->localize_path(path);
	ria->res_path = ria->local_path;
	//ria->set_local_path( ProjectSettings::get_singleton()->localize_path(p_path) );
	ria->open(f);
	return ria->save_as_binary(f, p_dst_path);
}

/*****************************************************************************************************/
/*****************************************************************************************************/
/*****************************************************************************************************/
/*****************************************************************************************************/
/*****************************************************************************************************/
/*****************************************************************************************************/
/*****************************************************************************************************/
/*****************************************************************************************************/
/*****************************************************************************************************/
/*****************************************************************************************************/

String ResourceFormatSaverTextInstance::_write_resources(void *ud, const RES &p_resource) {

	ResourceFormatSaverTextInstance *rsi = (ResourceFormatSaverTextInstance *)ud;
	return rsi->_write_resource(p_resource);
}

String ResourceFormatSaverTextInstance::_write_resource(const RES &res) {

	if (external_resources.has(res)) {

		return "ExtResource( " + itos(external_resources[res] + 1) + " )";
	} else {

		if (internal_resources.has(res)) {
			return "SubResource( " + itos(internal_resources[res]) + " )";
		} else if (res->get_path().length() && res->get_path().find("::") == -1) {

			//external resource
			String path = relative_paths ? local_path.path_to_file(res->get_path()) : res->get_path();
			return "Resource( \"" + path + "\" )";
		} else {
			ERR_EXPLAIN("Resource was not pre cached for the resource section, bug?");
			ERR_FAIL_V("null");
			//internal resource
		}
	}

	return "null";
}

void ResourceFormatSaverTextInstance::_find_resources(const Variant &p_variant, bool p_main) {

	switch (p_variant.get_type()) {
		case Variant::OBJECT: {

			RES res = p_variant.operator RefPtr();

			if (res.is_null() || external_resources.has(res))
				return;

			if (!p_main && (!bundle_resources) && res->get_path().length() && res->get_path().find("::") == -1) {
				int index = external_resources.size();
				external_resources[res] = index;
				return;
			}

			if (resource_set.has(res))
				return;

			List<PropertyInfo> property_list;

			res->get_property_list(&property_list);
			property_list.sort();

			List<PropertyInfo>::Element *I = property_list.front();

			while (I) {

				PropertyInfo pi = I->get();

				if (pi.usage & PROPERTY_USAGE_STORAGE) {

					Variant v = res->get(I->get().name);
					_find_resources(v);
				}

				I = I->next();
			}

			resource_set.insert(res); //saved after, so the childs it needs are available when loaded
			saved_resources.push_back(res);

		} break;
		case Variant::ARRAY: {

			Array varray = p_variant;
			int len = varray.size();
			for (int i = 0; i < len; i++) {

				Variant v = varray.get(i);
				_find_resources(v);
			}

		} break;
		case Variant::DICTIONARY: {

			Dictionary d = p_variant;
			List<Variant> keys;
			d.get_key_list(&keys);
			for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {

				Variant v = d[E->get()];
				_find_resources(v);
			}
		} break;
		default: {}
	}
}

static String _valprop(const String &p_name) {

	if (p_name.find("\"") != -1 || p_name.find("=") != -1 || p_name.find(" ") != -1)
		return "\"" + p_name.c_escape_multiline() + "\"";
	return p_name;
}

Error ResourceFormatSaverTextInstance::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {

	if (p_path.ends_with(".tscn")) {
		packed_scene = p_resource;
	}

	Error err;
	f = FileAccess::open(p_path, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V(err, ERR_CANT_OPEN);
	FileAccessRef _fref(f);

	local_path = ProjectSettings::get_singleton()->localize_path(p_path);

	relative_paths = p_flags & ResourceSaver::FLAG_RELATIVE_PATHS;
	skip_editor = p_flags & ResourceSaver::FLAG_OMIT_EDITOR_PROPERTIES;
	bundle_resources = p_flags & ResourceSaver::FLAG_BUNDLE_RESOURCES;
	takeover_paths = p_flags & ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS;
	if (!p_path.begins_with("res://")) {
		takeover_paths = false;
	}

	// save resources
	_find_resources(p_resource, true);

	if (packed_scene.is_valid()) {
		//add instances to external resources if saving a packed scene
		for (int i = 0; i < packed_scene->get_state()->get_node_count(); i++) {
			if (packed_scene->get_state()->is_node_instance_placeholder(i))
				continue;

			Ref<PackedScene> instance = packed_scene->get_state()->get_node_instance(i);
			if (instance.is_valid() && !external_resources.has(instance)) {
				int index = external_resources.size();
				external_resources[instance] = index;
			}
		}
	}

	ERR_FAIL_COND_V(err != OK, err);

	{
		String title = packed_scene.is_valid() ? "[gd_scene " : "[gd_resource ";
		if (packed_scene.is_null())
			title += "type=\"" + p_resource->get_class() + "\" ";
		int load_steps = saved_resources.size() + external_resources.size();
		/*
		if (packed_scene.is_valid()) {
			load_steps+=packed_scene->get_node_count();
		}
		//no, better to not use load steps from nodes, no point to that
		*/

		if (load_steps > 1) {
			title += "load_steps=" + itos(load_steps) + " ";
		}
		title += "format=" + itos(FORMAT_VERSION) + "";
		//title+="engine_version=\""+itos(VERSION_MAJOR)+"."+itos(VERSION_MINOR)+"\"";

		f->store_string(title);
		f->store_line("]\n"); //one empty line
	}

	Vector<RES> sorted_er;
	sorted_er.resize(external_resources.size());

	for (Map<RES, int>::Element *E = external_resources.front(); E; E = E->next()) {

		sorted_er[E->get()] = E->key();
	}

	for (int i = 0; i < sorted_er.size(); i++) {
		String p = sorted_er[i]->get_path();

		f->store_string("[ext_resource path=\"" + p + "\" type=\"" + sorted_er[i]->get_save_class() + "\" id=" + itos(i + 1) + "]\n"); //bundled
	}

	if (external_resources.size())
		f->store_line(String()); //separate

	Set<int> used_indices;

	for (List<RES>::Element *E = saved_resources.front(); E; E = E->next()) {

		RES res = E->get();
		if (E->next() && (res->get_path() == "" || res->get_path().find("::") != -1)) {

			if (res->get_subindex() != 0) {
				if (used_indices.has(res->get_subindex())) {
					res->set_subindex(0); //repeated
				} else {
					used_indices.insert(res->get_subindex());
				}
			}
		}
	}

	for (List<RES>::Element *E = saved_resources.front(); E; E = E->next()) {

		RES res = E->get();
		ERR_CONTINUE(!resource_set.has(res));
		bool main = (E->next() == NULL);

		if (main && packed_scene.is_valid())
			break; //save as a scene

		if (main) {
			f->store_line("[resource]\n");
		} else {
			String line = "[sub_resource ";
			if (res->get_subindex() == 0) {
				int new_subindex = 1;
				if (used_indices.size()) {
					new_subindex = used_indices.back()->get() + 1;
				}

				res->set_subindex(new_subindex);
				used_indices.insert(new_subindex);
			}

			int idx = res->get_subindex();
			line += "type=\"" + res->get_class() + "\" id=" + itos(idx);
			f->store_line(line + "]\n");
			if (takeover_paths) {
				res->set_path(p_path + "::" + itos(idx), true);
			}

			internal_resources[res] = idx;
#ifdef TOOLS_ENABLED
			res->set_edited(false);
#endif
		}

		List<PropertyInfo> property_list;
		res->get_property_list(&property_list);
		//property_list.sort();
		for (List<PropertyInfo>::Element *PE = property_list.front(); PE; PE = PE->next()) {

			if (skip_editor && PE->get().name.begins_with("__editor"))
				continue;

			if (PE->get().usage & PROPERTY_USAGE_STORAGE) {

				String name = PE->get().name;
				Variant value = res->get(name);

				if ((PE->get().usage & PROPERTY_USAGE_STORE_IF_NONZERO && value.is_zero()) || (PE->get().usage & PROPERTY_USAGE_STORE_IF_NONONE && value.is_one()))
					continue;

				if (PE->get().type == Variant::OBJECT && value.is_zero() && !(PE->get().usage & PROPERTY_USAGE_STORE_IF_NULL))
					continue;

				String vars;
				VariantWriter::write_to_string(value, vars, _write_resources, this);
				f->store_string(_valprop(name) + " = " + vars + "\n");
			}
		}

		f->store_string("\n");
	}

	if (packed_scene.is_valid()) {
		//if this is a scene, save nodes and connections!
		Ref<SceneState> state = packed_scene->get_state();
		for (int i = 0; i < state->get_node_count(); i++) {

			StringName type = state->get_node_type(i);
			StringName name = state->get_node_name(i);
			int index = state->get_node_index(i);
			NodePath path = state->get_node_path(i, true);
			NodePath owner = state->get_node_owner_path(i);
			Ref<PackedScene> instance = state->get_node_instance(i);
			String instance_placeholder = state->get_node_instance_placeholder(i);
			Vector<StringName> groups = state->get_node_groups(i);

			String header = "[node";
			header += " name=\"" + String(name) + "\"";
			if (type != StringName()) {
				header += " type=\"" + String(type) + "\"";
			}
			if (path != NodePath()) {
				header += " parent=\"" + String(path.simplified()) + "\"";
			}
			if (owner != NodePath() && owner != NodePath(".")) {
				header += " owner=\"" + String(owner.simplified()) + "\"";
			}
			if (index >= 0) {
				header += " index=\"" + itos(index) + "\"";
			}

			if (groups.size()) {
				String sgroups = " groups=[\n";
				for (int j = 0; j < groups.size(); j++) {
					sgroups += "\"" + String(groups[j]).c_escape() + "\",\n";
				}
				sgroups += "]";
				header += sgroups;
			}

			f->store_string(header);

			if (instance_placeholder != String()) {

				String vars;
				f->store_string(" instance_placeholder=");
				VariantWriter::write_to_string(instance_placeholder, vars, _write_resources, this);
				f->store_string(vars);
			}

			if (instance.is_valid()) {

				String vars;
				f->store_string(" instance=");
				VariantWriter::write_to_string(instance, vars, _write_resources, this);
				f->store_string(vars);
			}

			f->store_line("]\n");

			for (int j = 0; j < state->get_node_property_count(i); j++) {

				String vars;
				VariantWriter::write_to_string(state->get_node_property_value(i, j), vars, _write_resources, this);

				f->store_string(_valprop(String(state->get_node_property_name(i, j))) + " = " + vars + "\n");
			}

			if (state->get_node_property_count(i)) {
				//add space
				f->store_line(String());
			}
		}

		for (int i = 0; i < state->get_connection_count(); i++) {

			String connstr = "[connection";
			connstr += " signal=\"" + String(state->get_connection_signal(i)) + "\"";
			connstr += " from=\"" + String(state->get_connection_source(i).simplified()) + "\"";
			connstr += " to=\"" + String(state->get_connection_target(i).simplified()) + "\"";
			connstr += " method=\"" + String(state->get_connection_method(i)) + "\"";
			int flags = state->get_connection_flags(i);
			if (flags != Object::CONNECT_PERSIST) {
				connstr += " flags=" + itos(flags);
			}

			Array binds = state->get_connection_binds(i);
			f->store_string(connstr);
			if (binds.size()) {
				String vars;
				VariantWriter::write_to_string(binds, vars, _write_resources, this);
				f->store_string(" binds= " + vars);
			}

			f->store_line("]\n");
		}

		f->store_line(String());

		Vector<NodePath> editable_instances = state->get_editable_instances();
		for (int i = 0; i < editable_instances.size(); i++) {
			f->store_line("[editable path=\"" + editable_instances[i].operator String() + "\"]");
		}
	}

	if (f->get_error() != OK && f->get_error() != ERR_FILE_EOF) {
		f->close();
		return ERR_CANT_CREATE;
	}

	f->close();
	//memdelete(f);

	return OK;
}

Error ResourceFormatSaverText::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {

	if (p_path.ends_with(".sct") && p_resource->get_class() != "PackedScene") {
		return ERR_FILE_UNRECOGNIZED;
	}

	ResourceFormatSaverTextInstance saver;
	return saver.save(p_path, p_resource, p_flags);
}

bool ResourceFormatSaverText::recognize(const RES &p_resource) const {

	return true; // all recognized!
}
void ResourceFormatSaverText::get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const {

	if (p_resource->get_class() == "PackedScene")
		p_extensions->push_back("tscn"); //text scene
	else
		p_extensions->push_back("tres"); //text resource
}

ResourceFormatSaverText *ResourceFormatSaverText::singleton = NULL;
ResourceFormatSaverText::ResourceFormatSaverText() {
	singleton = this;
}
