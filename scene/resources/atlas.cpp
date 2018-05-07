/*************************************************************************/
/*  atlas.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "atlas.h"

#include "core/io/json.h"
#include "os/file_access.h"

void Atlas::reload_from_file() {

	String path = get_path();
	if (!path.is_resource_file())
		return;

	path = ResourceLoader::path_remap(path); //remap for translation
	path = ResourceLoader::import_remap(path); //remap for import
	if (!path.is_resource_file())
		return;

	load(path);
}

Error Atlas::load(const String &p_path) {

	Vector<uint8_t> buffer = FileAccess::get_file_as_array(p_path);
	ERR_FAIL_COND_V(buffer.size() == 0, ERR_FILE_CANT_OPEN);

	String str;
	str.parse_utf8((const char *)buffer.ptr(), buffer.size());

	String error_string;
	int error_line;
	Variant atlas_json;

	Error err = JSON::parse(str, atlas_json, error_string, error_line);
	if (err != OK) {
		ERR_PRINTS("Parse error on atlas " + p_path + " at line " + String::num(error_line) + ": " + error_string);
		return err;
	}

	ERR_FAIL_COND_V(atlas_json.get_type() != Variant::DICTIONARY, ERR_FILE_CORRUPT);
	const Dictionary atlas_dict = atlas_json;

	ERR_FAIL_COND_V(atlas_dict["textures"].get_type() != Variant::ARRAY, ERR_FILE_CORRUPT);
	const Array textures = atlas_dict["textures"];
	for (int i = 0; i < textures.size(); i++) {

		ERR_FAIL_COND_V(textures[i].get_type() != Variant::DICTIONARY, ERR_FILE_CORRUPT);
		const Dictionary texture = textures[i];

		ERR_FAIL_COND_V(texture["image"].get_type() != Variant::STRING, ERR_FILE_CORRUPT);
		String atlas_path = texture["image"];
		if (atlas_path.is_rel_path()) {
			const String base_dir = atlas_dict["base_dir"];
			atlas_path = base_dir + "/" + atlas_path;
		}

		if (!FileAccess::exists(atlas_path)) {
			return FAILED;
		}
		Ref<Texture> atlas_texture = ResourceLoader::load(atlas_path, "Texture");
		if (!atlas_texture.is_valid()) {
			ERR_PRINTS("Failed to load atlas texture at " + atlas_path + " specified in atlas " + p_path);
			return FAILED;
		}

		const Array texture_sprites = texture["sprites"];
		for (int j = 0; j < texture_sprites.size(); j++) {

			const Dictionary sprite = texture_sprites[j];

			SpriteData data;
			data.atlas = atlas_texture;

			const Dictionary region = sprite["region"];
			data.region = Rect2(region["x"], region["y"], region["w"], region["h"]);

			if (sprite.has("margin")) {
				const Dictionary margin = sprite["margin"];
				data.margin = Rect2(margin["x"], margin["y"], margin["w"], margin["h"]);
			} else {
				data.margin = Rect2(0, 0, 0, 0);
			}

			String name = sprite["filename"];
			name = name.get_file();
			int pos = name.find_last(".");
			if (pos > 0)
				name = name.substr(0, pos);

			sprites[name] = data;
		}
	}

	return OK;
}

Ref<Texture> Atlas::get_texture(const String &p_name) {

	SpriteData *sprite = sprites.getptr(p_name);

	if (!sprite) {
		ERR_PRINTS("Atlas has no texture named " + p_name);
		ERR_FAIL_V(Ref<AtlasTexture>());
	}

	if (sprite->cached.is_valid()) {
		return sprite->cached;
	}

	Ref<AtlasTexture> tex;
	tex.instance();
	tex->set_atlas(sprite->atlas);
	tex->set_region(sprite->region);
	tex->set_margin(sprite->margin);
	sprite->cached = tex;

	return tex;
}

void Atlas::_bind_methods() {

	ClassDB::bind_method(D_METHOD("load", "path"), &Atlas::load);
	ClassDB::bind_method(D_METHOD("get_texture", "name"), &Atlas::get_texture);
}

RES ResourceFormatLoaderAtlas::load(const String &p_path, const String &p_original_path, Error *r_error) {

	Ref<Atlas> atlas;
	atlas.instance();

	String path = p_path;
	path = ResourceLoader::import_remap(path); //remap for import

	Error err;
	if (FileAccess::exists(path))
		err = atlas->load(path);
	else
		err = ERR_FILE_NOT_FOUND;

	if (r_error)
		*r_error = err;
	if (err != OK)
		return RES();

	return atlas;
}

void ResourceFormatLoaderAtlas::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("gdpackedatlas");
}

bool ResourceFormatLoaderAtlas::handles_type(const String &p_type) const {

	return (p_type == "Atlas");
}

String ResourceFormatLoaderAtlas::get_resource_type(const String &p_path) const {

	String el = p_path.get_extension().to_lower();
	if (el == "gdpackedatlas")
		return "Atlas";
	return "";
}
