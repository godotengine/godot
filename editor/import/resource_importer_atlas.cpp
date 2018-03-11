/*************************************************************************/
/*  resource_importer_atlas.cpp                                          */
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

#include "resource_importer_atlas.h"

#include "core/io/json.h"
#include "io/marshalls.h"
#include "io/resource_saver.h"
#include "os/file_access.h"
#include "scene/resources/texture.h"

Error ResourceImporterAtlas::_pack_atlas(const String &p_base_dir, const String &p_atlas_path, const Array &p_sprites, Array &r_sprites) const {

	ERR_FAIL_COND_V(p_atlas_path.get_extension() != "png", FAILED);

	Vector<Size2i> sizes;
	Vector<Ref<Texture> > textures;

	for (int i = 0; i < p_sprites.size(); i++) {
		ERR_FAIL_COND_V(p_sprites[i].get_type() != Variant::STRING, FAILED);
		String sprite_path = p_sprites[i];
		if (sprite_path.is_rel_path()) {
			sprite_path = p_base_dir + "/" + sprite_path;
		}

		ERR_FAIL_COND_V(sprite_path.get_extension() != "png", FAILED);

		// load images manually without ResourceLoader to allow paths from outside the project.
		Vector<uint8_t> sprite_png = FileAccess::get_file_as_array(sprite_path);
		ERR_FAIL_COND_V(sprite_png.size() == 0, FAILED);

		Ref<Image> image;
		image.instance();
		Error err = image->load_png_from_buffer((Variant)sprite_png);
		ERR_FAIL_COND_V(err != OK, err);

		Ref<ImageTexture> texture;
		texture.instance();
		texture->create_from_image(image);

		textures.push_back(texture);
		Size2 size = texture->get_size();
		sizes.push_back(Size2i(size.width, size.height));
	}

	Vector<Point2i> positions;
	Size2i total_size;

	Geometry::make_atlas(sizes, positions, total_size);

	Ref<Image> image;
	image.instance();
	image->create(total_size.width, total_size.height, false, Image::FORMAT_RGBA8);

	for (int i = 0; i < textures.size(); i++) {
		image->blit_rect(textures[i]->get_data(), Rect2(0, 0, sizes[i].width, sizes[i].height), positions[i]);
	}

	for (int i = 0; i < p_sprites.size(); i++) {
		Dictionary region;
		region["x"] = positions[i].x;
		region["y"] = positions[i].y;
		region["w"] = sizes[i].width;
		region["h"] = sizes[i].height;

		Dictionary packed_sprite;
		packed_sprite["filename"] = p_sprites[i];
		packed_sprite["region"] = region;

		r_sprites.push_back(packed_sprite);
	}

	String atlas_path = p_atlas_path;
	if (atlas_path.is_rel_path()) {
		atlas_path = p_base_dir + "/" + atlas_path;
	}

	return image->save_png(atlas_path);
}

String ResourceImporterAtlas::get_importer_name() const {

	return "gdatlas";
}

String ResourceImporterAtlas::get_visible_name() const {

	return "Texture Atlas";
}
void ResourceImporterAtlas::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("gdatlas");
}

String ResourceImporterAtlas::get_save_extension() const {

	return "gdpackedatlas";
}

String ResourceImporterAtlas::get_resource_type() const {

	return "Atlas";
}

bool ResourceImporterAtlas::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {

	return true;
}

int ResourceImporterAtlas::get_preset_count() const {
	return 0;
}

String ResourceImporterAtlas::get_preset_name(int p_idx) const {

	return String();
}

void ResourceImporterAtlas::get_import_options(List<ImportOption> *r_options, int p_preset) const {
}

Error ResourceImporterAtlas::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files) {

	Vector<uint8_t> buffer = FileAccess::get_file_as_array(p_source_file);
	ERR_FAIL_COND_V(buffer.size() == 0, ERR_FILE_CANT_OPEN);

	String str;
	str.parse_utf8((const char *)buffer.ptr(), buffer.size());

	String error_string;
	int error_line;
	Variant atlas_json;

	Error err = JSON::parse(str, atlas_json, error_string, error_line);
	if (err != OK) {
		ERR_PRINTS("Parse error on atlas " + p_source_file + " at line " + String::num(error_line) + ": " + error_string);
		return err;
	}

	const String base_dir = p_source_file.get_base_dir();
	Dictionary atlas_dict = atlas_json;
	atlas_dict["base_dir"] = base_dir;

	ERR_FAIL_COND_V(atlas_dict["textures"].get_type() != Variant::ARRAY, ERR_FILE_CORRUPT);
	const Array textures = atlas_dict["textures"];
	for (int i = 0; i < textures.size(); i++) {
		ERR_FAIL_COND_V(textures[i].get_type() != Variant::DICTIONARY, ERR_FILE_CORRUPT);
		Dictionary texture = textures[i];
		ERR_FAIL_COND_V(texture["sprites"].get_type() != Variant::ARRAY, ERR_FILE_CORRUPT);
		const Array sprites = texture["sprites"];
		if (sprites.size() > 0 && sprites[0].get_type() == Variant::STRING) {
			Array packed_sprites;
			ERR_FAIL_COND_V(texture["image"].get_type() != Variant::STRING, ERR_FILE_CORRUPT);
			err = _pack_atlas(base_dir, texture["image"], sprites, packed_sprites);
			ERR_FAIL_COND_V(err != OK, err);
			texture["sprites"] = packed_sprites;
		}
	}

	const CharString new_atlas_json = JSON::print(atlas_dict).utf8();
	const String save_path = p_save_path + ".gdpackedatlas";
	FileAccess *file = FileAccess::open(save_path, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V(err != OK || !file, err);
	file->store_buffer((uint8_t *)new_atlas_json.get_data(), new_atlas_json.length());
	file->close();
	memdelete(file);

	return OK;
}

ResourceImporterAtlas::ResourceImporterAtlas() {
}
