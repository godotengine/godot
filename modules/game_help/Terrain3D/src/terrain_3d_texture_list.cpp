// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

// #include <godot_cpp/classes/image_texture.hpp>
// #include <godot_cpp/classes/resource_saver.hpp>
#include "scene/resources/image_texture.h"
#include "core/io/resource_saver.h"

#include "logger.h"
#include "terrain_3d_texture_list.h"
#include "util.h"

///////////////////////////
// Private Functions
///////////////////////////

void Terrain3DTextureList::_swap_textures(int p_old_id, int p_new_id) {
	if (p_old_id < 0 || p_old_id >= _textures.size()) {
		LOG(ERROR, "Old id out of range: ", p_old_id);
		return;
	}
	Ref<Terrain3DTexture> texture_a = _textures[p_old_id];

	p_new_id = CLAMP(p_new_id, 0, _textures.size() - 1);
	if (p_new_id == p_old_id) {
		// Texture_a new id was likely out of range, reset it
		texture_a->get_data()->_texture_id = p_old_id;
		return;
	}

	LOG(DEBUG, "Swapping textures id: ", p_old_id, " and id:", p_new_id);
	Ref<Terrain3DTexture> texture_b = _textures[p_new_id];
	texture_a->get_data()->_texture_id = p_new_id;
	texture_b->get_data()->_texture_id = p_old_id;
	_textures[p_new_id] = texture_a;
	_textures[p_old_id] = texture_b;

	update_list();
}

void Terrain3DTextureList::_update_texture_files() {
	LOG(DEBUG, "Received texture_changed signal");
	_generated_albedo_textures.clear();
	_generated_normal_textures.clear();
	_update_texture_data(true, false);
}

void Terrain3DTextureList::_update_texture_settings() {
	LOG(DEBUG, "Received setting_changed signal");
	_update_texture_data(false, true);
}

void Terrain3DTextureList::_update_texture_data(bool p_textures, bool p_settings) {
	bool changed = false;
	Array signal_args;

	if (!_textures.is_empty() && p_textures) {
		LOG(INFO, "Validating texture sizes");
		Vector2i albedo_size = Vector2i(0, 0);
		Vector2i normal_size = Vector2i(0, 0);

		Image::Format albedo_format = Image::FORMAT_MAX;
		Image::Format normal_format = Image::FORMAT_MAX;
		bool albedo_mipmaps = true;
		bool normal_mipmaps = true;

		// Detect image sizes and formats
		for (int i = 0; i < _textures.size(); i++) {
			Ref<Terrain3DTexture> texture_set = _textures[i];
			if (texture_set.is_null()) {
				continue;
			}
			Ref<Texture2D> albedo_tex = texture_set->get_albedo_texture();
			Ref<Texture2D> normal_tex = texture_set->get_normal_texture();

			// If this is the first texture, set expected size and format for the arrays
			if (albedo_tex.is_valid()) {
				Vector2i tex_size = albedo_tex->get_size();
				if (albedo_size.length() == 0.0) {
					albedo_size = tex_size;
				} else if (tex_size != albedo_size) {
					LOG(ERROR, "Texture ID ", i, " albedo size: ", tex_size, " doesn't match first texture: ", albedo_size);
					return;
				}
				Ref<Image> img = albedo_tex->get_image();
				Image::Format format = img->get_format();
				if (albedo_format == Image::FORMAT_MAX) {
					albedo_format = format;
					albedo_mipmaps = img->has_mipmaps();
				} else if (format != albedo_format) {
					LOG(ERROR, "Texture ID ", i, " albedo format: ", format, " doesn't match first texture: ", albedo_format);
					return;
				}
			}
			if (normal_tex.is_valid()) {
				Vector2i tex_size = normal_tex->get_size();
				if (normal_size.length() == 0.0) {
					normal_size = tex_size;
				} else if (tex_size != normal_size) {
					LOG(ERROR, "Texture ID ", i, " normal size: ", tex_size, " doesn't match first texture: ", normal_size);
					return;
				}
				Ref<Image> img = normal_tex->get_image();
				Image::Format format = img->get_format();
				if (normal_format == Image::FORMAT_MAX) {
					normal_format = format;
					normal_mipmaps = img->has_mipmaps();
				} else if (format != normal_format) {
					LOG(ERROR, "Texture ID ", i, " normal format: ", format, " doesn't match first texture: ", normal_format);
					return;
				}
			}
		}

		if (normal_size == Vector2i(0, 0)) {
			normal_size = albedo_size;
		} else if (albedo_size == Vector2i(0, 0)) {
			albedo_size = normal_size;
		}
		if (albedo_size == Vector2i(0, 0)) {
			albedo_size = Vector2i(1024, 1024);
			normal_size = Vector2i(1024, 1024);
		}

		// Generate TextureArrays and replace nulls with a empty image
		if (_generated_albedo_textures.is_dirty() && albedo_size != Vector2i(0, 0)) {
			LOG(INFO, "Regenerating albedo texture array");
			Array albedo_texture_array;

			for (int i = 0; i < _textures.size(); i++) {
				Ref<Terrain3DTexture> texture_set = _textures[i];
				if (texture_set.is_null()) {
					continue;
				}
				Ref<Texture2D> tex = texture_set->get_albedo_texture();
				Ref<Image> img;

				if (tex.is_null()) {
					img = Util::get_filled_image(albedo_size, COLOR_CHECKED, albedo_mipmaps, albedo_format);
					LOG(DEBUG, "ID ", i, " albedo texture is null. Creating a new one. Format: ", img->get_format());
					texture_set->get_data()->_albedo_texture = ImageTexture::create_from_image(img);
				} else {
					img = tex->get_image();
					LOG(DEBUG, "ID ", i, " albedo texture is valid. Format: ", img->get_format());
				}
				albedo_texture_array.push_back(img);
			}
			if (!albedo_texture_array.is_empty()) {
				_generated_albedo_textures.create(albedo_texture_array);
				changed = true;
			}
		}

		if (_generated_normal_textures.is_dirty() && normal_size != Vector2i(0, 0)) {
			LOG(INFO, "Regenerating normal texture arrays");

			Array normal_texture_array;

			for (int i = 0; i < _textures.size(); i++) {
				Ref<Terrain3DTexture> texture_set = _textures[i];
				if (texture_set.is_null()) {
					continue;
				}
				Ref<Texture2D> tex = texture_set->get_normal_texture();
				Ref<Image> img;

				if (tex.is_null()) {
					img = Util::get_filled_image(normal_size, COLOR_NORMAL, normal_mipmaps, normal_format);
					LOG(DEBUG, "ID ", i, " normal texture is null. Creating a new one. Format: ", img->get_format());
					texture_set->get_data()->_normal_texture = ImageTexture::create_from_image(img);
				} else {
					img = tex->get_image();
					LOG(DEBUG, "ID ", i, " Normal texture is valid. Format: ", img->get_format());
				}
				normal_texture_array.push_back(img);
			}
			if (!normal_texture_array.is_empty()) {
				_generated_normal_textures.create(normal_texture_array);
				changed = true;
			}
		}
	}
	signal_args.push_back(_textures.size());
	signal_args.push_back(_generated_albedo_textures.get_rid());
	signal_args.push_back(_generated_normal_textures.get_rid());

	if (!_textures.is_empty() && p_settings) {
		LOG(INFO, "Updating terrain color and scale arrays");
		PackedFloat32Array uv_scales;
		PackedFloat32Array uv_rotations;
		PackedColorArray colors;

		for (int i = 0; i < _textures.size(); i++) {
			Ref<Terrain3DTexture> texture_set = _textures[i];
			if (texture_set.is_null()) {
				continue;
			}
			uv_scales.push_back(texture_set->get_uv_scale());
			uv_rotations.push_back(texture_set->get_uv_rotation());
			colors.push_back(texture_set->get_albedo_color());
		}
		signal_args.push_back(uv_rotations);
		signal_args.push_back(uv_scales);
		signal_args.push_back(colors);
	}

	emit_signal("textures_changed", signal_args);
}

///////////////////////////
// Public Functions
///////////////////////////

Terrain3DTextureList::Terrain3DTextureList() {
}

Terrain3DTextureList::~Terrain3DTextureList() {
	_generated_albedo_textures.clear();
	_generated_normal_textures.clear();
}

void Terrain3DTextureList::update_list() {
	LOG(INFO, "Reconnecting texture signals");
	for (int i = 0; i < _textures.size(); i++) {
		Ref<Terrain3DTexture> texture_set = _textures[i];

		if (texture_set.is_null()) {
			LOG(ERROR, "Texture at index ", i, " is null, but shouldn't be.");
			continue;
		}
		if (!texture_set->is_connected("file_changed", Callable(this, "_update_texture_files"))) {
			LOG(DEBUG, "Connecting file_changed signal");
			texture_set->connect("file_changed", Callable(this, "_update_texture_files"));
		}
		if (!texture_set->is_connected("setting_changed", Callable(this, "_update_texture_settings"))) {
			LOG(DEBUG, "Connecting setting_changed signal");
			texture_set->connect("setting_changed", Callable(this, "_update_texture_settings"));
		}
	}
	_generated_albedo_textures.clear();
	_generated_normal_textures.clear();
	_update_texture_data(true, true);
}

void Terrain3DTextureList::set_texture(int p_index, const Ref<Terrain3DTexture> &p_texture) {
	LOG(INFO, "Setting texture index: ", p_index);
	if (p_index < 0 || p_index >= MAX_TEXTURES) {
		LOG(ERROR, "Invalid texture index: ", p_index, " range is 0-", MAX_TEXTURES);
		return;
	}
	//Delete texture
	if (p_texture.is_null()) {
		// If final texture, remove it
		if (p_index == get_texture_count() - 1) {
			LOG(DEBUG, "Deleting texture id: ", p_index);
			_textures.pop_back();
		} else if (p_index < get_texture_count()) {
			// Else just clear it
			Ref<Terrain3DTexture> texture = _textures[p_index];
			texture->clear();
			texture->get_data()->_texture_id = p_index;
		}
	} else {
		// Else Insert/Add Texture
		// At end if a high number
		if (p_index >= get_texture_count()) {
			p_texture->get_data()->_texture_id = get_texture_count();
			_textures.push_back(p_texture);
			if (!p_texture->is_connected("id_changed", Callable(this, "_swap_textures"))) {
				LOG(DEBUG, "Connecting to id_changed");
				p_texture->connect("id_changed", Callable(this, "_swap_textures"));
			}
		} else {
			// Else overwrite an existing slot
			_textures[p_index] = p_texture;
		}
	}
	update_list();
}

/**
 * set_textures attempts to keep the texture_id as saved in the resource file.
 * But if an ID is invalid or already taken, the new ID is changed to the next available one
 */
void Terrain3DTextureList::set_textures(const TypedArray<Terrain3DTexture> &p_textures) {
	LOG(INFO, "Setting textures");
	int max_size = CLAMP(p_textures.size(), 0, MAX_TEXTURES);
	_textures.resize(max_size);
	int filled_index = -1;
	// For all provided textures up to MAX SIZE
	for (int i = 0; i < max_size; i++) {
		Ref<Terrain3DTexture> texture = p_textures[i];
		int id = texture->get_texture_id();
		// If saved texture id is in range and doesn't exist, add it
		if (id >= 0 && id < max_size && !_textures[id]) {
			_textures[id] = texture;
		} else {
			// Else texture id is invalid or slot is already taken, insert in next available
			for (int j = filled_index + 1; j < max_size; j++) {
				if (!_textures[j]) {
					texture->set_texture_id(j);
					_textures[j] = texture;
					filled_index = j;
					break;
				}
			}
		}
		if (!texture->is_connected("id_changed", Callable(this, "_swap_textures"))) {
			LOG(DEBUG, "Connecting to id_changed");
			texture->connect("id_changed", Callable(this, "_swap_textures"));
		}
	}
	update_list();
}

void Terrain3DTextureList::save() {
	String path = get_path();
	if (path.get_extension() == "tres" || path.get_extension() == "res") {
		LOG(DEBUG, "Attempting to save texture list to external file: " + path);
		Error err;
		err = ResourceSaver::save(this, path, ResourceSaver::FLAG_COMPRESS);
		ERR_FAIL_COND(err);
		LOG(DEBUG, "ResourceSaver return error (0 is OK): ", err);
		LOG(INFO, "Finished saving texture list");
	}
}

///////////////////////////
// Protected Functions
///////////////////////////

void Terrain3DTextureList::_bind_methods() {
	// Private, but Public workaround until callable_mp is implemented
	// https://github.com/godotengine/godot-cpp/pull/1155
	ClassDB::bind_method(D_METHOD("_swap_textures", "old_id", "new_id"), &Terrain3DTextureList::_swap_textures);
	ClassDB::bind_method(D_METHOD("_update_texture_files"), &Terrain3DTextureList::_update_texture_files);
	ClassDB::bind_method(D_METHOD("_update_texture_settings"), &Terrain3DTextureList::_update_texture_settings);

	// Public
	ClassDB::bind_method(D_METHOD("set_texture", "index", "texture"), &Terrain3DTextureList::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture", "index"), &Terrain3DTextureList::get_texture);
	ClassDB::bind_method(D_METHOD("set_textures", "textures"), &Terrain3DTextureList::set_textures);
	ClassDB::bind_method(D_METHOD("get_textures"), &Terrain3DTextureList::get_textures);
	ClassDB::bind_method(D_METHOD("get_texture_count"), &Terrain3DTextureList::get_texture_count);

	ClassDB::bind_method(D_METHOD("save"), &Terrain3DTextureList::save);

	int ro_flags = PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY;
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "textures", PROPERTY_HINT_ARRAY_TYPE, vformat("%tex_size/%tex_size:%tex_size", Variant::OBJECT, PROPERTY_HINT_RESOURCE_TYPE, "Terrain3DTextureList"), ro_flags), "set_textures", "get_textures");

	BIND_CONSTANT(MAX_TEXTURES);

	ADD_SIGNAL(MethodInfo("textures_changed"));
}
