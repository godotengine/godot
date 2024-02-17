// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

//#include <godot_cpp/classes/image.hpp>
#include "core/io/image.h"

#include "logger.h"
#include "terrain_3d_texture.h"

///////////////////////////
// Private Functions
///////////////////////////

bool Terrain3DTexture::_is_texture_valid(const Ref<Texture2D> &p_texture) const {
	if (p_texture.is_null()) {
		LOG(DEBUG, "Provided texture is null.");
		return true;
	}

	Ref<Image> img = p_texture->get_image();
	Image::Format format = Image::FORMAT_MAX;
	if (img.is_valid()) {
		format = img->get_format();
	}
	if (format < 0 || format >= Image::FORMAT_MAX) {
		LOG(ERROR, "Invalid texture format. See documentation for format specification.");
		return false;
	}

	return true;
}

///////////////////////////
// Public Functions
///////////////////////////

Terrain3DTexture::Terrain3DTexture() {
}

Terrain3DTexture::~Terrain3DTexture() {
}

void Terrain3DTexture::clear() {
	_data._name = "New Texture";
	_data._texture_id = 0;
	_data._albedo_color = Color(1.0f, 1.0f, 1.0f, 1.0f);
	_data._albedo_texture.unref();
	_data._normal_texture.unref();
	_data._uv_scale = 0.1f;
	_data._uv_rotation = 0.0f;
}

void Terrain3DTexture::set_tex_name(String p_name) {
	_data._name = p_name;
	emit_signal("setting_changed");
}

void Terrain3DTexture::set_texture_id(int p_new_id) {
	int old_id = _data._texture_id;
	_data._texture_id = p_new_id;
	emit_signal("id_changed", old_id, p_new_id);
}

void Terrain3DTexture::set_albedo_color(Color p_color) {
	_data._albedo_color = p_color;
	emit_signal("setting_changed");
}

void Terrain3DTexture::set_albedo_texture(const Ref<Texture2D> &p_texture) {
	if (_is_texture_valid(p_texture)) {
		_data._albedo_texture = p_texture;
		emit_signal("file_changed");
	}
}

void Terrain3DTexture::set_normal_texture(const Ref<Texture2D> &p_texture) {
	if (_is_texture_valid(p_texture)) {
		_data._normal_texture = p_texture;
		emit_signal("file_changed");
	}
}

void Terrain3DTexture::set_uv_scale(real_t p_scale) {
	_data._uv_scale = p_scale;
	emit_signal("setting_changed");
}

void Terrain3DTexture::set_uv_rotation(real_t p_rotation) {
	_data._uv_rotation = CLAMP(p_rotation, 0.0f, 1.0f);
	emit_signal("setting_changed");
}

///////////////////////////
// Protected Functions
///////////////////////////

void Terrain3DTexture::_bind_methods() {
	ADD_SIGNAL(MethodInfo("id_changed"));
	ADD_SIGNAL(MethodInfo("file_changed"));
	ADD_SIGNAL(MethodInfo("setting_changed"));

	ClassDB::bind_method(D_METHOD("clear"), &Terrain3DTexture::clear);
	ClassDB::bind_method(D_METHOD("set_tex_name", "name"), &Terrain3DTexture::set_tex_name);
	ClassDB::bind_method(D_METHOD("get_tex_name"), &Terrain3DTexture::get_tex_name);
	ClassDB::bind_method(D_METHOD("set_texture_id", "id"), &Terrain3DTexture::set_texture_id);
	ClassDB::bind_method(D_METHOD("get_texture_id"), &Terrain3DTexture::get_texture_id);
	ClassDB::bind_method(D_METHOD("set_albedo_color", "color"), &Terrain3DTexture::set_albedo_color);
	ClassDB::bind_method(D_METHOD("get_albedo_color"), &Terrain3DTexture::get_albedo_color);
	ClassDB::bind_method(D_METHOD("set_albedo_texture", "texture"), &Terrain3DTexture::set_albedo_texture);
	ClassDB::bind_method(D_METHOD("get_albedo_texture"), &Terrain3DTexture::get_albedo_texture);
	ClassDB::bind_method(D_METHOD("set_normal_texture", "texture"), &Terrain3DTexture::set_normal_texture);
	ClassDB::bind_method(D_METHOD("get_normal_texture"), &Terrain3DTexture::get_normal_texture);
	ClassDB::bind_method(D_METHOD("set_uv_scale", "scale"), &Terrain3DTexture::set_uv_scale);
	ClassDB::bind_method(D_METHOD("get_uv_scale"), &Terrain3DTexture::get_uv_scale);
	ClassDB::bind_method(D_METHOD("set_uv_rotation", "scale"), &Terrain3DTexture::set_uv_rotation);
	ClassDB::bind_method(D_METHOD("get_uv_rotation"), &Terrain3DTexture::get_uv_rotation);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "tex_name", PROPERTY_HINT_NONE), "set_tex_name", "get_tex_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_id", PROPERTY_HINT_NONE), "set_texture_id", "get_texture_id");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "albedo_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_albedo_color", "get_albedo_color");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "albedo_texture", PROPERTY_HINT_RESOURCE_TYPE, "ImageTexture,CompressedTexture2D"), "set_albedo_texture", "get_albedo_texture");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "normal_texture", PROPERTY_HINT_RESOURCE_TYPE, "ImageTexture,CompressedTexture2D"), "set_normal_texture", "get_normal_texture");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "uv_scale", PROPERTY_HINT_RANGE, "0.001, 2.0"), "set_uv_scale", "get_uv_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "uv_rotation", PROPERTY_HINT_RANGE, "0.0, 1.0"), "set_uv_rotation", "get_uv_rotation");
}
