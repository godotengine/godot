/**************************************************************************/
/*  lottie_texture.cpp                                                    */
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

#include "lottie_texture.h"

#include "core/os/memory.h"

#include <thorvg.h>

LottieTexture2D::LottieTexture2D() {
	tvg::Result res = sw_canvas->push(tvg::cast(picture));
	if (res != tvg::Result::Success) {
		ERR_FAIL_MSG("LottieTexture2D: Couldn't insert ThorVG picture on canvas.");
	}
}

void LottieTexture2D::_load_lottie_json() {
	if (json.is_null()) {
		return;
	}
	String lottie_str = json->get_parsed_text();
	if (lottie_str.is_empty()) {
		// don't sort keys, otherwise ThorVG can't load it
		lottie_str = JSON::stringify(json->get_data(), "", false);
	}
	tvg::Result result = picture->load(lottie_str.utf8(), lottie_str.utf8().size(), "lottie", true);
	if (result != tvg::Result::Success && result != tvg::Result::InsufficientCondition) {
		ERR_FAIL_MSG(vformat("LottieTexture2D: Couldn't load Lottie: %s.",
				result == tvg::Result::InvalidArguments		? "InvalidArguments"
						: result == tvg::Result::NonSupport ? "NonSupport"
															: "Unknown Error"));
	}
}

void LottieTexture2D::_update_image() {
	if (origin_width < 0 && origin_height < 0) {
		float fw, fh;
		picture->size(&fw, &fh);
		origin_width = fw;
		origin_height = fh;
	}
	if (json.is_null() || frame_count <= 0) {
		if (image.is_null()) {
			image = Image::create_empty(1, 1, false, Image::FORMAT_RGBA8);
		}
		if (texture.is_null()) {
			texture = RenderingServer::get_singleton()->texture_2d_create(image);
		}
		return;
	}

	int _columns = columns <= 0 ? Math::ceil(Math::sqrt((float)frame_count)) : columns;
	int _rows = Math::ceil(((float)frame_count) / _columns);

	uint32_t w = MAX(1, round(origin_width * scale));
	uint32_t h = MAX(1, round(origin_height * scale));

	const uint32_t max_dimension = 16384;
	if (w * _columns > max_dimension || h * _rows > max_dimension) {
		WARN_PRINT(vformat(
				String::utf8("LottieTexture2D: Target canvas dimensions %d×%d (with scale %.2f, rows %d, columns %d) exceed the max supported dimensions %d×%d. The target canvas will be scaled down."),
				w, h, scale, _rows, _columns, max_dimension, max_dimension));
		w = MIN(w, max_dimension / _columns);
		h = MIN(h, max_dimension / _rows);
		scale = MIN(w / origin_width, h / origin_height);
	}
	picture->size(w, h);

	bool can_update = image.is_valid() && (uint32_t)image->get_width() == w * _columns && (uint32_t)image->get_height() == h * _rows;
	if (!can_update) {
		image = Image::create_empty(w * _columns, h * _rows, false, Image::FORMAT_RGBA8);
	}

	uint32_t *buffer = (uint32_t *)memalloc(sizeof(uint32_t) * w * h);
	memset(buffer, 0, sizeof(uint32_t) * w * h);

	sw_canvas->sync();
	tvg::Result res = sw_canvas->target(buffer, w, w, h, tvg::SwCanvas::ARGB8888S);
	if (res != tvg::Result::Success) {
		memfree(buffer);
		ERR_FAIL_MSG("LottieTexture2D: Couldn't set target on ThorVG canvas.");
	}

	for (int row = 0; row < _rows; row++) {
		for (int column = 0; column < _columns; column++) {
			if (row * _columns + column >= frame_count) {
				break;
			}
			float progress = ((float)(row * _columns + column)) / frame_count;
			float current_frame = frame_begin + (frame_end - frame_begin) * progress;

			animation->frame(current_frame);
			res = sw_canvas->update(picture);
			if (res != tvg::Result::Success) {
				memfree(buffer);
				ERR_FAIL_MSG("LottieTexture2D: Couldn't update ThorVG pictures on canvas.");
			}
			res = sw_canvas->draw();
			if (res != tvg::Result::Success) {
				memfree(buffer);
				ERR_FAIL_MSG("LottieTexture2D: Couldn't draw ThorVG pictures on canvas.");
			}
			res = sw_canvas->sync();
			if (res != tvg::Result::Success) {
				memfree(buffer);
				ERR_FAIL_MSG("LottieTexture2D: Couldn't sync ThorVG canvas.");
			}

			for (uint32_t y = 0; y < h; y++) {
				for (uint32_t x = 0; x < w; x++) {
					uint32_t n = buffer[y * w + x];
					Color color;
					color.set_r8((n >> 16) & 0xff);
					color.set_g8((n >> 8) & 0xff);
					color.set_b8(n & 0xff);
					color.set_a8((n >> 24) & 0xff);
					image->set_pixel(x + w * column, y + h * row, color);
				}
			}
			sw_canvas->clear(false);
		}
	}
	memfree(buffer);

	if (texture.is_null()) {
		texture = RenderingServer::get_singleton()->texture_2d_create(image);
	} else {
		if (can_update) {
			RenderingServer::get_singleton()->texture_2d_update(texture, image);
		} else {
			RID new_texture = RenderingServer::get_singleton()->texture_2d_create(image);
			RenderingServer::get_singleton()->texture_replace(texture, new_texture);
		}
	}
	emit_changed();
}

bool LottieTexture2D::validate_json(Ref<JSON> p_json) {
	String str = p_json->get_parsed_text();
	if (str.is_empty()) {
		str = p_json->stringify(p_json->get_data(), "", false);
	}
	// use ThorVG to check if it's Lottie file.
	std::unique_ptr<tvg::Picture> picture = tvg::Picture::gen();
	tvg::Result res = picture->load(str.utf8(), str.utf8().size(), "lottie", true);
	return res == tvg::Result::Success;
}

Ref<LottieTexture2D> LottieTexture2D::create_from_json(Ref<JSON> p_json) {
	if (!validate_json(p_json)) {
		return nullptr;
	}
	Ref<LottieTexture2D> ret = memnew(LottieTexture2D);
	Dictionary dict = p_json->get_data();
	float p_scale = dict.get("gd_scale", 1.0);
	float p_frame_begin = dict.get("gd_frame_begin", 0);
	float p_frame_end = dict.get("gd_frame_end", 0);
	int p_frame_count = dict.get("gd_frame_count", 1);
	int p_columns = dict.get("gd_columns", 0);
	ret->update(p_json, p_frame_begin, p_frame_end, p_frame_count, p_scale, p_columns);
	return ret;
}

void LottieTexture2D::update(Ref<JSON> p_json, float p_frame_begin, float p_frame_end, int p_frame_count, float p_scale, int p_columns) {
	frame_begin = p_frame_begin;
	frame_end = p_frame_end;
	frame_count = p_frame_count;
	scale = p_scale;
	json = p_json;
	columns = p_columns;
	_load_lottie_json();
	_update_image();
}

void LottieTexture2D::set_json(Ref<JSON> p_json) {
	json = p_json;
	_load_lottie_json();
	_update_image();
}

void LottieTexture2D::set_scale(float p_scale) {
	scale = p_scale;
	_update_image();
}

void LottieTexture2D::set_frame_begin(float p_frame_begin) {
	frame_begin = CLAMP(p_frame_begin, 0, get_lottie_frame_count());
	if (frame_begin > frame_end) {
		frame_end = frame_begin;
	}
	_update_image();
}

void LottieTexture2D::set_frame_end(float p_frame_end) {
	frame_end = CLAMP(p_frame_end, frame_begin, get_lottie_frame_count() - 1);
	_update_image();
}

void LottieTexture2D::set_frame_count(int p_frame_count) {
	frame_count = p_frame_count;
	_update_image();
};

void LottieTexture2D::set_columns(int p_columns) {
	columns = MIN(p_columns, frame_count);
	_update_image();
}

RID LottieTexture2D::get_rid() const {
	if (texture.is_null()) {
		texture = RenderingServer::get_singleton()->texture_2d_placeholder_create();
	}
	return texture;
}

LottieTexture2D::~LottieTexture2D() {
	if (texture.is_valid()) {
		RenderingServer::get_singleton()->free(texture);
	}
}

void LottieTexture2D::_bind_methods() {
	ClassDB::bind_static_method("LottieTexture2D", D_METHOD("validate_json", "p_json"), &LottieTexture2D::validate_json);
	ClassDB::bind_static_method("LottieTexture2D", D_METHOD("create_from_json", "p_json"), &LottieTexture2D::create_from_json);
	ClassDB::bind_method(D_METHOD("update", "p_json", "p_frame_begin", "p_frame_end", "p_frame_count", "p_scale", "p_columns"), &LottieTexture2D::update);
	ClassDB::bind_method(D_METHOD("set_json", "p_json"), &LottieTexture2D::set_json);
	ClassDB::bind_method(D_METHOD("get_json"), &LottieTexture2D::get_json);
	ClassDB::bind_method(D_METHOD("set_scale", "p_scale"), &LottieTexture2D::set_scale);
	ClassDB::bind_method(D_METHOD("get_scale"), &LottieTexture2D::get_scale);
	ClassDB::bind_method(D_METHOD("set_frame_begin", "p_frame_begin"), &LottieTexture2D::set_frame_begin);
	ClassDB::bind_method(D_METHOD("get_frame_begin"), &LottieTexture2D::get_frame_begin);
	ClassDB::bind_method(D_METHOD("set_frame_end", "p_frame_end"), &LottieTexture2D::set_frame_end);
	ClassDB::bind_method(D_METHOD("get_frame_end"), &LottieTexture2D::get_frame_end);
	ClassDB::bind_method(D_METHOD("set_frame_count", "p_frame_count"), &LottieTexture2D::set_frame_count);
	ClassDB::bind_method(D_METHOD("get_frame_count"), &LottieTexture2D::get_frame_count);
	ClassDB::bind_method(D_METHOD("set_columns", "p_columns"), &LottieTexture2D::set_columns);
	ClassDB::bind_method(D_METHOD("get_columns"), &LottieTexture2D::get_columns);
	ClassDB::bind_method(D_METHOD("get_lottie_duration"), &LottieTexture2D::get_lottie_duration);
	ClassDB::bind_method(D_METHOD("get_lottie_frame_count"), &LottieTexture2D::get_lottie_frame_count);
	ClassDB::bind_method(D_METHOD("get_lottie_image_size"), &LottieTexture2D::get_lottie_image_size);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "json", PROPERTY_HINT_RESOURCE_TYPE, "JSON"), "set_json", "get_json");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "scale"), "set_scale", "get_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "frame_begin"), "set_frame_begin", "get_frame_begin");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "frame_end"), "set_frame_end", "get_frame_end");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "frame_count"), "set_frame_count", "get_frame_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "columns"), "set_columns", "get_columns");
}

////////////////

Ref<Resource> ResourceFormatLoaderLottie::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	Ref<Resource> ret = Ref<Resource>();

	if (!FileAccess::exists(p_path)) {
		*r_error = ERR_FILE_NOT_FOUND;
		return ret;
	}

	Ref<JSON> json;
	json.instantiate();

	Error err = json->parse(FileAccess::get_file_as_string(p_path), true);
	if (err != OK) {
		String err_text = "Error parsing JSON file at '" + p_path + "', on line " + itos(json->get_error_line()) + ": " + json->get_error_message();
		if (r_error) {
			*r_error = err;
		}
		ERR_PRINT(err_text);
		return ret;
	}

	if (get_resource_type(p_path) == "LottieTexture2D") {
		ret = LottieTexture2D::create_from_json(json);
	} else {
		if (r_error) {
			*r_error = ERR_INVALID_DATA;
		}
		return ret;
	}

	if (r_error) {
		*r_error = OK;
	}

	return ret;
}

void ResourceFormatLoaderLottie::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("lottiejson");
}

bool ResourceFormatLoaderLottie::handles_type(const String &p_type) const {
	return p_type == "LottieTexture2D";
}

String ResourceFormatLoaderLottie::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el != "lottiejson") {
		return "";
	}
	return "LottieTexture2D";
}

////////////////

Error ResourceFormatSaverLottie::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	Ref<LottieTexture2D> lottie = p_resource;
	ERR_FAIL_COND_V(lottie.is_null(), ERR_INVALID_PARAMETER);

	// Lottie JSON object allows storing additional data
	Dictionary dict = lottie->get_json().is_valid() ? (Dictionary)lottie->get_json()->get_data() : Dictionary();
	dict["gd_scale"] = lottie->get_scale();
	dict["gd_frame_begin"] = lottie->get_frame_begin();
	dict["gd_frame_end"] = lottie->get_frame_end();
	dict["gd_frame_count"] = lottie->get_frame_count();
	dict["gd_columns"] = lottie->get_columns();

	String source = JSON::stringify(dict, "", false);

	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);

	ERR_FAIL_COND_V_MSG(err, err, "Cannot save lottie json '" + p_path + "'.");

	file->store_string(source);
	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		return ERR_CANT_CREATE;
	}

	return OK;
}

void ResourceFormatSaverLottie::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	Ref<LottieTexture2D> lottie = p_resource;
	if (lottie.is_valid()) {
		p_extensions->push_back("lottiejson");
	}
}

bool ResourceFormatSaverLottie::recognize(const Ref<Resource> &p_resource) const {
	return p_resource->is_class("LottieTexture2D");
}
