/**************************************************************************/
/*  image_loader_svg.cpp                                                  */
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

#include "lottie_sheet.h"

#include "core/os/memory.h"
#include "core/variant/variant.h"

#include <thorvg.h>

void LottieSheet::_load_data(String p_string, float p_scale) {
	ERR_FAIL_COND_MSG(Math::is_zero_approx(p_scale), "LottieSheet: Can't load Lottie with a scale of 0.");

	tvg::Result result = picture->load(p_string.utf8(), p_string.utf8().size(), "lottie", true);
	if (result != tvg::Result::Success) {
		return;
	}
	float fw, fh;
	picture->size(&fw, &fh);

	uint32_t width = MAX(1, round(fw * p_scale));
	uint32_t height = MAX(1, round(fh * p_scale));

	const uint32_t max_dimension = 16384;
	if (width > max_dimension || height > max_dimension) {
		WARN_PRINT(vformat(
				String::utf8("LottieSheet: Target canvas dimensions %d×%d (with scale %.2f) exceed the max supported dimensions %d×%d. The target canvas will be scaled down."),
				width, height, p_scale, max_dimension, max_dimension));
		width = MIN(width, max_dimension);
		height = MIN(height, max_dimension);
	}

	picture->size(width, height);
	this->width = width;
	this->height = height;
	image = Image::create_empty(width, height, false, Image::FORMAT_RGBA8);
	// Note: memalloc here, be sure to memfree before any return.
	buffer = (uint32_t *)(buffer == nullptr ? memalloc(sizeof(uint32_t) * width * height) : memrealloc(buffer, sizeof(uint32_t) * width * height));
}

Ref<LottieSheet> LottieSheet::load_json(Ref<JSON> p_json, float p_scale) {
	String data = p_json->get_parsed_text();
	if (data.is_empty()) {
		data = p_json->to_string();
	}
	Ref<LottieSheet> ret = memnew(LottieSheet);
	ret->_load_data(data, p_scale);
	ret->json = p_json;
	return ret;
}

Ref<LottieSheet> LottieSheet::load_string(String p_string, float p_scale) {
	Ref<LottieSheet> ret = memnew(LottieSheet);
	ret->_load_data(p_string, p_scale);
	ret->json->parse(p_string, true);
	return ret;
}

void LottieSheet::update_frame(float frame) {
	tvg::Result res = animation->frame(frame);
	if (res == tvg::Result::Success) {
		sw_canvas->update(picture);
	}

	res = sw_canvas->target(buffer, width, width, height, tvg::SwCanvas::ARGB8888S);
	if (res != tvg::Result::Success) {
		ERR_FAIL_MSG("LottieSheet: Couldn't set target on ThorVG canvas.");
	}

	res = sw_canvas->push(tvg::cast(picture));
	if (res != tvg::Result::Success) {
		ERR_FAIL_MSG("LottieSheet: Couldn't insert ThorVG picture on canvas.");
	}

	res = sw_canvas->draw();
	if (res != tvg::Result::Success) {
		ERR_FAIL_MSG("LottieSheet: Couldn't draw ThorVG pictures on canvas.");
	}

	res = sw_canvas->sync();
	if (res != tvg::Result::Success) {
		ERR_FAIL_MSG("LottieSheet: Couldn't sync ThorVG canvas.");
	}

	Vector<uint8_t> image_data;
	image_data.resize(width * height * sizeof(uint32_t));

	for (uint32_t y = 0; y < height; y++) {
		for (uint32_t x = 0; x < width; x++) {
			uint32_t n = buffer[y * width + x];
			const size_t offset = sizeof(uint32_t) * width * y + sizeof(uint32_t) * x;
			image_data.write[offset + 0] = (n >> 16) & 0xff;
			image_data.write[offset + 1] = (n >> 8) & 0xff;
			image_data.write[offset + 2] = n & 0xff;
			image_data.write[offset + 3] = (n >> 24) & 0xff;
		}
	}

	res = sw_canvas->clear(true);

	image->set_data(width, height, false, Image::FORMAT_RGBA8, image_data);
}

Ref<Image> LottieSheet::get_image() { return image; };

Ref<Image> LottieSheet::get_frame_image(float frame) {
	update_frame(frame);
	return image;
};

Vector2i LottieSheet::get_image_size() {
	return Vector2i(width, height);
}

float LottieSheet::get_total_frame() { return animation->totalFrame(); };

float LottieSheet::get_duration() { return animation->duration(); };

Ref<JSON> LottieSheet::get_json() { return json; }

void LottieSheet::set_json(Ref<JSON> p_json) {
	String data = p_json->get_parsed_text();
	if (data.is_empty()) {
		data = p_json->to_string();
	}
	_load_data(data, scale);
	json = p_json;
}

float LottieSheet::get_scale() { return scale; };
void LottieSheet::set_scale(float p_scale) {
	String data = json->get_parsed_text();
	if (data.is_empty()) {
		data = json->to_string();
	}
	_load_data(data, scale);
	scale = p_scale;
};

LottieSheet::~LottieSheet() { memfree(buffer); }

void LottieSheet::_bind_methods() {
	ClassDB::bind_static_method("LottieSheet", D_METHOD("load_string", "p_string", "p_scale"), &LottieSheet::load_string, DEFVAL(1));
	ClassDB::bind_static_method("LottieSheet", D_METHOD("load_json", "p_json", "p_scale"), &LottieSheet::load_json, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("get_json"), &LottieSheet::get_json);
	ClassDB::bind_method(D_METHOD("set_json", "p_json"), &LottieSheet::set_json);
	ClassDB::bind_method(D_METHOD("get_scale"), &LottieSheet::get_scale);
	ClassDB::bind_method(D_METHOD("set_scale", "p_scale"), &LottieSheet::set_scale);
	ClassDB::bind_method(D_METHOD("update_frame", "frame"), &LottieSheet::update_frame);
	ClassDB::bind_method(D_METHOD("get_image"), &LottieSheet::get_image);
	ClassDB::bind_method(D_METHOD("get_frame_image", "frame"), &LottieSheet::get_frame_image);
	ClassDB::bind_method(D_METHOD("get_image_size"), &LottieSheet::get_image_size);
	ClassDB::bind_method(D_METHOD("get_total_frame"), &LottieSheet::get_total_frame);
	ClassDB::bind_method(D_METHOD("get_duration"), &LottieSheet::get_duration);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "json"), "set_json", "get_json");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "scale"), "set_scale", "get_scale");
}
