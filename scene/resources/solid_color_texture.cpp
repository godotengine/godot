/**************************************************************************/
/*  solid_color_texture.cpp                                               */
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

#include "solid_color_texture.h"

void SolidColorTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_color", "color"), &SolidColorTexture::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &SolidColorTexture::get_color);
	ClassDB::bind_method(D_METHOD("set_size", "size"), &SolidColorTexture::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &SolidColorTexture::get_size);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "size", PROPERTY_HINT_NONE, "suffix:px"), "set_size", "get_size");
}

void SolidColorTexture::set_color(const Color &p_color) {
	if (color == p_color) {
		return;
	}
	color = p_color;
	_queue_update();
	emit_changed();
}

void SolidColorTexture::set_size(const Size2 &p_size) {
	ERR_FAIL_COND(!std::isfinite(p_size.x) || !std::isfinite(p_size.y));
	ERR_FAIL_COND(p_size.x <= 0 || p_size.y <= 0);
	if (size == p_size) {
		return;
	}
	size = p_size;
	_queue_update();
	emit_changed();
}

void SolidColorTexture::_queue_update() const {
	if (update_pending) {
		return;
	}
	update_pending = true;
	_update();
}

void SolidColorTexture::_update() const {
	Ref<Image> image = memnew(Image(size.x, size.y, false, Image::FORMAT_RGBA8));
	image->fill(color);

	if (texture.is_valid()) {
		RID new_texture = RS::get_singleton()->texture_2d_create(image);
		RS::get_singleton()->texture_replace(texture, new_texture);
	} else {
		texture = RS::get_singleton()->texture_2d_create(image);
	}
	RS::get_singleton()->texture_set_path(texture, get_path());

	update_pending = false;
}

Size2 SolidColorTexture::get_size() const {
	return size;
}

Color SolidColorTexture::get_color() const {
	return color;
}

RID SolidColorTexture::get_rid() const {
	if (!texture.is_valid()) {
		_update();
	}
	return texture;
}

Ref<Image> SolidColorTexture::get_image() const {
	if (!texture.is_valid()) {
		_update();
	}
	return RS::get_singleton()->texture_2d_get(texture);
}

SolidColorTexture::SolidColorTexture() {
	_queue_update();
}

SolidColorTexture::~SolidColorTexture() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
}
