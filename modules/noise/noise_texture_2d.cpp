/**************************************************************************/
/*  noise_texture_2d.cpp                                                  */
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

#include "noise_texture_2d.h"

#include "noise.h"

NoiseTexture2D::NoiseTexture2D() {
	noise = Ref<Noise>();

	_queue_update();
}

NoiseTexture2D::~NoiseTexture2D() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
	if (noise_thread.is_started()) {
		noise_thread.wait_to_finish();
	}
}

void NoiseTexture2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_width", "width"), &NoiseTexture2D::set_width);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &NoiseTexture2D::set_height);

	ClassDB::bind_method(D_METHOD("set_invert", "invert"), &NoiseTexture2D::set_invert);
	ClassDB::bind_method(D_METHOD("get_invert"), &NoiseTexture2D::get_invert);

	ClassDB::bind_method(D_METHOD("set_in_3d_space", "enable"), &NoiseTexture2D::set_in_3d_space);
	ClassDB::bind_method(D_METHOD("is_in_3d_space"), &NoiseTexture2D::is_in_3d_space);

	ClassDB::bind_method(D_METHOD("set_generate_mipmaps", "invert"), &NoiseTexture2D::set_generate_mipmaps);
	ClassDB::bind_method(D_METHOD("is_generating_mipmaps"), &NoiseTexture2D::is_generating_mipmaps);

	ClassDB::bind_method(D_METHOD("set_seamless", "seamless"), &NoiseTexture2D::set_seamless);
	ClassDB::bind_method(D_METHOD("get_seamless"), &NoiseTexture2D::get_seamless);

	ClassDB::bind_method(D_METHOD("set_seamless_blend_skirt", "seamless_blend_skirt"), &NoiseTexture2D::set_seamless_blend_skirt);
	ClassDB::bind_method(D_METHOD("get_seamless_blend_skirt"), &NoiseTexture2D::get_seamless_blend_skirt);

	ClassDB::bind_method(D_METHOD("set_as_normal_map", "as_normal_map"), &NoiseTexture2D::set_as_normal_map);
	ClassDB::bind_method(D_METHOD("is_normal_map"), &NoiseTexture2D::is_normal_map);

	ClassDB::bind_method(D_METHOD("set_bump_strength", "bump_strength"), &NoiseTexture2D::set_bump_strength);
	ClassDB::bind_method(D_METHOD("get_bump_strength"), &NoiseTexture2D::get_bump_strength);

	ClassDB::bind_method(D_METHOD("set_normalize", "normalize"), &NoiseTexture2D::set_normalize);
	ClassDB::bind_method(D_METHOD("is_normalized"), &NoiseTexture2D::is_normalized);

	ClassDB::bind_method(D_METHOD("set_color_ramp", "gradient"), &NoiseTexture2D::set_color_ramp);
	ClassDB::bind_method(D_METHOD("get_color_ramp"), &NoiseTexture2D::get_color_ramp);

	ClassDB::bind_method(D_METHOD("set_noise", "noise"), &NoiseTexture2D::set_noise);
	ClassDB::bind_method(D_METHOD("get_noise"), &NoiseTexture2D::get_noise);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,2048,1,or_greater,suffix:px"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "height", PROPERTY_HINT_RANGE, "1,2048,1,or_greater,suffix:px"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert"), "set_invert", "get_invert");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "in_3d_space"), "set_in_3d_space", "is_in_3d_space");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "generate_mipmaps"), "set_generate_mipmaps", "is_generating_mipmaps");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "seamless"), "set_seamless", "get_seamless");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "seamless_blend_skirt", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_seamless_blend_skirt", "get_seamless_blend_skirt");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "as_normal_map"), "set_as_normal_map", "is_normal_map");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bump_strength", PROPERTY_HINT_RANGE, "0,32,0.1,or_greater"), "set_bump_strength", "get_bump_strength");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "normalize"), "set_normalize", "is_normalized");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "color_ramp", PROPERTY_HINT_RESOURCE_TYPE, "Gradient"), "set_color_ramp", "get_color_ramp");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "noise", PROPERTY_HINT_RESOURCE_TYPE, "Noise"), "set_noise", "get_noise");
}

void NoiseTexture2D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "bump_strength") {
		if (!as_normal_map) {
			p_property.usage = PROPERTY_USAGE_NO_EDITOR;
		}
	}

	if (p_property.name == "seamless_blend_skirt") {
		if (!seamless) {
			p_property.usage = PROPERTY_USAGE_NO_EDITOR;
		}
	}
}

void NoiseTexture2D::_set_texture_image(const Ref<Image> &p_image) {
	image = p_image;
	if (image.is_valid()) {
		if (texture.is_valid()) {
			RID new_texture = RS::get_singleton()->texture_2d_create(p_image);
			RS::get_singleton()->texture_replace(texture, new_texture);
		} else {
			texture = RS::get_singleton()->texture_2d_create(p_image);
		}
		RS::get_singleton()->texture_set_path(texture, get_path());
	}
	emit_changed();
}

void NoiseTexture2D::_thread_done(const Ref<Image> &p_image) {
	_set_texture_image(p_image);
	noise_thread.wait_to_finish();
	if (regen_queued) {
		noise_thread.start(_thread_function, this);
		regen_queued = false;
	}
}

void NoiseTexture2D::_thread_function(void *p_ud) {
	NoiseTexture2D *tex = static_cast<NoiseTexture2D *>(p_ud);
	callable_mp(tex, &NoiseTexture2D::_thread_done).call_deferred(tex->_generate_texture());
}

void NoiseTexture2D::_queue_update() {
	if (update_queued) {
		return;
	}

	update_queued = true;
	callable_mp(this, &NoiseTexture2D::_update_texture).call_deferred();
}

Ref<Image> NoiseTexture2D::_generate_texture() {
	// Prevent memdelete due to unref() on other thread.
	Ref<Noise> ref_noise = noise;

	if (ref_noise.is_null()) {
		return Ref<Image>();
	}

	Ref<Image> new_image;

	if (seamless) {
		new_image = ref_noise->get_seamless_image(size.x, size.y, invert, in_3d_space, seamless_blend_skirt, normalize);
	} else {
		new_image = ref_noise->get_image(size.x, size.y, invert, in_3d_space, normalize);
	}
	if (color_ramp.is_valid()) {
		new_image = _modulate_with_gradient(new_image, color_ramp);
	}
	if (as_normal_map) {
		new_image->bump_map_to_normal_map(bump_strength);
	}
	if (generate_mipmaps) {
		new_image->generate_mipmaps();
	}

	return new_image;
}

Ref<Image> NoiseTexture2D::_modulate_with_gradient(Ref<Image> p_image, Ref<Gradient> p_gradient) {
	int width = p_image->get_width();
	int height = p_image->get_height();

	Ref<Image> new_image = Image::create_empty(width, height, false, Image::FORMAT_RGBA8);

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			Color pixel_color = p_image->get_pixel(col, row);
			Color ramp_color = p_gradient->get_color_at_offset(pixel_color.get_luminance());
			new_image->set_pixel(col, row, ramp_color);
		}
	}

	return new_image;
}

void NoiseTexture2D::_update_texture() {
	bool use_thread = true;
	if (first_time) {
		use_thread = false;
		first_time = false;
	}
	if (use_thread) {
		if (!noise_thread.is_started()) {
			noise_thread.start(_thread_function, this);
			regen_queued = false;
		} else {
			regen_queued = true;
		}

	} else {
		Ref<Image> new_image = _generate_texture();
		_set_texture_image(new_image);
	}
	update_queued = false;
}

void NoiseTexture2D::set_noise(Ref<Noise> p_noise) {
	if (p_noise == noise) {
		return;
	}
	if (noise.is_valid()) {
		noise->disconnect_changed(callable_mp(this, &NoiseTexture2D::_queue_update));
	}
	noise = p_noise;
	if (noise.is_valid()) {
		noise->connect_changed(callable_mp(this, &NoiseTexture2D::_queue_update));
	}
	_queue_update();
}

Ref<Noise> NoiseTexture2D::get_noise() {
	return noise;
}

void NoiseTexture2D::set_width(int p_width) {
	ERR_FAIL_COND(p_width <= 0);
	if (p_width == size.x) {
		return;
	}
	size.x = p_width;
	_queue_update();
}

void NoiseTexture2D::set_height(int p_height) {
	ERR_FAIL_COND(p_height <= 0);
	if (p_height == size.y) {
		return;
	}
	size.y = p_height;
	_queue_update();
}

void NoiseTexture2D::set_invert(bool p_invert) {
	if (p_invert == invert) {
		return;
	}
	invert = p_invert;
	_queue_update();
}

bool NoiseTexture2D::get_invert() const {
	return invert;
}

void NoiseTexture2D::set_in_3d_space(bool p_enable) {
	if (p_enable == in_3d_space) {
		return;
	}
	in_3d_space = p_enable;
	_queue_update();
}
bool NoiseTexture2D::is_in_3d_space() const {
	return in_3d_space;
}

void NoiseTexture2D::set_generate_mipmaps(bool p_enable) {
	if (p_enable == generate_mipmaps) {
		return;
	}
	generate_mipmaps = p_enable;
	_queue_update();
}

bool NoiseTexture2D::is_generating_mipmaps() const {
	return generate_mipmaps;
}

void NoiseTexture2D::set_seamless(bool p_seamless) {
	if (p_seamless == seamless) {
		return;
	}
	seamless = p_seamless;
	_queue_update();
	notify_property_list_changed();
}

bool NoiseTexture2D::get_seamless() {
	return seamless;
}

void NoiseTexture2D::set_seamless_blend_skirt(real_t p_blend_skirt) {
	ERR_FAIL_COND(p_blend_skirt < 0 || p_blend_skirt > 1);

	if (p_blend_skirt == seamless_blend_skirt) {
		return;
	}
	seamless_blend_skirt = p_blend_skirt;
	_queue_update();
}
real_t NoiseTexture2D::get_seamless_blend_skirt() {
	return seamless_blend_skirt;
}

void NoiseTexture2D::set_as_normal_map(bool p_as_normal_map) {
	if (p_as_normal_map == as_normal_map) {
		return;
	}
	as_normal_map = p_as_normal_map;
	_queue_update();
	notify_property_list_changed();
}

bool NoiseTexture2D::is_normal_map() {
	return as_normal_map;
}

void NoiseTexture2D::set_bump_strength(float p_bump_strength) {
	if (p_bump_strength == bump_strength) {
		return;
	}
	bump_strength = p_bump_strength;
	if (as_normal_map) {
		_queue_update();
	}
}

float NoiseTexture2D::get_bump_strength() {
	return bump_strength;
}

void NoiseTexture2D::set_color_ramp(const Ref<Gradient> &p_gradient) {
	if (p_gradient == color_ramp) {
		return;
	}
	if (color_ramp.is_valid()) {
		color_ramp->disconnect_changed(callable_mp(this, &NoiseTexture2D::_queue_update));
	}
	color_ramp = p_gradient;
	if (color_ramp.is_valid()) {
		color_ramp->connect_changed(callable_mp(this, &NoiseTexture2D::_queue_update));
	}
	_queue_update();
}

void NoiseTexture2D::set_normalize(bool p_normalize) {
	if (normalize == p_normalize) {
		return;
	}
	normalize = p_normalize;
	_queue_update();
}

bool NoiseTexture2D::is_normalized() const {
	return normalize;
}

Ref<Gradient> NoiseTexture2D::get_color_ramp() const {
	return color_ramp;
}

int NoiseTexture2D::get_width() const {
	return size.x;
}

int NoiseTexture2D::get_height() const {
	return size.y;
}

RID NoiseTexture2D::get_rid() const {
	if (!texture.is_valid()) {
		texture = RS::get_singleton()->texture_2d_placeholder_create();
	}

	return texture;
}

Ref<Image> NoiseTexture2D::get_image() const {
	return image;
}
