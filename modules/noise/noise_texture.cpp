/*************************************************************************/
/*  noise_texture.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "noise_texture.h"

#include "core/core_string_names.h"
#include "noise.h"

NoiseTexture::NoiseTexture() {
	noise = Ref<Noise>();

	_queue_update();
}

NoiseTexture::~NoiseTexture() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
	noise_thread.wait_to_finish();
}

void NoiseTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_texture"), &NoiseTexture::_update_texture);
	ClassDB::bind_method(D_METHOD("_generate_texture"), &NoiseTexture::_generate_texture);
	ClassDB::bind_method(D_METHOD("_thread_done", "image"), &NoiseTexture::_thread_done);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &NoiseTexture::set_width);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &NoiseTexture::set_height);

	ClassDB::bind_method(D_METHOD("set_invert", "invert"), &NoiseTexture::set_invert);
	ClassDB::bind_method(D_METHOD("get_invert"), &NoiseTexture::get_invert);

	ClassDB::bind_method(D_METHOD("set_in_3d_space", "enable"), &NoiseTexture::set_in_3d_space);
	ClassDB::bind_method(D_METHOD("is_in_3d_space"), &NoiseTexture::is_in_3d_space);

	ClassDB::bind_method(D_METHOD("set_generate_mipmaps", "invert"), &NoiseTexture::set_generate_mipmaps);
	ClassDB::bind_method(D_METHOD("is_generating_mipmaps"), &NoiseTexture::is_generating_mipmaps);

	ClassDB::bind_method(D_METHOD("set_seamless", "seamless"), &NoiseTexture::set_seamless);
	ClassDB::bind_method(D_METHOD("get_seamless"), &NoiseTexture::get_seamless);

	ClassDB::bind_method(D_METHOD("set_seamless_blend_skirt", "seamless_blend_skirt"), &NoiseTexture::set_seamless_blend_skirt);
	ClassDB::bind_method(D_METHOD("get_seamless_blend_skirt"), &NoiseTexture::get_seamless_blend_skirt);

	ClassDB::bind_method(D_METHOD("set_as_normal_map", "as_normal_map"), &NoiseTexture::set_as_normal_map);
	ClassDB::bind_method(D_METHOD("is_normal_map"), &NoiseTexture::is_normal_map);

	ClassDB::bind_method(D_METHOD("set_bump_strength", "bump_strength"), &NoiseTexture::set_bump_strength);
	ClassDB::bind_method(D_METHOD("get_bump_strength"), &NoiseTexture::get_bump_strength);

	ClassDB::bind_method(D_METHOD("set_color_ramp", "gradient"), &NoiseTexture::set_color_ramp);
	ClassDB::bind_method(D_METHOD("get_color_ramp"), &NoiseTexture::get_color_ramp);

	ClassDB::bind_method(D_METHOD("set_noise", "noise"), &NoiseTexture::set_noise);
	ClassDB::bind_method(D_METHOD("get_noise"), &NoiseTexture::get_noise);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,2048,1,or_greater"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "height", PROPERTY_HINT_RANGE, "1,2048,1,or_greater"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert"), "set_invert", "get_invert");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "in_3d_space"), "set_in_3d_space", "is_in_3d_space");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "generate_mipmaps"), "set_generate_mipmaps", "is_generating_mipmaps");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "seamless"), "set_seamless", "get_seamless");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "seamless_blend_skirt", PROPERTY_HINT_RANGE, "0.05,1,0.001"), "set_seamless_blend_skirt", "get_seamless_blend_skirt");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "as_normal_map"), "set_as_normal_map", "is_normal_map");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bump_strength", PROPERTY_HINT_RANGE, "0,32,0.1,or_greater"), "set_bump_strength", "get_bump_strength");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "color_ramp", PROPERTY_HINT_RESOURCE_TYPE, "Gradient"), "set_color_ramp", "get_color_ramp");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "noise", PROPERTY_HINT_RESOURCE_TYPE, "Noise"), "set_noise", "get_noise");
}

void NoiseTexture::_validate_property(PropertyInfo &property) const {
	if (property.name == "bump_strength") {
		if (!as_normal_map) {
			property.usage = PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}

	if (property.name == "seamless_blend_skirt") {
		if (!seamless) {
			property.usage = PROPERTY_USAGE_NO_EDITOR;
		}
	}
}

void NoiseTexture::_set_texture_image(const Ref<Image> &p_image) {
	image = p_image;
	if (image.is_valid()) {
		if (texture.is_valid()) {
			RID new_texture = RS::get_singleton()->texture_2d_create(p_image);
			RS::get_singleton()->texture_replace(texture, new_texture);
		} else {
			texture = RS::get_singleton()->texture_2d_create(p_image);
		}
	}
	emit_changed();
}

void NoiseTexture::_thread_done(const Ref<Image> &p_image) {
	_set_texture_image(p_image);
	noise_thread.wait_to_finish();
	if (regen_queued) {
		noise_thread.start(_thread_function, this);
		regen_queued = false;
	}
}

void NoiseTexture::_thread_function(void *p_ud) {
	NoiseTexture *tex = static_cast<NoiseTexture *>(p_ud);
	tex->call_deferred(SNAME("_thread_done"), tex->_generate_texture());
}

void NoiseTexture::_queue_update() {
	if (update_queued) {
		return;
	}

	update_queued = true;
	call_deferred(SNAME("_update_texture"));
}

Ref<Image> NoiseTexture::_generate_texture() {
	// Prevent memdelete due to unref() on other thread.
	Ref<Noise> ref_noise = noise;

	if (ref_noise.is_null()) {
		return Ref<Image>();
	}

	Ref<Image> image;

	if (seamless) {
		image = ref_noise->get_seamless_image(size.x, size.y, invert, in_3d_space, seamless_blend_skirt);
	} else {
		image = ref_noise->get_image(size.x, size.y, invert, in_3d_space);
	}
	if (color_ramp.is_valid()) {
		image = _modulate_with_gradient(image, color_ramp);
	}
	if (as_normal_map) {
		image->bump_map_to_normal_map(bump_strength);
	}
	if (generate_mipmaps) {
		image->generate_mipmaps();
	}

	return image;
}

Ref<Image> NoiseTexture::_modulate_with_gradient(Ref<Image> p_image, Ref<Gradient> p_gradient) {
	int width = p_image->get_width();
	int height = p_image->get_height();

	Ref<Image> new_image;
	new_image.instantiate();
	new_image->create(width, height, false, Image::FORMAT_RGBA8);

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			Color pixel_color = p_image->get_pixel(col, row);
			Color ramp_color = color_ramp->get_color_at_offset(pixel_color.get_luminance());
			new_image->set_pixel(col, row, ramp_color);
		}
	}

	return new_image;
}

void NoiseTexture::_update_texture() {
	bool use_thread = true;
	if (first_time) {
		use_thread = false;
		first_time = false;
	}
#ifdef NO_THREADS
	use_thread = false;
#endif
	if (use_thread) {
		if (!noise_thread.is_started()) {
			noise_thread.start(_thread_function, this);
			regen_queued = false;
		} else {
			regen_queued = true;
		}

	} else {
		Ref<Image> image = _generate_texture();
		_set_texture_image(image);
	}
	update_queued = false;
}

void NoiseTexture::set_noise(Ref<Noise> p_noise) {
	if (p_noise == noise) {
		return;
	}
	if (noise.is_valid()) {
		noise->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &NoiseTexture::_queue_update));
	}
	noise = p_noise;
	if (noise.is_valid()) {
		noise->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &NoiseTexture::_queue_update));
	}
	_queue_update();
}

Ref<Noise> NoiseTexture::get_noise() {
	return noise;
}

void NoiseTexture::set_width(int p_width) {
	ERR_FAIL_COND(p_width <= 0);
	if (p_width == size.x) {
		return;
	}
	size.x = p_width;
	_queue_update();
}

void NoiseTexture::set_height(int p_height) {
	ERR_FAIL_COND(p_height <= 0);
	if (p_height == size.y) {
		return;
	}
	size.y = p_height;
	_queue_update();
}

void NoiseTexture::set_invert(bool p_invert) {
	if (p_invert == invert) {
		return;
	}
	invert = p_invert;
	_queue_update();
}

bool NoiseTexture::get_invert() const {
	return invert;
}

void NoiseTexture::set_in_3d_space(bool p_enable) {
	if (p_enable == in_3d_space) {
		return;
	}
	in_3d_space = p_enable;
	_queue_update();
}
bool NoiseTexture::is_in_3d_space() const {
	return in_3d_space;
}

void NoiseTexture::set_generate_mipmaps(bool p_enable) {
	if (p_enable == generate_mipmaps) {
		return;
	}
	generate_mipmaps = p_enable;
	_queue_update();
}

bool NoiseTexture::is_generating_mipmaps() const {
	return generate_mipmaps;
}

void NoiseTexture::set_seamless(bool p_seamless) {
	if (p_seamless == seamless) {
		return;
	}
	seamless = p_seamless;
	_queue_update();
	notify_property_list_changed();
}

bool NoiseTexture::get_seamless() {
	return seamless;
}

void NoiseTexture::set_seamless_blend_skirt(real_t p_blend_skirt) {
	if (p_blend_skirt == seamless_blend_skirt) {
		return;
	}
	seamless_blend_skirt = p_blend_skirt;
	_queue_update();
}
real_t NoiseTexture::get_seamless_blend_skirt() {
	return seamless_blend_skirt;
}

void NoiseTexture::set_as_normal_map(bool p_as_normal_map) {
	if (p_as_normal_map == as_normal_map) {
		return;
	}
	as_normal_map = p_as_normal_map;
	_queue_update();
	notify_property_list_changed();
}

bool NoiseTexture::is_normal_map() {
	return as_normal_map;
}

void NoiseTexture::set_bump_strength(float p_bump_strength) {
	if (p_bump_strength == bump_strength) {
		return;
	}
	bump_strength = p_bump_strength;
	if (as_normal_map) {
		_queue_update();
	}
}

float NoiseTexture::get_bump_strength() {
	return bump_strength;
}

void NoiseTexture::set_color_ramp(const Ref<Gradient> &p_gradient) {
	if (p_gradient == color_ramp) {
		return;
	}
	if (color_ramp.is_valid()) {
		color_ramp->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &NoiseTexture::_queue_update));
	}
	color_ramp = p_gradient;
	if (color_ramp.is_valid()) {
		color_ramp->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &NoiseTexture::_queue_update));
	}
	_queue_update();
}

Ref<Gradient> NoiseTexture::get_color_ramp() const {
	return color_ramp;
}

int NoiseTexture::get_width() const {
	return size.x;
}

int NoiseTexture::get_height() const {
	return size.y;
}

RID NoiseTexture::get_rid() const {
	if (!texture.is_valid()) {
		texture = RS::get_singleton()->texture_2d_placeholder_create();
	}

	return texture;
}

Ref<Image> NoiseTexture::get_image() const {
	return image;
}
