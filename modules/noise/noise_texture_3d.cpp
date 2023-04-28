/**************************************************************************/
/*  noise_texture_3d.cpp                                                  */
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

#include "noise_texture_3d.h"

#include "core/core_string_names.h"
#include "noise.h"

NoiseTexture3D::NoiseTexture3D() {
	noise = Ref<Noise>();

	_queue_update();
}

NoiseTexture3D::~NoiseTexture3D() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
	noise_thread.wait_to_finish();
}

void NoiseTexture3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_texture"), &NoiseTexture3D::_update_texture);
	ClassDB::bind_method(D_METHOD("_generate_texture"), &NoiseTexture3D::_generate_texture);
	ClassDB::bind_method(D_METHOD("_thread_done", "image"), &NoiseTexture3D::_thread_done);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &NoiseTexture3D::set_width);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &NoiseTexture3D::set_height);
	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &NoiseTexture3D::set_depth);

	ClassDB::bind_method(D_METHOD("set_invert", "invert"), &NoiseTexture3D::set_invert);
	ClassDB::bind_method(D_METHOD("get_invert"), &NoiseTexture3D::get_invert);

	ClassDB::bind_method(D_METHOD("set_seamless", "seamless"), &NoiseTexture3D::set_seamless);
	ClassDB::bind_method(D_METHOD("get_seamless"), &NoiseTexture3D::get_seamless);

	ClassDB::bind_method(D_METHOD("set_seamless_blend_skirt", "seamless_blend_skirt"), &NoiseTexture3D::set_seamless_blend_skirt);
	ClassDB::bind_method(D_METHOD("get_seamless_blend_skirt"), &NoiseTexture3D::get_seamless_blend_skirt);

	ClassDB::bind_method(D_METHOD("set_normalize", "normalize"), &NoiseTexture3D::set_normalize);
	ClassDB::bind_method(D_METHOD("is_normalized"), &NoiseTexture3D::is_normalized);

	ClassDB::bind_method(D_METHOD("set_color_ramp", "gradient"), &NoiseTexture3D::set_color_ramp);
	ClassDB::bind_method(D_METHOD("get_color_ramp"), &NoiseTexture3D::get_color_ramp);

	ClassDB::bind_method(D_METHOD("set_noise", "noise"), &NoiseTexture3D::set_noise);
	ClassDB::bind_method(D_METHOD("get_noise"), &NoiseTexture3D::get_noise);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,2048,1,or_greater,suffix:px"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "height", PROPERTY_HINT_RANGE, "1,2048,1,or_greater,suffix:px"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "depth", PROPERTY_HINT_RANGE, "1,2048,1,or_greater,suffix:px"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert"), "set_invert", "get_invert");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "seamless"), "set_seamless", "get_seamless");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "seamless_blend_skirt", PROPERTY_HINT_RANGE, "0.05,1,0.001"), "set_seamless_blend_skirt", "get_seamless_blend_skirt");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "normalize"), "set_normalize", "is_normalized");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "color_ramp", PROPERTY_HINT_RESOURCE_TYPE, "Gradient"), "set_color_ramp", "get_color_ramp");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "noise", PROPERTY_HINT_RESOURCE_TYPE, "Noise"), "set_noise", "get_noise");
}

void NoiseTexture3D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "seamless_blend_skirt") {
		if (!seamless) {
			p_property.usage = PROPERTY_USAGE_NO_EDITOR;
		}
	}
}

void NoiseTexture3D::_set_texture_data(const TypedArray<Image> &p_data) {
	if (!p_data.is_empty()) {
		Vector<Ref<Image>> data;

		data.resize(p_data.size());

		for (int i = 0; i < data.size(); i++) {
			data.write[i] = p_data[i];
		}

		if (texture.is_valid()) {
			RID new_texture = RS::get_singleton()->texture_3d_create(data[0]->get_format(), data[0]->get_width(), data[0]->get_height(), data.size(), false, data);
			RS::get_singleton()->texture_replace(texture, new_texture);
		} else {
			texture = RS::get_singleton()->texture_3d_create(data[0]->get_format(), data[0]->get_width(), data[0]->get_height(), data.size(), false, data);
		}
	}
	emit_changed();
}

void NoiseTexture3D::_thread_done(const TypedArray<Image> &p_data) {
	_set_texture_data(p_data);
	noise_thread.wait_to_finish();
	if (regen_queued) {
		noise_thread.start(_thread_function, this);
		regen_queued = false;
	}
}

void NoiseTexture3D::_thread_function(void *p_ud) {
	NoiseTexture3D *tex = static_cast<NoiseTexture3D *>(p_ud);
	tex->call_deferred(SNAME("_thread_done"), tex->_generate_texture());
}

void NoiseTexture3D::_queue_update() {
	if (update_queued) {
		return;
	}

	update_queued = true;
	call_deferred(SNAME("_update_texture"));
}

TypedArray<Image> NoiseTexture3D::_generate_texture() {
	// Prevent memdelete due to unref() on other thread.
	Ref<Noise> ref_noise = noise;

	if (ref_noise.is_null()) {
		return TypedArray<Image>();
	}

	Vector<Ref<Image>> images;

	if (seamless) {
		images = _get_seamless(width, height, depth, invert, seamless_blend_skirt);
	} else {
		images.resize(depth);

		for (int i = 0; i < images.size(); i++) {
			images.write[i] = ref_noise->get_image(width, height, i, invert, true, false);
		}
	}

	// Normalize on whole texture at once rather than on each image individually as it would result in visible artifacts on z (depth) axis.
	if (normalize) {
		images = _normalize(images);
	}

	if (color_ramp.is_valid()) {
		for (int i = 0; i < images.size(); i++) {
			images.write[i] = _modulate_with_gradient(images[i], color_ramp);
		}
	}

	TypedArray<Image> new_data;
	new_data.resize(images.size());

	for (int i = 0; i < new_data.size(); i++) {
		new_data[i] = images[i];
	}

	return new_data;
}

Vector<Ref<Image>> NoiseTexture3D::_get_seamless(int p_width, int p_height, int p_depth, bool p_invert, real_t p_blend_skirt) {
	// Prevent memdelete due to unref() on other thread.
	Ref<Noise> ref_noise = noise;

	if (ref_noise.is_null()) {
		return Vector<Ref<Image>>();
	}

	int skirt_depth = MAX(1, p_depth * p_blend_skirt);
	int src_depth = p_depth + skirt_depth;

	Vector<Ref<Image>> images;
	images.resize(src_depth);

	for (int i = 0; i < src_depth; i++) {
		images.write[i] = ref_noise->get_seamless_image(p_width, p_height, i, p_invert, true, p_blend_skirt, false);
	}

	int half_depth = p_depth * 0.5;
	int skirt_edge_z = half_depth + skirt_depth;

	// swap halves on depth.
	for (int i = 0; i < half_depth; i++) {
		Ref<Image> img = images[i];
		images.write[i] = images[i + half_depth];
		images.write[i + half_depth] = img;
	}

	Vector<Ref<Image>> new_images = images;
	new_images.resize(p_depth);

	// scale seamless generation to third dimension.
	for (int z = half_depth; z < skirt_edge_z; z++) {
		int alpha = 255 * (1 - Math::smoothstep(0.1f, 0.9f, float(z - half_depth) / float(skirt_depth)));

		Vector<uint8_t> img = images[z % p_depth]->get_data();
		Vector<uint8_t> skirt = images[(z - half_depth) + p_depth]->get_data();

		Vector<uint8_t> dest;
		dest.resize(images[0]->get_width() * images[0]->get_height() * Image::get_format_pixel_size(images[0]->get_format()));

		for (int i = 0; i < img.size(); i++) {
			uint8_t fg, bg, out;

			fg = skirt[i];
			bg = img[i];

			uint16_t a;
			uint16_t inv_a;

			a = alpha + 1;
			inv_a = 256 - alpha;

			out = (uint8_t)((a * fg + inv_a * bg) >> 8);

			dest.write[i] = out;
		}

		Ref<Image> new_image = memnew(Image(images[0]->get_width(), images[0]->get_height(), false, images[0]->get_format(), dest));
		new_images.write[z % p_depth] = new_image;
	}

	return new_images;
}

Vector<Ref<Image>> NoiseTexture3D::_normalize(Vector<Ref<Image>> p_images) {
	real_t min_val = FLT_MAX;
	real_t max_val = -FLT_MAX;

	int w = p_images[0]->get_width();
	int h = p_images[0]->get_height();

	for (int i = 0; i < p_images.size(); i++) {
		Vector<uint8_t> data = p_images[i]->get_data();

		for (int j = 0; j < data.size(); j++) {
			if (data[j] > max_val) {
				max_val = data[j];
			}
			if (data[j] < min_val) {
				min_val = data[j];
			}
		}
	}

	Vector<Ref<Image>> new_images;
	new_images.resize(p_images.size());

	for (int i = 0; i < p_images.size(); i++) {
		Vector<uint8_t> data = p_images[i]->get_data();

		for (int j = 0; j < data.size(); j++) {
			uint8_t value;

			if (max_val == min_val) {
				value = 0;
			} else {
				value = static_cast<uint8_t>(CLAMP((data[j] - min_val) / (max_val - min_val) * 255.f, 0, 255));
			}

			data.write[j] = value;
		}

		Ref<Image> new_image = memnew(Image(w, h, false, Image::FORMAT_L8, data));
		new_images.write[i] = new_image;
	}

	return new_images;
}

Ref<Image> NoiseTexture3D::_modulate_with_gradient(Ref<Image> p_image, Ref<Gradient> p_gradient) {
	int w = p_image->get_width();
	int h = p_image->get_height();

	Ref<Image> new_image = Image::create_empty(w, h, false, Image::FORMAT_RGBA8);

	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			Color pixel_color = p_image->get_pixel(col, row);
			Color ramp_color = p_gradient->get_color_at_offset(pixel_color.get_luminance());
			new_image->set_pixel(col, row, ramp_color);
		}
	}

	return new_image;
}

void NoiseTexture3D::_update_texture() {
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
		TypedArray<Image> new_data = _generate_texture();
		_set_texture_data(new_data);
	}
	update_queued = false;
}

void NoiseTexture3D::set_noise(Ref<Noise> p_noise) {
	if (p_noise == noise) {
		return;
	}
	if (noise.is_valid()) {
		noise->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &NoiseTexture3D::_queue_update));
	}
	noise = p_noise;
	if (noise.is_valid()) {
		noise->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &NoiseTexture3D::_queue_update));
	}
	_queue_update();
}

Ref<Noise> NoiseTexture3D::get_noise() {
	return noise;
}

void NoiseTexture3D::set_width(int p_width) {
	ERR_FAIL_COND(p_width <= 0);
	if (p_width == width) {
		return;
	}
	width = p_width;
	_queue_update();
}

void NoiseTexture3D::set_height(int p_height) {
	ERR_FAIL_COND(p_height <= 0);
	if (p_height == height) {
		return;
	}
	height = p_height;
	_queue_update();
}

void NoiseTexture3D::set_depth(int p_depth) {
	ERR_FAIL_COND(p_depth <= 0);
	if (p_depth == depth) {
		return;
	}
	depth = p_depth;
	_queue_update();
}

void NoiseTexture3D::set_invert(bool p_invert) {
	if (p_invert == invert) {
		return;
	}
	invert = p_invert;
	_queue_update();
}

bool NoiseTexture3D::get_invert() const {
	return invert;
}

void NoiseTexture3D::set_seamless(bool p_seamless) {
	if (p_seamless == seamless) {
		return;
	}
	seamless = p_seamless;
	_queue_update();
	notify_property_list_changed();
}

bool NoiseTexture3D::get_seamless() {
	return seamless;
}

void NoiseTexture3D::set_seamless_blend_skirt(real_t p_blend_skirt) {
	ERR_FAIL_COND(p_blend_skirt < 0.05 || p_blend_skirt > 1);

	if (p_blend_skirt == seamless_blend_skirt) {
		return;
	}
	seamless_blend_skirt = p_blend_skirt;
	_queue_update();
}
real_t NoiseTexture3D::get_seamless_blend_skirt() {
	return seamless_blend_skirt;
}

void NoiseTexture3D::set_color_ramp(const Ref<Gradient> &p_gradient) {
	if (p_gradient == color_ramp) {
		return;
	}
	if (color_ramp.is_valid()) {
		color_ramp->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &NoiseTexture3D::_queue_update));
	}
	color_ramp = p_gradient;
	if (color_ramp.is_valid()) {
		color_ramp->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &NoiseTexture3D::_queue_update));
	}
	_queue_update();
}

void NoiseTexture3D::set_normalize(bool p_normalize) {
	if (normalize == p_normalize) {
		return;
	}
	normalize = p_normalize;
	_queue_update();
}

bool NoiseTexture3D::is_normalized() const {
	return normalize;
}

Ref<Gradient> NoiseTexture3D::get_color_ramp() const {
	return color_ramp;
}

int NoiseTexture3D::get_width() const {
	return width;
}

int NoiseTexture3D::get_height() const {
	return height;
}

int NoiseTexture3D::get_depth() const {
	return depth;
}

RID NoiseTexture3D::get_rid() const {
	if (!texture.is_valid()) {
		texture = RS::get_singleton()->texture_3d_placeholder_create();
	}

	return texture;
}

Vector<Ref<Image>> NoiseTexture3D::get_data() const {
	ERR_FAIL_COND_V(!texture.is_valid(), Vector<Ref<Image>>());
	return RS::get_singleton()->texture_3d_get(texture);
}

Image::Format NoiseTexture3D::get_format() const {
	ERR_FAIL_COND_V(!texture.is_valid(), Image::FORMAT_L8);
	return RS::get_singleton()->texture_3d_get(texture)[0]->get_format();
}
