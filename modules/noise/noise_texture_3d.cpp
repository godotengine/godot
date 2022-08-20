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

#include "noise_texture_3d.h"

#include "core/core_string_names.h"
#include "noise.h"

NoiseTexture3D::NoiseTexture3D() {
	noise = Ref<Noise>();

	_queue_update();
}

NoiseTexture3D::~NoiseTexture3D() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
	noise_thread.wait_to_finish();
}

void NoiseTexture3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_texture"), &NoiseTexture3D::_update_texture);
	ClassDB::bind_method(D_METHOD("_generate_texture"), &NoiseTexture3D::_generate_texture);
	ClassDB::bind_method(D_METHOD("_thread_done", "data"), &NoiseTexture3D::_thread_done);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &NoiseTexture3D::set_width);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &NoiseTexture3D::set_height);
	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &NoiseTexture3D::set_depth);

	ClassDB::bind_method(D_METHOD("set_invert", "invert"), &NoiseTexture3D::set_invert);
	ClassDB::bind_method(D_METHOD("get_invert"), &NoiseTexture3D::get_invert);

	ClassDB::bind_method(D_METHOD("set_generate_mipmaps", "invert"), &NoiseTexture3D::set_generate_mipmaps);
	ClassDB::bind_method(D_METHOD("is_generating_mipmaps"), &NoiseTexture3D::is_generating_mipmaps);

	ClassDB::bind_method(D_METHOD("set_color_ramp", "gradient"), &NoiseTexture3D::set_color_ramp);
	ClassDB::bind_method(D_METHOD("get_color_ramp"), &NoiseTexture3D::get_color_ramp);

	ClassDB::bind_method(D_METHOD("set_noise", "noise"), &NoiseTexture3D::set_noise);
	ClassDB::bind_method(D_METHOD("get_noise"), &NoiseTexture3D::get_noise);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,2048,1,or_greater,suffix:px"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "height", PROPERTY_HINT_RANGE, "1,2048,1,or_greater,suffix:px"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "depth", PROPERTY_HINT_RANGE, "1,2048,1,or_greater,suffix:px"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert"), "set_invert", "get_invert");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "generate_mipmaps"), "set_generate_mipmaps", "is_generating_mipmaps");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "color_ramp", PROPERTY_HINT_RESOURCE_TYPE, "Gradient"), "set_color_ramp", "get_color_ramp");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "noise", PROPERTY_HINT_RESOURCE_TYPE, "Noise"), "set_noise", "get_noise");
}

void NoiseTexture3D::_set_texture_data(const TypedArray<Image> &p_data) {
	Vector<Ref<Image>> images;
	images.resize(p_data.size());
	for (int i = 0; i < images.size(); i++) {
		images.write[i] = p_data[i];
	}

	if (images.size() >= 1) {
		if (texture.is_valid()) {
			RID new_texture = RS::get_singleton()->texture_3d_create(Image::Format::FORMAT_RGBA8, size.x, size.y, size.z, generate_mipmaps, images);
			RS::get_singleton()->texture_replace(texture, new_texture);
		} else {
			texture = RS::get_singleton()->texture_3d_create(Image::Format::FORMAT_RGBA8, size.x, size.y, size.z, generate_mipmaps, images);
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

	Vector<Ref<Image>> data;

	data.resize(get_depth());

	for (int i = 0; i < size.z; i++) {
		data.write[i] = ref_noise->get_image(size.x, size.y, i, invert, true);

		if (color_ramp.is_valid()) {
			data.write[i] = _modulate_with_gradient(data[i], color_ramp);
		}
	}

	if (generate_mipmaps) {
		data.append_array(_generate_mipmaps(data));
	}

	TypedArray<Image> images;
	images.resize(data.size());
	for (int i = 0; i < images.size(); i++) {
		images[i] = data[i];
	}

	return images;
}

Ref<Image> NoiseTexture3D::_modulate_with_gradient(Ref<Image> p_image, Ref<Gradient> p_gradient) {
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

Vector<Ref<Image>> NoiseTexture3D::_generate_mipmaps(const Vector<Ref<Image>> &p_data) {
	Vector<Ref<Image>> mipmap_images;
	Vector<Ref<Image>> parent_images = p_data;

	//create 3D mipmaps, this is horrible, though not used very often
	int w = p_data[0]->get_width();
	int h = p_data[0]->get_height();
	int d = p_data.size();

	while (w > 1 || h > 1 || d > 1) {
		Vector<Ref<Image>> mipmaps;
		int mm_w = MAX(1, w >> 1);
		int mm_h = MAX(1, h >> 1);
		int mm_d = MAX(1, d >> 1);

		for (int i = 0; i < mm_d; i++) {
			Ref<Image> mm;
			mm.instantiate();
			mm->create(mm_w, mm_h, false, p_data[0]->get_format());
			Vector3 pos;
			pos.z = float(i) * float(d) / float(mm_d) + 0.5;
			for (int x = 0; x < mm_w; x++) {
				for (int y = 0; y < mm_h; y++) {
					pos.x = float(x) * float(w) / float(mm_w) + 0.5;
					pos.y = float(y) * float(h) / float(mm_h) + 0.5;

					Vector3i posi = Vector3i(pos);
					Vector3 fract = pos - Vector3(posi);
					Vector3i posi_n = posi;
					if (posi_n.x < w - 1) {
						posi_n.x++;
					}
					if (posi_n.y < h - 1) {
						posi_n.y++;
					}
					if (posi_n.z < d - 1) {
						posi_n.z++;
					}

					Color c000 = parent_images[posi.z]->get_pixel(posi.x, posi.y);
					Color c100 = parent_images[posi.z]->get_pixel(posi_n.x, posi.y);
					Color c010 = parent_images[posi.z]->get_pixel(posi.x, posi_n.y);
					Color c110 = parent_images[posi.z]->get_pixel(posi_n.x, posi_n.y);
					Color c001 = parent_images[posi_n.z]->get_pixel(posi.x, posi.y);
					Color c101 = parent_images[posi_n.z]->get_pixel(posi_n.x, posi.y);
					Color c011 = parent_images[posi_n.z]->get_pixel(posi.x, posi_n.y);
					Color c111 = parent_images[posi_n.z]->get_pixel(posi_n.x, posi_n.y);

					Color cx00 = c000.lerp(c100, fract.x);
					Color cx01 = c001.lerp(c101, fract.x);
					Color cx10 = c010.lerp(c110, fract.x);
					Color cx11 = c011.lerp(c111, fract.x);

					Color cy0 = cx00.lerp(cx10, fract.y);
					Color cy1 = cx01.lerp(cx11, fract.y);

					Color cz = cy0.lerp(cy1, fract.z);

					mm->set_pixel(x, y, cz);
				}
			}

			mipmaps.push_back(mm);
		}

		w = mm_w;
		h = mm_h;
		d = mm_d;

		mipmap_images.append_array(mipmaps);
		parent_images = mipmaps;
	}
	return mipmap_images;
}

void NoiseTexture3D::_update_texture() {
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
		TypedArray<Image> data = _generate_texture();
		_set_texture_data(data);
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
	if (p_width == size.x) {
		return;
	}
	size.x = p_width;
	_queue_update();
}

void NoiseTexture3D::set_height(int p_height) {
	ERR_FAIL_COND(p_height <= 0);
	if (p_height == size.y) {
		return;
	}
	size.y = p_height;
	_queue_update();
}

void NoiseTexture3D::set_depth(int p_depth) {
	ERR_FAIL_COND(p_depth <= 0);
	if (p_depth == size.z) {
		return;
	}
	size.z = p_depth;
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

void NoiseTexture3D::set_generate_mipmaps(bool p_enable) {
	if (p_enable == generate_mipmaps) {
		return;
	}
	generate_mipmaps = p_enable;
	_queue_update();
}

bool NoiseTexture3D::is_generating_mipmaps() const {
	return generate_mipmaps;
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

Ref<Gradient> NoiseTexture3D::get_color_ramp() const {
	return color_ramp;
}

int NoiseTexture3D::get_width() const {
	return size.x;
}

int NoiseTexture3D::get_height() const {
	return size.y;
}

int NoiseTexture3D::get_depth() const {
	return size.z;
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
	return Image::Format::FORMAT_RGBA8;
}
