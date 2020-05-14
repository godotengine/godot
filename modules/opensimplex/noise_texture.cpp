/*************************************************************************/
/*  noise_texture.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

NoiseTexture::NoiseTexture() {
	update_queued = false;
	noise_thread = nullptr;
	regen_queued = false;
	first_time = true;

	size = Vector2i(512, 512);
	seamless = false;
	as_normalmap = false;
	bump_strength = 8.0;

	noise = Ref<OpenSimplexNoise>();

	_queue_update();
}

NoiseTexture::~NoiseTexture() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
	if (noise_thread) {
		Thread::wait_to_finish(noise_thread);
		memdelete(noise_thread);
	}
}

void NoiseTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_width", "width"), &NoiseTexture::set_width);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &NoiseTexture::set_height);

	ClassDB::bind_method(D_METHOD("set_noise", "noise"), &NoiseTexture::set_noise);
	ClassDB::bind_method(D_METHOD("get_noise"), &NoiseTexture::get_noise);

	ClassDB::bind_method(D_METHOD("set_seamless", "seamless"), &NoiseTexture::set_seamless);
	ClassDB::bind_method(D_METHOD("get_seamless"), &NoiseTexture::get_seamless);

	ClassDB::bind_method(D_METHOD("set_as_normalmap", "as_normalmap"), &NoiseTexture::set_as_normalmap);
	ClassDB::bind_method(D_METHOD("is_normalmap"), &NoiseTexture::is_normalmap);

	ClassDB::bind_method(D_METHOD("set_bump_strength", "bump_strength"), &NoiseTexture::set_bump_strength);
	ClassDB::bind_method(D_METHOD("get_bump_strength"), &NoiseTexture::get_bump_strength);

	ClassDB::bind_method(D_METHOD("_update_texture"), &NoiseTexture::_update_texture);
	ClassDB::bind_method(D_METHOD("_generate_texture"), &NoiseTexture::_generate_texture);
	ClassDB::bind_method(D_METHOD("_thread_done", "image"), &NoiseTexture::_thread_done);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,2048,1,or_greater"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "height", PROPERTY_HINT_RANGE, "1,2048,1,or_greater"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "seamless"), "set_seamless", "get_seamless");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "as_normalmap"), "set_as_normalmap", "is_normalmap");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bump_strength", PROPERTY_HINT_RANGE, "0,32,0.1,or_greater"), "set_bump_strength", "get_bump_strength");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "noise", PROPERTY_HINT_RESOURCE_TYPE, "OpenSimplexNoise"), "set_noise", "get_noise");
}

void NoiseTexture::_validate_property(PropertyInfo &property) const {
	if (property.name == "bump_strength") {
		if (!as_normalmap) {
			property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}
}

void NoiseTexture::_set_texture_data(const Ref<Image> &p_image) {
	data = p_image;
	if (data.is_valid()) {
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
	_set_texture_data(p_image);
	Thread::wait_to_finish(noise_thread);
	memdelete(noise_thread);
	noise_thread = nullptr;
	if (regen_queued) {
		noise_thread = Thread::create(_thread_function, this);
		regen_queued = false;
	}
}

void NoiseTexture::_thread_function(void *p_ud) {
	NoiseTexture *tex = (NoiseTexture *)p_ud;
	tex->call_deferred("_thread_done", tex->_generate_texture());
}

void NoiseTexture::_queue_update() {
	if (update_queued) {
		return;
	}

	update_queued = true;
	call_deferred("_update_texture");
}

Ref<Image> NoiseTexture::_generate_texture() {
	// Prevent memdelete due to unref() on other thread.
	Ref<OpenSimplexNoise> ref_noise = noise;

	if (ref_noise.is_null()) {
		return Ref<Image>();
	}

	Ref<Image> image;

	if (seamless) {
		image = ref_noise->get_seamless_image(size.x);
	} else {
		image = ref_noise->get_image(size.x, size.y);
	}

	if (as_normalmap) {
		image->bumpmap_to_normalmap(bump_strength);
	}

	return image;
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
		if (!noise_thread) {
			noise_thread = Thread::create(_thread_function, this);
			regen_queued = false;
		} else {
			regen_queued = true;
		}

	} else {
		Ref<Image> image = _generate_texture();
		_set_texture_data(image);
	}
	update_queued = false;
}

void NoiseTexture::set_noise(Ref<OpenSimplexNoise> p_noise) {
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

Ref<OpenSimplexNoise> NoiseTexture::get_noise() {
	return noise;
}

void NoiseTexture::set_width(int p_width) {
	if (p_width == size.x) {
		return;
	}
	size.x = p_width;
	_queue_update();
}

void NoiseTexture::set_height(int p_height) {
	if (p_height == size.y) {
		return;
	}
	size.y = p_height;
	_queue_update();
}

void NoiseTexture::set_seamless(bool p_seamless) {
	if (p_seamless == seamless) {
		return;
	}
	seamless = p_seamless;
	_queue_update();
}

bool NoiseTexture::get_seamless() {
	return seamless;
}

void NoiseTexture::set_as_normalmap(bool p_as_normalmap) {
	if (p_as_normalmap == as_normalmap) {
		return;
	}
	as_normalmap = p_as_normalmap;
	_queue_update();
	_change_notify();
}

bool NoiseTexture::is_normalmap() {
	return as_normalmap;
}

void NoiseTexture::set_bump_strength(float p_bump_strength) {
	if (p_bump_strength == bump_strength) {
		return;
	}
	bump_strength = p_bump_strength;
	if (as_normalmap) {
		_queue_update();
	}
}

float NoiseTexture::get_bump_strength() {
	return bump_strength;
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

Ref<Image> NoiseTexture::get_data() const {
	return data;
}
