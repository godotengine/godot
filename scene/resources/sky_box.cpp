/*************************************************************************/
/*  sky_box.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "sky_box.h"
#include "io/image_loader.h"

void Sky::set_radiance_size(RadianceSize p_size) {
	ERR_FAIL_INDEX(p_size, RADIANCE_SIZE_MAX);

	radiance_size = p_size;
	_radiance_changed();
}

Sky::RadianceSize Sky::get_radiance_size() const {

	return radiance_size;
}

void Sky::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_radiance_size", "size"), &Sky::set_radiance_size);
	ClassDB::bind_method(D_METHOD("get_radiance_size"), &Sky::get_radiance_size);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "radiance_size", PROPERTY_HINT_ENUM, "256,512,1024,2048"), "set_radiance_size", "get_radiance_size");

	BIND_CONSTANT(RADIANCE_SIZE_256);
	BIND_CONSTANT(RADIANCE_SIZE_512);
	BIND_CONSTANT(RADIANCE_SIZE_1024);
	BIND_CONSTANT(RADIANCE_SIZE_2048);
	BIND_CONSTANT(RADIANCE_SIZE_MAX);
}

Sky::Sky() {
	radiance_size = RADIANCE_SIZE_512;
}

/////////////////////////////////////////

void PanoramaSky::_radiance_changed() {

	if (panorama.is_valid()) {
		static const int size[RADIANCE_SIZE_MAX] = {
			256, 512, 1024, 2048
		};
		VS::get_singleton()->sky_set_texture(sky, panorama->get_rid(), size[get_radiance_size()]);
	}
}

void PanoramaSky::set_panorama(const Ref<Texture> &p_panorama) {

	panorama = p_panorama;

	if (panorama.is_valid()) {

		_radiance_changed();

	} else {
		VS::get_singleton()->sky_set_texture(sky, RID(), 0);
	}
}

Ref<Texture> PanoramaSky::get_panorama() const {

	return panorama;
}

RID PanoramaSky::get_rid() const {

	return sky;
}

void PanoramaSky::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_panorama", "texture:Texture"), &PanoramaSky::set_panorama);
	ClassDB::bind_method(D_METHOD("get_panorama:Texture"), &PanoramaSky::get_panorama);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "panorama", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_panorama", "get_panorama");
}

PanoramaSky::PanoramaSky() {

	sky = VS::get_singleton()->sky_create();
}

PanoramaSky::~PanoramaSky() {

	VS::get_singleton()->free(sky);
}
