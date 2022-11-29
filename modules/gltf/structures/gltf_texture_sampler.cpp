/*************************************************************************/
/*  gltf_texture_sampler.cpp                                             */
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

#include "gltf_texture_sampler.h"

void GLTFTextureSampler::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_mag_filter"), &GLTFTextureSampler::get_mag_filter);
	ClassDB::bind_method(D_METHOD("set_mag_filter", "filter_mode"), &GLTFTextureSampler::set_mag_filter);
	ClassDB::bind_method(D_METHOD("get_min_filter"), &GLTFTextureSampler::get_min_filter);
	ClassDB::bind_method(D_METHOD("set_min_filter", "filter_mode"), &GLTFTextureSampler::set_min_filter);
	ClassDB::bind_method(D_METHOD("get_wrap_s"), &GLTFTextureSampler::get_wrap_s);
	ClassDB::bind_method(D_METHOD("set_wrap_s", "wrap_mode"), &GLTFTextureSampler::set_wrap_s);
	ClassDB::bind_method(D_METHOD("get_wrap_t"), &GLTFTextureSampler::get_wrap_t);
	ClassDB::bind_method(D_METHOD("set_wrap_t", "wrap_mode"), &GLTFTextureSampler::set_wrap_t);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mag_filter"), "set_mag_filter", "get_mag_filter");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "min_filter"), "set_min_filter", "get_min_filter");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "wrap_s"), "set_wrap_s", "get_wrap_s");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "wrap_t"), "set_wrap_t", "get_wrap_t");
}
