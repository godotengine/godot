/*************************************************************************/
/*  gltf_texture.cpp                                                     */
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

#include "gltf_texture.h"

void GLTFTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_src_image"), &GLTFTexture::get_src_image);
	ClassDB::bind_method(D_METHOD("set_src_image", "src_image"), &GLTFTexture::set_src_image);
	ClassDB::bind_method(D_METHOD("get_sampler"), &GLTFTexture::get_sampler);
	ClassDB::bind_method(D_METHOD("set_sampler", "sampler"), &GLTFTexture::set_sampler);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "src_image"), "set_src_image", "get_src_image"); // int
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sampler"), "set_sampler", "get_sampler"); // int
}

GLTFImageIndex GLTFTexture::get_src_image() const {
	return src_image;
}

void GLTFTexture::set_src_image(GLTFImageIndex val) {
	src_image = val;
}

GLTFTextureSamplerIndex GLTFTexture::get_sampler() const {
	return sampler;
}

void GLTFTexture::set_sampler(GLTFTextureSamplerIndex val) {
	sampler = val;
}
