/**************************************************************************/
/*  external_texture.cpp                                                  */
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

#include "external_texture.h"
#include "drivers/gles3/storage/texture_storage.h"
#include "servers/rendering/rendering_server_globals.h"

void ExternalTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &ExternalTexture::set_size);
	ClassDB::bind_method(D_METHOD("get_external_texture_id"), &ExternalTexture::get_external_texture_id);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size"), "set_size", "get_size");
}

uint32_t ExternalTexture::get_external_texture_id() {
	return ((GLES3::TextureStorage *)RSG::texture_storage)->texture_get_texid(_texture);
}

void ExternalTexture::set_size(const Size2 &p_size) {
	if (p_size.width > 0 && p_size.height > 0) {
		_size = p_size;
		RenderingServer::get_singleton()->texture_set_size_override(_texture, _size.width, _size.height);
		notify_property_list_changed();
	}
}

Size2 ExternalTexture::get_size() const {
	return _size;
}

int ExternalTexture::get_width() const {
	return _size.width;
}

int ExternalTexture::get_height() const {
	return _size.height;
}

bool ExternalTexture::has_alpha() const {
	return false;
}

RID ExternalTexture::get_rid() const {
	return _texture;
}

Ref<Image> ExternalTexture::get_image() const {
	// not (yet) supported
	return Ref<Image>();
}

ExternalTexture::ExternalTexture() {
	_size = Size2(1.0, 1.0);
	Ref<Image> image = Image::create_empty(_size.width, _size.height, true, Image::FORMAT_RGBA8);
	_texture = RenderingServer::get_singleton()->texture_external_create(image);
	notify_property_list_changed();
}

ExternalTexture::~ExternalTexture() {
	if (_texture.is_valid()) {
		ERR_FAIL_NULL(RenderingServer::get_singleton());
		RenderingServer::get_singleton()->free(_texture);
	}
}
