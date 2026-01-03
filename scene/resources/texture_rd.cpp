/**************************************************************************/
/*  texture_rd.cpp                                                        */
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

#include "texture_rd.h"

#include "servers/rendering/rendering_server.h"

////////////////////////////////////////////////////////////////////////////
// Texture2DRD

void Texture2DRD::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture_rd_rid", "texture_rd_rid"), &Texture2DRD::set_texture_rd_rid);
	ClassDB::bind_method(D_METHOD("get_texture_rd_rid"), &Texture2DRD::get_texture_rd_rid);

	ADD_PROPERTY(PropertyInfo(Variant::RID, "texture_rd_rid", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_texture_rd_rid", "get_texture_rd_rid");
}

int Texture2DRD::get_width() const {
	return size.width;
}

int Texture2DRD::get_height() const {
	return size.height;
}

RID Texture2DRD::get_rid() const {
	if (texture_rid.is_null()) {
		// Initialize the resource RID if we have not to.
		// The RID should not change again until resource destruction,
		// because Material is not able to detect RID changes.
		// This also applies to the same lazy initialization in `set_texture_rd_rid()`
		// and `_set_texture_rd_rid()`.
		texture_rid = RS::get_singleton()->texture_2d_placeholder_create();
	}

	return texture_rid;
}

bool Texture2DRD::has_alpha() const {
	return false;
}

Ref<Image> Texture2DRD::get_image() const {
	ERR_FAIL_NULL_V(RS::get_singleton(), Ref<Image>());
	if (texture_rid.is_valid()) {
		return RS::get_singleton()->texture_2d_get(texture_rid);
	} else {
		return Ref<Image>();
	}
}

void Texture2DRD::set_texture_rd_rid(RID p_texture_rd_rid) {
	ERR_FAIL_NULL(RS::get_singleton());

	if (p_texture_rd_rid.is_valid()) {
		RS::get_singleton()->call_on_render_thread(callable_mp(this, &Texture2DRD::_set_texture_rd_rid).bind(p_texture_rd_rid));
	} else if (texture_rid.is_valid() && texture_rd_rid.is_valid()) {
		RID new_texture_rid = RS::get_singleton()->texture_2d_placeholder_create();
		RS::get_singleton()->texture_replace(texture_rid, new_texture_rid);
		size = Size2i();
		texture_rd_rid = RID();
		notify_property_list_changed();
		emit_changed();
	}
}

void Texture2DRD::_set_texture_rd_rid(RID p_texture_rd_rid) {
	ERR_FAIL_NULL(RD::get_singleton());
	ERR_FAIL_COND(!RD::get_singleton()->texture_is_valid(p_texture_rd_rid));

	RD::TextureFormat tf = RD::get_singleton()->texture_get_format(p_texture_rd_rid);
	ERR_FAIL_COND(tf.texture_type != RD::TEXTURE_TYPE_2D);
	ERR_FAIL_COND(tf.depth > 1);
	ERR_FAIL_COND(tf.array_layers > 1);

	size.width = tf.width;
	size.height = tf.height;

	texture_rd_rid = p_texture_rd_rid;

	if (texture_rid.is_valid()) {
		RS::get_singleton()->texture_replace(texture_rid, RS::get_singleton()->texture_rd_create(p_texture_rd_rid));
	} else {
		texture_rid = RS::get_singleton()->texture_rd_create(p_texture_rd_rid);
	}

	notify_property_list_changed();
	emit_changed();
}

RID Texture2DRD::get_texture_rd_rid() const {
	return texture_rd_rid;
}

Texture2DRD::Texture2DRD() {
	size = Size2i();
}

Texture2DRD::~Texture2DRD() {
	if (texture_rid.is_valid()) {
		ERR_FAIL_NULL(RS::get_singleton());
		RS::get_singleton()->free_rid(texture_rid);
		texture_rid = RID();
	}
}

////////////////////////////////////////////////////////////////////////////
// TextureLayeredRD

void TextureLayeredRD::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture_rd_rid", "texture_rd_rid"), &TextureLayeredRD::set_texture_rd_rid);
	ClassDB::bind_method(D_METHOD("get_texture_rd_rid"), &TextureLayeredRD::get_texture_rd_rid);

	ADD_PROPERTY(PropertyInfo(Variant::RID, "texture_rd_rid", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_texture_rd_rid", "get_texture_rd_rid");
}

TextureLayered::LayeredType TextureLayeredRD::get_layered_type() const {
	return layer_type;
}

Image::Format TextureLayeredRD::get_format() const {
	return image_format;
}

int TextureLayeredRD::get_width() const {
	return size.width;
}

int TextureLayeredRD::get_height() const {
	return size.height;
}

int TextureLayeredRD::get_layers() const {
	return (int)layers;
}

bool TextureLayeredRD::has_mipmaps() const {
	return mipmaps > 1;
}

RID TextureLayeredRD::get_rid() const {
	if (texture_rid.is_null()) {
		// Initialize the resource RID if we have not to.
		// The RID should not change again until resource destruction,
		// because Material is not able to detect RID changes.
		// This also applies to the same lazy initialization in `set_texture_rd_rid()`
		// and `_set_texture_rd_rid()`.
		texture_rid = RS::get_singleton()->texture_2d_placeholder_create();
	}

	return texture_rid;
}

Ref<Image> TextureLayeredRD::get_layer_data(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers, Ref<Image>());
	return RS::get_singleton()->texture_2d_layer_get(texture_rid, p_layer);
}

void TextureLayeredRD::set_texture_rd_rid(RID p_texture_rd_rid) {
	ERR_FAIL_NULL(RS::get_singleton());

	if (p_texture_rd_rid.is_valid()) {
		RS::get_singleton()->call_on_render_thread(callable_mp(this, &TextureLayeredRD::_set_texture_rd_rid).bind(p_texture_rd_rid));
	} else if (texture_rid.is_valid() && texture_rd_rid.is_valid()) {
		RID new_texture_rid = RS::get_singleton()->texture_2d_placeholder_create();
		RS::get_singleton()->texture_replace(texture_rid, new_texture_rid);
		image_format = Image::FORMAT_MAX;
		size = Size2i();
		layers = 0;
		mipmaps = 0;
		texture_rd_rid = RID();
		notify_property_list_changed();
		emit_changed();
	}
}

void TextureLayeredRD::_set_texture_rd_rid(RID p_texture_rd_rid) {
	ERR_FAIL_NULL(RD::get_singleton());
	ERR_FAIL_COND(!RD::get_singleton()->texture_is_valid(p_texture_rd_rid));

	RS::TextureLayeredType rs_layer_type;
	RD::TextureFormat tf = RD::get_singleton()->texture_get_format(p_texture_rd_rid);
	ERR_FAIL_COND(tf.texture_type != RD::TEXTURE_TYPE_2D_ARRAY && tf.texture_type != RD::TEXTURE_TYPE_CUBE && tf.texture_type != RD::TEXTURE_TYPE_CUBE_ARRAY);
	ERR_FAIL_COND(tf.depth > 1);
	switch (layer_type) {
		case LAYERED_TYPE_2D_ARRAY: {
			ERR_FAIL_COND(tf.array_layers <= 1);
			rs_layer_type = RS::TEXTURE_LAYERED_2D_ARRAY;
		} break;
		case LAYERED_TYPE_CUBEMAP: {
			ERR_FAIL_COND(tf.array_layers != 6);
			rs_layer_type = RS::TEXTURE_LAYERED_CUBEMAP;
		} break;
		case LAYERED_TYPE_CUBEMAP_ARRAY: {
			ERR_FAIL_COND((tf.array_layers == 0) || ((tf.array_layers % 6) != 0));
			rs_layer_type = RS::TEXTURE_LAYERED_CUBEMAP_ARRAY;
		} break;
		default: {
			ERR_FAIL_MSG("Unknown layer type selected");
		} break;
	}

	size.width = tf.width;
	size.height = tf.height;
	layers = tf.array_layers;
	mipmaps = tf.mipmaps;

	texture_rd_rid = p_texture_rd_rid;

	if (texture_rid.is_valid()) {
		RS::get_singleton()->texture_replace(texture_rid, RS::get_singleton()->texture_rd_create(p_texture_rd_rid, rs_layer_type));
	} else {
		texture_rid = RS::get_singleton()->texture_rd_create(p_texture_rd_rid, rs_layer_type);
	}

	image_format = RS::get_singleton()->texture_get_format(texture_rid);

	notify_property_list_changed();
	emit_changed();
}

RID TextureLayeredRD::get_texture_rd_rid() const {
	return texture_rd_rid;
}

TextureLayeredRD::TextureLayeredRD(LayeredType p_layer_type) {
	layer_type = p_layer_type;
	size = Size2i();
	image_format = Image::FORMAT_MAX;
	layers = 0;
	mipmaps = 0;
}

TextureLayeredRD::~TextureLayeredRD() {
	if (texture_rid.is_valid()) {
		ERR_FAIL_NULL(RS::get_singleton());
		RS::get_singleton()->free_rid(texture_rid);
		texture_rid = RID();
	}
}

////////////////////////////////////////////////////////////////////////////
// Texture3DRD

void Texture3DRD::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_texture_rd_rid", "texture_rd_rid"), &Texture3DRD::set_texture_rd_rid);
	ClassDB::bind_method(D_METHOD("get_texture_rd_rid"), &Texture3DRD::get_texture_rd_rid);

	ADD_PROPERTY(PropertyInfo(Variant::RID, "texture_rd_rid", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_texture_rd_rid", "get_texture_rd_rid");
}

Image::Format Texture3DRD::get_format() const {
	return image_format;
}

int Texture3DRD::get_width() const {
	return size.x;
}

int Texture3DRD::get_height() const {
	return size.y;
}

int Texture3DRD::get_depth() const {
	return size.z;
}

bool Texture3DRD::has_mipmaps() const {
	return mipmaps > 1;
}

RID Texture3DRD::get_rid() const {
	if (texture_rid.is_null()) {
		// Initialize the resource RID if we have not to.
		// The RID should not change again until resource destruction,
		// because Material is not able to detect RID changes.
		// This also applies to the same lazy initialization in `set_texture_rd_rid()`
		// and `_set_texture_rd_rid()`.
		texture_rid = RS::get_singleton()->texture_2d_placeholder_create();
	}

	return texture_rid;
}

void Texture3DRD::set_texture_rd_rid(RID p_texture_rd_rid) {
	ERR_FAIL_NULL(RS::get_singleton());

	if (p_texture_rd_rid.is_valid()) {
		RS::get_singleton()->call_on_render_thread(callable_mp(this, &Texture3DRD::_set_texture_rd_rid).bind(p_texture_rd_rid));
	} else if (texture_rid.is_valid() && texture_rd_rid.is_valid()) {
		RID new_texture_rid = RS::get_singleton()->texture_2d_placeholder_create();
		RS::get_singleton()->texture_replace(texture_rid, new_texture_rid);
		image_format = Image::FORMAT_MAX;
		size = Vector3i();
		mipmaps = 0;
		texture_rd_rid = RID();
		notify_property_list_changed();
		emit_changed();
	}
}

void Texture3DRD::_set_texture_rd_rid(RID p_texture_rd_rid) {
	ERR_FAIL_NULL(RD::get_singleton());
	ERR_FAIL_COND(!RD::get_singleton()->texture_is_valid(p_texture_rd_rid));

	RD::TextureFormat tf = RD::get_singleton()->texture_get_format(p_texture_rd_rid);
	ERR_FAIL_COND(tf.texture_type != RD::TEXTURE_TYPE_3D);
	ERR_FAIL_COND(tf.array_layers > 1);

	size.x = tf.width;
	size.y = tf.height;
	size.z = tf.depth;
	mipmaps = tf.mipmaps;

	texture_rd_rid = p_texture_rd_rid;

	if (texture_rid.is_valid()) {
		RS::get_singleton()->texture_replace(texture_rid, RS::get_singleton()->texture_rd_create(p_texture_rd_rid));
	} else {
		texture_rid = RS::get_singleton()->texture_rd_create(p_texture_rd_rid);
	}

	image_format = RS::get_singleton()->texture_get_format(texture_rid);

	notify_property_list_changed();
	emit_changed();
}

RID Texture3DRD::get_texture_rd_rid() const {
	return texture_rd_rid;
}

Texture3DRD::Texture3DRD() {
	image_format = Image::FORMAT_MAX;
	size = Vector3i();
	mipmaps = 0;
}

Texture3DRD::~Texture3DRD() {
	if (texture_rid.is_valid()) {
		ERR_FAIL_NULL(RS::get_singleton());
		RS::get_singleton()->free_rid(texture_rid);
		texture_rid = RID();
	}
}
