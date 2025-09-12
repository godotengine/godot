/**************************************************************************/
/*  drawable_texture.cpp                                                  */
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

#include "drawable_texture.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"

int DrawableTexture2D::get_width() const {
	return size.width;
}

int DrawableTexture2D::get_height() const {
	return size.height;
}

bool DrawableTexture2D::has_alpha() const {
	return true;
}

RID DrawableTexture2D::get_rid() const {
	if (texture.is_null()) {
		texture = RS::get_singleton()->texture_2d_placeholder_create();
	}
	return texture;
}

Ref<Image> DrawableTexture2D::get_image() const {
	if (texture.is_null()) {
		return Ref<Image>();
	}
	return RenderingServer::get_singleton()->texture_2d_get(texture);
}

void DrawableTexture2D::setup(Size2i p_size, RS::TextureDrawableFormat p_texture_format, bool p_use_mipmaps) {
	ERR_FAIL_COND(p_size.width <= 0 || p_size.width > 16384);
	ERR_FAIL_COND(p_size.height <= 0 || p_size.height > 16384);
	size = p_size;
	RID tex_new = RS::get_singleton()->texture_drawable_2d_create(size.width, size.height, p_texture_format, p_use_mipmaps);
	if (texture.is_null()) {
		texture = tex_new;
	} else {
		RS::get_singleton()->texture_replace(texture, tex_new);
	}
	emit_changed();
}

void DrawableTexture2D::blit_mesh_advanced(const Ref<Material> &p_material, const Ref<Mesh> &p_mesh, uint32_t p_surface_index, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color) {
	ERR_FAIL_COND(p_material.is_null());
	ERR_FAIL_COND(p_mesh.is_null());
	RS::get_singleton()->texture_drawable_blit_mesh_advanced(texture, p_material->get_rid(), p_mesh->get_rid(), p_surface_index, p_blend_mode, p_clear_color);
}

void DrawableTexture2D::blit_texture_rect(const Ref<Texture2D> &p_source_texture, Rect2 p_dst_rect, const Color &p_modulate, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color) {
	ERR_FAIL_COND(p_source_texture.is_null());
	RS::get_singleton()->texture_drawable_blit_texture_rect_region(texture, p_source_texture->get_rid(), p_dst_rect, Rect2(Vector2(), p_source_texture->get_size()), p_modulate, p_blend_mode, p_clear_color);
}

void DrawableTexture2D::blit_texture_rect_region(const Ref<Texture2D> &p_source_texture, Rect2 p_dst_rect, Rect2 p_src_rect, const Color &p_modulate, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color) {
	ERR_FAIL_COND(p_source_texture.is_null());
	RS::get_singleton()->texture_drawable_blit_texture_rect_region(texture, p_source_texture->get_rid(), p_dst_rect, p_src_rect, p_modulate, p_blend_mode, p_clear_color);
}

void DrawableTexture2D::generate_mipmaps() {
	RS::get_singleton()->texture_drawable_generate_mipmaps(texture);
}

DrawableTexture2D::~DrawableTexture2D() {
	RS::get_singleton()->free(texture);
}

void DrawableTexture2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("setup", "size", "texture_format", "use_mipmaps"), &DrawableTexture2D::setup, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("blit_mesh_advanced", "material", "mesh", "surface_index", "blend_mode", "clear_color"), &DrawableTexture2D::blit_mesh_advanced);
	ClassDB::bind_method(D_METHOD("blit_texture_rect", "source_texture", "dst_rect", "modulate", "blend_mode", "clear_color"), &DrawableTexture2D::blit_texture_rect);
	ClassDB::bind_method(D_METHOD("blit_texture_rect_region", "source_texture", "dst_rect", "src_rect", "modulate", "blend_mode", "clear_color"), &DrawableTexture2D::blit_texture_rect_region);
	ClassDB::bind_method(D_METHOD("generate_mipmaps"), &DrawableTexture2D::generate_mipmaps);
}

Image::Format DrawableTextureLayered::get_format() const {
	return image_format;
}

DrawableTextureLayered::LayeredType DrawableTextureLayered::get_layered_type() const {
	return layer_type;
}

int DrawableTextureLayered::get_width() const {
	return size.width;
}

int DrawableTextureLayered::get_height() const {
	return size.height;
}

int DrawableTextureLayered::get_layers() const {
	return layers;
}

bool DrawableTextureLayered::has_mipmaps() const {
	return mipmaps > 1;
}

RID DrawableTextureLayered::get_rid() const {
	if (texture.is_null()) {
		texture = RS::get_singleton()->texture_2d_layered_placeholder_create((RS::TextureLayeredType)layer_type);
	}
	return texture;
}

Ref<Image> DrawableTextureLayered::get_layer_data(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, (int)layers, Ref<Image>());
	return RS::get_singleton()->texture_2d_layer_get(texture, p_layer);
}

void DrawableTextureLayered::setup(Size2i p_size, int p_layers, LayeredType p_layer_type, RS::TextureDrawableFormat p_texture_format, bool p_use_mipmaps) {
	ERR_FAIL_COND(p_size.width <= 0 || p_size.width > 16384);
	ERR_FAIL_COND(p_size.height <= 0 || p_size.height > 16384);

	RS::TextureLayeredType rs_layer_type;
	switch (p_layer_type) {
		case LAYERED_TYPE_2D_ARRAY: {
			ERR_FAIL_COND(p_layers <= 1);
			rs_layer_type = RS::TEXTURE_LAYERED_2D_ARRAY;
		} break;
		case LAYERED_TYPE_CUBEMAP: {
			ERR_FAIL_COND(p_layers != 6);
			rs_layer_type = RS::TEXTURE_LAYERED_CUBEMAP;
		} break;
		case LAYERED_TYPE_CUBEMAP_ARRAY: {
			ERR_FAIL_COND((p_layers < 6) || ((p_layers % 6) != 0));
			rs_layer_type = RS::TEXTURE_LAYERED_CUBEMAP_ARRAY;
		} break;
		default: {
			ERR_FAIL_MSG("Unknown layer type selected");
		} break;
	}
	size = p_size;
	layers = p_layers;
	layer_type = p_layer_type;

	RID tex_new = RS::get_singleton()->texture_drawable_2d_layered_create(p_size.width, p_size.height, p_layers, rs_layer_type, p_texture_format, p_use_mipmaps);
	if (texture.is_null()) {
		texture = tex_new;
	} else {
		RS::get_singleton()->texture_replace(texture, tex_new);
	}

	image_format = RS::get_singleton()->texture_get_format(texture);
	mipmaps = 1 + (p_use_mipmaps ? Image::get_image_required_mipmaps(size.width, p_size.height, image_format) : 0);
	emit_changed();
}

void DrawableTextureLayered::blit_mesh_advanced(const Ref<Material> &p_material, const Ref<Mesh> &p_mesh, uint32_t p_surface_index, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color, int p_layer) {
	ERR_FAIL_COND(p_material.is_null());
	ERR_FAIL_COND(p_mesh.is_null());
	RS::get_singleton()->texture_drawable_blit_mesh_advanced(texture, p_material->get_rid(), p_mesh->get_rid(), p_surface_index, p_blend_mode, p_clear_color, p_layer);
}

void DrawableTextureLayered::blit_texture_rect(const Ref<Texture2D> &p_source_texture, Rect2 p_dst_rect, const Color &p_modulate, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color, int p_layer) {
	ERR_FAIL_COND(p_source_texture.is_null());
	RS::get_singleton()->texture_drawable_blit_texture_rect_region(texture, p_source_texture->get_rid(), p_dst_rect, Rect2(Vector2(), p_source_texture->get_size()), p_modulate, p_blend_mode, p_clear_color, p_layer);
}

void DrawableTextureLayered::blit_texture_rect_region(const Ref<Texture2D> &p_source_texture, Rect2 p_dst_rect, Rect2 p_src_rect, const Color &p_modulate, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color, int p_layer) {
	ERR_FAIL_COND(p_source_texture.is_null());
	RS::get_singleton()->texture_drawable_blit_texture_rect_region(texture, p_source_texture->get_rid(), p_dst_rect, p_src_rect, p_modulate, p_blend_mode, p_clear_color, p_layer);
}

void DrawableTextureLayered::generate_mipmaps(int p_layer) {
	RS::get_singleton()->texture_drawable_generate_mipmaps(texture, p_layer);
}

DrawableTextureLayered::DrawableTextureLayered(LayeredType p_layer_type) {
	layer_type = p_layer_type;
}

DrawableTextureLayered::~DrawableTextureLayered() {
	RS::get_singleton()->free(texture);
}

void DrawableTextureLayered::_bind_methods() {
	ClassDB::bind_method(D_METHOD("setup", "size", "layers", "layer_type", "texture_format", "use_mipmaps"), &DrawableTextureLayered::setup, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("blit_mesh_advanced", "material", "mesh", "surface_index", "blend_mode", "clear_color", "layer"), &DrawableTextureLayered::blit_mesh_advanced);
	ClassDB::bind_method(D_METHOD("blit_texture_rect", "source_texture", "dst_rect", "modulate", "blend_mode", "clear_color", "layer"), &DrawableTextureLayered::blit_texture_rect);
	ClassDB::bind_method(D_METHOD("blit_texture_rect_region", "source_texture", "dst_rect", "src_rect", "modulate", "blend_mode", "clear_color", "layer"), &DrawableTextureLayered::blit_texture_rect_region);
	ClassDB::bind_method(D_METHOD("generate_mipmaps", "layer"), &DrawableTextureLayered::generate_mipmaps);
}
