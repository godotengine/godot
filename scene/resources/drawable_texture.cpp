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
	return texture;
}

Ref<Image> DrawableTexture2D::get_image() const {
	if (texture.is_null()) {
		return Ref<Image>();
	}
	return RenderingServer::get_singleton()->texture_2d_get(texture);
}

void DrawableTexture2D::setup(Size2i p_size, RD::DataFormat p_texture_format, bool p_use_mipmaps) {
	ERR_FAIL_COND(p_size.width <= 0 || p_size.width > 16384);
	ERR_FAIL_COND(p_size.height <= 0 || p_size.height > 16384);
	size = p_size;
	RID drawable_texture = RS::get_singleton()->texture_drawable_create(size.width, size.height, p_texture_format, p_use_mipmaps);
	RS::get_singleton()->texture_replace(texture, drawable_texture);
	emit_changed();
}

void DrawableTexture2D::draw_mesh(const Ref<Material> &p_material, const Ref<Mesh> &p_mesh, uint32_t p_surface_index, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color) {
	ERR_FAIL_COND(p_material.is_null());
	ERR_FAIL_COND(p_mesh.is_null());
	RS::get_singleton()->texture_drawable_draw_mesh(texture, p_material->get_rid(), p_mesh->get_rid(), p_surface_index, p_blend_mode, p_clear_color);
}

void DrawableTexture2D::blit_rect(Rect2i p_rect, const Ref<Texture2D> &p_source_texture, const Color &p_modulate, RS::TextureDrawableBlendMode p_blend_mode, const Color &p_clear_color) {
	ERR_FAIL_COND(p_source_texture.is_null());
	RS::get_singleton()->texture_drawable_blit_rect(texture, p_rect, p_source_texture->get_rid(), p_modulate, p_blend_mode, p_clear_color);
}

DrawableTexture2D::DrawableTexture2D() {
	texture = RS::get_singleton()->texture_2d_placeholder_create();
}

DrawableTexture2D::~DrawableTexture2D() {
	RS::get_singleton()->free(texture);
}

void DrawableTexture2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("setup", "size", "texture_format", "use_mipmaps"), &DrawableTexture2D::setup, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_mesh", "material", "mesh", "surface_index", "blend_mode", "clear_color"), &DrawableTexture2D::draw_mesh);
	ClassDB::bind_method(D_METHOD("blit_rect", "rect", "source_texture", "modulate", "blend_mode", "clear_color"), &DrawableTexture2D::blit_rect);
}
