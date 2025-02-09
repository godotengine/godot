/**************************************************************************/
/*  drawable_texture_2d.cpp                                               */
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

#include "drawable_texture_2d.h"

int DrawableTexture2D::get_width() const {
	return w;
}

int DrawableTexture2D::get_height() const {
	return h;
}

RID DrawableTexture2D::get_rid() const {
	if (texture.is_null()) {
		// We are in trouble, create something temporary.
		// 4, 4, false, Image::FORMAT_RGBA8
		texture = RenderingServer::get_singleton()->texture_2d_placeholder_create();
	}
	return texture;
}

void DrawableTexture2D::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	if ((w | h) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, Size2(w, h)), texture, false, p_modulate, p_transpose);
}

void DrawableTexture2D::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	if ((w | h) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, texture, p_tile, p_modulate, p_transpose);
}

void DrawableTexture2D::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	if ((w | h) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, texture, p_src_rect, p_modulate, p_transpose, p_clip_uv);
}

// Initialized DrawableTexture
void DrawableTexture2D::setup(int p_width, int p_height, DrawableFormat p_format, bool p_use_mipmaps) {
	w = p_width;
	h = p_height;
	texture = RS::get_singleton()->texture_drawable_create(p_width, p_height, (RS::TextureDrawableFormat)p_format, p_use_mipmaps);
	// Store a reference to default_material for later.
	default_material = RS::get_singleton()->texture_drawable_get_default_material();
	notify_property_list_changed();
	emit_changed();
}

void DrawableTexture2D::blit_rect(const Rect2i p_rect, const Ref<Texture2D> &p_source, const Color &p_modulate, int p_mipmap, const Ref<Material> &p_material) {
	// Use user Shader if exists.
	RID material = default_material;
	if (!p_material.is_null()) {
		material = p_material->get_rid();
		if (p_material->get_shader_mode() != Shader::MODE_TEXTURE_BLIT) {
			WARN_PRINT("ShaderMaterial passed to blit_rect() is not a texture_blit shader. Using default instead");
		}
	}

	// Rendering server expects textureParameters as a TypedArray[RID]
	Array textures;
	textures.push_back(texture);
	Array src_textures;
	src_textures.push_back(p_source);

	RS::get_singleton()->texture_drawable_blit_rect(textures, p_rect, material, p_modulate, src_textures, p_mipmap);
	notify_property_list_changed();
	emit_changed();
}

void DrawableTexture2D::blit_rect_multi(const Rect2i p_rect, const TypedArray<Texture2D> &p_sources, const TypedArray<DrawableTexture2D> &p_extra_targets, const Color &p_modulate, int p_mipmap, const Ref<Material> &p_material) {
	RID material = default_material;
	if (!p_material.is_null()) {
		material = p_material->get_rid();
		if (p_material->get_shader_mode() != Shader::MODE_TEXTURE_BLIT) {
			WARN_PRINT("ShaderMaterial passed to blit_rect_multi() is not a texture_blit shader. Using default instead.");
		}
	}

	// Rendering server expects textureParameters as a TypedArray[RID]
	Array textures;
	textures.push_back(texture);
	int i = 0;
	while (i < p_extra_targets.size()) {
		textures.push_back(p_extra_targets[i]);
		i += 1;
	}
	i = 0;
	Array src_textures;
	while (i < p_sources.size()) {
		src_textures.push_back(p_sources[i]);
		i += 1;
	}

	RS::get_singleton()->texture_drawable_blit_rect(textures, p_rect, material, p_modulate, src_textures, p_mipmap);
	notify_property_list_changed();
	emit_changed();
}

Ref<Image> DrawableTexture2D::get_image() const {
	if (texture.is_valid()) {
		return RS::get_singleton()->texture_2d_get(texture);
	} else {
		return Ref<Image>();
	}
}

void DrawableTexture2D::generate_mipmaps() {
	if (texture.is_valid()) {
		RS::get_singleton()->texture_drawable_generate_mipmaps(texture);
	}
}

void DrawableTexture2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("setup", "width", "height", "format", "use_mipmaps"), &DrawableTexture2D::setup);
	ClassDB::bind_method(D_METHOD("blit_rect", "rect", "source", "modulate", "mipmap", "material"), &DrawableTexture2D::blit_rect, DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Ref<Material>()));
	ClassDB::bind_method(D_METHOD("blit_rect_multi", "rect", "sources", "extra_targets", "modulate", "mipmap", "material"), &DrawableTexture2D::blit_rect_multi, DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Ref<Material>()));
	ClassDB::bind_method(D_METHOD("generate_mipmaps"), &DrawableTexture2D::generate_mipmaps);

	BIND_ENUM_CONSTANT(DRAWABLE_FORMAT_RGBA8);
	BIND_ENUM_CONSTANT(DRAWABLE_FORMAT_RGBA8_SRGB);
	BIND_ENUM_CONSTANT(DRAWABLE_FORMAT_RGBAH);
	BIND_ENUM_CONSTANT(DRAWABLE_FORMAT_RGBAF);
}

DrawableTexture2D::DrawableTexture2D() {}

DrawableTexture2D::~DrawableTexture2D() {
	if (texture.is_valid()) {
		ERR_FAIL_NULL(RenderingServer::get_singleton());
		RenderingServer::get_singleton()->free(texture);
	}
}
