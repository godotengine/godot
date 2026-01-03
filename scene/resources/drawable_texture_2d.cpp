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

DrawableTexture2D::DrawableTexture2D() {
	default_material = RS::get_singleton()->texture_drawable_get_default_material();
	//_initialize();
}

DrawableTexture2D::~DrawableTexture2D() {
	if (texture.is_valid()) {
		ERR_FAIL_NULL(RenderingServer::get_singleton());
		RenderingServer::get_singleton()->free_rid(texture);
	}
}

// Initialize Texture Resource with a call to rendering server. Overwrite existing.
void DrawableTexture2D::_initialize() {
	if (texture.is_valid()) {
		RID new_texture = RS::get_singleton()->texture_drawable_create(width, height, (RS::TextureDrawableFormat)format, base_color, mipmaps);
		RS::get_singleton()->texture_replace(texture, new_texture);
	} else {
		texture = RS::get_singleton()->texture_drawable_create(width, height, (RS::TextureDrawableFormat)format, base_color, mipmaps);
	}
}

// Setup basic parameters on the Drawable Texture
void DrawableTexture2D::setup(int p_width, int p_height, DrawableFormat p_format, const Color &p_color, bool p_use_mipmaps) {
	ERR_FAIL_COND_MSG(p_width <= 0 || p_width > 16384, "Texture dimensions have to be in the 1 to 16384 range.");
	ERR_FAIL_COND_MSG(p_height <= 0 || p_height > 16384, "Texture dimensions have to be in the 1 to 16384 range.");
	width = p_width;
	height = p_height;
	format = p_format;
	mipmaps = p_use_mipmaps;
	base_color = p_color;
	_initialize();
	notify_property_list_changed();
	emit_changed();
}

void DrawableTexture2D::set_width(int p_width) {
	ERR_FAIL_COND_MSG(p_width <= 0 || p_width > 16384, "Texture dimensions have to be in the 1 to 16384 range.");
	if (width == p_width) {
		return;
	}
	width = p_width;
	//_initialize();
	notify_property_list_changed();
	emit_changed();
}

int DrawableTexture2D::get_width() const {
	return width;
}

void DrawableTexture2D::set_height(int p_height) {
	ERR_FAIL_COND_MSG(p_height <= 0 || p_height > 16384, "Texture dimensions have to be in the 1 to 16384 range.");
	if (height == p_height) {
		return;
	}
	height = p_height;
	//_initialize();
	notify_property_list_changed();
	emit_changed();
}

int DrawableTexture2D::get_height() const {
	return height;
}

void DrawableTexture2D::set_format(DrawableFormat p_format) {
	if (format == p_format) {
		return;
	}
	format = p_format;
	//_initialize();
	notify_property_list_changed();
	emit_changed();
}

DrawableTexture2D::DrawableFormat DrawableTexture2D::get_format() const {
	return format;
}

void DrawableTexture2D::set_use_mipmaps(bool p_mipmaps) {
	if (mipmaps == p_mipmaps) {
		return;
	}
	mipmaps = p_mipmaps;
	//_initialize();
	notify_property_list_changed();
	emit_changed();
}

bool DrawableTexture2D::get_use_mipmaps() const {
	return mipmaps;
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
	if ((width | height) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, Size2(width, height)), texture, false, p_modulate, p_transpose);
}

void DrawableTexture2D::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	if ((width | height) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, texture, p_tile, p_modulate, p_transpose);
}

void DrawableTexture2D::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	if ((width | height) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, texture, p_src_rect, p_modulate, p_transpose, p_clip_uv);
}

// Perform a blit operation from the given source to the given rect on self.
void DrawableTexture2D::blit_rect(const Rect2i p_rect, const Ref<Texture2D> &p_source, const Color &p_modulate, int p_mipmap, const Ref<Material> &p_material) {
	// Use user Shader if exists.
	RID material = default_material;
	if (p_material.is_valid()) {
		material = p_material->get_rid();
		if (p_material->get_shader_mode() != Shader::MODE_TEXTURE_BLIT) {
			WARN_PRINT("ShaderMaterial passed to blit_rect() is not a texture_blit shader. Using default instead.");
		}
	}

	// Rendering server expects textureParameters as a TypedArray[RID]
	Array textures;
	textures.push_back(texture);

	if (p_source.is_valid()) {
		ERR_FAIL_COND_MSG(texture == p_source->get_rid(), "Cannot use self as a source.");
	}
	Array src_textures;
	if (Ref<AtlasTexture>(p_source).is_valid()) {
		WARN_PRINT("AtlasTexture not supported as a source for blit_rect. Using default White.");
		src_textures.push_back(RID());
	} else {
		src_textures.push_back(p_source);
	}

	RS::get_singleton()->texture_drawable_blit_rect(textures, p_rect, material, p_modulate, src_textures, p_mipmap);
	notify_property_list_changed();
	//emit_changed();
}

// Perform a blit operation from the given sources to the given rect on self and extra targets
void DrawableTexture2D::blit_rect_multi(const Rect2i p_rect, const TypedArray<Texture2D> &p_sources, const TypedArray<DrawableTexture2D> &p_extra_targets, const Color &p_modulate, int p_mipmap, const Ref<Material> &p_material) {
	RID material = default_material;
	if (p_material.is_valid()) {
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
		textures.push_back(RID(p_extra_targets[i]));
		i += 1;
	}
	i = 0;
	Array src_textures;
	while (i < p_sources.size()) {
		if (Ref<AtlasTexture>(p_sources[i]).is_valid()) {
			WARN_PRINT("AtlasTexture not supported as a source for blit_rect. Using default White.");
			src_textures.push_back(RID());
		} else {
			src_textures.push_back(RID(p_sources[i]));
		}
		ERR_FAIL_COND_MSG(textures.has(RID(src_textures[i])), "Cannot use self as a source.");
		i += 1;
	}

	RS::get_singleton()->texture_drawable_blit_rect(textures, p_rect, material, p_modulate, src_textures, p_mipmap);
	notify_property_list_changed();
	//emit_changed();
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
	ClassDB::bind_method(D_METHOD("setup", "width", "height", "format", "color", "use_mipmaps"), &DrawableTexture2D::setup, DEFVAL(Color(1, 1, 1, 1)), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("blit_rect", "rect", "source", "modulate", "mipmap", "material"), &DrawableTexture2D::blit_rect, DEFVAL(Color(1, 1, 1, 1)), DEFVAL(0), DEFVAL(Ref<Material>()));
	ClassDB::bind_method(D_METHOD("blit_rect_multi", "rect", "sources", "extra_targets", "modulate", "mipmap", "material"), &DrawableTexture2D::blit_rect_multi, DEFVAL(Color(1, 1, 1, 1)), DEFVAL(0), DEFVAL(Ref<Material>()));
	ClassDB::bind_method(D_METHOD("generate_mipmaps"), &DrawableTexture2D::generate_mipmaps);
	//ClassDB::bind_method(D_METHOD("set_width", "width"), &DrawableTexture2D::set_width);
	//ClassDB::bind_method(D_METHOD("set_height", "height"), &DrawableTexture2D::set_height);
	//ClassDB::bind_method(D_METHOD("set_format", "format"), &DrawableTexture2D::set_format);
	//ClassDB::bind_method(D_METHOD("get_format"), &DrawableTexture2D::get_format);
	//ClassDB::bind_method(D_METHOD("set_use_mipmaps", "use_mipmaps"), &DrawableTexture2D::set_use_mipmaps);
	//ClassDB::bind_method(D_METHOD("get_use_mipmaps"), &DrawableTexture2D::get_use_mipmaps);

	BIND_ENUM_CONSTANT(DRAWABLE_FORMAT_RGBA8);
	BIND_ENUM_CONSTANT(DRAWABLE_FORMAT_RGBA8_SRGB);
	BIND_ENUM_CONSTANT(DRAWABLE_FORMAT_RGBAH);
	BIND_ENUM_CONSTANT(DRAWABLE_FORMAT_RGBAF);

	//ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,2048,or_greater,suffix:px"), "set_width", "get_width");
	//ADD_PROPERTY(PropertyInfo(Variant::INT, "height", PROPERTY_HINT_RANGE, "1,2048,or_greater,suffix:px"), "set_height", "get_height");
	//ADD_PROPERTY(PropertyInfo(Variant::INT, "format", PROPERTY_HINT_ENUM, "RGBA8, RBGA8_SRGB, RGBAH, RGBAF"), "set_format", "get_format");
	//ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_mipmaps"), "set_use_mipmaps", "get_use_mipmaps");
}
