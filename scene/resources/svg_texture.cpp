/**************************************************************************/
/*  svg_texture.cpp                                                       */
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

#include "svg_texture.h"

#include "core/io/image_loader.h"
#include "scene/resources/bit_map.h"
#include "scene/resources/placeholder_textures.h"

#include "modules/modules_enabled.gen.h" // For svg.
#ifdef MODULE_SVG_ENABLED
#include "modules/svg/image_loader_svg.h"
#endif

Ref<SVGTexture> SVGTexture::create_from_string(const String &p_source, float p_scale, float p_saturation, const Dictionary &p_color_map) {
	Ref<SVGTexture> svg_texture;
	svg_texture.instantiate();
	svg_texture->set_source(p_source);
	svg_texture->set_base_scale(p_scale);
	svg_texture->set_saturation(p_saturation);
	svg_texture->set_color_map(p_color_map);
	return svg_texture;
}

void SVGTexture::set_source(const String &p_source) {
	if (source == p_source) {
		return;
	}
	source = p_source;
	_update_texture();
}

String SVGTexture::get_source() const {
	return source;
}

void SVGTexture::set_base_scale(float p_scale) {
	if (base_scale == p_scale) {
		return;
	}
	ERR_FAIL_COND(p_scale <= 0.0);

	base_scale = p_scale;
	_update_texture();
}

float SVGTexture::get_base_scale() const {
	return base_scale;
}

void SVGTexture::set_saturation(float p_saturation) {
	if (saturation == p_saturation) {
		return;
	}

	saturation = p_saturation;
	_update_texture();
}

float SVGTexture::get_saturation() const {
	return saturation;
}

void SVGTexture::set_color_map(const Dictionary &p_color_map) {
	if (color_map == p_color_map) {
		return;
	}
	color_map = p_color_map;
	cmap.clear();
	for (const Variant *E = color_map.next(); E; E = color_map.next(E)) {
		cmap[*E] = color_map[*E];
	}
	_update_texture();
}

Dictionary SVGTexture::get_color_map() const {
	return color_map;
}

RID SVGTexture::_ensure_scale(float p_scale, bool p_set_size) const {
	const Ref<TextureCache> *cache_entry = scaled_textures.getptr(p_scale);
	if (cache_entry && cache_entry->is_valid()) {
		return (*cache_entry)->scaled_texture;
	}

	if (source.is_empty()) {
		return RID();
	}

	Ref<Image> img = memnew(Image);
#ifdef MODULE_SVG_ENABLED
	const bool upsample = !Math::is_equal_approx(Math::round(p_scale), p_scale);

	Error err = ImageLoaderSVG::create_image_from_string(img, source, p_scale, upsample, cmap);
	ERR_FAIL_COND_V_MSG(err != OK, RID(), "Failed generating icon, unsupported or invalid SVG data in default theme.");
#else
	img = Image::create_empty(Math::round(16 * p_scale), Math::round(16 * p_scale), false, Image::FORMAT_RGBA8);
#endif
	if (saturation != 1.0) {
		img->adjust_bcs(1.0, 1.0, saturation);
	}

	if (p_set_size) {
		size.x = img->get_width();
		if (size_override.x != 0) {
			size.x = size_override.x;
		}
		size.y = img->get_height();
		if (size_override.y != 0) {
			size.y = size_override.y;
		}
	}

	Ref<TextureCache> new_cache_entry;
	new_cache_entry.instantiate();
	new_cache_entry->scaled_texture = RenderingServer::get_singleton()->texture_2d_create(img);
	if (size_override.x != 0 || size_override.y != 0) {
		RenderingServer::get_singleton()->texture_set_size_override(new_cache_entry->scaled_texture, size.x, size.y);
	}
	scaled_textures.insert(p_scale, new_cache_entry);

	return new_cache_entry->scaled_texture;
}

void SVGTexture::_update_texture() {
	scaled_textures.clear();
	alpha_cache.unref();
	_ensure_scale(base_scale, true);

	emit_changed();
}

Ref<Image> SVGTexture::get_image() const {
	RID rid = _ensure_scale(base_scale);
	if (rid.is_valid()) {
		return RenderingServer::get_singleton()->texture_2d_get(rid);
	} else {
		return Ref<Image>();
	}
}

int SVGTexture::get_width() const {
	return size.x;
}

int SVGTexture::get_height() const {
	return size.y;
}

RID SVGTexture::get_rid() const {
	return _ensure_scale(base_scale);
}

RID SVGTexture::get_scaled_rid(float p_scale) const {
	return _ensure_scale(base_scale * p_scale);
}

bool SVGTexture::has_alpha() const {
	return true;
}

void SVGTexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	float oversampling = RenderingServer::get_singleton()->canvas_item_get_oversampling_factor(p_canvas_item);
	RID rid = _ensure_scale(base_scale * oversampling);

	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, size), rid, false, p_modulate, p_transpose);
}

void SVGTexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	float oversampling = RenderingServer::get_singleton()->canvas_item_get_oversampling_factor(p_canvas_item);
	RID rid = _ensure_scale(base_scale * oversampling);

	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, rid, p_tile, p_modulate, p_transpose);
}

void SVGTexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	float oversampling = RenderingServer::get_singleton()->canvas_item_get_oversampling_factor(p_canvas_item);
	RID rid = _ensure_scale(base_scale * oversampling);

	RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, rid, Rect2(p_src_rect.position * oversampling, p_src_rect.size * oversampling), p_modulate, p_transpose, p_clip_uv);
}

bool SVGTexture::is_pixel_opaque(int p_x, int p_y) const {
	if (!alpha_cache.is_valid()) {
		Ref<Image> img = get_image();
		if (img.is_valid()) {
			alpha_cache.instantiate();
			alpha_cache->create_from_image_alpha(img);
		}
	}

	if (alpha_cache.is_valid()) {
		int aw = int(alpha_cache->get_size().width);
		int ah = int(alpha_cache->get_size().height);
		if (aw == 0 || ah == 0) {
			return true;
		}

		int x = p_x * aw / size.x;
		int y = p_y * ah / size.y;

		x = CLAMP(x, 0, aw);
		y = CLAMP(y, 0, ah);

		return alpha_cache->get_bit(x, y);
	}

	return true;
}

void SVGTexture::set_size_override(const Size2i &p_size) {
	if (size_override == p_size) {
		return;
	}
	size_override = p_size;
	if (size_override.x != 0) {
		size.x = size_override.x;
	}
	if (size_override.y != 0) {
		size.y = size_override.y;
	}
	for (LRUCache<float, Ref<TextureCache>>::Iterator E = scaled_textures.begin(); E; ++E) {
		if (E->second.is_valid()) {
			RenderingServer::get_singleton()->texture_set_size_override(E->second->scaled_texture, size.x, size.y);
		}
	}
}

void SVGTexture::_bind_methods() {
	ClassDB::bind_static_method("SVGTexture", D_METHOD("create_from_string", "source", "scale", "saturation", "color_map"), &SVGTexture::create_from_string, DEFVAL(1.0), DEFVAL(1.0), DEFVAL(Dictionary()));

	ClassDB::bind_method(D_METHOD("set_source", "source"), &SVGTexture::set_source);
	ClassDB::bind_method(D_METHOD("get_source"), &SVGTexture::get_source);
	ClassDB::bind_method(D_METHOD("set_base_scale", "base_scale"), &SVGTexture::set_base_scale);
	ClassDB::bind_method(D_METHOD("get_base_scale"), &SVGTexture::get_base_scale);
	ClassDB::bind_method(D_METHOD("set_saturation", "saturation"), &SVGTexture::set_saturation);
	ClassDB::bind_method(D_METHOD("get_saturation"), &SVGTexture::get_saturation);
	ClassDB::bind_method(D_METHOD("set_color_map", "color_map"), &SVGTexture::set_color_map);
	ClassDB::bind_method(D_METHOD("get_color_map"), &SVGTexture::get_color_map);

	ClassDB::bind_method(D_METHOD("get_scaled_rid", "scale"), &SVGTexture::get_scaled_rid);
	ClassDB::bind_method(D_METHOD("set_size_override", "size"), &SVGTexture::set_size_override);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "source", PROPERTY_HINT_MULTILINE_TEXT), "set_source", "get_source");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "base_scale", PROPERTY_HINT_RANGE, "0.5,10.0,0.01"), "set_base_scale", "get_base_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "saturation", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_saturation", "get_saturation");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "color_map"), "set_color_map", "get_color_map");
}

SVGTexture::SVGTexture() {
	scaled_textures.set_capacity(16);
}

SVGTexture::~SVGTexture() {
	scaled_textures.clear();
}
