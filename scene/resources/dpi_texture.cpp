/**************************************************************************/
/*  dpi_texture.cpp                                                       */
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

#include "dpi_texture.h"

#include "scene/main/canvas_item.h"
#include "scene/main/viewport.h"
#include "scene/resources/bit_map.h"

#include "modules/modules_enabled.gen.h" // For svg.
#ifdef MODULE_SVG_ENABLED
#include "modules/svg/image_loader_svg.h"
#else
#include "core/io/image_loader.h"
#endif

Mutex DPITexture::mutex;
HashMap<double, DPITexture::ScalingLevel> DPITexture::scaling_levels;

void DPITexture::reference_scaling_level(double p_scale) {
	uint32_t oversampling = CLAMP(p_scale, 0.1, 100.0) * 64;
	if (oversampling == 64) {
		return;
	}
	double scale = double(oversampling) / 64.0;

	MutexLock lock(mutex);
	ScalingLevel *sl = scaling_levels.getptr(scale);
	if (sl) {
		sl->refcount++;
	} else {
		ScalingLevel new_sl;
		scaling_levels.insert(scale, new_sl);
	}
}

void DPITexture::unreference_scaling_level(double p_scale) {
	uint32_t oversampling = CLAMP(p_scale, 0.1, 100.0) * 64;
	if (oversampling == 64) {
		return;
	}
	double scale = double(oversampling) / 64.0;

	MutexLock lock(mutex);
	ScalingLevel *sl = scaling_levels.getptr(scale);
	if (sl) {
		sl->refcount--;
		if (sl->refcount == 0) {
			for (DPITexture *tx : sl->textures) {
				tx->_remove_scale(scale);
			}
			sl->textures.clear();
			scaling_levels.erase(scale);
		}
	}
}

Ref<DPITexture> DPITexture::create_from_string(const String &p_source, float p_scale, float p_saturation, const Dictionary &p_color_map) {
	Ref<DPITexture> dpi_texture;
	dpi_texture.instantiate();
	dpi_texture->update(p_source, p_scale, p_saturation, p_color_map);
	return dpi_texture;
}

void DPITexture::update(const String &p_source, float p_scale, float p_saturation, const Dictionary &p_color_map) {
	_block_emit_changed();
	set_source(p_source);
	set_base_scale(p_scale);
	set_saturation(p_saturation);
	set_color_map(p_color_map);
	_unblock_emit_changed();
}

void DPITexture::set_source(const String &p_source) {
	if (source == p_source) {
		return;
	}
	source = p_source;
	_update_texture();
}

String DPITexture::get_source() const {
	return source;
}

void DPITexture::set_base_scale(float p_scale) {
	p_scale = MAX(0.01, p_scale);
	if (base_scale == p_scale) {
		return;
	}

	base_scale = p_scale;
	_update_texture();
}

float DPITexture::get_base_scale() const {
	return base_scale;
}

void DPITexture::set_saturation(float p_saturation) {
	p_saturation = CLAMP(p_saturation, 0.0, 1.0);
	if (saturation == p_saturation) {
		return;
	}

	saturation = p_saturation;
	_update_texture();
}

float DPITexture::get_saturation() const {
	return saturation;
}

void DPITexture::set_color_map(const Dictionary &p_color_map) {
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

Dictionary DPITexture::get_color_map() const {
	return color_map;
}

void DPITexture::_remove_scale(double p_scale) {
	if (Math::is_equal_approx(p_scale, 1.0)) {
		return;
	}

	RID *rid = texture_cache.getptr(p_scale);
	if (rid) {
		if (rid->is_valid()) {
			RenderingServer::get_singleton()->free_rid(*rid);
		}
		texture_cache.erase(p_scale);
	}
}

RID DPITexture::_ensure_scale(double p_scale) const {
	uint32_t oversampling = CLAMP(p_scale, 0.1, 100.0) * 64;
	if (oversampling == 64) {
		if (base_texture.is_null()) {
			base_texture = _load_at_scale(p_scale, true);
		}
		return base_texture;
	}
	double scale = double(oversampling) / 64.0;

	RID *rid = texture_cache.getptr(scale);
	if (rid) {
		return *rid;
	}

	MutexLock lock(mutex);
	ScalingLevel *sl = scaling_levels.getptr(scale);
	ERR_FAIL_NULL_V_MSG(sl, RID(), "Invalid scaling level");
	sl->textures.insert(const_cast<DPITexture *>(this));

	RID new_rid = _load_at_scale(scale, false);
	texture_cache[scale] = new_rid;
	return new_rid;
}

RID DPITexture::_load_at_scale(double p_scale, bool p_set_size) const {
	Ref<Image> img;
	img.instantiate();
#ifdef MODULE_SVG_ENABLED
	const bool upsample = !Math::is_equal_approx(Math::round(p_scale * base_scale), p_scale * base_scale);

	Error err = ImageLoaderSVG::create_image_from_string(img, source, p_scale * base_scale, upsample, cmap);
	if (err != OK) {
		return RID();
	}
	if (saturation != 1.0) {
		img->adjust_bcs(1.0, 1.0, saturation);
	}
	img->fix_alpha_edges();
#else
	img = Image::create_empty(Math::round(16 * p_scale * base_scale), Math::round(16 * p_scale * base_scale), false, Image::FORMAT_RGBA8);
#endif

	Size2 current_size = size;
	if (p_set_size || size.is_zero_approx()) {
		size.x = img->get_width() / p_scale;
		base_size.x = size.x;
		if (size_override.x != 0) {
			size.x = size_override.x;
		}
		size.y = img->get_height() / p_scale;
		base_size.y = size.y;
		if (size_override.y != 0) {
			size.y = size_override.y;
		}
		current_size = size;
	}
	if (current_size.is_zero_approx()) {
		current_size.x = img->get_width();
		current_size.y = img->get_height();
	}

	RID rid = RenderingServer::get_singleton()->texture_2d_create(img);
	RenderingServer::get_singleton()->texture_set_size_override(rid, current_size.x, current_size.y);
	return rid;
}

void DPITexture::_clear() {
	for (KeyValue<double, RID> &tx : texture_cache) {
		if (tx.value.is_valid()) {
			RenderingServer::get_singleton()->free_rid(tx.value);
		}
	}
	texture_cache.clear();
	if (base_texture.is_valid()) {
		RenderingServer::get_singleton()->free_rid(base_texture);
	}
	base_texture = RID();
	alpha_cache.unref();
}

void DPITexture::_update_texture() {
	_clear();
	emit_changed();
}

Ref<Image> DPITexture::get_image() const {
	RID rid = _ensure_scale(1.0);
	if (rid.is_valid()) {
		return RenderingServer::get_singleton()->texture_2d_get(rid);
	} else {
		return Ref<Image>();
	}
}

int DPITexture::get_width() const {
	_ensure_scale(1.0);
	return size.x;
}

int DPITexture::get_height() const {
	_ensure_scale(1.0);
	return size.y;
}

RID DPITexture::get_rid() const {
	return _ensure_scale(1.0);
}

bool DPITexture::has_alpha() const {
	return true;
}

RID DPITexture::get_scaled_rid() const {
	double scale = TextServer::get_current_drawn_item_oversampling();
	if (scale == 0.0) {
		scale = 1.0;
	}
	return _ensure_scale(scale);
}

void DPITexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	RID rid = get_scaled_rid(); // Note: call `get_scaled_rid` before using `size` to ensure it is loaded.
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, size), rid, false, p_modulate, p_transpose);
}

void DPITexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, get_scaled_rid(), p_tile, p_modulate, p_transpose);
}

void DPITexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, get_scaled_rid(), p_src_rect, p_modulate, p_transpose, p_clip_uv);
}

bool DPITexture::is_pixel_opaque(int p_x, int p_y) const {
	if (alpha_cache.is_null()) {
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

		x = CLAMP(x, 0, aw - 1);
		y = CLAMP(y, 0, ah - 1);

		return alpha_cache->get_bit(x, y);
	}

	return true;
}

void DPITexture::set_size_override(const Size2i &p_size) {
	if (size_override == p_size) {
		return;
	}
	size_override = p_size;
	if (size_override.x == 0 || size_override.y == 0) {
		_ensure_scale(1.0);
		size = base_size;
	}
	if (size_override.x != 0) {
		size.x = size_override.x;
	}
	if (size_override.y != 0) {
		size.y = size_override.y;
	}
	for (KeyValue<double, RID> &tx : texture_cache) {
		if (tx.value.is_valid()) {
			RenderingServer::get_singleton()->texture_set_size_override(tx.value, size.x, size.y);
		}
	}
	if (base_texture.is_valid()) {
		RenderingServer::get_singleton()->texture_set_size_override(base_texture, size.x, size.y);
	}

	emit_changed();
}

void DPITexture::_bind_methods() {
	ClassDB::bind_static_method("DPITexture", D_METHOD("create_from_string", "source", "scale", "saturation", "color_map"), &DPITexture::create_from_string, DEFVAL(1.0), DEFVAL(1.0), DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("update", "source", "scale", "saturation", "color_map"), &DPITexture::update, DEFVAL(1.0), DEFVAL(1.0), DEFVAL(Dictionary()));

	ClassDB::bind_method(D_METHOD("set_source", "source"), &DPITexture::set_source);
	ClassDB::bind_method(D_METHOD("get_source"), &DPITexture::get_source);
	ClassDB::bind_method(D_METHOD("set_base_scale", "base_scale"), &DPITexture::set_base_scale);
	ClassDB::bind_method(D_METHOD("get_base_scale"), &DPITexture::get_base_scale);
	ClassDB::bind_method(D_METHOD("set_saturation", "saturation"), &DPITexture::set_saturation);
	ClassDB::bind_method(D_METHOD("get_saturation"), &DPITexture::get_saturation);
	ClassDB::bind_method(D_METHOD("set_color_map", "color_map"), &DPITexture::set_color_map);
	ClassDB::bind_method(D_METHOD("get_color_map"), &DPITexture::get_color_map);
	ClassDB::bind_method(D_METHOD("set_size_override", "size"), &DPITexture::set_size_override);
	ClassDB::bind_method(D_METHOD("get_scaled_rid"), &DPITexture::get_scaled_rid);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "_source", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_STORAGE), "set_source", "get_source");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "base_scale", PROPERTY_HINT_RANGE, "0.01,10.0,0.01"), "set_base_scale", "get_base_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "saturation", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_saturation", "get_saturation");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "color_map", PROPERTY_HINT_DICTIONARY_TYPE, "Color;Color"), "set_color_map", "get_color_map");
}

DPITexture::~DPITexture() {
	_clear();

	MutexLock lock(mutex);
	for (KeyValue<double, ScalingLevel> &sl : scaling_levels) {
		sl.value.textures.erase(this);
	}
}
