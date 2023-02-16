/**************************************************************************/
/*  psd_layer.cpp                                                         */
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

#include "psd_layer.h"
#include <core\core_string_names.h>

int PSDLayer::get_width() const {
	
	if (psd.is_valid() && psd->get_texture_layer(layer).is_valid()) {
		return psd->get_texture_layer(layer)->get_image()->get_width();
	}

	return 1;
	
}

int PSDLayer::get_height() const {
	if (psd.is_valid() && psd->get_texture_layer(layer).is_valid()) {
		return psd->get_texture_layer(layer)->get_image()->get_height();
	}

	return 1;
}

RID PSDLayer::get_rid() const {
	if (psd.is_valid() && psd->get_texture_layer(layer).is_valid()) {
		return psd->get_texture_layer(layer)->get_rid();
	}

	return RID();
}

bool PSDLayer::has_alpha() const {

	if (psd.is_valid() && psd->get_texture_layer(layer).is_valid()) {
		return psd->get_texture_layer(layer)->has_alpha();
	}

	return false;
}

void PSDLayer::set_psd_texture(const Ref<PSDTexture>& p_psd) {
	if (psd == p_psd) {
		return;
	}
	
	if (psd.is_valid()) {
		psd->disconnect(CoreStringNames::get_singleton()->changed, callable_mp((Resource*)this, &PSDLayer::emit_changed));
	}

	psd = p_psd;

	if (psd.is_valid()) {
		psd->connect(CoreStringNames::get_singleton()->changed, callable_mp((Resource*)this, &PSDLayer::emit_changed));
	}


	emit_changed();
}

Ref<PSDTexture> PSDLayer::get_psd_texture() const {
	return psd;
}

void PSDLayer::set_layer(const String p_layer) {
	layer = p_layer;

	emit_changed();
}

String PSDLayer::get_layer() const {
	return layer;
}



void PSDLayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_psd_texture", "psd"), &PSDLayer::set_psd_texture);
	ClassDB::bind_method(D_METHOD("get_psd_texture"), &PSDLayer::get_psd_texture);

	ClassDB::bind_method(D_METHOD("set_layer", "layer"), &PSDLayer::set_layer);
	ClassDB::bind_method(D_METHOD("get_layer"), &PSDLayer::get_layer);


	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "psd", PROPERTY_HINT_RESOURCE_TYPE, "PSDTexture"), "set_psd_texture", "get_psd_texture");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "layer"), "set_layer", "get_layer");
}

void PSDLayer::draw(RID p_canvas_item, const Point2& p_pos, const Color& p_modulate, bool p_transpose) const {
	if (!psd.is_valid() || !psd->get_texture_layer(layer).is_valid()) {
		return;
	}

	psd->get_texture_layer(layer)->draw(p_canvas_item, p_pos, p_modulate, p_transpose);
}

void PSDLayer::draw_rect(RID p_canvas_item, const Rect2& p_rect, bool p_tile, const Color& p_modulate, bool p_transpose) const {
	if (!psd.is_valid() || !psd->get_texture_layer(layer).is_valid()) {
		return;
	}

	psd->get_texture_layer(layer)->draw_rect(p_canvas_item, p_rect, p_tile, p_modulate, p_transpose);
}

void PSDLayer::draw_rect_region(RID p_canvas_item, const Rect2& p_rect, const Rect2& p_src_rect, const Color& p_modulate, bool p_transpose, bool p_clip_uv) const {
	if (!psd.is_valid() || !psd->get_texture_layer(layer).is_valid()) {
		return;
	}

	psd->get_texture_layer(layer)->draw_rect_region(p_canvas_item, p_rect, p_src_rect, p_modulate, p_transpose, p_clip_uv);
}

bool PSDLayer::get_rect_region(const Rect2& p_rect, const Rect2& p_src_rect, Rect2& r_rect, Rect2& r_src_rect) const {
	if (!psd.is_valid() || !psd->get_texture_layer(layer).is_valid()) {
		return false;
	}

	return psd->get_texture_layer(layer)->get_rect_region(p_rect, p_src_rect, r_rect, r_src_rect);
}

bool PSDLayer::is_pixel_opaque(int p_x, int p_y) const {
	if (!psd.is_valid() || !psd->get_texture_layer(layer).is_valid()) {
		return false;
	}

	return psd->get_texture_layer(layer)->is_pixel_opaque(p_x, p_y);
}

Ref<Image> PSDLayer::get_image() const {
	if (!psd.is_valid() || !psd->get_texture_layer(layer).is_valid() || !psd->get_texture_layer(layer)->get_image().is_valid()) {
		return Ref<Image>();
	}

	return psd->get_texture_layer(layer)->get_image();
}

PSDLayer::PSDLayer() {}
