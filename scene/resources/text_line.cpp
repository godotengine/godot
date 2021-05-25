/*************************************************************************/
/*  text_line.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "text_line.h"

void TextLine::_bind_methods() {
	ClassDB::bind_method(D_METHOD("clear"), &TextLine::clear);

	ClassDB::bind_method(D_METHOD("set_direction", "direction"), &TextLine::set_direction);
	ClassDB::bind_method(D_METHOD("get_direction"), &TextLine::get_direction);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "direction", PROPERTY_HINT_ENUM, "Auto,Light-to-right,Right-to-left"), "set_direction", "get_direction");

	ClassDB::bind_method(D_METHOD("set_orientation", "orientation"), &TextLine::set_orientation);
	ClassDB::bind_method(D_METHOD("get_orientation"), &TextLine::get_orientation);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "orientation", PROPERTY_HINT_ENUM, "Horizontal,Orientation"), "set_orientation", "get_orientation");

	ClassDB::bind_method(D_METHOD("set_preserve_invalid", "enabled"), &TextLine::set_preserve_invalid);
	ClassDB::bind_method(D_METHOD("get_preserve_invalid"), &TextLine::get_preserve_invalid);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "preserve_invalid"), "set_preserve_invalid", "get_preserve_invalid");

	ClassDB::bind_method(D_METHOD("set_preserve_control", "enabled"), &TextLine::set_preserve_control);
	ClassDB::bind_method(D_METHOD("get_preserve_control"), &TextLine::get_preserve_control);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "preserve_control"), "set_preserve_control", "get_preserve_control");

	ClassDB::bind_method(D_METHOD("set_bidi_override", "override"), &TextLine::_set_bidi_override);

	ClassDB::bind_method(D_METHOD("add_string", "text", "fonts", "size", "opentype_features", "language"), &TextLine::add_string, DEFVAL(Dictionary()), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("add_object", "key", "size", "inline_align", "length"), &TextLine::add_object, DEFVAL(VALIGN_CENTER), DEFVAL(1));
	ClassDB::bind_method(D_METHOD("resize_object", "key", "size", "inline_align"), &TextLine::resize_object, DEFVAL(VALIGN_CENTER));

	ClassDB::bind_method(D_METHOD("set_width", "width"), &TextLine::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &TextLine::get_width);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width"), "set_width", "get_width");

	ClassDB::bind_method(D_METHOD("set_align", "align"), &TextLine::set_align);
	ClassDB::bind_method(D_METHOD("get_align"), &TextLine::get_align);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "align", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_align", "get_align");

	ClassDB::bind_method(D_METHOD("tab_align", "tab_stops"), &TextLine::tab_align);

	ClassDB::bind_method(D_METHOD("set_flags", "flags"), &TextLine::set_flags);
	ClassDB::bind_method(D_METHOD("get_flags"), &TextLine::get_flags);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "flags", PROPERTY_HINT_FLAGS, "Kashida Justify,Word Justify,Trim Edge Spaces After Justify,Justify Only After Last Tab"), "set_flags", "get_flags");

	ClassDB::bind_method(D_METHOD("get_objects"), &TextLine::get_objects);
	ClassDB::bind_method(D_METHOD("get_object_rect", "key"), &TextLine::get_object_rect);

	ClassDB::bind_method(D_METHOD("get_size"), &TextLine::get_size);

	ClassDB::bind_method(D_METHOD("get_rid"), &TextLine::get_rid);

	ClassDB::bind_method(D_METHOD("get_line_ascent"), &TextLine::get_line_ascent);
	ClassDB::bind_method(D_METHOD("get_line_descent"), &TextLine::get_line_descent);
	ClassDB::bind_method(D_METHOD("get_line_width"), &TextLine::get_line_width);
	ClassDB::bind_method(D_METHOD("get_line_underline_position"), &TextLine::get_line_underline_position);
	ClassDB::bind_method(D_METHOD("get_line_underline_thickness"), &TextLine::get_line_underline_thickness);

	ClassDB::bind_method(D_METHOD("draw", "canvas", "pos", "color"), &TextLine::draw, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_outline", "canvas", "pos", "outline_size", "color"), &TextLine::draw_outline, DEFVAL(1), DEFVAL(Color(1, 1, 1)));

	ClassDB::bind_method(D_METHOD("hit_test", "coords"), &TextLine::hit_test);
}

void TextLine::_shape() {
	if (dirty) {
		if (!tab_stops.is_empty()) {
			TS->shaped_text_tab_align(rid, tab_stops);
		}
		if (align == HALIGN_FILL) {
			TS->shaped_text_fit_to_width(rid, width, flags);
		}
		dirty = false;
	}
}

RID TextLine::get_rid() const {
	return rid;
}

void TextLine::clear() {
	TS->shaped_text_clear(rid);
	spacing_top = 0;
	spacing_bottom = 0;
}

void TextLine::set_preserve_invalid(bool p_enabled) {
	TS->shaped_text_set_preserve_invalid(rid, p_enabled);
	dirty = true;
}

bool TextLine::get_preserve_invalid() const {
	return TS->shaped_text_get_preserve_invalid(rid);
}

void TextLine::set_preserve_control(bool p_enabled) {
	TS->shaped_text_set_preserve_control(rid, p_enabled);
	dirty = true;
}

bool TextLine::get_preserve_control() const {
	return TS->shaped_text_get_preserve_control(rid);
}

void TextLine::set_direction(TextServer::Direction p_direction) {
	TS->shaped_text_set_direction(rid, p_direction);
	dirty = true;
}

TextServer::Direction TextLine::get_direction() const {
	return TS->shaped_text_get_direction(rid);
}

void TextLine::set_orientation(TextServer::Orientation p_orientation) {
	TS->shaped_text_set_orientation(rid, p_orientation);
	dirty = true;
}

TextServer::Orientation TextLine::get_orientation() const {
	return TS->shaped_text_get_orientation(rid);
}

void TextLine::_set_bidi_override(const Array &p_override) {
	Vector<Vector2i> overrides;
	for (int i = 0; i < p_override.size(); i++) {
		overrides.push_back(p_override[i]);
	}
	set_bidi_override(overrides);
}

void TextLine::set_bidi_override(const Vector<Vector2i> &p_override) {
	TS->shaped_text_set_bidi_override(rid, p_override);
	dirty = true;
}

bool TextLine::add_string(const String &p_text, const Ref<Font> &p_fonts, int p_size, const Dictionary &p_opentype_features, const String &p_language) {
	ERR_FAIL_COND_V(p_fonts.is_null(), false);
	bool res = TS->shaped_text_add_string(rid, p_text, p_fonts->get_rids(), p_size, p_opentype_features, p_language);
	spacing_top = p_fonts->get_spacing(Font::SPACING_TOP);
	spacing_bottom = p_fonts->get_spacing(Font::SPACING_BOTTOM);
	dirty = true;
	return res;
}

bool TextLine::add_object(Variant p_key, const Size2 &p_size, VAlign p_inline_align, int p_length) {
	bool res = TS->shaped_text_add_object(rid, p_key, p_size, p_inline_align, p_length);
	dirty = true;
	return res;
}

bool TextLine::resize_object(Variant p_key, const Size2 &p_size, VAlign p_inline_align) {
	const_cast<TextLine *>(this)->_shape();
	return TS->shaped_text_resize_object(rid, p_key, p_size, p_inline_align);
}

Array TextLine::get_objects() const {
	return TS->shaped_text_get_objects(rid);
}

Rect2 TextLine::get_object_rect(Variant p_key) const {
	return TS->shaped_text_get_object_rect(rid, p_key);
}

void TextLine::set_align(HAlign p_align) {
	if (align != p_align) {
		if (align == HALIGN_FILL || p_align == HALIGN_FILL) {
			align = p_align;
			dirty = true;
		} else {
			align = p_align;
		}
	}
}

HAlign TextLine::get_align() const {
	return align;
}

void TextLine::tab_align(const Vector<float> &p_tab_stops) {
	tab_stops = p_tab_stops;
	dirty = true;
}

void TextLine::set_flags(uint8_t p_flags) {
	if (flags != p_flags) {
		flags = p_flags;
		dirty = true;
	}
}

uint8_t TextLine::get_flags() const {
	return flags;
}

void TextLine::set_width(float p_width) {
	width = p_width;
	if (align == HALIGN_FILL) {
		dirty = true;
	}
}

float TextLine::get_width() const {
	return width;
}

Size2 TextLine::get_size() const {
	const_cast<TextLine *>(this)->_shape();
	if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
		return Size2(TS->shaped_text_get_size(rid).x, TS->shaped_text_get_size(rid).y + spacing_top + spacing_bottom);
	} else {
		return Size2(TS->shaped_text_get_size(rid).x + spacing_top + spacing_bottom, TS->shaped_text_get_size(rid).y);
	}
}

float TextLine::get_line_ascent() const {
	const_cast<TextLine *>(this)->_shape();
	return TS->shaped_text_get_ascent(rid) + spacing_top;
}

float TextLine::get_line_descent() const {
	const_cast<TextLine *>(this)->_shape();
	return TS->shaped_text_get_descent(rid) + spacing_bottom;
}

float TextLine::get_line_width() const {
	const_cast<TextLine *>(this)->_shape();
	return TS->shaped_text_get_width(rid);
}

float TextLine::get_line_underline_position() const {
	const_cast<TextLine *>(this)->_shape();
	return TS->shaped_text_get_underline_position(rid);
}

float TextLine::get_line_underline_thickness() const {
	const_cast<TextLine *>(this)->_shape();
	return TS->shaped_text_get_underline_thickness(rid);
}

void TextLine::draw(RID p_canvas, const Vector2 &p_pos, const Color &p_color) const {
	const_cast<TextLine *>(this)->_shape();

	Vector2 ofs = p_pos;

	float length = TS->shaped_text_get_width(rid);
	if (width > 0) {
		switch (align) {
			case HALIGN_FILL:
			case HALIGN_LEFT:
				break;
			case HALIGN_CENTER: {
				if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
					ofs.x += Math::floor((width - length) / 2.0);
				} else {
					ofs.y += Math::floor((width - length) / 2.0);
				}
			} break;
			case HALIGN_RIGHT: {
				if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
					ofs.x += width - length;
				} else {
					ofs.y += width - length;
				}
			} break;
		}
	}

	float clip_l;
	if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(rid) + spacing_top;
		clip_l = MAX(0, p_pos.x - ofs.x);
	} else {
		ofs.x += TS->shaped_text_get_ascent(rid) + spacing_top;
		clip_l = MAX(0, p_pos.y - ofs.y);
	}
	return TS->shaped_text_draw(rid, p_canvas, ofs, clip_l, clip_l + width, p_color);
}

void TextLine::draw_outline(RID p_canvas, const Vector2 &p_pos, int p_outline_size, const Color &p_color) const {
	const_cast<TextLine *>(this)->_shape();

	Vector2 ofs = p_pos;

	float length = TS->shaped_text_get_width(rid);
	if (width > 0) {
		switch (align) {
			case HALIGN_FILL:
			case HALIGN_LEFT:
				break;
			case HALIGN_CENTER: {
				if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
					ofs.x += Math::floor((width - length) / 2.0);
				} else {
					ofs.y += Math::floor((width - length) / 2.0);
				}
			} break;
			case HALIGN_RIGHT: {
				if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
					ofs.x += width - length;
				} else {
					ofs.y += width - length;
				}
			} break;
		}
	}

	float clip_l;
	if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(rid) + spacing_top;
		clip_l = MAX(0, p_pos.x - ofs.x);
	} else {
		ofs.x += TS->shaped_text_get_ascent(rid) + spacing_top;
		clip_l = MAX(0, p_pos.y - ofs.y);
	}
	return TS->shaped_text_draw_outline(rid, p_canvas, ofs, clip_l, clip_l + width, p_outline_size, p_color);
}

int TextLine::hit_test(float p_coords) const {
	const_cast<TextLine *>(this)->_shape();

	return TS->shaped_text_hit_test_position(rid, p_coords);
}

TextLine::TextLine(const String &p_text, const Ref<Font> &p_fonts, int p_size, const Dictionary &p_opentype_features, const String &p_language, TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	rid = TS->create_shaped_text(p_direction, p_orientation);
	spacing_top = p_fonts->get_spacing(Font::SPACING_TOP);
	spacing_bottom = p_fonts->get_spacing(Font::SPACING_BOTTOM);
	TS->shaped_text_add_string(rid, p_text, p_fonts->get_rids(), p_size, p_opentype_features, p_language);
}

TextLine::TextLine() {
	rid = TS->create_shaped_text();
}

TextLine::~TextLine() {
	TS->free(rid);
}
