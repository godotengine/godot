/*************************************************************************/
/*  text_paragraph.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "scene/resources/text_paragraph.h"

void TextParagraph::_bind_methods() {
	ClassDB::bind_method(D_METHOD("clear"), &TextParagraph::clear);

	ClassDB::bind_method(D_METHOD("set_direction", "direction"), &TextParagraph::set_direction);
	ClassDB::bind_method(D_METHOD("get_direction"), &TextParagraph::get_direction);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "direction", PROPERTY_HINT_ENUM, "Auto,Light-to-right,Right-to-left"), "set_direction", "get_direction");

	ClassDB::bind_method(D_METHOD("set_orientation", "orientation"), &TextParagraph::set_orientation);
	ClassDB::bind_method(D_METHOD("get_orientation"), &TextParagraph::get_orientation);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "orientation", PROPERTY_HINT_ENUM, "Horizontal,Orientation"), "set_orientation", "get_orientation");

	ClassDB::bind_method(D_METHOD("set_preserve_invalid", "enabled"), &TextParagraph::set_preserve_invalid);
	ClassDB::bind_method(D_METHOD("get_preserve_invalid"), &TextParagraph::get_preserve_invalid);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "preserve_invalid"), "set_preserve_invalid", "get_preserve_invalid");

	ClassDB::bind_method(D_METHOD("set_preserve_control", "enabled"), &TextParagraph::set_preserve_control);
	ClassDB::bind_method(D_METHOD("get_preserve_control"), &TextParagraph::get_preserve_control);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "preserve_control"), "set_preserve_control", "get_preserve_control");

	ClassDB::bind_method(D_METHOD("set_bidi_override", "override"), &TextParagraph::_set_bidi_override);

	ClassDB::bind_method(D_METHOD("add_string", "text", "fonts", "size", "opentype_features", "language"), &TextParagraph::add_string, DEFVAL(Dictionary()), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("add_object", "key", "size", "inline_align", "length"), &TextParagraph::add_object, DEFVAL(VALIGN_CENTER), DEFVAL(1));
	ClassDB::bind_method(D_METHOD("resize_object", "key", "size", "inline_align"), &TextParagraph::resize_object, DEFVAL(VALIGN_CENTER));

	ClassDB::bind_method(D_METHOD("set_align", "align"), &TextParagraph::set_align);
	ClassDB::bind_method(D_METHOD("get_align"), &TextParagraph::get_align);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "align", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_align", "get_align");

	ClassDB::bind_method(D_METHOD("tab_align", "tab_stops"), &TextParagraph::tab_align);

	ClassDB::bind_method(D_METHOD("set_flags", "flags"), &TextParagraph::set_flags);
	ClassDB::bind_method(D_METHOD("get_flags"), &TextParagraph::get_flags);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "flags", PROPERTY_HINT_FLAGS, "Kashida justification,Word justification,Trim edge spaces after justification,Justification only after last tab,Break mandatory,Break words,Break graphemes"), "set_flags", "get_flags");

	ClassDB::bind_method(D_METHOD("set_width", "width"), &TextParagraph::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &TextParagraph::get_width);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width"), "set_width", "get_width");

	ClassDB::bind_method(D_METHOD("get_non_wraped_size"), &TextParagraph::get_non_wraped_size);
	ClassDB::bind_method(D_METHOD("get_size"), &TextParagraph::get_size);

	ClassDB::bind_method(D_METHOD("get_rid"), &TextParagraph::get_rid);
	ClassDB::bind_method(D_METHOD("get_line_rid", "line"), &TextParagraph::get_line_rid);

	ClassDB::bind_method(D_METHOD("get_line_count"), &TextParagraph::get_line_count);

	ClassDB::bind_method(D_METHOD("get_line_objects", "line"), &TextParagraph::get_line_objects);
	ClassDB::bind_method(D_METHOD("get_line_object_rect", "line", "key"), &TextParagraph::get_line_object_rect);
	ClassDB::bind_method(D_METHOD("get_line_size", "line"), &TextParagraph::get_line_size);
	ClassDB::bind_method(D_METHOD("get_line_range", "line"), &TextParagraph::get_line_range);
	ClassDB::bind_method(D_METHOD("get_line_ascent", "line"), &TextParagraph::get_line_ascent);
	ClassDB::bind_method(D_METHOD("get_line_descent", "line"), &TextParagraph::get_line_descent);
	ClassDB::bind_method(D_METHOD("get_line_width", "line"), &TextParagraph::get_line_width);
	ClassDB::bind_method(D_METHOD("get_line_underline_position", "line"), &TextParagraph::get_line_underline_position);
	ClassDB::bind_method(D_METHOD("get_line_underline_thickness", "line"), &TextParagraph::get_line_underline_thickness);

	ClassDB::bind_method(D_METHOD("draw", "canvas", "pos", "color"), &TextParagraph::draw, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_outline", "canvas", "outline_size", "color"), &TextParagraph::draw_outline, DEFVAL(1), DEFVAL(Color(1, 1, 1)));

	ClassDB::bind_method(D_METHOD("draw_line", "canvas", "pos", "line", "color"), &TextParagraph::draw_line, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_line_outline", "canvas", "pos", "line", "outline_size", "color"), &TextParagraph::draw_line_outline, DEFVAL(1), DEFVAL(Color(1, 1, 1)));

	ClassDB::bind_method(D_METHOD("hit_test", "coords"), &TextParagraph::hit_test);
}

void TextParagraph::_shape_lines() {
	if (dirty_lines) {
		for (int i = 0; i < lines.size(); i++) {
			TS->free(lines[i]);
		}
		lines.clear();

		if (!tab_stops.empty()) {
			TS->shaped_text_tab_align(rid, tab_stops);
		}

		Vector<Vector2i> line_breaks = TS->shaped_text_get_line_breaks(rid, width, 0, flags);
		for (int i = 0; i < line_breaks.size(); i++) {
			RID line = TS->shaped_text_substr(rid, line_breaks[i].x, line_breaks[i].y - line_breaks[i].x);
			if (!tab_stops.empty()) {
				TS->shaped_text_tab_align(line, tab_stops);
			}
			if (align == HALIGN_FILL && (line_breaks.size() == 1 || i < line_breaks.size() - 1)) {
				TS->shaped_text_fit_to_width(line, width, flags);
			}
			lines.push_back(line);
		}
		dirty_lines = false;
	}
}

RID TextParagraph::get_rid() const {
	return rid;
}

RID TextParagraph::get_line_rid(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines.size(), RID());
	return lines[p_line];
}

void TextParagraph::clear() {
	for (int i = 0; i < lines.size(); i++) {
		TS->free(lines[i]);
	}
	lines.clear();
	TS->shaped_text_clear(rid);
}

void TextParagraph::set_preserve_invalid(bool p_enabled) {
	TS->shaped_text_set_preserve_invalid(rid, p_enabled);
	dirty_lines = true;
}

bool TextParagraph::get_preserve_invalid() const {
	return TS->shaped_text_get_preserve_invalid(rid);
}

void TextParagraph::set_preserve_control(bool p_enabled) {
	TS->shaped_text_set_preserve_control(rid, p_enabled);
	dirty_lines = true;
}

bool TextParagraph::get_preserve_control() const {
	return TS->shaped_text_get_preserve_control(rid);
}

void TextParagraph::set_direction(TextServer::Direction p_direction) {
	TS->shaped_text_set_direction(rid, p_direction);
	dirty_lines = true;
}

TextServer::Direction TextParagraph::get_direction() const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	return TS->shaped_text_get_direction(rid);
}

void TextParagraph::set_orientation(TextServer::Orientation p_orientation) {
	TS->shaped_text_set_orientation(rid, p_orientation);
	dirty_lines = true;
}

TextServer::Orientation TextParagraph::get_orientation() const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	return TS->shaped_text_get_orientation(rid);
}

bool TextParagraph::add_string(const String &p_text, const Ref<Font> &p_fonts, int p_size, const Dictionary &p_opentype_features, const String &p_language) {
	bool res = TS->shaped_text_add_string(rid, p_text, p_fonts->get_rids(), p_size, p_opentype_features, p_language);
	dirty_lines = true;
	return res;
}

void TextParagraph::_set_bidi_override(const Array &p_override) {
	Vector<Vector2i> overrides;
	for (int i = 0; i < p_override.size(); i++) {
		overrides.push_back(p_override[i]);
	}
	set_bidi_override(overrides);
}

void TextParagraph::set_bidi_override(const Vector<Vector2i> &p_override) {
	TS->shaped_text_set_bidi_override(rid, p_override);
	dirty_lines = true;
}

bool TextParagraph::add_object(Variant p_key, const Size2 &p_size, VAlign p_inline_align, int p_length) {
	bool res = TS->shaped_text_add_object(rid, p_key, p_size, p_inline_align, p_length);
	dirty_lines = true;
	return res;
}

bool TextParagraph::resize_object(Variant p_key, const Size2 &p_size, VAlign p_inline_align) {
	bool res = TS->shaped_text_resize_object(rid, p_key, p_size, p_inline_align);
	dirty_lines = true;
	return res;
}

void TextParagraph::set_align(HAlign p_align) {
	if (align != p_align) {
		if (align == HALIGN_FILL || p_align == HALIGN_FILL) {
			align = p_align;
			dirty_lines = true;
		} else {
			align = p_align;
		}
	}
}

HAlign TextParagraph::get_align() const {
	return align;
}

void TextParagraph::tab_align(const Vector<float> &p_tab_stops) {
	tab_stops = p_tab_stops;
	dirty_lines = true;
}

void TextParagraph::set_flags(uint8_t p_flags) {
	if (flags != p_flags) {
		flags = p_flags;
		dirty_lines = true;
	}
}

uint8_t TextParagraph::get_flags() const {
	return flags;
}

void TextParagraph::set_width(float p_width) {
	if (width != p_width) {
		width = p_width;
		dirty_lines = true;
	}
}

float TextParagraph::get_width() const {
	return width;
}

Size2 TextParagraph::get_non_wraped_size() const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	return TS->shaped_text_get_size(rid);
}

Size2 TextParagraph::get_size() const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	Size2 size;
	for (int i = 0; i < lines.size(); i++) {
		Size2 lsize = TS->shaped_text_get_size(lines[i]);
		if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			size.x = MAX(size.x, lsize.x);
			size.y += lsize.y;
		} else {
			size.x += lsize.x;
			size.y = MAX(size.y, lsize.y);
		}
	}
	return size;
}

int TextParagraph::get_line_count() const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	return lines.size();
}

Array TextParagraph::get_line_objects(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines.size(), Array());
	return TS->shaped_text_get_objects(lines[p_line]);
}

Rect2 TextParagraph::get_line_object_rect(int p_line, Variant p_key) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines.size(), Rect2());
	Rect2 xrect = TS->shaped_text_get_object_rect(lines[p_line], p_key);
	for (int i = 0; i < p_line; i++) {
		Size2 lsize = TS->shaped_text_get_size(lines[i]);
		if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			xrect.position.y += lsize.y;
		} else {
			xrect.position.x += lsize.x;
		}
	}
	return xrect;
}

Size2 TextParagraph::get_line_size(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines.size(), Size2());
	return TS->shaped_text_get_size(lines[p_line]);
}

Vector2i TextParagraph::get_line_range(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines.size(), Vector2i());
	return TS->shaped_text_get_range(lines[p_line]);
}

float TextParagraph::get_line_ascent(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines.size(), 0.f);
	return TS->shaped_text_get_ascent(lines[p_line]);
}

float TextParagraph::get_line_descent(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines.size(), 0.f);
	return TS->shaped_text_get_descent(lines[p_line]);
}

float TextParagraph::get_line_width(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines.size(), 0.f);
	return TS->shaped_text_get_width(lines[p_line]);
}

float TextParagraph::get_line_underline_position(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines.size(), 0.f);
	return TS->shaped_text_get_underline_position(lines[p_line]);
}

float TextParagraph::get_line_underline_thickness(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines.size(), 0.f);
	return TS->shaped_text_get_underline_thickness(lines[p_line]);
}

void TextParagraph::draw(RID p_canvas, const Vector2 &p_pos, const Color &p_color) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	Vector2 ofs = p_pos;
	for (int i = 0; i < lines.size(); i++) {
		if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			ofs.x = p_pos.x;
			ofs.y += TS->shaped_text_get_ascent(lines[i]);
		} else {
			ofs.y = p_pos.y;
			ofs.x += TS->shaped_text_get_ascent(lines[i]);
		}
		float length = TS->shaped_text_get_width(lines[i]);
		if (width > 0) {
			switch (align) {
				case HALIGN_FILL:
				case HALIGN_LEFT:
					break;
				case HALIGN_CENTER: {
					if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
						ofs.x += Math::floor((width - length) / 2.0);
					} else {
						ofs.y += Math::floor((width - length) / 2.0);
					}
				} break;
				case HALIGN_RIGHT: {
					if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
						ofs.x += width - length;
					} else {
						ofs.y += width - length;
					}
				} break;
			}
		}
		float clip_l;
		if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			clip_l = MAX(0, p_pos.x - ofs.x);
		} else {
			clip_l = MAX(0, p_pos.y - ofs.y);
		}
		TS->shaped_text_draw(lines[i], p_canvas, ofs, clip_l, clip_l + width, p_color);
		if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			ofs.x = p_pos.x;
			ofs.y += TS->shaped_text_get_descent(lines[i]);
		} else {
			ofs.y = p_pos.y;
			ofs.x += TS->shaped_text_get_descent(lines[i]);
		}
	}
}

void TextParagraph::draw_outline(RID p_canvas, const Vector2 &p_pos, int p_outline_size, const Color &p_color) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	Vector2 ofs = p_pos;
	for (int i = 0; i < lines.size(); i++) {
		if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			ofs.x = p_pos.x;
			ofs.y += TS->shaped_text_get_ascent(lines[i]);
		} else {
			ofs.y = p_pos.y;
			ofs.x += TS->shaped_text_get_ascent(lines[i]);
		}
		float length = TS->shaped_text_get_width(lines[i]);
		if (width > 0) {
			switch (align) {
				case HALIGN_FILL:
				case HALIGN_LEFT:
					break;
				case HALIGN_CENTER: {
					if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
						ofs.x += Math::floor((width - length) / 2.0);
					} else {
						ofs.y += Math::floor((width - length) / 2.0);
					}
				} break;
				case HALIGN_RIGHT: {
					if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
						ofs.x += width - length;
					} else {
						ofs.y += width - length;
					}
				} break;
			}
		}
		float clip_l;
		if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			clip_l = MAX(0, p_pos.x - ofs.x);
		} else {
			clip_l = MAX(0, p_pos.y - ofs.y);
		}
		TS->shaped_text_draw_outline(lines[i], p_canvas, ofs, clip_l, clip_l + width, p_outline_size, p_color);
		if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			ofs.x = p_pos.x;
			ofs.y += TS->shaped_text_get_descent(lines[i]);
		} else {
			ofs.y = p_pos.y;
			ofs.x += TS->shaped_text_get_descent(lines[i]);
		}
	}
}

int TextParagraph::hit_test(const Point2 &p_coords) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	Vector2 ofs;
	if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
		if (ofs.y < 0)
			return 0;
	} else {
		if (ofs.x < 0)
			return 0;
	}
	for (int i = 0; i < lines.size(); i++) {
		if (TS->shaped_text_get_orientation(lines[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			if ((p_coords.y >= ofs.y) && (p_coords.y <= ofs.y + TS->shaped_text_get_size(lines[i]).y)) {
				return TS->shaped_text_hit_test_position(lines[i], p_coords.x);
			}
		} else {
			if ((p_coords.x >= ofs.x) && (p_coords.x <= ofs.x + TS->shaped_text_get_size(lines[i]).x)) {
				return TS->shaped_text_hit_test_position(lines[i], p_coords.y);
			}
		}
	}
	return TS->shaped_text_get_range(rid).y;
}

void TextParagraph::draw_line(RID p_canvas, const Vector2 &p_pos, int p_line, const Color &p_color) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND(p_line < 0 || p_line >= lines.size());

	Vector2 ofs = p_pos;
	if (TS->shaped_text_get_orientation(lines[p_line]) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(lines[p_line]);
	} else {
		ofs.x += TS->shaped_text_get_ascent(lines[p_line]);
	}
	return TS->shaped_text_draw(lines[p_line], p_canvas, ofs, -1, -1, p_color);
}

void TextParagraph::draw_line_outline(RID p_canvas, const Vector2 &p_pos, int p_line, int p_outline_size, const Color &p_color) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND(p_line < 0 || p_line >= lines.size());

	Vector2 ofs = p_pos;
	if (TS->shaped_text_get_orientation(lines[p_line]) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(lines[p_line]);
	} else {
		ofs.x += TS->shaped_text_get_ascent(lines[p_line]);
	}
	return TS->shaped_text_draw_outline(lines[p_line], p_canvas, ofs, -1, -1, p_outline_size, p_color);
}

TextParagraph::TextParagraph(const String &p_text, const Ref<Font> &p_fonts, int p_size, const Dictionary &p_opentype_features, const String &p_language, float p_width, TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	rid = TS->create_shaped_text(p_direction, p_orientation);
	TS->shaped_text_add_string(rid, p_text, p_fonts->get_rids(), p_size, p_opentype_features, p_language);
	width = p_width;
	dirty_lines = true;
}

TextParagraph::TextParagraph() {
	rid = TS->create_shaped_text();
}

TextParagraph::~TextParagraph() {
	for (int i = 0; i < lines.size(); i++) {
		TS->free(lines[i]);
	}
	lines.clear();
	TS->free(rid);
}
