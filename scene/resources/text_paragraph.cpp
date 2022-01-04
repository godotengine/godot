/*************************************************************************/
/*  text_paragraph.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

	ClassDB::bind_method(D_METHOD("set_custom_punctuation", "custom_punctuation"), &TextParagraph::set_custom_punctuation);
	ClassDB::bind_method(D_METHOD("get_custom_punctuation"), &TextParagraph::get_custom_punctuation);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "custom_punctuation"), "set_custom_punctuation", "get_custom_punctuation");

	ClassDB::bind_method(D_METHOD("set_orientation", "orientation"), &TextParagraph::set_orientation);
	ClassDB::bind_method(D_METHOD("get_orientation"), &TextParagraph::get_orientation);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "orientation", PROPERTY_HINT_ENUM, "Horizontal,Orientation"), "set_orientation", "get_orientation");

	ClassDB::bind_method(D_METHOD("set_preserve_invalid", "enabled"), &TextParagraph::set_preserve_invalid);
	ClassDB::bind_method(D_METHOD("get_preserve_invalid"), &TextParagraph::get_preserve_invalid);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "preserve_invalid"), "set_preserve_invalid", "get_preserve_invalid");

	ClassDB::bind_method(D_METHOD("set_preserve_control", "enabled"), &TextParagraph::set_preserve_control);
	ClassDB::bind_method(D_METHOD("get_preserve_control"), &TextParagraph::get_preserve_control);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "preserve_control"), "set_preserve_control", "get_preserve_control");

	ClassDB::bind_method(D_METHOD("set_bidi_override", "override"), &TextParagraph::set_bidi_override);

	ClassDB::bind_method(D_METHOD("set_dropcap", "text", "fonts", "size", "dropcap_margins", "opentype_features", "language"), &TextParagraph::set_dropcap, DEFVAL(Rect2()), DEFVAL(Dictionary()), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("clear_dropcap"), &TextParagraph::clear_dropcap);

	ClassDB::bind_method(D_METHOD("add_string", "text", "fonts", "size", "opentype_features", "language"), &TextParagraph::add_string, DEFVAL(Dictionary()), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("add_object", "key", "size", "inline_align", "length"), &TextParagraph::add_object, DEFVAL(INLINE_ALIGNMENT_CENTER), DEFVAL(1));
	ClassDB::bind_method(D_METHOD("resize_object", "key", "size", "inline_align"), &TextParagraph::resize_object, DEFVAL(INLINE_ALIGNMENT_CENTER));

	ClassDB::bind_method(D_METHOD("set_alignment", "alignment"), &TextParagraph::set_alignment);
	ClassDB::bind_method(D_METHOD("get_alignment"), &TextParagraph::get_alignment);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "alignment", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_alignment", "get_alignment");

	ClassDB::bind_method(D_METHOD("tab_align", "tab_stops"), &TextParagraph::tab_align);

	ClassDB::bind_method(D_METHOD("set_flags", "flags"), &TextParagraph::set_flags);
	ClassDB::bind_method(D_METHOD("get_flags"), &TextParagraph::get_flags);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "flags", PROPERTY_HINT_FLAGS, "Kashida Justify,Word Justify,Trim Edge Spaces After Justify,Justify Only After Last Tab,Break Mandatory,Break Words,Break Graphemes"), "set_flags", "get_flags");

	ClassDB::bind_method(D_METHOD("set_text_overrun_behavior", "overrun_behavior"), &TextParagraph::set_text_overrun_behavior);
	ClassDB::bind_method(D_METHOD("get_text_overrun_behavior"), &TextParagraph::get_text_overrun_behavior);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_overrun_behavior", PROPERTY_HINT_ENUM, "Trim Nothing,Trim Characters,Trim Words,Ellipsis,Word Ellipsis"), "set_text_overrun_behavior", "get_text_overrun_behavior");

	ClassDB::bind_method(D_METHOD("set_width", "width"), &TextParagraph::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &TextParagraph::get_width);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width"), "set_width", "get_width");

	ClassDB::bind_method(D_METHOD("get_non_wrapped_size"), &TextParagraph::get_non_wrapped_size);
	ClassDB::bind_method(D_METHOD("get_size"), &TextParagraph::get_size);

	ClassDB::bind_method(D_METHOD("get_rid"), &TextParagraph::get_rid);
	ClassDB::bind_method(D_METHOD("get_line_rid", "line"), &TextParagraph::get_line_rid);
	ClassDB::bind_method(D_METHOD("get_dropcap_rid"), &TextParagraph::get_dropcap_rid);

	ClassDB::bind_method(D_METHOD("get_line_count"), &TextParagraph::get_line_count);

	ClassDB::bind_method(D_METHOD("set_max_lines_visible", "max_lines_visible"), &TextParagraph::set_max_lines_visible);
	ClassDB::bind_method(D_METHOD("get_max_lines_visible"), &TextParagraph::get_max_lines_visible);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_lines_visible"), "set_max_lines_visible", "get_max_lines_visible");

	ClassDB::bind_method(D_METHOD("get_line_objects", "line"), &TextParagraph::get_line_objects);
	ClassDB::bind_method(D_METHOD("get_line_object_rect", "line", "key"), &TextParagraph::get_line_object_rect);
	ClassDB::bind_method(D_METHOD("get_line_size", "line"), &TextParagraph::get_line_size);
	ClassDB::bind_method(D_METHOD("get_line_range", "line"), &TextParagraph::get_line_range);
	ClassDB::bind_method(D_METHOD("get_line_ascent", "line"), &TextParagraph::get_line_ascent);
	ClassDB::bind_method(D_METHOD("get_line_descent", "line"), &TextParagraph::get_line_descent);
	ClassDB::bind_method(D_METHOD("get_line_width", "line"), &TextParagraph::get_line_width);
	ClassDB::bind_method(D_METHOD("get_line_underline_position", "line"), &TextParagraph::get_line_underline_position);
	ClassDB::bind_method(D_METHOD("get_line_underline_thickness", "line"), &TextParagraph::get_line_underline_thickness);

	ClassDB::bind_method(D_METHOD("get_spacing_top"), &TextParagraph::get_spacing_top);
	ClassDB::bind_method(D_METHOD("get_spacing_bottom"), &TextParagraph::get_spacing_bottom);

	ClassDB::bind_method(D_METHOD("get_dropcap_size"), &TextParagraph::get_dropcap_size);
	ClassDB::bind_method(D_METHOD("get_dropcap_lines"), &TextParagraph::get_dropcap_lines);

	ClassDB::bind_method(D_METHOD("draw", "canvas", "pos", "color", "dc_color"), &TextParagraph::draw, DEFVAL(Color(1, 1, 1)), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_outline", "canvas", "pos", "outline_size", "color", "dc_color"), &TextParagraph::draw_outline, DEFVAL(1), DEFVAL(Color(1, 1, 1)), DEFVAL(Color(1, 1, 1)));

	ClassDB::bind_method(D_METHOD("draw_line", "canvas", "pos", "line", "color"), &TextParagraph::draw_line, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_line_outline", "canvas", "pos", "line", "outline_size", "color"), &TextParagraph::draw_line_outline, DEFVAL(1), DEFVAL(Color(1, 1, 1)));

	ClassDB::bind_method(D_METHOD("draw_dropcap", "canvas", "pos", "color"), &TextParagraph::draw_dropcap, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_dropcap_outline", "canvas", "pos", "outline_size", "color"), &TextParagraph::draw_dropcap_outline, DEFVAL(1), DEFVAL(Color(1, 1, 1)));

	ClassDB::bind_method(D_METHOD("hit_test", "coords"), &TextParagraph::hit_test);

	BIND_ENUM_CONSTANT(OVERRUN_NO_TRIMMING);
	BIND_ENUM_CONSTANT(OVERRUN_TRIM_CHAR);
	BIND_ENUM_CONSTANT(OVERRUN_TRIM_WORD);
	BIND_ENUM_CONSTANT(OVERRUN_TRIM_ELLIPSIS);
	BIND_ENUM_CONSTANT(OVERRUN_TRIM_WORD_ELLIPSIS);
}

void TextParagraph::_shape_lines() {
	if (lines_dirty) {
		for (int i = 0; i < lines_rid.size(); i++) {
			TS->free(lines_rid[i]);
		}
		lines_rid.clear();

		if (!tab_stops.is_empty()) {
			TS->shaped_text_tab_align(rid, tab_stops);
		}

		float h_offset = 0.f;
		float v_offset = 0.f;
		int start = 0;
		dropcap_lines = 0;

		if (TS->shaped_text_get_orientation(dropcap_rid) == TextServer::ORIENTATION_HORIZONTAL) {
			h_offset = TS->shaped_text_get_size(dropcap_rid).x + dropcap_margins.size.x + dropcap_margins.position.x;
			v_offset = TS->shaped_text_get_size(dropcap_rid).y + dropcap_margins.size.y + dropcap_margins.position.y;
		} else {
			h_offset = TS->shaped_text_get_size(dropcap_rid).y + dropcap_margins.size.y + dropcap_margins.position.y;
			v_offset = TS->shaped_text_get_size(dropcap_rid).x + dropcap_margins.size.x + dropcap_margins.position.x;
		}

		if (h_offset > 0) {
			// Dropcap, flow around.
			PackedInt32Array line_breaks = TS->shaped_text_get_line_breaks(rid, width - h_offset, 0, flags);
			for (int i = 0; i < line_breaks.size(); i = i + 2) {
				RID line = TS->shaped_text_substr(rid, line_breaks[i], line_breaks[i + 1] - line_breaks[i]);
				float h = (TS->shaped_text_get_orientation(line) == TextServer::ORIENTATION_HORIZONTAL) ? TS->shaped_text_get_size(line).y : TS->shaped_text_get_size(line).x;
				if (v_offset < h) {
					TS->free(line);
					break;
				}
				if (!tab_stops.is_empty()) {
					TS->shaped_text_tab_align(line, tab_stops);
				}
				dropcap_lines++;
				v_offset -= h;
				start = line_breaks[i + 1];
				lines_rid.push_back(line);
			}
		}
		// Use fixed for the rest of lines.
		PackedInt32Array line_breaks = TS->shaped_text_get_line_breaks(rid, width, start, flags);
		for (int i = 0; i < line_breaks.size(); i = i + 2) {
			RID line = TS->shaped_text_substr(rid, line_breaks[i], line_breaks[i + 1] - line_breaks[i]);
			if (!tab_stops.is_empty()) {
				TS->shaped_text_tab_align(line, tab_stops);
			}
			lines_rid.push_back(line);
		}

		uint16_t overrun_flags = TextServer::OVERRUN_NO_TRIMMING;
		if (overrun_behavior != OVERRUN_NO_TRIMMING) {
			switch (overrun_behavior) {
				case OVERRUN_TRIM_WORD_ELLIPSIS:
					overrun_flags |= TextServer::OVERRUN_TRIM;
					overrun_flags |= TextServer::OVERRUN_TRIM_WORD_ONLY;
					overrun_flags |= TextServer::OVERRUN_ADD_ELLIPSIS;
					break;
				case OVERRUN_TRIM_ELLIPSIS:
					overrun_flags |= TextServer::OVERRUN_TRIM;
					overrun_flags |= TextServer::OVERRUN_ADD_ELLIPSIS;
					break;
				case OVERRUN_TRIM_WORD:
					overrun_flags |= TextServer::OVERRUN_TRIM;
					overrun_flags |= TextServer::OVERRUN_TRIM_WORD_ONLY;
					break;
				case OVERRUN_TRIM_CHAR:
					overrun_flags |= TextServer::OVERRUN_TRIM;
					break;
				case OVERRUN_NO_TRIMMING:
					break;
			}
		}

		bool autowrap_enabled = ((flags & TextServer::BREAK_WORD_BOUND) == TextServer::BREAK_WORD_BOUND) || ((flags & TextServer::BREAK_GRAPHEME_BOUND) == TextServer::BREAK_GRAPHEME_BOUND);

		// Fill after min_size calculation.
		if (autowrap_enabled) {
			int visible_lines = (max_lines_visible >= 0) ? MIN(max_lines_visible, lines_rid.size()) : lines_rid.size();
			bool lines_hidden = visible_lines > 0 && visible_lines < lines_rid.size();
			if (lines_hidden) {
				overrun_flags |= TextServer::OVERRUN_ENFORCE_ELLIPSIS;
			}
			if (alignment == HORIZONTAL_ALIGNMENT_FILL) {
				for (int i = 0; i < lines_rid.size(); i++) {
					if (i < visible_lines - 1 || lines_rid.size() == 1) {
						TS->shaped_text_fit_to_width(lines_rid[i], width, flags);
					} else if (i == (visible_lines - 1)) {
						TS->shaped_text_overrun_trim_to_width(lines_rid[visible_lines - 1], width, overrun_flags);
					}
				}

			} else if (lines_hidden) {
				TS->shaped_text_overrun_trim_to_width(lines_rid[visible_lines - 1], width, overrun_flags);
			}

		} else {
			// Autowrap disabled.
			for (int i = 0; i < lines_rid.size(); i++) {
				if (alignment == HORIZONTAL_ALIGNMENT_FILL) {
					TS->shaped_text_fit_to_width(lines_rid[i], width, flags);
					overrun_flags |= TextServer::OVERRUN_JUSTIFICATION_AWARE;
					TS->shaped_text_overrun_trim_to_width(lines_rid[i], width, overrun_flags);
					TS->shaped_text_fit_to_width(lines_rid[i], width, flags | TextServer::JUSTIFICATION_CONSTRAIN_ELLIPSIS);
				} else {
					TS->shaped_text_overrun_trim_to_width(lines_rid[i], width, overrun_flags);
				}
			}
		}
		lines_dirty = false;
	}
}

RID TextParagraph::get_rid() const {
	return rid;
}

RID TextParagraph::get_line_rid(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines_rid.size(), RID());
	return lines_rid[p_line];
}

RID TextParagraph::get_dropcap_rid() const {
	return dropcap_rid;
}

void TextParagraph::clear() {
	spacing_top = 0;
	spacing_bottom = 0;
	for (int i = 0; i < lines_rid.size(); i++) {
		TS->free(lines_rid[i]);
	}
	lines_rid.clear();
	TS->shaped_text_clear(rid);
	TS->shaped_text_clear(dropcap_rid);
}

void TextParagraph::set_preserve_invalid(bool p_enabled) {
	TS->shaped_text_set_preserve_invalid(rid, p_enabled);
	TS->shaped_text_set_preserve_invalid(dropcap_rid, p_enabled);
	lines_dirty = true;
}

bool TextParagraph::get_preserve_invalid() const {
	return TS->shaped_text_get_preserve_invalid(rid);
}

void TextParagraph::set_preserve_control(bool p_enabled) {
	TS->shaped_text_set_preserve_control(rid, p_enabled);
	TS->shaped_text_set_preserve_control(dropcap_rid, p_enabled);
	lines_dirty = true;
}

bool TextParagraph::get_preserve_control() const {
	return TS->shaped_text_get_preserve_control(rid);
}

void TextParagraph::set_direction(TextServer::Direction p_direction) {
	TS->shaped_text_set_direction(rid, p_direction);
	TS->shaped_text_set_direction(dropcap_rid, p_direction);
	lines_dirty = true;
}

TextServer::Direction TextParagraph::get_direction() const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	return TS->shaped_text_get_direction(rid);
}

void TextParagraph::set_custom_punctuation(const String &p_punct) {
	TS->shaped_text_set_custom_punctuation(rid, p_punct);
	lines_dirty = true;
}

String TextParagraph::get_custom_punctuation() const {
	return TS->shaped_text_get_custom_punctuation(rid);
}

void TextParagraph::set_orientation(TextServer::Orientation p_orientation) {
	TS->shaped_text_set_orientation(rid, p_orientation);
	TS->shaped_text_set_orientation(dropcap_rid, p_orientation);
	lines_dirty = true;
}

TextServer::Orientation TextParagraph::get_orientation() const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	return TS->shaped_text_get_orientation(rid);
}

bool TextParagraph::set_dropcap(const String &p_text, const Ref<Font> &p_fonts, int p_size, const Rect2 &p_dropcap_margins, const Dictionary &p_opentype_features, const String &p_language) {
	ERR_FAIL_COND_V(p_fonts.is_null(), false);
	TS->shaped_text_clear(dropcap_rid);
	dropcap_margins = p_dropcap_margins;
	bool res = TS->shaped_text_add_string(dropcap_rid, p_text, p_fonts->get_rids(), p_size, p_opentype_features, p_language);
	lines_dirty = true;
	return res;
}

void TextParagraph::clear_dropcap() {
	dropcap_margins = Rect2();
	TS->shaped_text_clear(dropcap_rid);
	lines_dirty = true;
}

bool TextParagraph::add_string(const String &p_text, const Ref<Font> &p_fonts, int p_size, const Dictionary &p_opentype_features, const String &p_language) {
	ERR_FAIL_COND_V(p_fonts.is_null(), false);
	bool res = TS->shaped_text_add_string(rid, p_text, p_fonts->get_rids(), p_size, p_opentype_features, p_language);
	spacing_top = p_fonts->get_spacing(TextServer::SPACING_TOP);
	spacing_bottom = p_fonts->get_spacing(TextServer::SPACING_BOTTOM);
	lines_dirty = true;
	return res;
}

int TextParagraph::get_spacing_top() const {
	return spacing_top;
}

int TextParagraph::get_spacing_bottom() const {
	return spacing_bottom;
}

void TextParagraph::set_bidi_override(const Array &p_override) {
	TS->shaped_text_set_bidi_override(rid, p_override);
	lines_dirty = true;
}

bool TextParagraph::add_object(Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align, int p_length) {
	bool res = TS->shaped_text_add_object(rid, p_key, p_size, p_inline_align, p_length);
	lines_dirty = true;
	return res;
}

bool TextParagraph::resize_object(Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align) {
	bool res = TS->shaped_text_resize_object(rid, p_key, p_size, p_inline_align);
	lines_dirty = true;
	return res;
}

void TextParagraph::set_alignment(HorizontalAlignment p_alignment) {
	if (alignment != p_alignment) {
		if (alignment == HORIZONTAL_ALIGNMENT_FILL || p_alignment == HORIZONTAL_ALIGNMENT_FILL) {
			alignment = p_alignment;
			lines_dirty = true;
		} else {
			alignment = p_alignment;
		}
	}
}

HorizontalAlignment TextParagraph::get_alignment() const {
	return alignment;
}

void TextParagraph::tab_align(const Vector<float> &p_tab_stops) {
	tab_stops = p_tab_stops;
	lines_dirty = true;
}

void TextParagraph::set_flags(uint16_t p_flags) {
	if (flags != p_flags) {
		flags = p_flags;
		lines_dirty = true;
	}
}

uint16_t TextParagraph::get_flags() const {
	return flags;
}

void TextParagraph::set_text_overrun_behavior(TextParagraph::OverrunBehavior p_behavior) {
	if (overrun_behavior != p_behavior) {
		overrun_behavior = p_behavior;
		lines_dirty = true;
	}
}

TextParagraph::OverrunBehavior TextParagraph::get_text_overrun_behavior() const {
	return overrun_behavior;
}

void TextParagraph::set_width(float p_width) {
	if (width != p_width) {
		width = p_width;
		lines_dirty = true;
	}
}

float TextParagraph::get_width() const {
	return width;
}

Size2 TextParagraph::get_non_wrapped_size() const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
		return Size2(TS->shaped_text_get_size(rid).x, TS->shaped_text_get_size(rid).y + spacing_top + spacing_bottom);
	} else {
		return Size2(TS->shaped_text_get_size(rid).x + spacing_top + spacing_bottom, TS->shaped_text_get_size(rid).y);
	}
}

Size2 TextParagraph::get_size() const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	Size2 size;
	int visible_lines = (max_lines_visible >= 0) ? MIN(max_lines_visible, lines_rid.size()) : lines_rid.size();
	for (int i = 0; i < visible_lines; i++) {
		Size2 lsize = TS->shaped_text_get_size(lines_rid[i]);
		if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			size.x = MAX(size.x, lsize.x);
			size.y += lsize.y + spacing_top + spacing_bottom;
		} else {
			size.x += lsize.x + spacing_top + spacing_bottom;
			size.y = MAX(size.y, lsize.y);
		}
	}
	return size;
}

int TextParagraph::get_line_count() const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	return lines_rid.size();
}

void TextParagraph::set_max_lines_visible(int p_lines) {
	if (p_lines != max_lines_visible) {
		max_lines_visible = p_lines;
		lines_dirty = true;
	}
}

int TextParagraph::get_max_lines_visible() const {
	return max_lines_visible;
}

Array TextParagraph::get_line_objects(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines_rid.size(), Array());
	return TS->shaped_text_get_objects(lines_rid[p_line]);
}

Rect2 TextParagraph::get_line_object_rect(int p_line, Variant p_key) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines_rid.size(), Rect2());
	Rect2 xrect = TS->shaped_text_get_object_rect(lines_rid[p_line], p_key);
	for (int i = 0; i < p_line; i++) {
		Size2 lsize = TS->shaped_text_get_size(lines_rid[i]);
		if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			xrect.position.y += lsize.y + spacing_top + spacing_bottom;
		} else {
			xrect.position.x += lsize.x + spacing_top + spacing_bottom;
		}
	}
	return xrect;
}

Size2 TextParagraph::get_line_size(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines_rid.size(), Size2());
	if (TS->shaped_text_get_orientation(lines_rid[p_line]) == TextServer::ORIENTATION_HORIZONTAL) {
		return Size2(TS->shaped_text_get_size(lines_rid[p_line]).x, TS->shaped_text_get_size(lines_rid[p_line]).y + spacing_top + spacing_bottom);
	} else {
		return Size2(TS->shaped_text_get_size(lines_rid[p_line]).x + spacing_top + spacing_bottom, TS->shaped_text_get_size(lines_rid[p_line]).y);
	}
}

Vector2i TextParagraph::get_line_range(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines_rid.size(), Vector2i());
	return TS->shaped_text_get_range(lines_rid[p_line]);
}

float TextParagraph::get_line_ascent(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines_rid.size(), 0.f);
	return TS->shaped_text_get_ascent(lines_rid[p_line]) + spacing_top;
}

float TextParagraph::get_line_descent(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines_rid.size(), 0.f);
	return TS->shaped_text_get_descent(lines_rid[p_line]) + spacing_bottom;
}

float TextParagraph::get_line_width(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines_rid.size(), 0.f);
	return TS->shaped_text_get_width(lines_rid[p_line]);
}

float TextParagraph::get_line_underline_position(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines_rid.size(), 0.f);
	return TS->shaped_text_get_underline_position(lines_rid[p_line]);
}

float TextParagraph::get_line_underline_thickness(int p_line) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= lines_rid.size(), 0.f);
	return TS->shaped_text_get_underline_thickness(lines_rid[p_line]);
}

Size2 TextParagraph::get_dropcap_size() const {
	return TS->shaped_text_get_size(dropcap_rid) + dropcap_margins.size + dropcap_margins.position;
}

int TextParagraph::get_dropcap_lines() const {
	return dropcap_lines;
}

void TextParagraph::draw(RID p_canvas, const Vector2 &p_pos, const Color &p_color, const Color &p_dc_color) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	Vector2 ofs = p_pos;
	float h_offset = 0.f;
	if (TS->shaped_text_get_orientation(dropcap_rid) == TextServer::ORIENTATION_HORIZONTAL) {
		h_offset = TS->shaped_text_get_size(dropcap_rid).x + dropcap_margins.size.x + dropcap_margins.position.x;
	} else {
		h_offset = TS->shaped_text_get_size(dropcap_rid).y + dropcap_margins.size.y + dropcap_margins.position.y;
	}

	if (h_offset > 0) {
		// Draw dropcap.
		Vector2 dc_off = ofs;
		if (TS->shaped_text_get_direction(dropcap_rid) == TextServer::DIRECTION_RTL) {
			if (TS->shaped_text_get_orientation(dropcap_rid) == TextServer::ORIENTATION_HORIZONTAL) {
				dc_off.x += width - h_offset;
			} else {
				dc_off.y += width - h_offset;
			}
		}
		TS->shaped_text_draw(dropcap_rid, p_canvas, dc_off + Vector2(0, TS->shaped_text_get_ascent(dropcap_rid) + dropcap_margins.size.y + dropcap_margins.position.y / 2), -1, -1, p_dc_color);
	}

	int lines_visible = (max_lines_visible >= 0) ? MIN(max_lines_visible, lines_rid.size()) : lines_rid.size();

	for (int i = 0; i < lines_visible; i++) {
		float l_width = width;
		if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			ofs.x = p_pos.x;
			ofs.y += TS->shaped_text_get_ascent(lines_rid[i]) + spacing_top;
			if (i <= dropcap_lines) {
				if (TS->shaped_text_get_direction(dropcap_rid) == TextServer::DIRECTION_LTR) {
					ofs.x -= h_offset;
				}
				l_width -= h_offset;
			}
		} else {
			ofs.y = p_pos.y;
			ofs.x += TS->shaped_text_get_ascent(lines_rid[i]) + spacing_top;
			if (i <= dropcap_lines) {
				if (TS->shaped_text_get_direction(dropcap_rid) == TextServer::DIRECTION_LTR) {
					ofs.x -= h_offset;
				}
				l_width -= h_offset;
			}
		}
		float line_width = TS->shaped_text_get_width(lines_rid[i]);
		if (width > 0) {
			switch (alignment) {
				case HORIZONTAL_ALIGNMENT_FILL:
					if (TS->shaped_text_get_direction(lines_rid[i]) == TextServer::DIRECTION_RTL) {
						if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
							ofs.x += l_width - line_width;
						} else {
							ofs.y += l_width - line_width;
						}
					}
					break;
				case HORIZONTAL_ALIGNMENT_LEFT:
					break;
				case HORIZONTAL_ALIGNMENT_CENTER: {
					if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
						ofs.x += Math::floor((l_width - line_width) / 2.0);
					} else {
						ofs.y += Math::floor((l_width - line_width) / 2.0);
					}
				} break;
				case HORIZONTAL_ALIGNMENT_RIGHT: {
					if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
						ofs.x += l_width - line_width;
					} else {
						ofs.y += l_width - line_width;
					}
				} break;
			}
		}
		float clip_l;
		if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			clip_l = MAX(0, p_pos.x - ofs.x);
		} else {
			clip_l = MAX(0, p_pos.y - ofs.y);
		}
		TS->shaped_text_draw(lines_rid[i], p_canvas, ofs, clip_l, clip_l + l_width, p_color);
		if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			ofs.x = p_pos.x;
			ofs.y += TS->shaped_text_get_descent(lines_rid[i]) + spacing_bottom;
		} else {
			ofs.y = p_pos.y;
			ofs.x += TS->shaped_text_get_descent(lines_rid[i]) + spacing_bottom;
		}
	}
}

void TextParagraph::draw_outline(RID p_canvas, const Vector2 &p_pos, int p_outline_size, const Color &p_color, const Color &p_dc_color) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	Vector2 ofs = p_pos;

	float h_offset = 0.f;
	if (TS->shaped_text_get_orientation(dropcap_rid) == TextServer::ORIENTATION_HORIZONTAL) {
		h_offset = TS->shaped_text_get_size(dropcap_rid).x + dropcap_margins.size.x + dropcap_margins.position.x;
	} else {
		h_offset = TS->shaped_text_get_size(dropcap_rid).y + dropcap_margins.size.y + dropcap_margins.position.y;
	}

	if (h_offset > 0) {
		// Draw dropcap.
		Vector2 dc_off = ofs;
		if (TS->shaped_text_get_direction(dropcap_rid) == TextServer::DIRECTION_RTL) {
			if (TS->shaped_text_get_orientation(dropcap_rid) == TextServer::ORIENTATION_HORIZONTAL) {
				dc_off.x += width - h_offset;
			} else {
				dc_off.y += width - h_offset;
			}
		}
		TS->shaped_text_draw_outline(dropcap_rid, p_canvas, dc_off + Vector2(dropcap_margins.position.x, TS->shaped_text_get_ascent(dropcap_rid) + dropcap_margins.position.y), -1, -1, p_outline_size, p_dc_color);
	}

	for (int i = 0; i < lines_rid.size(); i++) {
		float l_width = width;
		if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			ofs.x = p_pos.x;
			ofs.y += TS->shaped_text_get_ascent(lines_rid[i]) + spacing_top;
			if (i <= dropcap_lines) {
				if (TS->shaped_text_get_direction(dropcap_rid) == TextServer::DIRECTION_LTR) {
					ofs.x -= h_offset;
				}
				l_width -= h_offset;
			}
		} else {
			ofs.y = p_pos.y;
			ofs.x += TS->shaped_text_get_ascent(lines_rid[i]) + spacing_top;
			if (i <= dropcap_lines) {
				if (TS->shaped_text_get_direction(dropcap_rid) == TextServer::DIRECTION_LTR) {
					ofs.x -= h_offset;
				}
				l_width -= h_offset;
			}
		}
		float length = TS->shaped_text_get_width(lines_rid[i]);
		if (width > 0) {
			switch (alignment) {
				case HORIZONTAL_ALIGNMENT_FILL:
					if (TS->shaped_text_get_direction(lines_rid[i]) == TextServer::DIRECTION_RTL) {
						if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
							ofs.x += l_width - length;
						} else {
							ofs.y += l_width - length;
						}
					}
					break;
				case HORIZONTAL_ALIGNMENT_LEFT:
					break;
				case HORIZONTAL_ALIGNMENT_CENTER: {
					if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
						ofs.x += Math::floor((l_width - length) / 2.0);
					} else {
						ofs.y += Math::floor((l_width - length) / 2.0);
					}
				} break;
				case HORIZONTAL_ALIGNMENT_RIGHT: {
					if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
						ofs.x += l_width - length;
					} else {
						ofs.y += l_width - length;
					}
				} break;
			}
		}
		float clip_l;
		if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			clip_l = MAX(0, p_pos.x - ofs.x);
		} else {
			clip_l = MAX(0, p_pos.y - ofs.y);
		}
		TS->shaped_text_draw_outline(lines_rid[i], p_canvas, ofs, clip_l, clip_l + l_width, p_outline_size, p_color);
		if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			ofs.x = p_pos.x;
			ofs.y += TS->shaped_text_get_descent(lines_rid[i]) + spacing_bottom;
		} else {
			ofs.y = p_pos.y;
			ofs.x += TS->shaped_text_get_descent(lines_rid[i]) + spacing_bottom;
		}
	}
}

int TextParagraph::hit_test(const Point2 &p_coords) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	Vector2 ofs;
	if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
		if (ofs.y < 0) {
			return 0;
		}
	} else {
		if (ofs.x < 0) {
			return 0;
		}
	}
	for (int i = 0; i < lines_rid.size(); i++) {
		if (TS->shaped_text_get_orientation(lines_rid[i]) == TextServer::ORIENTATION_HORIZONTAL) {
			if ((p_coords.y >= ofs.y) && (p_coords.y <= ofs.y + TS->shaped_text_get_size(lines_rid[i]).y)) {
				return TS->shaped_text_hit_test_position(lines_rid[i], p_coords.x);
			}
			ofs.y += TS->shaped_text_get_size(lines_rid[i]).y + spacing_bottom + spacing_top;
		} else {
			if ((p_coords.x >= ofs.x) && (p_coords.x <= ofs.x + TS->shaped_text_get_size(lines_rid[i]).x)) {
				return TS->shaped_text_hit_test_position(lines_rid[i], p_coords.y);
			}
			ofs.y += TS->shaped_text_get_size(lines_rid[i]).x + spacing_bottom + spacing_top;
		}
	}
	return TS->shaped_text_get_range(rid).y;
}

void TextParagraph::draw_dropcap(RID p_canvas, const Vector2 &p_pos, const Color &p_color) const {
	Vector2 ofs = p_pos;
	float h_offset = 0.f;
	if (TS->shaped_text_get_orientation(dropcap_rid) == TextServer::ORIENTATION_HORIZONTAL) {
		h_offset = TS->shaped_text_get_size(dropcap_rid).x + dropcap_margins.size.x + dropcap_margins.position.x;
	} else {
		h_offset = TS->shaped_text_get_size(dropcap_rid).y + dropcap_margins.size.y + dropcap_margins.position.y;
	}

	if (h_offset > 0) {
		// Draw dropcap.
		if (TS->shaped_text_get_direction(dropcap_rid) == TextServer::DIRECTION_RTL) {
			if (TS->shaped_text_get_orientation(dropcap_rid) == TextServer::ORIENTATION_HORIZONTAL) {
				ofs.x += width - h_offset;
			} else {
				ofs.y += width - h_offset;
			}
		}
		TS->shaped_text_draw(dropcap_rid, p_canvas, ofs + Vector2(dropcap_margins.position.x, TS->shaped_text_get_ascent(dropcap_rid) + dropcap_margins.position.y), -1, -1, p_color);
	}
}

void TextParagraph::draw_dropcap_outline(RID p_canvas, const Vector2 &p_pos, int p_outline_size, const Color &p_color) const {
	Vector2 ofs = p_pos;
	float h_offset = 0.f;
	if (TS->shaped_text_get_orientation(dropcap_rid) == TextServer::ORIENTATION_HORIZONTAL) {
		h_offset = TS->shaped_text_get_size(dropcap_rid).x + dropcap_margins.size.x + dropcap_margins.position.x;
	} else {
		h_offset = TS->shaped_text_get_size(dropcap_rid).y + dropcap_margins.size.y + dropcap_margins.position.y;
	}

	if (h_offset > 0) {
		// Draw dropcap.
		if (TS->shaped_text_get_direction(dropcap_rid) == TextServer::DIRECTION_RTL) {
			if (TS->shaped_text_get_orientation(dropcap_rid) == TextServer::ORIENTATION_HORIZONTAL) {
				ofs.x += width - h_offset;
			} else {
				ofs.y += width - h_offset;
			}
		}
		TS->shaped_text_draw_outline(dropcap_rid, p_canvas, ofs + Vector2(dropcap_margins.position.x, TS->shaped_text_get_ascent(dropcap_rid) + dropcap_margins.position.y), -1, -1, p_outline_size, p_color);
	}
}

void TextParagraph::draw_line(RID p_canvas, const Vector2 &p_pos, int p_line, const Color &p_color) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND(p_line < 0 || p_line >= lines_rid.size());

	Vector2 ofs = p_pos;

	if (TS->shaped_text_get_orientation(lines_rid[p_line]) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(lines_rid[p_line]) + spacing_top;
	} else {
		ofs.x += TS->shaped_text_get_ascent(lines_rid[p_line]) + spacing_top;
	}
	return TS->shaped_text_draw(lines_rid[p_line], p_canvas, ofs, -1, -1, p_color);
}

void TextParagraph::draw_line_outline(RID p_canvas, const Vector2 &p_pos, int p_line, int p_outline_size, const Color &p_color) const {
	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND(p_line < 0 || p_line >= lines_rid.size());

	Vector2 ofs = p_pos;
	if (TS->shaped_text_get_orientation(lines_rid[p_line]) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(lines_rid[p_line]) + spacing_top;
	} else {
		ofs.x += TS->shaped_text_get_ascent(lines_rid[p_line]) + spacing_top;
	}
	return TS->shaped_text_draw_outline(lines_rid[p_line], p_canvas, ofs, -1, -1, p_outline_size, p_color);
}

TextParagraph::TextParagraph(const String &p_text, const Ref<Font> &p_fonts, int p_size, const Dictionary &p_opentype_features, const String &p_language, float p_width, TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	rid = TS->create_shaped_text(p_direction, p_orientation);
	TS->shaped_text_add_string(rid, p_text, p_fonts->get_rids(), p_size, p_opentype_features, p_language);
	spacing_top = p_fonts->get_spacing(TextServer::SPACING_TOP);
	spacing_bottom = p_fonts->get_spacing(TextServer::SPACING_BOTTOM);
	width = p_width;
}

TextParagraph::TextParagraph() {
	rid = TS->create_shaped_text();
	dropcap_rid = TS->create_shaped_text();
}

TextParagraph::~TextParagraph() {
	for (int i = 0; i < lines_rid.size(); i++) {
		TS->free(lines_rid[i]);
	}
	lines_rid.clear();
	TS->free(rid);
	TS->free(dropcap_rid);
}
