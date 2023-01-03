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

	ADD_PROPERTY(PropertyInfo(Variant::INT, "orientation", PROPERTY_HINT_ENUM, "Horizontal,Vertical Upright,Vertical Mixed,Vertical Sideways"), "set_orientation", "get_orientation");

	ClassDB::bind_method(D_METHOD("set_preserve_invalid", "enabled"), &TextParagraph::set_preserve_invalid);
	ClassDB::bind_method(D_METHOD("get_preserve_invalid"), &TextParagraph::get_preserve_invalid);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "preserve_invalid"), "set_preserve_invalid", "get_preserve_invalid");

	ClassDB::bind_method(D_METHOD("set_preserve_control", "enabled"), &TextParagraph::set_preserve_control);
	ClassDB::bind_method(D_METHOD("get_preserve_control"), &TextParagraph::get_preserve_control);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "preserve_control"), "set_preserve_control", "get_preserve_control");

	ClassDB::bind_method(D_METHOD("set_bidi_override", "override"), &TextParagraph::set_bidi_override);

	ClassDB::bind_method(D_METHOD("set_dropcap", "text", "font", "font_size", "dropcap_margins", "language"), &TextParagraph::set_dropcap, DEFVAL(Rect2()), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("clear_dropcap"), &TextParagraph::clear_dropcap);

	ClassDB::bind_method(D_METHOD("get_span_count"), &TextParagraph::get_span_count);
	ClassDB::bind_method(D_METHOD("update_span_font", "span", "font", "font_size"), &TextParagraph::update_span_font);

	ClassDB::bind_method(D_METHOD("add_string", "text", "font", "font_size", "language", "meta"), &TextParagraph::add_string, DEFVAL(""), DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("add_object", "key", "size", "inline_align", "length", "baseline"), &TextParagraph::add_object, DEFVAL(INLINE_ALIGNMENT_CENTER), DEFVAL(1), DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("resize_object", "key", "size", "inline_align", "baseline"), &TextParagraph::resize_object, DEFVAL(INLINE_ALIGNMENT_CENTER), DEFVAL(0.0));

	ClassDB::bind_method(D_METHOD("set_horizontal_alignment", "horizontal_alignment"), &TextParagraph::set_horizontal_alignment);
	ClassDB::bind_method(D_METHOD("get_horizontal_alignment"), &TextParagraph::get_horizontal_alignment);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "horizontal_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_horizontal_alignment", "get_horizontal_alignment");

	ClassDB::bind_method(D_METHOD("set_vertical_alignment", "vertical_alignment"), &TextParagraph::set_vertical_alignment);
	ClassDB::bind_method(D_METHOD("get_vertical_alignment"), &TextParagraph::get_vertical_alignment);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "vertical_alignment", PROPERTY_HINT_ENUM, "Top,Center,Bottom,Fill"), "set_vertical_alignment", "get_vertical_alignment");

	ClassDB::bind_method(D_METHOD("set_uniform_line_height", "uniform_line_height"), &TextParagraph::set_uniform_line_height);
	ClassDB::bind_method(D_METHOD("get_uniform_line_height"), &TextParagraph::get_uniform_line_height);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "uniform_line_height"), "set_uniform_line_height", "get_uniform_line_height");

	ClassDB::bind_method(D_METHOD("set_invert_line_order", "invert_line_order"), &TextParagraph::set_invert_line_order);
	ClassDB::bind_method(D_METHOD("get_invert_line_order"), &TextParagraph::get_invert_line_order);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert_line_order"), "set_invert_line_order", "get_invert_line_order");

	ClassDB::bind_method(D_METHOD("set_clip", "clip"), &TextParagraph::set_clip);
	ClassDB::bind_method(D_METHOD("get_clip"), &TextParagraph::get_clip);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip"), "set_clip", "get_clip");

	ClassDB::bind_method(D_METHOD("tab_align", "tab_stops"), &TextParagraph::tab_align);

	ClassDB::bind_method(D_METHOD("set_break_flags", "flags"), &TextParagraph::set_break_flags);
	ClassDB::bind_method(D_METHOD("get_break_flags"), &TextParagraph::get_break_flags);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "break_flags", PROPERTY_HINT_FLAGS, "Mandatory,Word Bound,Grapheme Bound,Adaptive,Trim Spaces"), "set_break_flags", "get_break_flags");

	ClassDB::bind_method(D_METHOD("set_justification_flags", "flags"), &TextParagraph::set_justification_flags);
	ClassDB::bind_method(D_METHOD("get_justification_flags"), &TextParagraph::get_justification_flags);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "justification_flags", PROPERTY_HINT_FLAGS, "Kashida Justification,Word Justification,Trim Edge Spaces After Justification,Justify Only After Last Tab,Constrain Ellipsis"), "set_justification_flags", "get_justification_flags");

	ClassDB::bind_method(D_METHOD("set_text_overrun_behavior", "overrun_behavior"), &TextParagraph::set_text_overrun_behavior);
	ClassDB::bind_method(D_METHOD("get_text_overrun_behavior"), &TextParagraph::get_text_overrun_behavior);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_overrun_behavior", PROPERTY_HINT_ENUM, "Trim Nothing,Trim Characters,Trim Words,Ellipsis,Word Ellipsis"), "set_text_overrun_behavior", "get_text_overrun_behavior");

	ClassDB::bind_method(D_METHOD("set_width", "width"), &TextParagraph::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &TextParagraph::get_width);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width"), "set_width", "get_width");

	ClassDB::bind_method(D_METHOD("set_height", "height"), &TextParagraph::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &TextParagraph::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");

	ClassDB::bind_method(D_METHOD("set_extra_line_spacing", "spacing"), &TextParagraph::set_extra_line_spacing);
	ClassDB::bind_method(D_METHOD("get_extra_line_spacing"), &TextParagraph::get_extra_line_spacing);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "extra_line_spacing"), "set_extra_line_spacing", "get_extra_line_spacing");

	ClassDB::bind_method(D_METHOD("get_non_wrapped_size"), &TextParagraph::get_non_wrapped_size);
	ClassDB::bind_method(D_METHOD("get_size"), &TextParagraph::get_size);

	ClassDB::bind_method(D_METHOD("get_rid"), &TextParagraph::get_rid);
	ClassDB::bind_method(D_METHOD("get_line_rid", "line"), &TextParagraph::get_line_rid);
	ClassDB::bind_method(D_METHOD("get_dropcap_rid"), &TextParagraph::get_dropcap_rid);

	ClassDB::bind_method(D_METHOD("get_line_count"), &TextParagraph::get_line_count);
	ClassDB::bind_method(D_METHOD("get_visible_line_count"), &TextParagraph::get_visible_line_count);

	ClassDB::bind_method(D_METHOD("set_lines_skipped", "lines_skipped"), &TextParagraph::set_lines_skipped);
	ClassDB::bind_method(D_METHOD("get_lines_skipped"), &TextParagraph::get_lines_skipped);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "lines_skipped"), "set_lines_skipped", "get_lines_skipped");

	ClassDB::bind_method(D_METHOD("set_max_lines_visible", "max_lines_visible"), &TextParagraph::set_max_lines_visible);
	ClassDB::bind_method(D_METHOD("get_max_lines_visible"), &TextParagraph::get_max_lines_visible);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_lines_visible"), "set_max_lines_visible", "get_max_lines_visible");

	ClassDB::bind_method(D_METHOD("get_line_objects", "line"), &TextParagraph::get_line_objects);
	ClassDB::bind_method(D_METHOD("get_line_object_rect", "line", "key"), &TextParagraph::get_line_object_rect);
	ClassDB::bind_method(D_METHOD("get_line_size", "line"), &TextParagraph::get_line_size);
	ClassDB::bind_method(D_METHOD("get_line_offset", "line"), &TextParagraph::get_line_offset);
	ClassDB::bind_method(D_METHOD("get_line_range", "line"), &TextParagraph::get_line_range);
	ClassDB::bind_method(D_METHOD("get_line_ascent", "line"), &TextParagraph::get_line_ascent);
	ClassDB::bind_method(D_METHOD("get_line_descent", "line"), &TextParagraph::get_line_descent);
	ClassDB::bind_method(D_METHOD("get_line_width", "line"), &TextParagraph::get_line_width);
	ClassDB::bind_method(D_METHOD("get_line_underline_position", "line"), &TextParagraph::get_line_underline_position);
	ClassDB::bind_method(D_METHOD("get_line_underline_thickness", "line"), &TextParagraph::get_line_underline_thickness);

	ClassDB::bind_method(D_METHOD("get_dropcap_size"), &TextParagraph::get_dropcap_size);
	ClassDB::bind_method(D_METHOD("get_dropcap_lines"), &TextParagraph::get_dropcap_lines);
	ClassDB::bind_method(D_METHOD("get_dropcap_offset"), &TextParagraph::get_dropcap_offset);

	ClassDB::bind_method(D_METHOD("has_invalid_glyphs"), &TextParagraph::has_invalid_glyphs);
	ClassDB::bind_method(D_METHOD("get_glyph_count"), &TextParagraph::get_glyph_count);

	ClassDB::bind_method(D_METHOD("draw", "canvas", "pos", "color", "dc_color"), &TextParagraph::draw, DEFVAL(Color(1, 1, 1)), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_outline", "canvas", "pos", "outline_size", "color", "dc_color"), &TextParagraph::draw_outline, DEFVAL(1), DEFVAL(Color(1, 1, 1)), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_custom", "canvas", "pos", "callback"), &TextParagraph::_draw_custom);

	ClassDB::bind_method(D_METHOD("draw_underline_custom", "canvas", "pos", "line_type", "start", "end", "callback"), &TextParagraph::_draw_underline_custom);

	ClassDB::bind_method(D_METHOD("draw_line", "canvas", "pos", "line", "color"), &TextParagraph::draw_line, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_line_outline", "canvas", "pos", "line", "outline_size", "color"), &TextParagraph::draw_line_outline, DEFVAL(1), DEFVAL(Color(1, 1, 1)));

	ClassDB::bind_method(D_METHOD("draw_dropcap", "canvas", "pos", "color"), &TextParagraph::draw_dropcap, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_dropcap_outline", "canvas", "pos", "outline_size", "color"), &TextParagraph::draw_dropcap_outline, DEFVAL(1), DEFVAL(Color(1, 1, 1)));

	ClassDB::bind_method(D_METHOD("hit_test", "coords"), &TextParagraph::hit_test);
}

void TextParagraph::_shape_lines() {
	if (lines_dirty) {
		for (int i = 0; i < (int)lines_rid.size(); i++) {
			TS->free_rid(lines_rid[i]);
		}
		lines_rid.clear();

		if (!tab_stops.is_empty()) {
			TS->shaped_text_tab_align(rid, tab_stops);
		}

		int start = 0;
		dropcap_lines = 0;

		Size2 dc_vb = TS->shaped_text_get_vertical_bounds(dropcap_rid);
		float dc_width = TS->shaped_text_get_width(dropcap_rid) + dropcap_margins.position.x + dropcap_margins.size.x;
		float dc_height = dc_vb.x + dc_vb.y + dropcap_margins.position.y + dropcap_margins.size.y;

		if (dc_width > 0) {
			// Dropcap, flow around.
			PackedInt32Array line_breaks = TS->shaped_text_get_line_breaks(rid, width - dc_width, 0, brk_flags);
			TextServer::Orientation orientation = TS->shaped_text_get_orientation(rid);
			for (int i = 0; i < line_breaks.size() && dc_height > 0.0; i = i + 2) {
				RID line = TS->shaped_text_substr(rid, line_breaks[i], line_breaks[i + 1] - line_breaks[i]);
				float h = (orientation == TextServer::ORIENTATION_HORIZONTAL) ? TS->shaped_text_get_size(line).y : TS->shaped_text_get_size(line).x;
				if (!tab_stops.is_empty()) {
					TS->shaped_text_tab_align(line, tab_stops);
				}
				dropcap_lines++;
				dc_height -= (h + line_spacing);
				start = line_breaks[i + 1];
				lines_rid.push_back(line);
			}
		}

		// Use fixed for the rest of lines.
		PackedInt32Array line_breaks = TS->shaped_text_get_line_breaks(rid, width, start, brk_flags);
		for (int i = 0; i < line_breaks.size(); i = i + 2) {
			RID line = TS->shaped_text_substr(rid, line_breaks[i], line_breaks[i + 1] - line_breaks[i]);
			if (!tab_stops.is_empty()) {
				TS->shaped_text_tab_align(line, tab_stops);
			}
			lines_rid.push_back(line);
		}

		BitField<TextServer::TextOverrunFlag> overrun_flags = TextServer::OVERRUN_NO_TRIM;
		if (overrun_behavior != TextServer::OVERRUN_NO_TRIMMING) {
			switch (overrun_behavior) {
				case TextServer::OVERRUN_TRIM_WORD_ELLIPSIS:
					overrun_flags.set_flag(TextServer::OVERRUN_TRIM);
					overrun_flags.set_flag(TextServer::OVERRUN_TRIM_WORD_ONLY);
					overrun_flags.set_flag(TextServer::OVERRUN_ADD_ELLIPSIS);
					break;
				case TextServer::OVERRUN_TRIM_ELLIPSIS:
					overrun_flags.set_flag(TextServer::OVERRUN_TRIM);
					overrun_flags.set_flag(TextServer::OVERRUN_ADD_ELLIPSIS);
					break;
				case TextServer::OVERRUN_TRIM_WORD:
					overrun_flags.set_flag(TextServer::OVERRUN_TRIM);
					overrun_flags.set_flag(TextServer::OVERRUN_TRIM_WORD_ONLY);
					break;
				case TextServer::OVERRUN_TRIM_CHAR:
					overrun_flags.set_flag(TextServer::OVERRUN_TRIM);
					break;
				case TextServer::OVERRUN_NO_TRIMMING:
					break;
			}
		}

		bool autowrap_enabled = brk_flags.has_flag(TextServer::BREAK_WORD_BOUND) || brk_flags.has_flag(TextServer::BREAK_GRAPHEME_BOUND);

		// Fill after min_size calculation.
		if (autowrap_enabled) {
			int visible_lines = (max_lines_visible >= 0) ? MIN(max_lines_visible, (int)lines_rid.size()) : (int)lines_rid.size();
			bool lines_hidden = visible_lines > 0 && visible_lines < (int)lines_rid.size();
			if (lines_hidden) {
				overrun_flags.set_flag(TextServer::OVERRUN_ENFORCE_ELLIPSIS);
			}
			if (horizontal_alignment == HORIZONTAL_ALIGNMENT_FILL) {
				for (int i = 0; i < (int)lines_rid.size(); i++) {
					float lw = (i < dropcap_lines) ? (width - dc_width) : (width);
					if (i < visible_lines - 1 || (int)lines_rid.size() == 1) {
						TS->shaped_text_fit_to_width(lines_rid[i], lw, jst_flags);
					} else if (i == (visible_lines - 1)) {
						TS->shaped_text_overrun_trim_to_width(lines_rid[visible_lines - 1], lw, overrun_flags);
					}
				}
			} else if (lines_hidden) {
				TS->shaped_text_overrun_trim_to_width(lines_rid[visible_lines - 1], width, overrun_flags);
			}
		} else {
			// Autowrap disabled.
			for (int i = 0; i < (int)lines_rid.size(); i++) {
				float lw = (i < dropcap_lines) ? (width - dc_width) : (width);
				if (horizontal_alignment == HORIZONTAL_ALIGNMENT_FILL) {
					TS->shaped_text_fit_to_width(lines_rid[i], lw, jst_flags);
					overrun_flags.set_flag(TextServer::OVERRUN_JUSTIFICATION_AWARE);
					TS->shaped_text_overrun_trim_to_width(lines_rid[i], lw, overrun_flags);
					TS->shaped_text_fit_to_width(lines_rid[i], lw, jst_flags | TextServer::JUSTIFICATION_CONSTRAIN_ELLIPSIS);
				} else {
					TS->shaped_text_overrun_trim_to_width(lines_rid[i], lw, overrun_flags);
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
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= (int)lines_rid.size(), RID());
	return lines_rid[p_line];
}

RID TextParagraph::get_dropcap_rid() const {
	return dropcap_rid;
}

void TextParagraph::clear() {
	_THREAD_SAFE_METHOD_

	for (int i = 0; i < (int)lines_rid.size(); i++) {
		TS->free_rid(lines_rid[i]);
	}
	lines_rid.clear();
	TS->shaped_text_clear(rid);
	TS->shaped_text_clear(dropcap_rid);
}

void TextParagraph::set_preserve_invalid(bool p_enabled) {
	_THREAD_SAFE_METHOD_

	TS->shaped_text_set_preserve_invalid(rid, p_enabled);
	TS->shaped_text_set_preserve_invalid(dropcap_rid, p_enabled);
	lines_dirty = true;
	lines_offsets_dirty = true;
}

bool TextParagraph::get_preserve_invalid() const {
	_THREAD_SAFE_METHOD_

	return TS->shaped_text_get_preserve_invalid(rid);
}

void TextParagraph::set_preserve_control(bool p_enabled) {
	_THREAD_SAFE_METHOD_

	TS->shaped_text_set_preserve_control(rid, p_enabled);
	TS->shaped_text_set_preserve_control(dropcap_rid, p_enabled);
	lines_dirty = true;
	lines_offsets_dirty = true;
}

bool TextParagraph::get_preserve_control() const {
	_THREAD_SAFE_METHOD_

	return TS->shaped_text_get_preserve_control(rid);
}

void TextParagraph::set_direction(TextServer::Direction p_direction) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND((int)p_direction < 0 || (int)p_direction > 2);

	TS->shaped_text_set_direction(rid, p_direction);
	TS->shaped_text_set_direction(dropcap_rid, p_direction);
	lines_dirty = true;
	lines_offsets_dirty = true;
}

TextServer::Direction TextParagraph::get_direction() const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	return TS->shaped_text_get_direction(rid);
}

void TextParagraph::set_custom_punctuation(const String &p_punct) {
	_THREAD_SAFE_METHOD_

	TS->shaped_text_set_custom_punctuation(rid, p_punct);
	lines_dirty = true;
	lines_offsets_dirty = true;
}

String TextParagraph::get_custom_punctuation() const {
	_THREAD_SAFE_METHOD_

	return TS->shaped_text_get_custom_punctuation(rid);
}

void TextParagraph::set_orientation(TextServer::Orientation p_orientation) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND((int)p_orientation < 0 || (int)p_orientation > 3);

	TS->shaped_text_set_orientation(rid, p_orientation);
	TS->shaped_text_set_orientation(dropcap_rid, p_orientation);
	lines_dirty = true;
	lines_offsets_dirty = true;
}

TextServer::Orientation TextParagraph::get_orientation() const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	return TS->shaped_text_get_orientation(rid);
}

bool TextParagraph::set_dropcap(const String &p_text, const Ref<Font> &p_font, int p_font_size, const Rect2 &p_dropcap_margins, const String &p_language) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(p_font.is_null(), false);
	TS->shaped_text_clear(dropcap_rid);
	dropcap_margins = p_dropcap_margins;
	bool res = TS->shaped_text_add_string(dropcap_rid, p_text, p_font->get_rids(), p_font_size, p_font->get_opentype_features(), p_language);
	for (int i = 0; i < TextServer::SPACING_MAX; i++) {
		TS->shaped_text_set_spacing(dropcap_rid, TextServer::SpacingType(i), p_font->get_spacing(TextServer::SpacingType(i)));
	}
	lines_dirty = true;
	lines_offsets_dirty = true;
	return res;
}

void TextParagraph::clear_dropcap() {
	_THREAD_SAFE_METHOD_
	dropcap_margins = Rect2();
	TS->shaped_text_clear(dropcap_rid);
	lines_dirty = true;
	lines_offsets_dirty = true;
}

int TextParagraph::get_span_count() const {
	_THREAD_SAFE_METHOD_
	return TS->shaped_get_span_count(rid);
}

void TextParagraph::update_span_font(int p_span, const Ref<Font> &p_font, int p_font_size) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND(p_font.is_null());

	TS->shaped_set_span_update_font(rid, p_span, p_font->get_rids(), p_font_size, p_font->get_opentype_features());
	lines_dirty = true;
	lines_offsets_dirty = true;
}

bool TextParagraph::add_string(const String &p_text, const Ref<Font> &p_font, int p_font_size, const String &p_language, const Variant &p_meta) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(p_font.is_null(), false);
	bool res = TS->shaped_text_add_string(rid, p_text, p_font->get_rids(), p_font_size, p_font->get_opentype_features(), p_language, p_meta);
	for (int i = 0; i < TextServer::SPACING_MAX; i++) {
		TS->shaped_text_set_spacing(rid, TextServer::SpacingType(i), p_font->get_spacing(TextServer::SpacingType(i)));
	}
	lines_dirty = true;
	lines_offsets_dirty = true;
	return res;
}

void TextParagraph::set_bidi_override(const Array &p_override) {
	_THREAD_SAFE_METHOD_

	TS->shaped_text_set_bidi_override(rid, p_override);
	lines_dirty = true;
	lines_offsets_dirty = true;
}

bool TextParagraph::add_object(Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align, int p_length, float p_baseline) {
	_THREAD_SAFE_METHOD_

	bool res = TS->shaped_text_add_object(rid, p_key, p_size, p_inline_align, p_length, p_baseline);
	lines_dirty = true;
	lines_offsets_dirty = true;
	return res;
}

bool TextParagraph::resize_object(Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align, float p_baseline) {
	_THREAD_SAFE_METHOD_

	bool res = TS->shaped_text_resize_object(rid, p_key, p_size, p_inline_align, p_baseline);
	lines_dirty = true;
	lines_offsets_dirty = true;
	return res;
}

void TextParagraph::set_horizontal_alignment(HorizontalAlignment p_alignment) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX((int)p_alignment, 4);
	if (horizontal_alignment != p_alignment) {
		if (horizontal_alignment == HORIZONTAL_ALIGNMENT_FILL || p_alignment == HORIZONTAL_ALIGNMENT_FILL) {
			horizontal_alignment = p_alignment;
			lines_dirty = true;
		} else {
			horizontal_alignment = p_alignment;
		}
		lines_offsets_dirty = true;
	}
}

HorizontalAlignment TextParagraph::get_horizontal_alignment() const {
	return horizontal_alignment;
}

void TextParagraph::set_vertical_alignment(VerticalAlignment p_alignment) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX((int)p_alignment, 4);
	if (vertical_alignment != p_alignment) {
		lines_offsets_dirty = true;
		vertical_alignment = p_alignment;
	}
}

VerticalAlignment TextParagraph::get_vertical_alignment() const {
	return vertical_alignment;
}

void TextParagraph::set_uniform_line_height(bool p_enabled) {
	_THREAD_SAFE_METHOD_

	if (uniform_line_height != p_enabled) {
		lines_offsets_dirty = true;
		uniform_line_height = p_enabled;
	}
}

bool TextParagraph::get_uniform_line_height() const {
	return uniform_line_height;
}

void TextParagraph::set_invert_line_order(bool p_enabled) {
	_THREAD_SAFE_METHOD_

	if (invert_line_order != p_enabled) {
		lines_offsets_dirty = true;
		invert_line_order = p_enabled;
	}
}

bool TextParagraph::get_invert_line_order() const {
	return invert_line_order;
}

void TextParagraph::set_clip(bool p_enabled) {
	clip = p_enabled;
}

bool TextParagraph::get_clip() const {
	return clip;
}

void TextParagraph::tab_align(const Vector<float> &p_tab_stops) {
	_THREAD_SAFE_METHOD_

	tab_stops = p_tab_stops;
	lines_dirty = true;
	lines_offsets_dirty = true;
}

void TextParagraph::set_justification_flags(BitField<TextServer::JustificationFlag> p_flags) {
	_THREAD_SAFE_METHOD_

	if (jst_flags != p_flags) {
		jst_flags = p_flags;
		lines_dirty = true;
		lines_offsets_dirty = true;
	}
}

BitField<TextServer::JustificationFlag> TextParagraph::get_justification_flags() const {
	return jst_flags;
}

void TextParagraph::set_break_flags(BitField<TextServer::LineBreakFlag> p_flags) {
	_THREAD_SAFE_METHOD_

	if (brk_flags != p_flags) {
		brk_flags = p_flags;
		lines_dirty = true;
		lines_offsets_dirty = true;
	}
}

BitField<TextServer::LineBreakFlag> TextParagraph::get_break_flags() const {
	return brk_flags;
}

void TextParagraph::set_text_overrun_behavior(TextServer::OverrunBehavior p_behavior) {
	_THREAD_SAFE_METHOD_

	if (overrun_behavior != p_behavior) {
		overrun_behavior = p_behavior;
		lines_dirty = true;
		lines_offsets_dirty = true;
	}
}

TextServer::OverrunBehavior TextParagraph::get_text_overrun_behavior() const {
	return overrun_behavior;
}

void TextParagraph::set_width(float p_width) {
	_THREAD_SAFE_METHOD_

	if (width != p_width) {
		width = p_width;
		lines_dirty = true;
		lines_offsets_dirty = true;
	}
}

float TextParagraph::get_width() const {
	return width;
}

void TextParagraph::set_height(float p_height) {
	_THREAD_SAFE_METHOD_

	if (height != p_height) {
		height = p_height;
		lines_offsets_dirty = true;
	}
}

float TextParagraph::get_height() const {
	return height;
}

void TextParagraph::set_extra_line_spacing(float p_spacing) {
	_THREAD_SAFE_METHOD_

	if (line_spacing != p_spacing) {
		line_spacing = p_spacing;
		lines_offsets_dirty = true;
	}
}

float TextParagraph::get_extra_line_spacing() const {
	return line_spacing;
}

Size2 TextParagraph::get_non_wrapped_size() const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
		return Size2(TS->shaped_text_get_size(rid).x, TS->shaped_text_get_size(rid).y);
	} else {
		return Size2(TS->shaped_text_get_size(rid).x, TS->shaped_text_get_size(rid).y);
	}
}

Size2 TextParagraph::get_size() const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	Size2 size;
	int visible_lines = (max_lines_visible >= 0) ? MIN(max_lines_visible, (int)lines_rid.size()) : (int)lines_rid.size();
	int last_line = MIN((int)lines_rid.size(), visible_lines + lines_skipped);
	TextServer::Orientation orientation = TS->shaped_text_get_orientation(rid);

	for (int i = lines_skipped; i < last_line; i++) {
		Size2 lsize = TS->shaped_text_get_size(lines_rid[i]);
		if (orientation == TextServer::ORIENTATION_HORIZONTAL) {
			size.x = MAX(size.x, lsize.x);
			size.y += lsize.y;
			if (i < last_line - 1) {
				size.y += line_spacing;
			}
		} else {
			size.y = MAX(size.y, lsize.y);
			size.x += lsize.x;
			if (i < last_line - 1) {
				size.x += line_spacing;
			}
		}
	}
	return size;
}

int TextParagraph::get_line_count() const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	return (int)lines_rid.size();
}

int TextParagraph::get_visible_line_count() const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	const_cast<TextParagraph *>(this)->_update_line_offsets();

	TextServer::Orientation orientation = TS->shaped_text_get_orientation(rid);

	int lines_visible = (max_lines_visible >= 0) ? MIN(max_lines_visible, (int)lines_rid.size()) : (int)lines_rid.size();
	int last_line = MIN((int)lines_rid.size(), lines_visible + lines_skipped);

	if (clip) {
		int lines_visible_clip = 0;
		for (int i = lines_skipped; i < last_line; i++) {
			Vector2 ofs = get_line_offset(i);
			bool clip_line = false;
			if (height > 0) {
				if (orientation == TextServer::ORIENTATION_HORIZONTAL) {
					clip_line = ((ofs.y - TS->shaped_text_get_ascent(lines_rid[i])) < 0.0 || (ofs.y + TS->shaped_text_get_descent(lines_rid[i])) > height);
				} else {
					clip_line = ((ofs.x - TS->shaped_text_get_descent(lines_rid[i])) < 0.0 || (ofs.x + TS->shaped_text_get_ascent(lines_rid[i])) > height);
				}
			}
			if (!clip_line) {
				lines_visible_clip++;
			}
		}
		return lines_visible_clip;
	} else {
		return lines_visible;
	}
}

void TextParagraph::set_max_lines_visible(int p_lines) {
	_THREAD_SAFE_METHOD_

	if (p_lines != max_lines_visible) {
		max_lines_visible = p_lines;
		lines_dirty = true;
		lines_offsets_dirty = true;
	}
}

int TextParagraph::get_max_lines_visible() const {
	return max_lines_visible;
}

void TextParagraph::set_lines_skipped(int p_lines) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND(p_lines < 0);

	if (p_lines != lines_skipped) {
		lines_skipped = p_lines;
		lines_offsets_dirty = true;
	}
}

int TextParagraph::get_lines_skipped() const {
	return lines_skipped;
}

Array TextParagraph::get_line_objects(int p_line) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= (int)lines_rid.size(), Array());
	return TS->shaped_text_get_objects(lines_rid[p_line]);
}

void TextParagraph::_update_line_offsets() {
	if (lines_offsets_dirty) {
		_shape_lines();

		vsep = line_spacing;
		line_offsets.clear();
		line_offsets.resize(lines_rid.size());

		TextServer::Orientation orientation = TS->shaped_text_get_orientation(rid);

		Size2 dc_vb = TS->shaped_text_get_vertical_bounds(dropcap_rid);
		float dc_width = TS->shaped_text_get_width(dropcap_rid) + dropcap_margins.position.x + dropcap_margins.size.x;

		int visible_lines = (max_lines_visible >= 0) ? MIN(max_lines_visible, (int)lines_rid.size()) : (int)lines_rid.size();
		int last_line = MIN((int)lines_rid.size(), visible_lines + lines_skipped);

		int dc_max_line_width = 0;
		for (int i = 0; i < dropcap_lines; i++) {
			dc_max_line_width = MAX(dc_max_line_width, TS->shaped_text_get_width(lines_rid[i]));
		}

		int max_line_ascent = 0;
		int max_line_descent = 0;
		int max_line_width = dc_width + dc_max_line_width;
		int total_h = 0;
		for (int i = 0; i < (int)lines_rid.size(); i++) {
			max_line_width = MAX(max_line_width, TS->shaped_text_get_width(lines_rid[i]));
			max_line_ascent = MAX(max_line_ascent, TS->shaped_text_get_ascent(lines_rid[i]));
			max_line_descent = MAX(max_line_descent, TS->shaped_text_get_descent(lines_rid[i]));
			if (!uniform_line_height && i >= lines_skipped && i < last_line) {
				total_h += TS->shaped_text_get_ascent(lines_rid[i]) + TS->shaped_text_get_descent(lines_rid[i]);
				if (i != last_line - 1) {
					total_h += line_spacing;
				}
			}
		}

		if (uniform_line_height) {
			line_height = max_line_ascent + max_line_descent;
			total_h = (last_line - lines_skipped) * (max_line_ascent + max_line_descent) + (last_line - lines_skipped - 1) * line_spacing;
		} else {
			line_height = 0.0;
		}

		Vector2 ofs;
		if (visible_lines > 0 && height > 0) {
			switch (vertical_alignment) {
				case VERTICAL_ALIGNMENT_FILL: {
					if (visible_lines > 1) {
						vsep = (height - total_h) / (visible_lines - 1);
					} else {
						vsep = 0;
					}
				}
					[[fallthrough]];
				case VERTICAL_ALIGNMENT_TOP: {
					switch (orientation) {
						case TextServer::ORIENTATION_HORIZONTAL: {
							if (invert_line_order) {
								ofs.y = height;
							} else {
								ofs.y = 0.0;
							}
						} break;
						case TextServer::ORIENTATION_VERTICAL_UPRIGHT:
						case TextServer::ORIENTATION_VERTICAL_MIXED:
						case TextServer::ORIENTATION_VERTICAL_SIDEWAYS: {
							if (invert_line_order) {
								ofs.x = 0.0;
							} else {
								ofs.x = height;
							}
						} break;
					}
				} break;
				case VERTICAL_ALIGNMENT_CENTER: {
					switch (orientation) {
						case TextServer::ORIENTATION_HORIZONTAL: {
							if (invert_line_order) {
								ofs.y = (height + total_h) / 2.0;
							} else {
								ofs.y = (height - total_h) / 2.0;
							}
						} break;
						case TextServer::ORIENTATION_VERTICAL_UPRIGHT:
						case TextServer::ORIENTATION_VERTICAL_MIXED:
						case TextServer::ORIENTATION_VERTICAL_SIDEWAYS: {
							if (invert_line_order) {
								ofs.x = (height - total_h) / 2;
							} else {
								ofs.x = (height + total_h) / 2;
							}
						} break;
					}
				} break;
				case VERTICAL_ALIGNMENT_BOTTOM: {
					switch (orientation) {
						case TextServer::ORIENTATION_HORIZONTAL: {
							if (invert_line_order) {
								ofs.y = total_h;
							} else {
								ofs.y = height - total_h;
							}
						} break;
						case TextServer::ORIENTATION_VERTICAL_UPRIGHT:
						case TextServer::ORIENTATION_VERTICAL_MIXED:
						case TextServer::ORIENTATION_VERTICAL_SIDEWAYS: {
							if (invert_line_order) {
								ofs.x = height - total_h;
							} else {
								ofs.x = total_h;
							}
						} break;
					}
				} break;
			}
		}
		switch (orientation) {
			case TextServer::ORIENTATION_HORIZONTAL: {
				if (invert_line_order) {
					dc_offset.y = ofs.y + dc_vb.y + dropcap_margins.size.y;
				} else {
					dc_offset.y = ofs.y + dc_vb.x + dropcap_margins.position.y;
				}
				if (horizontal_alignment == HORIZONTAL_ALIGNMENT_CENTER) {
					dc_offset.x = (width - (dc_width + dc_max_line_width)) / 2.0;
				} else if (horizontal_alignment == HORIZONTAL_ALIGNMENT_RIGHT) {
					dc_offset.x = (width - (dc_width + dc_max_line_width));
				}
			} break;
			case TextServer::ORIENTATION_VERTICAL_UPRIGHT:
			case TextServer::ORIENTATION_VERTICAL_MIXED:
			case TextServer::ORIENTATION_VERTICAL_SIDEWAYS: {
				if (invert_line_order) {
					dc_offset.x = ofs.x + dc_vb.y + dropcap_margins.size.y;
				} else {
					dc_offset.x = ofs.x - (dc_vb.x + dropcap_margins.position.y);
				}
				if (horizontal_alignment == HORIZONTAL_ALIGNMENT_CENTER) {
					dc_offset.y = (width - (dc_width + dc_max_line_width)) / 2.0;
				} else if (horizontal_alignment == HORIZONTAL_ALIGNMENT_RIGHT) {
					dc_offset.y = (width - (dc_width + dc_max_line_width));
				}
			} break;
		}

		for (int i = lines_skipped; i < last_line; i++) {
			if (orientation == TextServer::ORIENTATION_HORIZONTAL) {
				if (invert_line_order) {
					ofs.y -= (uniform_line_height) ? max_line_descent : TS->shaped_text_get_descent(lines_rid[i]);
				} else {
					ofs.y += (uniform_line_height) ? max_line_ascent : TS->shaped_text_get_ascent(lines_rid[i]);
				}
			} else {
				if (invert_line_order) {
					ofs.x += (uniform_line_height) ? max_line_descent : TS->shaped_text_get_descent(lines_rid[i]);
				} else {
					ofs.x -= (uniform_line_height) ? max_line_ascent : TS->shaped_text_get_ascent(lines_rid[i]);
				}
			}
			float line_width = TS->shaped_text_get_width(lines_rid[i]);
			float avail_width = width;
			float start_off = 0.0;
			if (i < dropcap_lines) {
				avail_width = dc_max_line_width;
				if (orientation == TextServer::ORIENTATION_HORIZONTAL) {
					start_off = dc_offset.x + dc_width;
				} else {
					start_off = dc_offset.y + dc_width;
				}
			}
			if (avail_width > 0) {
				switch (horizontal_alignment) {
					case HORIZONTAL_ALIGNMENT_FILL:
					case HORIZONTAL_ALIGNMENT_LEFT: {
						switch (orientation) {
							case TextServer::ORIENTATION_HORIZONTAL: {
								ofs.x = start_off;
							} break;
							case TextServer::ORIENTATION_VERTICAL_UPRIGHT:
							case TextServer::ORIENTATION_VERTICAL_MIXED:
							case TextServer::ORIENTATION_VERTICAL_SIDEWAYS: {
								ofs.y = start_off;
							} break;
						}
					} break;
					case HORIZONTAL_ALIGNMENT_CENTER: {
						switch (orientation) {
							case TextServer::ORIENTATION_HORIZONTAL: {
								ofs.x = start_off + (avail_width - line_width) / 2.0;
							} break;
							case TextServer::ORIENTATION_VERTICAL_UPRIGHT:
							case TextServer::ORIENTATION_VERTICAL_MIXED:
							case TextServer::ORIENTATION_VERTICAL_SIDEWAYS: {
								ofs.y = start_off + (avail_width - line_width) / 2.0;
							} break;
						}
					} break;
					case HORIZONTAL_ALIGNMENT_RIGHT: {
						switch (orientation) {
							case TextServer::ORIENTATION_HORIZONTAL: {
								ofs.x = start_off + (avail_width - line_width);
							} break;
							case TextServer::ORIENTATION_VERTICAL_UPRIGHT:
							case TextServer::ORIENTATION_VERTICAL_MIXED:
							case TextServer::ORIENTATION_VERTICAL_SIDEWAYS: {
								ofs.y = start_off + (avail_width - line_width);
							} break;
						}
					} break;
				}
			}
			line_offsets.write[i] = ofs;
			if (orientation == TextServer::ORIENTATION_HORIZONTAL) {
				if (invert_line_order) {
					ofs.y -= (uniform_line_height) ? (max_line_ascent + vsep) : (TS->shaped_text_get_ascent(lines_rid[i]) + vsep);
				} else {
					ofs.y += (uniform_line_height) ? (max_line_descent + vsep) : (TS->shaped_text_get_descent(lines_rid[i]) + vsep);
				}
			} else {
				if (invert_line_order) {
					ofs.x += (uniform_line_height) ? (max_line_ascent + vsep) : (TS->shaped_text_get_ascent(lines_rid[i]) + vsep);
				} else {
					ofs.x -= (uniform_line_height) ? (max_line_descent + vsep) : (TS->shaped_text_get_descent(lines_rid[i]) + vsep);
				}
			}
		}
		if (orientation == TextServer::ORIENTATION_HORIZONTAL) {
			dc_offset.x += dropcap_margins.position.x;
		} else {
			dc_offset.y += dropcap_margins.position.x;
		}

		lines_offsets_dirty = false;
	}
}

Vector2 TextParagraph::get_dropcap_offset() const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_update_line_offsets();
	return dc_offset;
}

Vector2 TextParagraph::get_line_offset(int p_line) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_update_line_offsets();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= (int)lines_rid.size(), Vector2());
	return line_offsets[p_line];
}

Rect2 TextParagraph::get_line_object_rect(int p_line, Variant p_key) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= (int)lines_rid.size(), Rect2());

	Vector2 ofs = get_line_offset(p_line);
	Rect2 rect = TS->shaped_text_get_object_rect(lines_rid[p_line], p_key);
	rect.position += ofs;

	return rect;
}

Size2 TextParagraph::get_line_size(int p_line) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= (int)lines_rid.size(), Size2());
	if (TS->shaped_text_get_orientation(lines_rid[p_line]) == TextServer::ORIENTATION_HORIZONTAL) {
		return Size2(TS->shaped_text_get_size(lines_rid[p_line]).x, (uniform_line_height) ? line_height : TS->shaped_text_get_size(lines_rid[p_line]).y);
	} else {
		return Size2((uniform_line_height) ? line_height : TS->shaped_text_get_size(lines_rid[p_line]).x, TS->shaped_text_get_size(lines_rid[p_line]).y);
	}
}

Vector2i TextParagraph::get_line_range(int p_line) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= (int)lines_rid.size(), Vector2i());
	return TS->shaped_text_get_range(lines_rid[p_line]);
}

float TextParagraph::get_line_ascent(int p_line) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= (int)lines_rid.size(), 0.f);
	return TS->shaped_text_get_ascent(lines_rid[p_line]);
}

float TextParagraph::get_line_descent(int p_line) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= (int)lines_rid.size(), 0.f);
	return TS->shaped_text_get_descent(lines_rid[p_line]);
}

float TextParagraph::get_line_width(int p_line) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= (int)lines_rid.size(), 0.f);
	return TS->shaped_text_get_width(lines_rid[p_line]);
}

float TextParagraph::get_line_underline_position(int p_line) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= (int)lines_rid.size(), 0.f);
	return TS->shaped_text_get_underline_position(lines_rid[p_line]);
}

float TextParagraph::get_line_underline_thickness(int p_line) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND_V(p_line < 0 || p_line >= (int)lines_rid.size(), 0.f);
	return TS->shaped_text_get_underline_thickness(lines_rid[p_line]);
}

Size2 TextParagraph::get_dropcap_size() const {
	_THREAD_SAFE_METHOD_

	return TS->shaped_text_get_size(dropcap_rid) + dropcap_margins.size + dropcap_margins.position;
}

int TextParagraph::get_dropcap_lines() const {
	return dropcap_lines;
}

bool TextParagraph::has_invalid_glyphs() const {
	const Glyph *glyph = TS->shaped_text_get_glyphs(dropcap_rid);
	int64_t glyph_count = TS->shaped_text_get_glyph_count(dropcap_rid);
	for (int64_t i = 0; i < glyph_count; i++) {
		if (glyph[i].font_rid == RID()) {
			return true;
		}
	}

	int lines_visible = (max_lines_visible >= 0) ? MIN(max_lines_visible, (int)lines_rid.size()) : (int)lines_rid.size();
	int last_line = MIN((int)lines_rid.size(), lines_visible + lines_skipped);
	for (int i = lines_skipped; i < last_line; i++) {
		glyph = TS->shaped_text_get_glyphs(lines_rid[i]);
		glyph_count = TS->shaped_text_get_glyph_count(lines_rid[i]);
		for (int64_t j = 0; j < glyph_count; j++) {
			if (glyph[j].font_rid == RID()) {
				return true;
			}
		}
	}
	return false;
}

int TextParagraph::get_glyph_count() const {
	int count = TS->shaped_text_get_glyph_count(dropcap_rid) + TS->shaped_text_get_ellipsis_glyph_count(dropcap_rid);

	int lines_visible = (max_lines_visible >= 0) ? MIN(max_lines_visible, (int)lines_rid.size()) : (int)lines_rid.size();
	int last_line = MIN((int)lines_rid.size(), lines_visible + lines_skipped);
	for (int i = lines_skipped; i < last_line; i++) {
		count += TS->shaped_text_get_glyph_count(lines_rid[i]) + TS->shaped_text_get_ellipsis_glyph_count(lines_rid[i]);
	}

	return count;
}

void TextParagraph::draw_underline_custom(const Vector2 &p_pos, TextServer::LinePosition p_line, int p_start, int p_end, std::function<bool(const Rect2 &, int)> p_draw_fn) const {
	TextServer::Orientation orientation = TS->shaped_text_get_orientation(rid);

	draw_custom(
			p_pos,
			[&](const Glyph &p_gl, const Vector2 &p_ofs, int p_line_id) {
				if (p_gl.font_rid != RID() && p_gl.start >= p_start && p_gl.end <= p_end) {
					float l_ofs = 0.0;
					switch (p_line) {
						case TextServer::UNDERLINE: {
							l_ofs = TS->font_get_underline_position(p_gl.font_rid, p_gl.font_size);
						} break;
						case TextServer::OVERDERLINE: {
							l_ofs = -TS->font_get_ascent(p_gl.font_rid, p_gl.font_size);
						} break;
						case TextServer::STRIKETHROUGH: {
							l_ofs = -(TS->font_get_ascent(p_gl.font_rid, p_gl.font_size) + TS->font_get_descent(p_gl.font_rid, p_gl.font_size)) / 2.0;
						} break;
					}
					float l_h = TS->font_get_underline_thickness(p_gl.font_rid, p_gl.font_size);
					float l_w = p_gl.advance;
					if (orientation == TextServer::ORIENTATION_HORIZONTAL) {
						return p_draw_fn(Rect2(p_ofs + Vector2(0, l_ofs), Vector2(l_w, l_h)), p_line_id);
					} else {
						return p_draw_fn(Rect2(p_ofs + Vector2(-l_ofs, 0), Vector2(l_h, l_w)), p_line_id);
					}
				}
				return true;
			});
}

void TextParagraph::_draw_underline_custom(RID p_canvas, const Vector2 &p_pos, TextServer::LinePosition p_line, int p_start, int p_end, const Callable &p_callback) const {
	draw_underline_custom(
			p_pos,
			p_line,
			p_start,
			p_end,
			[&](const Rect2 &p_rect, int p_line_id) {
				Variant args[] = { p_rect, p_canvas, p_line_id };
				const Variant *args_ptr[] = { &args[0], &args[1], &args[2] };
				Variant ret;
				Callable::CallError ce;
				p_callback.callp(args_ptr, 3, ret, ce);
				if (ce.error != Callable::CallError::CALL_OK) {
					ERR_PRINT_ONCE("Error calling glyph draw callback method " + Variant::get_callable_error_text(p_callback, args_ptr, 3, ce));
				}
				return ret.operator bool();
			});
}

void TextParagraph::draw_custom(const Vector2 &p_pos, std::function<bool(const Glyph &, const Vector2 &, int)> p_draw_fn) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	const_cast<TextParagraph *>(this)->_update_line_offsets();

	TextServer::Orientation orientation = TS->shaped_text_get_orientation(rid);

	// Draw dropcap.
	if (lines_skipped < dropcap_lines) {
		Vector2 ofs = dc_offset;
		bool clip_line = false;
		float clip_l = -1.0;
		float clip_r = -1.0;
		if (clip) {
			if (height > 0) {
				if (orientation == TextServer::ORIENTATION_HORIZONTAL) {
					clip_line = ((ofs.y + TS->shaped_text_get_vertical_bounds(dropcap_rid).y) < 0.0 || (ofs.y - TS->shaped_text_get_vertical_bounds(dropcap_rid).x) > height);
				} else {
					clip_line = ((ofs.x + TS->shaped_text_get_vertical_bounds(dropcap_rid).x) < 0.0 || (ofs.x - TS->shaped_text_get_vertical_bounds(dropcap_rid).y) > height);
				}
			}
			if (width > 0.0) {
				if (orientation == TextServer::ORIENTATION_HORIZONTAL) {
					clip_l = Math::floor(MAX(0.0, -ofs.x));
					clip_r = Math::ceil(width + clip_l);
				} else {
					clip_l = Math::floor(MAX(0, -ofs.y));
					clip_r = Math::ceil(width + clip_l);
				}
			}
		}
		if (!clip_line) {
			TS->shaped_text_draw_custom(dropcap_rid, ofs + p_pos, clip_l, clip_r, p_draw_fn, -1);
		}
	}

	int lines_visible = (max_lines_visible >= 0) ? MIN(max_lines_visible, (int)lines_rid.size()) : (int)lines_rid.size();
	int last_line = MIN((int)lines_rid.size(), lines_visible + lines_skipped);

	for (int i = lines_skipped; i < last_line; i++) {
		Vector2 ofs = get_line_offset(i);
		float clip_l = -1.0;
		float clip_r = -1.0;
		if (clip) {
			if (height > 0.0) {
				if (orientation == TextServer::ORIENTATION_HORIZONTAL) {
					if ((ofs.y + TS->shaped_text_get_descent(lines_rid[i])) < 0.0 || (ofs.y - TS->shaped_text_get_ascent(lines_rid[i])) > height) {
						continue;
					}
				} else {
					if ((ofs.x + TS->shaped_text_get_ascent(lines_rid[i])) < 0.0 || (ofs.x - TS->shaped_text_get_descent(lines_rid[i])) > height) {
						continue;
					}
				}
			}
			if (width > 0.0) {
				if (orientation == TextServer::ORIENTATION_HORIZONTAL) {
					clip_l = Math::floor(MAX(0, -ofs.x));
					clip_r = Math::ceil(width + clip_l);
				} else {
					clip_l = Math::floor(MAX(0, -ofs.y));
					clip_r = Math::ceil(width + clip_l);
				}
			}
		}
		TS->shaped_text_draw_custom(lines_rid[i], ofs + p_pos, clip_l, clip_r, p_draw_fn, i);
	}
}

void TextParagraph::_draw_custom(RID p_canvas, const Vector2 &p_pos, const Callable &p_callback) const {
	draw_custom(
			p_pos,
			[&](const Glyph &p_gl, const Vector2 &p_ofs, int p_line_id) {
				Dictionary glyph;

				glyph["start"] = p_gl.start;
				glyph["end"] = p_gl.end;
				glyph["repeat"] = p_gl.repeat;
				glyph["count"] = p_gl.count;
				glyph["flags"] = p_gl.flags;
				glyph["offset"] = Vector2(p_gl.x_off, p_gl.y_off);
				glyph["advance"] = p_gl.advance;
				glyph["font_rid"] = p_gl.font_rid;
				glyph["font_size"] = p_gl.font_size;
				glyph["index"] = p_gl.index;

				Variant args[] = { glyph, p_ofs, p_canvas, p_line_id };
				const Variant *args_ptr[] = { &args[0], &args[1], &args[2], &args[3] };
				Variant ret;
				Callable::CallError ce;
				p_callback.callp(args_ptr, 4, ret, ce);
				if (ce.error != Callable::CallError::CALL_OK) {
					ERR_PRINT_ONCE("Error calling glyph draw callback method " + Variant::get_callable_error_text(p_callback, args_ptr, 4, ce));
				}
				return ret.operator bool();
			});
}

void TextParagraph::draw(RID p_canvas, const Vector2 &p_pos, const Color &p_color, const Color &p_dc_color) const {
	bool hex_codes = TS->shaped_text_get_preserve_control(rid) || TS->shaped_text_get_preserve_invalid(rid);
	draw_custom(
			p_pos,
			[&](const Glyph &p_gl, const Vector2 &p_ofs, int p_line_id) {
				if (p_gl.font_rid != RID()) {
					TS->font_draw_glyph(p_gl.font_rid, p_canvas, p_gl.font_size, p_ofs, p_gl.index, (p_line_id == -1) ? p_dc_color : p_color);
				} else if (hex_codes && ((p_gl.flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL)) {
					TS->draw_hex_code_box(p_canvas, p_gl.font_size, p_ofs, p_gl.index, (p_line_id == -1) ? p_dc_color : p_color);
				}
				return true;
			});
}

void TextParagraph::draw_outline(RID p_canvas, const Vector2 &p_pos, int p_outline_size, const Color &p_color, const Color &p_dc_color) const {
	draw_custom(
			p_pos,
			[&](const Glyph &p_gl, const Vector2 &p_ofs, int p_line_id) {
				if (p_gl.font_rid != RID()) {
					TS->font_draw_glyph_outline(p_gl.font_rid, p_canvas, p_gl.font_size, p_outline_size, p_ofs, p_gl.index, (p_line_id == -1) ? p_dc_color : p_color);
				}
				return true;
			});
}

int TextParagraph::hit_test(const Point2 &p_coords) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	const_cast<TextParagraph *>(this)->_update_line_offsets();

	TextServer::Orientation orientation = TS->shaped_text_get_orientation(rid);

	for (int i = 0; i < (int)lines_rid.size(); i++) {
		Vector2 ofs = get_line_offset(i);
		if (orientation == TextServer::ORIENTATION_HORIZONTAL) {
			if ((p_coords.y >= ofs.y) && (p_coords.y <= ofs.y + TS->shaped_text_get_size(lines_rid[i]).y)) {
				return TS->shaped_text_hit_test_position(lines_rid[i], p_coords.x);
			}
		} else {
			if ((p_coords.x >= ofs.x) && (p_coords.x <= ofs.x + TS->shaped_text_get_size(lines_rid[i]).x)) {
				return TS->shaped_text_hit_test_position(lines_rid[i], p_coords.y);
			}
		}
	}
	return TS->shaped_text_get_range(rid).y;
}

void TextParagraph::draw_dropcap(RID p_canvas, const Vector2 &p_pos, const Color &p_color) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();

	TS->shaped_text_draw(dropcap_rid, p_canvas, p_pos, -1, -1, p_color);
}

void TextParagraph::draw_dropcap_outline(RID p_canvas, const Vector2 &p_pos, int p_outline_size, const Color &p_color) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();

	TS->shaped_text_draw_outline(dropcap_rid, p_canvas, p_pos, -1, -1, p_outline_size, p_color);
}

void TextParagraph::draw_line(RID p_canvas, const Vector2 &p_pos, int p_line, const Color &p_color) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND(p_line < 0 || p_line >= (int)lines_rid.size());

	Vector2 ofs = p_pos;

	if (TS->shaped_text_get_orientation(lines_rid[p_line]) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(lines_rid[p_line]);
	} else {
		ofs.x += TS->shaped_text_get_ascent(lines_rid[p_line]);
	}
	return TS->shaped_text_draw(lines_rid[p_line], p_canvas, ofs, -1, -1, p_color);
}

void TextParagraph::draw_line_outline(RID p_canvas, const Vector2 &p_pos, int p_line, int p_outline_size, const Color &p_color) const {
	_THREAD_SAFE_METHOD_

	const_cast<TextParagraph *>(this)->_shape_lines();
	ERR_FAIL_COND(p_line < 0 || p_line >= (int)lines_rid.size());

	Vector2 ofs = p_pos;
	if (TS->shaped_text_get_orientation(lines_rid[p_line]) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(lines_rid[p_line]);
	} else {
		ofs.x += TS->shaped_text_get_ascent(lines_rid[p_line]);
	}
	return TS->shaped_text_draw_outline(lines_rid[p_line], p_canvas, ofs, -1, -1, p_outline_size, p_color);
}

TextParagraph::TextParagraph(const String &p_text, const Ref<Font> &p_font, int p_font_size, const String &p_language, float p_width, float p_height, TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	rid = TS->create_shaped_text(p_direction, p_orientation);
	if (p_font.is_valid()) {
		TS->shaped_text_add_string(rid, p_text, p_font->get_rids(), p_font_size, p_font->get_opentype_features(), p_language);
		for (int i = 0; i < TextServer::SPACING_MAX; i++) {
			TS->shaped_text_set_spacing(rid, TextServer::SpacingType(i), p_font->get_spacing(TextServer::SpacingType(i)));
		}
	}
	width = p_width;
	height = p_height;
}

TextParagraph::TextParagraph() {
	rid = TS->create_shaped_text();
	dropcap_rid = TS->create_shaped_text();
}

TextParagraph::~TextParagraph() {
	for (int i = 0; i < (int)lines_rid.size(); i++) {
		TS->free_rid(lines_rid[i]);
	}
	lines_rid.clear();
	TS->free_rid(rid);
	TS->free_rid(dropcap_rid);
}
