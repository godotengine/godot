/*************************************************************************/
/*  text_line.cpp                                                        */
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

#include "text_line.h"

void TextLine::_bind_methods() {
	ClassDB::bind_method(D_METHOD("clear"), &TextLine::clear);

	ClassDB::bind_method(D_METHOD("set_direction", "direction"), &TextLine::set_direction);
	ClassDB::bind_method(D_METHOD("get_direction"), &TextLine::get_direction);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "direction", PROPERTY_HINT_ENUM, "Auto,Left-to-right,Right-to-left"), "set_direction", "get_direction");

	ClassDB::bind_method(D_METHOD("set_orientation", "orientation"), &TextLine::set_orientation);
	ClassDB::bind_method(D_METHOD("get_orientation"), &TextLine::get_orientation);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "orientation", PROPERTY_HINT_ENUM, "Horizontal,Orientation"), "set_orientation", "get_orientation");

	ClassDB::bind_method(D_METHOD("set_preserve_invalid", "enabled"), &TextLine::set_preserve_invalid);
	ClassDB::bind_method(D_METHOD("get_preserve_invalid"), &TextLine::get_preserve_invalid);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "preserve_invalid"), "set_preserve_invalid", "get_preserve_invalid");

	ClassDB::bind_method(D_METHOD("set_preserve_control", "enabled"), &TextLine::set_preserve_control);
	ClassDB::bind_method(D_METHOD("get_preserve_control"), &TextLine::get_preserve_control);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "preserve_control"), "set_preserve_control", "get_preserve_control");

	ClassDB::bind_method(D_METHOD("set_bidi_override", "override"), &TextLine::set_bidi_override);

	ClassDB::bind_method(D_METHOD("get_span_count"), &TextLine::get_span_count);
	ClassDB::bind_method(D_METHOD("update_span_font", "span", "font", "font_size"), &TextLine::update_span_font);

	ClassDB::bind_method(D_METHOD("add_string", "text", "font", "font_size", "language", "meta"), &TextLine::add_string, DEFVAL(""), DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("add_object", "key", "size", "inline_align", "length", "baseline"), &TextLine::add_object, DEFVAL(INLINE_ALIGNMENT_CENTER), DEFVAL(1), DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("resize_object", "key", "size", "inline_align", "baseline"), &TextLine::resize_object, DEFVAL(INLINE_ALIGNMENT_CENTER), DEFVAL(0.0));

	ClassDB::bind_method(D_METHOD("set_width", "width"), &TextLine::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &TextLine::get_width);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width"), "set_width", "get_width");

	ClassDB::bind_method(D_METHOD("set_horizontal_alignment", "alignment"), &TextLine::set_horizontal_alignment);
	ClassDB::bind_method(D_METHOD("get_horizontal_alignment"), &TextLine::get_horizontal_alignment);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "alignment", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_horizontal_alignment", "get_horizontal_alignment");

	ClassDB::bind_method(D_METHOD("set_clip", "clip"), &TextLine::set_clip);
	ClassDB::bind_method(D_METHOD("get_clip"), &TextLine::get_clip);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip"), "set_clip", "get_clip");

	ClassDB::bind_method(D_METHOD("tab_align", "tab_stops"), &TextLine::tab_align);

	ClassDB::bind_method(D_METHOD("set_flags", "flags"), &TextLine::set_flags);
	ClassDB::bind_method(D_METHOD("get_flags"), &TextLine::get_flags);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "flags", PROPERTY_HINT_FLAGS, "Kashida Justification,Word Justification,Trim Edge Spaces After Justification,Justify Only After Last Tab,Constrain Ellipsis"), "set_flags", "get_flags");

	ClassDB::bind_method(D_METHOD("set_text_overrun_behavior", "overrun_behavior"), &TextLine::set_text_overrun_behavior);
	ClassDB::bind_method(D_METHOD("get_text_overrun_behavior"), &TextLine::get_text_overrun_behavior);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_overrun_behavior", PROPERTY_HINT_ENUM, "Trim Nothing,Trim Characters,Trim Words,Ellipsis,Word Ellipsis"), "set_text_overrun_behavior", "get_text_overrun_behavior");

	ClassDB::bind_method(D_METHOD("get_objects"), &TextLine::get_objects);
	ClassDB::bind_method(D_METHOD("get_object_rect", "key"), &TextLine::get_object_rect);

	ClassDB::bind_method(D_METHOD("get_size"), &TextLine::get_size);

	ClassDB::bind_method(D_METHOD("get_rid"), &TextLine::get_rid);

	ClassDB::bind_method(D_METHOD("get_line_ascent"), &TextLine::get_line_ascent);
	ClassDB::bind_method(D_METHOD("get_line_descent"), &TextLine::get_line_descent);
	ClassDB::bind_method(D_METHOD("get_line_width"), &TextLine::get_line_width);
	ClassDB::bind_method(D_METHOD("get_line_underline_position"), &TextLine::get_line_underline_position);
	ClassDB::bind_method(D_METHOD("get_line_underline_thickness"), &TextLine::get_line_underline_thickness);

	ClassDB::bind_method(D_METHOD("has_invalid_glyphs"), &TextLine::has_invalid_glyphs);
	ClassDB::bind_method(D_METHOD("get_glyph_count"), &TextLine::get_glyph_count);

	ClassDB::bind_method(D_METHOD("draw", "canvas", "pos", "color"), &TextLine::draw, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_outline", "canvas", "pos", "outline_size", "color"), &TextLine::draw_outline, DEFVAL(1), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("draw_custom", "canvas", "pos", "callback"), &TextLine::_draw_custom);

	ClassDB::bind_method(D_METHOD("draw_underline_custom", "canvas", "pos", "line_type", "start", "end", "callback"), &TextLine::_draw_underline_custom);

	ClassDB::bind_method(D_METHOD("hit_test", "coords"), &TextLine::hit_test);
}

void TextLine::_shape() {
	if (dirty) {
		if (!tab_stops.is_empty()) {
			TS->shaped_text_tab_align(rid, tab_stops);
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

			if (alignment == HORIZONTAL_ALIGNMENT_FILL) {
				TS->shaped_text_fit_to_width(rid, width, flags);
				overrun_flags.set_flag(TextServer::OVERRUN_JUSTIFICATION_AWARE);
				TS->shaped_text_overrun_trim_to_width(rid, width, overrun_flags);
			} else {
				TS->shaped_text_overrun_trim_to_width(rid, width, overrun_flags);
			}
		} else if (alignment == HORIZONTAL_ALIGNMENT_FILL) {
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

void TextLine::set_bidi_override(const Array &p_override) {
	TS->shaped_text_set_bidi_override(rid, p_override);
	dirty = true;
}

int TextLine::get_span_count() const {
	return TS->shaped_get_span_count(rid);
}

void TextLine::update_span_font(int p_span, const Ref<Font> &p_font, int p_font_size) {
	ERR_FAIL_COND(p_font.is_null());

	TS->shaped_set_span_update_font(rid, p_span, p_font->get_rids(), p_font_size, p_font->get_opentype_features());
}

bool TextLine::add_string(const String &p_text, const Ref<Font> &p_font, int p_font_size, const String &p_language, const Variant &p_meta) {
	ERR_FAIL_COND_V(p_font.is_null(), false);
	bool res = TS->shaped_text_add_string(rid, p_text, p_font->get_rids(), p_font_size, p_font->get_opentype_features(), p_language, p_meta);
	for (int i = 0; i < TextServer::SPACING_MAX; i++) {
		TS->shaped_text_set_spacing(rid, TextServer::SpacingType(i), p_font->get_spacing(TextServer::SpacingType(i)));
	}
	dirty = true;
	return res;
}

bool TextLine::add_object(Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align, int p_length, float p_baseline) {
	bool res = TS->shaped_text_add_object(rid, p_key, p_size, p_inline_align, p_length, p_baseline);
	dirty = true;
	return res;
}

bool TextLine::resize_object(Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align, float p_baseline) {
	const_cast<TextLine *>(this)->_shape();
	return TS->shaped_text_resize_object(rid, p_key, p_size, p_inline_align, p_baseline);
}

Array TextLine::get_objects() const {
	return TS->shaped_text_get_objects(rid);
}

Rect2 TextLine::get_object_rect(Variant p_key) const {
	Vector2 ofs;

	float length = TS->shaped_text_get_width(rid);
	if (width > 0) {
		switch (alignment) {
			case HORIZONTAL_ALIGNMENT_FILL:
			case HORIZONTAL_ALIGNMENT_LEFT:
				break;
			case HORIZONTAL_ALIGNMENT_CENTER: {
				if (length <= width) {
					if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
						ofs.x += Math::floor((width - length) / 2.0);
					} else {
						ofs.y += Math::floor((width - length) / 2.0);
					}
				} else if (TS->shaped_text_get_inferred_direction(rid) == TextServer::DIRECTION_RTL) {
					if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
						ofs.x += width - length;
					} else {
						ofs.y += width - length;
					}
				}
			} break;
			case HORIZONTAL_ALIGNMENT_RIGHT: {
				if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
					ofs.x += width - length;
				} else {
					ofs.y += width - length;
				}
			} break;
		}
	}
	if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(rid);
	} else {
		ofs.x += TS->shaped_text_get_ascent(rid);
	}

	Rect2 rect = TS->shaped_text_get_object_rect(rid, p_key);
	rect.position += ofs;

	return rect;
}

void TextLine::set_horizontal_alignment(HorizontalAlignment p_alignment) {
	if (alignment != p_alignment) {
		if (alignment == HORIZONTAL_ALIGNMENT_FILL || p_alignment == HORIZONTAL_ALIGNMENT_FILL) {
			alignment = p_alignment;
			dirty = true;
		} else {
			alignment = p_alignment;
		}
	}
}

HorizontalAlignment TextLine::get_horizontal_alignment() const {
	return alignment;
}

void TextLine::set_clip(bool p_enabled) {
	clip = p_enabled;
}

bool TextLine::get_clip() const {
	return clip;
}

void TextLine::tab_align(const Vector<float> &p_tab_stops) {
	tab_stops = p_tab_stops;
	dirty = true;
}

void TextLine::set_flags(BitField<TextServer::JustificationFlag> p_flags) {
	if (flags != p_flags) {
		flags = p_flags;
		dirty = true;
	}
}

BitField<TextServer::JustificationFlag> TextLine::get_flags() const {
	return flags;
}

void TextLine::set_text_overrun_behavior(TextServer::OverrunBehavior p_behavior) {
	if (overrun_behavior != p_behavior) {
		overrun_behavior = p_behavior;
		dirty = true;
	}
}

TextServer::OverrunBehavior TextLine::get_text_overrun_behavior() const {
	return overrun_behavior;
}

void TextLine::set_width(float p_width) {
	width = p_width;
	if (alignment == HORIZONTAL_ALIGNMENT_FILL || overrun_behavior != TextServer::OVERRUN_NO_TRIMMING) {
		dirty = true;
	}
}

float TextLine::get_width() const {
	return width;
}

Size2 TextLine::get_size() const {
	const_cast<TextLine *>(this)->_shape();
	return TS->shaped_text_get_size(rid);
}

float TextLine::get_line_ascent() const {
	const_cast<TextLine *>(this)->_shape();
	return TS->shaped_text_get_ascent(rid);
}

float TextLine::get_line_descent() const {
	const_cast<TextLine *>(this)->_shape();
	return TS->shaped_text_get_descent(rid);
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

bool TextLine::has_invalid_glyphs() const {
	const Glyph *glyph = TS->shaped_text_get_glyphs(rid);
	int64_t glyph_count = TS->shaped_text_get_glyph_count(rid);
	for (int64_t i = 0; i < glyph_count; i++) {
		if (glyph[i].font_rid == RID()) {
			return true;
		}
	}
	return false;
}

int TextLine::get_glyph_count() const {
	return TS->shaped_text_get_glyph_count(rid) + TS->shaped_text_get_ellipsis_glyph_count(rid);
}

void TextLine::draw_underline_custom(const Vector2 &p_pos, TextServer::LinePosition p_line, int p_start, int p_end, std::function<bool(const Rect2 &)> p_draw_fn) const {
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
						return p_draw_fn(Rect2(p_ofs + Vector2(0, l_ofs), Vector2(l_w, l_h)));
					} else {
						return p_draw_fn(Rect2(p_ofs + Vector2(-l_ofs, 0), Vector2(l_h, l_w)));
					}
				}
				return true;
			});
}

void TextLine::_draw_underline_custom(RID p_canvas, const Vector2 &p_pos, TextServer::LinePosition p_line, int p_start, int p_end, const Callable &p_callback) const {
	draw_underline_custom(
			p_pos,
			p_line,
			p_start,
			p_end,
			[&](const Rect2 &p_rect) {
				Variant args[] = { p_rect, p_canvas };
				const Variant *args_ptr[] = { &args[0], &args[1] };
				Variant ret;
				Callable::CallError ce;
				p_callback.callp(args_ptr, 2, ret, ce);
				if (ce.error != Callable::CallError::CALL_OK) {
					ERR_PRINT_ONCE("Error calling glyph draw callback method " + Variant::get_callable_error_text(p_callback, args_ptr, 2, ce));
				}
				return ret.operator bool();
			});
}

void TextLine::draw_custom(const Vector2 &p_pos, std::function<bool(const Glyph &, const Vector2 &, int)> p_draw_fn) const {
	const_cast<TextLine *>(this)->_shape();

	Vector2 ofs = p_pos;

	float length = TS->shaped_text_get_width(rid);
	if (width > 0) {
		switch (alignment) {
			case HORIZONTAL_ALIGNMENT_FILL:
			case HORIZONTAL_ALIGNMENT_LEFT:
				break;
			case HORIZONTAL_ALIGNMENT_CENTER: {
				if (length <= width) {
					if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
						ofs.x += Math::floor((width - length) / 2.0);
					} else {
						ofs.y += Math::floor((width - length) / 2.0);
					}
				} else if (TS->shaped_text_get_inferred_direction(rid) == TextServer::DIRECTION_RTL) {
					if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
						ofs.x += width - length;
					} else {
						ofs.y += width - length;
					}
				}
			} break;
			case HORIZONTAL_ALIGNMENT_RIGHT: {
				if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
					ofs.x += width - length;
				} else {
					ofs.y += width - length;
				}
			} break;
		}
	}

	float clip_l = 0.0;
	float clip_r = 0.0;
	if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(rid);
		if (clip && width > 0) {
			clip_l = Math::floor(MAX(0, p_pos.x - ofs.x));
			clip_r = Math::ceil(clip_l + width);
		}
	} else {
		ofs.x += TS->shaped_text_get_ascent(rid);
		if (clip && width > 0) {
			clip_l = Math::floor(MAX(0, p_pos.y - ofs.y));
			clip_r = Math::ceil(clip_l + width);
		}
	}
	TS->shaped_text_draw_custom(rid, ofs, clip_l, clip_r, p_draw_fn, 0);
}

void TextLine::_draw_custom(RID p_canvas, const Vector2 &p_pos, const Callable &p_callback) const {
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

				Variant args[] = { glyph, p_ofs, p_canvas };
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

void TextLine::draw(RID p_canvas, const Vector2 &p_pos, const Color &p_color) const {
	bool hex_codes = TS->shaped_text_get_preserve_control(rid) || TS->shaped_text_get_preserve_invalid(rid);
	draw_custom(
			p_pos,
			[&](const Glyph &p_gl, const Vector2 &p_ofs, int p_line_id) {
				if (p_gl.font_rid != RID()) {
					TS->font_draw_glyph(p_gl.font_rid, p_canvas, p_gl.font_size, p_ofs, p_gl.index, p_color);
				} else if (hex_codes && ((p_gl.flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL)) {
					TS->draw_hex_code_box(p_canvas, p_gl.font_size, p_ofs, p_gl.index, p_color);
				}
				return true;
			});
}

void TextLine::draw_outline(RID p_canvas, const Vector2 &p_pos, int p_outline_size, const Color &p_color) const {
	draw_custom(
			p_pos,
			[&](const Glyph &p_gl, const Vector2 &p_ofs, int p_line_id) {
				if (p_gl.font_rid != RID()) {
					TS->font_draw_glyph_outline(p_gl.font_rid, p_canvas, p_gl.font_size, p_outline_size, p_ofs, p_gl.index, p_color);
				}
				return true;
			});
}

int TextLine::hit_test(float p_coords) const {
	const_cast<TextLine *>(this)->_shape();

	return TS->shaped_text_hit_test_position(rid, p_coords);
}

TextLine::TextLine(const String &p_text, const Ref<Font> &p_font, int p_font_size, const String &p_language, TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	rid = TS->create_shaped_text(p_direction, p_orientation);
	if (p_font.is_valid()) {
		TS->shaped_text_add_string(rid, p_text, p_font->get_rids(), p_font_size, p_font->get_opentype_features(), p_language);
		for (int i = 0; i < TextServer::SPACING_MAX; i++) {
			TS->shaped_text_set_spacing(rid, TextServer::SpacingType(i), p_font->get_spacing(TextServer::SpacingType(i)));
		}
	}
}

TextLine::TextLine() {
	rid = TS->create_shaped_text();
}

TextLine::~TextLine() {
	TS->free_rid(rid);
}
