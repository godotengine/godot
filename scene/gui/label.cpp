/*************************************************************************/
/*  label.cpp                                                            */
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

#include "label.h"

#include "core/config/project_settings.h"
#include "core/core_string_names.h"
#include "core/string/print_string.h"
#include "core/string/translation.h"

#include "servers/text_server.h"

void Label::set_autowrap_mode(TextServer::AutowrapMode p_mode) {
	if (autowrap_mode != p_mode) {
		autowrap_mode = p_mode;

		BitField<TextServer::LineBreakFlag> autowrap_flags = TextServer::BREAK_MANDATORY;
		switch (autowrap_mode) {
			case TextServer::AUTOWRAP_WORD_SMART:
				autowrap_flags = TextServer::BREAK_WORD_BOUND | TextServer::BREAK_ADAPTIVE | TextServer::BREAK_MANDATORY;
				break;
			case TextServer::AUTOWRAP_WORD:
				autowrap_flags = TextServer::BREAK_WORD_BOUND | TextServer::BREAK_MANDATORY;
				break;
			case TextServer::AUTOWRAP_ARBITRARY:
				autowrap_flags = TextServer::BREAK_GRAPHEME_BOUND | TextServer::BREAK_MANDATORY;
				break;
			case TextServer::AUTOWRAP_OFF:
				break;
		}
		autowrap_flags = autowrap_flags | TextServer::BREAK_TRIM_EDGE_SPACES;
		text_para->set_break_flags(autowrap_flags);

		queue_redraw();

		if (text_para->get_clip() || text_para->get_text_overrun_behavior() != TextServer::OVERRUN_NO_TRIMMING) {
			update_minimum_size();
		}
	}
}

TextServer::AutowrapMode Label::get_autowrap_mode() const {
	return autowrap_mode;
}

void Label::set_uppercase(bool p_uppercase) {
	if (uppercase != p_uppercase) {
		uppercase = p_uppercase;
		_update_text();
		queue_redraw();
	}
}

bool Label::is_uppercase() const {
	return uppercase;
}

int Label::get_line_height(int p_line) const {
	if (p_line >= 0 && p_line < text_para->get_line_count()) {
		if (text_para->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
			return text_para->get_line_size(p_line).y;
		} else {
			return text_para->get_line_size(p_line).x;
		}
	} else if (text_para->get_line_count() > 0) {
		int h = 0;
		for (int i = 0; i < text_para->get_line_count(); i++) {
			if (text_para->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
				h = MAX(h, text_para->get_line_size(i).y);
			} else {
				h = MAX(h, text_para->get_line_size(i).x);
			}
		}
		return h;
	} else {
		Ref<Font> font = (settings.is_valid() && settings->get_font().is_valid()) ? settings->get_font() : theme_cache.font;
		int font_size = settings.is_valid() ? settings->get_font_size() : theme_cache.font_size;
		return font->get_height(font_size);
	}
}

void Label::_update_text() {
	const Ref<Font> &font = (settings.is_valid() && settings->get_font().is_valid()) ? settings->get_font() : theme_cache.font;
	int font_size = settings.is_valid() ? settings->get_font_size() : theme_cache.font_size;

	text_para->clear();
	if (font.is_valid()) {
		String txt = (uppercase) ? TS->string_to_upper(xl_text, language) : xl_text;
		if (visible_chars >= 0 && visible_chars_behavior == TextServer::VC_CHARS_BEFORE_SHAPING) {
			txt = txt.substr(0, visible_chars);
		}
		text_para->add_string(txt, font, font_size, language);
		text_para->set_bidi_override(structured_text_parser(st_parser, st_args, txt));
		text_set = true;
	} else {
		text_set = false;
	}
}

void Label::_update_fonts() {
	if (!text_set) {
		_update_text();
	} else {
		const Ref<Font> &font = (settings.is_valid() && settings->get_font().is_valid()) ? settings->get_font() : theme_cache.font;
		int font_size = settings.is_valid() ? settings->get_font_size() : theme_cache.font_size;

		if (font.is_valid()) {
			int spans = text_para->get_span_count();
			for (int i = 0; i < spans; i++) {
				text_para->update_span_font(i, font, font_size);
			}
		}
	}
}

void Label::_update_visible() {
	Ref<StyleBox> style = theme_cache.normal_style;
	int lines_visible = (text_para->get_max_lines_visible() >= 0) ? MIN(text_para->get_max_lines_visible(), text_para->get_line_count()) : text_para->get_line_count();
	int last_line = MIN(text_para->get_line_count(), lines_visible + text_para->get_lines_skipped());

	if (text_para->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
		minsize.height = 0;
	} else {
		minsize.width = 0;
	}
	for (int64_t i = text_para->get_lines_skipped(); i < last_line; i++) {
		if (text_para->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
			minsize.height += text_para->get_line_size(i).y + text_para->get_extra_line_spacing();
			if (minsize.height > (get_size().height - style->get_minimum_size().height + text_para->get_extra_line_spacing())) {
				break;
			}
		} else {
			minsize.width += text_para->get_line_size(i).x + text_para->get_extra_line_spacing();
			if (minsize.width > (get_size().width - style->get_minimum_size().width + text_para->get_extra_line_spacing())) {
				break;
			}
		}
	}
}

void Label::_update_theme_item_cache() {
	Control::_update_theme_item_cache();

	theme_cache.normal_style = get_theme_stylebox(SNAME("normal"));
	theme_cache.font = get_theme_font(SNAME("font"));

	theme_cache.font_size = get_theme_font_size(SNAME("font_size"));
	theme_cache.line_spacing = get_theme_constant(SNAME("line_spacing"));
	theme_cache.font_color = get_theme_color(SNAME("font_color"));
	theme_cache.font_shadow_color = get_theme_color(SNAME("font_shadow_color"));
	theme_cache.font_shadow_offset = Point2(get_theme_constant(SNAME("shadow_offset_x")), get_theme_constant(SNAME("shadow_offset_y")));
	theme_cache.font_outline_color = get_theme_color(SNAME("font_outline_color"));
	theme_cache.font_outline_size = get_theme_constant(SNAME("outline_size"));
	theme_cache.font_shadow_outline_size = get_theme_constant(SNAME("shadow_outline_size"));
}

PackedStringArray Label::get_configuration_warnings() const {
	PackedStringArray warnings = Control::get_configuration_warnings();

	// Ensure that the font can render all of the required glyphs.
	Ref<Font> font;
	if (settings.is_valid()) {
		font = settings->get_font();
	}
	if (font.is_null()) {
		font = theme_cache.font;
	}

	if (font.is_valid()) {
		if (text_para->has_invalid_glyphs()) {
			warnings.push_back(RTR("The current font does not support rendering one or more characters used in this Label's text."));
		}
	}

	return warnings;
}

void Label::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			String new_text = atr(text);
			if (new_text == xl_text) {
				return; // Nothing new.
			}
			xl_text = new_text;
			if (visible_ratio < 1) {
				visible_chars = get_total_character_count() * visible_ratio;
			}
			_update_text();

			queue_redraw();
			update_configuration_warnings();
		} break;

		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
				text_para->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
			}
			queue_redraw();
		} break;

		case NOTIFICATION_DRAW: {
			if (text_para->get_clip()) {
				RenderingServer::get_singleton()->canvas_item_set_clip(get_canvas_item(), true);
			}

			RID ci = get_canvas_item();

			bool has_settings = settings.is_valid();

			Size2 string_size;
			Ref<StyleBox> style = theme_cache.normal_style;
			Ref<Font> font = (has_settings && settings->get_font().is_valid()) ? settings->get_font() : theme_cache.font;
			Color font_color = has_settings ? settings->get_font_color() : theme_cache.font_color;
			Color font_shadow_color = has_settings ? settings->get_shadow_color() : theme_cache.font_shadow_color;
			Point2 shadow_ofs = has_settings ? settings->get_shadow_offset() : theme_cache.font_shadow_offset;
			Color font_outline_color = has_settings ? settings->get_outline_color() : theme_cache.font_outline_color;
			int outline_size = has_settings ? settings->get_outline_size() : theme_cache.font_outline_size;
			int shadow_outline_size = has_settings ? settings->get_shadow_size() : theme_cache.font_shadow_outline_size;
			bool rtl_layout = is_layout_rtl();
			bool hex_codes = text_para->get_preserve_invalid() || text_para->get_preserve_control();

			style->draw(ci, Rect2(Point2(0, 0), get_size()));

			bool trim_chars = (visible_chars >= 0) && (visible_chars_behavior == TextServer::VC_CHARS_AFTER_SHAPING);
			bool trim_glyphs_ltr = (visible_chars >= 0) && ((visible_chars_behavior == TextServer::VC_GLYPHS_LTR) || ((visible_chars_behavior == TextServer::VC_GLYPHS_AUTO) && !rtl_layout));
			bool trim_glyphs_rtl = (visible_chars >= 0) && ((visible_chars_behavior == TextServer::VC_GLYPHS_RTL) || ((visible_chars_behavior == TextServer::VC_GLYPHS_AUTO) && rtl_layout));

			// Get real total height.
			int total_glyphs = text_para->get_glyph_count();
			int visible_glyphs = total_glyphs * visible_ratio;

			// Draw outline and shadow. Note: Do not merge this into the single loop with the main text, to prevent overlaps.
			int processed_glyphs = 0;
			text_para->draw_custom(
					style->get_offset(),
					[&](const Glyph &p_gl, const Vector2 &p_ofs, int p_line_id) {
						bool skip = (p_gl.font_rid != RID()) && ((trim_chars && p_gl.end > visible_chars) || (trim_glyphs_ltr && (processed_glyphs >= visible_glyphs)) || (trim_glyphs_rtl && (processed_glyphs < total_glyphs - visible_glyphs)));
						if (!skip) {
							if (font_shadow_color.a > 0) {
								TS->font_draw_glyph(p_gl.font_rid, ci, p_gl.font_size, p_ofs + shadow_ofs, p_gl.index, font_shadow_color);
							}
							if (font_shadow_color.a > 0 && shadow_outline_size > 0) {
								TS->font_draw_glyph_outline(p_gl.font_rid, ci, p_gl.font_size, shadow_outline_size, p_ofs + shadow_ofs, p_gl.index, font_shadow_color);
							}
							if (font_outline_color.a != 0.0 && outline_size > 0) {
								TS->font_draw_glyph_outline(p_gl.font_rid, ci, p_gl.font_size, outline_size, p_ofs, p_gl.index, font_outline_color);
							}
						}
						processed_glyphs++;
						return true;
					});

			// Draw main text. Note: Do not merge this into the single loop with the outline, to prevent overlaps.
			processed_glyphs = 0;
			text_para->draw_custom(
					style->get_offset(),
					[&](const Glyph &p_gl, const Vector2 &p_ofs, int p_line_id) {
						bool skip = (trim_chars && p_gl.end > visible_chars) || (trim_glyphs_ltr && (processed_glyphs >= visible_glyphs)) || (trim_glyphs_rtl && (processed_glyphs < total_glyphs - visible_glyphs));
						if (!skip) {
							if (p_gl.font_rid != RID()) {
								TS->font_draw_glyph(p_gl.font_rid, ci, p_gl.font_size, p_ofs, p_gl.index, font_color);
							} else if (hex_codes && ((p_gl.flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL)) {
								TS->draw_hex_code_box(ci, p_gl.font_size, p_ofs, p_gl.index, font_color);
							}
						}
						processed_glyphs++;
						return true;
					});
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			_update_fonts();

			text_para->set_extra_line_spacing(settings.is_valid() ? settings->get_line_spacing() : theme_cache.line_spacing);
			Size2 size = get_size() - theme_cache.normal_style->get_minimum_size();
			if (text_para->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
				text_para->set_width(size.x);
				text_para->set_height(size.y);
			} else {
				text_para->set_width(size.y);
				text_para->set_height(size.x);
			}
			queue_redraw();
		} break;

		case NOTIFICATION_RESIZED: {
			Size2 size = get_size() - theme_cache.normal_style->get_minimum_size();
			if (text_para->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
				text_para->set_width(size.x);
				text_para->set_height(size.y);
			} else {
				text_para->set_width(size.y);
				text_para->set_height(size.x);
			}
		} break;
	}
}

Size2 Label::get_minimum_size() const {
	Size2 min_size = minsize;

	const Ref<Font> &font = (settings.is_valid() && settings->get_font().is_valid()) ? settings->get_font() : theme_cache.font;
	int font_size = settings.is_valid() ? settings->get_font_size() : theme_cache.font_size;
	Size2 min_style = theme_cache.normal_style->get_minimum_size();

	if (text_para->get_orientation() == TextServer::ORIENTATION_HORIZONTAL) {
		min_size.height = MAX(min_size.height, font->get_height(font_size) + font->get_spacing(TextServer::SPACING_TOP) + font->get_spacing(TextServer::SPACING_BOTTOM));

		if (autowrap_mode != TextServer::AUTOWRAP_OFF) {
			return Size2(1, (text_para->get_clip() || text_para->get_text_overrun_behavior() != TextServer::OVERRUN_NO_TRIMMING) ? 1 : min_size.height) + min_style;
		} else {
			if (text_para->get_clip() || text_para->get_text_overrun_behavior() != TextServer::OVERRUN_NO_TRIMMING) {
				min_size.width = 1;
			}
			return min_size + min_style;
		}
	} else {
		min_size.width = MAX(min_size.width, font->get_height(font_size) + font->get_spacing(TextServer::SPACING_TOP) + font->get_spacing(TextServer::SPACING_BOTTOM));

		if (autowrap_mode != TextServer::AUTOWRAP_OFF) {
			return Size2((text_para->get_clip() || text_para->get_text_overrun_behavior() != TextServer::OVERRUN_NO_TRIMMING) ? 1 : min_size.width, 1) + min_style;
		} else {
			if (text_para->get_clip() || text_para->get_text_overrun_behavior() != TextServer::OVERRUN_NO_TRIMMING) {
				min_size.height = 1;
			}
			return min_size + min_style;
		}
	}
}

int Label::get_line_count() const {
	if (!is_inside_tree()) {
		return 1;
	}

	return text_para->get_line_count();
}

int Label::get_visible_line_count() const {
	return text_para->get_visible_line_count();
}

void Label::set_horizontal_alignment(HorizontalAlignment p_alignment) {
	ERR_FAIL_INDEX((int)p_alignment, 4);
	if (text_para->get_horizontal_alignment() != p_alignment) {
		text_para->set_horizontal_alignment(p_alignment);
		queue_redraw();
	}
}

HorizontalAlignment Label::get_horizontal_alignment() const {
	return text_para->get_horizontal_alignment();
}

void Label::set_vertical_alignment(VerticalAlignment p_alignment) {
	ERR_FAIL_INDEX((int)p_alignment, 4);
	if (text_para->get_vertical_alignment() != p_alignment) {
		text_para->set_vertical_alignment(p_alignment);
		queue_redraw();
	}
}

VerticalAlignment Label::get_vertical_alignment() const {
	return text_para->get_vertical_alignment();
}

void Label::set_text(const String &p_string) {
	if (text == p_string) {
		return;
	}
	text = p_string;
	xl_text = atr(p_string);
	if (visible_ratio < 1) {
		visible_chars = get_total_character_count() * visible_ratio;
	}

	_update_text();
	queue_redraw();
	update_minimum_size();
	update_configuration_warnings();
}

void Label::_invalidate_fonts() {
	_update_fonts();
	queue_redraw();
}

void Label::set_label_settings(const Ref<LabelSettings> &p_settings) {
	if (settings != p_settings) {
		if (settings.is_valid()) {
			settings->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &Label::_invalidate_fonts));
		}
		settings = p_settings;
		if (settings.is_valid()) {
			settings->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &Label::_invalidate_fonts), CONNECT_REFERENCE_COUNTED);
		}
		_invalidate_fonts();
	}
}

Ref<LabelSettings> Label::get_label_settings() const {
	return settings;
}

void Label::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < 0 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
			text_para->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
		} else {
			text_para->set_direction((TextServer::Direction)p_text_direction);
		}
		queue_redraw();
	}
}

Control::TextDirection Label::get_text_direction() const {
	return text_direction;
}

void Label::set_orientation(TextServer::Orientation p_orientation) {
	ERR_FAIL_COND((int)p_orientation < 0 || (int)p_orientation > 3);
	if (text_para->get_orientation() != p_orientation) {
		text_para->set_orientation(p_orientation);
		queue_redraw();
	}
}

TextServer::Orientation Label::get_orientation() const {
	return text_para->get_orientation();
}

void Label::set_uniform_line_height(bool p_enabled) {
	if (text_para->get_uniform_line_height() != p_enabled) {
		text_para->set_uniform_line_height(p_enabled);
		_update_visible();
		queue_redraw();
	}
}

bool Label::get_uniform_line_height() const {
	return text_para->get_uniform_line_height();
}

void Label::set_invert_line_order(bool p_enabled) {
	if (text_para->get_invert_line_order() != p_enabled) {
		text_para->set_invert_line_order(p_enabled);
		_update_visible();
		queue_redraw();
	}
}

bool Label::get_invert_line_order() const {
	return text_para->get_invert_line_order();
}

void Label::set_structured_text_bidi_override(TextServer::StructuredTextParser p_parser) {
	if (st_parser != p_parser) {
		st_parser = p_parser;
		_update_text();
		queue_redraw();
	}
}

TextServer::StructuredTextParser Label::get_structured_text_bidi_override() const {
	return st_parser;
}

void Label::set_structured_text_bidi_override_options(Array p_args) {
	if (st_args != p_args) {
		st_args = p_args;
		_update_text();
		queue_redraw();
	}
}

Array Label::get_structured_text_bidi_override_options() const {
	return st_args;
}

void Label::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		_update_text();
		queue_redraw();
	}
}

String Label::get_language() const {
	return language;
}

void Label::set_clip_text(bool p_clip) {
	if (text_para->get_clip() != p_clip) {
		text_para->set_clip(p_clip);
		queue_redraw();
		update_minimum_size();
	}
}

bool Label::is_clipping_text() const {
	return text_para->get_clip();
}

void Label::set_text_overrun_behavior(TextServer::OverrunBehavior p_behavior) {
	if (text_para->get_text_overrun_behavior() != p_behavior) {
		text_para->set_text_overrun_behavior(p_behavior);

		queue_redraw();
		if (text_para->get_clip() || p_behavior != TextServer::OVERRUN_NO_TRIMMING) {
			update_minimum_size();
		}
	}
}

TextServer::OverrunBehavior Label::get_text_overrun_behavior() const {
	return text_para->get_text_overrun_behavior();
}

String Label::get_text() const {
	return text;
}

void Label::set_visible_characters(int p_amount) {
	if (visible_chars != p_amount) {
		visible_chars = p_amount;
		if (get_total_character_count() > 0) {
			visible_ratio = (float)p_amount / (float)get_total_character_count();
		} else {
			visible_ratio = 1.0;
		}
		if (visible_chars_behavior == TextServer::VC_CHARS_BEFORE_SHAPING) {
			_update_text();
		}
		queue_redraw();
	}
}

int Label::get_visible_characters() const {
	return visible_chars;
}

void Label::set_visible_ratio(float p_ratio) {
	if (visible_ratio != p_ratio) {
		if (p_ratio >= 1.0) {
			visible_chars = -1;
			visible_ratio = 1.0;
		} else if (p_ratio < 0.0) {
			visible_chars = 0;
			visible_ratio = 0.0;
		} else {
			visible_chars = get_total_character_count() * p_ratio;
			visible_ratio = p_ratio;
		}

		if (visible_chars_behavior == TextServer::VC_CHARS_BEFORE_SHAPING) {
			_update_text();
		}
		queue_redraw();
	}
}

float Label::get_visible_ratio() const {
	return visible_ratio;
}

void Label::set_visible_characters_behavior(TextServer::VisibleCharactersBehavior p_behavior) {
	if (visible_chars_behavior != p_behavior) {
		visible_chars_behavior = p_behavior;
		if (visible_chars_behavior == TextServer::VC_CHARS_BEFORE_SHAPING) {
			_update_text();
		}
		queue_redraw();
	}
}

TextServer::VisibleCharactersBehavior Label::get_visible_characters_behavior() const {
	return visible_chars_behavior;
}

void Label::set_lines_skipped(int p_lines) {
	ERR_FAIL_COND(p_lines < 0);

	if (text_para->get_lines_skipped() != p_lines) {
		text_para->set_lines_skipped(p_lines);
		_update_visible();
		queue_redraw();
	}
}

int Label::get_lines_skipped() const {
	return text_para->get_lines_skipped();
}

void Label::set_max_lines_visible(int p_lines) {
	if (text_para->get_max_lines_visible() != p_lines) {
		text_para->set_max_lines_visible(p_lines);
		_update_visible();
		queue_redraw();
	}
}

int Label::get_max_lines_visible() const {
	return text_para->get_max_lines_visible();
}

int Label::get_total_character_count() const {
	return xl_text.length();
}

void Label::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_horizontal_alignment", "alignment"), &Label::set_horizontal_alignment);
	ClassDB::bind_method(D_METHOD("get_horizontal_alignment"), &Label::get_horizontal_alignment);
	ClassDB::bind_method(D_METHOD("set_vertical_alignment", "alignment"), &Label::set_vertical_alignment);
	ClassDB::bind_method(D_METHOD("get_vertical_alignment"), &Label::get_vertical_alignment);
	ClassDB::bind_method(D_METHOD("set_text", "text"), &Label::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &Label::get_text);
	ClassDB::bind_method(D_METHOD("set_label_settings", "settings"), &Label::set_label_settings);
	ClassDB::bind_method(D_METHOD("get_label_settings"), &Label::get_label_settings);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &Label::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &Label::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_orientation", "orientation"), &Label::set_orientation);
	ClassDB::bind_method(D_METHOD("get_orientation"), &Label::get_orientation);
	ClassDB::bind_method(D_METHOD("set_uniform_line_height", "enabled"), &Label::set_uniform_line_height);
	ClassDB::bind_method(D_METHOD("get_uniform_line_height"), &Label::get_uniform_line_height);
	ClassDB::bind_method(D_METHOD("set_invert_line_order", "enabled"), &Label::set_invert_line_order);
	ClassDB::bind_method(D_METHOD("get_invert_line_order"), &Label::get_invert_line_order);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &Label::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &Label::get_language);
	ClassDB::bind_method(D_METHOD("set_autowrap_mode", "autowrap_mode"), &Label::set_autowrap_mode);
	ClassDB::bind_method(D_METHOD("get_autowrap_mode"), &Label::get_autowrap_mode);
	ClassDB::bind_method(D_METHOD("set_clip_text", "enable"), &Label::set_clip_text);
	ClassDB::bind_method(D_METHOD("is_clipping_text"), &Label::is_clipping_text);
	ClassDB::bind_method(D_METHOD("set_text_overrun_behavior", "overrun_behavior"), &Label::set_text_overrun_behavior);
	ClassDB::bind_method(D_METHOD("get_text_overrun_behavior"), &Label::get_text_overrun_behavior);
	ClassDB::bind_method(D_METHOD("set_uppercase", "enable"), &Label::set_uppercase);
	ClassDB::bind_method(D_METHOD("is_uppercase"), &Label::is_uppercase);
	ClassDB::bind_method(D_METHOD("get_line_height", "line"), &Label::get_line_height, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("get_line_count"), &Label::get_line_count);
	ClassDB::bind_method(D_METHOD("get_visible_line_count"), &Label::get_visible_line_count);
	ClassDB::bind_method(D_METHOD("get_total_character_count"), &Label::get_total_character_count);
	ClassDB::bind_method(D_METHOD("set_visible_characters", "amount"), &Label::set_visible_characters);
	ClassDB::bind_method(D_METHOD("get_visible_characters"), &Label::get_visible_characters);
	ClassDB::bind_method(D_METHOD("get_visible_characters_behavior"), &Label::get_visible_characters_behavior);
	ClassDB::bind_method(D_METHOD("set_visible_characters_behavior", "behavior"), &Label::set_visible_characters_behavior);
	ClassDB::bind_method(D_METHOD("set_visible_ratio", "ratio"), &Label::set_visible_ratio);
	ClassDB::bind_method(D_METHOD("get_visible_ratio"), &Label::get_visible_ratio);
	ClassDB::bind_method(D_METHOD("set_lines_skipped", "lines_skipped"), &Label::set_lines_skipped);
	ClassDB::bind_method(D_METHOD("get_lines_skipped"), &Label::get_lines_skipped);
	ClassDB::bind_method(D_METHOD("set_max_lines_visible", "lines_visible"), &Label::set_max_lines_visible);
	ClassDB::bind_method(D_METHOD("get_max_lines_visible"), &Label::get_max_lines_visible);
	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override", "parser"), &Label::set_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override"), &Label::get_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override_options", "args"), &Label::set_structured_text_bidi_override_options);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override_options"), &Label::get_structured_text_bidi_override_options);

	ClassDB::bind_method(D_METHOD("_invalidate_fonts"), &Label::_invalidate_fonts);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT, "", PROPERTY_USAGE_DEFAULT_INTL), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "label_settings", PROPERTY_HINT_RESOURCE_TYPE, "LabelSettings"), "set_label_settings", "get_label_settings");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "horizontal_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_horizontal_alignment", "get_horizontal_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "vertical_alignment", PROPERTY_HINT_ENUM, "Top,Center,Bottom,Fill"), "set_vertical_alignment", "get_vertical_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "autowrap_mode", PROPERTY_HINT_ENUM, "Off,Arbitrary,Word,Word (Smart)"), "set_autowrap_mode", "get_autowrap_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_text"), "set_clip_text", "is_clipping_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_overrun_behavior", PROPERTY_HINT_ENUM, "Trim Nothing,Trim Characters,Trim Words,Ellipsis,Word Ellipsis"), "set_text_overrun_behavior", "get_text_overrun_behavior");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "uppercase"), "set_uppercase", "is_uppercase");

	ADD_GROUP("Displayed Text", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "orientation", PROPERTY_HINT_ENUM, "Horizontal,Vertical Upright,Vertical Mixed,Vertical Sideways"), "set_orientation", "get_orientation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "uniform_line_height"), "set_uniform_line_height", "get_uniform_line_height");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert_line_order"), "set_invert_line_order", "get_invert_line_order");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "lines_skipped", PROPERTY_HINT_RANGE, "0,999,1"), "set_lines_skipped", "get_lines_skipped");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_lines_visible", PROPERTY_HINT_RANGE, "-1,999,1"), "set_max_lines_visible", "get_max_lines_visible");
	// Note: "visible_characters" and "visible_ratio" should be set after "text" to be correctly applied.
	ADD_PROPERTY(PropertyInfo(Variant::INT, "visible_characters", PROPERTY_HINT_RANGE, "-1,128000,1"), "set_visible_characters", "get_visible_characters");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "visible_characters_behavior", PROPERTY_HINT_ENUM, "Characters Before Shaping,Characters After Shaping,Glyphs (Layout Direction),Glyphs (Left-to-Right),Glyphs (Right-to-Left)"), "set_visible_characters_behavior", "get_visible_characters_behavior");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "visible_ratio", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_visible_ratio", "get_visible_ratio");

	ADD_GROUP("BiDi", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language", PROPERTY_HINT_LOCALE_ID, ""), "set_language", "get_language");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "structured_text_bidi_override", PROPERTY_HINT_ENUM, "Default,URI,File,Email,List,None,Custom"), "set_structured_text_bidi_override", "get_structured_text_bidi_override");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "structured_text_bidi_override_options"), "set_structured_text_bidi_override_options", "get_structured_text_bidi_override_options");
}

Label::Label(const String &p_text) {
	text_para.instantiate();
	text_para->set_break_flags(TextServer::BREAK_MANDATORY | TextServer::BREAK_TRIM_EDGE_SPACES);
	text_para->set_clip(false);

	set_mouse_filter(MOUSE_FILTER_IGNORE);
	set_text(p_text);
	set_v_size_flags(SIZE_SHRINK_CENTER);
}

Label::~Label() {
}
