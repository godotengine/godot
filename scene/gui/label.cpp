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
#include "core/string/print_string.h"
#include "core/string/translation.h"

#include "servers/text_server.h"

void Label::set_autowrap_mode(Label::AutowrapMode p_mode) {
	if (autowrap_mode != p_mode) {
		autowrap_mode = p_mode;
		lines_dirty = true;
	}
	update();

	if (clip || overrun_behavior != OVERRUN_NO_TRIMMING) {
		update_minimum_size();
	}
}

Label::AutowrapMode Label::get_autowrap_mode() const {
	return autowrap_mode;
}

void Label::set_uppercase(bool p_uppercase) {
	uppercase = p_uppercase;
	dirty = true;

	update();
}

bool Label::is_uppercase() const {
	return uppercase;
}

int Label::get_line_height(int p_line) const {
	Ref<Font> font = get_theme_font(SNAME("font"));
	if (p_line >= 0 && p_line < lines_rid.size()) {
		return TS->shaped_text_get_size(lines_rid[p_line]).y + font->get_spacing(TextServer::SPACING_TOP) + font->get_spacing(TextServer::SPACING_BOTTOM);
	} else if (lines_rid.size() > 0) {
		int h = 0;
		for (int i = 0; i < lines_rid.size(); i++) {
			h = MAX(h, TS->shaped_text_get_size(lines_rid[i]).y) + font->get_spacing(TextServer::SPACING_TOP) + font->get_spacing(TextServer::SPACING_BOTTOM);
		}
		return h;
	} else {
		return font->get_height(get_theme_font_size(SNAME("font_size")));
	}
}

void Label::_shape() {
	Ref<StyleBox> style = get_theme_stylebox(SNAME("normal"), SNAME("Label"));
	int width = (get_size().width - style->get_minimum_size().width);

	if (dirty) {
		String lang = (!language.is_empty()) ? language : TranslationServer::get_singleton()->get_tool_locale();
		TS->shaped_text_clear(text_rid);
		if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
			TS->shaped_text_set_direction(text_rid, is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
		} else {
			TS->shaped_text_set_direction(text_rid, (TextServer::Direction)text_direction);
		}
		const Ref<Font> &font = get_theme_font(SNAME("font"));
		int font_size = get_theme_font_size(SNAME("font_size"));
		ERR_FAIL_COND(font.is_null());
		String text = (uppercase) ? TS->string_to_upper(xl_text, lang) : xl_text;
		if (visible_chars >= 0 && visible_chars_behavior == VC_CHARS_BEFORE_SHAPING) {
			text = text.substr(0, visible_chars);
		}
		TS->shaped_text_add_string(text_rid, text, font->get_rids(), font_size, opentype_features, lang);
		TS->shaped_text_set_bidi_override(text_rid, structured_text_parser(st_parser, st_args, text));
		dirty = false;
		lines_dirty = true;
	}

	if (lines_dirty) {
		for (int i = 0; i < lines_rid.size(); i++) {
			TS->free(lines_rid[i]);
		}
		lines_rid.clear();

		uint16_t autowrap_flags = TextServer::BREAK_MANDATORY;
		switch (autowrap_mode) {
			case AUTOWRAP_WORD_SMART:
				autowrap_flags = TextServer::BREAK_WORD_BOUND_ADAPTIVE | TextServer::BREAK_MANDATORY;
				break;
			case AUTOWRAP_WORD:
				autowrap_flags = TextServer::BREAK_WORD_BOUND | TextServer::BREAK_MANDATORY;
				break;
			case AUTOWRAP_ARBITRARY:
				autowrap_flags = TextServer::BREAK_GRAPHEME_BOUND | TextServer::BREAK_MANDATORY;
				break;
			case AUTOWRAP_OFF:
				break;
		}
		PackedInt32Array line_breaks = TS->shaped_text_get_line_breaks(text_rid, width, 0, autowrap_flags);

		for (int i = 0; i < line_breaks.size(); i = i + 2) {
			RID line = TS->shaped_text_substr(text_rid, line_breaks[i], line_breaks[i + 1] - line_breaks[i]);
			lines_rid.push_back(line);
		}
	}

	if (xl_text.length() == 0) {
		minsize = Size2(1, get_line_height());
		return;
	}

	if (autowrap_mode == AUTOWRAP_OFF) {
		minsize.width = 0.0f;
		for (int i = 0; i < lines_rid.size(); i++) {
			if (minsize.width < TS->shaped_text_get_size(lines_rid[i]).x) {
				minsize.width = TS->shaped_text_get_size(lines_rid[i]).x;
			}
		}
	}

	if (lines_dirty) {
		uint16_t overrun_flags = TextServer::OVERRUN_NO_TRIMMING;
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

		// Fill after min_size calculation.

		if (autowrap_mode != AUTOWRAP_OFF) {
			int visible_lines = get_visible_line_count();
			bool lines_hidden = visible_lines > 0 && visible_lines < lines_rid.size();
			if (lines_hidden) {
				overrun_flags |= TextServer::OVERRUN_ENFORCE_ELLIPSIS;
			}
			if (horizontal_alignment == HORIZONTAL_ALIGNMENT_FILL) {
				for (int i = 0; i < lines_rid.size(); i++) {
					if (i < visible_lines - 1 || lines_rid.size() == 1) {
						TS->shaped_text_fit_to_width(lines_rid[i], width);
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
				if (horizontal_alignment == HORIZONTAL_ALIGNMENT_FILL) {
					TS->shaped_text_fit_to_width(lines_rid[i], width);
					overrun_flags |= TextServer::OVERRUN_JUSTIFICATION_AWARE;
					TS->shaped_text_overrun_trim_to_width(lines_rid[i], width, overrun_flags);
					TS->shaped_text_fit_to_width(lines_rid[i], width, TextServer::JUSTIFICATION_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_CONSTRAIN_ELLIPSIS);
				} else {
					TS->shaped_text_overrun_trim_to_width(lines_rid[i], width, overrun_flags);
				}
			}
		}
		lines_dirty = false;
	}

	_update_visible();

	if (autowrap_mode == AUTOWRAP_OFF || !clip || overrun_behavior == OVERRUN_NO_TRIMMING) {
		update_minimum_size();
	}
}

void Label::_update_visible() {
	int line_spacing = get_theme_constant(SNAME("line_spacing"), SNAME("Label"));
	Ref<StyleBox> style = get_theme_stylebox(SNAME("normal"), SNAME("Label"));
	Ref<Font> font = get_theme_font(SNAME("font"));
	int lines_visible = lines_rid.size();

	if (max_lines_visible >= 0 && lines_visible > max_lines_visible) {
		lines_visible = max_lines_visible;
	}

	minsize.height = 0;
	int last_line = MIN(lines_rid.size(), lines_visible + lines_skipped);
	for (int64_t i = lines_skipped; i < last_line; i++) {
		minsize.height += TS->shaped_text_get_size(lines_rid[i]).y + font->get_spacing(TextServer::SPACING_TOP) + font->get_spacing(TextServer::SPACING_BOTTOM) + line_spacing;
		if (minsize.height > (get_size().height - style->get_minimum_size().height + line_spacing)) {
			break;
		}
	}
}

inline void draw_glyph(const Glyph &p_gl, const RID &p_canvas, const Color &p_font_color, const Vector2 &p_ofs) {
	if (p_gl.font_rid != RID()) {
		TS->font_draw_glyph(p_gl.font_rid, p_canvas, p_gl.font_size, p_ofs + Vector2(p_gl.x_off, p_gl.y_off), p_gl.index, p_font_color);
	} else {
		TS->draw_hex_code_box(p_canvas, p_gl.font_size, p_ofs + Vector2(p_gl.x_off, p_gl.y_off), p_gl.index, p_font_color);
	}
}

inline void draw_glyph_outline(const Glyph &p_gl, const RID &p_canvas, const Color &p_font_color, const Color &p_font_shadow_color, const Color &p_font_outline_color, const int &p_shadow_outline_size, const int &p_outline_size, const Vector2 &p_ofs, const Vector2 &shadow_ofs) {
	if (p_gl.font_rid != RID()) {
		if (p_font_shadow_color.a > 0) {
			TS->font_draw_glyph(p_gl.font_rid, p_canvas, p_gl.font_size, p_ofs + Vector2(p_gl.x_off, p_gl.y_off) + shadow_ofs, p_gl.index, p_font_shadow_color);
		}
		if (p_font_shadow_color.a > 0 && p_shadow_outline_size > 0) {
			TS->font_draw_glyph_outline(p_gl.font_rid, p_canvas, p_gl.font_size, p_shadow_outline_size, p_ofs + Vector2(p_gl.x_off, p_gl.y_off) + shadow_ofs, p_gl.index, p_font_shadow_color);
		}
		if (p_font_outline_color.a != 0.0 && p_outline_size > 0) {
			TS->font_draw_glyph_outline(p_gl.font_rid, p_canvas, p_gl.font_size, p_outline_size, p_ofs + Vector2(p_gl.x_off, p_gl.y_off), p_gl.index, p_font_outline_color);
		}
	}
}

void Label::_notification(int p_what) {
	if (p_what == NOTIFICATION_TRANSLATION_CHANGED) {
		String new_text = atr(text);
		if (new_text == xl_text) {
			return; // Nothing new.
		}
		xl_text = new_text;
		if (percent_visible < 1) {
			visible_chars = get_total_character_count() * percent_visible;
		}
		dirty = true;

		update();
	}

	if (p_what == NOTIFICATION_LAYOUT_DIRECTION_CHANGED) {
		update();
	}

	if (p_what == NOTIFICATION_DRAW) {
		if (clip) {
			RenderingServer::get_singleton()->canvas_item_set_clip(get_canvas_item(), true);
		}

		if (dirty || lines_dirty) {
			_shape();
		}

		RID ci = get_canvas_item();

		Size2 string_size;
		Size2 size = get_size();
		Ref<StyleBox> style = get_theme_stylebox(SNAME("normal"));
		Ref<Font> font = get_theme_font(SNAME("font"));
		Color font_color = get_theme_color(SNAME("font_color"));
		Color font_shadow_color = get_theme_color(SNAME("font_shadow_color"));
		Point2 shadow_ofs(get_theme_constant(SNAME("shadow_offset_x")), get_theme_constant(SNAME("shadow_offset_y")));
		int line_spacing = get_theme_constant(SNAME("line_spacing"));
		Color font_outline_color = get_theme_color(SNAME("font_outline_color"));
		int outline_size = get_theme_constant(SNAME("outline_size"));
		int shadow_outline_size = get_theme_constant(SNAME("shadow_outline_size"));
		bool rtl = (TS->shaped_text_get_inferred_direction(text_rid) == TextServer::DIRECTION_RTL);
		bool rtl_layout = is_layout_rtl();

		style->draw(ci, Rect2(Point2(0, 0), get_size()));

		float total_h = 0.0;
		int lines_visible = 0;

		// Get number of lines to fit to the height.
		for (int64_t i = lines_skipped; i < lines_rid.size(); i++) {
			total_h += TS->shaped_text_get_size(lines_rid[i]).y + font->get_spacing(TextServer::SPACING_TOP) + font->get_spacing(TextServer::SPACING_BOTTOM) + line_spacing;
			if (total_h > (get_size().height - style->get_minimum_size().height + line_spacing)) {
				break;
			}
			lines_visible++;
		}

		if (max_lines_visible >= 0 && lines_visible > max_lines_visible) {
			lines_visible = max_lines_visible;
		}

		int last_line = MIN(lines_rid.size(), lines_visible + lines_skipped);
		bool trim_chars = (visible_chars >= 0) && (visible_chars_behavior == VC_CHARS_AFTER_SHAPING);
		bool trim_glyphs_ltr = (visible_chars >= 0) && ((visible_chars_behavior == VC_GLYPHS_LTR) || ((visible_chars_behavior == VC_GLYPHS_AUTO) && !rtl_layout));
		bool trim_glyphs_rtl = (visible_chars >= 0) && ((visible_chars_behavior == VC_GLYPHS_RTL) || ((visible_chars_behavior == VC_GLYPHS_AUTO) && rtl_layout));

		// Get real total height.
		int total_glyphs = 0;
		total_h = 0;
		for (int64_t i = lines_skipped; i < last_line; i++) {
			total_h += TS->shaped_text_get_size(lines_rid[i]).y + font->get_spacing(TextServer::SPACING_TOP) + font->get_spacing(TextServer::SPACING_BOTTOM) + line_spacing;
			total_glyphs += TS->shaped_text_get_glyph_count(lines_rid[i]) + TS->shaped_text_get_ellipsis_glyph_count(lines_rid[i]);
		}
		int visible_glyphs = total_glyphs * percent_visible;
		int processed_glyphs = 0;
		total_h += style->get_margin(SIDE_TOP) + style->get_margin(SIDE_BOTTOM);

		int vbegin = 0, vsep = 0;
		if (lines_visible > 0) {
			switch (vertical_alignment) {
				case VERTICAL_ALIGNMENT_TOP: {
					// Nothing.
				} break;
				case VERTICAL_ALIGNMENT_CENTER: {
					vbegin = (size.y - (total_h - line_spacing)) / 2;
					vsep = 0;

				} break;
				case VERTICAL_ALIGNMENT_BOTTOM: {
					vbegin = size.y - (total_h - line_spacing);
					vsep = 0;

				} break;
				case VERTICAL_ALIGNMENT_FILL: {
					vbegin = 0;
					if (lines_visible > 1) {
						vsep = (size.y - (total_h - line_spacing)) / (lines_visible - 1);
					} else {
						vsep = 0;
					}

				} break;
			}
		}

		Vector2 ofs;
		ofs.y = style->get_offset().y + vbegin;
		for (int i = lines_skipped; i < last_line; i++) {
			Size2 line_size = TS->shaped_text_get_size(lines_rid[i]);
			ofs.x = 0;
			ofs.y += TS->shaped_text_get_ascent(lines_rid[i]) + font->get_spacing(TextServer::SPACING_TOP);
			switch (horizontal_alignment) {
				case HORIZONTAL_ALIGNMENT_FILL:
					if (rtl && autowrap_mode != AUTOWRAP_OFF) {
						ofs.x = int(size.width - style->get_margin(SIDE_RIGHT) - line_size.width);
					} else {
						ofs.x = style->get_offset().x;
					}
					break;
				case HORIZONTAL_ALIGNMENT_LEFT: {
					if (rtl_layout) {
						ofs.x = int(size.width - style->get_margin(SIDE_RIGHT) - line_size.width);
					} else {
						ofs.x = style->get_offset().x;
					}
				} break;
				case HORIZONTAL_ALIGNMENT_CENTER: {
					ofs.x = int(size.width - line_size.width) / 2;
				} break;
				case HORIZONTAL_ALIGNMENT_RIGHT: {
					if (rtl_layout) {
						ofs.x = style->get_offset().x;
					} else {
						ofs.x = int(size.width - style->get_margin(SIDE_RIGHT) - line_size.width);
					}
				} break;
			}

			const Glyph *glyphs = TS->shaped_text_get_glyphs(lines_rid[i]);
			int gl_size = TS->shaped_text_get_glyph_count(lines_rid[i]);

			int ellipsis_pos = TS->shaped_text_get_ellipsis_pos(lines_rid[i]);
			int trim_pos = TS->shaped_text_get_trim_pos(lines_rid[i]);

			const Glyph *ellipsis_glyphs = TS->shaped_text_get_ellipsis_glyphs(lines_rid[i]);
			int ellipsis_gl_size = TS->shaped_text_get_ellipsis_glyph_count(lines_rid[i]);

			// Draw outline. Note: Do not merge this into the single loop with the main text, to prevent overlaps.
			int processed_glyphs_ol = processed_glyphs;
			if ((outline_size > 0 && font_outline_color.a != 0) || (font_shadow_color.a != 0)) {
				Vector2 offset = ofs;
				// Draw RTL ellipsis string when necessary.
				if (rtl && ellipsis_pos >= 0) {
					for (int gl_idx = ellipsis_gl_size - 1; gl_idx >= 0; gl_idx--) {
						for (int j = 0; j < ellipsis_glyphs[gl_idx].repeat; j++) {
							bool skip = (trim_chars && ellipsis_glyphs[gl_idx].end > visible_chars) || (trim_glyphs_ltr && (processed_glyphs_ol >= visible_glyphs)) || (trim_glyphs_rtl && (processed_glyphs_ol < total_glyphs - visible_glyphs));
							//Draw glyph outlines and shadow.
							if (!skip) {
								draw_glyph_outline(ellipsis_glyphs[gl_idx], ci, font_color, font_shadow_color, font_outline_color, shadow_outline_size, outline_size, offset, shadow_ofs);
							}
							processed_glyphs_ol++;
							offset.x += ellipsis_glyphs[gl_idx].advance;
						}
					}
				}

				// Draw main text.
				for (int j = 0; j < gl_size; j++) {
					// Trim when necessary.
					if (trim_pos >= 0) {
						if (rtl) {
							if (j < trim_pos) {
								continue;
							}
						} else {
							if (j >= trim_pos) {
								break;
							}
						}
					}
					for (int k = 0; k < glyphs[j].repeat; k++) {
						bool skip = (trim_chars && glyphs[j].end > visible_chars) || (trim_glyphs_ltr && (processed_glyphs_ol >= visible_glyphs)) || (trim_glyphs_rtl && (processed_glyphs_ol < total_glyphs - visible_glyphs));

						// Draw glyph outlines and shadow.
						if (!skip) {
							draw_glyph_outline(glyphs[j], ci, font_color, font_shadow_color, font_outline_color, shadow_outline_size, outline_size, offset, shadow_ofs);
						}
						processed_glyphs_ol++;
						offset.x += glyphs[j].advance;
					}
				}
				// Draw LTR ellipsis string when necessary.
				if (!rtl && ellipsis_pos >= 0) {
					for (int gl_idx = 0; gl_idx < ellipsis_gl_size; gl_idx++) {
						for (int j = 0; j < ellipsis_glyphs[gl_idx].repeat; j++) {
							bool skip = (trim_chars && ellipsis_glyphs[gl_idx].end > visible_chars) || (trim_glyphs_ltr && (processed_glyphs_ol >= visible_glyphs)) || (trim_glyphs_rtl && (processed_glyphs_ol < total_glyphs - visible_glyphs));
							//Draw glyph outlines and shadow.
							if (!skip) {
								draw_glyph_outline(ellipsis_glyphs[gl_idx], ci, font_color, font_shadow_color, font_outline_color, shadow_outline_size, outline_size, offset, shadow_ofs);
							}
							processed_glyphs_ol++;
							offset.x += ellipsis_glyphs[gl_idx].advance;
						}
					}
				}
			}

			// Draw main text. Note: Do not merge this into the single loop with the outline, to prevent overlaps.

			// Draw RTL ellipsis string when necessary.
			if (rtl && ellipsis_pos >= 0) {
				for (int gl_idx = ellipsis_gl_size - 1; gl_idx >= 0; gl_idx--) {
					for (int j = 0; j < ellipsis_glyphs[gl_idx].repeat; j++) {
						bool skip = (trim_chars && ellipsis_glyphs[gl_idx].end > visible_chars) || (trim_glyphs_ltr && (processed_glyphs >= visible_glyphs)) || (trim_glyphs_rtl && (processed_glyphs < total_glyphs - visible_glyphs));
						//Draw glyph outlines and shadow.
						if (!skip) {
							draw_glyph(ellipsis_glyphs[gl_idx], ci, font_color, ofs);
						}
						processed_glyphs++;
						ofs.x += ellipsis_glyphs[gl_idx].advance;
					}
				}
			}

			// Draw main text.
			for (int j = 0; j < gl_size; j++) {
				// Trim when necessary.
				if (trim_pos >= 0) {
					if (rtl) {
						if (j < trim_pos) {
							continue;
						}
					} else {
						if (j >= trim_pos) {
							break;
						}
					}
				}
				for (int k = 0; k < glyphs[j].repeat; k++) {
					bool skip = (trim_chars && glyphs[j].end > visible_chars) || (trim_glyphs_ltr && (processed_glyphs >= visible_glyphs)) || (trim_glyphs_rtl && (processed_glyphs < total_glyphs - visible_glyphs));

					// Draw glyph outlines and shadow.
					if (!skip) {
						draw_glyph(glyphs[j], ci, font_color, ofs);
					}
					processed_glyphs++;
					ofs.x += glyphs[j].advance;
				}
			}
			// Draw LTR ellipsis string when necessary.
			if (!rtl && ellipsis_pos >= 0) {
				for (int gl_idx = 0; gl_idx < ellipsis_gl_size; gl_idx++) {
					for (int j = 0; j < ellipsis_glyphs[gl_idx].repeat; j++) {
						bool skip = (trim_chars && ellipsis_glyphs[gl_idx].end > visible_chars) || (trim_glyphs_ltr && (processed_glyphs >= visible_glyphs)) || (trim_glyphs_rtl && (processed_glyphs < total_glyphs - visible_glyphs));
						//Draw glyph outlines and shadow.
						if (!skip) {
							draw_glyph(ellipsis_glyphs[gl_idx], ci, font_color, ofs);
						}
						processed_glyphs++;
						ofs.x += ellipsis_glyphs[gl_idx].advance;
					}
				}
			}
			ofs.y += TS->shaped_text_get_descent(lines_rid[i]) + vsep + line_spacing + font->get_spacing(TextServer::SPACING_BOTTOM);
		}
	}

	if (p_what == NOTIFICATION_THEME_CHANGED) {
		dirty = true;
		update();
	}
	if (p_what == NOTIFICATION_RESIZED) {
		lines_dirty = true;
	}
}

Size2 Label::get_minimum_size() const {
	// don't want to mutable everything
	if (dirty || lines_dirty) {
		const_cast<Label *>(this)->_shape();
	}

	Size2 min_size = minsize;

	Ref<Font> font = get_theme_font(SNAME("font"));
	min_size.height = MAX(min_size.height, font->get_height(get_theme_font_size(SNAME("font_size"))) + font->get_spacing(TextServer::SPACING_TOP) + font->get_spacing(TextServer::SPACING_BOTTOM));

	Size2 min_style = get_theme_stylebox(SNAME("normal"))->get_minimum_size();
	if (autowrap_mode != AUTOWRAP_OFF) {
		return Size2(1, (clip || overrun_behavior != OVERRUN_NO_TRIMMING) ? 1 : min_size.height) + min_style;
	} else {
		if (clip || overrun_behavior != OVERRUN_NO_TRIMMING) {
			min_size.width = 1;
		}
		return min_size + min_style;
	}
}

int Label::get_line_count() const {
	if (!is_inside_tree()) {
		return 1;
	}
	if (dirty || lines_dirty) {
		const_cast<Label *>(this)->_shape();
	}

	return lines_rid.size();
}

int Label::get_visible_line_count() const {
	Ref<Font> font = get_theme_font(SNAME("font"));
	Ref<StyleBox> style = get_theme_stylebox(SNAME("normal"));
	int line_spacing = get_theme_constant(SNAME("line_spacing"));
	int lines_visible = 0;
	float total_h = 0.0;
	for (int64_t i = lines_skipped; i < lines_rid.size(); i++) {
		total_h += TS->shaped_text_get_size(lines_rid[i]).y + font->get_spacing(TextServer::SPACING_TOP) + font->get_spacing(TextServer::SPACING_BOTTOM) + line_spacing;
		if (total_h > (get_size().height - style->get_minimum_size().height + line_spacing)) {
			break;
		}
		lines_visible++;
	}

	if (lines_visible > lines_rid.size()) {
		lines_visible = lines_rid.size();
	}

	if (max_lines_visible >= 0 && lines_visible > max_lines_visible) {
		lines_visible = max_lines_visible;
	}

	return lines_visible;
}

void Label::set_horizontal_alignment(HorizontalAlignment p_alignment) {
	ERR_FAIL_INDEX((int)p_alignment, 4);
	if (horizontal_alignment != p_alignment) {
		if (horizontal_alignment == HORIZONTAL_ALIGNMENT_FILL || p_alignment == HORIZONTAL_ALIGNMENT_FILL) {
			lines_dirty = true; // Reshape lines.
		}
		horizontal_alignment = p_alignment;
	}
	update();
}

HorizontalAlignment Label::get_horizontal_alignment() const {
	return horizontal_alignment;
}

void Label::set_vertical_alignment(VerticalAlignment p_alignment) {
	ERR_FAIL_INDEX((int)p_alignment, 4);
	vertical_alignment = p_alignment;
	update();
}

VerticalAlignment Label::get_vertical_alignment() const {
	return vertical_alignment;
}

void Label::set_text(const String &p_string) {
	if (text == p_string) {
		return;
	}
	text = p_string;
	xl_text = atr(p_string);
	dirty = true;
	if (percent_visible < 1) {
		visible_chars = get_total_character_count() * percent_visible;
	}
	update();
	update_minimum_size();
}

void Label::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		dirty = true;
		update();
	}
}

void Label::set_structured_text_bidi_override(Control::StructuredTextParser p_parser) {
	if (st_parser != p_parser) {
		st_parser = p_parser;
		dirty = true;
		update();
	}
}

Control::StructuredTextParser Label::get_structured_text_bidi_override() const {
	return st_parser;
}

void Label::set_structured_text_bidi_override_options(Array p_args) {
	st_args = p_args;
	dirty = true;
	update();
}

Array Label::get_structured_text_bidi_override_options() const {
	return st_args;
}

Control::TextDirection Label::get_text_direction() const {
	return text_direction;
}

void Label::clear_opentype_features() {
	opentype_features.clear();
	dirty = true;
	update();
}

void Label::set_opentype_feature(const String &p_name, int p_value) {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag) || (int)opentype_features[tag] != p_value) {
		opentype_features[tag] = p_value;
		dirty = true;
		update();
	}
}

int Label::get_opentype_feature(const String &p_name) const {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag)) {
		return -1;
	}
	return opentype_features[tag];
}

void Label::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		dirty = true;
		update();
	}
}

String Label::get_language() const {
	return language;
}

void Label::set_clip_text(bool p_clip) {
	clip = p_clip;
	update();
	update_minimum_size();
}

bool Label::is_clipping_text() const {
	return clip;
}

void Label::set_text_overrun_behavior(Label::OverrunBehavior p_behavior) {
	if (overrun_behavior != p_behavior) {
		overrun_behavior = p_behavior;
		lines_dirty = true;
	}
	update();
	if (clip || overrun_behavior != OVERRUN_NO_TRIMMING) {
		update_minimum_size();
	}
}

Label::OverrunBehavior Label::get_text_overrun_behavior() const {
	return overrun_behavior;
}

String Label::get_text() const {
	return text;
}

void Label::set_visible_characters(int p_amount) {
	if (visible_chars != p_amount) {
		visible_chars = p_amount;
		if (get_total_character_count() > 0) {
			percent_visible = (float)p_amount / (float)get_total_character_count();
		} else {
			percent_visible = 1.0;
		}
		if (visible_chars_behavior == VC_CHARS_BEFORE_SHAPING) {
			dirty = true;
		}
		update();
	}
}

int Label::get_visible_characters() const {
	return visible_chars;
}

void Label::set_percent_visible(float p_percent) {
	if (percent_visible != p_percent) {
		if (p_percent < 0 || p_percent >= 1) {
			visible_chars = -1;
			percent_visible = 1;
		} else {
			visible_chars = get_total_character_count() * p_percent;
			percent_visible = p_percent;
		}
		if (visible_chars_behavior == VC_CHARS_BEFORE_SHAPING) {
			dirty = true;
		}
		update();
	}
}

float Label::get_percent_visible() const {
	return percent_visible;
}

Label::VisibleCharactersBehavior Label::get_visible_characters_behavior() const {
	return visible_chars_behavior;
}

void Label::set_visible_characters_behavior(Label::VisibleCharactersBehavior p_behavior) {
	if (visible_chars_behavior != p_behavior) {
		visible_chars_behavior = p_behavior;
		dirty = true;
		update();
	}
}

void Label::set_lines_skipped(int p_lines) {
	ERR_FAIL_COND(p_lines < 0);
	lines_skipped = p_lines;
	_update_visible();
	update();
}

int Label::get_lines_skipped() const {
	return lines_skipped;
}

void Label::set_max_lines_visible(int p_lines) {
	max_lines_visible = p_lines;
	_update_visible();
	update();
}

int Label::get_max_lines_visible() const {
	return max_lines_visible;
}

int Label::get_total_character_count() const {
	if (dirty || lines_dirty) {
		const_cast<Label *>(this)->_shape();
	}

	return xl_text.length();
}

bool Label::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		int value = p_value;
		if (value == -1) {
			if (opentype_features.has(tag)) {
				opentype_features.erase(tag);
				dirty = true;
				update();
			}
		} else {
			if (!opentype_features.has(tag) || (int)opentype_features[tag] != value) {
				opentype_features[tag] = value;
				dirty = true;
				update();
			}
		}
		notify_property_list_changed();
		return true;
	}

	return false;
}

bool Label::_get(const StringName &p_name, Variant &r_ret) const {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		if (opentype_features.has(tag)) {
			r_ret = opentype_features[tag];
			return true;
		} else {
			r_ret = -1;
			return true;
		}
	}
	return false;
}

void Label::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const Variant *ftr = opentype_features.next(nullptr); ftr != nullptr; ftr = opentype_features.next(ftr)) {
		String name = TS->tag_to_name(*ftr);
		p_list->push_back(PropertyInfo(Variant::INT, "opentype_features/" + name));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "opentype_features/_new", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
}

void Label::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_horizontal_alignment", "alignment"), &Label::set_horizontal_alignment);
	ClassDB::bind_method(D_METHOD("get_horizontal_alignment"), &Label::get_horizontal_alignment);
	ClassDB::bind_method(D_METHOD("set_vertical_alignment", "alignment"), &Label::set_vertical_alignment);
	ClassDB::bind_method(D_METHOD("get_vertical_alignment"), &Label::get_vertical_alignment);
	ClassDB::bind_method(D_METHOD("set_text", "text"), &Label::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &Label::get_text);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &Label::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &Label::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_opentype_feature", "tag", "value"), &Label::set_opentype_feature);
	ClassDB::bind_method(D_METHOD("get_opentype_feature", "tag"), &Label::get_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_opentype_features"), &Label::clear_opentype_features);
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
	ClassDB::bind_method(D_METHOD("set_percent_visible", "percent_visible"), &Label::set_percent_visible);
	ClassDB::bind_method(D_METHOD("get_percent_visible"), &Label::get_percent_visible);
	ClassDB::bind_method(D_METHOD("set_lines_skipped", "lines_skipped"), &Label::set_lines_skipped);
	ClassDB::bind_method(D_METHOD("get_lines_skipped"), &Label::get_lines_skipped);
	ClassDB::bind_method(D_METHOD("set_max_lines_visible", "lines_visible"), &Label::set_max_lines_visible);
	ClassDB::bind_method(D_METHOD("get_max_lines_visible"), &Label::get_max_lines_visible);
	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override", "parser"), &Label::set_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override"), &Label::get_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override_options", "args"), &Label::set_structured_text_bidi_override_options);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override_options"), &Label::get_structured_text_bidi_override_options);

	BIND_ENUM_CONSTANT(AUTOWRAP_OFF);
	BIND_ENUM_CONSTANT(AUTOWRAP_ARBITRARY);
	BIND_ENUM_CONSTANT(AUTOWRAP_WORD);
	BIND_ENUM_CONSTANT(AUTOWRAP_WORD_SMART);

	BIND_ENUM_CONSTANT(OVERRUN_NO_TRIMMING);
	BIND_ENUM_CONSTANT(OVERRUN_TRIM_CHAR);
	BIND_ENUM_CONSTANT(OVERRUN_TRIM_WORD);
	BIND_ENUM_CONSTANT(OVERRUN_TRIM_ELLIPSIS);
	BIND_ENUM_CONSTANT(OVERRUN_TRIM_WORD_ELLIPSIS);

	BIND_ENUM_CONSTANT(VC_CHARS_BEFORE_SHAPING);
	BIND_ENUM_CONSTANT(VC_CHARS_AFTER_SHAPING);
	BIND_ENUM_CONSTANT(VC_GLYPHS_AUTO);
	BIND_ENUM_CONSTANT(VC_GLYPHS_LTR);
	BIND_ENUM_CONSTANT(VC_GLYPHS_RTL);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT, "", PROPERTY_USAGE_DEFAULT_INTL), "set_text", "get_text");
	ADD_GROUP("Locale", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language", PROPERTY_HINT_LOCALE_ID, ""), "set_language", "get_language");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "horizontal_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_horizontal_alignment", "get_horizontal_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "vertical_alignment", PROPERTY_HINT_ENUM, "Top,Center,Bottom,Fill"), "set_vertical_alignment", "get_vertical_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "autowrap_mode", PROPERTY_HINT_ENUM, "Off,Arbitrary,Word,Word (Smart)"), "set_autowrap_mode", "get_autowrap_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_text"), "set_clip_text", "is_clipping_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_overrun_behavior", PROPERTY_HINT_ENUM, "Trim Nothing,Trim Characters,Trim Words,Ellipsis,Word Ellipsis"), "set_text_overrun_behavior", "get_text_overrun_behavior");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "uppercase"), "set_uppercase", "is_uppercase");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "visible_characters", PROPERTY_HINT_RANGE, "-1,128000,1"), "set_visible_characters", "get_visible_characters");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "visible_characters_behavior", PROPERTY_HINT_ENUM, "Characters Before Shaping,Characters After Shaping,Glyphs (Layout Direction),Glyphs (Left-to-Right),Glyphs (Right-to-Left)"), "set_visible_characters_behavior", "get_visible_characters_behavior");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "percent_visible", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_percent_visible", "get_percent_visible");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "lines_skipped", PROPERTY_HINT_RANGE, "0,999,1"), "set_lines_skipped", "get_lines_skipped");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_lines_visible", PROPERTY_HINT_RANGE, "-1,999,1"), "set_max_lines_visible", "get_max_lines_visible");
	ADD_GROUP("Structured Text", "structured_text_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "structured_text_bidi_override", PROPERTY_HINT_ENUM, "Default,URI,File,Email,List,None,Custom"), "set_structured_text_bidi_override", "get_structured_text_bidi_override");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "structured_text_bidi_override_options"), "set_structured_text_bidi_override_options", "get_structured_text_bidi_override_options");
}

Label::Label(const String &p_text) {
	text_rid = TS->create_shaped_text();

	set_mouse_filter(MOUSE_FILTER_IGNORE);
	set_text(p_text);
	set_v_size_flags(SIZE_SHRINK_CENTER);
}

Label::~Label() {
	for (int i = 0; i < lines_rid.size(); i++) {
		TS->free(lines_rid[i]);
	}
	lines_rid.clear();
	TS->free(text_rid);
}
