/*************************************************************************/
/*  label.cpp                                                            */
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

#include "label.h"

#include "core/config/project_settings.h"
#include "core/string/print_string.h"
#include "core/string/translation.h"

#include "servers/text_server.h"

void Label::set_autowrap(bool p_autowrap) {
	if (autowrap != p_autowrap) {
		autowrap = p_autowrap;
		lines_dirty = true;
	}
	update();

	if (clip) {
		minimum_size_changed();
	}
}

bool Label::has_autowrap() const {
	return autowrap;
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
	Ref<Font> font = get_theme_font("font");
	if (p_line >= 0 && p_line < lines_rid.size()) {
		return TS->shaped_text_get_size(lines_rid[p_line]).y + font->get_spacing(Font::SPACING_TOP) + font->get_spacing(Font::SPACING_BOTTOM);
	} else if (lines_rid.size() > 0) {
		int h = 0;
		for (int i = 0; i < lines_rid.size(); i++) {
			h = MAX(h, TS->shaped_text_get_size(lines_rid[i]).y) + font->get_spacing(Font::SPACING_TOP) + font->get_spacing(Font::SPACING_BOTTOM);
		}
		return h;
	} else {
		return font->get_height(get_theme_font_size("font_size"));
	}
}

void Label::_shape() {
	Ref<StyleBox> style = get_theme_stylebox("normal", "Label");
	int width = (get_size().width - style->get_minimum_size().width);

	if (dirty) {
		TS->shaped_text_clear(text_rid);
		if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
			TS->shaped_text_set_direction(text_rid, is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
		} else {
			TS->shaped_text_set_direction(text_rid, (TextServer::Direction)text_direction);
		}
		TS->shaped_text_add_string(text_rid, (uppercase) ? xl_text.to_upper() : xl_text, get_theme_font("font")->get_rids(), get_theme_font_size("font_size"), opentype_features, (language != "") ? language : TranslationServer::get_singleton()->get_tool_locale());
		TS->shaped_text_set_bidi_override(text_rid, structured_text_parser(st_parser, st_args, xl_text));
		dirty = false;
		lines_dirty = true;
	}
	if (lines_dirty) {
		for (int i = 0; i < lines_rid.size(); i++) {
			TS->free(lines_rid[i]);
		}
		lines_rid.clear();

		Vector<Vector2i> lines = TS->shaped_text_get_line_breaks(text_rid, width, 0, (autowrap) ? (TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND) : TextServer::BREAK_MANDATORY);
		for (int i = 0; i < lines.size(); i++) {
			RID line = TS->shaped_text_substr(text_rid, lines[i].x, lines[i].y - lines[i].x);
			lines_rid.push_back(line);
		}
	}

	if (xl_text.length() == 0) {
		minsize = Size2(1, get_line_height());
		return;
	}
	if (!autowrap) {
		minsize.width = 0.0f;
		for (int i = 0; i < lines_rid.size(); i++) {
			if (minsize.width < TS->shaped_text_get_size(lines_rid[i]).x) {
				minsize.width = TS->shaped_text_get_size(lines_rid[i]).x;
			}
		}
	}

	if (lines_dirty) { // Fill after min_size calculation.
		if (align == ALIGN_FILL) {
			for (int i = 0; i < lines_rid.size(); i++) {
				TS->shaped_text_fit_to_width(lines_rid.write[i], width);
			}
		}
		lines_dirty = false;
	}

	_update_visible();

	if (!autowrap || !clip) {
		minimum_size_changed();
	}
}

void Label::_update_visible() {
	int line_spacing = get_theme_constant("line_spacing", "Label");
	Ref<StyleBox> style = get_theme_stylebox("normal", "Label");
	Ref<Font> font = get_theme_font("font");
	int lines_visible = lines_rid.size();

	if (max_lines_visible >= 0 && lines_visible > max_lines_visible) {
		lines_visible = max_lines_visible;
	}

	minsize.height = 0;
	int last_line = MIN(lines_rid.size(), lines_visible + lines_skipped);
	for (int64_t i = lines_skipped; i < last_line; i++) {
		minsize.height += TS->shaped_text_get_size(lines_rid[i]).y + font->get_spacing(Font::SPACING_TOP) + font->get_spacing(Font::SPACING_BOTTOM) + line_spacing;
		if (minsize.height > (get_size().height - style->get_minimum_size().height + line_spacing)) {
			break;
		}
	}
}

void Label::_notification(int p_what) {
	if (p_what == NOTIFICATION_TRANSLATION_CHANGED) {
		String new_text = tr(text);
		if (new_text == xl_text) {
			return; //nothing new
		}
		xl_text = new_text;
		dirty = true;

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
		Ref<StyleBox> style = get_theme_stylebox("normal");
		Ref<Font> font = get_theme_font("font");
		Color font_color = get_theme_color("font_color");
		Color font_shadow_color = get_theme_color("font_shadow_color");
		Point2 shadow_ofs(get_theme_constant("shadow_offset_x"), get_theme_constant("shadow_offset_y"));
		int line_spacing = get_theme_constant("line_spacing");
		Color font_outline_color = get_theme_color("font_outline_color");
		int outline_size = get_theme_constant("outline_size");
		int shadow_outline_size = get_theme_constant("shadow_outline_size");
		bool rtl = is_layout_rtl();

		style->draw(ci, Rect2(Point2(0, 0), get_size()));

		float total_h = 0.0;
		int lines_visible = 0;

		// Get number of lines to fit to the height.
		for (int64_t i = lines_skipped; i < lines_rid.size(); i++) {
			total_h += TS->shaped_text_get_size(lines_rid[i]).y + font->get_spacing(Font::SPACING_TOP) + font->get_spacing(Font::SPACING_BOTTOM) + line_spacing;
			if (total_h > (get_size().height - style->get_minimum_size().height + line_spacing)) {
				break;
			}
			lines_visible++;
		}

		if (max_lines_visible >= 0 && lines_visible > max_lines_visible) {
			lines_visible = max_lines_visible;
		}

		int last_line = MIN(lines_rid.size(), lines_visible + lines_skipped);

		// Get real total height.
		total_h = 0;
		for (int64_t i = lines_skipped; i < last_line; i++) {
			total_h += TS->shaped_text_get_size(lines_rid[i]).y + font->get_spacing(Font::SPACING_TOP) + font->get_spacing(Font::SPACING_BOTTOM) + line_spacing;
		}

		int vbegin = 0, vsep = 0;
		if (lines_visible > 0) {
			switch (valign) {
				case VALIGN_TOP: {
					//nothing
				} break;
				case VALIGN_CENTER: {
					vbegin = (size.y - (total_h - line_spacing)) / 2;
					vsep = 0;

				} break;
				case VALIGN_BOTTOM: {
					vbegin = size.y - (total_h - line_spacing);
					vsep = 0;

				} break;
				case VALIGN_FILL: {
					vbegin = 0;
					if (lines_visible > 1) {
						vsep = (size.y - (total_h - line_spacing)) / (lines_visible - 1);
					} else {
						vsep = 0;
					}

				} break;
			}
		}

		int visible_glyphs = -1;
		int glyhps_drawn = 0;
		if (percent_visible < 1) {
			int total_glyphs = 0;
			for (int i = lines_skipped; i < last_line; i++) {
				const Vector<TextServer::Glyph> visual = TS->shaped_text_get_glyphs(lines_rid[i]);
				const TextServer::Glyph *glyphs = visual.ptr();
				int gl_size = visual.size();
				for (int j = 0; j < gl_size; j++) {
					if ((glyphs[j].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
						total_glyphs++;
					}
				}
			}

			visible_glyphs = MIN(total_glyphs, visible_chars);
		}

		Vector2 ofs;
		ofs.y = style->get_offset().y + vbegin;
		for (int i = lines_skipped; i < last_line; i++) {
			ofs.x = 0;
			ofs.y += TS->shaped_text_get_ascent(lines_rid[i]) + font->get_spacing(Font::SPACING_TOP);
			switch (align) {
				case ALIGN_FILL:
				case ALIGN_LEFT: {
					if (rtl) {
						ofs.x = int(size.width - style->get_margin(SIDE_RIGHT) - TS->shaped_text_get_size(lines_rid[i]).x);
					} else {
						ofs.x = style->get_offset().x;
					}
				} break;
				case ALIGN_CENTER: {
					ofs.x = int(size.width - TS->shaped_text_get_size(lines_rid[i]).x) / 2;
				} break;
				case ALIGN_RIGHT: {
					if (rtl) {
						ofs.x = style->get_offset().x;
					} else {
						ofs.x = int(size.width - style->get_margin(SIDE_RIGHT) - TS->shaped_text_get_size(lines_rid[i]).x);
					}
				} break;
			}

			const Vector<TextServer::Glyph> visual = TS->shaped_text_get_glyphs(lines_rid[i]);
			const TextServer::Glyph *glyphs = visual.ptr();
			int gl_size = visual.size();

			float x = ofs.x;
			int outlines_drawn = glyhps_drawn;
			for (int j = 0; j < gl_size; j++) {
				for (int k = 0; k < glyphs[j].repeat; k++) {
					if (glyphs[j].font_rid != RID()) {
						if (font_shadow_color.a > 0) {
							TS->font_draw_glyph(glyphs[j].font_rid, ci, glyphs[j].font_size, ofs + Vector2(glyphs[j].x_off, glyphs[j].y_off) + shadow_ofs, glyphs[j].index, font_shadow_color);
							if (shadow_outline_size > 0) {
								//draw shadow
								TS->font_draw_glyph_outline(glyphs[j].font_rid, ci, glyphs[j].font_size, shadow_outline_size, ofs + Vector2(glyphs[j].x_off, glyphs[j].y_off) + Vector2(-shadow_ofs.x, shadow_ofs.y), glyphs[j].index, font_shadow_color);
								TS->font_draw_glyph_outline(glyphs[j].font_rid, ci, glyphs[j].font_size, shadow_outline_size, ofs + Vector2(glyphs[j].x_off, glyphs[j].y_off) + Vector2(shadow_ofs.x, -shadow_ofs.y), glyphs[j].index, font_shadow_color);
								TS->font_draw_glyph_outline(glyphs[j].font_rid, ci, glyphs[j].font_size, shadow_outline_size, ofs + Vector2(glyphs[j].x_off, glyphs[j].y_off) + Vector2(-shadow_ofs.x, -shadow_ofs.y), glyphs[j].index, font_shadow_color);
							}
						}
						if (font_outline_color.a != 0.0 && outline_size > 0) {
							TS->font_draw_glyph_outline(glyphs[j].font_rid, ci, glyphs[j].font_size, outline_size, ofs + Vector2(glyphs[j].x_off, glyphs[j].y_off), glyphs[j].index, font_outline_color);
						}
					}
					ofs.x += glyphs[j].advance;
				}
				if (visible_glyphs != -1) {
					if ((glyphs[j].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
						outlines_drawn++;
						if (outlines_drawn >= visible_glyphs) {
							break;
						}
					}
				}
			}
			ofs.x = x;

			for (int j = 0; j < gl_size; j++) {
				for (int k = 0; k < glyphs[j].repeat; k++) {
					if (glyphs[j].font_rid != RID()) {
						TS->font_draw_glyph(glyphs[j].font_rid, ci, glyphs[j].font_size, ofs + Vector2(glyphs[j].x_off, glyphs[j].y_off), glyphs[j].index, font_color);
					} else if ((glyphs[j].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
						TS->draw_hex_code_box(ci, glyphs[j].font_size, ofs + Vector2(glyphs[j].x_off, glyphs[j].y_off), glyphs[j].index, font_color);
					}
					ofs.x += glyphs[j].advance;
				}
				if (visible_glyphs != -1) {
					if ((glyphs[j].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
						glyhps_drawn++;
						if (glyhps_drawn >= visible_glyphs) {
							return;
						}
					}
				}
			}

			ofs.y += TS->shaped_text_get_descent(lines_rid[i]) + vsep + line_spacing + font->get_spacing(Font::SPACING_BOTTOM);
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

	Ref<Font> font = get_theme_font("font");
	min_size.height = MAX(min_size.height, font->get_height(get_theme_font_size("font_size")) + font->get_spacing(Font::SPACING_TOP) + font->get_spacing(Font::SPACING_BOTTOM));

	Size2 min_style = get_theme_stylebox("normal")->get_minimum_size();
	if (autowrap) {
		return Size2(1, clip ? 1 : min_size.height) + min_style;
	} else {
		if (clip) {
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
	Ref<Font> font = get_theme_font("font");
	Ref<StyleBox> style = get_theme_stylebox("normal");
	int line_spacing = get_theme_constant("line_spacing");
	int lines_visible = 0;
	float total_h = 0.0;
	for (int64_t i = lines_skipped; i < lines_rid.size(); i++) {
		total_h += TS->shaped_text_get_size(lines_rid[i]).y + font->get_spacing(Font::SPACING_TOP) + font->get_spacing(Font::SPACING_BOTTOM) + line_spacing;
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

void Label::set_align(Align p_align) {
	ERR_FAIL_INDEX((int)p_align, 4);
	if (align != p_align) {
		if (align == ALIGN_FILL || p_align == ALIGN_FILL) {
			lines_dirty = true; // Reshape lines.
		}
		align = p_align;
	}
	update();
}

Label::Align Label::get_align() const {
	return align;
}

void Label::set_valign(VAlign p_align) {
	ERR_FAIL_INDEX((int)p_align, 4);
	valign = p_align;
	update();
}

Label::VAlign Label::get_valign() const {
	return valign;
}

void Label::set_text(const String &p_string) {
	if (text == p_string) {
		return;
	}
	text = p_string;
	xl_text = tr(p_string);
	dirty = true;
	if (percent_visible < 1) {
		visible_chars = get_total_character_count() * percent_visible;
	}
	update();
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
	minimum_size_changed();
}

bool Label::is_clipping_text() const {
	return clip;
}

String Label::get_text() const {
	return text;
}

void Label::set_visible_characters(int p_amount) {
	visible_chars = p_amount;
	if (get_total_character_count() > 0) {
		percent_visible = (float)p_amount / (float)get_total_character_count();
	} else {
		percent_visible = 1.0;
	}
	update();
}

int Label::get_visible_characters() const {
	return visible_chars;
}

void Label::set_percent_visible(float p_percent) {
	if (p_percent < 0 || p_percent >= 1) {
		visible_chars = -1;
		percent_visible = 1;

	} else {
		visible_chars = get_total_character_count() * p_percent;
		percent_visible = p_percent;
	}
	update();
}

float Label::get_percent_visible() const {
	return percent_visible;
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
		double value = p_value;
		if (value == -1) {
			if (opentype_features.has(tag)) {
				opentype_features.erase(tag);
				dirty = true;
				update();
			}
		} else {
			if ((double)opentype_features[tag] != value) {
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
		p_list->push_back(PropertyInfo(Variant::FLOAT, "opentype_features/" + name));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "opentype_features/_new", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
}

void Label::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_align", "align"), &Label::set_align);
	ClassDB::bind_method(D_METHOD("get_align"), &Label::get_align);
	ClassDB::bind_method(D_METHOD("set_valign", "valign"), &Label::set_valign);
	ClassDB::bind_method(D_METHOD("get_valign"), &Label::get_valign);
	ClassDB::bind_method(D_METHOD("set_text", "text"), &Label::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &Label::get_text);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &Label::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &Label::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_opentype_feature", "tag", "value"), &Label::set_opentype_feature);
	ClassDB::bind_method(D_METHOD("get_opentype_feature", "tag"), &Label::get_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_opentype_features"), &Label::clear_opentype_features);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &Label::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &Label::get_language);
	ClassDB::bind_method(D_METHOD("set_autowrap", "enable"), &Label::set_autowrap);
	ClassDB::bind_method(D_METHOD("has_autowrap"), &Label::has_autowrap);
	ClassDB::bind_method(D_METHOD("set_clip_text", "enable"), &Label::set_clip_text);
	ClassDB::bind_method(D_METHOD("is_clipping_text"), &Label::is_clipping_text);
	ClassDB::bind_method(D_METHOD("set_uppercase", "enable"), &Label::set_uppercase);
	ClassDB::bind_method(D_METHOD("is_uppercase"), &Label::is_uppercase);
	ClassDB::bind_method(D_METHOD("get_line_height", "line"), &Label::get_line_height, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("get_line_count"), &Label::get_line_count);
	ClassDB::bind_method(D_METHOD("get_visible_line_count"), &Label::get_visible_line_count);
	ClassDB::bind_method(D_METHOD("get_total_character_count"), &Label::get_total_character_count);
	ClassDB::bind_method(D_METHOD("set_visible_characters", "amount"), &Label::set_visible_characters);
	ClassDB::bind_method(D_METHOD("get_visible_characters"), &Label::get_visible_characters);
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

	BIND_ENUM_CONSTANT(ALIGN_LEFT);
	BIND_ENUM_CONSTANT(ALIGN_CENTER);
	BIND_ENUM_CONSTANT(ALIGN_RIGHT);
	BIND_ENUM_CONSTANT(ALIGN_FILL);

	BIND_ENUM_CONSTANT(VALIGN_TOP);
	BIND_ENUM_CONSTANT(VALIGN_CENTER);
	BIND_ENUM_CONSTANT(VALIGN_BOTTOM);
	BIND_ENUM_CONSTANT(VALIGN_FILL);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT, "", PROPERTY_USAGE_DEFAULT_INTL), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language"), "set_language", "get_language");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "align", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_align", "get_align");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "valign", PROPERTY_HINT_ENUM, "Top,Center,Bottom,Fill"), "set_valign", "get_valign");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autowrap"), "set_autowrap", "has_autowrap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_text"), "set_clip_text", "is_clipping_text");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "uppercase"), "set_uppercase", "is_uppercase");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "visible_characters", PROPERTY_HINT_RANGE, "-1,128000,1", PROPERTY_USAGE_EDITOR), "set_visible_characters", "get_visible_characters");
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
