/*************************************************************************/
/*  label.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "core/print_string.h"
#include "core/project_settings.h"
#include "core/translation.h"

void Label::set_autowrap(bool p_autowrap) {

	if (autowrap != p_autowrap) {
		autowrap = p_autowrap;
		_lines_dirty = true;
	}
	update();
}

bool Label::has_autowrap() const {

	return autowrap;
}

void Label::set_uppercase(bool p_uppercase) {

	if (uppercase != p_uppercase) {
		uppercase = p_uppercase;
		s_paragraph->set_text((uppercase) ? xl_text.to_upper() : xl_text);
		_lines_dirty = true;
	}
	update();
}

bool Label::is_uppercase() const {

	return uppercase;
}

int Label::get_line_height() const {

	float total_h = 0.0;
	for (int i = 0; i < s_lines.size(); i++) {
		total_h += s_lines[i]->get_height();
	}
	return MAX(get_font("font")->get_height(), total_h / s_lines.size());
}

void Label::_notification(int p_what) {

	if (p_what == NOTIFICATION_TRANSLATION_CHANGED) {

		String new_text = tr(text);
		if (new_text == xl_text)
			return; //nothing new
		xl_text = new_text;

		s_paragraph->set_text(xl_text);
		_reshape_lines();

		update();
	}

	if (p_what == NOTIFICATION_DRAW) {

		if (clip) {
			VisualServer::get_singleton()->canvas_item_set_clip(get_canvas_item(), true);
		}

		if (_lines_dirty)
			_reshape_lines();

		RID ci = get_canvas_item();

		Size2 string_size;
		Size2 size = get_size();
		Ref<StyleBox> style = get_stylebox("normal");
		Ref<Font> font = get_font("font");
		Color font_color = get_color("font_color");
		Color font_color_shadow = get_color("font_color_shadow");
		bool use_outline = get_constant("shadow_as_outline");
		Point2 shadow_ofs(get_constant("shadow_offset_x"), get_constant("shadow_offset_y"));
		int line_spacing = get_constant("line_spacing");
		Color font_outline_modulate = get_color("font_outline_modulate");

		style->draw(ci, Rect2(Point2(0, 0), get_size()));

		VisualServer::get_singleton()->canvas_item_set_distance_field_mode(get_canvas_item(), font.is_valid() && font->is_distance_field_hint());

		int vbegin = 0, vsep = 0;

		float total_h = 0.0;
		int lines_visible = 0;
		for (int i = lines_skipped; i < s_lines.size(); i++) {
			total_h += s_lines[i]->get_height() + line_spacing;
			if (total_h > (get_size().height - get_stylebox("normal")->get_minimum_size().height + line_spacing)) {
				break;
			}
			lines_visible++;
		}

		if (lines_visible > s_lines.size())
			lines_visible = s_lines.size();

		if (max_lines_visible >= 0 && lines_visible > max_lines_visible)
			lines_visible = max_lines_visible;

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

		Vector2 ofs;
		ofs.y = style->get_offset().y + vbegin;
		for (int j = lines_skipped; j < s_lines.size(); j++) {
			ofs.y += s_lines[j]->get_ascent();
			switch (align) {
				case ALIGN_FILL:
				case ALIGN_LEFT: {

					ofs.x = style->get_offset().x;
				} break;
				case ALIGN_CENTER: {

					ofs.x = int(size.width - s_lines[j]->get_width()) / 2;
				} break;
				case ALIGN_RIGHT: {

					ofs.x = int(size.width - style->get_margin(MARGIN_RIGHT) - s_lines[j]->get_width());
				} break;
			}
			if (font_color_shadow.a > 0) {
				s_lines[j]->draw(ci, ofs + shadow_ofs, font_color_shadow, false);
				if (use_outline) {
					//draw shadow
					s_lines[j]->draw(ci, ofs + Vector2(-shadow_ofs.x, shadow_ofs.y), font_color_shadow, false);
					s_lines[j]->draw(ci, ofs + Vector2(shadow_ofs.x, -shadow_ofs.y), font_color_shadow, false);
					s_lines[j]->draw(ci, ofs + Vector2(-shadow_ofs.x, -shadow_ofs.y), font_color_shadow, false);
				}
			}

			s_lines[j]->draw(ci, ofs, font->has_outline() ? font_outline_modulate : font_color, font->has_outline());

			if (font->has_outline()) {
				s_lines[j]->draw(ci, ofs, font_color, false);
			}
			ofs.y += s_lines[j]->get_descent() + vsep + line_spacing;
		}
	}
	if (p_what == NOTIFICATION_THEME_CHANGED) {

		_lines_dirty = true;
		update();
	}
	if (p_what == NOTIFICATION_RESIZED) {

		_lines_dirty = true;
	}
}

Size2 Label::get_minimum_size() const {

	Size2 min_style = get_stylebox("normal")->get_minimum_size();

	// don't want to mutable everything
	if (_lines_dirty)
		const_cast<Label *>(this)->_reshape_lines();

	if (autowrap)
		return Size2(1, clip ? 1 : minsize.height) + min_style;
	else {
		Size2 ms = minsize;
		if (clip)
			ms.width = 1;
		return ms + min_style;
	}
}

int Label::get_line_count() const {

	if (!is_inside_tree())
		return 1;

	if (_lines_dirty)
		const_cast<Label *>(this)->_reshape_lines();

	return s_lines.size();
}

int Label::get_visible_line_count() const {

	int line_spacing = get_constant("line_spacing");

	if (_lines_dirty)
		const_cast<Label *>(this)->_reshape_lines();

	float total_h = 0.0;
	int lines_visible = 0;
	for (int i = lines_skipped; i < s_lines.size(); i++) {
		total_h += s_lines[i]->get_height() + line_spacing;
		if (total_h > (get_size().height - get_stylebox("normal")->get_minimum_size().height + line_spacing)) {
			break;
		}
		lines_visible++;
	}

	if (lines_visible > s_lines.size())
		lines_visible = s_lines.size();

	if (max_lines_visible >= 0 && lines_visible > max_lines_visible)
		lines_visible = max_lines_visible;

	return lines_visible;
}

void Label::_reshape_lines() {

	Ref<StyleBox> style = get_stylebox("normal");
	Ref<Font> font = get_font("font");
	int line_spacing = get_constant("line_spacing");

	s_paragraph->set_base_font(font);
	s_lines.clear();

	int width = (get_size().width - style->get_minimum_size().width);

	if (xl_text.size() == 0) {
		minsize = Size2(width, font->get_height());
		return;
	}

	Vector<int> l_lines = s_paragraph->break_lines(width, (autowrap) ? TEXT_BREAK_MANDATORY_AND_WORD_BOUND : TEXT_BREAK_MANDATORY);

	int line_start = 0;
	for (int i = 0; i < l_lines.size(); i++) {
		Ref<ShapedString> _ln = s_paragraph->substr(line_start, l_lines[i]);
		if (!_ln.is_null()) s_lines.push_back(_ln);
		line_start = l_lines[i];
	}

	if (!autowrap) {
		minsize.width = 0.0f;
		for (int i = 0; i < s_lines.size(); i++) {
			if (minsize.width < s_lines[i]->get_width()) {
				minsize.width = s_lines[i]->get_width();
			}
		}
	}

	if (max_lines_visible > 0 && s_lines.size() > max_lines_visible) {
		minsize.height = (font->get_height() * max_lines_visible) + (line_spacing * (max_lines_visible - 1));
	} else {
		minsize.height = (font->get_height() * s_lines.size()) + (line_spacing * (s_lines.size() - 1));
	}

	if (align == ALIGN_FILL) {
		for (int i = 0; i < s_lines.size(); i++) {
			s_lines.write[i]->extend_to_width(width, TEXT_JUSTIFICATION_KASHIDA_AND_WHITESPACE);
		}
	}

	if (!autowrap || !clip) {
		//helps speed up some labels that may change a lot, as no resizing is requested. Do not change.
		minimum_size_changed();
	}

	_lines_dirty = false;
}

void Label::set_text_direction(TextDirection p_text_direction) {

	ERR_FAIL_INDEX((int)p_text_direction, 4);
	if (base_direction != p_text_direction) {
		base_direction = p_text_direction;
		s_paragraph->set_base_direction(base_direction);
		_lines_dirty = true;
		update();
	}
}

TextDirection Label::get_text_direction() const {

	return base_direction;
}

void Label::set_ot_features(const String &p_features) {

	if (ot_features != p_features) {
		ot_features = p_features;
		s_paragraph->set_features(ot_features);
		_lines_dirty = true;
		update();
	}
}

String Label::get_ot_features() const {

	return ot_features;
}

void Label::set_language(const String &p_language) {

	if (language != p_language) {
		language = p_language;
		s_paragraph->set_language(language);
		_lines_dirty = true;
		update();
	}
}

String Label::get_language() const {

	return language;
}

void Label::set_align(Align p_align) {

	ERR_FAIL_INDEX((int)p_align, 4);

	if (align != p_align) {
		align = p_align;
		_lines_dirty = true;
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

	if (text == p_string)
		return;
	text = p_string;
	xl_text = tr(p_string);
	s_paragraph->set_text(xl_text);
	_lines_dirty = true;
	if (percent_visible < 1)
		visible_chars = get_total_character_count() * percent_visible;
	update();
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
	}
	_change_notify("percent_visible");
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
	_change_notify("visible_chars");
	update();
}

float Label::get_percent_visible() const {

	return percent_visible;
}

void Label::set_lines_skipped(int p_lines) {

	lines_skipped = p_lines;
	update();
}

int Label::get_lines_skipped() const {

	return lines_skipped;
}

void Label::set_max_lines_visible(int p_lines) {

	max_lines_visible = p_lines;
	update();
}

int Label::get_max_lines_visible() const {

	return max_lines_visible;
}

int Label::get_total_character_count() const {

	/* ??? */
	return xl_text.size();
}

void Label::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_align", "align"), &Label::set_align);
	ClassDB::bind_method(D_METHOD("get_align"), &Label::get_align);
	ClassDB::bind_method(D_METHOD("set_valign", "valign"), &Label::set_valign);
	ClassDB::bind_method(D_METHOD("get_valign"), &Label::get_valign);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &Label::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &Label::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_ot_features", "features"), &Label::set_ot_features);
	ClassDB::bind_method(D_METHOD("get_ot_features"), &Label::get_ot_features);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &Label::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &Label::get_language);

	ClassDB::bind_method(D_METHOD("set_text", "text"), &Label::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &Label::get_text);
	ClassDB::bind_method(D_METHOD("set_autowrap", "enable"), &Label::set_autowrap);
	ClassDB::bind_method(D_METHOD("has_autowrap"), &Label::has_autowrap);
	ClassDB::bind_method(D_METHOD("set_clip_text", "enable"), &Label::set_clip_text);
	ClassDB::bind_method(D_METHOD("is_clipping_text"), &Label::is_clipping_text);
	ClassDB::bind_method(D_METHOD("set_uppercase", "enable"), &Label::set_uppercase);
	ClassDB::bind_method(D_METHOD("is_uppercase"), &Label::is_uppercase);
	ClassDB::bind_method(D_METHOD("get_line_height"), &Label::get_line_height);
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

	BIND_ENUM_CONSTANT(ALIGN_LEFT);
	BIND_ENUM_CONSTANT(ALIGN_CENTER);
	BIND_ENUM_CONSTANT(ALIGN_RIGHT);
	BIND_ENUM_CONSTANT(ALIGN_FILL);

	BIND_ENUM_CONSTANT(VALIGN_TOP);
	BIND_ENUM_CONSTANT(VALIGN_CENTER);
	BIND_ENUM_CONSTANT(VALIGN_BOTTOM);
	BIND_ENUM_CONSTANT(VALIGN_FILL);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT, "", PROPERTY_USAGE_DEFAULT_INTL), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "align", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_align", "get_align");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "valign", PROPERTY_HINT_ENUM, "Top,Center,Bottom,Fill"), "set_valign", "get_valign");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "LTR,RTL,Locale,Auto"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "ot_features"), "set_ot_features", "get_ot_features");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language"), "set_language", "get_language");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autowrap"), "set_autowrap", "has_autowrap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_text"), "set_clip_text", "is_clipping_text");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "uppercase"), "set_uppercase", "is_uppercase");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "visible_characters", PROPERTY_HINT_RANGE, "-1,128000,1", PROPERTY_USAGE_EDITOR), "set_visible_characters", "get_visible_characters");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "percent_visible", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_percent_visible", "get_percent_visible");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "lines_skipped", PROPERTY_HINT_RANGE, "0,999,1"), "set_lines_skipped", "get_lines_skipped");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_lines_visible", PROPERTY_HINT_RANGE, "-1,999,1"), "set_max_lines_visible", "get_max_lines_visible");
}

Label::Label(const String &p_text) {

	s_paragraph.instance();

	base_direction = TEXT_DIRECTION_AUTO;
	ot_features = "";
	language = "";

	align = ALIGN_LEFT;
	valign = VALIGN_TOP;
	xl_text = "";

	_lines_dirty = true;

	autowrap = false;

	set_v_size_flags(0);
	clip = false;
	set_mouse_filter(MOUSE_FILTER_IGNORE);

	visible_chars = -1;
	percent_visible = 1;
	lines_skipped = 0;
	max_lines_visible = -1;
	set_text(p_text);
	uppercase = false;
	set_v_size_flags(SIZE_SHRINK_CENTER);
}
