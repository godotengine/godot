/*************************************************************************/
/*  link_button.cpp                                                      */
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

#include "link_button.h"

void LinkButton::set_text(const String &p_text) {
	text = p_text;
	update();
	minimum_size_changed();
}

String LinkButton::get_text() const {
	return text;
}

void LinkButton::set_underline_mode(UnderlineMode p_underline_mode) {
	underline_mode = p_underline_mode;
	update();
}

LinkButton::UnderlineMode LinkButton::get_underline_mode() const {
	return underline_mode;
}

Size2 LinkButton::get_minimum_size() const {
	return get_theme_font("font")->get_string_size(text);
}

void LinkButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			Size2 size = get_size();
			Color color;
			bool do_underline = false;

			switch (get_draw_mode()) {
				case DRAW_NORMAL: {
					color = get_theme_color("font_color");
					do_underline = underline_mode == UNDERLINE_MODE_ALWAYS;
				} break;
				case DRAW_HOVER_PRESSED:
				case DRAW_PRESSED: {
					if (has_theme_color("font_color_pressed")) {
						color = get_theme_color("font_color_pressed");
					} else {
						color = get_theme_color("font_color");
					}

					do_underline = underline_mode != UNDERLINE_MODE_NEVER;

				} break;
				case DRAW_HOVER: {
					color = get_theme_color("font_color_hover");
					do_underline = underline_mode != UNDERLINE_MODE_NEVER;

				} break;
				case DRAW_DISABLED: {
					color = get_theme_color("font_color_disabled");
					do_underline = underline_mode == UNDERLINE_MODE_ALWAYS;

				} break;
			}

			if (has_focus()) {
				Ref<StyleBox> style = get_theme_stylebox("focus");
				style->draw(ci, Rect2(Point2(), size));
			}

			Ref<Font> font = get_theme_font("font");

			draw_string(font, Vector2(0, font->get_ascent()), text, color);

			if (do_underline) {
				int underline_spacing = get_theme_constant("underline_spacing") + font->get_underline_position();
				int width = font->get_string_size(text).width;
				int y = font->get_ascent() + underline_spacing;

				draw_line(Vector2(0, y), Vector2(width, y), color, font->get_underline_thickness());
			}

		} break;
	}
}

void LinkButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_text", "text"), &LinkButton::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &LinkButton::get_text);

	ClassDB::bind_method(D_METHOD("set_underline_mode", "underline_mode"), &LinkButton::set_underline_mode);
	ClassDB::bind_method(D_METHOD("get_underline_mode"), &LinkButton::get_underline_mode);

	BIND_ENUM_CONSTANT(UNDERLINE_MODE_ALWAYS);
	BIND_ENUM_CONSTANT(UNDERLINE_MODE_ON_HOVER);
	BIND_ENUM_CONSTANT(UNDERLINE_MODE_NEVER);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text"), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "underline", PROPERTY_HINT_ENUM, "Always,On Hover,Never"), "set_underline_mode", "get_underline_mode");
}

LinkButton::LinkButton() {
	underline_mode = UNDERLINE_MODE_ALWAYS;
	set_default_cursor_shape(CURSOR_POINTING_HAND);
}
