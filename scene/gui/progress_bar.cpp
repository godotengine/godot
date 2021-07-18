/*************************************************************************/
/*  progress_bar.cpp                                                     */
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

#include "progress_bar.h"
#include "scene/resources/text_line.h"

Size2 ProgressBar::get_minimum_size() const {
	Ref<StyleBox> bg = get_theme_stylebox("bg");
	Ref<StyleBox> fg = get_theme_stylebox("fg");
	Ref<Font> font = get_theme_font("font");
	int font_size = get_theme_font_size("font_size");

	Size2 minimum_size = bg->get_minimum_size();
	minimum_size.height = MAX(minimum_size.height, fg->get_minimum_size().height);
	minimum_size.width = MAX(minimum_size.width, fg->get_minimum_size().width);

	switch (progress_label_format) {
		case FORMAT_NONE: // this is needed, else the progressbar will collapse
			minimum_size.width = MAX(minimum_size.width, 1);
			minimum_size.height = MAX(minimum_size.height, 1);
			break;
		case FORMAT_PERCENTAGE:
		case FORMAT_DECIMAL:
		case FORMAT_FRACTION:
			String txt = "100%";
			TextLine tl = TextLine(txt, font, font_size);
			minimum_size.height = MAX(minimum_size.height, bg->get_minimum_size().height + tl.get_size().y);
			break;
	}

	return minimum_size;
}

void ProgressBar::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		Ref<StyleBox> bg = get_theme_stylebox("bg");
		Ref<StyleBox> fg = get_theme_stylebox("fg");
		Ref<Font> font = get_theme_font("font");
		int font_size = get_theme_font_size("font_size");
		Color font_color = get_theme_color("font_color");

		draw_style_box(bg, Rect2(Point2(), get_size()));
		float r = get_as_ratio();
		int mp = fg->get_minimum_size().width;
		int p = r * (get_size().width - mp);
		if (p > 0) {
			if (is_layout_rtl()) {
				draw_style_box(fg, Rect2(Point2(p, 0), Size2(fg->get_minimum_size().width, get_size().height)));
			} else {
				draw_style_box(fg, Rect2(Point2(0, 0), Size2(p + fg->get_minimum_size().width, get_size().height)));
			}
		}

		if (progress_label_format != FORMAT_NONE) {
			String txt = "";

			switch (progress_label_format) {
				case FORMAT_PERCENTAGE:
					txt = TS->format_number(itos(int(get_as_ratio() * 100))) + TS->percent_sign();
					break;
				case FORMAT_DECIMAL:
					txt = TS->format_number(rtos(Math::snapped(get_as_ratio(), 0.001)));
					break;
				case FORMAT_FRACTION:
					txt = TS->format_number(rtos(Math::snapped(get_value(), 0.001))) + "/" + TS->format_number(rtos(Math::snapped(get_max(), 0.001)));
					break;
				default:
					break;
			}

			TextLine tl = TextLine(txt, font, font_size);
			Vector2 text_pos = (Point2(get_size().width - tl.get_size().x, get_size().height - tl.get_size().y) / 2).round();
			Color font_outline_color = get_theme_color("font_outline_color");
			int outline_size = get_theme_constant("outline_size");
			if (outline_size > 0 && font_outline_color.a > 0) {
				tl.draw_outline(get_canvas_item(), text_pos, outline_size, font_outline_color);
			}
			tl.draw(get_canvas_item(), text_pos, font_color);
		}
	}
}

void ProgressBar::set_progress_label_format(ProgressLabelFormat p_type) {
	progress_label_format = p_type;
	update();
}

ProgressBar::ProgressLabelFormat ProgressBar::get_progress_label_format() const {
	return progress_label_format;
}

void ProgressBar::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_progress_label_format", "progress_label_format"), &ProgressBar::set_progress_label_format);
	ClassDB::bind_method(D_METHOD("get_progress_label_format"), &ProgressBar::get_progress_label_format);

	BIND_ENUM_CONSTANT(FORMAT_NONE);
	BIND_ENUM_CONSTANT(FORMAT_PERCENTAGE);
	BIND_ENUM_CONSTANT(FORMAT_DECIMAL);
	BIND_ENUM_CONSTANT(FORMAT_FRACTION);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "progress_label_format", PROPERTY_HINT_ENUM, "None,Percentage,Decimal,Fraction"), "set_progress_label_format", "get_progress_label_format");
}

ProgressBar::ProgressBar() {
	set_v_size_flags(0);
	set_step(0.01);
}
