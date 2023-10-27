/**************************************************************************/
/*  progress_bar.cpp                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "progress_bar.h"

#include "scene/resources/text_line.h"
#include "scene/theme/theme_db.h"

Size2 ProgressBar::get_minimum_size() const {
	Size2 minimum_size = theme_cache.background_style->get_minimum_size();
	minimum_size.height = MAX(minimum_size.height, theme_cache.fill_style->get_minimum_size().height);
	minimum_size.width = MAX(minimum_size.width, theme_cache.fill_style->get_minimum_size().width);
	if (show_percentage) {
		String txt = "100%";
		TextLine tl = TextLine(txt, theme_cache.font, theme_cache.font_size);
		minimum_size.height = MAX(minimum_size.height, theme_cache.background_style->get_minimum_size().height + tl.get_size().y);
	} else { // this is needed, else the progressbar will collapse
		minimum_size.width = MAX(minimum_size.width, 1);
		minimum_size.height = MAX(minimum_size.height, 1);
	}
	return minimum_size;
}

void ProgressBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			draw_style_box(theme_cache.background_style, Rect2(Point2(), get_size()));

			float r = get_as_ratio();

			switch (mode) {
				case FILL_BEGIN_TO_END:
				case FILL_END_TO_BEGIN: {
					int mp = theme_cache.fill_style->get_minimum_size().width;
					int p = round(r * (get_size().width - mp));
					// We want FILL_BEGIN_TO_END to map to right to left when UI layout is RTL,
					// and left to right otherwise. And likewise for FILL_END_TO_BEGIN.
					bool right_to_left = is_layout_rtl() ? (mode == FILL_BEGIN_TO_END) : (mode == FILL_END_TO_BEGIN);
					if (p > 0) {
						if (right_to_left) {
							int p_remaining = round((1.0 - r) * (get_size().width - mp));
							draw_style_box(theme_cache.fill_style, Rect2(Point2(p_remaining, 0), Size2(p + theme_cache.fill_style->get_minimum_size().width, get_size().height)));
						} else {
							draw_style_box(theme_cache.fill_style, Rect2(Point2(0, 0), Size2(p + theme_cache.fill_style->get_minimum_size().width, get_size().height)));
						}
					}
				} break;
				case FILL_TOP_TO_BOTTOM:
				case FILL_BOTTOM_TO_TOP: {
					int mp = theme_cache.fill_style->get_minimum_size().height;
					int p = round(r * (get_size().height - mp));

					if (p > 0) {
						if (mode == FILL_TOP_TO_BOTTOM) {
							draw_style_box(theme_cache.fill_style, Rect2(Point2(0, 0), Size2(get_size().width, p + theme_cache.fill_style->get_minimum_size().height)));
						} else {
							int p_remaining = round((1.0 - r) * (get_size().height - mp));
							draw_style_box(theme_cache.fill_style, Rect2(Point2(0, p_remaining), Size2(get_size().width, p + theme_cache.fill_style->get_minimum_size().height)));
						}
					}
				} break;
				case FILL_MODE_MAX:
					break;
			}

			if (show_percentage) {
				String txt = itos(int(get_as_ratio() * 100));
				if (is_localizing_numeral_system()) {
					txt = TS->format_number(txt) + TS->percent_sign();
				} else {
					txt += String("%");
				}
				TextLine tl = TextLine(txt, theme_cache.font, theme_cache.font_size);
				Vector2 text_pos = (Point2(get_size().width - tl.get_size().x, get_size().height - tl.get_size().y) / 2).round();

				if (theme_cache.font_outline_size > 0 && theme_cache.font_outline_color.a > 0) {
					tl.draw_outline(get_canvas_item(), text_pos, theme_cache.font_outline_size, theme_cache.font_outline_color);
				}

				tl.draw(get_canvas_item(), text_pos, theme_cache.font_color);
			}
		} break;
	}
}

void ProgressBar::set_fill_mode(int p_fill) {
	ERR_FAIL_INDEX(p_fill, FILL_MODE_MAX);
	mode = (FillMode)p_fill;
	queue_redraw();
}

int ProgressBar::get_fill_mode() {
	return mode;
}

void ProgressBar::set_show_percentage(bool p_visible) {
	if (show_percentage == p_visible) {
		return;
	}
	show_percentage = p_visible;
	update_minimum_size();
	queue_redraw();
}

bool ProgressBar::is_percentage_shown() const {
	return show_percentage;
}

void ProgressBar::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_fill_mode", "mode"), &ProgressBar::set_fill_mode);
	ClassDB::bind_method(D_METHOD("get_fill_mode"), &ProgressBar::get_fill_mode);
	ClassDB::bind_method(D_METHOD("set_show_percentage", "visible"), &ProgressBar::set_show_percentage);
	ClassDB::bind_method(D_METHOD("is_percentage_shown"), &ProgressBar::is_percentage_shown);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "fill_mode", PROPERTY_HINT_ENUM, "Begin to End,End to Begin,Top to Bottom,Bottom to Top"), "set_fill_mode", "get_fill_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_percentage"), "set_show_percentage", "is_percentage_shown");

	BIND_ENUM_CONSTANT(FILL_BEGIN_TO_END);
	BIND_ENUM_CONSTANT(FILL_END_TO_BEGIN);
	BIND_ENUM_CONSTANT(FILL_TOP_TO_BOTTOM);
	BIND_ENUM_CONSTANT(FILL_BOTTOM_TO_TOP);

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, ProgressBar, background_style, "background");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, ProgressBar, fill_style, "fill");

	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT, ProgressBar, font);
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT_SIZE, ProgressBar, font_size);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, ProgressBar, font_color);
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_CONSTANT, ProgressBar, font_outline_size, "outline_size");
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, ProgressBar, font_outline_color);
}

ProgressBar::ProgressBar() {
	set_v_size_flags(0);
	set_step(0.01);
}
