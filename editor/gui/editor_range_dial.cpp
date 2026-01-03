/**************************************************************************/
/*  editor_range_dial.cpp                                                 */
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

#include "editor_range_dial.h"
#include "editor/editor_string_names.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"

void EditorRangeDial::add_zoom_from_mouse_dist(double p_relative) {
	if (inverted_zoom_y) {
		p_relative = -p_relative;
	}
	double zoom_factor = 1.0 + (p_relative * zoom_speed * 0.001);
	set_zoom(zoom / zoom_factor);
}

double EditorRangeDial::scale_diff(double p_diff) {
	return p_diff / zoom * 0.1;
}

void EditorRangeDial::set_highlighting_range(bool p_enable) {
	highlighting_range = p_enable;
	queue_redraw();
}

bool EditorRangeDial::is_highlighting_range() const {
	return highlighting_range;
}

void EditorRangeDial::set_value_no_step(double p_value, bool p_rounded) {
	if (!is_greater_allowed() && p_value > get_max() - get_page()) {
		p_value = get_max() - get_page();
	}
	if (!is_lesser_allowed() && p_value < get_min()) {
		p_value = get_min();
	}

	value_no_snap = p_value;

	if (p_rounded) {
		set_value(Math::round(value_no_snap));
	} else {
		set_value(value_no_snap);
	}
	queue_redraw();
}

void EditorRangeDial::set_zoom_from_value(double p_value) {
	if (Math::is_zero_approx(p_value)) {
		set_zoom(get_rect().size.x / (zoom_speed * 2) / 2);
		return;
	}

	if (p_value < 1.0) {
		int precision = is_using_rounded_values() ? 0 : MAX(0, -floor(log10(p_value)));
		set_zoom(Math::pow(10, (double)precision + 2));
	} else {
		set_zoom(get_rect().size.x / (MAX(1.0, Math::abs(p_value)) * 2) / 2);
	}
}

void EditorRangeDial::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			const StringName &sn_label = SNAME("Label");
			Ref<Font> font = get_theme_font(SNAME("rulers"), EditorStringName(EditorFonts));
			Color font_color = get_theme_color(SNAME("font_color"), sn_label);
			Color accent_color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
			Color highlight_color = get_theme_color(SNAME("highlight_color"), EditorStringName(Editor));
			highlight_color.a = 0.05;

			const int font_size = get_theme_font_size(SceneStringName(font_size), sn_label);
			const int precision = is_using_rounded_values() ? 0 : MAX(0, -floor(log10(get_step())));
			const String max_value_str(String::num(get_max(), precision));
			const String min_value_str(String::num(get_min(), precision));

			const double middle = get_rect().size.x / 2 - (value_no_snap * zoom);
			const double number_x_min = middle + get_min() * zoom;
			const double number_x_max = middle + get_max() * zoom;

			Color mono_color, mono_color_outside, mono_color_highlight, mono_color_highlight_outside;
			mono_color = mono_color_outside = mono_color_highlight = mono_color_highlight_outside = get_theme_color(SNAME("mono_color"), EditorStringName(Editor));
			mono_color_highlight.a = 0.75;
			mono_color_highlight_outside.a = 0.25;

			bool internal_highlighting_range = highlighting_range;

			if (get_min() == get_max()) {
				internal_highlighting_range = false;
			}

			// only draw the background if either not allowing lesser or greater values
			if (internal_highlighting_range) {
				const double min_x = MIN(MAX(0, number_x_min), get_rect().size.x);
				const double max_x = MAX(MIN(get_rect().size.x, number_x_max), 0);
				draw_rect(Rect2(max_x, 0, min_x - max_x, get_rect().size.y), highlight_color);
			}

			int min_exp = is_using_rounded_values() ? 0 : -MAX(0, -floor(log10(get_step())));
			int max_exp;

			double mx = EDITOR_GET("interface/inspector/max_drag_zoom_out_value");
			double zout = Math::abs(get_min()) + get_max();

			max_exp = int(log10(MIN(zout, mx * 2)));

			for (int i = min_exp; i <= max_exp; i++) {
				const double level = Math::pow(10.0, i);
				const double size = level * zoom;
				if (size < Math::pow(10.0, 1) || size > Math::pow(10.0, 2)) {
					continue;
				}

				const double powsize = Math::pow(size / 100.0, 3.0);
				const double alpha = powsize / (powsize + Math::pow((1 - size / 100), 3));
				const double span = get_rect().size.x / size;

				PackedVector2Array lines, lines_highlight, lines_outside, lines_highlight_outside;

				for (int n = -span; n < span; n++) {
					const double number_x = get_rect().size.x / 2 - Math::fmod(value_no_snap * zoom, size) + n * size;
					double n_value = Math::snapped((number_x - middle) / zoom, 1 / (Math::pow(10, (double)precision + 1)));
					if (Math::abs(n_value) < CMP_EPSILON2) {
						n_value = 0.0;
					}
					String current_value(String::num(n_value, precision));
					Size2 string_size = font->get_string_size(current_value, HORIZONTAL_ALIGNMENT_CENTER, -1.0f, font_size);
					Color number_font_color = font_color;
					if (number_x - string_size.x / 2 > get_rect().size.x || number_x + string_size.x / 2 < 0) {
						continue;
					}

					if (internal_highlighting_range) {
						// fade the text color if it is near the min or max text (number_x_min or number_x_max) in distance 64
						double distance_to_min = Math::abs(number_x - number_x_min);
						double distance_to_max = Math::abs(number_x - number_x_max);
						if (distance_to_min < 64.0 || distance_to_max < 64.0) {
							if (distance_to_min < distance_to_max) {
								number_font_color.a *= distance_to_min / 64.0;
							} else {
								number_font_color.a *= distance_to_max / 64.0;
							}
						}
					}

					bool outside = false;
					if (internal_highlighting_range) {
						if ((!is_lesser_allowed() && (n_value <= get_min() + CMP_EPSILON2)) || (!is_greater_allowed() && (n_value >= get_max() - CMP_EPSILON2))) {
							continue;
						} else {
							if (Math::abs(n_value - get_min()) < CMP_EPSILON2 || Math::abs(n_value - get_max()) < CMP_EPSILON2) {
								continue;
							}
							if ((n_value <= get_min() + CMP_EPSILON2) || (n_value >= get_max() - CMP_EPSILON2)) {
								number_font_color.a = 0.25;
								outside = true;
							}
						}
					}

					Vector2 line_start(number_x, 0);
					Vector2 line_end(number_x, get_rect().size.y - string_size.y);

					if (Math::fmod(Math::abs(n_value), level * 10) < CMP_EPSILON2 || Math::fmod(Math::abs(n_value), level * 10) >= (level * 10) - CMP_EPSILON2) {
						if (outside) {
							lines_highlight_outside.push_back(line_start);
							lines_highlight_outside.push_back(line_end);
						} else {
							lines_highlight.push_back(line_start);
							lines_highlight.push_back(line_end);
						}

						draw_string(font, Point2(number_x - (string_size.x / 2), get_rect().size.y).floor(), current_value, HORIZONTAL_ALIGNMENT_CENTER, -1, font_size, number_font_color);
					} else {
						if (outside) {
							lines_outside.push_back(line_start);
							lines_outside.push_back(line_end);
						} else {
							lines.push_back(line_start);
							lines.push_back(line_end);
						}
						number_font_color.a *= alpha;
						draw_string(font, Point2(number_x - (string_size.x / 2), get_rect().size.y).floor(), current_value, HORIZONTAL_ALIGNMENT_CENTER, -1, font_size, number_font_color);
					}
				}

				if (internal_highlighting_range) {
					const Size2 max_value_string_size = font->get_string_size(max_value_str, HORIZONTAL_ALIGNMENT_CENTER, -1.0f, font_size);
					const Size2 min_value_string_size = font->get_string_size(min_value_str, HORIZONTAL_ALIGNMENT_CENTER, -1.0f, font_size);

					if (number_x_max - max_value_string_size.x / 2 < get_rect().size.x && number_x_max + max_value_string_size.x / 2 > 0) {
						draw_line(Point2(number_x_max, 0), Point2(number_x_max, get_rect().size.y - max_value_string_size.y), accent_color);
						draw_string(font, Point2(number_x_max - (max_value_string_size.x / 2), get_rect().size.y).floor(), max_value_str, HORIZONTAL_ALIGNMENT_CENTER, -1, font_size, accent_color);
					}
					if (number_x_min + min_value_string_size.x > 0 && number_x_min - min_value_string_size.x < get_rect().size.x) {
						draw_line(Point2(number_x_min, 0), Point2(number_x_min, get_rect().size.y - max_value_string_size.y), accent_color);
						draw_string(font, Point2(number_x_min - (min_value_string_size.x / 2), get_rect().size.y).floor(), min_value_str, HORIZONTAL_ALIGNMENT_CENTER, -1, font_size, accent_color);
					}
				}

				if (!lines.is_empty()) {
					mono_color.a = 0.5 * alpha;
					draw_multiline(lines, mono_color);
				}
				if (!lines_outside.is_empty()) {
					mono_color_outside.a = 0.25 * alpha;
					draw_multiline(lines_outside, mono_color_outside);
				}
				if (!lines_highlight.is_empty()) {
					draw_multiline(lines_highlight, mono_color_highlight);
				}
				if (!lines_highlight_outside.is_empty()) {
					draw_multiline(lines_highlight_outside, mono_color_highlight_outside);
				}
				if (!lines_highlight_outside.is_empty()) {
					draw_multiline(lines_highlight_outside, mono_color_highlight_outside);
				}
			}
			draw_line(Vector2(get_rect().size.x / 2, 0), Vector2(get_rect().size.x / 2, get_rect().size.y), Color(0.96, 0.2, 0.32));
		}
	}
}

void EditorRangeDial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_zoom"), &EditorRangeDial::get_zoom);
	ClassDB::bind_method(D_METHOD("set_zoom", "amount"), &EditorRangeDial::set_zoom);

	ClassDB::bind_method(D_METHOD("add_zoom_from_mouse_dist", "relative"), &EditorRangeDial::add_zoom_from_mouse_dist);
	ClassDB::bind_method(D_METHOD("scale_diff", "diff"), &EditorRangeDial::scale_diff);

	ClassDB::bind_method(D_METHOD("set_highlighting_range", "enable"), &EditorRangeDial::set_highlighting_range);
	ClassDB::bind_method(D_METHOD("is_highlighting_range"), &EditorRangeDial::is_highlighting_range);

	ADD_SIGNAL(MethodInfo("zoom_changed"));
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "zoom"), "set_zoom", "get_zoom");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "highlighting_range", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR), "set_highlighting_range", "is_highlighting_range");
}

void EditorRangeDial::set_zoom(double p_zoom) {
	double min_zoom;

	double zout = Math::abs(get_min()) + get_max();
	double mx = EDITOR_GET("interface/inspector/max_drag_zoom_out_value");
	min_zoom = get_rect().size.x / MIN(zout, mx * 2) / 2;

	int precision = is_using_rounded_values() ? 0 : MAX(0, -floor(log10(get_step())));
	double max_zoom = Math::pow(10, (double)precision + 2);
	if (min_zoom > max_zoom) {
		zoom = max_zoom;
	} else {
		zoom = CLAMP(p_zoom, min_zoom, max_zoom);
	}
	emit_signal("zoom_changed");
	queue_redraw();
}

double EditorRangeDial::get_zoom() const {
	return zoom;
}

EditorRangeDial::EditorRangeDial() {
	zoom = 1;
	value_no_snap = 0.0;

	zoom_speed = EDITOR_GET("interface/inspector/zoom_drag_speed");
	inverted_zoom_y = EDITOR_GET("interface/inspector/invert_drag_zoom_y");
}

void EditorRangeDialPopup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_dial"), &EditorRangeDialPopup::get_dial);
}

EditorRangeDial *EditorRangeDialPopup::get_dial() const {
	return dial;
}

void EditorRangeDialPopup::_pre_popup() {
	set_min_size(Vector2(350, 48) * EDSCALE);
	reset_size();
	set_flag(Window::FLAG_NO_FOCUS, true);
	set_flag(Window::FLAG_POPUP, false);
	set_flag(Window::FLAG_MOUSE_PASSTHROUGH, true);
	set_wrap_controls(true);
}

EditorRangeDialPopup::EditorRangeDialPopup() {
	dial = memnew(EditorRangeDial);
	dial->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	add_child(dial);
}
