/*************************************************************************/
/*  gradient_edit.cpp                                                    */
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

#include "gradient_edit.h"

#include "core/os/keyboard.h"

GradientEdit::GradientEdit() {
	set_focus_mode(FOCUS_ALL);

	popup = memnew(PopupPanel);
	picker = memnew(ColorPicker);
	popup->add_child(picker);

	gradient_cache.instantiate();
	preview_texture.instantiate();

	preview_texture->set_width(1024);
	add_child(popup, false, INTERNAL_MODE_FRONT);
}

int GradientEdit::_get_point_from_pos(int x) {
	int result = -1;
	int total_w = get_size().width - get_size().height - draw_spacing;
	float min_distance = 1e20;
	for (int i = 0; i < points.size(); i++) {
		// Check if we clicked at point.
		float distance = ABS(x - points[i].offset * total_w);
		float min = (draw_point_width / 2 * 1.7); //make it easier to grab
		if (distance <= min && distance < min_distance) {
			result = i;
			min_distance = distance;
		}
	}
	return result;
}

void GradientEdit::_show_color_picker() {
	if (grabbed == -1) {
		return;
	}
	picker->set_pick_color(points[grabbed].color);
	Size2 minsize = popup->get_contents_minimum_size();
	bool show_above = false;
	if (get_global_position().y + get_size().y + minsize.y > get_viewport_rect().size.y) {
		show_above = true;
	}
	if (show_above) {
		popup->set_position(get_screen_position() - Vector2(0, minsize.y));
	} else {
		popup->set_position(get_screen_position() + Vector2(0, get_size().y));
	}
	popup->popup();
}

GradientEdit::~GradientEdit() {
}

void GradientEdit::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventKey> k = p_event;

	if (k.is_valid() && k->is_pressed() && k->get_keycode() == KEY_DELETE && grabbed != -1) {
		points.remove(grabbed);
		grabbed = -1;
		grabbing = false;
		update();
		emit_signal(SNAME("ramp_changed"));
		accept_event();
	}

	Ref<InputEventMouseButton> mb = p_event;
	// Show color picker on double click.
	if (mb.is_valid() && mb->get_button_index() == 1 && mb->is_double_click() && mb->is_pressed()) {
		grabbed = _get_point_from_pos(mb->get_position().x);
		_show_color_picker();
		accept_event();
	}

	// Delete point on right click.
	if (mb.is_valid() && mb->get_button_index() == 2 && mb->is_pressed()) {
		grabbed = _get_point_from_pos(mb->get_position().x);
		if (grabbed != -1) {
			points.remove(grabbed);
			grabbed = -1;
			grabbing = false;
			update();
			emit_signal(SNAME("ramp_changed"));
			accept_event();
		}
	}

	// Hold alt key to duplicate selected color.
	if (mb.is_valid() && mb->get_button_index() == 1 && mb->is_pressed() && mb->is_alt_pressed()) {
		int x = mb->get_position().x;
		grabbed = _get_point_from_pos(x);

		if (grabbed != -1) {
			int total_w = get_size().width - get_size().height - draw_spacing;
			Gradient::Point new_point = points[grabbed];
			new_point.offset = CLAMP(x / float(total_w), 0, 1);

			points.push_back(new_point);
			points.sort();
			for (int i = 0; i < points.size(); ++i) {
				if (points[i].offset == new_point.offset) {
					grabbed = i;
					break;
				}
			}

			emit_signal(SNAME("ramp_changed"));
			update();
		}
	}

	// Select.
	if (mb.is_valid() && mb->get_button_index() == 1 && mb->is_pressed()) {
		update();
		int x = mb->get_position().x;
		int total_w = get_size().width - get_size().height - draw_spacing;

		//Check if color selector was clicked.
		if (x > total_w + draw_spacing) {
			_show_color_picker();
			return;
		}

		grabbing = true;

		grabbed = _get_point_from_pos(x);
		//grab or select
		if (grabbed != -1) {
			return;
		}

		// Insert point.
		Gradient::Point new_point;
		new_point.offset = CLAMP(x / float(total_w), 0, 1);

		Gradient::Point prev;
		Gradient::Point next;

		int pos = -1;
		for (int i = 0; i < points.size(); i++) {
			if (points[i].offset < new_point.offset) {
				pos = i;
			}
		}

		if (pos == -1) {
			prev.color = Color(0, 0, 0);
			prev.offset = 0;
			if (points.size()) {
				next = points[0];
			} else {
				next.color = Color(1, 1, 1);
				next.offset = 1.0;
			}
		} else {
			if (pos == points.size() - 1) {
				next.color = Color(1, 1, 1);
				next.offset = 1.0;
			} else {
				next = points[pos + 1];
			}
			prev = points[pos];
		}

		new_point.color = prev.color.lerp(next.color, (new_point.offset - prev.offset) / (next.offset - prev.offset));

		points.push_back(new_point);
		points.sort();
		for (int i = 0; i < points.size(); i++) {
			if (points[i].offset == new_point.offset) {
				grabbed = i;
				break;
			}
		}

		emit_signal(SNAME("ramp_changed"));
	}

	if (mb.is_valid() && mb->get_button_index() == 1 && !mb->is_pressed()) {
		if (grabbing) {
			grabbing = false;
			emit_signal(SNAME("ramp_changed"));
		}
		update();
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid() && grabbing) {
		int total_w = get_size().width - get_size().height - draw_spacing;

		int x = mm->get_position().x;

		float newofs = CLAMP(x / float(total_w), 0, 1);

		// Snap to "round" coordinates if holding Ctrl.
		// Be more precise if holding Shift as well.
		if (mm->is_ctrl_pressed()) {
			newofs = Math::snapped(newofs, mm->is_shift_pressed() ? 0.025 : 0.1);
		} else if (mm->is_shift_pressed()) {
			// Snap to nearest point if holding just Shift
			const float snap_threshold = 0.03;
			float smallest_ofs = snap_threshold;
			bool found = false;
			int nearest_point = 0;
			for (int i = 0; i < points.size(); ++i) {
				if (i != grabbed) {
					float temp_ofs = ABS(points[i].offset - newofs);
					if (temp_ofs < smallest_ofs) {
						smallest_ofs = temp_ofs;
						nearest_point = i;
						if (found) {
							break;
						}
						found = true;
					}
				}
			}
			if (found) {
				if (points[nearest_point].offset < newofs) {
					newofs = points[nearest_point].offset + 0.00001;
				} else {
					newofs = points[nearest_point].offset - 0.00001;
				}
				newofs = CLAMP(newofs, 0, 1);
			}
		}

		bool valid = true;
		for (int i = 0; i < points.size(); i++) {
			if (points[i].offset == newofs && i != grabbed) {
				valid = false;
				break;
			}
		}

		if (!valid || grabbed == -1) {
			return;
		}
		points.write[grabbed].offset = newofs;

		points.sort();
		for (int i = 0; i < points.size(); i++) {
			if (points[i].offset == newofs) {
				grabbed = i;
				break;
			}
		}

		emit_signal(SNAME("ramp_changed"));

		update();
	}
}

void GradientEdit::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		if (!picker->is_connected("color_changed", callable_mp(this, &GradientEdit::_color_changed))) {
			picker->connect("color_changed", callable_mp(this, &GradientEdit::_color_changed));
		}
	}

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		draw_spacing = BASE_SPACING * get_theme_default_base_scale();
		draw_point_width = BASE_POINT_WIDTH * get_theme_default_base_scale();
	}

	if (p_what == NOTIFICATION_DRAW) {
		int w = get_size().x;
		int h = get_size().y;

		if (w == 0 || h == 0) {
			return; // Safety check. We have division by 'h'. And in any case there is nothing to draw with such size.
		}

		int total_w = get_size().width - get_size().height - draw_spacing;

		// Draw checker pattern for ramp.
		draw_texture_rect(get_theme_icon(SNAME("GuiMiniCheckerboard"), SNAME("EditorIcons")), Rect2(0, 0, total_w, h), true);

		// Draw color ramp.

		gradient_cache->set_points(points);
		gradient_cache->set_interpolation_mode(interpolation_mode);
		preview_texture->set_gradient(gradient_cache);
		draw_texture_rect(preview_texture, Rect2(0, 0, total_w, h));

		// Draw point markers.
		for (int i = 0; i < points.size(); i++) {
			Color col = points[i].color.inverted();
			col.a = 0.9;

			draw_line(Vector2(points[i].offset * total_w, 0), Vector2(points[i].offset * total_w, h / 2), col);
			Rect2 rect = Rect2(points[i].offset * total_w - draw_point_width / 2, h / 2, draw_point_width, h / 2);
			draw_rect(rect, points[i].color, true);
			draw_rect(rect, col, false);
			if (grabbed == i) {
				rect = rect.grow(-1);
				if (has_focus()) {
					draw_rect(rect, Color(1, 0, 0, 0.9), false);
				} else {
					draw_rect(rect, Color(0.6, 0, 0, 0.9), false);
				}

				rect = rect.grow(-1);
				draw_rect(rect, col, false);
			}
		}

		//Draw "button" for color selector
		draw_texture_rect(get_theme_icon(SNAME("GuiMiniCheckerboard"), SNAME("EditorIcons")), Rect2(total_w + draw_spacing, 0, h, h), true);
		if (grabbed != -1) {
			//Draw with selection color
			draw_rect(Rect2(total_w + draw_spacing, 0, h, h), points[grabbed].color);
		} else {
			//if no color selected draw grey color with 'X' on top.
			draw_rect(Rect2(total_w + draw_spacing, 0, h, h), Color(0.5, 0.5, 0.5, 1));
			draw_line(Vector2(total_w + draw_spacing, 0), Vector2(total_w + draw_spacing + h, h), Color(1, 1, 1, 0.6));
			draw_line(Vector2(total_w + draw_spacing, h), Vector2(total_w + draw_spacing + h, 0), Color(1, 1, 1, 0.6));
		}

		// Draw borders around color ramp if in focus.
		if (has_focus()) {
			draw_line(Vector2(-1, -1), Vector2(total_w + 1, -1), Color(1, 1, 1, 0.6));
			draw_line(Vector2(total_w + 1, -1), Vector2(total_w + 1, h + 1), Color(1, 1, 1, 0.6));
			draw_line(Vector2(total_w + 1, h + 1), Vector2(-1, h + 1), Color(1, 1, 1, 0.6));
			draw_line(Vector2(-1, -1), Vector2(-1, h + 1), Color(1, 1, 1, 0.6));
		}
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (!is_visible()) {
			grabbing = false;
		}
	}
}

Size2 GradientEdit::get_minimum_size() const {
	return Vector2(0, 16);
}

void GradientEdit::_color_changed(const Color &p_color) {
	if (grabbed == -1) {
		return;
	}
	points.write[grabbed].color = p_color;
	update();
	emit_signal(SNAME("ramp_changed"));
}

void GradientEdit::set_ramp(const Vector<real_t> &p_offsets, const Vector<Color> &p_colors) {
	ERR_FAIL_COND(p_offsets.size() != p_colors.size());
	points.clear();
	for (int i = 0; i < p_offsets.size(); i++) {
		Gradient::Point p;
		p.offset = p_offsets[i];
		p.color = p_colors[i];
		points.push_back(p);
	}

	points.sort();
	update();
}

Vector<real_t> GradientEdit::get_offsets() const {
	Vector<real_t> ret;
	for (int i = 0; i < points.size(); i++) {
		ret.push_back(points[i].offset);
	}
	return ret;
}

Vector<Color> GradientEdit::get_colors() const {
	Vector<Color> ret;
	for (int i = 0; i < points.size(); i++) {
		ret.push_back(points[i].color);
	}
	return ret;
}

void GradientEdit::set_points(Vector<Gradient::Point> &p_points) {
	if (points.size() != p_points.size()) {
		grabbed = -1;
	}
	points.clear();
	points = p_points;
	points.sort();
}

Vector<Gradient::Point> &GradientEdit::get_points() {
	return points;
}

void GradientEdit::set_interpolation_mode(Gradient::InterpolationMode p_interp_mode) {
	interpolation_mode = p_interp_mode;
}

Gradient::InterpolationMode GradientEdit::get_interpolation_mode() {
	return interpolation_mode;
}

void GradientEdit::_bind_methods() {
	ADD_SIGNAL(MethodInfo("ramp_changed"));
}
