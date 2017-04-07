/*************************************************************************/
/*  color_ramp_edit.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "color_ramp_edit.h"
#include "os/keyboard.h"

ColorRampEdit::ColorRampEdit() {
	grabbed = -1;
	grabbing = false;
	set_focus_mode(FOCUS_ALL);

	popup = memnew(PopupPanel);
	picker = memnew(ColorPicker);
	popup->add_child(picker);

	add_child(popup);

	checker = Ref<ImageTexture>(memnew(ImageTexture));
	checker->create_from_image(Image(checker_bg_png), ImageTexture::FLAG_REPEAT);
}

int ColorRampEdit::_get_point_from_pos(int x) {
	int result = -1;
	int total_w = get_size().width - get_size().height - 3;
	for (int i = 0; i < points.size(); i++) {
		//Check if we clicked at point
		if (ABS(x - points[i].offset * total_w + 1) < (POINT_WIDTH / 2 + 1)) {
			result = i;
		}
	}
	return result;
}

void ColorRampEdit::_show_color_picker() {
	if (grabbed == -1)
		return;
	Size2 ms = Size2(350, picker->get_combined_minimum_size().height + 10);
	picker->set_pick_color(points[grabbed].color);
	popup->set_pos(get_global_pos() - Vector2(ms.width - get_size().width, ms.height));
	popup->set_size(ms);
	popup->popup();
}

ColorRampEdit::~ColorRampEdit() {
}

void ColorRampEdit::_gui_input(const InputEvent &p_event) {

	if (p_event.type == InputEvent::KEY && p_event.key.pressed && p_event.key.scancode == KEY_DELETE && grabbed != -1) {

		points.remove(grabbed);
		grabbed = -1;
		grabbing = false;
		update();
		emit_signal("ramp_changed");
		accept_event();
	}

	//Show color picker on double click.
	if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.button_index == 1 && p_event.mouse_button.doubleclick && p_event.mouse_button.pressed) {
		grabbed = _get_point_from_pos(p_event.mouse_button.x);
		_show_color_picker();
		accept_event();
	}

	//Delete point on right click
	if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.button_index == 2 && p_event.mouse_button.pressed) {
		grabbed = _get_point_from_pos(p_event.mouse_button.x);
		if (grabbed != -1) {
			points.remove(grabbed);
			grabbed = -1;
			grabbing = false;
			update();
			emit_signal("ramp_changed");
			accept_event();
		}
	}

	//Hold alt key to duplicate selected color
	if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.button_index == 1 && p_event.mouse_button.pressed && p_event.key.mod.alt) {

		int x = p_event.mouse_button.x;
		grabbed = _get_point_from_pos(x);

		if (grabbed != -1) {
			int total_w = get_size().width - get_size().height - 3;
			ColorRamp::Point newPoint = points[grabbed];
			newPoint.offset = CLAMP(x / float(total_w), 0, 1);

			points.push_back(newPoint);
			points.sort();
			for (int i = 0; i < points.size(); ++i) {
				if (points[i].offset == newPoint.offset) {
					grabbed = i;
					break;
				}
			}

			emit_signal("ramp_changed");
			update();
		}
	}

	if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.button_index == 1 && p_event.mouse_button.pressed) {

		update();
		int x = p_event.mouse_button.x;
		int total_w = get_size().width - get_size().height - 3;

		//Check if color selector was clicked.
		if (x > total_w + 3) {
			_show_color_picker();
			return;
		}

		grabbing = true;

		grabbed = _get_point_from_pos(x);
		//grab or select
		if (grabbed != -1) {
			return;
		}

		//insert
		ColorRamp::Point newPoint;
		newPoint.offset = CLAMP(x / float(total_w), 0, 1);

		ColorRamp::Point prev;
		ColorRamp::Point next;

		int pos = -1;
		for (int i = 0; i < points.size(); i++) {
			if (points[i].offset < newPoint.offset)
				pos = i;
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

		newPoint.color = prev.color.linear_interpolate(next.color, (newPoint.offset - prev.offset) / (next.offset - prev.offset));

		points.push_back(newPoint);
		points.sort();
		for (int i = 0; i < points.size(); i++) {
			if (points[i].offset == newPoint.offset) {
				grabbed = i;
				break;
			}
		}

		emit_signal("ramp_changed");
	}

	if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.button_index == 1 && !p_event.mouse_button.pressed) {

		if (grabbing) {
			grabbing = false;
			emit_signal("ramp_changed");
		}
		update();
	}

	if (p_event.type == InputEvent::MOUSE_MOTION && grabbing) {

		int total_w = get_size().width - get_size().height - 3;

		int x = p_event.mouse_motion.x;
		float newofs = CLAMP(x / float(total_w), 0, 1);

		//Snap to nearest point if holding shift
		if (p_event.key.mod.shift) {
			float snap_treshhold = 0.03;
			float smallest_ofs = snap_treshhold;
			bool founded = false;
			int nearest_point;
			for (int i = 0; i < points.size(); ++i) {
				if (i != grabbed) {
					float temp_ofs = ABS(points[i].offset - newofs);
					if (temp_ofs < smallest_ofs) {
						smallest_ofs = temp_ofs;
						nearest_point = i;
						if (founded)
							break;
						founded = true;
					}
				}
			}
			if (founded) {
				if (points[nearest_point].offset < newofs)
					newofs = points[nearest_point].offset + 0.00001;
				else
					newofs = points[nearest_point].offset - 0.00001;
				newofs = CLAMP(newofs, 0, 1);
			}
		}

		bool valid = true;
		for (int i = 0; i < points.size(); i++) {

			if (points[i].offset == newofs && i != grabbed) {
				valid = false;
			}
		}

		if (!valid)
			return;

		points[grabbed].offset = newofs;

		points.sort();
		for (int i = 0; i < points.size(); i++) {
			if (points[i].offset == newofs) {
				grabbed = i;
				break;
			}
		}

		emit_signal("ramp_changed");

		update();
	}
}

void ColorRampEdit::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {
		if (!picker->is_connected("color_changed", this, "_color_changed")) {
			picker->connect("color_changed", this, "_color_changed");
		}
	}
	if (p_what == NOTIFICATION_DRAW) {

		int w = get_size().x;
		int h = get_size().y;

		if (w == 0 || h == 0)
			return; //Safety check. We have division by 'h'. And in any case there is nothing to draw with such size

		int total_w = get_size().width - get_size().height - 3;

		//Draw checker pattern for ramp
		_draw_checker(0, 0, total_w, h);

		//Draw color ramp
		ColorRamp::Point prev;
		prev.offset = 0;
		if (points.size() == 0)
			prev.color = Color(0, 0, 0); //Draw black rectangle if we have no points
		else
			prev.color = points[0].color; //Extend color of first point to the beginning.

		for (int i = -1; i < points.size(); i++) {

			ColorRamp::Point next;
			//If there is no next point
			if (i + 1 == points.size()) {
				if (points.size() == 0)
					next.color = Color(0, 0, 0); //Draw black rectangle if we have no points
				else
					next.color = points[i].color; //Extend color of last point to the end.
				next.offset = 1;
			} else {
				next = points[i + 1];
			}

			if (prev.offset == next.offset) {
				prev = next;
				continue;
			}

			Vector<Vector2> points;
			Vector<Color> colors;
			points.push_back(Vector2(prev.offset * total_w, h));
			points.push_back(Vector2(prev.offset * total_w, 0));
			points.push_back(Vector2(next.offset * total_w, 0));
			points.push_back(Vector2(next.offset * total_w, h));
			colors.push_back(prev.color);
			colors.push_back(prev.color);
			colors.push_back(next.color);
			colors.push_back(next.color);
			draw_primitive(points, colors, Vector<Point2>());
			prev = next;
		}

		//Draw point markers
		for (int i = 0; i < points.size(); i++) {

			Color col = i == grabbed ? Color(1, 0.0, 0.0, 0.9) : points[i].color.contrasted();
			col.a = 0.9;

			draw_line(Vector2(points[i].offset * total_w, 0), Vector2(points[i].offset * total_w, h / 2), col);
			draw_rect(Rect2(points[i].offset * total_w - POINT_WIDTH / 2, h / 2, POINT_WIDTH, h / 2), Color(0.6, 0.6, 0.6, i == grabbed ? 0.9 : 0.4));
			draw_line(Vector2(points[i].offset * total_w - POINT_WIDTH / 2, h / 2), Vector2(points[i].offset * total_w - POINT_WIDTH / 2, h - 1), col);
			draw_line(Vector2(points[i].offset * total_w + POINT_WIDTH / 2, h / 2), Vector2(points[i].offset * total_w + POINT_WIDTH / 2, h - 1), col);
			draw_line(Vector2(points[i].offset * total_w - POINT_WIDTH / 2, h / 2), Vector2(points[i].offset * total_w + POINT_WIDTH / 2, h / 2), col);
			draw_line(Vector2(points[i].offset * total_w - POINT_WIDTH / 2, h - 1), Vector2(points[i].offset * total_w + POINT_WIDTH / 2, h - 1), col);
		}

		//Draw "button" for color selector
		_draw_checker(total_w + 3, 0, h, h);
		if (grabbed != -1) {
			//Draw with selection color
			draw_rect(Rect2(total_w + 3, 0, h, h), points[grabbed].color);
		} else {
			//if no color selected draw grey color with 'X' on top.
			draw_rect(Rect2(total_w + 3, 0, h, h), Color(0.5, 0.5, 0.5, 1));
			draw_line(Vector2(total_w + 3, 0), Vector2(total_w + 3 + h, h), Color(1, 1, 1, 0.6));
			draw_line(Vector2(total_w + 3, h), Vector2(total_w + 3 + h, 0), Color(1, 1, 1, 0.6));
		}

		//Draw borders around color ramp if in focus
		if (has_focus()) {

			draw_line(Vector2(-1, -1), Vector2(total_w + 1, -1), Color(1, 1, 1, 0.6));
			draw_line(Vector2(total_w + 1, -1), Vector2(total_w + 1, h + 1), Color(1, 1, 1, 0.6));
			draw_line(Vector2(total_w + 1, h + 1), Vector2(-1, h + 1), Color(1, 1, 1, 0.6));
			draw_line(Vector2(-1, -1), Vector2(-1, h + 1), Color(1, 1, 1, 0.6));
		}
	}
}

void ColorRampEdit::_draw_checker(int x, int y, int w, int h) {
	//Draw it with polygon to insert UVs for scale
	Vector<Vector2> backPoints;
	backPoints.push_back(Vector2(x, y));
	backPoints.push_back(Vector2(x, y + h));
	backPoints.push_back(Vector2(x + w, y + h));
	backPoints.push_back(Vector2(x + w, y));
	Vector<Color> colorPoints;
	colorPoints.push_back(Color(1, 1, 1, 1));
	colorPoints.push_back(Color(1, 1, 1, 1));
	colorPoints.push_back(Color(1, 1, 1, 1));
	colorPoints.push_back(Color(1, 1, 1, 1));
	Vector<Vector2> uvPoints;
	//Draw checker pattern pixel-perfect and scale it by 2.
	uvPoints.push_back(Vector2(x, y));
	uvPoints.push_back(Vector2(x, y + h * .5f / checker->get_height()));
	uvPoints.push_back(Vector2(x + w * .5f / checker->get_width(), y + h * .5f / checker->get_height()));
	uvPoints.push_back(Vector2(x + w * .5f / checker->get_width(), y));
	draw_polygon(backPoints, colorPoints, uvPoints, checker);
}

Size2 ColorRampEdit::get_minimum_size() const {

	return Vector2(0, 16);
}

void ColorRampEdit::_color_changed(const Color &p_color) {

	if (grabbed == -1)
		return;
	points[grabbed].color = p_color;
	update();
	emit_signal("ramp_changed");
}

void ColorRampEdit::set_ramp(const Vector<float> &p_offsets, const Vector<Color> &p_colors) {

	ERR_FAIL_COND(p_offsets.size() != p_colors.size());
	points.clear();
	for (int i = 0; i < p_offsets.size(); i++) {
		ColorRamp::Point p;
		p.offset = p_offsets[i];
		p.color = p_colors[i];
		points.push_back(p);
	}

	points.sort();
	update();
}

Vector<float> ColorRampEdit::get_offsets() const {
	Vector<float> ret;
	for (int i = 0; i < points.size(); i++)
		ret.push_back(points[i].offset);
	return ret;
}

Vector<Color> ColorRampEdit::get_colors() const {
	Vector<Color> ret;
	for (int i = 0; i < points.size(); i++)
		ret.push_back(points[i].color);
	return ret;
}

void ColorRampEdit::set_points(Vector<ColorRamp::Point> &p_points) {
	if (points.size() != p_points.size())
		grabbed = -1;
	points.clear();
	points = p_points;
}

Vector<ColorRamp::Point> &ColorRampEdit::get_points() {
	return points;
}

void ColorRampEdit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_gui_input"), &ColorRampEdit::_gui_input);
	ClassDB::bind_method(D_METHOD("_color_changed"), &ColorRampEdit::_color_changed);
	ADD_SIGNAL(MethodInfo("ramp_changed"));
}
