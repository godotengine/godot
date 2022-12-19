/*************************************************************************/
/*  viewport_rotation_control.cpp                                        */
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

#include "viewport_rotation_control.h"

void ViewportRotationControl::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			axis_menu_options.clear();
			axis_menu_options.push_back(Node3DEditorViewport::VIEW_RIGHT);
			axis_menu_options.push_back(Node3DEditorViewport::VIEW_TOP);
			axis_menu_options.push_back(Node3DEditorViewport::VIEW_REAR);
			axis_menu_options.push_back(Node3DEditorViewport::VIEW_LEFT);
			axis_menu_options.push_back(Node3DEditorViewport::VIEW_BOTTOM);
			axis_menu_options.push_back(Node3DEditorViewport::VIEW_FRONT);

			axis_colors.clear();
			axis_colors.push_back(get_theme_color(SNAME("axis_x_color"), SNAME("Editor")));
			axis_colors.push_back(get_theme_color(SNAME("axis_y_color"), SNAME("Editor")));
			axis_colors.push_back(get_theme_color(SNAME("axis_z_color"), SNAME("Editor")));
			queue_redraw();

			if (!is_connected("mouse_exited", callable_mp(this, &ViewportRotationControl::_on_mouse_exited))) {
				connect("mouse_exited", callable_mp(this, &ViewportRotationControl::_on_mouse_exited));
			}
		} break;

		case NOTIFICATION_DRAW: {
			if (viewport != nullptr) {
				_draw();
			}
		} break;
	}
}

void ViewportRotationControl::_draw() {
	const Vector2i center = get_size() / 2.0;
	const real_t radius = get_size().x / 2.0;

	if (focused_axis > -2 || orbiting_index != -1) {
		draw_circle(center, radius, Color(0.5, 0.5, 0.5, 0.25));
	}

	Vector<Axis2D> axis_to_draw;
	_get_sorted_axis(axis_to_draw);
	for (int i = 0; i < axis_to_draw.size(); ++i) {
		_draw_axis(axis_to_draw[i]);
	}
}

void ViewportRotationControl::_draw_axis(const Axis2D &p_axis) {
	const bool focused = focused_axis == p_axis.axis;
	const bool positive = p_axis.axis < 3;
	const int direction = p_axis.axis % 3;

	const Color axis_color = axis_colors[direction];
	const double alpha = focused ? 1.0 : ((p_axis.z_axis + 1.0) / 2.0) * 0.5 + 0.5;
	const Color c = focused ? Color(0.9, 0.9, 0.9) : Color(axis_color, alpha);

	if (positive) {
		// Draw axis lines for the positive axes.
		const Vector2i center = get_size() / 2.0;
		draw_line(center, p_axis.screen_point, c, 1.5 * EDSCALE);

		draw_circle(p_axis.screen_point, AXIS_CIRCLE_RADIUS, c);

		// Draw the axis letter for the positive axes.
		const String axis_name = direction == 0 ? "X" : (direction == 1 ? "Y" : "Z");
		draw_char(get_theme_font(SNAME("rotation_control"), SNAME("EditorFonts")), p_axis.screen_point + Vector2i(Math::round(-4.0 * EDSCALE), Math::round(5.0 * EDSCALE)), axis_name, get_theme_font_size(SNAME("rotation_control_size"), SNAME("EditorFonts")), Color(0.0, 0.0, 0.0, alpha));
	} else {
		// Draw an outline around the negative axes.
		draw_circle(p_axis.screen_point, AXIS_CIRCLE_RADIUS, c);
		draw_circle(p_axis.screen_point, AXIS_CIRCLE_RADIUS * 0.8, c.darkened(0.4));
	}
}

void ViewportRotationControl::_get_sorted_axis(Vector<Axis2D> &r_axis) {
	const Vector2i center = get_size() / 2.0;
	const real_t radius = get_size().x / 2.0 - AXIS_CIRCLE_RADIUS - 2.0 * EDSCALE;
	const Basis camera_basis = viewport->to_camera_transform(viewport->cursor).get_basis().inverse();

	for (int i = 0; i < 3; ++i) {
		Vector3 axis_3d = camera_basis.get_column(i);
		Vector2i axis_vector = Vector2(axis_3d.x, -axis_3d.y) * radius;

		if (Math::abs(axis_3d.z) < 1.0) {
			Axis2D pos_axis;
			pos_axis.axis = i;
			pos_axis.screen_point = center + axis_vector;
			pos_axis.z_axis = axis_3d.z;
			r_axis.push_back(pos_axis);

			Axis2D neg_axis;
			neg_axis.axis = i + 3;
			neg_axis.screen_point = center - axis_vector;
			neg_axis.z_axis = -axis_3d.z;
			r_axis.push_back(neg_axis);
		} else {
			// Special case when the camera is aligned with one axis
			Axis2D axis;
			axis.axis = i + (axis_3d.z < 0 ? 0 : 3);
			axis.screen_point = center;
			axis.z_axis = 1.0;
			r_axis.push_back(axis);
		}
	}

	r_axis.sort_custom<Axis2DCompare>();
}

void ViewportRotationControl::_process_click(int p_index, Vector2 p_position, bool p_pressed) {
	if (orbiting_index != -1 && orbiting_index != p_index) {
		return;
	}
	if (p_pressed) {
		if (p_position.distance_to(get_size() / 2.0) < get_size().x / 2.0) {
			orbiting_index = p_index;
		}
	} else {
		if (focused_axis > -1) {
			viewport->_menu_option(axis_menu_options[focused_axis]);
			_update_focus();
		}
		orbiting_index = -1;
		if (Input::get_singleton()->get_mouse_mode() == Input::MOUSE_MODE_CAPTURED) {
			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
			Input::get_singleton()->warp_mouse(orbiting_mouse_start);
		}
	}
}

void ViewportRotationControl::_process_drag(Ref<InputEventWithModifiers> p_event, int p_index, Vector2 p_position, Vector2 p_relative_position) {
	if (orbiting_index == p_index) {
		if (Input::get_singleton()->get_mouse_mode() == Input::MOUSE_MODE_VISIBLE) {
			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
			orbiting_mouse_start = p_position;
		}
		viewport->_nav_orbit(p_event, p_relative_position);
		focused_axis = -1;
	} else {
		_update_focus();
	}
}

void ViewportRotationControl::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	// Mouse events
	const Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT) {
		_process_click(100, mb->get_position(), mb->is_pressed());
	}

	const Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		_process_drag(mm, 100, mm->get_global_position(), viewport->_get_warped_mouse_motion(mm));
	}

	// Touch events
	const Ref<InputEventScreenTouch> screen_touch = p_event;
	if (screen_touch.is_valid()) {
		_process_click(screen_touch->get_index(), screen_touch->get_position(), screen_touch->is_pressed());
	}

	const Ref<InputEventScreenDrag> screen_drag = p_event;
	if (screen_drag.is_valid()) {
		_process_drag(screen_drag, screen_drag->get_index(), screen_drag->get_position(), screen_drag->get_relative());
	}
}

void ViewportRotationControl::_update_focus() {
	int original_focus = focused_axis;
	focused_axis = -2;
	Vector2 mouse_pos = get_local_mouse_position();

	if (mouse_pos.distance_to(get_size() / 2.0) < get_size().x / 2.0) {
		focused_axis = -1;
	}

	Vector<Axis2D> axes;
	_get_sorted_axis(axes);

	for (int i = 0; i < axes.size(); i++) {
		const Axis2D &axis = axes[i];
		if (mouse_pos.distance_to(axis.screen_point) < AXIS_CIRCLE_RADIUS) {
			focused_axis = axis.axis;
		}
	}

	if (focused_axis != original_focus) {
		queue_redraw();
	}
}

void ViewportRotationControl::_on_mouse_exited() {
	focused_axis = -2;
	queue_redraw();
}

void ViewportRotationControl::set_viewport(Node3DEditorViewport *p_viewport) {
	viewport = p_viewport;
}
