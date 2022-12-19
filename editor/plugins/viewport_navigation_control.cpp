/*************************************************************************/
/*  viewport_navigation_control.cpp                                      */
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

#include "viewport_navigation_control.h"

#include "editor/editor_settings.h"

void ViewportNavigationControl::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (!is_connected("mouse_exited", callable_mp(this, &ViewportNavigationControl::_on_mouse_exited))) {
				connect("mouse_exited", callable_mp(this, &ViewportNavigationControl::_on_mouse_exited));
			}
			if (!is_connected("mouse_entered", callable_mp(this, &ViewportNavigationControl::_on_mouse_entered))) {
				connect("mouse_entered", callable_mp(this, &ViewportNavigationControl::_on_mouse_entered));
			}
		} break;

		case NOTIFICATION_DRAW: {
			if (viewport != nullptr) {
				_draw();
				_update_navigation();
			}
		} break;
	}
}

void ViewportNavigationControl::_draw() {
	if (nav_mode == Node3DEditorViewport::NAVIGATION_NONE) {
		return;
	}

	Vector2 center = get_size() / 2.0;
	float radius = get_size().x / 2.0;

	const bool focused = focused_index != -1;
	draw_circle(center, radius, Color(0.5, 0.5, 0.5, focused || hovered ? 0.35 : 0.15));

	const Color c = focused ? Color(0.9, 0.9, 0.9, 0.9) : Color(0.5, 0.5, 0.5, 0.25);

	Vector2 circle_pos = focused ? center.move_toward(focused_pos, radius) : center;

	draw_circle(circle_pos, AXIS_CIRCLE_RADIUS, c);
	draw_circle(circle_pos, AXIS_CIRCLE_RADIUS * 0.8, c.darkened(0.4));
}

void ViewportNavigationControl::_process_click(int p_index, Vector2 p_position, bool p_pressed) {
	hovered = false;
	queue_redraw();

	if (focused_index != -1 && focused_index != p_index) {
		return;
	}
	if (p_pressed) {
		if (p_position.distance_to(get_size() / 2.0) < get_size().x / 2.0) {
			focused_pos = p_position;
			focused_index = p_index;
			queue_redraw();
		}
	} else {
		focused_index = -1;
		if (Input::get_singleton()->get_mouse_mode() == Input::MOUSE_MODE_CAPTURED) {
			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
			Input::get_singleton()->warp_mouse(focused_mouse_start);
		}
	}
}

void ViewportNavigationControl::_process_drag(int p_index, Vector2 p_position, Vector2 p_relative_position) {
	if (focused_index == p_index) {
		if (Input::get_singleton()->get_mouse_mode() == Input::MOUSE_MODE_VISIBLE) {
			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
			focused_mouse_start = p_position;
		}
		focused_pos += p_relative_position;
		queue_redraw();
	}
}

void ViewportNavigationControl::gui_input(const Ref<InputEvent> &p_event) {
	// Mouse events
	const Ref<InputEventMouseButton> mouse_button = p_event;
	if (mouse_button.is_valid() && mouse_button->get_button_index() == MouseButton::LEFT) {
		_process_click(100, mouse_button->get_position(), mouse_button->is_pressed());
	}

	const Ref<InputEventMouseMotion> mouse_motion = p_event;
	if (mouse_motion.is_valid()) {
		_process_drag(100, mouse_motion->get_global_position(), viewport->_get_warped_mouse_motion(mouse_motion));
	}

	// Touch events
	const Ref<InputEventScreenTouch> screen_touch = p_event;
	if (screen_touch.is_valid()) {
		_process_click(screen_touch->get_index(), screen_touch->get_position(), screen_touch->is_pressed());
	}

	const Ref<InputEventScreenDrag> screen_drag = p_event;
	if (screen_drag.is_valid()) {
		_process_drag(screen_drag->get_index(), screen_drag->get_position(), screen_drag->get_relative());
	}
}

void ViewportNavigationControl::_update_navigation() {
	if (focused_index == -1) {
		return;
	}

	Vector2 delta = focused_pos - (get_size() / 2.0);
	Vector2 delta_normalized = delta.normalized();
	switch (nav_mode) {
		case Node3DEditorViewport::NavigationMode::NAVIGATION_MOVE: {
			real_t speed_multiplier = MIN(delta.length() / (get_size().x * 100.0), 3.0);
			real_t speed = viewport->freelook_speed * speed_multiplier;

			const Node3DEditorViewport::FreelookNavigationScheme navigation_scheme = (Node3DEditorViewport::FreelookNavigationScheme)EditorSettings::get_singleton()->get("editors/3d/freelook/freelook_navigation_scheme").operator int();

			Vector3 forward;
			if (navigation_scheme == Node3DEditorViewport::FreelookNavigationScheme::FREELOOK_FULLY_AXIS_LOCKED) {
				// Forward/backward keys will always go straight forward/backward, never moving on the Y axis.
				forward = Vector3(0, 0, delta_normalized.y).rotated(Vector3(0, 1, 0), viewport->camera->get_rotation().y);
			} else {
				// Forward/backward keys will be relative to the camera pitch.
				forward = viewport->camera->get_transform().basis.xform(Vector3(0, 0, delta_normalized.y));
			}

			const Vector3 right = viewport->camera->get_transform().basis.xform(Vector3(delta_normalized.x, 0, 0));

			const Vector3 direction = forward + right;
			const Vector3 motion = direction * speed;
			viewport->cursor.pos += motion;
			viewport->cursor.eye_pos += motion;
		} break;

		case Node3DEditorViewport::NavigationMode::NAVIGATION_LOOK: {
			real_t speed_multiplier = MIN(delta.length() / (get_size().x * 2.5), 3.0);
			real_t speed = viewport->freelook_speed * speed_multiplier;
			viewport->_nav_look(nullptr, delta_normalized * speed);
		} break;

		case Node3DEditorViewport::NAVIGATION_PAN: {
			real_t speed_multiplier = MIN(delta.length() / (get_size().x), 3.0);
			real_t speed = viewport->freelook_speed * speed_multiplier;
			viewport->_nav_pan(nullptr, -delta_normalized * speed);
		} break;
		case Node3DEditorViewport::NAVIGATION_ZOOM: {
			real_t speed_multiplier = MIN(delta.length() / (get_size().x), 3.0);
			real_t speed = viewport->freelook_speed * speed_multiplier;
			viewport->_nav_zoom(nullptr, delta_normalized * speed);
		} break;
		case Node3DEditorViewport::NAVIGATION_ORBIT: {
			real_t speed_multiplier = MIN(delta.length() / (get_size().x), 3.0);
			real_t speed = viewport->freelook_speed * speed_multiplier;
			viewport->_nav_orbit(nullptr, delta_normalized * speed);
		} break;
		case Node3DEditorViewport::NAVIGATION_NONE: {
		} break;
	}
}

void ViewportNavigationControl::_on_mouse_entered() {
	hovered = true;
	queue_redraw();
}

void ViewportNavigationControl::_on_mouse_exited() {
	hovered = false;
	queue_redraw();
}

void ViewportNavigationControl::set_navigation_mode(Node3DEditorViewport::NavigationMode p_nav_mode) {
	nav_mode = p_nav_mode;
}

void ViewportNavigationControl::set_viewport(Node3DEditorViewport *p_viewport) {
	viewport = p_viewport;
}
