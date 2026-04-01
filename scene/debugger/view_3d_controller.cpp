/**************************************************************************/
/*  view_3d_controller.cpp                                                */
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

#ifndef _3D_DISABLED

#include "view_3d_controller.h"

#include "core/config/engine.h"
#include "core/input/input.h"
#include "core/input/shortcut.h"
#include "core/object/class_db.h" // IWYU pragma: keep. `ADD_SIGNAL` macro.
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

using namespace View3DControllerConsts;

Transform3D View3DController::_to_camera_transform(const Cursor &p_cursor) const {
	Transform3D camera_transform;
	camera_transform.translate_local(p_cursor.pos);
	camera_transform.basis.rotate(Vector3(1, 0, 0), -p_cursor.x_rot);
	camera_transform.basis.rotate(Vector3(0, 1, 0), -p_cursor.y_rot);

	if (orthogonal) {
		camera_transform.translate_local(0, 0, (zfar - znear) / 2.0);
	} else {
		camera_transform.translate_local(0, 0, p_cursor.distance);
	}

	return camera_transform;
}

bool View3DController::_is_shortcut_pressed(const ShortcutName p_name, const bool p_true_if_null) {
	Ref<Shortcut> shortcut = inputs[p_name];
	if (shortcut.is_null()) {
		return p_true_if_null;
	}

	const Array shortcuts = shortcut->get_events();
	Ref<InputEventKey> k;
	if (shortcuts.size() > 0) {
		k = shortcuts.front();
	}

	if (k.is_null()) {
		return p_true_if_null;
	}

#define EMULATE_NUMPAD_KEY(p_code) \
	(emulate_numpad && p_code >= Key::KEY_0 && p_code <= Key::KEY_9 ? p_code - Key::KEY_0 + Key::KP_0 : p_code)

	if (k->get_physical_keycode() == Key::NONE) {
		return Input::get_singleton()->is_key_pressed(EMULATE_NUMPAD_KEY(k->get_keycode()));
	}

	return Input::get_singleton()->is_physical_key_pressed(EMULATE_NUMPAD_KEY(k->get_physical_keycode()));

#undef EMULATE_NUMPAD_KEY
}

bool View3DController::_is_shortcut_empty(const ShortcutName p_name) {
	Ref<Shortcut> shortcut = inputs[p_name];
	if (shortcut.is_null()) {
		return true;
	}

	const Array shortcuts = shortcut->get_events();
	Ref<InputEventKey> k;
	if (shortcuts.size() > 0) {
		k = shortcuts.front();
	}

	return k.is_null();
}

View3DController::NavigationMode View3DController::_get_nav_mode_from_shortcuts(NavigationMouseButton p_mouse_button, const Vector<ShortcutCheck> &p_shortcut_checks, bool p_not_empty) {
	if (p_not_empty) {
		for (const ShortcutCheck &shortcut_check : p_shortcut_checks) {
			if (shortcut_check.mod_pressed && shortcut_check.not_empty) {
				return shortcut_check.result_nav_mode;
			}
		}
	} else {
		for (const ShortcutCheck &shortcut_check : p_shortcut_checks) {
			if (shortcut_check.mouse_preference == p_mouse_button && shortcut_check.mod_pressed) {
				return shortcut_check.result_nav_mode;
			}
		}
	}

	return NAV_MODE_NONE;
}

bool View3DController::gui_input(const Ref<InputEvent> &p_event, const Rect2 &p_surface_rect) {
	Ref<InputEventMouseButton> b = p_event;
	if (b.is_valid()) {
		if (b->get_button_index() == MouseButton::RIGHT && b->is_pressed() && navigating) {
			cancel_navigation();
			return true;
		}

		const real_t zoom_factor = 1 + (ZOOM_FREELOOK_MULTIPLIER - 1) * b->get_factor();
		switch (b->get_button_index()) {
			case MouseButton::WHEEL_UP: {
				if (freelook) {
					scale_freelook_speed(zoom_factor);
				} else {
					scale_cursor_distance(1.0 / zoom_factor);
				}
			} break;
			case MouseButton::WHEEL_DOWN: {
				if (freelook) {
					scale_freelook_speed(1.0 / zoom_factor);
				} else {
					scale_cursor_distance(zoom_factor);
				}
			} break;
			default: {
				return false;
			}
		}

		return true;
	}

	Vector<ShortcutCheck> shortcut_checks;

	if (Input::get_singleton()->get_mouse_mode() != Input::MouseMode::MOUSE_MODE_CAPTURED) {
#define GET_SHORTCUT_COUNT(p_name) (inputs[p_name].is_null() ? 0 : inputs[p_name]->get_events().size())

		bool orbit_mod_pressed = _is_shortcut_pressed(SHORTCUT_ORBIT_MOD_1, true) && _is_shortcut_pressed(SHORTCUT_ORBIT_MOD_2, true);
		bool pan_mod_pressed = _is_shortcut_pressed(SHORTCUT_PAN_MOD_1, true) && _is_shortcut_pressed(SHORTCUT_PAN_MOD_2, true);
		bool zoom_mod_pressed = _is_shortcut_pressed(SHORTCUT_ZOOM_MOD_1, true) && _is_shortcut_pressed(SHORTCUT_ZOOM_MOD_2, true);
		int orbit_mod_input_count = GET_SHORTCUT_COUNT(SHORTCUT_ORBIT_MOD_1) + GET_SHORTCUT_COUNT(SHORTCUT_ORBIT_MOD_2);
		int pan_mod_input_count = GET_SHORTCUT_COUNT(SHORTCUT_PAN_MOD_1) + GET_SHORTCUT_COUNT(SHORTCUT_PAN_MOD_2);
		int zoom_mod_input_count = GET_SHORTCUT_COUNT(SHORTCUT_ZOOM_MOD_1) + GET_SHORTCUT_COUNT(SHORTCUT_ZOOM_MOD_2);
		bool orbit_not_empty = !_is_shortcut_empty(SHORTCUT_ORBIT_MOD_1) || !_is_shortcut_empty(SHORTCUT_ORBIT_MOD_2);
		bool pan_not_empty = !_is_shortcut_empty(SHORTCUT_PAN_MOD_2) || !_is_shortcut_empty(SHORTCUT_PAN_MOD_2);
		bool zoom_not_empty = !_is_shortcut_empty(SHORTCUT_ZOOM_MOD_1) || !_is_shortcut_empty(SHORTCUT_ZOOM_MOD_2);
		shortcut_checks.push_back(ShortcutCheck(orbit_mod_pressed, orbit_not_empty, orbit_mod_input_count, orbit_mouse_button, NAV_MODE_ORBIT));
		shortcut_checks.push_back(ShortcutCheck(pan_mod_pressed, pan_not_empty, pan_mod_input_count, pan_mouse_button, NAV_MODE_PAN));
		shortcut_checks.push_back(ShortcutCheck(zoom_mod_pressed, zoom_not_empty, zoom_mod_input_count, zoom_mouse_button, NAV_MODE_ZOOM));
		shortcut_checks.sort_custom<ShortcutCheckSetComparator>();

#undef GET_SHORTCUT_COUNT
	}

	Ref<InputEventMouseMotion> m = p_event;
	if (m.is_valid()) {
		if (m->get_button_mask() == MouseButtonMask::NONE) {
			navigation_cancelled = false;
		}

		if (navigation_cancelled) {
			return false;
		}

		if (!navigating) {
			previous_cursor = cursor;
		}

		NavigationMode nav_mode = NAV_MODE_NONE;

		if (m->get_button_mask().has_flag(MouseButtonMask::LEFT)) {
			NavigationMode change_nav_from_shortcut = _get_nav_mode_from_shortcuts(NAV_MOUSE_BUTTON_LEFT, shortcut_checks, false);
			if (change_nav_from_shortcut != NAV_MODE_NONE) {
				nav_mode = change_nav_from_shortcut;
			}
		} else if (freelook || m->get_button_mask().has_flag(MouseButtonMask::RIGHT)) {
			NavigationMode change_nav_from_shortcut = _get_nav_mode_from_shortcuts(NAV_MOUSE_BUTTON_RIGHT, shortcut_checks, false);
			if (m->get_button_mask().has_flag(MouseButtonMask::RIGHT) && change_nav_from_shortcut != NAV_MODE_NONE) {
				nav_mode = change_nav_from_shortcut;
			} else if (freelook) {
				nav_mode = NAV_MODE_LOOK;
			} else if (orthogonal) {
				nav_mode = NAV_MODE_PAN;
			}

		} else if (m->get_button_mask().has_flag(MouseButtonMask::MIDDLE)) {
			NavigationMode change_nav_from_shortcut = _get_nav_mode_from_shortcuts(NAV_MOUSE_BUTTON_MIDDLE, shortcut_checks, false);
			if (change_nav_from_shortcut != NAV_MODE_NONE) {
				nav_mode = change_nav_from_shortcut;
			}

		} else if (m->get_button_mask().has_flag(MouseButtonMask::MB_XBUTTON1)) {
			NavigationMode change_nav_from_shortcut = _get_nav_mode_from_shortcuts(NAV_MOUSE_BUTTON_4, shortcut_checks, false);
			if (change_nav_from_shortcut != NAV_MODE_NONE) {
				nav_mode = change_nav_from_shortcut;
			}

		} else if (m->get_button_mask().has_flag(MouseButtonMask::MB_XBUTTON2)) {
			NavigationMode change_nav_from_shortcut = _get_nav_mode_from_shortcuts(NAV_MOUSE_BUTTON_5, shortcut_checks, false);
			if (change_nav_from_shortcut != NAV_MODE_NONE) {
				nav_mode = change_nav_from_shortcut;
			}

		} else if (emulate_3_button_mouse) {
			// Handle trackpad (no external mouse) use case.
			NavigationMode change_nav_from_shortcut = _get_nav_mode_from_shortcuts(NAV_MOUSE_BUTTON_LEFT, shortcut_checks, true);
			if (change_nav_from_shortcut != NAV_MODE_NONE) {
				nav_mode = change_nav_from_shortcut;
			}
		}

		switch (nav_mode) {
			case NAV_MODE_PAN: {
				cursor_pan(m, get_warped_mouse_motion(m, p_surface_rect));
			} break;

			case NAV_MODE_ZOOM: {
				cursor_zoom(m, m->get_relative());
			} break;

			case NAV_MODE_ORBIT: {
				cursor_orbit(m, get_warped_mouse_motion(m, p_surface_rect));
			} break;

			case NAV_MODE_LOOK: {
				cursor_look(m, get_warped_mouse_motion(m, p_surface_rect));
			} break;

			default: {
				navigating = false;
				return false;
			}
		}

		if (!freelook) {
			navigating = true;
		}
		return true;
	}

	Ref<InputEventMagnifyGesture> magnify_gesture = p_event;
	if (magnify_gesture.is_valid()) {
		if (freelook) {
			scale_freelook_speed(magnify_gesture->get_factor());
		} else {
			scale_cursor_distance(1.0 / magnify_gesture->get_factor());
		}

		return true;
	}

	Ref<InputEventPanGesture> pan_gesture = p_event;
	if (pan_gesture.is_valid()) {
		NavigationMode nav_mode = NAV_MODE_NONE;

		for (const ShortcutCheck &shortcut_check_set : shortcut_checks) {
			if (shortcut_check_set.mod_pressed) {
				nav_mode = shortcut_check_set.result_nav_mode;
				break;
			}
		}

		switch (nav_mode) {
			case NAV_MODE_PAN: {
				cursor_pan(pan_gesture, -pan_gesture->get_delta());
			} break;

			case NAV_MODE_ZOOM: {
				cursor_zoom(pan_gesture, pan_gesture->get_delta());
			} break;

			case NAV_MODE_ORBIT: {
				cursor_orbit(pan_gesture, -pan_gesture->get_delta());
			} break;

			case NAV_MODE_LOOK: {
				cursor_look(pan_gesture, pan_gesture->get_delta());
			} break;

			default: {
				return false;
			}
		}

		return true;
	}

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->is_pressed() && !k->is_echo() && k->get_keycode() == Key::ESCAPE && navigating) {
		cancel_navigation();
		return true;
	}

	bool pressed = false;
	float old_fov_scale = cursor.fov_scale;

	if (_is_shortcut_pressed(SHORTCUT_FOV_DECREASE)) {
		cursor.fov_scale = CLAMP(cursor.fov_scale - 0.05, CAMERA_MIN_FOV_SCALE, CAMERA_MAX_FOV_SCALE);
		pressed = true;
	}
	if (_is_shortcut_pressed(SHORTCUT_FOV_INCREASE)) {
		cursor.fov_scale = CLAMP(cursor.fov_scale + 0.05, CAMERA_MIN_FOV_SCALE, CAMERA_MAX_FOV_SCALE);
		pressed = true;
	}
	if (_is_shortcut_pressed(SHORTCUT_FOV_RESET)) {
		cursor.fov_scale = 1;
		pressed = true;
	}

	if (old_fov_scale != cursor.fov_scale) {
		emit_signal(SNAME("fov_scaled"));
	}

	return pressed;
}

void View3DController::cancel_navigation() {
	navigating = false;
	navigation_cancelled = true;
	cursor = previous_cursor;
}

void View3DController::cursor_pan(const Ref<InputEventWithModifiers> &p_event, const Vector2 &p_relative) {
	float pan_speed = translation_sensitivity / 150.0;
	if (p_event.is_valid() && navigation_scheme == NAV_SCHEME_MAYA && p_event->is_shift_pressed()) {
		pan_speed *= 10;
	}

	Transform3D camera_transform;

	camera_transform.translate_local(cursor.pos);
	camera_transform.basis.rotate(Vector3(1, 0, 0), -cursor.x_rot);
	camera_transform.basis.rotate(Vector3(0, 1, 0), -cursor.y_rot);
	Vector3 translation(
			(invert_x_axis ? -1 : 1) * -p_relative.x * pan_speed,
			(invert_y_axis ? -1 : 1) * p_relative.y * pan_speed,
			0);
	translation *= cursor.distance / DISTANCE_DEFAULT;
	camera_transform.translate_local(translation);
	cursor.pos = camera_transform.origin;

	emit_signal(SNAME("cursor_panned"));
}

void View3DController::cursor_orbit(const Ref<InputEventWithModifiers> &p_event, const Vector2 &p_relative) {
	if (lock_rotation) {
		cursor_pan(p_event, p_relative);
		return;
	}

	const float radians_per_pixel = Math::deg_to_rad(orbit_sensitivity);

	cursor.unsnapped_x_rot += p_relative.y * radians_per_pixel * (invert_y_axis ? -1 : 1);
	cursor.unsnapped_x_rot = CLAMP(cursor.unsnapped_x_rot, -1.57, 1.57);
	cursor.unsnapped_y_rot += p_relative.x * radians_per_pixel * (invert_x_axis ? -1 : 1);

	cursor.x_rot = cursor.unsnapped_x_rot;
	cursor.y_rot = cursor.unsnapped_y_rot;

	ViewType new_view_type = VIEW_TYPE_USER;

	bool snap_modifier_configured = !_is_shortcut_empty(SHORTCUT_ORBIT_SNAP_MOD_1) || !_is_shortcut_empty(SHORTCUT_ORBIT_SNAP_MOD_2);
	if (snap_modifier_configured && _is_shortcut_pressed(SHORTCUT_ORBIT_SNAP_MOD_1, true) && _is_shortcut_pressed(SHORTCUT_ORBIT_SNAP_MOD_2, true)) {
		const float snap_angle = Math::deg_to_rad(45.0);
		const float snap_threshold = Math::deg_to_rad(angle_snap_threshold);

		float x_rot_snapped = Math::snapped(cursor.unsnapped_x_rot, snap_angle);
		float y_rot_snapped = Math::snapped(cursor.unsnapped_y_rot, snap_angle);

		float x_dist = Math::abs(cursor.unsnapped_x_rot - x_rot_snapped);
		float y_dist = Math::abs(cursor.unsnapped_y_rot - y_rot_snapped);

		if (x_dist < snap_threshold && y_dist < snap_threshold) {
			cursor.x_rot = x_rot_snapped;
			cursor.y_rot = y_rot_snapped;

			float y_rot_wrapped = Math::wrapf(y_rot_snapped, (float)-Math::PI, (float)Math::PI);

			if (Math::abs(x_rot_snapped) < snap_threshold) {
				// Only switch to ortho for 90-degree views.
				if (Math::abs(y_rot_wrapped) < snap_threshold) {
					new_view_type = VIEW_TYPE_FRONT;
				} else if (Math::abs(Math::abs(y_rot_wrapped) - Math::PI) < snap_threshold) {
					new_view_type = VIEW_TYPE_REAR;
				} else if (Math::abs(y_rot_wrapped - Math::PI / 2.0) < snap_threshold) {
					new_view_type = VIEW_TYPE_LEFT;
				} else if (Math::abs(y_rot_wrapped + Math::PI / 2.0) < snap_threshold) {
					new_view_type = VIEW_TYPE_RIGHT;
				}

			} else if (Math::abs(Math::abs(x_rot_snapped) - Math::PI / 2.0) < snap_threshold) {
				if (Math::abs(y_rot_wrapped) < snap_threshold ||
						Math::abs(Math::abs(y_rot_wrapped) - Math::PI) < snap_threshold ||
						Math::abs(y_rot_wrapped - Math::PI / 2.0) < snap_threshold ||
						Math::abs(y_rot_wrapped + Math::PI / 2.0) < snap_threshold) {
					new_view_type = x_rot_snapped > 0 ? VIEW_TYPE_TOP : VIEW_TYPE_BOTTOM;
				}
			}
		}
	}

	set_view_type(new_view_type);
}

void View3DController::cursor_look(const Ref<InputEventWithModifiers> &p_event, const Vector2 &p_relative) {
	if (orthogonal) {
		cursor_pan(p_event, p_relative);
		return;
	}

	// Scale mouse sensitivity with camera FOV scale when zoomed in to make it easier to point at things.
	const float degrees_per_pixel = freelook_sensitivity * MIN(1.0, cursor.fov_scale);
	const float radians_per_pixel = Math::deg_to_rad(degrees_per_pixel);

	// Note: do NOT assume the camera has the "current" transform, because it is interpolated and may have "lag".
	const Transform3D prev_camera_transform = to_camera_transform();

	if (freelook_invert_y_axis) {
		cursor.x_rot -= p_relative.y * radians_per_pixel;
	} else {
		cursor.x_rot += p_relative.y * radians_per_pixel;
	}

	// Clamp the Y rotation to roughly -90..90 degrees so the user can't look upside-down and end up disoriented.
	cursor.x_rot = CLAMP(cursor.x_rot, -1.57, 1.57);
	cursor.unsnapped_x_rot = cursor.x_rot;

	cursor.y_rot += p_relative.x * radians_per_pixel;
	cursor.unsnapped_y_rot = cursor.y_rot;

	// Look is like the opposite of Orbit: the focus point rotates around the camera
	Transform3D camera_transform = to_camera_transform();
	Vector3 pos = camera_transform.xform(Vector3(0, 0, 0));
	Vector3 prev_pos = prev_camera_transform.xform(Vector3(0, 0, 0));
	Vector3 diff = prev_pos - pos;
	cursor.pos += diff;

	set_view_type(VIEW_TYPE_USER);
}

void View3DController::cursor_zoom(const Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative) {
	float zoom_speed = 1 / 80.0;
	if (p_event.is_valid() && navigation_scheme == NAV_SCHEME_MAYA && p_event->is_shift_pressed()) {
		zoom_speed *= 10;
	}

	if (zoom_style == ZOOM_HORIZONTAL) {
		if (p_relative.x > 0) {
			scale_cursor_distance(1 - p_relative.x * zoom_speed);
		} else if (p_relative.x < 0) {
			scale_cursor_distance(1.0 / (1 + p_relative.x * zoom_speed));
		}
	} else {
		if (p_relative.y > 0) {
			scale_cursor_distance(1 + p_relative.y * zoom_speed);
		} else if (p_relative.y < 0) {
			scale_cursor_distance(1.0 / (1 - p_relative.y * zoom_speed));
		}
	}
}

void View3DController::update_camera(const real_t p_delta) {
	View3DController::Cursor old_camera_cursor = cursor_interp;
	cursor_interp = cursor;
	bool equal = true;

	if (p_delta > 0) {
		// Perform smoothing.

		if (freelook) {
			// Higher inertia should increase "lag" (lerp with factor between 0 and 1).
			// Inertia of zero should produce instant movement (lerp with factor of 1) in this case it returns a really high value and gets clamped to 1.
			float factor = (1.0 / freelook_inertia) * p_delta;

			// We interpolate a different point here, because in freelook mode the focus point (cursor.pos) orbits around eye_pos
			cursor_interp.eye_pos = old_camera_cursor.eye_pos.lerp(cursor.eye_pos, CLAMP(factor, 0, 1));
		}

		cursor_interp.x_rot = Math::lerp(old_camera_cursor.x_rot, cursor.x_rot, MIN(1.f, p_delta * (1 / orbit_inertia)));
		cursor_interp.y_rot = Math::lerp(old_camera_cursor.y_rot, cursor.y_rot, MIN(1.f, p_delta * (1 / orbit_inertia)));

		if (Math::abs(cursor_interp.x_rot - cursor.x_rot) < 0.1) {
			cursor_interp.x_rot = cursor.x_rot;
		}
		if (Math::abs(cursor_interp.y_rot - cursor.y_rot) < 0.1) {
			cursor_interp.y_rot = cursor.y_rot;
		}

		if (freelook) {
			Vector3 forward = _to_camera_transform(cursor_interp).basis.xform(Vector3(0, 0, -1));
			cursor_interp.pos = cursor_interp.eye_pos + forward * cursor_interp.distance;
		} else {
			cursor_interp.pos = old_camera_cursor.pos.lerp(cursor.pos, MIN(1.f, p_delta * (1 / translation_inertia)));
			cursor_interp.distance = Math::lerp(old_camera_cursor.distance, cursor.distance, MIN((float)1.0, p_delta * (1 / zoom_inertia)));
		}

		// Apply camera transform.

		const real_t tolerance = 0.001;
		if (!Math::is_equal_approx(old_camera_cursor.x_rot, cursor_interp.x_rot, tolerance) || !Math::is_equal_approx(old_camera_cursor.y_rot, cursor_interp.y_rot, tolerance)) {
			equal = false;
		} else if (!old_camera_cursor.pos.is_equal_approx(cursor_interp.pos)) {
			equal = false;
		} else if (!Math::is_equal_approx(old_camera_cursor.distance, cursor_interp.distance, tolerance)) {
			equal = false;
		} else if (!Math::is_equal_approx(old_camera_cursor.fov_scale, cursor_interp.fov_scale, tolerance)) {
			equal = false;
		}
	}

	if (p_delta == 0 || !equal) {
		emit_signal(SNAME("cursor_interpolated"));
	}
}

void View3DController::update_freelook(const float p_delta) {
	if (!freelook) {
		return;
	}

	const Transform3D camera_transform = to_camera_transform();

	Vector3 forward;
	if (freelook_scheme == FREELOOK_FULLY_AXIS_LOCKED) {
		// Forward/backward keys will always go straight forward/backward, never moving on the Y axis.
		forward = Vector3(0, 0, -1).rotated(Vector3(0, 1, 0), camera_transform.get_basis().get_euler().y);
	} else {
		// Forward/backward keys will be relative to the camera pitch.
		forward = camera_transform.basis.xform(Vector3(0, 0, -1));
	}

	const Vector3 right = camera_transform.basis.xform(Vector3(1, 0, 0));

	Vector3 up;
	if (freelook_scheme == View3DController::FREELOOK_PARTIALLY_AXIS_LOCKED || freelook_scheme == View3DController::FREELOOK_FULLY_AXIS_LOCKED) {
		// Up/down keys will always go up/down regardless of camera pitch.
		up = Vector3(0, 1, 0);
	} else {
		// Up/down keys will be relative to the camera pitch.
		up = camera_transform.basis.xform(Vector3(0, 1, 0));
	}

	Vector3 direction;
	if (_is_shortcut_pressed(SHORTCUT_FREELOOK_LEFT)) {
		direction -= right;
	}
	if (_is_shortcut_pressed(SHORTCUT_FREELOOK_RIGHT)) {
		direction += right;
	}
	if (_is_shortcut_pressed(SHORTCUT_FREELOOK_FORWARD)) {
		direction += forward;
	}
	if (_is_shortcut_pressed(SHORTCUT_FREELOOK_BACKWARDS)) {
		direction -= forward;
	}
	if (_is_shortcut_pressed(SHORTCUT_FREELOOK_UP)) {
		direction += up;
	}
	if (_is_shortcut_pressed(SHORTCUT_FREELOOK_DOWN)) {
		direction -= up;
	}

	real_t speed = freelook_speed;

	if (_is_shortcut_pressed(SHORTCUT_FREELOOK_SPEED_MOD)) {
		speed *= 3.0;
	}
	if (_is_shortcut_pressed(SHORTCUT_FREELOOK_SLOW_MOD)) {
		speed *= 0.333333;
	}

	const Vector3 motion = direction * speed * p_delta;
	cursor.pos += motion;
	cursor.eye_pos += motion;
}

void View3DController::scale_freelook_speed(const float p_scale) {
	float min_speed = MAX(znear * 4, ZOOM_FREELOOK_MIN);
	float max_speed = MIN(zfar / 4, ZOOM_FREELOOK_MAX);
	if (unlikely(min_speed > max_speed)) {
		freelook_speed = (min_speed + max_speed) / 2;
	} else {
		freelook_speed = CLAMP(freelook_speed * p_scale, min_speed, max_speed);
	}

	emit_signal(SNAME("freelook_speed_scaled"));
}

void View3DController::scale_cursor_distance(const float p_scale) {
	float min_distance = MAX(znear * 4, ZOOM_FREELOOK_MIN);
	float max_distance = MIN(zfar / 4, ZOOM_FREELOOK_MAX);
	if (unlikely(min_distance > max_distance)) {
		cursor.distance = (min_distance + max_distance) / 2;
	} else {
		cursor.distance = CLAMP(cursor.distance * p_scale, min_distance, max_distance);
	}

	if (cursor.distance == max_distance || cursor.distance == min_distance) {
		zoom_failed_attempts_count++;
	} else {
		zoom_failed_attempts_count = 0;
	}

	emit_signal(SNAME("cursor_distance_scaled"));
}

void View3DController::set_shortcut(const ShortcutName p_name, const Ref<Shortcut> &p_shortcut) {
	ERR_FAIL_INDEX(0, SHORTCUT_MAX);
	ERR_FAIL_COND(p_shortcut.is_null());
	inputs[p_name] = p_shortcut;
}

void View3DController::set_view_type(const ViewType p_view) {
	ViewType view_type_old = view_type;
	OrthogonalMode orthogonal_old = orthogonal;

	view_type = p_view;
	if (view_type != VIEW_TYPE_USER) {
		if (auto_orthogonal_allowed && orthogonal != ORTHOGONAL_ENABLED) {
			orthogonal = ORTHOGONAL_AUTO;
		}
	} else if (orthogonal == ORTHOGONAL_AUTO) {
		orthogonal = ORTHOGONAL_DISABLED;
	}

	if (view_type_old != view_type || orthogonal_old != orthogonal) {
		emit_signal(SNAME("view_state_changed"));
	}
}

String View3DController::get_view_type_name() const {
	String name;

	switch (view_type) {
		case VIEW_TYPE_USER: {
			if (orthogonal) {
				name = RTR("Orthogonal");
			} else {
				name = RTR("Perspective");
			}
		} break;
		case VIEW_TYPE_TOP: {
			if (orthogonal) {
				name = RTR("Top Orthogonal");
			} else {
				name = RTR("Top Perspective");
			}
		} break;
		case VIEW_TYPE_BOTTOM: {
			if (orthogonal) {
				name = RTR("Bottom Orthogonal");
			} else {
				name = RTR("Bottom Perspective");
			}
		} break;
		case VIEW_TYPE_LEFT: {
			if (orthogonal) {
				name = RTR("Left Orthogonal");
			} else {
				name = RTR("Left Perspective");
			}
		} break;
		case VIEW_TYPE_RIGHT: {
			if (orthogonal) {
				name = RTR("Right Orthogonal");
			} else {
				name = RTR("Right Perspective");
			}
		} break;
		case VIEW_TYPE_FRONT: {
			if (orthogonal) {
				name = RTR("Front Orthogonal");
			} else {
				name = RTR("Front Perspective");
			}
		} break;
		case VIEW_TYPE_REAR: {
			if (orthogonal) {
				name = RTR("Rear Orthogonal");
			} else {
				name = RTR("Rear Perspective");
			}
		} break;
	}

	if (orthogonal == ORTHOGONAL_AUTO) {
		// TRANSLATORS: This will be appended to the view name when Auto Orthogonal is enabled.
		name += " " + RTR("[auto]");
	}

	return name;
}

void View3DController::set_freelook_enabled(const bool p_enabled) {
	if (freelook == p_enabled) {
		return;
	}

	freelook = p_enabled;

	// Sync interpolated cursor to cursor to "cut" interpolation jumps due to changing referential.
	cursor = cursor_interp;

	if (freelook) {
		// Make sure eye_pos is synced, because freelook referential is eye pos rather than orbit pos.
		Vector3 forward = to_camera_transform().basis.xform(Vector3(0, 0, -1));
		cursor.eye_pos = cursor.pos - cursor.distance * forward;
		// Also sync the interpolated cursor's eye_pos, otherwise switching to freelook will be trippy if inertia is active.
		cursor_interp.eye_pos = cursor.eye_pos;

		if (freelook_speed_zoom_link) {
			// Re-adjust freelook speed from the current zoom level.
			freelook_speed = freelook_base_speed * cursor.distance;
		}

		previous_mouse_position = SceneTree::get_singleton()->get_root()->get_mouse_position();

		// Hide mouse like in an FPS (warping doesn't work).
		if (Engine::get_singleton()->is_editor_hint()) {
			Input::get_singleton()->set_mouse_mode(Input::MouseMode::MOUSE_MODE_CAPTURED);
		} else {
			Input::get_singleton()->set_mouse_mode_override(Input::MouseMode::MOUSE_MODE_CAPTURED);
		}
	} else {
		// Restore mouse.
		if (Engine::get_singleton()->is_editor_hint()) {
			Input::get_singleton()->set_mouse_mode(Input::MouseMode::MOUSE_MODE_VISIBLE);
		} else {
			Input::get_singleton()->set_mouse_mode_override(Input::MouseMode::MOUSE_MODE_VISIBLE);
		}

		// Restore the previous mouse position when leaving freelook mode.
		// This is done because leaving `Input.MOUSE_MODE_CAPTURED` will center the cursor
		// due to OS limitations.
		Input::get_singleton()->warp_mouse(previous_mouse_position);
	}

	emit_signal(SNAME("freelook_changed"));
}

void View3DController::set_freelook_base_speed(const float p_speed) {
	freelook_base_speed = p_speed;
	freelook_speed = p_speed;
}

void View3DController::force_auto_orthogonal() {
	if (auto_orthogonal_allowed) {
		orthogonal = ORTHOGONAL_AUTO;
	}
}

void View3DController::set_auto_orthogonal_allowed(const bool p_enabled) {
	auto_orthogonal_allowed = p_enabled;

	if (!p_enabled && orthogonal == ORTHOGONAL_AUTO) {
		orthogonal = ORTHOGONAL_ENABLED;
	}
}

Point2 View3DController::get_warped_mouse_motion(const Ref<InputEventMouseMotion> &p_event, const Rect2 &p_surface_rect) const {
	if (warped_mouse_panning) {
		return Input::get_singleton()->warp_mouse_motion(p_event, p_surface_rect);
	}

	return p_event->get_relative();
}

void View3DController::_bind_methods() {
	ADD_SIGNAL(MethodInfo("view_state_changed"));

	ADD_SIGNAL(MethodInfo("fov_scaled"));

	ADD_SIGNAL(MethodInfo("freelook_changed"));
	ADD_SIGNAL(MethodInfo("freelook_speed_scaled"));

	ADD_SIGNAL(MethodInfo("cursor_panned"));
	ADD_SIGNAL(MethodInfo("cursor_interpolated"));
	ADD_SIGNAL(MethodInfo("cursor_distance_scaled"));
}

#endif // _3D_DISABLED
