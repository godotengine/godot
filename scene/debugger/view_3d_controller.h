/**************************************************************************/
/*  view_3d_controller.h                                                  */
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

#pragma once

#ifndef _3D_DISABLED

#include "core/object/ref_counted.h"

namespace View3DControllerConsts {
constexpr float DISTANCE_DEFAULT = 4;

constexpr float ZOOM_FREELOOK_MULTIPLIER = 1.08;
constexpr float ZOOM_FREELOOK_MIN = 0.01;
#ifdef REAL_T_IS_DOUBLE
constexpr double ZOOM_FREELOOK_MAX = 1'000'000'000'000;
#else
constexpr float ZOOM_FREELOOK_MAX = 10'000;
#endif

constexpr float CAMERA_MIN_FOV_SCALE = 0.1;
constexpr float CAMERA_MAX_FOV_SCALE = 2.5;
} //namespace View3DControllerConsts

class InputEvent;
class InputEventMouseMotion;
class InputEventWithModifiers;
class Shortcut;

class View3DController : public RefCounted {
	GDCLASS(View3DController, RefCounted);

public:
	enum NavigationMode {
		NAV_MODE_NONE,
		NAV_MODE_PAN,
		NAV_MODE_ZOOM,
		NAV_MODE_ORBIT,
		NAV_MODE_LOOK,
		NAV_MODE_MOVE,
	};

	enum NavigationScheme {
		NAV_SCHEME_GODOT,
		NAV_SCHEME_MAYA,
		NAV_SCHEME_MODO,
		NAV_SCHEME_CUSTOM,
		NAV_SCHEME_TABLET,
	};

	enum NavigationMouseButton {
		NAV_MOUSE_BUTTON_LEFT,
		NAV_MOUSE_BUTTON_MIDDLE,
		NAV_MOUSE_BUTTON_RIGHT,
		NAV_MOUSE_BUTTON_4,
		NAV_MOUSE_BUTTON_5,
	};

	enum ZoomStyle {
		ZOOM_VERTICAL,
		ZOOM_HORIZONTAL,
	};

	enum FreelookScheme {
		FREELOOK_DEFAULT,
		FREELOOK_PARTIALLY_AXIS_LOCKED,
		FREELOOK_FULLY_AXIS_LOCKED,
	};

	enum ViewType {
		VIEW_TYPE_USER,
		VIEW_TYPE_TOP,
		VIEW_TYPE_BOTTOM,
		VIEW_TYPE_LEFT,
		VIEW_TYPE_RIGHT,
		VIEW_TYPE_FRONT,
		VIEW_TYPE_REAR,
	};

	enum OrthogonalMode {
		ORTHOGONAL_DISABLED,
		ORTHOGONAL_ENABLED,
		ORTHOGONAL_AUTO,
	};

	enum ShortcutName {
		SHORTCUT_FOV_INCREASE,
		SHORTCUT_FOV_DECREASE,
		SHORTCUT_FOV_RESET,

		SHORTCUT_PAN_MOD_1,
		SHORTCUT_PAN_MOD_2,

		SHORTCUT_ORBIT_MOD_1,
		SHORTCUT_ORBIT_MOD_2,
		SHORTCUT_ORBIT_SNAP_MOD_1,
		SHORTCUT_ORBIT_SNAP_MOD_2,

		SHORTCUT_ZOOM_MOD_1,
		SHORTCUT_ZOOM_MOD_2,

		SHORTCUT_FREELOOK_FORWARD,
		SHORTCUT_FREELOOK_BACKWARDS,
		SHORTCUT_FREELOOK_LEFT,
		SHORTCUT_FREELOOK_RIGHT,
		SHORTCUT_FREELOOK_UP,
		SHORTCUT_FREELOOK_DOWN,
		SHORTCUT_FREELOOK_SPEED_MOD,
		SHORTCUT_FREELOOK_SLOW_MOD,

		SHORTCUT_MAX,
	};

	struct Cursor {
		Vector3 pos;
		real_t x_rot;
		real_t y_rot;
		real_t distance;
		real_t fov_scale;
		real_t unsnapped_x_rot;
		real_t unsnapped_y_rot;
		Vector3 eye_pos; // Used for freelook.
		// TODO: These variables are not related to cursor manipulation, and specific
		// to Node3DEditorPlugin. So remove them in the future.
		bool region_select;
		Point2 region_begin;
		Point2 region_end;

		Cursor() {
			// These rotations place the camera in +X +Y +Z, aka south east, facing north west.
			x_rot = 0.5;
			y_rot = -0.5;
			unsnapped_x_rot = x_rot;
			unsnapped_y_rot = y_rot;
			distance = 4;
			fov_scale = 1.0;
			region_select = false;
		}
	};

	// Viewport camera supports movement smoothing,
	// so one cursor is the real cursor, while the other can be an interpolated version.
	Cursor cursor; // Immediate cursor.

private:
	Cursor cursor_interp; // That one may be interpolated (don't modify this one except for smoothing purposes).
	Cursor previous_cursor; // Storing previous cursor state for canceling purposes.
	bool navigating = false;
	bool navigation_cancelled = false;
	void cancel_navigation();

protected:
	static void _bind_methods();

private:
	HashMap<int, Ref<Shortcut>> inputs;

	NavigationScheme navigation_scheme = NAV_SCHEME_GODOT;
	ViewType view_type = VIEW_TYPE_USER;

	NavigationMouseButton pan_mouse_button = NAV_MOUSE_BUTTON_MIDDLE;

	NavigationMouseButton orbit_mouse_button = NAV_MOUSE_BUTTON_MIDDLE;
	float orbit_sensitivity = 0;
	float orbit_inertia = 0;

	bool freelook = false;
	FreelookScheme freelook_scheme = FREELOOK_DEFAULT;
	float freelook_speed = 0;
	float freelook_base_speed = 0;
	float freelook_speed_zoom_link = 0;
	float freelook_sensitivity = 0;
	float freelook_inertia = 0;
	bool freelook_invert_y_axis = false;

	ZoomStyle zoom_style = ZOOM_VERTICAL;
	NavigationMouseButton zoom_mouse_button = NAV_MOUSE_BUTTON_MIDDLE;
	float zoom_inertia = 0;
	int zoom_failed_attempts_count = 0;

	float translation_sensitivity = 0;
	float translation_inertia = 0;

	float angle_snap_threshold = 0;
	bool warped_mouse_panning = false;

	bool emulate_3_button_mouse = false;
	bool emulate_numpad = true;

	OrthogonalMode orthogonal = ORTHOGONAL_DISABLED;
	bool auto_orthogonal_allowed = false;

	bool lock_rotation = false;

	float znear = 0;
	float zfar = 0;

	bool invert_x_axis = false;
	bool invert_y_axis = false;

	Point2 previous_mouse_position;

	Transform3D _to_camera_transform(const Cursor &p_cursor) const;

	struct ShortcutCheck {
		bool mod_pressed = false;
		bool not_empty = true;
		int input_count = 0;
		NavigationMouseButton mouse_preference = NAV_MOUSE_BUTTON_LEFT;
		NavigationMode result_nav_mode = NAV_MODE_NONE;

		ShortcutCheck(bool p_mod_pressed, bool p_not_empty, int p_input_count, const NavigationMouseButton &p_mouse_preference, const NavigationMode &p_result_nav_mode) :
				mod_pressed(p_mod_pressed), not_empty(p_not_empty), input_count(p_input_count), mouse_preference(p_mouse_preference), result_nav_mode(p_result_nav_mode) {
		}
	};

	struct ShortcutCheckSetComparator {
		_FORCE_INLINE_ bool operator()(const ShortcutCheck &A, const ShortcutCheck &B) const {
			return A.input_count > B.input_count;
		}
	};

	bool _is_shortcut_pressed(const ShortcutName p_name, const bool p_true_if_null = false);
	bool _is_shortcut_empty(const ShortcutName p_name);
	NavigationMode _get_nav_mode_from_shortcuts(NavigationMouseButton p_mouse_button, const Vector<ShortcutCheck> &p_shortcut_checks, bool p_not_empty);

public:
	bool gui_input(const Ref<InputEvent> &p_event, const Rect2 &p_surface_rect);
	bool is_navigating() const { return navigating; }

	void cursor_pan(const Ref<InputEventWithModifiers> &p_event, const Vector2 &p_relative);
	void cursor_look(const Ref<InputEventWithModifiers> &p_event, const Vector2 &p_relative);
	void cursor_orbit(const Ref<InputEventWithModifiers> &p_event, const Vector2 &p_relative);
	void cursor_zoom(const Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative);

	void update_camera(const real_t p_delta = 0);

	void update_freelook(const float p_delta);
	void scale_freelook_speed(const float p_scale);

	void scale_cursor_distance(const float p_scale);

	inline Transform3D to_camera_transform() const { return _to_camera_transform(cursor); }
	inline Transform3D interp_to_camera_transform() const { return _to_camera_transform(cursor_interp); }

	void set_shortcut(const ShortcutName p_name, const Ref<Shortcut> &p_shortcut);

	void set_navigation_scheme(const NavigationScheme p_scheme) { navigation_scheme = p_scheme; }

	void set_view_type(const ViewType p_view);
	ViewType get_view_type() const { return view_type; }
	String get_view_type_name() const;

	void set_pan_mouse_button(const NavigationMouseButton p_button) { pan_mouse_button = p_button; }

	void set_orbit_sensitivity(const float p_sensitivity) { orbit_sensitivity = p_sensitivity; }
	void set_orbit_inertia(const float p_inertia) { orbit_inertia = p_inertia; }
	void set_orbit_mouse_button(const NavigationMouseButton p_button) { orbit_mouse_button = p_button; }

	void set_freelook_enabled(const bool p_enabled);
	bool is_freelook_enabled() const { return freelook; }
	void set_freelook_scheme(FreelookScheme p_scheme) { freelook_scheme = p_scheme; }
	FreelookScheme get_freelook_scheme() const { return freelook_scheme; }
	void set_freelook_base_speed(const float p_speed);
	float get_freelook_speed() const { return freelook_speed; }
	void set_freelook_sensitivity(const float p_sensitivity) { freelook_sensitivity = p_sensitivity; }
	void set_freelook_inertia(const float p_inertia) { freelook_inertia = p_inertia; }
	void set_freelook_speed_zoom_link(const bool p_enabled) { freelook_speed_zoom_link = p_enabled; }
	void set_freelook_invert_y_axis(const bool p_enabled) { freelook_invert_y_axis = p_enabled; }

	void set_zoom_style(ZoomStyle p_style) { zoom_style = p_style; }
	void set_zoom_inertia(const float p_inertia) { zoom_inertia = p_inertia; }
	void set_zoom_mouse_button(const NavigationMouseButton p_button) { zoom_mouse_button = p_button; }

	void set_translation_sensitivity(const float p_sensitivity) { translation_sensitivity = p_sensitivity; }
	void set_translation_inertia(const float p_inertia) { translation_inertia = p_inertia; }

	void set_angle_snap_threshold(const float p_threshold) { angle_snap_threshold = p_threshold; }

	void set_emulate_3_button_mouse(const bool p_enabled) { emulate_3_button_mouse = p_enabled; }
	void set_emulate_numpad(const bool p_enabled) { emulate_numpad = p_enabled; }

	void set_orthogonal(const bool p_enabled) { orthogonal = p_enabled ? ORTHOGONAL_ENABLED : ORTHOGONAL_DISABLED; }
	bool is_orthogonal() const { return orthogonal != ORTHOGONAL_DISABLED; }
	OrthogonalMode get_orthogonal_mode() const { return orthogonal; }
	void force_auto_orthogonal();
	void set_auto_orthogonal_allowed(const bool p_enabled);

	void set_lock_rotation(const bool p_locked) { lock_rotation = p_locked; }
	bool is_locking_rotation() { return lock_rotation; }

	void set_z_near(const float p_near) { znear = p_near; }
	void set_z_far(const float p_far) { zfar = p_far; }

	void set_invert_x_axis(const bool p_invert) { invert_x_axis = p_invert; }
	void set_invert_y_axis(const bool p_invert) { invert_y_axis = p_invert; }

	void set_warped_mouse_panning(const bool p_enabled) { warped_mouse_panning = p_enabled; }
	Point2 get_warped_mouse_motion(const Ref<InputEventMouseMotion> &p_event, const Rect2 &p_surface_rect) const;

	int get_zoom_failed_attempts_count() const { return zoom_failed_attempts_count; }
};

#endif // _3D_DISABLED
