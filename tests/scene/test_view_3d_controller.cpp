/**************************************************************************/
/*  test_view_3d_controller.cpp                                           */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_view_3d_controller)

#ifndef _3D_DISABLED

#include "core/input/input.h"
#include "core/input/shortcut.h"
#include "scene/debugger/view_3d_controller.h"

namespace TestView3DController {

static void _setup_shortcut(Ref<View3DController> &p_controller, View3DController::ShortcutName p_name, Key p_key) {
	Ref<Shortcut> shortcut;
	shortcut.instantiate();
	Array events;
	Ref<InputEventKey> key_event = InputEventKey::create_reference(p_key);
	events.push_back(key_event);
	shortcut->set_events(events);
	p_controller->set_shortcut(p_name, shortcut);
}

static void _press_key(Key p_key, bool p_pressed) {
	Ref<InputEventKey> key_event = InputEventKey::create_reference(p_key);
	key_event->set_pressed(p_pressed);
	Input::get_singleton()->parse_input_event(key_event);
}

static void _clear_keys() {
	_press_key(Key::W, false);
	_press_key(Key::S, false);
	_press_key(Key::A, false);
	_press_key(Key::D, false);
	_press_key(Key::E, false);
	_press_key(Key::Q, false);
}

TEST_CASE("[SceneTree][View3DController] Freelook vertical movement is independent of camera pitch") {
	Ref<View3DController> controller;
	controller.instantiate();

	_setup_shortcut(controller, View3DController::SHORTCUT_FREELOOK_FORWARD, Key::W);
	_setup_shortcut(controller, View3DController::SHORTCUT_FREELOOK_BACKWARDS, Key::S);
	_setup_shortcut(controller, View3DController::SHORTCUT_FREELOOK_LEFT, Key::A);
	_setup_shortcut(controller, View3DController::SHORTCUT_FREELOOK_RIGHT, Key::D);
	_setup_shortcut(controller, View3DController::SHORTCUT_FREELOOK_UP, Key::E);
	_setup_shortcut(controller, View3DController::SHORTCUT_FREELOOK_DOWN, Key::Q);

	controller->set_freelook_enabled(true);
	controller->set_freelook_base_speed(10.0f);
	controller->set_freelook_scheme(View3DController::FREELOOK_DEFAULT);

	const float test_pitches[] = {
		0.0f, // Looking straight ahead
		Math::deg_to_rad(-30.0f), // Looking slightly down
		Math::deg_to_rad(-60.0f), // Looking significantly down
		Math::deg_to_rad(-85.0f), // Looking almost straight down
		Math::deg_to_rad(30.0f), // Looking slightly up
		Math::deg_to_rad(60.0f), // Looking significantly up
	};

	SUBCASE("E (Up) produces purely positive global Y movement regardless of pitch") {
		for (float pitch : test_pitches) {
			_clear_keys();
			_press_key(Key::E, true);

			controller->cursor.x_rot = pitch;
			controller->cursor.y_rot = 0.0f;
			controller->cursor.pos_x = 0.0;
			controller->cursor.pos_y = 0.0;
			controller->cursor.pos_z = 0.0;

			const float delta = 1.0f;
			controller->update_freelook(delta);

			const double expected_y = 10.0; // speed * delta
			CHECK_MESSAGE(Math::is_equal_approx(controller->cursor.pos_y, expected_y, 0.001), "Y movement should equal speed * delta for E (Up)");
			CHECK_MESSAGE(Math::is_zero_approx(controller->cursor.pos_x), "X position must not change when pressing E (Up)");
			CHECK_MESSAGE(Math::is_zero_approx(controller->cursor.pos_z), "Z position must not change when pressing E (Up) - no horizontal drift");
		}
		_clear_keys();
	}

	SUBCASE("Q (Down) produces purely negative global Y movement regardless of pitch") {
		for (float pitch : test_pitches) {
			_clear_keys();
			_press_key(Key::Q, true);

			controller->cursor.x_rot = pitch;
			controller->cursor.y_rot = 0.0f;
			controller->cursor.pos_x = 0.0;
			controller->cursor.pos_y = 0.0;
			controller->cursor.pos_z = 0.0;

			const float delta = 1.0f;
			controller->update_freelook(delta);

			const double expected_y = -10.0; // -speed * delta
			CHECK_MESSAGE(Math::is_equal_approx(controller->cursor.pos_y, expected_y, 0.001), "Y movement should equal -speed * delta for Q (Down)");
			CHECK_MESSAGE(Math::is_zero_approx(controller->cursor.pos_x), "X position must not change when pressing Q (Down)");
			CHECK_MESSAGE(Math::is_zero_approx(controller->cursor.pos_z), "Z position must not change when pressing Q (Down) - no horizontal drift");
		}
		_clear_keys();
	}

	SUBCASE("W/A/S/D movement remains camera-oriented") {
		_clear_keys();
		_press_key(Key::W, true);

		// Pitch down 45 degrees (positive x_rot tilts down in Godot)
		controller->cursor.x_rot = Math::deg_to_rad(45.0f);
		controller->cursor.y_rot = 0.0f;
		controller->cursor.pos_x = 0.0;
		controller->cursor.pos_y = 0.0;
		controller->cursor.pos_z = 0.0;

		controller->update_freelook(1.0f);

		// Forward vector when pitch is down 45 degrees should move negative Z and negative Y
		CHECK(controller->cursor.pos_z < -0.1);
		CHECK(controller->cursor.pos_y < -0.1);
		CHECK_MESSAGE(Math::is_zero_approx(controller->cursor.pos_x), "X position should not change for forward move with zero yaw");

		_clear_keys();
	}

	SUBCASE("Diagonal combinations (W + E, W + Q, A + E, D + E, W + D + E)") {
		_clear_keys();

		// Pitch down 45 degrees
		controller->cursor.x_rot = Math::deg_to_rad(45.0f);
		controller->cursor.y_rot = 0.0f;

		// Test W + E
		_press_key(Key::W, true);
		_press_key(Key::E, true);
		controller->cursor.pos_x = 0.0;
		controller->cursor.pos_y = 0.0;
		controller->cursor.pos_z = 0.0;

		controller->update_freelook(1.0f);

		// Direction is forward + up(0, 1, 0)
		Transform3D camera_transform = controller->to_camera_transform();
		Vector3 expected_forward = camera_transform.basis.xform(Vector3(0, 0, -1));
		Vector3 expected_up = Vector3(0, 1, 0);
		Vector3 expected_dir = expected_forward + expected_up;
		Vector3 expected_motion = expected_dir * 10.0f * 1.0f;

		CHECK(Math::is_equal_approx(controller->cursor.pos_x, (double)expected_motion.x, 0.01));
		CHECK(Math::is_equal_approx(controller->cursor.pos_y, (double)expected_motion.y, 0.01));
		CHECK(Math::is_equal_approx(controller->cursor.pos_z, (double)expected_motion.z, 0.01));

		_clear_keys();

		// Test W + D + E
		_press_key(Key::W, true);
		_press_key(Key::D, true);
		_press_key(Key::E, true);
		controller->cursor.pos_x = 0.0;
		controller->cursor.pos_y = 0.0;
		controller->cursor.pos_z = 0.0;

		controller->update_freelook(1.0f);

		Vector3 expected_right = camera_transform.basis.xform(Vector3(1, 0, 0));
		expected_dir = expected_forward + expected_right + expected_up;
		expected_motion = expected_dir * 10.0f * 1.0f;

		CHECK(Math::is_equal_approx(controller->cursor.pos_x, (double)expected_motion.x, 0.01));
		CHECK(Math::is_equal_approx(controller->cursor.pos_y, (double)expected_motion.y, 0.01));
		CHECK(Math::is_equal_approx(controller->cursor.pos_z, (double)expected_motion.z, 0.01));

		_clear_keys();
	}

	SUBCASE("All Freelook schemes use global Y for vertical motion") {
		const View3DController::FreelookScheme schemes[] = {
			View3DController::FREELOOK_DEFAULT,
			View3DController::FREELOOK_PARTIALLY_AXIS_LOCKED,
			View3DController::FREELOOK_FULLY_AXIS_LOCKED,
		};

		for (View3DController::FreelookScheme scheme : schemes) {
			controller->set_freelook_scheme(scheme);
			_clear_keys();
			_press_key(Key::E, true);

			controller->cursor.x_rot = Math::deg_to_rad(-60.0f);
			controller->cursor.y_rot = 0.0f;
			controller->cursor.pos_x = 0.0;
			controller->cursor.pos_y = 0.0;
			controller->cursor.pos_z = 0.0;

			controller->update_freelook(1.0f);

			CHECK_MESSAGE(Math::is_equal_approx(controller->cursor.pos_y, 10.0, 0.001), "Y movement must be positive speed * delta");
			CHECK_MESSAGE(Math::is_zero_approx(controller->cursor.pos_x), "X position must not change");
			CHECK_MESSAGE(Math::is_zero_approx(controller->cursor.pos_z), "Z position must not change");
		}
		_clear_keys();
	}

	SUBCASE("Toggling freelook_vertical_movement_global setting") {
		controller->set_freelook_scheme(View3DController::FREELOOK_DEFAULT);
		_clear_keys();
		_press_key(Key::E, true);

		// Pitch down 45 degrees
		controller->cursor.x_rot = Math::deg_to_rad(45.0f);
		controller->cursor.y_rot = 0.0f;
		controller->cursor.pos_x = 0.0;
		controller->cursor.pos_y = 0.0;
		controller->cursor.pos_z = 0.0;

		// 1. By default, freelook_vertical_movement_global is true -> pure global Y move
		CHECK(controller->is_freelook_vertical_movement_global() == true);
		controller->update_freelook(1.0f);
		CHECK_MESSAGE(Math::is_equal_approx(controller->cursor.pos_y, 10.0, 0.001), "Y movement must be positive speed * delta");
		CHECK_MESSAGE(Math::is_zero_approx(controller->cursor.pos_z), "Z position must not drift when global Y is enabled");

		// 2. When disabled, Freelook Up follows camera local basis (has Z component when pitched)
		controller->set_freelook_vertical_movement_global(false);
		CHECK(controller->is_freelook_vertical_movement_global() == false);
		controller->cursor.pos_x = 0.0;
		controller->cursor.pos_y = 0.0;
		controller->cursor.pos_z = 0.0;
		controller->update_freelook(1.0f);
		CHECK_MESSAGE(!Math::is_zero_approx(controller->cursor.pos_z), "Z position should drift along camera pitch when global Y is disabled");

		// 3. Re-enabling restores global Y behavior
		controller->set_freelook_vertical_movement_global(true);
		controller->cursor.pos_x = 0.0;
		controller->cursor.pos_y = 0.0;
		controller->cursor.pos_z = 0.0;
		controller->update_freelook(1.0f);
		CHECK_MESSAGE(Math::is_zero_approx(controller->cursor.pos_z), "Z position must not drift when global Y is re-enabled");

		_clear_keys();
	}
}

} // namespace TestView3DController

#endif // _3D_DISABLED
