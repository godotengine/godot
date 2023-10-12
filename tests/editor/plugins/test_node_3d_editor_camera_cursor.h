/**************************************************************************/
/*  test_node_3d_editor_camera_cursor.h                                   */
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

#ifndef TEST_NODE_3D_EDITOR_CAMERA_CURSOR_H
#define TEST_NODE_3D_EDITOR_CAMERA_CURSOR_H

#include "editor/plugins/node_3d_editor_camera_cursor.h"

#include "tests/test_macros.h"

namespace TestNode3DEditorCameraCursor {

TEST_CASE("[Node3DEditorCameraCursor] Default cursor") {
	Node3DEditorCameraCursor cursor;

	CHECK_MESSAGE(
		cursor.get_current_values().position == cursor.get_target_values().position,
		"Default cursor should have equal target and current position.");
	CHECK_MESSAGE(
		cursor.get_current_values().eye_position == cursor.get_target_values().eye_position,
		"Default cursor should have equal target and current position.");
	CHECK_MESSAGE(
		cursor.get_current_values().x_rot == cursor.get_target_values().x_rot,
		"Default cursor should have equal target and current x_rot.");
	CHECK_MESSAGE(
		cursor.get_current_values().y_rot == cursor.get_target_values().y_rot,
		"Default cursor should have equal target and current y_rot.");
	CHECK_MESSAGE(
		cursor.get_current_values().distance == cursor.get_target_values().distance,
		"Default cursor should have equal target and current distance.");
	CHECK_MESSAGE(
		cursor.get_current_values().fov_scale == cursor.get_target_values().fov_scale,
		"Default cursor should have equal target and current fov_scale.");
	CHECK_MESSAGE(
		cursor.get_current_values().position == Vector3(0.0, 0.0, 0.0),
		"Initial position should be zero.");
	CHECK_MESSAGE(
		cursor.get_current_values().eye_position.is_equal_approx(Vector3(1.682942, 1.917702, 3.080605)),
		"Unexpected initial eye_position ", cursor.get_current_values().eye_position);
	CHECK_MESSAGE(
		cursor.get_current_values().distance == 4,
		"Unexpected initial distance.");
	CHECK_MESSAGE(
		cursor.get_current_values().fov_scale == 1.0,
		"Unexpected initial fov_scale.");
	CHECK_MESSAGE(
		cursor.get_current_values().x_rot == 0.5,
		"Unexpected initial x_rot.");
	CHECK_MESSAGE(
		cursor.get_current_values().y_rot == -0.5,
		"Unexpected initial y_rot.");
}

TEST_CASE("[Node3DEditorCameraCursor] Move") {
	Node3DEditorCameraCursor cursor;
	cursor.move(Vector3(10.0, 20.0, 30.0));

	CHECK_MESSAGE(
		cursor.get_target_values().position == Vector3(10.0, 20.0, 30.0),
		"Unexpected position after move.");
	CHECK_MESSAGE(
		cursor.get_target_values().eye_position.is_equal_approx(Vector3(11.682942, 21.917702, 33.080605)),
		"Unexpected eye_position ", cursor.get_target_values().eye_position, " after move.");

	cursor.move(Vector3(100.0, 100.0, 100.0));

	CHECK_MESSAGE(
		cursor.get_target_values().position == Vector3(110.0, 120.0, 130.0),
		"Unexpected position after move for the second time.");
	CHECK_MESSAGE(
		cursor.get_target_values().eye_position.is_equal_approx(Vector3(111.682942, 121.917702, 133.080605)),
		"Unexpected eye_position ", cursor.get_target_values().eye_position, " after move for the second time.");

	CHECK_MESSAGE(
		cursor.get_current_values().position == Vector3(0.0, 0.0, 0.0),
		"Should move only the target values.");
}

TEST_CASE("[Node3DEditorCameraCursor] Move to") {
	Node3DEditorCameraCursor cursor;
	cursor.move(Vector3(10.0, 20.0, 30.0));
	cursor.move_to(Vector3(100.0, 200.0, 300.0));

	CHECK_MESSAGE(
		cursor.get_target_values().position == Vector3(100.0, 200.0, 300.0),
		"Unexpected position after move to a position.");
	CHECK_MESSAGE(
		cursor.get_target_values().eye_position.is_equal_approx(Vector3(101.682942, 201.917702, 303.080605)),
		"Unexpected eye_position ", cursor.get_target_values().eye_position, " after move to a position.");
}

TEST_CASE("[Node3DEditorCameraCursor] Orbit to") {
	Node3DEditorCameraCursor cursor;
	cursor.move_to(Vector3(100.0, 0.0, 0.0));
	cursor.orbit_to(0.0, Math::deg_to_rad(90.0));

	CHECK_MESSAGE(
		cursor.get_target_values().position == Vector3(100.0, 0.0, 0.0),
		"Unexpected position after rotate to an angle.");
	CHECK_MESSAGE(
		cursor.get_target_values().x_rot == 0.0,
		"Unexpected x_rot after rotate to an angle.");
	CHECK_MESSAGE(
		Math::is_equal_approx(cursor.get_target_values().y_rot, (real_t) Math::deg_to_rad(90.0)),
		"Unexpected y_rot after rotate to an angle.");
	CHECK_MESSAGE(
		cursor.get_target_values().eye_position.is_equal_approx(Vector3(96.0, 0.0, 0.0)),
		"Unexpected eye_position ", cursor.get_target_values().eye_position, " after rotate to an angle.");
}

TEST_CASE("[Node3DEditorCameraCursor] Orbit") {
	Node3DEditorCameraCursor cursor;
	cursor.move_to(Vector3(100.0, 0.0, 0.0));
	cursor.orbit_to(Math::deg_to_rad(-45.0), Math::deg_to_rad(30.0));
	cursor.orbit(Math::deg_to_rad(45.0), Math::deg_to_rad(60.0));

	CHECK_MESSAGE(
		cursor.get_target_values().position == Vector3(100.0, 0.0, 0.0),
		"Unexpected position after rotatee.");
	CHECK_MESSAGE(
		cursor.get_target_values().x_rot == 0.0,
		"Unexpected x_rot after rotate.");
	CHECK_MESSAGE(
		Math::is_equal_approx(cursor.get_target_values().y_rot, (real_t)Math::deg_to_rad(90.0)),
		"Unexpected y_rot after rotate.");
	CHECK_MESSAGE(
		cursor.get_target_values().eye_position.is_equal_approx(Vector3(96.0, 0.0, 0.0)),
		"Unexpected eye_position ", cursor.get_target_values().eye_position, " after rotate.");
}

TEST_CASE("[Node3DEditorCameraCursor] Look to") {
	Node3DEditorCameraCursor cursor;
	cursor.orbit_to(0.0, 0.0);
	cursor.move_to(Vector3(100.0, 0.0, 0.0));
	cursor.look_to(0.0, Math::deg_to_rad(90.0));

	CHECK_MESSAGE(
		cursor.get_target_values().eye_position == Vector3(100.0, 0.0, 4.0),
		"Unexpected eye_position after rotate to an angle.");
	CHECK_MESSAGE(
		cursor.get_target_values().x_rot == 0.0,
		"Unexpected x_rot after rotate to an angle.");
	CHECK_MESSAGE(
		Math::is_equal_approx(cursor.get_target_values().y_rot, (real_t)Math::deg_to_rad(90.0)),
		"Unexpected y_rot after rotate to an angle.");
	CHECK_MESSAGE(
		cursor.get_target_values().position.is_equal_approx(Vector3(104.0, 0.0, 4.0)),
		"Unexpected position ", cursor.get_target_values().position, " after rotate to an angle.");
}

TEST_CASE("[Node3DEditorCameraCursor] Look") {
	Node3DEditorCameraCursor cursor;
	cursor.orbit_to(0.0, 0.0);
	cursor.move_to(Vector3(100.0, 0.0, 0.0));
	cursor.look(0.0, Math::deg_to_rad(90.0));

	CHECK_MESSAGE(
		cursor.get_target_values().eye_position == Vector3(100.0, 0.0, 4.0),
		"Unexpected position after rotatee.");
	CHECK_MESSAGE(
		cursor.get_target_values().x_rot == 0.0,
		"Unexpected x_rot after rotate.");
	CHECK_MESSAGE(
		Math::is_equal_approx(cursor.get_target_values().y_rot, (real_t)Math::deg_to_rad(90.0)),
		"Unexpected y_rot after rotate.");
	CHECK_MESSAGE(
		cursor.get_target_values().position.is_equal_approx(Vector3(104.0, 0.0, 4.0)),
		"Unexpected position ", cursor.get_target_values().position, " after rotate.");
}

TEST_CASE("[Node3DEditorCameraCursor] Move distance") {
	Node3DEditorCameraCursor cursor;
	cursor.orbit_to(0.0, 0.0);
	cursor.move_distance(2.0);

	CHECK_MESSAGE(
		cursor.get_target_values().position == Vector3(0.0, 0.0, 0.0),
		"Unexpected position ", cursor.get_target_values().position, " after move the distance.");
	CHECK_MESSAGE(
		cursor.get_target_values().eye_position.is_equal_approx(Vector3(0.0, 0.0, 6.0)),
		"Unexpected eye_position ", cursor.get_target_values().eye_position, " after move the distance.");

	cursor.move_distance(-4.0);

	CHECK_MESSAGE(
		cursor.get_target_values().eye_position.is_equal_approx(Vector3(0.0, 0.0, 2.0)),
		"Unexpected eye_position ", cursor.get_target_values().eye_position, " after move the distance again.");

	cursor.move_distance(-3.0);

	CHECK_MESSAGE(
		cursor.get_target_values().eye_position.is_equal_approx(Vector3(0.0, 0.0, 0.0)),
		"Eye position ", cursor.get_target_values().eye_position, " sould be zero because it should clamp the distance to never be smaller than 0.");
}

TEST_CASE("[Node3DEditorCameraCursor] Move distance to") {
	Node3DEditorCameraCursor cursor;
	cursor.orbit_to(0.0, 0.0);
	cursor.move_distance_to(2.0);

	CHECK_MESSAGE(
		cursor.get_target_values().position == Vector3(0.0, 0.0, 0.0),
		"Unexpected position ", cursor.get_target_values().position, " after move the distance.");
	CHECK_MESSAGE(
		cursor.get_target_values().eye_position.is_equal_approx(Vector3(0.0, 0.0, 2.0)),
		"Unexpected eye_position ", cursor.get_target_values().eye_position, " after move the distance.");

	cursor.move_distance_to(-1.0);

	CHECK_MESSAGE(
		cursor.get_target_values().eye_position.is_equal_approx(Vector3(0.0, 0.0, 0.0)),
		"Eye position ", cursor.get_target_values().eye_position, " sould be zero because it should clamp the distance to never be smaller than 0.");
}

TEST_CASE("[Node3DEditorCameraCursor] Move freelook to") {
	Node3DEditorCameraCursor cursor;
	cursor.set_freelook_mode(true);
	cursor.orbit_to(0.0, Math::deg_to_rad(90.0));
	cursor.move_to(Vector3(100.0, 200.0, 300.0));
	cursor.move_distance_to(2.0);
	cursor.stop_interpolation(true);

	cursor.move_freelook(Vector3(10.0, 5.0, -1.0), 2.0, 3.0);

	CHECK_MESSAGE(
		cursor.get_target_values().position == Vector3(94.0, 230.0, 360.0),
		"Unexpected position ", cursor.get_target_values().position, " after freelooking move.");
	CHECK_MESSAGE(
		cursor.get_target_values().eye_position == Vector3(92.0, 230.0, 360.0),
		"Unexpected eye_position ", cursor.get_target_values().eye_position, " after freelooking move.");

	cursor.stop_interpolation(true);
	cursor.set_freelook_mode(false);
	cursor.move_freelook(Vector3(10.0, 5.0, -1.0), 2.0, 3.0);
	CHECK_MESSAGE(
		cursor.get_target_values().position == Vector3(94.0, 230.0, 360.0),
		"Should not move position ", cursor.get_target_values().position, " after disable freelook mode.");
}

TEST_CASE("[Node3DEditorCameraCursor] Get camera transform") {
	Node3DEditorCameraCursor cursor;
	cursor.move_to(Vector3(100.0, 0.0, 200.0));
	cursor.orbit_to(0.0, Math::deg_to_rad(90.0));
	Transform3D target_camera_transform = cursor.get_target_camera_transform();
	Transform3D current_camera_transform = cursor.get_current_camera_transform();

	CHECK_MESSAGE(
		target_camera_transform.origin.is_equal_approx(Vector3(96.0, 0.0, 200.0)),
		"Unexpected transform origin ", target_camera_transform.origin);
	CHECK_MESSAGE(
		target_camera_transform.basis.get_euler().is_equal_approx(Vector3(0.0, -Math::deg_to_rad(90.0), 0.0)),
		"Unexpected transform rotation ", target_camera_transform.basis.get_euler());
	CHECK_MESSAGE(
		current_camera_transform.origin.is_equal_approx(Vector3(1.682942, 1.917702, 3.080605)),
		"Current transform ", current_camera_transform.origin, " not expected be equal to the target transform.");

	cursor.stop_interpolation(true);
	current_camera_transform = cursor.get_current_camera_transform();

	CHECK_MESSAGE(
		current_camera_transform.origin.is_equal_approx(Vector3(96.0, 0.0, 200.0)),
		"Unexpected current transform origin ", current_camera_transform.origin);
}

TEST_CASE("[Node3DEditorCameraCursor] Set camera transform") {
	Transform3D transform;
	transform.origin = Vector3(100.0, 0.0, 200.0);
	transform.basis.set_euler(Vector3(0.0, Math::deg_to_rad(90.0), 0.0));
	Node3DEditorCameraCursor cursor;
	cursor.move_distance(10.0);
	cursor.set_camera_transform(transform);

	CHECK_MESSAGE(
		cursor.get_target_values().position.is_equal_approx(Vector3(96.0, 0.0, 200.0)),
		"Unexpected position ", cursor.get_target_values().position);
	CHECK_MESSAGE(
		cursor.get_target_values().eye_position.is_equal_approx(Vector3(100.0, 0.0, 200.0)),
		"Unexpected eye_position ", cursor.get_target_values().eye_position);
	CHECK_MESSAGE(
		Math::is_equal_approx(cursor.get_target_values().x_rot, (real_t) 0.0),
		"Unexpected x_rot ", cursor.get_target_values().x_rot);
	CHECK_MESSAGE(
		Math::is_equal_approx(cursor.get_target_values().y_rot, -Math::deg_to_rad((real_t)90.0)),
		"Unexpected y_rot ", cursor.get_target_values().y_rot);
	CHECK_MESSAGE(
		cursor.get_target_values().distance == 4.0,
		"Should restore default distance.");
}

TEST_CASE("[Node3DEditorCameraCursor] Stop interpolation") {
	Node3DEditorCameraCursor cursor;
	cursor.orbit_to(0.1, 0.2);
	cursor.move_to(Vector3(100.0, 200.0, 300.0));
	cursor.set_fov_scale(2.0);

	CHECK(cursor.get_current_values().position != Vector3(100.0, 200.0, 300.0));
	CHECK(cursor.get_current_values().x_rot != (real_t)0.1);
	CHECK(cursor.get_current_values().y_rot != (real_t)0.2);
	CHECK(cursor.get_current_values().fov_scale != (real_t)2.0);
	CHECK(cursor.get_target_values().position == Vector3(100.0, 200.0, 300.0));
	CHECK(cursor.get_target_values().x_rot == (real_t)0.1);
	CHECK(cursor.get_target_values().y_rot == (real_t)0.2);
	CHECK(cursor.get_target_values().fov_scale == (real_t)2.0);

	cursor.stop_interpolation(true);
	CHECK(cursor.get_current_values().position == Vector3(100.0, 200.0, 300.0));
	CHECK(cursor.get_current_values().x_rot == (real_t)0.1);
	CHECK(cursor.get_current_values().y_rot == (real_t)0.2);
	CHECK(cursor.get_current_values().fov_scale == (real_t)2.0);
	CHECK(cursor.get_target_values().position == Vector3(100.0, 200.0, 300.0));
	CHECK(cursor.get_target_values().x_rot == (real_t)0.1);
	CHECK(cursor.get_target_values().y_rot == (real_t)0.2);
	CHECK(cursor.get_target_values().fov_scale == (real_t)2.0);

	cursor.orbit_to(0.15, 0.25);
	cursor.move_to(Vector3(1000.0, 2000.0, 3000.0));
	cursor.set_fov_scale(3.0);
	cursor.stop_interpolation(false);
	CHECK(cursor.get_current_values().position == Vector3(100.0, 200.0, 300.0));
	CHECK(cursor.get_current_values().x_rot == (real_t)0.1);
	CHECK(cursor.get_current_values().y_rot == (real_t)0.2);
	CHECK(cursor.get_current_values().fov_scale == (real_t)2.0);
	CHECK(cursor.get_target_values().position == Vector3(100.0, 200.0, 300.0));
	CHECK(cursor.get_target_values().x_rot == (real_t)0.1);
	CHECK(cursor.get_target_values().y_rot == (real_t)0.2);
	CHECK(cursor.get_target_values().fov_scale == (real_t)2.0);
}

} // namespace TestNode3DEditorCameraCursor

#endif // TEST_NODE_3D_EDITOR_CAMERA_CURSOR_H
