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

#include "editor/editor_settings.h"
#include "tests/test_macros.h"

namespace TestNode3DEditorCameraCursor {

TEST_CASE("[Node3DEditorCameraCursor][Editor] Values operator ==") {
	Node3DEditorCameraCursor::Values values1, values2;

	CHECK(values1 == values2);
	values2.position.x += 10.0;
	CHECK(values1 != values2);
	values2 = Node3DEditorCameraCursor::Values();
	values2.eye_position.x += 10.0;
	CHECK(values1 != values2);
	values2 = Node3DEditorCameraCursor::Values();
	values2.x_rot += 10.0;
	CHECK(values1 != values2);
	values2 = Node3DEditorCameraCursor::Values();
	values2.y_rot += 10.0;
	CHECK(values1 != values2);
	values2 = Node3DEditorCameraCursor::Values();
	values2.distance += 10.0;
	CHECK(values1 != values2);
}

TEST_CASE("[Node3DEditorCameraCursor][Editor] Default cursor") {
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

TEST_CASE("[Node3DEditorCameraCursor][Editor] Move") {
	Node3DEditorCameraCursor cursor;

	SUBCASE("Move from original position") {
		cursor.move(Vector3(10.0, 20.0, 30.0));
		CHECK_MESSAGE(
				cursor.get_target_values().position == Vector3(10.0, 20.0, 30.0),
				"Unexpected position after move.");
		CHECK_MESSAGE(
				cursor.get_target_values().eye_position.is_equal_approx(Vector3(11.682942, 21.917702, 33.080605)),
				"Unexpected eye_position ", cursor.get_target_values().eye_position, " after move.");
	}

	SUBCASE("Move again from previous position") {
		cursor.move(Vector3(10.0, 20.0, 30.0));
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
}

TEST_CASE("[Node3DEditorCameraCursor][Editor] Move to") {
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

TEST_CASE("[Node3DEditorCameraCursor][Editor] Orbit to") {
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
			Math::is_equal_approx(cursor.get_target_values().y_rot, (real_t)Math::deg_to_rad(90.0)),
			"Unexpected y_rot after rotate to an angle.");
	CHECK_MESSAGE(
			cursor.get_target_values().eye_position.is_equal_approx(Vector3(96.0, 0.0, 0.0)),
			"Unexpected eye_position ", cursor.get_target_values().eye_position, " after rotate to an angle.");
}

TEST_CASE("[Node3DEditorCameraCursor][Editor] Orbit") {
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

TEST_CASE("[Node3DEditorCameraCursor][Editor] Look to") {
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

TEST_CASE("[Node3DEditorCameraCursor][Editor] Look") {
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

TEST_CASE("[Node3DEditorCameraCursor][Editor] Get and set freelook mode") {
	Node3DEditorCameraCursor cursor;

	SUBCASE("Initial state") {
		CHECK(!cursor.get_freelook_mode());
	}

	SUBCASE("Set to true") {
		cursor.set_freelook_mode(true);
		CHECK(cursor.get_freelook_mode());
	}

	SUBCASE("Set to false") {
		cursor.set_freelook_mode(true);
		cursor.set_freelook_mode(false);
		CHECK(!cursor.get_freelook_mode());
	}
}

TEST_CASE("[Node3DEditorCameraCursor][Editor] Move distance") {
	Node3DEditorCameraCursor cursor;
	cursor.orbit_to(0.0, 0.0);

	SUBCASE("Move the distance from the initial value") {
		cursor.move_distance(2.0);
		CHECK_MESSAGE(
				cursor.get_target_values().position == Vector3(0.0, 0.0, 0.0),
				"Unexpected position ", cursor.get_target_values().position, " after move the distance.");
		CHECK_MESSAGE(
				cursor.get_target_values().eye_position.is_equal_approx(Vector3(0.0, 0.0, 6.0)),
				"Unexpected eye_position ", cursor.get_target_values().eye_position, " after move the distance.");
	}

	SUBCASE("Move the distance again in oposition direction") {
		cursor.move_distance(2.0);
		cursor.move_distance(-4.0);
		CHECK_MESSAGE(
				cursor.get_target_values().eye_position.is_equal_approx(Vector3(0.0, 0.0, 2.0)),
				"Unexpected eye_position ", cursor.get_target_values().eye_position, " after move the distance again.");
	}

	SUBCASE("Move the distance below zero") {
		cursor.move_distance(-5.0);
		CHECK_MESSAGE(
				cursor.get_target_values().eye_position.is_equal_approx(Vector3(0.0, 0.0, 0.0)),
				"Eye position ", cursor.get_target_values().eye_position, " sould be zero because it should clamp the distance to never be smaller than 0.");
	}
}

TEST_CASE("[Node3DEditorCameraCursor][Editor] Move distance to") {
	Node3DEditorCameraCursor cursor;
	cursor.orbit_to(0.0, 0.0);

	SUBCASE("Move the distance to a value") {
		cursor.move_distance_to(2.0);
		CHECK_MESSAGE(
				cursor.get_target_values().position == Vector3(0.0, 0.0, 0.0),
				"Unexpected position ", cursor.get_target_values().position, " after move the distance.");
		CHECK_MESSAGE(
				cursor.get_target_values().eye_position.is_equal_approx(Vector3(0.0, 0.0, 2.0)),
				"Unexpected eye_position ", cursor.get_target_values().eye_position, " after move the distance.");
	}

	SUBCASE("Move the distance to a value below zero") {
		cursor.move_distance_to(-1.0);
		CHECK_MESSAGE(
				cursor.get_target_values().eye_position.is_equal_approx(Vector3(0.0, 0.0, 0.0)),
				"Eye position ", cursor.get_target_values().eye_position, " should be zero because it should clamp the distance to never be smaller than 0.");
	}
}

TEST_CASE("[Node3DEditorCameraCursor][Editor] Move freelook") {
	Node3DEditorCameraCursor cursor;
	cursor.set_freelook_mode(true);
	cursor.orbit_to(0.0, Math::deg_to_rad(90.0));
	cursor.move_to(Vector3(100.0, 200.0, 300.0));
	cursor.move_distance_to(2.0);
	cursor.stop_interpolation(true);
	EditorSettings *editor_settings = memnew(EditorSettings);

	SUBCASE("Move in freelook mode") {
		cursor.move_freelook(Vector3(10.0, 5.0, -1.0), 2.0, 3.0);
		CHECK_MESSAGE(
				cursor.get_target_values().position == Vector3(94.0, 230.0, 360.0),
				"Unexpected position ", cursor.get_target_values().position, " after freelooking move.");
		CHECK_MESSAGE(
				cursor.get_target_values().eye_position == Vector3(92.0, 230.0, 360.0),
				"Unexpected eye_position ", cursor.get_target_values().eye_position, " after freelooking move.");
	}

	SUBCASE("Move in freelook mode with freelook mode disabled") {
		Node3DEditorCameraCursor::Values previous_target_values = cursor.get_target_values();
		cursor.set_freelook_mode(false);
		cursor.move_freelook(Vector3(10.0, 5.0, -1.0), 2.0, 3.0);
		CHECK_MESSAGE(
				cursor.get_target_values().position == previous_target_values.position,
				"Should not move position ", cursor.get_target_values().position, " after disable freelook mode.");
	}

	memdelete(editor_settings);
}

TEST_CASE("[Node3DEditorCameraCursor][Editor] Get camera transform") {
	Node3DEditorCameraCursor cursor;
	cursor.move_to(Vector3(100.0, 0.0, 200.0));
	cursor.orbit_to(0.0, Math::deg_to_rad(90.0));

	SUBCASE("Target and current transform") {
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
	}

	SUBCASE("Current transform after finish interpolation") {
		cursor.stop_interpolation(true);
		Transform3D current_camera_transform = cursor.get_current_camera_transform();
		CHECK_MESSAGE(
				current_camera_transform.origin.is_equal_approx(Vector3(96.0, 0.0, 200.0)),
				"Unexpected current transform origin ", current_camera_transform.origin);
	}

	SUBCASE("Transform in orthogonal mode") {
		cursor.set_orthogonal(10.0, 50.0);
		Transform3D current_camera_transform = cursor.get_current_camera_transform();
		CHECK_MESSAGE(
				current_camera_transform.origin.is_equal_approx(Vector3(80.0, 0.0, 200.0)),
				"Unexpected current transform origin ", current_camera_transform.origin);
	}
}

TEST_CASE("[Node3DEditorCameraCursor][Editor] Set camera transform") {
	Node3DEditorCameraCursor cursor;
	cursor.move_distance(10.0);
	Transform3D transform;
	transform.origin = Vector3(100.0, 0.0, 200.0);
	transform.basis.set_euler(Vector3(0.0, Math::deg_to_rad(90.0), 0.0));

	SUBCASE("Transform in perspective mode") {
		cursor.set_perspective();
		cursor.set_camera_transform(transform);

		CHECK_MESSAGE(
				cursor.get_target_values().position.is_equal_approx(Vector3(96.0, 0.0, 200.0)),
				"Unexpected position ", cursor.get_target_values().position);
		CHECK_MESSAGE(
				cursor.get_target_values().eye_position.is_equal_approx(Vector3(100.0, 0.0, 200.0)),
				"Unexpected eye_position ", cursor.get_target_values().eye_position);
		CHECK_MESSAGE(
				Math::is_equal_approx(cursor.get_target_values().x_rot, (real_t)0.0),
				"Unexpected x_rot ", cursor.get_target_values().x_rot);
		CHECK_MESSAGE(
				Math::is_equal_approx(cursor.get_target_values().y_rot, -Math::deg_to_rad((real_t)90.0)),
				"Unexpected y_rot ", cursor.get_target_values().y_rot);
		CHECK_MESSAGE(
				cursor.get_target_values().distance == 4.0,
				"Should restore default distance.");
	}

	SUBCASE("Transform in orthogonal mode") {
		cursor.set_orthogonal(10.0, 50.0);
		cursor.set_camera_transform(transform);

		CHECK_MESSAGE(
				cursor.get_target_values().position.is_equal_approx(Vector3(80.0, 0.0, 200.0)),
				"Unexpected position ", cursor.get_target_values().position);
		CHECK_MESSAGE(
				cursor.get_target_values().eye_position.is_equal_approx(Vector3(84.0, 0.0, 200.0)),
				"Unexpected eye_position ", cursor.get_target_values().eye_position);
		CHECK_MESSAGE(
				Math::is_equal_approx(cursor.get_target_values().x_rot, (real_t)0.0),
				"Unexpected x_rot ", cursor.get_target_values().x_rot);
		CHECK_MESSAGE(
				Math::is_equal_approx(cursor.get_target_values().y_rot, -Math::deg_to_rad((real_t)90.0)),
				"Unexpected y_rot ", cursor.get_target_values().y_rot);
		CHECK_MESSAGE(
				cursor.get_target_values().distance == 4.0,
				"Should restore default distance.");
	}
}

TEST_CASE("[Node3DEditorCameraCursor][Editor] Toggle perspective / orthogonal") {
	Node3DEditorCameraCursor cursor;
	cursor.move_to(Vector3(100.0, 0.0, 200.0));
	cursor.orbit_to(0.0, Math::deg_to_rad(90.0));

	SUBCASE("Toggle to orthogonal should stop interpolation") {
		cursor.set_orthogonal(10.0, 50.0);
		CHECK(cursor.get_current_values() == cursor.get_target_values());
	}

	SUBCASE("Toggle to perspective should stop interpolation") {
		cursor.set_perspective();
		CHECK(cursor.get_current_values() == cursor.get_target_values());
	}
}

TEST_CASE("[Node3DEditorCameraCursor][Editor] Stop interpolation") {
	Node3DEditorCameraCursor cursor;
	cursor.orbit_to(0.1, 0.2);
	cursor.move_to(Vector3(100.0, 200.0, 300.0));
	cursor.set_fov_scale(2.0);
	Node3DEditorCameraCursor::Values previous_current_values = cursor.get_current_values();
	Node3DEditorCameraCursor::Values previous_target_values = cursor.get_target_values();

	SUBCASE("Before stop the interpolation") {
		CHECK(previous_current_values.position != Vector3(100.0, 200.0, 300.0));
		CHECK(previous_current_values.x_rot != (real_t)0.1);
		CHECK(previous_current_values.y_rot != (real_t)0.2);
		CHECK(previous_current_values.fov_scale != (real_t)2.0);
		CHECK(cursor.get_target_values() != cursor.get_current_values());
	}

	SUBCASE("Stop the interpolation going to the end") {
		cursor.stop_interpolation(true);
		CHECK(cursor.get_current_values() == previous_target_values);
		CHECK(cursor.get_target_values() == cursor.get_current_values());
	}

	SUBCASE("Stop the interpolation keeping current values") {
		cursor.stop_interpolation(false);
		CHECK(cursor.get_current_values() == previous_current_values);
		CHECK(cursor.get_target_values() == cursor.get_current_values());
	}
}

TEST_CASE("[Node3DEditorCameraCursor][Editor] Update interpolation") {
	Node3DEditorCameraCursor cursor;
	Node3DEditorCameraCursor::Values previous_current_values = cursor.get_current_values();

	SUBCASE("Should change nothing if there aren't updates in target values") {
		CHECK(!cursor.update_interpolation(0.01));
		CHECK(previous_current_values == cursor.get_current_values());
	}

	SUBCASE("Should change nothing in freelook mode") {
		cursor.set_freelook_mode(true);
		CHECK(!cursor.update_interpolation(0.01));
		CHECK(previous_current_values == cursor.get_current_values());
	}

	SUBCASE("Should update rotation") {
		cursor.orbit(0.25, 0.25);
		CHECK(cursor.update_interpolation(0.01));
		CHECK(cursor.get_current_values().x_rot > previous_current_values.x_rot);
		CHECK(cursor.get_current_values().x_rot < cursor.get_target_values().x_rot);
		CHECK(cursor.get_current_values().y_rot > previous_current_values.y_rot);
		CHECK(cursor.get_current_values().y_rot < cursor.get_target_values().y_rot);
	}

	SUBCASE("Should update rotation in freelook mode") {
		previous_current_values = cursor.get_current_values();
		cursor.set_freelook_mode(true);
		cursor.look(0.25, 0.25);

		CHECK(cursor.update_interpolation(0.01));
		CHECK(cursor.get_current_values().x_rot > previous_current_values.x_rot);
		CHECK(cursor.get_current_values().x_rot < cursor.get_target_values().x_rot);
		CHECK(cursor.get_current_values().y_rot > previous_current_values.y_rot);
		CHECK(cursor.get_current_values().y_rot < cursor.get_target_values().y_rot);
	}

	SUBCASE("Should return false after reaching target") {
		cursor.stop_interpolation(true);
		CHECK(!cursor.update_interpolation(0.01));
		CHECK(cursor.get_current_values() == cursor.get_target_values());
	}

	SUBCASE("Should update positions") {
		previous_current_values = cursor.get_current_values();
		cursor.move(Vector3(10.0, 10.0, 10.0));
		CHECK(cursor.update_interpolation(0.01));

		CHECK(cursor.get_current_values().position.x > previous_current_values.position.x);
		CHECK(cursor.get_current_values().position.x < cursor.get_target_values().position.x);
		CHECK(cursor.get_current_values().position.y > previous_current_values.position.y);
		CHECK(cursor.get_current_values().position.y < cursor.get_target_values().position.y);
		CHECK(cursor.get_current_values().position.z > previous_current_values.position.z);
		CHECK(cursor.get_current_values().position.z < cursor.get_target_values().position.z);

		CHECK(cursor.get_current_values().eye_position.x > previous_current_values.eye_position.x);
		CHECK(cursor.get_current_values().eye_position.x < cursor.get_target_values().eye_position.x);
		CHECK(cursor.get_current_values().eye_position.y > previous_current_values.eye_position.y);
		CHECK(cursor.get_current_values().eye_position.y < cursor.get_target_values().eye_position.y);
		CHECK(cursor.get_current_values().eye_position.z > previous_current_values.eye_position.z);
		CHECK(cursor.get_current_values().eye_position.z < cursor.get_target_values().eye_position.z);
	}

	SUBCASE("Should update positions in freelook mode") {
		previous_current_values = cursor.get_current_values();
		cursor.set_freelook_mode(true);
		cursor.move(Vector3(10.0, 10.0, 10.0));
		CHECK(cursor.update_interpolation(0.01));

		CHECK(cursor.get_current_values().position.x > previous_current_values.position.x);
		CHECK(cursor.get_current_values().position.x < cursor.get_target_values().position.x);
		CHECK(cursor.get_current_values().position.y > previous_current_values.position.y);
		CHECK(cursor.get_current_values().position.y < cursor.get_target_values().position.y);
		CHECK(cursor.get_current_values().position.z > previous_current_values.position.z);
		CHECK(cursor.get_current_values().position.z < cursor.get_target_values().position.z);

		CHECK(cursor.get_current_values().eye_position.x > previous_current_values.eye_position.x);
		CHECK(cursor.get_current_values().eye_position.x < cursor.get_target_values().eye_position.x);
		CHECK(cursor.get_current_values().eye_position.y > previous_current_values.eye_position.y);
		CHECK(cursor.get_current_values().eye_position.y < cursor.get_target_values().eye_position.y);
		CHECK(cursor.get_current_values().eye_position.z > previous_current_values.eye_position.z);
		CHECK(cursor.get_current_values().eye_position.z < cursor.get_target_values().eye_position.z);
	}
}

} // namespace TestNode3DEditorCameraCursor

#endif // TEST_NODE_3D_EDITOR_CAMERA_CURSOR_H
