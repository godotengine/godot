/**************************************************************************/
/*  test_camera_3d.h                                                      */
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

#ifndef TEST_CAMERA_3D_H
#define TEST_CAMERA_3D_H

#include "scene/3d/camera_3d.h"
#include "scene/main/window.h"
#include "tests/test_macros.h"

namespace TestCamera3D {

TEST_CASE("[Camera][SceneTree] Camera 3D Relative Position Tests") {
	Camera3D *camera = memnew(Camera3D);
	SceneTree::get_singleton()->get_root()->add_child(camera);

	Vector3 camera_origin = camera->get_camera_transform().get_origin();
	Basis camera_basis = camera->get_camera_transform().basis;

	// Test is_position_behind
	CHECK_MESSAGE(camera->is_position_behind(camera_origin + camera_basis[2]), "A point directly in front of the camera should be behind the camera.");
	CHECK_FALSE_MESSAGE(camera->is_position_behind(camera_origin - camera_basis[2]), "A point directly behind the camera should be behind the camera.");

	// Test is_position_in_frustum
	CHECK_MESSAGE(camera->is_position_in_frustum(camera_origin - camera_basis[2] * (1 + camera->get_near())), "A point directly in front of the camera should be inside the frustum.");
	CHECK_FALSE_MESSAGE(camera->is_position_in_frustum(camera_origin - camera_basis[2] * camera->get_far()), "A point on the camera's far plane should not be considered inside the frustum.");

	memdelete(camera);
}

TEST_CASE("[Camera][SceneTree] Camera 3D Projection Tests") {
	Camera3D *camera = memnew(Camera3D);
	SceneTree::get_singleton()->get_root()->add_child(camera);
	Vector2 viewport_size = SceneTree::get_singleton()->get_root()->get_size();

	// Get the names of enum values as human-readable strings.
	List<StringName> projection_constant_names;
	ClassDB::get_enum_constants(Camera3D::get_class_static(), "ProjectionType", &projection_constant_names);

	// Iterate over projection types for each test.
	for (int projection_int = 0; projection_int < Camera3D::ProjectionType::PROJECTION_TYPE_MAX; projection_int++) {
		Camera3D::ProjectionType projection_type = (Camera3D::ProjectionType)projection_int;
		camera->set_projection(projection_type);

		bool frustum = true; // Test whether projected points are within the camera's frustum.
		bool inverse = true; // Test whether projection and unprojection are inverse operations.

		for (int i = 0; i < 10; i++) {
			float z_depth = Math::random(camera->get_near(), camera->get_far());
			Vector2 projection_point(viewport_size.x * Math::randf(), viewport_size.y * Math::randf());
			Vector3 projected_point = camera->project_position(projection_point, z_depth);

			if (!camera->is_position_in_frustum(projected_point)) {
				frustum = false;
			}
			Vector2 unprojected_point = camera->unproject_position(projected_point);
			// Fudging precision is needed here, because the values will never be *that* close to each other.
			bool x_approx = projection_point.x == doctest::Approx(unprojected_point.x);
			bool y_approx = projection_point.y == doctest::Approx(unprojected_point.y);
			if (!(x_approx && y_approx)) {
				inverse = false;
			}
		}
		CHECK_MESSAGE(frustum, vformat("Points on screen projected within the camera's near and far distances with projection mode %s should be inside the camera's frustum.", projection_constant_names[projection_int]));
		CHECK_MESSAGE(inverse, vformat("Points projected then unprojected with projection mode %s should be equal to the originals.", projection_constant_names[projection_int]));
	}

	memdelete(camera);
}

} //namespace TestCamera3D

#endif // TEST_CAMERA_3D_H
