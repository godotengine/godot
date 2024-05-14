/**************************************************************************/
/*  test_ik_kusudama_3d.h                                                 */
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

#include "modules/many_bone_ik/src/ik_kusudama_3d.h"
#include "tests/test_macros.h"

#ifndef TEST_IK_KUSUDAMA_3D_H
#define TEST_IK_KUSUDAMA_3D_H

namespace TestIKKusudama3D {
TEST_CASE("[Modules][ManyBoneIK][IKKusudama3D] Verify limit cone containment") {
	Ref<IKKusudama3D> kusudama;
	kusudama.instantiate();

	// Control point for the limit cone (straight up) and 30 degrees cone opening
	Vector3 control_point = Vector3(0, 1, 0);
	real_t radius = Math_PI / 6; // 30 degrees

	Ref<IKOpenCone3D> cone;
	cone.instantiate();
	cone->set_attached_to(kusudama);
	cone->set_tangent_circle_center_next_1(Vector3(0.0f, -1.0f, 0.0f));
	cone->set_tangent_circle_center_next_2(Vector3(0.0f, 1.0f, 0.0f));
	cone->set_radius(MAX(1.0e-38, radius));
	cone->set_control_point(control_point.normalized());

	kusudama->add_open_cone(cone);

	// Assert that Kusudama correctly sets tangent points based on added cones

	// Get the number of limit cones
	TypedArray<IKOpenCone3D> open_cones = kusudama->get_open_cones();
	REQUIRE(open_cones.size() == 1); // Ensure there is exactly one limit cone

	// Initialize boundary information vector with two elements
	Vector<double> bounds;
	bounds.resize(2);
	bounds.write[0] = 0; // Placeholder for distance from boundary
	bounds.write[1] = 0; // Placeholder for index of limit cone or segment exceeded

	// Test a point outside the bounds – should return a different point from input
	Vector3 test_point_outside = Vector3(0.5, 0.5, 1).normalized();
	Vector3 returned_point_outside = kusudama->get_local_point_in_limits(test_point_outside, &bounds);
	CHECK(returned_point_outside != test_point_outside); // Verify adjustment
	CHECK(bounds[0] < 0); // Negative value indicates outside bounds

	// Test a point inside the bounds – should return the same point as input
	Vector3 test_point_inside = Vector3(0, 0.9, 0.1).normalized();
	Vector3 returned_point_inside = kusudama->get_local_point_in_limits(test_point_inside, &bounds);
	CHECK(returned_point_inside == test_point_inside); // Point remains unchanged
	CHECK(bounds[0] >= 0); // Non-negative value indicates inside bounds or on boundary
}

TEST_CASE("[Modules][ManyBoneIK][IKKusudama3D] Adding and retrieving Limit Cones") {
	Ref<IKKusudama3D> kusudama;
	kusudama.instantiate();

	Vector3 point_on_sphere(1, 0, 0); // Unit sphere point
	double radius = Math_PI / 4; // 45 degrees

	Ref<IKOpenCone3D> cone;
	cone.instantiate();
	cone->set_attached_to(kusudama);
	cone->set_tangent_circle_center_next_1(Vector3(0.0f, -1.0f, 0.0f));
	cone->set_tangent_circle_center_next_2(Vector3(0.0f, 1.0f, 0.0f));
	cone->set_radius(MAX(1.0e-38, radius));
	cone->set_control_point(point_on_sphere.normalized());

	kusudama->add_open_cone(cone);

	TypedArray<IKOpenCone3D> open_cones = kusudama->get_open_cones();
	CHECK(open_cones.size() == 1); // Expect one limit cone

	Ref<IKOpenCone3D> retrieved_cone = open_cones[0];
	CHECK(retrieved_cone.is_valid()); // Validate retrieved cone
	CHECK(Math::is_equal_approx(retrieved_cone->get_radius(), radius)); // Radius check
	CHECK(retrieved_cone->get_closest_path_point(Ref<IKOpenCone3D>(), point_on_sphere) == point_on_sphere);
	CHECK(retrieved_cone->get_closest_path_point(retrieved_cone, point_on_sphere) == point_on_sphere); // Check match

	Vector3 different_point_on_sphere(-1, 0, 0); // Opposite sphere point

	Ref<IKOpenCone3D> cone_2;
	cone_2.instantiate();
	cone_2->set_attached_to(kusudama);
	cone_2->set_tangent_circle_center_next_1(Vector3(0.0f, -1.0f, 0.0f));
	cone_2->set_tangent_circle_center_next_2(Vector3(0.0f, 1.0f, 0.0f));
	cone_2->set_radius(MAX(1.0e-38, radius));
	cone_2->set_control_point(different_point_on_sphere.normalized());
	kusudama->add_open_cone(cone_2);

	open_cones = kusudama->get_open_cones();
	CHECK(open_cones.size() == 2); // Now expect two cones

	Ref<IKOpenCone3D> second_retrieved_cone = open_cones[1];
	CHECK(second_retrieved_cone.is_valid()); // Validate second cone
	CHECK(Math::is_equal_approx(second_retrieved_cone->get_radius(), radius)); // Radius check
	CHECK(second_retrieved_cone->get_closest_path_point(Ref<IKOpenCone3D>(), different_point_on_sphere) == different_point_on_sphere);
}

TEST_CASE("[Modules][ManyBoneIK][IKOpenCone3D] get_closest_path_point") {
	Ref<IKOpenCone3D> cone;
	cone.instantiate();
	double radius = Math_PI / 4; // 45 degrees

	cone->set_tangent_circle_center_next_1(Vector3(0.0f, -1.0f, 0.0f));
	cone->set_tangent_circle_center_next_2(Vector3(0.0f, 1.0f, 0.0f));
	cone->set_radius(MAX(1.0e-38, radius)); // Set radius

	Vector3 input_inside(1, 0, 0); // Input vector inside the cone
	Vector3 input_outside(-1, 0, 0); // Input vector outside the cone

	// Test when next is null
	Vector3 result = cone->get_closest_path_point(Ref<IKOpenCone3D>(), input_inside);
	CHECK(!(Math::is_nan(result.x) && Math::is_nan(result.y) && Math::is_nan(result.z))); // Result should be a number
	CHECK(Math::acos(cone->get_control_point().dot(result.normalized())) <= radius); // Result should be inside the cone

	result = cone->get_closest_path_point(Ref<IKOpenCone3D>(), input_outside);
	CHECK(!(Math::is_nan(result.x) && Math::is_nan(result.y) && Math::is_nan(result.z))); // Result should be a number
	CHECK(Math::acos(cone->get_control_point().dot(result.normalized())) <= radius); // Result should be inside the cone

	// Test when next is not null
	Ref<IKOpenCone3D> next;
	next.instantiate();

	next->set_tangent_circle_center_next_1(Vector3(0.0f, -1.0f, 0.0f));
	next->set_tangent_circle_center_next_2(Vector3(0.0f, 1.0f, 0.0f));
	next->set_radius(MAX(1.0e-38, radius)); // Set radius for next cone
	Vector3 control_point_next(0, 1, 0); // Control point for next cone
	next->set_control_point(control_point_next.normalized()); // Set control point for next cone

	result = cone->get_closest_path_point(next, input_inside);
	CHECK(!(Math::is_nan(result.x) && Math::is_nan(result.y) && Math::is_nan(result.z))); // Result should be a number
	CHECK(Math::acos(cone->get_control_point().dot(result.normalized())) <= radius); // Result should be inside the cone
	CHECK(Math::acos(next->get_control_point().dot(result.normalized())) <= radius); // Result should be inside the next cone

	// TODO: If the point is in the tangents of the cone sequence.

	result = cone->get_closest_path_point(next, input_outside);
	CHECK(!(Math::is_nan(result.x) && Math::is_nan(result.y) && Math::is_nan(result.z))); // Result should be a number
	CHECK(Math::acos(cone->get_control_point().dot(result.normalized())) <= radius); // Result should be inside the cone
	CHECK(Math::acos(next->get_control_point().dot(result.normalized())) <= radius); // Result should be inside the next cone

	// TODO: If the point is in the tangents of the cone sequence.
}

TEST_CASE("[Modules][ManyBoneIK][IKKusudama3D] Verify limit cone removal") {
	Ref<IKKusudama3D> kusudama;
	kusudama.instantiate();

	// Add a couple of limit cones
	Vector3 first_control_point = Vector3(1, 0, 0);
	real_t first_radius = Math_PI / 4; // 45 degrees

	Ref<IKOpenCone3D> cone_3;
	cone_3.instantiate();
	cone_3->set_attached_to(kusudama);
	cone_3->set_tangent_circle_center_next_1(Vector3(0.0f, -1.0f, 0.0f));
	cone_3->set_tangent_circle_center_next_2(Vector3(0.0f, 1.0f, 0.0f));
	cone_3->set_radius(MAX(1.0e-38, first_radius));
	cone_3->set_control_point(first_control_point.normalized());

	kusudama->add_open_cone(cone_3);

	Vector3 second_control_point = Vector3(0, 1, 0);
	real_t second_radius = Math_PI / 6; // 30 degrees

	Ref<IKOpenCone3D> cone_4;
	cone_4.instantiate();
	cone_4->set_attached_to(kusudama);
	cone_4->set_tangent_circle_center_next_1(Vector3(0.0f, -1.0f, 0.0f));
	cone_4->set_tangent_circle_center_next_2(Vector3(0.0f, 1.0f, 0.0f));
	cone_4->set_radius(MAX(1.0e-38, second_radius));
	cone_4->set_control_point(second_control_point.normalized());

	kusudama->add_open_cone(cone_4);

	// Initial checks (expected two limit cones)
	TypedArray<IKOpenCone3D> open_cones = kusudama->get_open_cones();
	REQUIRE(open_cones.size() == 2);

	// Re-check limit cones
	open_cones = kusudama->get_open_cones();

	// Remove the first limit cone
	kusudama->remove_open_cone(open_cones[0]);

	// Re-check limit cones
	open_cones = kusudama->get_open_cones();
	CHECK(open_cones.size() == 1); // Only one limit cone should be left
	Ref<IKOpenCone3D> open_cone = open_cones[0];
	CHECK(open_cone->get_control_point() == second_control_point); // Ensure the remaining cone is the correct one
}

TEST_CASE("[Modules][ManyBoneIK][IKKusudama3D] Check limit cones clear functionality") {
	Ref<IKKusudama3D> kusudama;
	kusudama.instantiate();

	Ref<IKOpenCone3D> cone_5;
	cone_5.instantiate();
	cone_5->set_attached_to(kusudama);
	cone_5->set_tangent_circle_center_next_1(Vector3(0.0f, -1.0f, 0.0f));
	cone_5->set_tangent_circle_center_next_2(Vector3(0.0f, 1.0f, 0.0f));
	cone_5->set_radius(MAX(1.0e-38, Math_PI / 4));
	cone_5->set_control_point(Vector3(1, 0, 0).normalized());

	kusudama->add_open_cone(cone_5); // 45 degrees

	Ref<IKOpenCone3D> cone_6;
	cone_6.instantiate();
	cone_6->set_attached_to(kusudama);
	cone_6->set_tangent_circle_center_next_1(Vector3(0.0f, -1.0f, 0.0f));
	cone_6->set_tangent_circle_center_next_2(Vector3(0.0f, 1.0f, 0.0f));
	cone_6->set_radius(MAX(1.0e-38, Math_PI / 6));
	cone_6->set_control_point(Vector3(0, 1, 0).normalized());
	kusudama->add_open_cone(cone_6); // 30 degrees

	Ref<IKOpenCone3D> cone_7;
	cone_7.instantiate();
	cone_7->set_attached_to(kusudama);
	cone_7->set_tangent_circle_center_next_1(Vector3(0.0f, -1.0f, 0.0f));
	cone_7->set_tangent_circle_center_next_2(Vector3(0.0f, 1.0f, 0.0f));
	cone_7->set_radius(MAX(1.0e-38, Math_PI / 3));
	cone_7->set_control_point(Vector3(0, 1, 0).normalized());
	kusudama->add_open_cone(cone_7); // 60 degrees

	// Initial checks (three limit cones expected)
	TypedArray<IKOpenCone3D> open_cones = kusudama->get_open_cones();
	REQUIRE(open_cones.size() == 3);

	kusudama->clear_open_cones();

	// Re-check limit cones - there should be none
	open_cones = kusudama->get_open_cones();
	CHECK(open_cones.size() == 0); // Expect no limit cones to remain
}
} // namespace TestIKKusudama3D

#endif // TEST_IK_KUSUDAMA_3D_H
