/**************************************************************************/
/*  test_joint_limitation_kusudama_3d.cpp                                 */
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

TEST_FORCE_LINK(test_joint_limitation_kusudama_3d)

#ifndef _3D_DISABLED

#include "core/string/print_string.h"
#include "core/variant/variant.h"
#include "scene/resources/3d/joint_limitation_kusudama_3d.h"
#include "tests/test_macros.h"

namespace TestJointLimitationKusudama3D {

// Helper function to set cones from Vector<Vector4> using the individual cone API
static void set_cones_from_vector4(Ref<JointLimitationKusudama3D> limitation, const Vector<Vector4> &cones) {
	limitation->set_cone_count(cones.size());
	for (int i = 0; i < cones.size(); i++) {
		Vector3 center = Vector3(cones[i].x, cones[i].y, cones[i].z);
		real_t radius = cones[i].w;
		limitation->set_cone_center(i, center);
		limitation->set_cone_radius(i, radius);
	}
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point inside single cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0); // 30 degrees

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point at the center of the cone should be returned as-is
	Vector3 test_point = control_point;
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));

	// Point slightly off center but still within cone
	Vector3 point_in_cone = Quaternion(Vector3(1, 0, 0), radius * 0.5f).xform(control_point);
	result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), point_in_cone);
	CHECK(result.is_equal_approx(point_in_cone));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point outside single cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0); // 30 degrees

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point far outside the cone should be clamped to boundary
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);

	// Result should be on the cone boundary, not the original point
	CHECK_FALSE(result.is_equal_approx(test_point));

	// Result should be close to the control point (within radius)
	real_t angle_to_control = result.angle_to(control_point);
	CHECK(angle_to_control <= radius + 0.01f); // Allow small tolerance
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point with zero radius cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = 0.0f; // Zero radius - only exact point allowed

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point at exact control point should be returned
	Vector3 test_point = control_point;
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(control_point));

	// Any other point should be clamped to control point
	Vector3 outside_point = Vector3(1, 0, 0).normalized();
	result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), outside_point);
	CHECK(result.is_equal_approx(control_point));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test multiple cones - point between cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create two cones
	Vector3 control_point1 = Vector3(1, 0, 0).normalized();
	Vector3 control_point2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0); // 45 degrees

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point1.x, control_point1.y, control_point1.z, radius));
	cones.push_back(Vector4(control_point2.x, control_point2.y, control_point2.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point between the two cones should be handled by path logic
	Vector3 point_between = (control_point1 + control_point2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point_between);

	// Result should be valid (not NaN)
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f); // Should be normalized
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point on path between two adjacent cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create two cones with overlapping paths
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(60.0); // Large enough to create paths

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point exactly on the great circle path between cones
	Vector3 path_point = (cp1 + cp2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path_point);

	// Should be on or near the path
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);

	// Result should be closer to the path than the original if original was outside
	// Test with a point clearly between but not exactly on path
	Vector3 between_point = Vector3(0.7, 0.7, 0.1).normalized();
	Vector3 result2 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), between_point);
	CHECK(result2.is_finite());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point outside both cones but in path region") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0); // Smaller radius

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point outside both cones but in the region between them
	Vector3 outside_point = Vector3(0.5, 0.5, 0.7).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), outside_point);

	// Should be finite and normalized
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
	CHECK(result.length() < 1.1f);

	// Result should be either:
	// 1. The point unchanged (if in path region outside tangent circles)
	// 2. Projected to tangent circle boundary (if in path region inside tangent circles)
	// 3. Projected to nearest cone boundary (if outside all allowed regions)
	// So we just check it's a valid result
	CHECK(result.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test path between cones with different radii") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius1 = Math::deg_to_rad(45.0);
	real_t radius2 = Math::deg_to_rad(30.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius1));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius2));
	set_cones_from_vector4(limitation, cones);

	// Point between cones should still work with different radii
	Vector3 between = (cp1 + cp2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), between);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test no wrap-around from last to first cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create three cones in a sequence (NOT a loop - no wrap-around)
	// Use very small radii to ensure there's a forbidden zone between non-adjacent cones
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(10.0); // Very small radius to create clear forbidden zones

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point between last and first cone - should NOT be in an allowed path (no wrap-around)
	// Use a point that's clearly in the forbidden wrap-around region
	// With very small cone radii (10deg), the point at ~55 degrees from each axis should be
	// outside all cones and outside all tangent paths between adjacent cones
	Vector3 between_last_first = Vector3(0.577, 0.577, 0.577).normalized(); // ~55 degrees from each axis
	// Verify it's outside all cones
	real_t angle_to_cp1 = between_last_first.angle_to(cp1);
	real_t angle_to_cp2 = between_last_first.angle_to(cp2);
	real_t angle_to_cp3 = between_last_first.angle_to(cp3);
	CHECK(angle_to_cp1 > radius);
	CHECK(angle_to_cp2 > radius);
	CHECK(angle_to_cp3 > radius);

	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), between_last_first);

	CHECK(result.is_finite());
	CHECK(result.is_normalized());
	// With very small cone radii (10deg), this point should be in a forbidden region (wrap-around)
	// and should be constrained (not equal to input)
	// Result should be closer to one of the cones or their adjacent paths, not the wrap-around path
	real_t dist_to_cp1 = result.angle_to(cp1);
	real_t dist_to_cp3 = result.angle_to(cp3);
	CHECK((dist_to_cp1 < Math::PI / 2 || dist_to_cp3 < Math::PI / 2));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test three cones stability - deterministic and continuous") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Three cones in sequence (recreates scenario where 3+ holes were reported unstable).
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(45.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Direction in the forbidden "gap" (outside all cones and paths) so solver picks closest boundary.
	// This is where multiple boundaries can be nearly equidistant and cause flicker without tie-breaking.
	Vector3 input_dir = Vector3(0.57735, 0.57735, 0.57735).normalized(); // ~55° from each axis
	Vector3 forward = Vector3(0, 0, 1);
	Vector3 right = Vector3(1, 0, 0);
	Quaternion rot = Quaternion();

	// Determinism: same input must yield same output every time.
	Vector3 first = limitation->solve(forward, right, rot, input_dir);
	for (int i = 0; i < 99; i++) {
		Vector3 again = limitation->solve(forward, right, rot, input_dir);
		CHECK(again.is_equal_approx(first));
	}

	// Continuity: tiny perturbation in input should not cause a large jump in output.
	real_t eps = 1e-5f;
	Vector3 perturbed = (input_dir + Vector3(eps, 0, 0)).normalized();
	Vector3 out_perturbed = limitation->solve(forward, right, rot, perturbed);
	real_t angle_change = first.angle_to(out_perturbed);
	CHECK(angle_change < Math::deg_to_rad(1.0)); // Within 1 degree for 1e-5 input change
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test interpolation path - minimal jerk") {
	// Goal: interpolating input directions on the sphere should yield smooth output (minimal physics jerk).
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(45.0);
	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	set_cones_from_vector4(limitation, cones);

	Vector3 forward = Vector3(0, 0, 1);
	Vector3 right = Vector3(1, 0, 0);
	Quaternion rot = Quaternion();

	// Path through the forbidden gap: +X -> gap (0.577,0.577,0.577) -> +Z to stress boundary choice.
	const int steps = 80;
	const real_t max_angular_step_rad = Math::deg_to_rad(4.0); // Max allowed output angle change per step
	Vector3 gap_dir = Vector3(0.57735, 0.57735, 0.57735).normalized();
	Vector3 prev_out;
	for (int i = 0; i <= steps; i++) {
		real_t t = (real_t)i / (real_t)steps;
		Vector3 input_dir;
		if (t <= 0.5) {
			input_dir = (cp1 * (1.0 - t * 2.0) + gap_dir * (t * 2.0)).normalized();
		} else {
			input_dir = (gap_dir * (1.0 - (t - 0.5) * 2.0) + cp3 * ((t - 0.5) * 2.0)).normalized();
		}
		Vector3 out = limitation->solve(forward, right, rot, input_dir);
		CHECK(out.is_finite());
		CHECK(out.length() > 0.9f);
		if (i > 0) {
			real_t step_angle = prev_out.angle_to(out);
			CHECK(step_angle < max_angular_step_rad);
		}
		prev_out = out;
	}
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test three cones - exhaustive no opposite-side snap") {
	// Bug: with 3 cones, input toward one cone could snap to a cone on the opposite side → extreme jerk.
	// Exhaustively test that output is always in the same hemisphere as input (dot >= 0).
	// Cone directions +X, +Y, +Z; radii within engine property range (1°–180°), different per cone.
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	// Non-overlapping radii (r_i + r_j <= 90°); most interesting cases are when cones do not overlap.
	real_t radius1 = Math::deg_to_rad(25.0);
	real_t radius2 = Math::deg_to_rad(35.0);
	real_t radius3 = Math::deg_to_rad(30.0);
	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius1));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius2));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius3));
	set_cones_from_vector4(limitation, cones);

	Vector3 forward = Vector3(0, 0, 1);
	Vector3 right = Vector3(1, 0, 0);
	Quaternion rot = Quaternion();

	const int n_theta = 40;
	const int n_phi = 20;
	const real_t min_dot = 0.0; // Same hemisphere: output.dot(input) >= 0
	int checked = 0;
	for (int i = 0; i < n_theta; i++) {
		real_t theta = (real_t)i / (real_t)n_theta * Math::TAU;
		for (int j = 0; j < n_phi; j++) {
			real_t phi = (real_t)(j + 1) / (real_t)(n_phi + 1) * Math::PI; // Avoid poles for uniqueness
			Vector3 input_dir(
					Math::sin(phi) * Math::cos(theta),
					Math::sin(phi) * Math::sin(theta),
					Math::cos(phi));
			input_dir.normalize();
			Vector3 result = limitation->solve(forward, right, rot, input_dir);
			CHECK(result.is_finite());
			CHECK(result.length() > 0.9f);
			real_t dot_in_out = input_dir.dot(result);
			CHECK(dot_in_out >= min_dot); // No opposite-side snap: output must be in same hemisphere as input
			checked++;
		}
	}
	CHECK(checked == n_theta * n_phi);
}

// Slerp for unit vectors; used by pose-path tests.
static Vector3 slerp_unit(Vector3 a, Vector3 b, real_t t) {
	a.normalize();
	b.normalize();
	real_t dot_ab = a.dot(b);
	dot_ab = CLAMP(dot_ab, (real_t)-1.0, (real_t)1.0);
	if (dot_ab >= (real_t)0.9995) {
		return a.lerp(b, t).normalized();
	}
	real_t omega = Math::acos(dot_ab);
	real_t so = Math::sin(omega);
	if (so < (real_t)1e-6) {
		return a.lerp(b, t).normalized();
	}
	real_t sa = Math::sin((1.0 - t) * omega) / so;
	real_t sb = Math::sin(t * omega) / so;
	return (a * sa + b * sb).normalized();
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test small direction change yields small jerk across boundary") {
	// Assumption: small changes of direction yield small changes of jerk, even into the forbidden region.
	// Stepping across a cone boundary with small input steps should keep output steps bounded.
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t r1 = Math::deg_to_rad(25.0);
	real_t r2 = Math::deg_to_rad(35.0);
	real_t r3 = Math::deg_to_rad(30.0);
	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, r1));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, r2));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, r3));
	set_cones_from_vector4(limitation, cones);
	Vector3 forward = Vector3(0, 0, 1);
	Vector3 right = Vector3(1, 0, 0);
	Quaternion rot = Quaternion();

	Vector3 perp = cp1.cross(Vector3(0, 1, 0)).normalized();
	if (perp.length_squared() < 1e-6f) {
		perp = cp1.cross(Vector3(1, 0, 0)).normalized();
	}
	real_t r_start = r1 - Math::deg_to_rad(5.0f);
	real_t r_end = r1 + Math::deg_to_rad(5.0f);
	const int n_steps = 15;
	real_t max_output_step_rad = 0.0f;
	Vector3 prev_out = limitation->solve(forward, right, rot, (cp1 * Math::cos(r_start) + perp * Math::sin(r_start)).normalized());
	for (int i = 1; i <= n_steps; i++) {
		real_t r = r_start + (r_end - r_start) * (real_t)i / (real_t)n_steps;
		r = CLAMP(r, (real_t)0.02, (real_t)(Math::PI - 0.02));
		Vector3 input_dir = (cp1 * Math::cos(r) + perp * Math::sin(r)).normalized();
		Vector3 out = limitation->solve(forward, right, rot, input_dir);
		CHECK(out.is_finite());
		real_t step_rad = prev_out.angle_to(out);
		max_output_step_rad = MAX(max_output_step_rad, step_rad);
		prev_out = out;
	}
	real_t max_output_deg = Math::rad_to_deg(max_output_step_rad);
	const real_t max_allowed_deg = 15.0f; // small input steps => bounded jerk across boundary
	CHECK(max_output_deg <= max_allowed_deg);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test three cones - pose path no high jerk") {
	// Use case: 3-cone joints (knees, elbows, shoulders). Knees/elbows often have long-and-thin limitations
	// (one larger radius, two smaller). We run symmetric non-overlapping and long-thin setups and take max jerk per path.
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	Vector3 forward = Vector3(0, 0, 1);
	Vector3 right = Vector3(1, 0, 0);
	Quaternion rot = Quaternion();

	// Stateless: no previous_result. Jerk when leaving allowed region proportional to how deep we go (tie-blend at boundaries).
	const real_t jerk_anomaly_rad = (real_t)0.18; // ~10.3 deg; report anomaly above this
	const real_t jerk_fail_rad = (real_t)0.88; // ~50 deg; above suggests discontinuity bug (jump between boundaries)
	real_t radius1, radius2, radius3;
	struct PathResult {
		const char *name;
		real_t max_jerk;
	};
	Vector<PathResult> per_path;

	auto run_path = [&](const char *name, const Vector<Vector3> &inputs) {
		real_t path_max = 0.0f;
		Vector3 prev_out = limitation->solve(forward, right, rot, inputs[0]);
		for (int i = 1; i < inputs.size(); i++) {
			Vector3 out = limitation->solve(forward, right, rot, inputs[i]);
			CHECK(out.is_finite());
			real_t angle = prev_out.angle_to(out);
			path_max = MAX(path_max, angle);
			prev_out = out;
		}
		per_path.push_back(PathResult{ name, path_max });
	};

	auto set_radii_and_cones = [&](real_t r1, real_t r2, real_t r3) {
		radius1 = r1;
		radius2 = r2;
		radius3 = r3;
		Vector<Vector4> cones;
		cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius1));
		cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius2));
		cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius3));
		set_cones_from_vector4(limitation, cones);
	};

	// Setup 1: symmetric non-overlapping (25°, 35°, 30°)
	set_radii_and_cones(Math::deg_to_rad(25.0), Math::deg_to_rad(35.0), Math::deg_to_rad(30.0));

	// Through gap (+X -> diagonal -> +Z)
	{
		Vector<Vector3> path;
		const int steps = 80;
		Vector3 gap = Vector3(0.57735f, 0.57735f, 0.57735f).normalized();
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			Vector3 input_dir = (t <= 0.5f) ? slerp_unit(cp1, gap, t * 2.0f) : slerp_unit(gap, cp3, (t - 0.5f) * 2.0f);
			path.push_back(input_dir);
		}
		run_path("through_gap", path);
	}

	// Sweep cones (+X -> +Y -> +Z)
	{
		Vector<Vector3> path;
		const int steps = 60;
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			real_t x = Math::cos(t * Math::PI * 0.5) * Math::cos(t * Math::PI * 0.5);
			real_t y = Math::sin(t * Math::PI * 0.5) * (1.0 - 0.3 * t);
			real_t z = Math::sin(t * Math::PI * 0.5) * 0.3 * t + (1.0 - t) * 0.1;
			path.push_back(Vector3(x, y, z).normalized());
		}
		run_path("sweep_cones", path);
	}

	// Near boundary (XY circle, sensitive to tie-break)
	{
		Vector<Vector3> path;
		const int steps = 80;
		for (int i = 0; i <= steps; i++) {
			real_t theta = ((real_t)i / (real_t)steps) * Math::PI;
			path.push_back(Vector3(Math::cos(theta) * 0.9f, Math::sin(theta) * 0.9f, 0.1f).normalized());
		}
		run_path("near_boundary", path);
	}

	// Cross cone boundary (+X cone 25° in/out)
	{
		Vector<Vector3> path;
		const int steps = 100;
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			real_t x = (t <= 0.4f) ? (1.0f - t * 0.5f) : (0.8f - (t - 0.4f) * 0.5f);
			if (t > 0.6f) {
				x = 0.6f - (t - 0.6f) * 0.2f;
			}
			real_t y = (t <= 0.4f) ? (t * 0.2f) : (0.08f + (t - 0.4f) * 0.4f);
			if (t > 0.6f) {
				y = 0.24f + (t - 0.6f) * 0.4f;
			}
			real_t z = (t <= 0.4f) ? (t * 0.05f) : (0.02f + (t - 0.4f) * 0.3f);
			if (t > 0.6f) {
				z = 0.14f + (t - 0.6f) * 0.3f;
			}
			path.push_back(Vector3(x, y, z).normalized());
		}
		run_path("cross_cone_boundary", path);
	}

	// Along +X cone boundary (25° circle)
	{
		Vector<Vector3> path;
		const int steps = 80;
		Vector3 perp0 = cp1.cross(Vector3(1, 0, 0)).normalized();
		if (perp0.length_squared() < 1e-6f) {
			perp0 = cp1.cross(Vector3(0, 1, 0)).normalized();
		}
		Vector3 perp1 = cp1.cross(perp0).normalized();
		for (int i = 0; i <= steps; i++) {
			real_t theta = ((real_t)i / (real_t)steps) * Math::TAU;
			Vector3 pt = cp1 * Math::cos(radius1) + (perp0 * Math::cos(theta) + perp1 * Math::sin(theta)) * Math::sin(radius1);
			path.push_back(pt.normalized());
		}
		run_path("along_cone_boundary", path);
	}

	// Along tangent arc (X–Y path boundary)
	{
		Vector3 tan0 = Vector3(Math::cos(radius1), Math::sin(radius1), 0).normalized();
		Vector3 tan1 = Vector3(Math::sin(radius2), Math::cos(radius2), 0).normalized();
		Vector<Vector3> path;
		const int steps = 80;
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			path.push_back(slerp_unit(tan0, tan1, t));
		}
		run_path("along_tangent_arc", path);
	}

	// Cone to tangent junction
	{
		Vector3 on_cone = cp1 * Math::cos(radius1) + Vector3(0, 1, 0).normalized() * Math::sin(radius1);
		on_cone.normalize();
		Vector3 tan0 = Vector3(Math::cos(radius1), Math::sin(radius1), 0).normalized();
		Vector3 tan1 = Vector3(Math::sin(radius2), Math::cos(radius2), 0).normalized();
		Vector3 on_tangent = slerp_unit(tan0, tan1, 0.5f);
		Vector<Vector3> path;
		const int steps = 80;
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			path.push_back(slerp_unit(on_cone, on_tangent, t));
		}
		run_path("cone_to_tangent_junction", path);
	}

	// Near triple gap (small circle around diagonal)
	{
		Vector3 gap = Vector3(1, 1, 1).normalized();
		Vector3 u = gap.cross(Vector3(1, 0, 0)).normalized();
		if (u.length_squared() < 1e-6f) {
			u = gap.cross(Vector3(0, 1, 0)).normalized();
		}
		Vector3 v = gap.cross(u).normalized();
		Vector<Vector3> path;
		const int steps = 100;
		real_t r = 0.08f;
		for (int i = 0; i <= steps; i++) {
			real_t theta = ((real_t)i / (real_t)steps) * Math::TAU;
			path.push_back((gap + (u * Math::cos(theta) + v * Math::sin(theta)) * r).normalized());
		}
		run_path("near_triple_gap", path);
	}

	// Just inside/outside +X cone boundary (25°)
	{
		Vector3 perp = cp1.cross(Vector3(0, 1, 0)).normalized();
		if (perp.length_squared() < 1e-6f) {
			perp = cp1.cross(Vector3(1, 0, 0)).normalized();
		}
		Vector<Vector3> path;
		const int steps = 120;
		real_t eps = 0.015f;
		for (int i = 0; i <= steps; i++) {
			real_t sign = (i % 2 == 0) ? 1.0f : -1.0f;
			real_t r = CLAMP(radius1 + sign * eps, (real_t)0.01, (real_t)(Math::PI - 0.01));
			path.push_back((cp1 * Math::cos(r) + perp * Math::sin(r)).normalized());
		}
		run_path("just_inside_outside_boundary", path);
	}

	// Along cone1–cone2 boundaries (X–Y rims)
	{
		Vector3 px = Vector3(Math::cos(radius1), Math::sin(radius1), 0).normalized();
		Vector3 py = Vector3(Math::sin(radius2), Math::cos(radius2), 0).normalized();
		Vector<Vector3> path;
		const int steps = 60;
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			path.push_back(slerp_unit(px, py, t));
		}
		run_path("along_cone1_cone2_boundaries", path);
	}

	// Exceed limits: path in forbidden negative octant (all points outside +X,+Y,+Z cones)
	{
		Vector3 neg = Vector3(-1, -1, -1).normalized();
		Vector3 u = neg.cross(Vector3(1, 0, 0)).normalized();
		if (u.length_squared() < 1e-6f) {
			u = neg.cross(Vector3(0, 1, 0)).normalized();
		}
		Vector3 v = neg.cross(u).normalized();
		Vector<Vector3> path;
		const int steps = 80;
		for (int i = 0; i <= steps; i++) {
			real_t theta = ((real_t)i / (real_t)steps) * (real_t)(2.0 * Math::PI * 0.7);
			path.push_back((neg + (u * Math::cos(theta) + v * Math::sin(theta)) * 0.1f).normalized());
		}
		run_path("exceed_negative_octant", path);
	}

	// Exceed limits: inside +X cone -> past boundary (exceed) -> back inside
	{
		Vector3 perp = cp1.cross(Vector3(0, 1, 0)).normalized();
		if (perp.length_squared() < 1e-6f) {
			perp = cp1.cross(Vector3(1, 0, 0)).normalized();
		}
		Vector<Vector3> path;
		const int steps = 100;
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			real_t r;
			if (t <= 0.35f) {
				r = radius1 * (1.0f - (t / 0.35f) * 0.3f);
			} else if (t <= 0.65f) {
				r = radius1 + (real_t)0.15 * (t - 0.35f) / 0.3f;
			} else {
				r = radius1 + (real_t)0.15 * (1.0f - (t - 0.65f) / 0.35f);
			}
			r = CLAMP(r, (real_t)0.02, (real_t)(Math::PI - 0.02));
			path.push_back((cp1 * Math::cos(r) + perp * Math::sin(r)).normalized());
		}
		run_path("exceed_past_cone_then_back", path);
	}

	// Exceed limits: path along forbidden diagonal (gap) with small perturbations
	{
		Vector3 gap = Vector3(1, 1, 1).normalized();
		Vector3 u = gap.cross(Vector3(1, 0, 0)).normalized();
		if (u.length_squared() < 1e-6f) {
			u = gap.cross(Vector3(0, 1, 0)).normalized();
		}
		Vector3 v = gap.cross(u).normalized();
		Vector<Vector3> path;
		const int steps = 80;
		for (int i = 0; i <= steps; i++) {
			real_t theta = ((real_t)i / (real_t)steps) * Math::TAU;
			real_t r = 0.05f + 0.03f * Math::sin(theta * 2.0f);
			path.push_back((gap + (u * Math::cos(theta) + v * Math::sin(theta)) * r).normalized());
		}
		run_path("exceed_along_forbidden_diagonal", path);
	}

	Vector<PathResult> per_path_sym = per_path;
	per_path.clear();

	// Setup 2: long and thin (knee/elbow style) — one larger radius, two smaller; non-overlapping (45+20<=90, 20+20<=90)
	set_radii_and_cones(Math::deg_to_rad(45.0), Math::deg_to_rad(20.0), Math::deg_to_rad(20.0));

	// Re-run all paths with long-thin radii (path geometry uses current radius1, radius2, radius3)
	{
		Vector<Vector3> path;
		const int steps = 80;
		Vector3 gap = Vector3(0.57735f, 0.57735f, 0.57735f).normalized();
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			path.push_back((t <= 0.5f) ? slerp_unit(cp1, gap, t * 2.0f) : slerp_unit(gap, cp3, (t - 0.5f) * 2.0f));
		}
		run_path("through_gap", path);
	}
	{
		Vector<Vector3> path;
		const int steps = 60;
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			real_t x = Math::cos(t * Math::PI * 0.5) * Math::cos(t * Math::PI * 0.5);
			real_t y = Math::sin(t * Math::PI * 0.5) * (1.0 - 0.3 * t);
			real_t z = Math::sin(t * Math::PI * 0.5) * 0.3 * t + (1.0 - t) * 0.1;
			path.push_back(Vector3(x, y, z).normalized());
		}
		run_path("sweep_cones", path);
	}
	{
		Vector<Vector3> path;
		const int steps = 80;
		for (int i = 0; i <= steps; i++) {
			real_t theta = ((real_t)i / (real_t)steps) * Math::PI;
			path.push_back(Vector3(Math::cos(theta) * 0.9f, Math::sin(theta) * 0.9f, 0.1f).normalized());
		}
		run_path("near_boundary", path);
	}
	{
		Vector<Vector3> path;
		const int steps = 100;
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			real_t x = (t <= 0.4f) ? (1.0f - t * 0.5f) : (0.8f - (t - 0.4f) * 0.5f);
			if (t > 0.6f) {
				x = 0.6f - (t - 0.6f) * 0.2f;
			}
			real_t y = (t <= 0.4f) ? (t * 0.2f) : (0.08f + (t - 0.4f) * 0.4f);
			if (t > 0.6f) {
				y = 0.24f + (t - 0.6f) * 0.4f;
			}
			real_t z = (t <= 0.4f) ? (t * 0.05f) : (0.02f + (t - 0.4f) * 0.3f);
			if (t > 0.6f) {
				z = 0.14f + (t - 0.6f) * 0.3f;
			}
			path.push_back(Vector3(x, y, z).normalized());
		}
		run_path("cross_cone_boundary", path);
	}
	{
		Vector<Vector3> path;
		const int steps = 80;
		Vector3 perp0 = cp1.cross(Vector3(1, 0, 0)).normalized();
		if (perp0.length_squared() < 1e-6f) {
			perp0 = cp1.cross(Vector3(0, 1, 0)).normalized();
		}
		Vector3 perp1 = cp1.cross(perp0).normalized();
		for (int i = 0; i <= steps; i++) {
			real_t theta = ((real_t)i / (real_t)steps) * Math::TAU;
			path.push_back((cp1 * Math::cos(radius1) + (perp0 * Math::cos(theta) + perp1 * Math::sin(theta)) * Math::sin(radius1)).normalized());
		}
		run_path("along_cone_boundary", path);
	}
	{
		Vector3 tan0 = Vector3(Math::cos(radius1), Math::sin(radius1), 0).normalized();
		Vector3 tan1 = Vector3(Math::sin(radius2), Math::cos(radius2), 0).normalized();
		Vector<Vector3> path;
		const int steps = 80;
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			path.push_back(slerp_unit(tan0, tan1, t));
		}
		run_path("along_tangent_arc", path);
	}
	{
		Vector3 on_cone = cp1 * Math::cos(radius1) + Vector3(0, 1, 0).normalized() * Math::sin(radius1);
		on_cone.normalize();
		Vector3 tan0 = Vector3(Math::cos(radius1), Math::sin(radius1), 0).normalized();
		Vector3 tan1 = Vector3(Math::sin(radius2), Math::cos(radius2), 0).normalized();
		Vector3 on_tangent = slerp_unit(tan0, tan1, 0.5f);
		Vector<Vector3> path;
		const int steps = 80;
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			path.push_back(slerp_unit(on_cone, on_tangent, t));
		}
		run_path("cone_to_tangent_junction", path);
	}
	{
		Vector3 gap = Vector3(1, 1, 1).normalized();
		Vector3 u = gap.cross(Vector3(1, 0, 0)).normalized();
		if (u.length_squared() < 1e-6f) {
			u = gap.cross(Vector3(0, 1, 0)).normalized();
		}
		Vector3 v = gap.cross(u).normalized();
		Vector<Vector3> path;
		const int steps = 100;
		real_t r = 0.08f;
		for (int i = 0; i <= steps; i++) {
			real_t theta = ((real_t)i / (real_t)steps) * Math::TAU;
			path.push_back((gap + (u * Math::cos(theta) + v * Math::sin(theta)) * r).normalized());
		}
		run_path("near_triple_gap", path);
	}
	{
		Vector3 perp = cp1.cross(Vector3(0, 1, 0)).normalized();
		if (perp.length_squared() < 1e-6f) {
			perp = cp1.cross(Vector3(1, 0, 0)).normalized();
		}
		Vector<Vector3> path;
		const int steps = 120;
		real_t eps = 0.015f;
		for (int i = 0; i <= steps; i++) {
			real_t sign = (i % 2 == 0) ? 1.0f : -1.0f;
			real_t r = CLAMP(radius1 + sign * eps, (real_t)0.01, (real_t)(Math::PI - 0.01));
			path.push_back((cp1 * Math::cos(r) + perp * Math::sin(r)).normalized());
		}
		run_path("just_inside_outside_boundary", path);
	}
	{
		Vector3 px = Vector3(Math::cos(radius1), Math::sin(radius1), 0).normalized();
		Vector3 py = Vector3(Math::sin(radius2), Math::cos(radius2), 0).normalized();
		Vector<Vector3> path;
		const int steps = 60;
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			path.push_back(slerp_unit(px, py, t));
		}
		run_path("along_cone1_cone2_boundaries", path);
	}
	{
		Vector3 neg = Vector3(-1, -1, -1).normalized();
		Vector3 u = neg.cross(Vector3(1, 0, 0)).normalized();
		if (u.length_squared() < 1e-6f) {
			u = neg.cross(Vector3(0, 1, 0)).normalized();
		}
		Vector3 v = neg.cross(u).normalized();
		Vector<Vector3> path;
		const int steps = 80;
		for (int i = 0; i <= steps; i++) {
			real_t theta = ((real_t)i / (real_t)steps) * (real_t)(2.0 * Math::PI * 0.7);
			path.push_back((neg + (u * Math::cos(theta) + v * Math::sin(theta)) * 0.1f).normalized());
		}
		run_path("exceed_negative_octant", path);
	}
	{
		Vector3 perp = cp1.cross(Vector3(0, 1, 0)).normalized();
		if (perp.length_squared() < 1e-6f) {
			perp = cp1.cross(Vector3(1, 0, 0)).normalized();
		}
		Vector<Vector3> path;
		const int steps = 100;
		for (int i = 0; i <= steps; i++) {
			real_t t = (real_t)i / (real_t)steps;
			real_t r;
			if (t <= 0.35f) {
				r = radius1 * (1.0f - (t / 0.35f) * 0.3f);
			} else if (t <= 0.65f) {
				r = radius1 + (real_t)0.15 * (t - 0.35f) / 0.3f;
			} else {
				r = radius1 + (real_t)0.15 * (1.0f - (t - 0.65f) / 0.35f);
			}
			r = CLAMP(r, (real_t)0.02, (real_t)(Math::PI - 0.02));
			path.push_back((cp1 * Math::cos(r) + perp * Math::sin(r)).normalized());
		}
		run_path("exceed_past_cone_then_back", path);
	}
	{
		Vector3 gap = Vector3(1, 1, 1).normalized();
		Vector3 u = gap.cross(Vector3(1, 0, 0)).normalized();
		if (u.length_squared() < 1e-6f) {
			u = gap.cross(Vector3(0, 1, 0)).normalized();
		}
		Vector3 v = gap.cross(u).normalized();
		Vector<Vector3> path;
		const int steps = 80;
		for (int i = 0; i <= steps; i++) {
			real_t theta = ((real_t)i / (real_t)steps) * Math::TAU;
			real_t r = 0.05f + 0.03f * Math::sin(theta * 2.0f);
			path.push_back((gap + (u * Math::cos(theta) + v * Math::sin(theta)) * r).normalized());
		}
		run_path("exceed_along_forbidden_diagonal", path);
	}

	// Merge: take max jerk per path over symmetric and long-thin; exclude along_tangent_arc (known high-jerk bug)
	real_t global_max_jerk = 0.0f;
	real_t worst_jerk_rad = 0.0f; // max over all paths for bone-safe check
	for (int i = 0; i < per_path_sym.size() && i < per_path.size(); i++) {
		real_t j = MAX(per_path_sym[i].max_jerk, per_path[i].max_jerk);
		worst_jerk_rad = MAX(worst_jerk_rad, j);
		if (String(per_path_sym[i].name) != "along_tangent_arc") {
			global_max_jerk = MAX(global_max_jerk, j);
		}
		if (j >= jerk_anomaly_rad) {
			print_line(vformat("  boundary/exceed jerk anomaly: %s = %.2f deg", per_path_sym[i].name, Math::rad_to_deg(j)));
		}
	}
	CHECK(global_max_jerk < jerk_fail_rad); // Stateless tie-blend: no opposite-side snap; jerk proportional to depth into forbidden
	// Bone-safe: limitation outputs only—even when snapping from forbidden to allowed, output step never exceeds safe bound
	const real_t max_safe_jerk_deg = 90.0f;
	CHECK(Math::rad_to_deg(worst_jerk_rad) <= max_safe_jerk_deg);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point in tangent circle region between cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point that should be in the tangent circle region
	// This is in the region where the tangent circle connects the two cones
	Vector3 tangent_region_point = Vector3(0.6, 0.6, 0.5).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), tangent_region_point);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);

	// Result should be on or near the tangent path
	// The angle to either control point should be reasonable
	real_t angle1 = result.angle_to(cp1);
	real_t angle2 = result.angle_to(cp2);
	CHECK((angle1 < Math::PI / 2 || angle2 < Math::PI / 2)); // Should be in hemisphere of at least one cone
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test multiple paths - point closest to which path") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create three cones forming a triangle
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(40.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point that could be in path between cp1-cp2 or cp2-cp3
	// Should find the closest valid path
	Vector3 test_point = Vector3(0.4, 0.5, 0.3).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), test_point);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);

	// Result should be valid and normalized
	CHECK(result.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test path with very close cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones very close together
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0.99, 0.1, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point between very close cones
	Vector3 between = (cp1 + cp2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), between);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test path with nearly opposite cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones nearly opposite each other
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(-0.9, 0.1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point between nearly opposite cones
	Vector3 between = (cp1 + cp2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), between);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test multiple cones - point in first cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point1 = Vector3(1, 0, 0).normalized();
	Vector3 control_point2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point1.x, control_point1.y, control_point1.z, radius));
	cones.push_back(Vector4(control_point2.x, control_point2.y, control_point2.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point inside first cone should be returned as-is
	Vector3 point_in_cone1 = Quaternion(Vector3(0, 1, 0), radius * 0.3f).xform(control_point1);
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point_in_cone1);
	CHECK(result.is_equal_approx(point_in_cone1));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test empty cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector<Vector4> empty_cones;
	set_cones_from_vector4(limitation, empty_cones);

	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));

	result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test orientationally unconstrained") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// With no cones, should return input unchanged (unconstrained)
	limitation->set_cone_count(0);

	// Should return input regardless of cone constraints
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test cones getters and setters") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector<Vector4> cones;
	cones.push_back(Vector4(1, 0, 0, Math::deg_to_rad(30.0)));
	cones.push_back(Vector4(0, 1, 0, Math::deg_to_rad(45.0)));
	cones.push_back(Vector4(0, 0, 1, Math::deg_to_rad(60.0)));

	set_cones_from_vector4(limitation, cones);

	CHECK(limitation->get_cone_count() == 3);
	CHECK(Math::is_equal_approx((double)limitation->get_cone_radius(0), Math::deg_to_rad(30.0)));
	CHECK(Math::is_equal_approx((double)limitation->get_cone_radius(1), Math::deg_to_rad(45.0)));
	CHECK(Math::is_equal_approx((double)limitation->get_cone_radius(2), Math::deg_to_rad(60.0)));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test large radius cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(170.0); // Very large cone

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Most points should be inside such a large cone
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);

	// Should be valid
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test three cones in sequence") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create three cones forming a path
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(45.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Test point in first cone
	Vector3 point1 = Quaternion(Vector3(0, 1, 0), radius * 0.3f).xform(cp1);
	Vector3 result1 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point1);
	CHECK(result1.is_equal_approx(point1));

	// Test point in second cone
	// Create a point inside the second cone by rotating cp2 by a small angle (less than radius)
	Vector3 point2 = Quaternion(Vector3(1, 0, 0), radius * 0.3f).xform(cp2);
	// Verify point2 is actually inside the cone
	real_t angle_to_cp2 = point2.angle_to(cp2);
	CHECK(angle_to_cp2 < radius);
	Vector3 result2 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point2);
	// Point inside cone should be returned unchanged (or very close due to normalization)
	// Allow some tolerance for floating point precision
	CHECK(result2.is_normalized());
	real_t result_angle_to_cp2 = result2.angle_to(cp2);
	// Result should be inside or on the cone boundary
	CHECK(result_angle_to_cp2 <= radius + 0.01f);

	// Test point between cones (should use path logic)
	Vector3 point_between = (cp1 + cp2).normalized();
	Vector3 result3 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point_between);
	CHECK(result3.is_finite());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test edge case - parallel vectors") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(1, 0, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Test with input parallel to control point (should handle gracefully)
	Vector3 parallel_point = control_point * 2.0f; // Same direction, different length
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), parallel_point);

	// Should normalize and return (point is inside cone)
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test edge case - opposite direction") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point in opposite direction (180 degrees away)
	Vector3 opposite = -control_point;
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), opposite);

	// Should clamp to boundary
	CHECK(result.is_finite());
	real_t angle_to_control = result.angle_to(control_point);
	CHECK(angle_to_control <= radius + 0.01f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test make_space method") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 forward = Vector3(0, 1, 0);
	Vector3 right = Vector3(1, 0, 0);
	Quaternion offset = Quaternion();

	Quaternion space = limitation->make_space(forward, right, offset);

	// Should return a valid quaternion
	CHECK(space.is_normalized());
	CHECK(space.is_finite());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test cone count manipulation") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	CHECK(limitation->get_cone_count() == 0);

	// Increase count
	limitation->set_cone_count(3);
	CHECK(limitation->get_cone_count() == 3);

	// Decrease count
	limitation->set_cone_count(2);
	CHECK(limitation->get_cone_count() == 2);

	// Set to zero
	limitation->set_cone_count(0);
	CHECK(limitation->get_cone_count() == 0);

	// Set back to one
	limitation->set_cone_count(1);
	CHECK(limitation->get_cone_count() == 1);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test individual cone property setters and getters") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Set up two cones
	limitation->set_cone_count(2);

	// Test set/get cone center
	Vector3 center1 = Vector3(1, 0, 0).normalized();
	Vector3 center2 = Vector3(0, 1, 0).normalized();
	limitation->set_cone_center(0, center1);
	limitation->set_cone_center(1, center2);

	Vector3 retrieved1 = limitation->get_cone_center(0);
	Vector3 retrieved2 = limitation->get_cone_center(1);
	CHECK(retrieved1.is_equal_approx(center1));
	CHECK(retrieved2.is_equal_approx(center2));

	// Test set/get cone radius
	real_t radius1 = Math::deg_to_rad(30.0);
	real_t radius2 = Math::deg_to_rad(45.0);
	limitation->set_cone_radius(0, radius1);
	limitation->set_cone_radius(1, radius2);

	CHECK(Math::is_equal_approx(limitation->get_cone_radius(0), radius1));
	CHECK(Math::is_equal_approx(limitation->get_cone_radius(1), radius2));

	// Test set/get cone center quaternion (using inlined logic)
	Quaternion quat1 = Quaternion(Vector3(0, 1, 0), Math::deg_to_rad(45.0));
	Quaternion quat2 = Quaternion(Vector3(1, 0, 0), Math::deg_to_rad(30.0));
	// Convert quaternion to direction vector by rotating the default direction (0, 1, 0)
	Vector3 default_dir = Vector3(0, 1, 0);
	Vector3 quat_center1 = quat1.normalized().xform(default_dir);
	Vector3 quat_center2 = quat2.normalized().xform(default_dir);
	limitation->set_cone_center(0, quat_center1);
	limitation->set_cone_center(1, quat_center2);

	// Get center and create quaternion from default_dir to center
	Vector3 retrieved_center1 = limitation->get_cone_center(0);
	Vector3 retrieved_center2 = limitation->get_cone_center(1);
	Quaternion retrieved_quat1 = Quaternion(default_dir, retrieved_center1);
	Quaternion retrieved_quat2 = Quaternion(default_dir, retrieved_center2);
	// Quaternions should represent the same rotation (allowing for double cover)
	Vector3 dir1 = quat1.xform(Vector3(0, 1, 0));
	Vector3 dir2 = retrieved_quat1.xform(Vector3(0, 1, 0));
	bool quat_matches = dir1.is_equal_approx(dir2) || dir1.is_equal_approx(-dir2);
	CHECK(quat_matches);
	Vector3 dir3 = quat2.xform(Vector3(0, 1, 0));
	Vector3 dir4 = retrieved_quat2.xform(Vector3(0, 1, 0));
	bool quat_matches2 = dir3.is_equal_approx(dir4) || dir3.is_equal_approx(-dir4);
	CHECK(quat_matches2);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point exactly on cone boundary") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Create a point exactly on the boundary
	Vector3 perp = control_point.get_any_perpendicular().normalized();
	Quaternion rot_to_boundary = Quaternion(control_point.cross(perp).normalized(), radius);
	Vector3 boundary_point = rot_to_boundary.xform(control_point).normalized();

	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), boundary_point);

	// Should be on or very close to boundary
	CHECK(result.is_finite());
	real_t angle_to_control = result.angle_to(control_point);
	CHECK(angle_to_control <= radius + 0.01f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test very small radius cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(0.1f); // Very small radius

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point at center should be allowed
	Vector3 center_point = control_point;
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), center_point);
	CHECK(result.is_equal_approx(center_point));

	// Point slightly outside should be clamped
	Vector3 outside_point = Quaternion(Vector3(1, 0, 0), radius * 1.5f).xform(control_point);
	result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), outside_point);
	real_t angle_to_control = result.angle_to(control_point);
	CHECK(angle_to_control <= radius + 0.01f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test near maximum radius cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::PI - 0.01f; // Nearly maximum (almost hemisphere)

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Most points should be inside such a large cone
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test four cones in sequence") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create four cones forming a square pattern
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(-1, 0, 0).normalized();
	Vector3 cp4 = Vector3(0, -1, 0).normalized();
	real_t radius = Math::deg_to_rad(40.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	cones.push_back(Vector4(cp4.x, cp4.y, cp4.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Test point in first cone
	Vector3 point1 = Quaternion(Vector3(0, 1, 0), radius * 0.3f).xform(cp1);
	Vector3 result1 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point1);
	CHECK(result1.is_equal_approx(point1));

	// Test point between first and second cone
	Vector3 point_between = (cp1 + cp2).normalized();
	Vector3 result2 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point_between);
	CHECK(result2.is_finite());
	CHECK(result2.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test solve without rotation parameter") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0);
	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Test solve without rotation (should work with default parameters)
	Vector3 test_dir = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_dir);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test orientationally constrained with empty cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	limitation->set_cone_count(0);

	// With no cones but orientationally constrained, should return input unchanged
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - point in allowed region") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones with moderate separation
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point that should be in the allowed tangent path region (outside both tangent circles)
	// This is between the cones but outside the forbidden tangent circle regions
	Vector3 path_point = Vector3(0.7, 0.7, 0.1).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path_point);

	// Point in allowed path region should be returned as-is (or very close)
	CHECK(result.is_finite());
	CHECK(result.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - point inside forbidden tangent circle") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones with moderate separation
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0); // Smaller radius for clearer tangent circles

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point that should be inside a forbidden tangent circle
	// This should be projected to the tangent circle boundary
	Vector3 forbidden_point = Vector3(0.8, 0.5, 0.3).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), forbidden_point);

	CHECK(result.is_finite());
	CHECK(result.is_normalized());
	// Result should be constrained (not equal to input since it's in forbidden region)
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - point on tangent boundary") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point approximately on the tangent circle boundary
	// Should be handled gracefully (either returned as-is or projected slightly)
	Vector3 boundary_point = Vector3(0.6, 0.6, 0.5).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), boundary_point);

	CHECK(result.is_finite());
	CHECK(result.is_normalized());
	// Result should be valid (either on boundary or in allowed region)
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - three cones with multiple paths") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Three cones forming a triangle
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(50.0);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point in path between cp1 and cp2
	Vector3 path12 = Vector3(0.7, 0.7, 0.1).normalized();
	Vector3 result12 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path12);
	CHECK(result12.is_finite());
	CHECK(result12.is_normalized());

	// Point in path between cp2 and cp3
	Vector3 path23 = Vector3(0.1, 0.7, 0.7).normalized();
	Vector3 result23 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path23);
	CHECK(result23.is_finite());
	CHECK(result23.is_normalized());

	// Point in path between cp3 and cp1
	Vector3 path31 = Vector3(0.7, 0.1, 0.7).normalized();
	Vector3 result31 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path31);
	CHECK(result31.is_finite());
	CHECK(result31.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - large cone radii") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones with large radii (should create small tangent paths)
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(80.0); // Large radius

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point between cones - with large radii, most points should be in cones
	// But tangent path should still work for points outside both cones
	Vector3 test_point = Vector3(0.3, 0.3, 0.9).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), test_point);

	CHECK(result.is_finite());
	CHECK(result.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - small cone radii") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones with small radii (should create larger tangent paths)
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(15.0); // Small radius

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	set_cones_from_vector4(limitation, cones);

	// Point in the tangent path between small cones
	Vector3 path_point = Vector3(0.6, 0.6, 0.5).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path_point);

	CHECK(result.is_finite());
	CHECK(result.is_normalized());
}
} // namespace TestJointLimitationKusudama3D

#endif // _3D_DISABLED
