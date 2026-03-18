#pragma once

#include "test_macros.h"

#include "core/math/math_funcs.h"
#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/math/basis.h"
#include "core/math/quaternion.h"

namespace TestGaussianSplatting {

/**
 * Test suite for validating the view transform pipeline.
 *
 * These tests verify that:
 * 1. Camera transform -> View transform conversion is correct
 * 2. Matrix packing for GPU (column-major) is correct
 * 3. World-space points transform correctly to view space
 * 4. The inverse relationship holds: cam_transform * view_transform ≈ identity
 */

// Tolerance for floating point comparisons
constexpr float TRANSFORM_EPSILON = 1e-5f;

static bool is_near(float a, float b, float epsilon = TRANSFORM_EPSILON) {
	return Math::abs(a - b) < epsilon;
}

static bool is_near_vector(const Vector3 &a, const Vector3 &b, float epsilon = TRANSFORM_EPSILON) {
	return is_near(a.x, b.x, epsilon) && is_near(a.y, b.y, epsilon) && is_near(a.z, b.z, epsilon);
}

static bool is_identity_basis(const Basis &b, float epsilon = TRANSFORM_EPSILON) {
	return is_near(b[0][0], 1.0f, epsilon) && is_near(b[0][1], 0.0f, epsilon) && is_near(b[0][2], 0.0f, epsilon) &&
		   is_near(b[1][0], 0.0f, epsilon) && is_near(b[1][1], 1.0f, epsilon) && is_near(b[1][2], 0.0f, epsilon) &&
		   is_near(b[2][0], 0.0f, epsilon) && is_near(b[2][1], 0.0f, epsilon) && is_near(b[2][2], 1.0f, epsilon);
}

/**
 * Result structure for transform validation
 */
struct TransformValidationResult {
	bool passed = true;
	String error_message;

	// Detailed data for debugging
	Transform3D cam_transform;
	Transform3D view_transform;
	Transform3D product; // cam * view - should be identity
	Vector3 test_world_point;
	Vector3 test_view_point;
	Vector3 expected_view_point;

	void fail(const String &msg) {
		passed = false;
		error_message = msg;
	}
};

/**
 * Test 1: Verify that cam_transform.affine_inverse() produces correct view matrix
 *
 * For a camera at position C with rotation R:
 * - cam_transform.origin = C
 * - cam_transform.basis = R
 * - view_transform = cam_transform.affine_inverse()
 * - cam_transform * view_transform should equal identity
 */
static TransformValidationResult test_affine_inverse_identity() {
	TransformValidationResult result;

	// Create a camera transform: position (5, 3, -10), rotated 45 degrees around Y
	Quaternion rotation = Quaternion(Vector3(0, 1, 0), Math::deg_to_rad(45.0f));
	result.cam_transform = Transform3D(Basis(rotation), Vector3(5.0f, 3.0f, -10.0f));

	// Compute view transform
	result.view_transform = result.cam_transform.affine_inverse();

	// Product should be identity
	result.product = result.cam_transform * result.view_transform;

	// Check if product is identity
	if (!is_near_vector(result.product.origin, Vector3(0, 0, 0), 1e-4f)) {
		result.fail(vformat("cam * view origin != (0,0,0), got (%.6f, %.6f, %.6f)",
			result.product.origin.x, result.product.origin.y, result.product.origin.z));
		return result;
	}

	if (!is_identity_basis(result.product.basis, 1e-4f)) {
		result.fail("cam * view basis != identity");
		return result;
	}

	return result;
}

/**
 * Test 2: Verify world-to-view transformation for a known point
 *
 * A point at the camera's position should transform to origin in view space.
 * A point in front of the camera should have negative Z in view space (Godot convention).
 */
static TransformValidationResult test_world_to_view_transform() {
	TransformValidationResult result;

	// Camera at (0, 2, 5) looking along -Z (default orientation)
	result.cam_transform = Transform3D(Basis(), Vector3(0.0f, 2.0f, 5.0f));
	result.view_transform = result.cam_transform.affine_inverse();

	// Test point at camera position - should become origin
	result.test_world_point = Vector3(0.0f, 2.0f, 5.0f);
	result.test_view_point = result.view_transform.xform(result.test_world_point);
	result.expected_view_point = Vector3(0.0f, 0.0f, 0.0f);

	if (!is_near_vector(result.test_view_point, result.expected_view_point, 1e-4f)) {
		result.fail(vformat("Point at camera position should be origin in view space. Got (%.4f, %.4f, %.4f)",
			result.test_view_point.x, result.test_view_point.y, result.test_view_point.z));
		return result;
	}

	// Test point in front of camera - should have negative Z
	result.test_world_point = Vector3(0.0f, 2.0f, 0.0f); // 5 units in front
	result.test_view_point = result.view_transform.xform(result.test_world_point);

	if (result.test_view_point.z >= 0.0f) {
		result.fail(vformat("Point in front of camera should have negative Z in view space. Got Z=%.4f",
			result.test_view_point.z));
		return result;
	}

	return result;
}

/**
 * Test 3: Verify that rotating camera rotates view space correctly
 *
 * When camera rotates right (positive Y rotation), points that were in front
 * should now appear to the left in view space.
 */
static TransformValidationResult test_camera_rotation_effect() {
	TransformValidationResult result;

	// Camera at origin, rotated 90 degrees to the right (around Y)
	Quaternion rotation = Quaternion(Vector3(0, 1, 0), Math::deg_to_rad(90.0f));
	result.cam_transform = Transform3D(Basis(rotation), Vector3(0.0f, 0.0f, 0.0f));
	result.view_transform = result.cam_transform.affine_inverse();

	// A point at (0, 0, -5) in world space (originally in front of unrotated camera)
	// After camera rotates 90 degrees right, this point should be to the LEFT in view space
	result.test_world_point = Vector3(0.0f, 0.0f, -5.0f);
	result.test_view_point = result.view_transform.xform(result.test_world_point);

	// In view space, X should be positive (point is to our left after we rotated right)
	// Z should be near zero (point is to the side, not in front/behind)
	if (result.test_view_point.x < 0.0f) {
		result.fail(vformat("Point should be to the LEFT (positive X) in view space after camera rotates right. Got X=%.4f",
			result.test_view_point.x));
		return result;
	}

	return result;
}

/**
 * Test 4: Verify GPU matrix packing produces correct column-major layout
 *
 * GLSL mat4 expects column-major storage:
 * float[0-3] = column 0, float[4-7] = column 1, etc.
 */
struct GPUMatrixPackResult {
	bool passed = true;
	String error_message;
	float gpu_matrix[16];
	Transform3D source_transform;

	void fail(const String &msg) {
		passed = false;
		error_message = msg;
	}
};

static void pack_transform_to_gpu_matrix(const Transform3D &t, float out_matrix[16]) {
	// Pack exactly as tile_renderer.cpp does
	for (int column = 0; column < 3; column++) {
		Vector3 column_vec = t.basis.get_column(column);
		out_matrix[column * 4 + 0] = column_vec.x;
		out_matrix[column * 4 + 1] = column_vec.y;
		out_matrix[column * 4 + 2] = column_vec.z;
		out_matrix[column * 4 + 3] = 0.0f;
	}
	out_matrix[12] = t.origin.x;
	out_matrix[13] = t.origin.y;
	out_matrix[14] = t.origin.z;
	out_matrix[15] = 1.0f;
}

static GPUMatrixPackResult test_gpu_matrix_packing() {
	GPUMatrixPackResult result;

	// Create a known transform
	Quaternion rotation = Quaternion(Vector3(0, 1, 0), Math::deg_to_rad(30.0f));
	result.source_transform = Transform3D(Basis(rotation), Vector3(1.0f, 2.0f, 3.0f));

	pack_transform_to_gpu_matrix(result.source_transform, result.gpu_matrix);

	// Verify column 0 matches basis.get_column(0)
	Vector3 col0 = result.source_transform.basis.get_column(0);
	if (!is_near(result.gpu_matrix[0], col0.x) ||
		!is_near(result.gpu_matrix[1], col0.y) ||
		!is_near(result.gpu_matrix[2], col0.z)) {
		result.fail("GPU matrix column 0 doesn't match basis.get_column(0)");
		return result;
	}

	// Verify column 3 (translation) is origin
	if (!is_near(result.gpu_matrix[12], result.source_transform.origin.x) ||
		!is_near(result.gpu_matrix[13], result.source_transform.origin.y) ||
		!is_near(result.gpu_matrix[14], result.source_transform.origin.z)) {
		result.fail("GPU matrix column 3 doesn't match origin");
		return result;
	}

	// Verify M[15] = 1.0 for proper homogeneous coordinates
	if (!is_near(result.gpu_matrix[15], 1.0f)) {
		result.fail("GPU matrix [15] should be 1.0");
		return result;
	}

	return result;
}

/**
 * Test 5: Simulate full transform pipeline and verify result
 *
 * This test simulates what happens in the shader:
 * view_pos = view_matrix * vec4(world_pos, 1.0)
 */
static TransformValidationResult test_full_pipeline_simulation() {
	TransformValidationResult result;

	// Camera setup: at (3, 2, 8) looking toward origin
	Vector3 cam_pos = Vector3(3.0f, 2.0f, 8.0f);
	Vector3 look_at = Vector3(0.0f, 0.0f, 0.0f);
	Vector3 up = Vector3(0.0f, 1.0f, 0.0f);

	// Build camera transform (look_at style)
	Vector3 forward = (look_at - cam_pos).normalized();
	Vector3 right = up.cross(forward).normalized();
	Vector3 cam_up = forward.cross(right);

	Basis cam_basis;
	cam_basis.set_column(0, right);
	cam_basis.set_column(1, cam_up);
	cam_basis.set_column(2, -forward); // Camera looks down -Z

	result.cam_transform = Transform3D(cam_basis, cam_pos);
	result.view_transform = result.cam_transform.affine_inverse();

	// World point at origin
	result.test_world_point = Vector3(0.0f, 0.0f, 0.0f);

	// Pack for GPU
	float gpu_matrix[16];
	pack_transform_to_gpu_matrix(result.view_transform, gpu_matrix);

	// Simulate GPU transform: result = M * v (column-major)
	float vx = result.test_world_point.x;
	float vy = result.test_world_point.y;
	float vz = result.test_world_point.z;
	float vw = 1.0f;

	result.test_view_point.x = gpu_matrix[0]*vx + gpu_matrix[4]*vy + gpu_matrix[8]*vz + gpu_matrix[12]*vw;
	result.test_view_point.y = gpu_matrix[1]*vx + gpu_matrix[5]*vy + gpu_matrix[9]*vz + gpu_matrix[13]*vw;
	result.test_view_point.z = gpu_matrix[2]*vx + gpu_matrix[6]*vy + gpu_matrix[10]*vz + gpu_matrix[14]*vw;

	// Origin should be in front of camera (negative Z in view space)
	if (result.test_view_point.z >= 0.0f) {
		result.fail(vformat("Origin should be in front of camera (negative Z). Got Z=%.4f", result.test_view_point.z));
		return result;
	}

	// Verify using Transform3D.xform gives same result
	Vector3 cpu_view_point = result.view_transform.xform(result.test_world_point);
	if (!is_near_vector(result.test_view_point, cpu_view_point, 1e-4f)) {
		result.fail(vformat("GPU simulation (%.4f,%.4f,%.4f) != CPU xform (%.4f,%.4f,%.4f)",
			result.test_view_point.x, result.test_view_point.y, result.test_view_point.z,
			cpu_view_point.x, cpu_view_point.y, cpu_view_point.z));
		return result;
	}

	return result;
}

/**
 * Run all view transform tests and return summary
 */
static String run_view_transform_tests() {
	String summary = "=== View Transform Pipeline Tests ===\n\n";
	int passed = 0;
	int failed = 0;

	// Test 1
	{
		auto r = test_affine_inverse_identity();
		if (r.passed) {
			summary += "[PASS] Test 1: affine_inverse identity property\n";
			passed++;
		} else {
			summary += vformat("[FAIL] Test 1: %s\n", r.error_message);
			failed++;
		}
	}

	// Test 2
	{
		auto r = test_world_to_view_transform();
		if (r.passed) {
			summary += "[PASS] Test 2: world-to-view transform\n";
			passed++;
		} else {
			summary += vformat("[FAIL] Test 2: %s\n", r.error_message);
			failed++;
		}
	}

	// Test 3
	{
		auto r = test_camera_rotation_effect();
		if (r.passed) {
			summary += "[PASS] Test 3: camera rotation effect\n";
			passed++;
		} else {
			summary += vformat("[FAIL] Test 3: %s\n", r.error_message);
			failed++;
		}
	}

	// Test 4
	{
		auto r = test_gpu_matrix_packing();
		if (r.passed) {
			summary += "[PASS] Test 4: GPU matrix packing\n";
			passed++;
		} else {
			summary += vformat("[FAIL] Test 4: %s\n", r.error_message);
			failed++;
		}
	}

	// Test 5
	{
		auto r = test_full_pipeline_simulation();
		if (r.passed) {
			summary += "[PASS] Test 5: full pipeline simulation\n";
			passed++;
		} else {
			summary += vformat("[FAIL] Test 5: %s\n", r.error_message);
			failed++;
		}
	}

	summary += vformat("\n=== Results: %d passed, %d failed ===\n", passed, failed);

	return summary;
}

// Doctest TEST_CASE wrappers for automatic discovery
TEST_CASE("[GaussianSplatting][ViewTransform] Affine inverse identity property") {
	auto result = test_affine_inverse_identity();
	CHECK_MESSAGE(result.passed, result.error_message.utf8().get_data());
}

TEST_CASE("[GaussianSplatting][ViewTransform] World to view transform") {
	auto result = test_world_to_view_transform();
	CHECK_MESSAGE(result.passed, result.error_message.utf8().get_data());
}

TEST_CASE("[GaussianSplatting][ViewTransform] Camera rotation effect") {
	auto result = test_camera_rotation_effect();
	CHECK_MESSAGE(result.passed, result.error_message.utf8().get_data());
}

TEST_CASE("[GaussianSplatting][ViewTransform] GPU matrix packing") {
	auto result = test_gpu_matrix_packing();
	CHECK_MESSAGE(result.passed, result.error_message.utf8().get_data());
}

TEST_CASE("[GaussianSplatting][ViewTransform] Full pipeline simulation") {
	auto result = test_full_pipeline_simulation();
	CHECK_MESSAGE(result.passed, result.error_message.utf8().get_data());
}

} // namespace TestGaussianSplatting
