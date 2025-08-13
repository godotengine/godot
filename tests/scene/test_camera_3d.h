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

#pragma once

#include "scene/3d/camera_3d.h"
#include "scene/main/viewport.h"
#include "scene/main/window.h"

#include "tests/test_macros.h"

TEST_CASE("[SceneTree][Camera3D] Getters and setters") {
	Camera3D *test_camera = memnew(Camera3D);

	SUBCASE("Cull mask") {
		constexpr int cull_mask = (1 << 5) | (1 << 7) | (1 << 9);
		constexpr int set_enable_layer = 3;
		constexpr int set_disable_layer = 5;
		test_camera->set_cull_mask(cull_mask);
		CHECK(test_camera->get_cull_mask() == cull_mask);
		test_camera->set_cull_mask_value(set_enable_layer, true);
		CHECK(test_camera->get_cull_mask_value(set_enable_layer));
		test_camera->set_cull_mask_value(set_disable_layer, false);
		CHECK_FALSE(test_camera->get_cull_mask_value(set_disable_layer));
	}

	SUBCASE("Attributes") {
		Ref<CameraAttributes> attributes = memnew(CameraAttributes);
		test_camera->set_attributes(attributes);
		CHECK(test_camera->get_attributes() == attributes);
		Ref<CameraAttributesPhysical> physical_attributes = memnew(CameraAttributesPhysical);
		test_camera->set_attributes(physical_attributes);
		CHECK(test_camera->get_attributes() == physical_attributes);
	}

	SUBCASE("Camera frustum properties") {
		constexpr float depth_near = 0.2f;
		constexpr float depth_far = 995.0f;
		constexpr float fov = 120.0f;
		constexpr float size = 7.0f;
		constexpr float h_offset = 1.1f;
		constexpr float v_offset = -1.6f;
		const Vector2 frustum_offset(5, 7);
		test_camera->set_near(depth_near);
		CHECK(test_camera->get_near() == depth_near);
		test_camera->set_far(depth_far);
		CHECK(test_camera->get_far() == depth_far);
		test_camera->set_fov(fov);
		CHECK(test_camera->get_fov() == fov);
		test_camera->set_size(size);
		CHECK(test_camera->get_size() == size);
		test_camera->set_h_offset(h_offset);
		CHECK(test_camera->get_h_offset() == h_offset);
		test_camera->set_v_offset(v_offset);
		CHECK(test_camera->get_v_offset() == v_offset);
		test_camera->set_frustum_offset(frustum_offset);
		CHECK(test_camera->get_frustum_offset() == frustum_offset);
		test_camera->set_keep_aspect_mode(Camera3D::KeepAspect::KEEP_HEIGHT);
		CHECK(test_camera->get_keep_aspect_mode() == Camera3D::KeepAspect::KEEP_HEIGHT);
		test_camera->set_keep_aspect_mode(Camera3D::KeepAspect::KEEP_WIDTH);
		CHECK(test_camera->get_keep_aspect_mode() == Camera3D::KeepAspect::KEEP_WIDTH);
	}

	SUBCASE("Projection mode") {
		test_camera->set_projection(Camera3D::ProjectionType::PROJECTION_ORTHOGONAL);
		CHECK(test_camera->get_projection() == Camera3D::ProjectionType::PROJECTION_ORTHOGONAL);
		test_camera->set_projection(Camera3D::ProjectionType::PROJECTION_PERSPECTIVE);
		CHECK(test_camera->get_projection() == Camera3D::ProjectionType::PROJECTION_PERSPECTIVE);
	}

	SUBCASE("Helper setters") {
		constexpr float fov = 90.0f, size = 6.0f;
		constexpr float near1 = 0.1f, near2 = 0.5f;
		constexpr float far1 = 1001.0f, far2 = 1005.0f;
		test_camera->set_perspective(fov, near1, far1);
		CHECK(test_camera->get_projection() == Camera3D::ProjectionType::PROJECTION_PERSPECTIVE);
		CHECK(test_camera->get_near() == near1);
		CHECK(test_camera->get_far() == far1);
		CHECK(test_camera->get_fov() == fov);
		test_camera->set_orthogonal(size, near2, far2);
		CHECK(test_camera->get_projection() == Camera3D::ProjectionType::PROJECTION_ORTHOGONAL);
		CHECK(test_camera->get_near() == near2);
		CHECK(test_camera->get_far() == far2);
		CHECK(test_camera->get_size() == size);
	}

	SUBCASE("Frustum planes") {
		constexpr real_t sqrt2_half = 1.41421356237 / 2;

		constexpr real_t fov = 90.0, size = 6.0;
		constexpr real_t near1 = 0.1, near2 = 0.5;
		constexpr real_t far1 = 11.0, far2 = 15.0;

		SubViewport *mock_viewport = memnew(SubViewport);

		mock_viewport->set_size(Vector2i(512, 512)); // aspect 1
		SceneTree::get_singleton()->get_root()->add_child(mock_viewport);
		mock_viewport->add_child(test_camera);

		Vector<Plane> planes;

		test_camera->set_perspective(fov, near1, far1);
		planes = test_camera->get_frustum();
		CHECK(planes[Projection::PLANE_NEAR].normalized().is_equal_approx(Plane(0, 0, 1, -near1)));
		CAPTURE(far1); // FIXME
		CAPTURE(planes[Projection::PLANE_FAR]); // FIXME
		CAPTURE(planes[Projection::PLANE_FAR].normalized()); // FIXME
		CHECK(planes[Projection::PLANE_FAR].normalized().is_equal_approx(Plane(0, 0, -1, far1)));
		CHECK(planes[Projection::PLANE_LEFT].normalized().is_equal_approx(Plane(-sqrt2_half, 0, sqrt2_half, 0)));
		CHECK(planes[Projection::PLANE_TOP].normalized().is_equal_approx(Plane(0, sqrt2_half, sqrt2_half, 0)));
		CHECK(planes[Projection::PLANE_RIGHT].normalized().is_equal_approx(Plane(sqrt2_half, 0, sqrt2_half, 0)));
		CHECK(planes[Projection::PLANE_BOTTOM].normalized().is_equal_approx(Plane(0, -sqrt2_half, sqrt2_half, 0)));

		test_camera->set_orthogonal(size, near2, far2);
		planes = test_camera->get_frustum();
		CHECK(planes[Projection::PLANE_NEAR].normalized().is_equal_approx(Plane(0, 0, 1, -near2)));
		CHECK(planes[Projection::PLANE_FAR].normalized().is_equal_approx(Plane(0, 0, -1, far2)));
		CHECK(planes[Projection::PLANE_LEFT].normalized().is_equal_approx(Plane(-1, 0, 0, size / 2)));
		CHECK(planes[Projection::PLANE_TOP].normalized().is_equal_approx(Plane(0, 1, 0, size / 2)));
		CHECK(planes[Projection::PLANE_RIGHT].normalized().is_equal_approx(Plane(1, 0, 0, size / 2)));
		CHECK(planes[Projection::PLANE_BOTTOM].normalized().is_equal_approx(Plane(0, -1, 0, size / 2)));

		mock_viewport->remove_child(test_camera);
		memdelete(mock_viewport);
	}

	SUBCASE("Frustum points") {
		constexpr real_t fov = 90.0, size = 6.0;
		constexpr real_t near1 = 0.1, near2 = 0.5;
		constexpr real_t far1 = 11.0, far2 = 15.0;
		const real_t tan_fov = Math::tan(Math::deg_to_rad(fov / 2));

		SubViewport *mock_viewport = memnew(SubViewport);

		mock_viewport->set_size(Vector2i(512, 512)); // aspect 1
		SceneTree::get_singleton()->get_root()->add_child(mock_viewport);
		mock_viewport->add_child(test_camera);

		Vector<Vector3> points;

		test_camera->set_perspective(fov, near1, far1);
		points = test_camera->get_near_plane_points();
		CHECK(points[0].is_equal_approx(Vector3())); // CENTER
		CHECK(points[1].is_equal_approx(Vector3(-tan_fov, tan_fov, -1) * near1)); // NEAR, LEFT, TOP
		CHECK(points[2].is_equal_approx(Vector3(-tan_fov, -tan_fov, -1) * near1)); // NEAR, LEFT, BOTTOM
		CHECK(points[3].is_equal_approx(Vector3(tan_fov, tan_fov, -1) * near1)); // NEAR, RIGHT, TOP
		CHECK(points[4].is_equal_approx(Vector3(tan_fov, -tan_fov, -1) * near1)); // NEAR, RIGHT, BOTTOM

		test_camera->set_orthogonal(size, near2, far2);
		points = test_camera->get_near_plane_points();
		CHECK(points[0].is_equal_approx(Vector3())); // CENTER
		CHECK(points[1].is_equal_approx(Vector3(-size / 2, size / 2, -near2))); //NEAR, LEFT, TOP
		CHECK(points[2].is_equal_approx(Vector3(-size / 2, -size / 2, -near2))); //NEAR, LEFT, BOTTOM
		CHECK(points[3].is_equal_approx(Vector3(size / 2, size / 2, -near2))); //NEAR, RIGHT, TOP
		CHECK(points[4].is_equal_approx(Vector3(size / 2, -size / 2, -near2))); //NEAR, RIGHT, BOTTOM

		// Overriding the viewport's 2d size should not influence the result

		mock_viewport->set_size_2d_override(mock_viewport->get_size() * 2);

		test_camera->set_perspective(fov, near1, far1);
		points = test_camera->get_near_plane_points();
		CHECK(points[0].is_equal_approx(Vector3())); // CENTER
		CHECK(points[1].is_equal_approx(Vector3(-tan_fov, tan_fov, -1) * near1)); // NEAR, LEFT, TOP
		CHECK(points[2].is_equal_approx(Vector3(-tan_fov, -tan_fov, -1) * near1)); // NEAR, LEFT, BOTTOM
		CHECK(points[3].is_equal_approx(Vector3(tan_fov, tan_fov, -1) * near1)); // NEAR, RIGHT, TOP
		CHECK(points[4].is_equal_approx(Vector3(tan_fov, -tan_fov, -1) * near1)); // NEAR, RIGHT, BOTTOM

		test_camera->set_orthogonal(size, near2, far2);
		points = test_camera->get_near_plane_points();
		CHECK(points[0].is_equal_approx(Vector3())); // CENTER
		CHECK(points[1].is_equal_approx(Vector3(-size / 2, size / 2, -near2))); //NEAR, LEFT, TOP
		CHECK(points[2].is_equal_approx(Vector3(-size / 2, -size / 2, -near2))); //NEAR, LEFT, BOTTOM
		CHECK(points[3].is_equal_approx(Vector3(size / 2, size / 2, -near2))); //NEAR, RIGHT, TOP
		CHECK(points[4].is_equal_approx(Vector3(size / 2, -size / 2, -near2))); //NEAR, RIGHT, BOTTOM
		mock_viewport->remove_child(test_camera);
		memdelete(mock_viewport);
	}

	SUBCASE("Doppler tracking") {
		test_camera->set_doppler_tracking(Camera3D::DopplerTracking::DOPPLER_TRACKING_IDLE_STEP);
		CHECK(test_camera->get_doppler_tracking() == Camera3D::DopplerTracking::DOPPLER_TRACKING_IDLE_STEP);
		test_camera->set_doppler_tracking(Camera3D::DopplerTracking::DOPPLER_TRACKING_PHYSICS_STEP);
		CHECK(test_camera->get_doppler_tracking() == Camera3D::DopplerTracking::DOPPLER_TRACKING_PHYSICS_STEP);
		test_camera->set_doppler_tracking(Camera3D::DopplerTracking::DOPPLER_TRACKING_DISABLED);
		CHECK(test_camera->get_doppler_tracking() == Camera3D::DopplerTracking::DOPPLER_TRACKING_DISABLED);
	}

	memdelete(test_camera);
}

TEST_CASE("[SceneTree][Camera3D] Position queries") {
	// Cameras need a viewport to know how to compute their frustums, so we make a fake one here.
	Camera3D *test_camera = memnew(Camera3D);
	SubViewport *mock_viewport = memnew(SubViewport);
	// 4:2.
	mock_viewport->set_size(Vector2(400, 200));
	SceneTree::get_singleton()->get_root()->add_child(mock_viewport);
	mock_viewport->add_child(test_camera);
	test_camera->set_keep_aspect_mode(Camera3D::KeepAspect::KEEP_WIDTH);
	REQUIRE_MESSAGE(test_camera->is_current(), "Camera3D should be made current upon entering tree.");

	SUBCASE("Orthogonal projection") {
		test_camera->set_projection(Camera3D::ProjectionType::PROJECTION_ORTHOGONAL);
		// The orthogonal case is simpler, so we test a more random position + rotation combination here.
		// For the other cases we'll use zero translation and rotation instead.
		test_camera->set_global_position(Vector3(1, 2, 3));
		test_camera->look_at(Vector3(-4, 5, 1));
		// Width = 5, Aspect Ratio = 400 / 200 = 2, so Height is 2.5.
		test_camera->set_orthogonal(5.0f, 0.5f, 1000.0f);
		const Basis basis = test_camera->get_global_basis();
		// Subtract near so offset starts from the near plane.
		const Vector3 offset1 = basis.xform(Vector3(-1.5f, 3.5f, 0.2f - test_camera->get_near()));
		const Vector3 offset2 = basis.xform(Vector3(2.0f, -0.5f, -0.6f - test_camera->get_near()));
		const Vector3 offset3 = basis.xform(Vector3(-3.0f, 1.0f, -0.6f - test_camera->get_near()));
		const Vector3 offset4 = basis.xform(Vector3(-2.0f, 1.5f, -0.6f - test_camera->get_near()));
		const Vector3 offset5 = basis.xform(Vector3(0, 0, 10000.0f - test_camera->get_near()));

		SUBCASE("is_position_behind") {
			CHECK(test_camera->is_position_behind(test_camera->get_global_position() + offset1));
			CHECK_FALSE(test_camera->is_position_behind(test_camera->get_global_position() + offset2));

			SUBCASE("h/v offset should have no effect on the result of is_position_behind") {
				test_camera->set_h_offset(-11.0f);
				test_camera->set_v_offset(22.1f);
				CHECK(test_camera->is_position_behind(test_camera->get_global_position() + offset1));
				test_camera->set_h_offset(4.7f);
				test_camera->set_v_offset(-3.0f);
				CHECK_FALSE(test_camera->is_position_behind(test_camera->get_global_position() + offset2));
			}
			// Reset h/v offsets.
			test_camera->set_h_offset(0);
			test_camera->set_v_offset(0);
		}

		SUBCASE("is_position_in_frustum") {
			// If the point is behind the near plane, it is outside the camera frustum.
			// So offset1 is not in frustum.
			CHECK_FALSE(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset1));
			// If |right| > 5 / 2 or |up| > 2.5 / 2, the point is outside the camera frustum.
			// So offset2 is in frustum and offset3 and offset4 are not.
			CHECK(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset2));
			CHECK_FALSE(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset3));
			CHECK_FALSE(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset4));
			// offset5 is beyond the far plane, so it is not in frustum.
			CHECK_FALSE(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset5));

			// Overriding the viewport's 2d size should not influence the result

			mock_viewport->set_size_2d_override(mock_viewport->get_size() / 10);
			CHECK_FALSE(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset1));
			CHECK(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset2));
			CHECK_FALSE(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset3));
			CHECK_FALSE(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset4));
			CHECK_FALSE(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset5));

			mock_viewport->set_size_2d_override(mock_viewport->get_size() * 10);
			CHECK_FALSE(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset1));
			CHECK(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset2));
			CHECK_FALSE(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset3));
			CHECK_FALSE(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset4));
			CHECK_FALSE(test_camera->is_position_in_frustum(test_camera->get_global_position() + offset5));
		}
	}

	SUBCASE("Perspective projection") {
		test_camera->set_projection(Camera3D::ProjectionType::PROJECTION_PERSPECTIVE);
		// Camera at origin, looking at +Z.
		test_camera->set_global_position(Vector3(0, 0, 0));
		test_camera->set_global_rotation(Vector3(0, 0, 0));
		// Keep width, so horizontal fov = 120.
		// Since the near plane distance is 1,
		// with trig we know the near plane's width is 2 * sqrt(3), so its height is sqrt(3).
		test_camera->set_perspective(120.0f, 1.0f, 1000.0f);

		SUBCASE("is_position_behind") {
			CHECK_FALSE(test_camera->is_position_behind(Vector3(0, 0, -1.5f)));
			CHECK(test_camera->is_position_behind(Vector3(2, 0, -0.2f)));
		}

		SUBCASE("is_position_in_frustum") {
			CHECK(test_camera->is_position_in_frustum(Vector3(-1.3f, 0, -1.1f)));
			CHECK_FALSE(test_camera->is_position_in_frustum(Vector3(2, 0, -1.1f)));
			CHECK(test_camera->is_position_in_frustum(Vector3(1, 0.5f, -1.1f)));
			CHECK_FALSE(test_camera->is_position_in_frustum(Vector3(1, 1, -1.1f)));
			CHECK(test_camera->is_position_in_frustum(Vector3(0, 0, -1.5f)));
			CHECK_FALSE(test_camera->is_position_in_frustum(Vector3(0, 0, -0.5f)));
		}
	}

	memdelete(test_camera);
	memdelete(mock_viewport);
}

// This template enables testing different combinations of Camera3D's KeepAspect, Viewport's size and size_2d_override in a programmatic way
// 1/ It exploits the fact that an XY-sized viewport with keep HEIGHT is the 45° mirror of an YX-sized viewport with keep WIDTH :
//    If aspect is KEEP_WIDTH, Extents::size and Extents::remap_xy() mirror X and Y
// 2/ If width_override is not 0, Extents::remap_xy() scales Vector2s to keep them at same relative position as if there were no override
//
// These 2 effects combined allow testing multiple combinations with the same input and reference values :
// - Create a TEST_CASE_TEMPLATE and set all the template parameter combinations you want to test
// - Use T::size, T::keep_aspect, T::is_2d_override and T::size_override to setup your Viewport and Camera3D accordingly
// - In your CHECKs, wrap all the Vector2s and Vector3s that depend linearly on the viewport's size with T::remap_xy()
//
// See below test cases for concrete examples

template <Camera3D::KeepAspect aspect, int width, int height, int width_override>
struct Extents {
	static constexpr Camera3D::KeepAspect keep_aspect = aspect;
	static constexpr bool is_2d_override = width_override != 0;
	static constexpr float override_scale = is_2d_override ? float(width_override) / float(width) : 1;
	static const Vector2 size;
	static const Vector2 size_override;

	static Vector3 remap_xy(Vector3 vec) {
		return keep_aspect == Camera3D::KeepAspect::KEEP_WIDTH ? Vector3(Math::abs(vec.y) * SIGN(vec.x), Math::abs(vec.x) * SIGN(vec.y), vec.z) : vec;
	}

	static Vector2 remap_xy(Vector2 vec) {
		return override_scale * (keep_aspect == Camera3D::KeepAspect::KEEP_WIDTH ? Vector2(Math::abs(vec.y) * SIGN(vec.x), Math::abs(vec.x) * SIGN(vec.y)) : vec);
	}
};

template <Camera3D::KeepAspect aspect, int width, int height, int width_override>
const Vector2 Extents<aspect, width, height, width_override>::size = aspect == Camera3D::KeepAspect::KEEP_WIDTH ? Vector2(height, width) : Vector2(width, height);

template <Camera3D::KeepAspect aspect, int width, int height, int width_override>
const Vector2 Extents<aspect, width, height, width_override>::size_override = is_2d_override ? size * float(width_override) / float(width) : Vector2();

TEST_CASE_TEMPLATE("[SceneTree][Camera3D] Project/Unproject position", T,
		Extents<Camera3D::KeepAspect::KEEP_HEIGHT, 400, 200, 0>,
		Extents<Camera3D::KeepAspect::KEEP_HEIGHT, 400, 200, 200>,
		Extents<Camera3D::KeepAspect::KEEP_WIDTH, 400, 200, 0>,
		Extents<Camera3D::KeepAspect::KEEP_WIDTH, 400, 200, 100>) {
	// Cameras need a viewport to know how to compute their frustums, so we make a fake one here.
	Camera3D *test_camera = memnew(Camera3D);
	SubViewport *mock_viewport = memnew(SubViewport);
	mock_viewport->set_size(T::size);
	SceneTree::get_singleton()->get_root()->add_child(mock_viewport);
	mock_viewport->add_child(test_camera);
	test_camera->set_global_position(Vector3(0, 0, 0));
	test_camera->set_global_rotation(Vector3(0, 0, 0));
	test_camera->set_keep_aspect_mode(T::keep_aspect);
	mock_viewport->set_size_2d_override(T::size_override);
	mock_viewport->set_size_2d_override_stretch(T::is_2d_override);

	CAPTURE(T::size);
	CAPTURE(T::keep_aspect);
	CAPTURE(T::is_2d_override);
	CAPTURE(T::size_override);

	SUBCASE("project_position") {
		SUBCASE("Orthogonal projection") {
			test_camera->set_orthogonal(5.0f, 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->project_position(T::remap_xy(Vector2(200, 100)), 0.5f).is_equal_approx(T::remap_xy(Vector3(0, 0, -0.5f))));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(200, 100)), test_camera->get_far()).is_equal_approx(T::remap_xy(Vector3(0, 0, -test_camera->get_far()))));
			// Top left.
			CHECK(test_camera->project_position(T::remap_xy(Vector2(0, 0)), 1.5f).is_equal_approx(T::remap_xy(Vector3(-5.0f, 2.5f, -1.5f))));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(0, 0)), test_camera->get_near()).is_equal_approx(T::remap_xy(Vector3(-5.0f, 2.5f, -test_camera->get_near()))));
			// Bottom right.
			CHECK(test_camera->project_position(T::remap_xy(Vector2(400, 200)), 5.0f).is_equal_approx(T::remap_xy(Vector3(5.0f, -2.5f, -5.0f))));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(400, 200)), test_camera->get_far()).is_equal_approx(T::remap_xy(Vector3(5.0f, -2.5f, -test_camera->get_far()))));
		}

		SUBCASE("Perspective projection") {
			test_camera->set_perspective(120.0f, 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->project_position(T::remap_xy(Vector2(200, 100)), 0.5f).is_equal_approx(T::remap_xy(Vector3(0, 0, -0.5f))));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(200, 100)), 100.0f).is_equal_approx(T::remap_xy(Vector3(0, 0, -100.0f))));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(200, 100)), test_camera->get_far()).is_equal_approx(T::remap_xy(Vector3(0, 0, -1.0f) * test_camera->get_far())));
			// 3/4th way to Top left.
			CHECK(test_camera->project_position(T::remap_xy(Vector2(100, 50)), 0.5f).is_equal_approx(T::remap_xy(Vector3(-Math::SQRT3 * 0.5f, Math::SQRT3 * 0.25f, -0.5f))));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(100, 50)), 1.0f).is_equal_approx(T::remap_xy(Vector3(-Math::SQRT3, Math::SQRT3 * 0.5f, -1.0f))));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(100, 50)), test_camera->get_near()).is_equal_approx(T::remap_xy(Vector3(-Math::SQRT3, Math::SQRT3 * 0.5f, -1.0f) * test_camera->get_near())));
			// 3/4th way to Bottom right.
			CHECK(test_camera->project_position(T::remap_xy(Vector2(300, 150)), 0.5f).is_equal_approx(T::remap_xy(Vector3(Math::SQRT3 * 0.5f, -Math::SQRT3 * 0.25f, -0.5f))));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(300, 150)), 1.0f).is_equal_approx(T::remap_xy(Vector3(Math::SQRT3, -Math::SQRT3 * 0.5f, -1.0f))));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(300, 150)), test_camera->get_far()).is_equal_approx(T::remap_xy(Vector3(Math::SQRT3, -Math::SQRT3 * 0.5f, -1.0f) * test_camera->get_far())));
		}

		SUBCASE("Frustum projection") {
			test_camera->set_frustum(Math::SQRT3, Vector2(4.0f, -7.0f), 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->project_position(T::remap_xy(Vector2(200, 100)), 0.5f).is_equal_approx(Vector3(4.0f, -7.0f, -0.5f)));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(200, 100)), 100.0f).is_equal_approx(Vector3(8.0f, -14.0f, -1.0f) * 100.0f));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(200, 100)), test_camera->get_far()).is_equal_approx(Vector3(8.0f, -14.0f, -1.0f) * test_camera->get_far()));
			// 3/4th way to Top left.
			CHECK(test_camera->project_position(T::remap_xy(Vector2(100, 50)), 0.5f).is_equal_approx(T::remap_xy(Vector3(-Math::SQRT3 * 0.5f, Math::SQRT3 * 0.25f, 0.0f)) + Vector3(4.0f, -7.0f, -0.5f)));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(100, 50)), 1.0f).is_equal_approx(T::remap_xy(Vector3(-Math::SQRT3, Math::SQRT3 * 0.5f, 0.0f)) + Vector3(8.0f, -14.0f, -1.0f)));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(100, 50)), test_camera->get_near()).is_equal_approx((T::remap_xy(Vector3(-Math::SQRT3, Math::SQRT3 * 0.5f, 0.0f)) + Vector3(8.0f, -14.0f, -1.0f)) * test_camera->get_near()));
			// 3/4th way to Bottom right.
			CHECK(test_camera->project_position(T::remap_xy(Vector2(300, 150)), 0.5f).is_equal_approx(T::remap_xy(Vector3(Math::SQRT3 * 0.5f, -Math::SQRT3 * 0.25f, 0.0f)) + Vector3(4.0f, -7.0f, -0.5f)));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(300, 150)), 1.0f).is_equal_approx(T::remap_xy(Vector3(Math::SQRT3, -Math::SQRT3 * 0.5f, 0.0f)) + Vector3(8.0f, -14.0f, -1.0f)));
			CHECK(test_camera->project_position(T::remap_xy(Vector2(300, 150)), test_camera->get_far()).is_equal_approx((T::remap_xy(Vector3(Math::SQRT3, -Math::SQRT3 * 0.5f, 0.0f)) + Vector3(8.0f, -14.0f, -1.0f)) * test_camera->get_far()));
		}
	}

	// Uses cases that are the inverse of the above sub-case.
	SUBCASE("unproject_position") {
		SUBCASE("Orthogonal projection") {
			test_camera->set_orthogonal(5.0f, 0.5f, 1000.0f);
			// Center
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(0, 0, -0.5f))).is_equal_approx(T::remap_xy(Vector2(200, 100))));
			// Top left
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(-5.0f, 2.5f, -1.5f))).is_equal_approx(T::remap_xy(Vector2(0, 0))));
			// Bottom right
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(5.0f, -2.5f, -5.0f))).is_equal_approx(T::remap_xy(Vector2(400, 200))));
		}

		SUBCASE("Perspective projection") {
			test_camera->set_perspective(120.0f, 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(0, 0, -0.5f))).is_equal_approx(T::remap_xy(Vector2(200, 100))));
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(0, 0, -100.0f))).is_equal_approx(T::remap_xy(Vector2(200, 100))));
			// 3/4th way to Top left.
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(-Math::SQRT3 * 0.5f, Math::SQRT3 * 0.25f, -0.5f))).is_equal_approx(T::remap_xy(Vector2(100, 50))));
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(-Math::SQRT3, Math::SQRT3 * 0.5f, -1.0f))).is_equal_approx(T::remap_xy(Vector2(100, 50))));
			// 3/4th way to Bottom right.
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(Math::SQRT3 * 0.5f, -Math::SQRT3 * 0.25f, -0.5f))).is_equal_approx(T::remap_xy(Vector2(300, 150))));
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(Math::SQRT3, -Math::SQRT3 * 0.5f, -1.0f))).is_equal_approx(T::remap_xy(Vector2(300, 150))));
		}

		SUBCASE("Frustum projection") {
			test_camera->set_frustum(Math::SQRT3, Vector2(4.0f, -7.0f), 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->unproject_position(Vector3(4.0f, -7.0f, -0.5f)).is_equal_approx(T::remap_xy(Vector2(200, 100))));
			CHECK(test_camera->unproject_position(Vector3(800.0f, -1400.0f, -100.0f)).is_equal_approx(T::remap_xy(Vector2(200, 100))));
			// 3/4th way to Top left.
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(-Math::SQRT3 * 0.5f, Math::SQRT3 * 0.25f, 0.0f)) + Vector3(4.0f, -7.0f, -0.5f)).is_equal_approx(T::remap_xy(Vector2(100, 50))));
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(-Math::SQRT3, Math::SQRT3 * 0.5f, 0.0f)) + Vector3(8.0f, -14.0f, -1.0f)).is_equal_approx(T::remap_xy(Vector2(100, 50))));
			// 3/4th way to Bottom right.
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(Math::SQRT3 * 0.5f, -Math::SQRT3 * 0.25f, 0.0f)) + Vector3(4.0f, -7.0f, -0.5f)).is_equal_approx(T::remap_xy(Vector2(300, 150))));
			CHECK(test_camera->unproject_position(T::remap_xy(Vector3(Math::SQRT3, -Math::SQRT3 * 0.5f, 0.0f)) + Vector3(8.0f, -14.0f, -1.0f)).is_equal_approx(T::remap_xy(Vector2(300, 150))));
		}
	}

	memdelete(test_camera);
	memdelete(mock_viewport);
}

TEST_CASE_TEMPLATE("[SceneTree][Camera3D] Project ray", T,
		Extents<Camera3D::KeepAspect::KEEP_HEIGHT, 400, 200, 0>,
		Extents<Camera3D::KeepAspect::KEEP_HEIGHT, 400, 200, 200>,
		Extents<Camera3D::KeepAspect::KEEP_WIDTH, 400, 200, 0>,
		Extents<Camera3D::KeepAspect::KEEP_WIDTH, 400, 200, 100>) {
	// Cameras need a viewport to know how to compute their frustums, so we make a fake one here.
	Camera3D *test_camera = memnew(Camera3D);
	SubViewport *mock_viewport = memnew(SubViewport);
	// 4:2.
	mock_viewport->set_size(T::size);
	SceneTree::get_singleton()->get_root()->add_child(mock_viewport);
	mock_viewport->add_child(test_camera);
	test_camera->set_global_position(Vector3(0, 0, 0));
	test_camera->set_global_rotation(Vector3(0, 0, 0));
	test_camera->set_keep_aspect_mode(T::keep_aspect);
	mock_viewport->set_size_2d_override(T::size_override);
	mock_viewport->set_size_2d_override_stretch(T::is_2d_override);

	CAPTURE(T::size);
	CAPTURE(T::keep_aspect);
	CAPTURE(T::is_2d_override);
	CAPTURE(T::size_override);

	SUBCASE("project_ray_origin") {
		SUBCASE("Orthogonal projection") {
			test_camera->set_orthogonal(5.0f, 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->project_ray_origin(T::remap_xy(Vector2(200, 100))).is_equal_approx(T::remap_xy(Vector3(0, 0, -0.5f))));
			// Top left.
			CHECK(test_camera->project_ray_origin(T::remap_xy(Vector2(0, 0))).is_equal_approx(T::remap_xy(Vector3(-5.0f, 2.5f, -0.5f))));
			// Bottom right.
			CHECK(test_camera->project_ray_origin(T::remap_xy(Vector2(400, 200))).is_equal_approx(T::remap_xy(Vector3(5.0f, -2.5f, -0.5f))));
		}

		SUBCASE("Perspective projection") {
			test_camera->set_perspective(120.0f, 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->project_ray_origin(T::remap_xy(Vector2(200, 100))).is_equal_approx(T::remap_xy(Vector3(0, 0, 0))));
			// Top left.
			CHECK(test_camera->project_ray_origin(T::remap_xy(Vector2(0, 0))).is_equal_approx(T::remap_xy(Vector3(0, 0, 0))));
			// Bottom right.
			CHECK(test_camera->project_ray_origin(T::remap_xy(Vector2(400, 200))).is_equal_approx(T::remap_xy(Vector3(0, 0, 0))));
		}

		SUBCASE("Frustum projection") {
			test_camera->set_frustum(Math::SQRT3, Vector2(4.0f, -7.0f), 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->project_ray_origin(T::remap_xy(Vector2(200, 100))).is_equal_approx(T::remap_xy(Vector3(0, 0, 0))));
			// Top left.
			CHECK(test_camera->project_ray_origin(T::remap_xy(Vector2(0, 0))).is_equal_approx(T::remap_xy(Vector3(0, 0, 0))));
			// Bottom right.
			CHECK(test_camera->project_ray_origin(T::remap_xy(Vector2(400, 200))).is_equal_approx(T::remap_xy(Vector3(0, 0, 0))));
		}
	}

	SUBCASE("project_ray_normal") {
		SUBCASE("Orthogonal projection") {
			test_camera->set_orthogonal(5.0f, 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->project_ray_normal(T::remap_xy(Vector2(200, 100))).is_equal_approx(T::remap_xy(Vector3(0, 0, -1))));
			// Top left.
			CHECK(test_camera->project_ray_normal(T::remap_xy(Vector2(0, 0))).is_equal_approx(T::remap_xy(Vector3(0, 0, -1))));
			// Bottom right.
			CHECK(test_camera->project_ray_normal(T::remap_xy(Vector2(400, 200))).is_equal_approx(T::remap_xy(Vector3(0, 0, -1))));
		}

		SUBCASE("Perspective projection") {
			test_camera->set_perspective(120.0f, 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->project_ray_normal(T::remap_xy(Vector2(200, 100))).is_equal_approx(T::remap_xy(Vector3(0, 0, -1))));
			// Top left.
			CHECK(test_camera->project_ray_normal(T::remap_xy(Vector2(0, 0))).is_equal_approx(T::remap_xy(Vector3(-Math::SQRT3, Math::SQRT3 / 2, -0.5f).normalized())));
			// Bottom right.
			CHECK(test_camera->project_ray_normal(T::remap_xy(Vector2(400, 200))).is_equal_approx(T::remap_xy(Vector3(Math::SQRT3, -Math::SQRT3 / 2, -0.5f).normalized())));
		}

		SUBCASE("Frustum projection") {
			test_camera->set_frustum(Math::SQRT3, Vector2(4.0f, -7.0f), 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->project_ray_normal(T::remap_xy(Vector2(200, 100))).is_equal_approx(Vector3(4.0f, -7.0f, -0.5f).normalized()));
			// Top left.
			CHECK(test_camera->project_ray_normal(T::remap_xy(Vector2(0, 0))).is_equal_approx((T::remap_xy(Vector3(-Math::SQRT3, Math::SQRT3 / 2, 0.0f)) + Vector3(4.0f, -7.0f, -0.5f)).normalized()));
			// Bottom right.
			CHECK(test_camera->project_ray_normal(T::remap_xy(Vector2(400, 200))).is_equal_approx((T::remap_xy(Vector3(Math::SQRT3, -Math::SQRT3 / 2, 0.0f)) + Vector3(4.0f, -7.0f, -0.5f)).normalized()));
		}
	}

	SUBCASE("project_local_ray_normal") {
		test_camera->set_rotation_degrees(Vector3(60, 60, 60));

		SUBCASE("Orthogonal projection") {
			test_camera->set_orthogonal(5.0f, 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->project_local_ray_normal(T::remap_xy(Vector2(200, 100))).is_equal_approx(T::remap_xy(Vector3(0, 0, -1))));
			// Top left.
			CHECK(test_camera->project_local_ray_normal(T::remap_xy(Vector2(0, 0))).is_equal_approx(T::remap_xy(Vector3(0, 0, -1))));
			// Bottom right.
			CHECK(test_camera->project_local_ray_normal(T::remap_xy(Vector2(400, 200))).is_equal_approx(T::remap_xy(Vector3(0, 0, -1))));
		}

		SUBCASE("Perspective projection") {
			test_camera->set_perspective(120.0f, 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->project_local_ray_normal(T::remap_xy(Vector2(200, 100))).is_equal_approx(T::remap_xy(Vector3(0, 0, -1))));
			// Top left.
			CHECK(test_camera->project_local_ray_normal(T::remap_xy(Vector2(0, 0))).is_equal_approx(T::remap_xy(Vector3(-Math::SQRT3, Math::SQRT3 / 2, -0.5f).normalized())));
			// Bottom right.
			CHECK(test_camera->project_local_ray_normal(T::remap_xy(Vector2(400, 200))).is_equal_approx(T::remap_xy(Vector3(Math::SQRT3, -Math::SQRT3 / 2, -0.5f).normalized())));
		}

		SUBCASE("Frustum projection") {
			test_camera->set_frustum(Math::SQRT3, Vector2(4.0f, -7.0f), 0.5f, 1000.0f);
			// Center.
			CHECK(test_camera->project_local_ray_normal(T::remap_xy(Vector2(200, 100))).is_equal_approx(Vector3(4.0f, -7.0f, -0.5f).normalized()));
			// Top left.
			CHECK(test_camera->project_local_ray_normal(T::remap_xy(Vector2(0, 0))).is_equal_approx((T::remap_xy(Vector3(-Math::SQRT3, Math::SQRT3 / 2, 0.0f)) + Vector3(4.0f, -7.0f, -0.5f)).normalized()));
			// Bottom right.
			CHECK(test_camera->project_local_ray_normal(T::remap_xy(Vector2(400, 200))).is_equal_approx((T::remap_xy(Vector3(Math::SQRT3, -Math::SQRT3 / 2, 0.0f)) + Vector3(4.0f, -7.0f, -0.5f)).normalized()));
		}
	}

	memdelete(test_camera);
	memdelete(mock_viewport);
}
