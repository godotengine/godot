/**************************************************************************/
/*  test_frustum.h                                                        */
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

#include "core/math/frustum.h"

#include "tests/test_macros.h"

namespace TestFrustum {

TEST_CASE("[Frustum] Construction from planes") {
	constexpr real_t sqrt_2_half = 1.41421356237309504880 / 2;

	Vector<Plane> planes;
	planes.push_back(Plane(0, 0, 1, -0.02)); // NEAR with znear = 0.02
	planes.push_back(Plane(0, 0, -1, 50)); // FAR with zfar = 50
	planes.push_back(Plane(-sqrt_2_half, 0, sqrt_2_half, 0)); // LEFT with fov 90° and aspect 1.0
	planes.push_back(Plane(0, sqrt_2_half, sqrt_2_half, 0)); // TOP with fov 90° and aspect 1.0
	planes.push_back(Plane(sqrt_2_half, 0, sqrt_2_half, 0)); // RIGHT with fov 90° and aspect 1.0
	planes.push_back(Plane(0, -sqrt_2_half, sqrt_2_half, 0)); // BOTTOM with fov 90° and aspect 1.0

	Frustum frustum(planes);

	CHECK_EQ(frustum.planes[0], planes[0]);
	CHECK_EQ(frustum.planes[1], planes[1]);
	CHECK_EQ(frustum.planes[2], planes[2]);
	CHECK_EQ(frustum.planes[3], planes[3]);
	CHECK_EQ(frustum.planes[4], planes[4]);
	CHECK_EQ(frustum.planes[5], planes[5]);
}

TEST_CASE("[Frustum] Orthographic construction") {
	Frustum frustum;
	Projection ref;

	frustum.set_orthogonal(-4, 5, -6, 7, 0.2, 9);
	ref.set_orthogonal(-4, 5, -6, 7, 0.2, 9);

	Vector<Plane> truth = ref.get_projection_planes(Transform3D());

	CHECK(frustum.planes[0].is_equal_approx(truth[0].normalized()));
	CHECK(frustum.planes[1].is_equal_approx(truth[1].normalized()));
	CHECK(frustum.planes[2].is_equal_approx(truth[2].normalized()));
	CHECK(frustum.planes[3].is_equal_approx(truth[3].normalized()));
	CHECK(frustum.planes[4].is_equal_approx(truth[4].normalized()));
	CHECK(frustum.planes[5].is_equal_approx(truth[5].normalized()));
}

TEST_CASE("[Frustum] Perspective construction") {
	Frustum frustum;
	Projection ref;

	frustum.set_perspective(50, 0.8, 0.2, 10.0);
	ref.set_perspective(50, 0.8, 0.2, 10.0);

	Vector<Plane> truth = ref.get_projection_planes(Transform3D());

	CHECK(frustum.planes[0].is_equal_approx(truth[0].normalized()));
	CHECK(frustum.planes[1].is_equal_approx(truth[1].normalized()));
	CHECK(frustum.planes[2].is_equal_approx(truth[2].normalized()));
	CHECK(frustum.planes[3].is_equal_approx(truth[3].normalized()));
	CHECK(frustum.planes[4].is_equal_approx(truth[4].normalized()));
	CHECK(frustum.planes[5].is_equal_approx(truth[5].normalized()));
}

TEST_CASE("[Frustum] Frustum construction") {
	Frustum frustum;
	Projection ref;

	frustum.set_frustum(-4, 5, -6, 7, 0.2, 9);
	ref.set_frustum(-4, 5, -6, 7, 0.2, 9);

	Vector<Plane> truth = ref.get_projection_planes(Transform3D());

	CHECK(frustum.planes[0].is_equal_approx(truth[0].normalized()));
	CHECK(frustum.planes[1].is_equal_approx(truth[1].normalized()));
	CHECK(frustum.planes[2].is_equal_approx(truth[2].normalized()));
	CHECK(frustum.planes[3].is_equal_approx(truth[3].normalized()));
	CHECK(frustum.planes[4].is_equal_approx(truth[4].normalized()));
	CHECK(frustum.planes[5].is_equal_approx(truth[5].normalized()));

	frustum.set_frustum(1.0, 4.0 / 3.0, Vector2(0.5, -0.25), 0.5, 50, true);
	ref.set_frustum(1.0, 4.0 / 3.0, Vector2(0.5, -0.25), 0.5, 50, true);

	truth = ref.get_projection_planes(Transform3D());

	CHECK(frustum.planes[0].is_equal_approx(truth[0].normalized()));
	CHECK(frustum.planes[1].is_equal_approx(truth[1].normalized()));
	CHECK(frustum.planes[2].is_equal_approx(truth[2].normalized()));
	CHECK(frustum.planes[3].is_equal_approx(truth[3].normalized()));
	CHECK(frustum.planes[4].is_equal_approx(truth[4].normalized()));
	CHECK(frustum.planes[5].is_equal_approx(truth[5].normalized()));
}

TEST_CASE("[Frustum] Perspective values extraction") {
	Frustum persp;
	persp.set_perspective(90, 0.5, 1, 50, true);

	double znear = persp.get_z_near();
	double zfar = persp.get_z_far();

	CHECK_EQ(znear, doctest::Approx(1));
	CHECK_EQ(zfar, doctest::Approx(50));

	persp.set_perspective(38, 1.3, 0.2, 8, false);

	znear = persp.get_z_near();
	zfar = persp.get_z_far();

	CHECK_EQ(znear, doctest::Approx(0.2));
	CHECK_EQ(zfar, doctest::Approx(8));

	persp.set_perspective(47, 2.5, 0.9, 14, true);

	znear = persp.get_z_near();
	zfar = persp.get_z_far();

	CHECK_EQ(znear, doctest::Approx(0.9));
	CHECK_EQ(zfar, doctest::Approx(14));
}

TEST_CASE("[Frustum] Frustum values extraction") {
	Frustum frustum;
	frustum.set_frustum(1.0, 4.0 / 3.0, Vector2(0.5, -0.25), 0.5, 50, true);

	double znear = frustum.get_z_near();
	double zfar = frustum.get_z_far();

	CHECK_EQ(znear, doctest::Approx(0.5));
	CHECK_EQ(zfar, doctest::Approx(50));

	frustum.set_frustum(2.0, 1.5, Vector2(-0.5, 2), 2, 12, false);

	znear = frustum.get_z_near();
	zfar = frustum.get_z_far();

	CHECK_EQ(znear, doctest::Approx(2));
	CHECK_EQ(zfar, doctest::Approx(12));
}

TEST_CASE("[Frustum] Orthographic values extraction") {
	Frustum ortho;
	ortho.set_orthogonal(-2, 3, -0.5, 1.5, 1.2, 15);

	double znear = ortho.get_z_near();
	double zfar = ortho.get_z_far();

	CHECK_EQ(znear, doctest::Approx(1.2));
	CHECK_EQ(zfar, doctest::Approx(15));

	ortho.set_orthogonal(-7, 2, 2.5, 5.5, 0.5, 6);

	znear = ortho.get_z_near();
	zfar = ortho.get_z_far();

	CHECK_EQ(znear, doctest::Approx(0.5));
	CHECK_EQ(zfar, doctest::Approx(6));
}

TEST_CASE("[Frustum] Perspective extents") {
	constexpr real_t sqrt3 = 1.7320508;
	Frustum persp;
	persp.set_perspective(90, 1, 1, 40, false);
	Vector2 ne = persp.get_viewport_half_extents();
	Vector2 fe = persp.get_far_plane_half_extents();
	Rect2 nr = persp.get_viewport_rect();
	Rect2 fr = persp.get_far_plane_rect();

	CHECK(ne.is_equal_approx(Vector2(1, 1) * 1));
	CHECK(fe.is_equal_approx(Vector2(1, 1) * 40));
	CHECK(nr.is_equal_approx(Rect2(-Vector2(1, 1), 2 * Vector2(1, 1))));
	CHECK(fr.is_equal_approx(Rect2(-Vector2(1, 1) * 40, 2 * Vector2(1, 1) * 40)));

	persp.set_perspective(120, sqrt3, 0.8, 10, true);
	ne = persp.get_viewport_half_extents();
	fe = persp.get_far_plane_half_extents();
	nr = persp.get_viewport_rect();
	fr = persp.get_far_plane_rect();

	CHECK(ne.is_equal_approx(Vector2(sqrt3, 1.0) * 0.8));
	CHECK(fe.is_equal_approx(Vector2(sqrt3, 1.0) * 10));
	CHECK(nr.is_equal_approx(Rect2(-Vector2(sqrt3, 1.0) * 0.8, 2 * Vector2(sqrt3, 1.0) * 0.8)));
	CHECK(fr.is_equal_approx(Rect2(-Vector2(sqrt3, 1.0) * 10, 2 * Vector2(sqrt3, 1.0) * 10)));

	persp.set_perspective(60, 1.2, 0.5, 15, false);
	ne = persp.get_viewport_half_extents();
	fe = persp.get_far_plane_half_extents();
	nr = persp.get_viewport_rect();
	fr = persp.get_far_plane_rect();

	CHECK(ne.is_equal_approx(Vector2(sqrt3 / 3 * 1.2, sqrt3 / 3) * 0.5));
	CHECK(fe.is_equal_approx(Vector2(sqrt3 / 3 * 1.2, sqrt3 / 3) * 15));
	CHECK(nr.is_equal_approx(Rect2(-Vector2(sqrt3 / 3 * 1.2, sqrt3 / 3) * 0.5, 2 * Vector2(sqrt3 / 3 * 1.2, sqrt3 / 3) * 0.5)));
	CHECK(fr.is_equal_approx(Rect2(-Vector2(sqrt3 / 3 * 1.2, sqrt3 / 3) * 15, 2 * Vector2(sqrt3 / 3 * 1.2, sqrt3 / 3) * 15)));
}

TEST_CASE("[Frustum] Orthographic extents") {
	Frustum ortho;
	ortho.set_orthogonal(-3, 3, -1.5, 1.5, 1.2, 15);
	Vector2 ne = ortho.get_viewport_half_extents();
	Vector2 fe = ortho.get_far_plane_half_extents();
	Rect2 nr = ortho.get_viewport_rect();
	Rect2 fr = ortho.get_far_plane_rect();

	CHECK(ne.is_equal_approx(Vector2(3, 1.5)));
	CHECK(fe.is_equal_approx(Vector2(3, 1.5)));
	CHECK(nr.is_equal_approx(Rect2(Vector2(-3, -1.5), 2 * Vector2(3, 1.5))));
	CHECK(fr.is_equal_approx(Rect2(Vector2(-3, -1.5), 2 * Vector2(3, 1.5))));

	ortho.set_orthogonal(-7, 7, -2.5, 2.5, 0.5, 6);
	ne = ortho.get_viewport_half_extents();
	fe = ortho.get_far_plane_half_extents();
	nr = ortho.get_viewport_rect();
	fr = ortho.get_far_plane_rect();

	CHECK(ne.is_equal_approx(Vector2(7, 2.5)));
	CHECK(fe.is_equal_approx(Vector2(7, 2.5)));
	CHECK(nr.is_equal_approx(Rect2(Vector2(-7, -2.5), 2 * Vector2(7, 2.5))));
	CHECK(fr.is_equal_approx(Rect2(Vector2(-7, -2.5), 2 * Vector2(7, 2.5))));

	ortho.set_orthogonal(-2, 6, -3.5, 0.5, 1.5, 5.5);
	nr = ortho.get_viewport_rect();
	fr = ortho.get_far_plane_rect();

	CHECK(nr.is_equal_approx(Rect2(Vector2(-2, -3.5), Vector2(8, 4))));
	CHECK(fr.is_equal_approx(Rect2(Vector2(-2, -3.5), Vector2(8, 4))));
}

TEST_CASE("[Frustum] Frustum extents") {
	Frustum frustum;
	frustum.set_frustum(-3, 3, -1.5, 1.5, 1.2, 15);
	Rect2 nr = frustum.get_viewport_rect();
	Rect2 fr = frustum.get_far_plane_rect();

	CHECK(nr.is_equal_approx(Rect2(Vector2(-3, -1.5), 2 * Vector2(3, 1.5))));
	CHECK(fr.is_equal_approx(Rect2(12.5 * Vector2(-3, -1.5), 25 * Vector2(3, 1.5))));

	frustum.set_frustum(-2, 6, -3.5, 0.5, 2, 6);
	nr = frustum.get_viewport_rect();
	fr = frustum.get_far_plane_rect();

	CHECK(nr.is_equal_approx(Rect2(Vector2(-2, -3.5), Vector2(8, 4))));
	CHECK(fr.is_equal_approx(Rect2(3 * Vector2(-2, -3.5), 3 * Vector2(8, 4))));
}

TEST_CASE("[Frustum] Endpoints") {
	constexpr real_t sqrt3 = 1.7320508;
	Frustum persp;
	persp.set_perspective(90, 1, 1, 40, false);
	Vector3 ep[8];
	persp.get_endpoints(Transform3D(), ep);

	CHECK(ep[0].is_equal_approx(Vector3(-1, 1, -1) * 40));
	CHECK(ep[1].is_equal_approx(Vector3(-1, -1, -1) * 40));
	CHECK(ep[2].is_equal_approx(Vector3(1, 1, -1) * 40));
	CHECK(ep[3].is_equal_approx(Vector3(1, -1, -1) * 40));
	CHECK(ep[4].is_equal_approx(Vector3(-1, 1, -1) * 1));
	CHECK(ep[5].is_equal_approx(Vector3(-1, -1, -1) * 1));
	CHECK(ep[6].is_equal_approx(Vector3(1, 1, -1) * 1));
	CHECK(ep[7].is_equal_approx(Vector3(1, -1, -1) * 1));

	persp.set_perspective(120, sqrt3, 0.8, 10, true);
	persp.get_endpoints(Transform3D(), ep);

	CHECK(ep[0].is_equal_approx(Vector3(-sqrt3, 1, -1) * 10));
	CHECK(ep[1].is_equal_approx(Vector3(-sqrt3, -1, -1) * 10));
	CHECK(ep[2].is_equal_approx(Vector3(sqrt3, 1, -1) * 10));
	CHECK(ep[3].is_equal_approx(Vector3(sqrt3, -1, -1) * 10));
	CHECK(ep[4].is_equal_approx(Vector3(-sqrt3, 1, -1) * 0.8));
	CHECK(ep[5].is_equal_approx(Vector3(-sqrt3, -1, -1) * 0.8));
	CHECK(ep[6].is_equal_approx(Vector3(sqrt3, 1, -1) * 0.8));
	CHECK(ep[7].is_equal_approx(Vector3(sqrt3, -1, -1) * 0.8));

	persp.set_perspective(60, 1.2, 0.5, 15, false);
	persp.get_endpoints(Transform3D(), ep);

	CHECK(ep[0].is_equal_approx(Vector3(-sqrt3 / 3 * 1.2, sqrt3 / 3, -1) * 15));
	CHECK(ep[1].is_equal_approx(Vector3(-sqrt3 / 3 * 1.2, -sqrt3 / 3, -1) * 15));
	CHECK(ep[2].is_equal_approx(Vector3(sqrt3 / 3 * 1.2, sqrt3 / 3, -1) * 15));
	CHECK(ep[3].is_equal_approx(Vector3(sqrt3 / 3 * 1.2, -sqrt3 / 3, -1) * 15));
	CHECK(ep[4].is_equal_approx(Vector3(-sqrt3 / 3 * 1.2, sqrt3 / 3, -1) * 0.5));
	CHECK(ep[5].is_equal_approx(Vector3(-sqrt3 / 3 * 1.2, -sqrt3 / 3, -1) * 0.5));
	CHECK(ep[6].is_equal_approx(Vector3(sqrt3 / 3 * 1.2, sqrt3 / 3, -1) * 0.5));
	CHECK(ep[7].is_equal_approx(Vector3(sqrt3 / 3 * 1.2, -sqrt3 / 3, -1) * 0.5));
}

} //namespace TestFrustum
