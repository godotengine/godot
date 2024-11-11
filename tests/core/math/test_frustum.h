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

#ifndef TEST_FRUSTUM_H
#define TEST_FRUSTUM_H

#include "core/math/frustum.h"

#include "tests/test_macros.h"

namespace TestFrustum {

TEST_CASE("[Frustum] Construction from planes") {
	constexpr real_t sqrt2half = 1.41421356237309504880 / 2;

	Vector<Plane> planes;
	planes.push_back(Plane(0, 0, 1, -0.02)); // NEAR with znear = 0.02
	planes.push_back(Plane(0, 0, -1, 50)); // FAR with zfar = 50
	planes.push_back(Plane(-sqrt2half, 0, sqrt2half, 0)); // LEFT with fov 90° and aspect 1.0
	planes.push_back(Plane(0, sqrt2half, sqrt2half, 0)); // TOP with fov 90° and aspect 1.0
	planes.push_back(Plane(sqrt2half, 0, sqrt2half, 0)); // RIGHT with fov 90° and aspect 1.0
	planes.push_back(Plane(0, -sqrt2half, sqrt2half, 0)); // BOTTOM with fov 90° and aspect 1.0

	Frustum frustum(planes);

	CHECK(frustum.planes[0] == planes[0]);
	CHECK(frustum.planes[1] == planes[1]);
	CHECK(frustum.planes[2] == planes[2]);
	CHECK(frustum.planes[3] == planes[3]);
	CHECK(frustum.planes[4] == planes[4]);
	CHECK(frustum.planes[5] == planes[5]);
}

TEST_CASE("[Frustum] Orthographic tests") {
	Frustum frustum_perspective;
	Frustum frustum_orthographic;

	frustum_perspective.set_perspective(50, 0.8, 0.2, 10.0);
	frustum_orthographic.set_orthogonal(-4, 5, -6, 7, 0.2, 9);

	CHECK(!frustum_perspective.is_orthogonal());
	CHECK(frustum_orthographic.is_orthogonal());
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
}

TEST_CASE("[Frustum] get_fovy()") {
	real_t fov = Frustum::get_fovy(90, 0.5);
	CHECK(fov == doctest::Approx(53.1301));
}

TEST_CASE("[Frustum] Perspective camera values retrieval") {
	Frustum frustum;

	frustum.set_perspective(30, 0.8, 0.2, 10.0);

	CHECK(frustum.get_z_near() == doctest::Approx(0.2));
	CHECK(frustum.get_z_far() == doctest::Approx(10));
	CHECK(Frustum::get_fovy(frustum.get_fov(), 1.0 / 0.8) == doctest::Approx(30));
	CHECK(frustum.get_aspect() == doctest::Approx(0.8));
}

TEST_CASE("[Frustum] Half extents") {
	constexpr real_t sqrt2 = 1.41421356237309504880;

	Frustum frustum;

	frustum.set_perspective(90, 0.8, 0.1, 40, false);
	Vector2 near1 = frustum.get_viewport_half_extents();
	CHECK(near1.is_equal_approx(Vector2(0.08, 0.1)));

	frustum.set_perspective(45, 0.5, 0.01, 50, true);
	Vector2 near2 = frustum.get_viewport_half_extents();
	CHECK(near2.is_equal_approx(Vector2((sqrt2 - 1) * 0.01, (sqrt2 - 1) * 0.02)));
}

} //namespace TestFrustum

#endif // TEST_FRUSTUM_H
