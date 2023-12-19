/**************************************************************************/
/*  test_projection.h                                                     */
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

#ifndef TEST_PROJECTION_H
#define TEST_PROJECTION_H

#include "tests/test_macros.h"

namespace TestProjection {

static bool isIdentity(const Projection &p_projection) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (p_projection[i][j] != doctest::Approx(i == j ? 1.0 : 0.0))
				return false;
		}
	}
	return true;
}

TEST_CASE("[Projection] Constructor") {
	Projection actual;
	CHECK_MESSAGE(isIdentity(actual),
			"Default constructor should initialize the projection to the identity.");
}

TEST_CASE("[Projection] create_depth_correction") {
	SUBCASE("With vertical flip") {
		Projection expected(Vector4(1, 0, 0, 0), Vector4(0, -1, 0, 0), Vector4(0, 0, 0.5f, 0), Vector4(0, 0, 0.5, 1.0f));
		Projection actual = Projection(
				Vector4(-29.32, 43.39, -94.85, -89.95),
				Vector4(-55.01, 55.50, -87.92, -36.67),
				Vector4(48.63, 19.80, 75.48, 44.12),
				Vector4(-2.16, 70.96, -59.63, 23.87))
									.create_depth_correction(true);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}

	SUBCASE("Without vertical flip") {
		Projection expected(Vector4(1, 0, 0, 0), Vector4(0, 1, 0, 0), Vector4(0, 0, 0.5f, 0), Vector4(0, 0, 0.5, 1.0f));
		Projection actual = Projection(
				Vector4(-29.32, 43.39, -94.85, -89.95),
				Vector4(-55.01, 55.50, -87.92, -36.67),
				Vector4(48.63, 19.80, 75.48, 44.12),
				Vector4(-2.16, 70.96, -59.63, 23.87))
									.create_depth_correction(false);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}
}

TEST_CASE("[Projection] create_fit_aabb") {
	AABB aabb(Vector3(1.0, 1.0, 1.0), Vector3(2.0, 2.0, 2.0));
	Projection expected(
			Vector4(1.0, 0.0, 0.0, 0.),
			Vector4(0.0, 1.0, 0.0, 0.0),
			Vector4(0.0, 0.0, 1.0, 0.0),
			Vector4(-2.0, -2.0, -2.0, 1.0));
	Projection actual = Projection().create_fit_aabb(aabb);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
		}
	}
}

TEST_CASE("[Projection] create_for_hmd") {
	SUBCASE("Projection for eye 1") {
		Projection expected(
				Vector4(0.735632, 0.0, 0.0, 0.0),
				Vector4(0.0, 0.735632, 0.0, 0.0),
				Vector4(-0.114943, 0.0, -3.0, -1.0),
				Vector4(0.0, 0.0, -4.0, 0.0));
		Projection actual = Projection().create_for_hmd(1, 1.0, 6.0, 14.5, 4.0, 1.5, 1.0, 2.0);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}

	SUBCASE("Projection for eye 2") {
		Projection expected(
				Vector4(0.735632, 0.0, 0.0, 0.0),
				Vector4(0.0, 0.735632, 0.0, 0.0),
				Vector4(0.114943, 0.0, -3.0, -1.0),
				Vector4(0.0, 0.0, -4.0, 0.0));
		Projection actual = Projection().create_for_hmd(2, 1.0, 6.0, 14.5, 4.0, 1.5, 1.0, 2.0);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}
}

TEST_CASE("[Projection] create_frustum") {
	SUBCASE("Invalid parameter p_right <= p_left") {
		ERR_PRINT_OFF;
		Projection actual = Projection().create_frustum(5.098039, 5.098039, -2.666666, 2.666666, 1.0, 2.0);
		ERR_PRINT_ON;

		CHECK(isIdentity(actual));
	}

	SUBCASE("Invalid parameter p_top <= p_bottom") {
		ERR_PRINT_OFF;
		Projection actual = Projection().create_frustum(-5.098039, 5.098039, 2.666666, 2.666666, 1.0, 2.0);
		ERR_PRINT_ON;

		CHECK(isIdentity(actual));
	}

	SUBCASE("Invalid parameter p_far <= p_near") {
		ERR_PRINT_OFF;
		Projection actual = Projection().create_frustum(-5.098039, 5.098039, -2.666666, 2.666666, 1.0, 1.0);
		ERR_PRINT_ON;

		CHECK(isIdentity(actual));
	}

	SUBCASE("Valid parameters") {
		Projection expected(
				Vector4(0.196154, 0.0, 0.0, 0.0),
				Vector4(0.0, 0.375, 0.0, 0.0),
				Vector4(0.0, 0.0, -3.0, -1.0),
				Vector4(0.0, 0.0, -4.0, 0.0));
		Projection actual = Projection().create_frustum(-5.098039, 5.098039, -2.666666, 2.666666, 1.0, 2.0);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}
}

TEST_CASE("[Projection] create_frustum_aspect") {
	SUBCASE("Valid parameters") {
		Projection expected(
				Vector4(4, 0, 0, 0),
				Vector4(0, 2, 0, 0),
				Vector4(4, 2, -3, -1),
				Vector4(0, 0, -4, 0));
		Projection actual = Projection().create_frustum_aspect(1.0, 0.5, Vector2(1.0, 1.0), 1.0, 2.0);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}
}

TEST_CASE("[Projection] create_light_atlas_rect") {
	Projection expected(
			Vector4(10.0, 0, 0, 0),
			Vector4(0, 10.0, 0, 0),
			Vector4(0, 0, 1.0, 0),
			Vector4(0, 0, 0, 1.0));
	Rect2 rect = Rect2(Vector2(0.0, 0.0),
			Vector2(10.0, 10.0));
	Projection actual = Projection().create_light_atlas_rect(rect);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
		}
	}
}

TEST_CASE("[Projection] create_orthogonal") {
	Projection expected(
			Vector4(0.2, 0, 0, 0),
			Vector4(0, 0.2, 0, 0),
			Vector4(0, 0, -2, 0),
			Vector4(-3, -3, -3, 1));
	Projection actual = Projection().create_orthogonal(10, 20, 10, 20, 1.0, 2.0);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
		}
	}
}

TEST_CASE("[Projection] create_orthogonal_aspect") {
	SUBCASE("Flipped y") {
		Projection expected(
				Vector4(0.2, 0, 0, 0),
				Vector4(0, 0.26, 0, 0),
				Vector4(0, 0, -2, 0),
				Vector4(0, 0, -3, 1));
		Projection actual = Projection().create_orthogonal_aspect(10, 1.3, 1.0, 2.0, true);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}

	SUBCASE("Not flipped y") {
		Projection expected(
				Vector4(0.153846, 0, 0, 0),
				Vector4(0, 0.2, 0, 0),
				Vector4(0, 0, -2, 0),
				Vector4(0, 0, -3, 1));
		Projection actual = Projection().create_orthogonal_aspect(10, 1.3, 1.0, 2.0, false);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}
}

TEST_CASE("[Projection] create_perspective") {
	SUBCASE("Flipped y") {
		Projection expected(
				Vector4(76.39, 0, 0, 0),
				Vector4(0, 99.307, 0, 0),
				Vector4(0, 0, -3, -1),
				Vector4(0, 0, -4, 0));
		Projection actual = Projection().create_perspective(1.5, 1.3, 1.0, 2.0, true);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}

	SUBCASE("Not flipped y") {
		Projection expected(
				Vector4(58.7615, 0, 0, 0),
				Vector4(0, 76.39, 0, 0),
				Vector4(0, 0, -3, -1),
				Vector4(0, 0, -4, 0));
		Projection actual = Projection().create_perspective(1.5, 1.3, 1.0, 2.0, false);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}
}

TEST_CASE("[Projection] create_perspective_hmd") {
	SUBCASE("No y flipping, eye 1") {
		Projection expected(
				Vector4(57.4362, 0, 0, 0),
				Vector4(0, 76.39, 0, 0),
				Vector4(41.6412, 0, -3, -1),
				Vector4(416.412, 0, -4, 0));
		Projection actual = Projection().create_perspective_hmd(1.5, 1.33, 1.0, 2.0, false, 1, 14.5, 10.0);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}

	SUBCASE("No y flipping, eye 2") {
		Projection expected(
				Vector4(57.4362, 0, 0, 0),
				Vector4(0, 76.39, 0, 0),
				Vector4(-41.6412, 0, -3, -1),
				Vector4(-416.412, 0, -4, 0));
		Projection actual = Projection().create_perspective_hmd(1.5, 1.33, 1.0, 2.0, false, 2, 14.5, 10.0);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}

	SUBCASE("Y flipping, eye 1") {
		Projection expected(
				Vector4(76.3899, 0, 0, 0),
				Vector4(0, 101.599, 0, 0),
				Vector4(55.3827, 0, -3, -1),
				Vector4(553.827, 0, -4, 0));
		Projection actual = Projection().create_perspective_hmd(1.5, 1.33, 1.0, 2.0, true, 1, 14.5, 10.0);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}

	SUBCASE("Y flipping, eye 2") {
		Projection expected(
				Vector4(76.3899, 0, 0, 0),
				Vector4(0, 101.599, 0, 0),
				Vector4(-55.3827, 0, -3, -1),
				Vector4(-553.827, 0, -4, 0));
		Projection actual = Projection().create_perspective_hmd(1.5, 1.33, 1.0, 2.0, true, 2, 14.5, 10.0);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}
}

TEST_CASE("[Projection] Determinant") {
	SUBCASE("Determinant calculation of identity matrix should return 1.0f") {
		float expected = 1.0;
		float actual = Projection().determinant();

		CHECK(expected == doctest::Approx(actual));
	}

	SUBCASE("Determinant calculation should return the right value") {
		float expected = 17317848.60;
		float actual = Projection(
				Vector4(-29.32, 43.39, -94.85, -89.95),
				Vector4(-55.01, 55.50, -87.92, -36.67),
				Vector4(48.63, 19.80, 75.48, 44.12),
				Vector4(-2.16, 70.96, -59.63, 23.87))
							   .determinant();

		CHECK(expected == doctest::Approx(actual));
	}
}

TEST_CASE("[Projection] flipped_y") {
	Projection expected(
			Vector4(-29.32, 43.39, -94.85, -89.95),
			Vector4(55.01, -55.50, 87.92, 36.67),
			Vector4(48.63, 19.80, 75.48, 44.12),
			Vector4(-2.16, 70.96, -59.63, 23.87));
	Projection actual = Projection(
			Vector4(-29.32, 43.39, -94.85, -89.95),
			Vector4(-55.01, 55.50, -87.92, -36.67),
			Vector4(48.63, 19.80, 75.48, 44.12),
			Vector4(-2.16, 70.96, -59.63, 23.87))
								.flipped_y();

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
		}
	}
}

TEST_CASE("[Projection] get_aspect") {
	real_t expected = 0.218863;
	real_t actual = Projection(
			Vector4(-29.32, 43.39, -94.85, -89.95),
			Vector4(55.01, -55.50, 87.92, 36.67),
			Vector4(48.63, 19.80, 75.48, 44.12),
			Vector4(-2.16, 70.96, -59.63, 23.87))
							.get_aspect();

	CHECK(expected == doctest::Approx(actual));
}

TEST_CASE("[Projection] get_far_plane_half_extents") {
	Vector2 expected = Vector2(0.200337, 0.158066);
	Vector2 actual = Projection(
			Vector4(-29.32, 43.39, -94.85, -89.95),
			Vector4(55.01, -55.50, 87.92, 36.67),
			Vector4(48.63, 19.80, 75.48, 44.12),
			Vector4(-2.16, 70.96, -59.63, 23.87))
							 .get_far_plane_half_extents();

	CHECK(expected.x == doctest::Approx(actual.x));
	CHECK(expected.y == doctest::Approx(actual.y));
}

TEST_CASE("[Projection] get_fov") {
	real_t expected = 64.8575;
	real_t actual = Projection(
			Vector4(-29.32, 43.39, -94.85, -89.95),
			Vector4(55.01, -55.50, 87.92, 36.67),
			Vector4(48.63, 19.80, 75.48, 44.12),
			Vector4(-2.16, 70.96, -59.63, 23.87))
							.get_fov();

	CHECK(expected == doctest::Approx(actual));
}

TEST_CASE("[Projection] get_fovy") {
	real_t expected = 1.32997;
	real_t actual = Projection().get_fovy(1.0, 1.33);

	CHECK(expected == doctest::Approx(actual));
}

TEST_CASE("[Projection] get_lod_multiplier") {
	float expected = 1;
	float actual = Projection().get_lod_multiplier();

	CHECK(expected == doctest::Approx(actual));
}

TEST_CASE("[Projection] get_pixels_per_meter") {
	int expected = 55;
	int actual = Projection(
			Vector4(-29.32, 43.39, -94.85, -89.95),
			Vector4(55.01, -55.50, 87.92, 36.67),
			Vector4(48.63, 19.80, 75.48, 44.12),
			Vector4(-2.16, 70.96, -59.63, 23.87))
						 .get_pixels_per_meter(64);

	CHECK(expected == actual);
}

TEST_CASE("[Projection] get_projection_plane") {
	SUBCASE("PLANE_NEAR") {
		Plane expected = Plane(Vector3(0, 0, -1), 1);
		Plane actual = Projection().get_projection_plane(Projection::Planes::PLANE_NEAR);

		CHECK(expected == actual);
	}

	SUBCASE("PLANE_FAR") {
		Plane expected = Plane(Vector3(0, 0, 1), 1);
		Plane actual = Projection().get_projection_plane(Projection::Planes::PLANE_FAR);

		CHECK(expected == actual);
	}

	SUBCASE("PLANE_LEFT") {
		Plane expected = Plane(Vector3(-1, 0, 0), 1);
		Plane actual = Projection().get_projection_plane(Projection::Planes::PLANE_LEFT);

		CHECK(expected == actual);
	}

	SUBCASE("PLANE_TOP") {
		Plane expected = Plane(Vector3(0, 1, 0), 1);
		Plane actual = Projection().get_projection_plane(Projection::Planes::PLANE_TOP);

		CHECK(expected == actual);
	}

	SUBCASE("PLANE_RIGHT") {
		Plane expected = Plane(Vector3(1, 0, 0), 1);
		Plane actual = Projection().get_projection_plane(Projection::Planes::PLANE_RIGHT);

		CHECK(expected == actual);
	}

	SUBCASE("PLANE_BOTTOM") {
		Plane expected = Plane(Vector3(0, -1, 0), 1);
		Plane actual = Projection().get_projection_plane(Projection::Planes::PLANE_BOTTOM);

		CHECK(expected == actual);
	}
}

TEST_CASE("[Projection] get_viewport_half_extents ") {
	SUBCASE("get_viewport_half_extents of a generic matrix") {
		Vector2 expected(0.189962f, 0.867947f);
		Vector2 actual = Projection(
				Vector4(-29.32, 43.39, -94.85, -89.95),
				Vector4(55.01, -55.50, 87.92, 36.67),
				Vector4(48.63, 19.80, 75.48, 44.12),
				Vector4(-2.16, 70.96, -59.63, 23.87))
								 .get_viewport_half_extents();

		CHECK(expected.x == doctest::Approx(actual.x));
		CHECK(expected.y == doctest::Approx(actual.y));
	}

	SUBCASE("get_viewport_half_extents of null matrix") {
		Vector2 expected(0.0, 0.0);
		Vector2 actual = Projection(
				Vector4(0, 0, 0, 0),
				Vector4(0, 0, 0, 0),
				Vector4(0, 0, 0, 0),
				Vector4(0, 0, 0, 0))
								 .get_viewport_half_extents();

		CHECK(expected.is_equal_approx(actual));
	}
}

TEST_CASE("[Projection] get_z_far") {
	real_t expected = 1;
	real_t actual = Projection().get_z_far();

	CHECK(expected == actual);
}

TEST_CASE("[Projection] get_z_near") {
	real_t expected = -1;
	real_t actual = Projection().get_z_near();

	CHECK(expected == actual);
}

TEST_CASE("[Projection] inverse") {
	SUBCASE("The inverse of the identity is the identity") {
		Projection expected;
		Projection actual = Projection().inverse();

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}

	SUBCASE("Calculate the inverse of a matrix") {
		Projection expected(
				Vector4(0.0145519, -0.0304995, -0.00444437, 0.0161966),
				Vector4(0.00190803, 0.01359, 0.0164218, -0.00228562),
				Vector4(-0.00297949, 0.0167925, 0.0164924, -0.015914),
				Vector4(-0.0117984, -0.00121019, -0.00802058, 0.0103989));
		Projection actual = Projection(
				Vector4(-29.32, 43.39, -94.85, -89.95),
				Vector4(-55.01, 55.50, -87.92, -36.67),
				Vector4(48.63, 19.80, 75.48, 44.12),
				Vector4(-2.16, 70.96, -59.63, 23.87))
									.inverse();

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
			}
		}
	}
}

TEST_CASE("[Projection] is_orthogonal ") {
	SUBCASE("The identity matrix is an orthoogonal matrix") {
		Projection identity;
		CHECK(identity.is_orthogonal());
	}

	SUBCASE("An orthogonal matrix should return true") {
		Projection orthogonal_matrix(
				Vector4(0.6, -0.8, 0, 0),
				Vector4(0.8, 0.6, 0, 0),
				Vector4(0, 0, 1, 0),
				Vector4(0, 0, 0, 1));

		CHECK(orthogonal_matrix.is_orthogonal());
	}

	SUBCASE("A non orthogonal matrix should return false") {
		Projection not_orthogonal_matrix(
				Vector4(1, 2, 3, 4),
				Vector4(5, 6, 7, 8),
				Vector4(9, 10, 11, 12),
				Vector4(13, 14, 15, 16));

		CHECK_FALSE(not_orthogonal_matrix.is_orthogonal());
	}
}

TEST_CASE("[Projection] jitter_offseted") {
	Projection expected(
			Vector4(1, 0, 0, 0),
			Vector4(0, 1, 0, 0),
			Vector4(0, 0, 1, 0),
			Vector4(1, 1, 0, 1));
	Projection actual = Projection().jitter_offseted(Vector2(1, 1));

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
		}
	}
}

TEST_CASE("[Projection] perspective_znear_adjusted") {
	Projection expected(
			Vector4(-29.32, 43.39, -94.85, -89.95),
			Vector4(-55.01, 55.5, -87.92, -36.67),
			Vector4(48.63, 19.8, -6.19296, 44.12),
			Vector4(-2.16, 70.96, -7.19296, 23.87));
	Projection actual = Projection(
			Vector4(-29.32, 43.39, -94.85, -89.95),
			Vector4(-55.01, 55.50, -87.92, -36.67),
			Vector4(48.63, 19.80, 75.48, 44.12),
			Vector4(-2.16, 70.96, -59.63, 23.87))
								.perspective_znear_adjusted(1);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			CHECK(expected[i][j] == doctest::Approx(actual[i][j]));
		}
	}
}
} //namespace TestProjection

#endif // TEST_PROJECTION_H
