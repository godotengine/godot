/**************************************************************************/
/*  test_astar.h                                                          */
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

#include "core/math/a_star.h"

#include "tests/test_macros.h"

namespace TestAStar {

class ABCX : public AStar3D {
public:
	enum {
		A,
		B,
		C,
		X,
	};

	ABCX() {
		add_point(A, Vector3(0, 0, 0));
		add_point(B, Vector3(1, 0, 0));
		add_point(C, Vector3(0, 1, 0));
		add_point(X, Vector3(0, 0, 1));
		connect_points(A, B);
		connect_points(A, C);
		connect_points(B, C);
		connect_points(X, A);
	}

	// Disable heuristic completely.
	real_t _compute_cost(int64_t p_from, int64_t p_to) {
		if (p_from == A && p_to == C) {
			return 1000;
		}
		return 100;
	}
};

TEST_CASE("[AStar3D] ABC path") {
	ABCX abcx;
	Vector<int64_t> path = abcx.get_id_path(ABCX::A, ABCX::C);
	REQUIRE(path.size() == 3);
	CHECK(path[0] == ABCX::A);
	CHECK(path[1] == ABCX::B);
	CHECK(path[2] == ABCX::C);
}

TEST_CASE("[AStar3D] ABCX path") {
	ABCX abcx;
	Vector<int64_t> path = abcx.get_id_path(ABCX::X, ABCX::C);
	REQUIRE(path.size() == 4);
	CHECK(path[0] == ABCX::X);
	CHECK(path[1] == ABCX::A);
	CHECK(path[2] == ABCX::B);
	CHECK(path[3] == ABCX::C);
}

TEST_CASE("[AStar3D] Add/Remove") {
	AStar3D a;

	// Manual tests.
	a.add_point(1, Vector3(0, 0, 0));
	a.add_point(2, Vector3(0, 1, 0));
	a.add_point(3, Vector3(1, 1, 0));
	a.add_point(4, Vector3(2, 0, 0));
	a.connect_points(1, 2, true);
	a.connect_points(1, 3, true);
	a.connect_points(1, 4, false);

	CHECK(a.are_points_connected(2, 1));
	CHECK(a.are_points_connected(4, 1));
	CHECK(a.are_points_connected(2, 1, false));
	CHECK_FALSE(a.are_points_connected(4, 1, false));

	a.disconnect_points(1, 2, true);
	CHECK(a.get_point_connections(1).size() == 2); // 3, 4
	CHECK(a.get_point_connections(2).size() == 0);

	a.disconnect_points(4, 1, false);
	CHECK(a.get_point_connections(1).size() == 2); // 3, 4
	CHECK(a.get_point_connections(4).size() == 0);

	a.disconnect_points(4, 1, true);
	CHECK(a.get_point_connections(1).size() == 1); // 3
	CHECK(a.get_point_connections(4).size() == 0);

	a.connect_points(2, 3, false);
	CHECK(a.get_point_connections(2).size() == 1); // 3
	CHECK(a.get_point_connections(3).size() == 1); // 1

	a.connect_points(2, 3, true);
	CHECK(a.get_point_connections(2).size() == 1); // 3
	CHECK(a.get_point_connections(3).size() == 2); // 1, 2

	a.disconnect_points(2, 3, false);
	CHECK(a.get_point_connections(2).size() == 0);
	CHECK(a.get_point_connections(3).size() == 2); // 1, 2

	a.connect_points(4, 3, true);
	CHECK(a.get_point_connections(3).size() == 3); // 1, 2, 4
	CHECK(a.get_point_connections(4).size() == 1); // 3

	a.disconnect_points(3, 4, false);
	CHECK(a.get_point_connections(3).size() == 2); // 1, 2
	CHECK(a.get_point_connections(4).size() == 1); // 3

	a.remove_point(3);
	CHECK(a.get_point_connections(1).size() == 0);
	CHECK(a.get_point_connections(2).size() == 0);
	CHECK(a.get_point_connections(4).size() == 0);

	a.add_point(0, Vector3(0, -1, 0));
	a.add_point(3, Vector3(2, 1, 0));
	// 0: (0, -1)
	// 1: (0, 0)
	// 2: (0, 1)
	// 3: (2, 1)
	// 4: (2, 0)

	// Tests for get_closest_position_in_segment.
	a.connect_points(2, 3);
	CHECK(a.get_closest_position_in_segment(Vector3(0.5, 0.5, 0)) == Vector3(0.5, 1, 0));

	a.connect_points(3, 4);
	a.connect_points(0, 3);
	a.connect_points(1, 4);
	a.disconnect_points(1, 4, false);
	a.disconnect_points(4, 3, false);
	a.disconnect_points(3, 4, false);
	// Remaining edges: <2, 3>, <0, 3>, <1, 4> (directed).
	CHECK(a.get_closest_position_in_segment(Vector3(2, 0.5, 0)) == Vector3(1.75, 0.75, 0));
	CHECK(a.get_closest_position_in_segment(Vector3(-1, 0.2, 0)) == Vector3(0, 0, 0));
	CHECK(a.get_closest_position_in_segment(Vector3(3, 2, 0)) == Vector3(2, 1, 0));

	Math::seed(0);

	// Random tests for connectivity checks
	for (int i = 0; i < 20000; i++) {
		int u = Math::rand() % 5;
		int v = Math::rand() % 4;
		if (u == v) {
			v = 4;
		}
		if (Math::rand() % 2 == 1) {
			// Add a (possibly existing) directed edge and confirm connectivity.
			a.connect_points(u, v, false);
			CHECK(a.are_points_connected(u, v, false));
		} else {
			// Remove a (possibly nonexistent) directed edge and confirm disconnectivity.
			a.disconnect_points(u, v, false);
			CHECK_FALSE(a.are_points_connected(u, v, false));
		}
	}

	// Random tests for point removal.
	for (int i = 0; i < 20000; i++) {
		a.clear();
		for (int j = 0; j < 5; j++) {
			a.add_point(j, Vector3(0, 0, 0));
		}

		// Add or remove random edges.
		for (int j = 0; j < 10; j++) {
			int u = Math::rand() % 5;
			int v = Math::rand() % 4;
			if (u == v) {
				v = 4;
			}
			if (Math::rand() % 2 == 1) {
				a.connect_points(u, v, false);
			} else {
				a.disconnect_points(u, v, false);
			}
		}

		// Remove point 0.
		a.remove_point(0);
		// White box: this will check all edges remaining in the segments set.
		for (int j = 1; j < 5; j++) {
			CHECK_FALSE(a.are_points_connected(0, j, true));
		}
	}
}

TEST_CASE("[AStar3D] Path from disabled point is empty") {
	AStar3D a;
	Vector3 p1(0, 0, 0);
	Vector3 p2(0, 1, 0);
	a.add_point(1, p1);
	a.add_point(2, p2);
	a.connect_points(1, 2);

	CHECK_EQ(a.get_id_path(1, 1), Vector<int64_t>{ 1 });
	CHECK_EQ(a.get_id_path(1, 2), Vector<int64_t>{ 1, 2 });

	CHECK_EQ(a.get_point_path(1, 1), Vector<Vector3>{ p1 });
	CHECK_EQ(a.get_point_path(1, 2), Vector<Vector3>{ p1, p2 });

	a.set_point_disabled(1, true);

	CHECK(a.get_id_path(1, 1).is_empty());
	CHECK(a.get_id_path(1, 2).is_empty());

	CHECK(a.get_point_path(1, 1).is_empty());
	CHECK(a.get_point_path(1, 2).is_empty());
}
} // namespace TestAStar
