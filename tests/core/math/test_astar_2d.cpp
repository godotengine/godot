/**************************************************************************/
/*  test_astar_2d.cpp                                                     */
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

TEST_FORCE_LINK(test_astar_2d)

#include "core/math/a_star.h"

namespace TestAStar2D {

class ABCX : public AStar2D {
public:
	enum {
		A,
		B,
		C,
		X,
	};

	ABCX() {
		add_point(A, Vector2(0, 0));
		add_point(B, Vector2(1, 0));
		add_point(C, Vector2(0, 1));
		add_point(X, Vector2(1, 1));
		connect_points(A, B);
		connect_points(A, C);
		connect_points(B, C);
		connect_points(X, A);
	}

	// Disable heuristic completely; force the path A->B->C instead of A->C.
	real_t _compute_cost(int64_t p_from, int64_t p_to) {
		if (p_from == A && p_to == C) {
			return 1000;
		}
		return 100;
	}
};

TEST_CASE("[AStar2D] ABC path") {
	Ref<ABCX> abcx;
	abcx.instantiate();
	Vector<int64_t> path = abcx->get_id_path(ABCX::A, ABCX::C);
	REQUIRE(path.size() == 3);
	CHECK(path[0] == ABCX::A);
	CHECK(path[1] == ABCX::B);
	CHECK(path[2] == ABCX::C);
}

TEST_CASE("[AStar2D] ABCX path") {
	Ref<ABCX> abcx;
	abcx.instantiate();
	Vector<int64_t> path = abcx->get_id_path(ABCX::X, ABCX::C);
	REQUIRE(path.size() == 4);
	CHECK(path[0] == ABCX::X);
	CHECK(path[1] == ABCX::A);
	CHECK(path[2] == ABCX::B);
	CHECK(path[3] == ABCX::C);
}

TEST_CASE("[AStar2D] Add/Remove") {
	Ref<AStar2D> a;
	a.instantiate();

	// Manual tests.
	a->add_point(1, Vector2(0, 0));
	a->add_point(2, Vector2(0, 1));
	a->add_point(3, Vector2(1, 1));
	a->add_point(4, Vector2(2, 0));
	a->connect_points(1, 2, true);
	a->connect_points(1, 3, true);
	a->connect_points(1, 4, false);

	CHECK(a->are_points_connected(2, 1));
	CHECK(a->are_points_connected(4, 1));
	CHECK(a->are_points_connected(2, 1, false));
	CHECK_FALSE(a->are_points_connected(4, 1, false));

	a->disconnect_points(1, 2, true);
	CHECK(a->get_point_connections(1).size() == 2); // 3, 4
	CHECK(a->get_point_connections(2).size() == 0);

	a->disconnect_points(4, 1, false);
	CHECK(a->get_point_connections(1).size() == 2); // 3, 4
	CHECK(a->get_point_connections(4).size() == 0);

	a->disconnect_points(4, 1, true);
	CHECK(a->get_point_connections(1).size() == 1); // 3
	CHECK(a->get_point_connections(4).size() == 0);

	a->connect_points(2, 3, false);
	CHECK(a->get_point_connections(2).size() == 1); // 3
	CHECK(a->get_point_connections(3).size() == 1); // 1

	a->connect_points(2, 3, true);
	CHECK(a->get_point_connections(2).size() == 1); // 3
	CHECK(a->get_point_connections(3).size() == 2); // 1, 2

	a->disconnect_points(2, 3, false);
	CHECK(a->get_point_connections(2).size() == 0);
	CHECK(a->get_point_connections(3).size() == 2); // 1, 2

	a->connect_points(4, 3, true);
	CHECK(a->get_point_connections(3).size() == 3); // 1, 2, 4
	CHECK(a->get_point_connections(4).size() == 1); // 3

	a->disconnect_points(3, 4, false);
	CHECK(a->get_point_connections(3).size() == 2); // 1, 2
	CHECK(a->get_point_connections(4).size() == 1); // 3

	a->remove_point(3);
	CHECK(a->get_point_connections(1).size() == 0);
	CHECK(a->get_point_connections(2).size() == 0);
	CHECK(a->get_point_connections(4).size() == 0);

	a->add_point(0, Vector2(0, -1));
	a->add_point(3, Vector2(2, 1));
	// 0: (0, -1)
	// 1: (0, 0)
	// 2: (0, 1)
	// 3: (2, 1)
	// 4: (2, 0)

	// Tests for get_closest_position_in_segment.
	a->connect_points(2, 3);
	CHECK(a->get_closest_position_in_segment(Vector2(0.5, 0.5)) == Vector2(0.5, 1));

	a->connect_points(3, 4);
	a->connect_points(0, 3);
	a->connect_points(1, 4);
	a->disconnect_points(1, 4, false);
	a->disconnect_points(4, 3, false);
	a->disconnect_points(3, 4, false);
	// Remaining edges: <2, 3>, <0, 3>, <1, 4> (directed).
	CHECK(a->get_closest_position_in_segment(Vector2(2, 0.5)) == Vector2(1.75, 0.75));
	CHECK(a->get_closest_position_in_segment(Vector2(-1, 0.2)) == Vector2(0, 0));
	CHECK(a->get_closest_position_in_segment(Vector2(3, 2)) == Vector2(2, 1));

	Math::seed(0);

	// Random tests for connectivity checks.
	for (int i = 0; i < 20000; i++) {
		int u = Math::rand() % 5;
		int v = Math::rand() % 4;
		if (u == v) {
			v = 4;
		}
		if (Math::rand() % 2 == 1) {
			// Add a (possibly existing) directed edge and confirm connectivity.
			a->connect_points(u, v, false);
			CHECK(a->are_points_connected(u, v, false));
		} else {
			// Remove a (possibly nonexistent) directed edge and confirm disconnectivity.
			a->disconnect_points(u, v, false);
			CHECK_FALSE(a->are_points_connected(u, v, false));
		}
	}

	// Random tests for point removal.
	for (int i = 0; i < 20000; i++) {
		a->clear();
		for (int j = 0; j < 5; j++) {
			a->add_point(j, Vector2(0, 0));
		}

		// Add or remove random edges.
		for (int j = 0; j < 10; j++) {
			int u = Math::rand() % 5;
			int v = Math::rand() % 4;
			if (u == v) {
				v = 4;
			}
			if (Math::rand() % 2 == 1) {
				a->connect_points(u, v, false);
			} else {
				a->disconnect_points(u, v, false);
			}
		}

		// Remove point 0.
		a->remove_point(0);
		// White box: this will check all edges remaining in the segments set.
		for (int j = 1; j < 5; j++) {
			CHECK_FALSE(a->are_points_connected(0, j, true));
		}
	}
}

TEST_CASE("[AStar2D] Path from disabled point is empty") {
	Ref<AStar2D> a;
	a.instantiate();
	Vector2 p1(0, 0);
	Vector2 p2(0, 1);
	a->add_point(1, p1);
	a->add_point(2, p2);
	a->connect_points(1, 2);

	CHECK_EQ(a->get_id_path(1, 1), Vector<int64_t>{ 1 });
	CHECK_EQ(a->get_id_path(1, 2), Vector<int64_t>{ 1, 2 });

	CHECK_EQ(a->get_point_path(1, 1), Vector<Vector2>{ p1 });
	CHECK_EQ(a->get_point_path(1, 2), Vector<Vector2>{ p1, p2 });

	a->set_point_disabled(1, true);

	CHECK(a->get_id_path(1, 1).is_empty());
	CHECK(a->get_id_path(1, 2).is_empty());

	CHECK(a->get_point_path(1, 1).is_empty());
	CHECK(a->get_point_path(1, 2).is_empty());
}

TEST_CASE("[AStar2D] Empty graph has no path and no points") {
	Ref<AStar2D> a;
	a.instantiate();
	CHECK(a->get_point_count() == 0);
	CHECK_FALSE(a->has_point(1));
	CHECK(a->get_point_ids().is_empty());
	CHECK(a->get_id_path(1, 2).is_empty());
	CHECK(a->get_point_path(1, 2).is_empty());
}

TEST_CASE("[AStar2D] Point properties") {
	Ref<AStar2D> a;
	a.instantiate();
	a->add_point(1, Vector2(2, 3));
	a->add_point(2, Vector2(-1, 4), 2.5);

	CHECK(a->has_point(1));
	CHECK(a->has_point(2));
	CHECK_FALSE(a->has_point(3));
	CHECK(a->get_point_count() == 2);

	CHECK(a->get_point_position(1) == Vector2(2, 3));
	CHECK(a->get_point_position(2) == Vector2(-1, 4));
	CHECK(a->get_point_weight_scale(1) == doctest::Approx(1.0));
	CHECK(a->get_point_weight_scale(2) == doctest::Approx(2.5));

	a->set_point_position(1, Vector2(5, 6));
	CHECK(a->get_point_position(1) == Vector2(5, 6));

	a->set_point_weight_scale(2, 4.0);
	CHECK(a->get_point_weight_scale(2) == doctest::Approx(4.0));

	CHECK_FALSE(a->is_point_disabled(1));
	a->set_point_disabled(1, true);
	CHECK(a->is_point_disabled(1));
	a->set_point_disabled(1, false);
	CHECK_FALSE(a->is_point_disabled(1));
}

TEST_CASE("[AStar2D] Re-adding existing point updates its data") {
	Ref<AStar2D> a;
	a.instantiate();
	a->add_point(1, Vector2(0, 0), 1.0);
	a->add_point(1, Vector2(10, 20), 3.0);

	CHECK(a->get_point_count() == 1);
	CHECK(a->get_point_position(1) == Vector2(10, 20));
	CHECK(a->get_point_weight_scale(1) == doctest::Approx(3.0));
}

TEST_CASE("[AStar2D] get_available_point_id returns unused ids") {
	Ref<AStar2D> a;
	a.instantiate();
	CHECK(a->get_available_point_id() == 0);
	a->add_point(0, Vector2(0, 0));
	CHECK(a->get_available_point_id() == 1);
	a->add_point(1, Vector2(1, 0));
	CHECK(a->get_available_point_id() == 2);
	a->remove_point(0);
	// Lowest unused id should now be 0 again.
	CHECK(a->get_available_point_id() == 0);
}

TEST_CASE("[AStar2D] get_point_ids returns all added ids") {
	Ref<AStar2D> a;
	a.instantiate();
	a->add_point(10, Vector2(0, 0));
	a->add_point(20, Vector2(1, 0));
	a->add_point(30, Vector2(0, 1));

	PackedInt64Array ids = a->get_point_ids();
	REQUIRE(ids.size() == 3);
	// Order is not guaranteed; check via membership.
	CHECK(ids.has(10));
	CHECK(ids.has(20));
	CHECK(ids.has(30));
}

TEST_CASE("[AStar2D] get_closest_point honors include_disabled flag") {
	Ref<AStar2D> a;
	a.instantiate();
	a->add_point(1, Vector2(0, 0));
	a->add_point(2, Vector2(10, 0));
	a->add_point(3, Vector2(20, 0));

	CHECK(a->get_closest_point(Vector2(11, 0)) == 2);

	a->set_point_disabled(2, true);
	// Disabled point 2 is skipped; nearest enabled is 1 (distance 11) or 3 (distance 9).
	CHECK(a->get_closest_point(Vector2(11, 0)) == 3);
	// When include_disabled is true, the disabled point can still win.
	CHECK(a->get_closest_point(Vector2(11, 0), true) == 2);
}

TEST_CASE("[AStar2D] get_closest_point on empty graph returns -1") {
	Ref<AStar2D> a;
	a.instantiate();
	CHECK(a->get_closest_point(Vector2(0, 0)) == -1);
}

TEST_CASE("[AStar2D] clear removes all points and edges") {
	Ref<AStar2D> a;
	a.instantiate();
	a->add_point(1, Vector2(0, 0));
	a->add_point(2, Vector2(1, 0));
	a->connect_points(1, 2);
	CHECK(a->get_point_count() == 2);

	a->clear();
	CHECK(a->get_point_count() == 0);
	CHECK_FALSE(a->has_point(1));
	CHECK_FALSE(a->has_point(2));
	CHECK(a->get_point_ids().is_empty());
}

TEST_CASE("[AStar2D] reserve_space increases capacity") {
	Ref<AStar2D> a;
	a.instantiate();
	int64_t initial_capacity = a->get_point_capacity();
	a->reserve_space(initial_capacity + 128);
	CHECK(a->get_point_capacity() >= initial_capacity + 128);
}

TEST_CASE("[AStar2D] Path uses point weight scale") {
	// Two parallel routes from start (0) to end (3).
	// Upper route: 0 -> 1 -> 3 with point 1 having a high weight_scale.
	// Lower route: 0 -> 2 -> 3 with point 2 having default weight_scale.
	// The lower route should be chosen.
	Ref<AStar2D> a;
	a.instantiate();
	a->add_point(0, Vector2(0, 0));
	a->add_point(1, Vector2(1, 1), 100.0);
	a->add_point(2, Vector2(1, -1), 1.0);
	a->add_point(3, Vector2(2, 0));
	a->connect_points(0, 1);
	a->connect_points(1, 3);
	a->connect_points(0, 2);
	a->connect_points(2, 3);

	Vector<int64_t> path = a->get_id_path(0, 3);
	REQUIRE(path.size() == 3);
	CHECK(path[0] == 0);
	CHECK(path[1] == 2);
	CHECK(path[2] == 3);
}

TEST_CASE("[AStar2D] Disconnected components have no path") {
	Ref<AStar2D> a;
	a.instantiate();
	a->add_point(1, Vector2(0, 0));
	a->add_point(2, Vector2(1, 0));
	a->add_point(3, Vector2(10, 10));
	a->add_point(4, Vector2(11, 10));
	a->connect_points(1, 2);
	a->connect_points(3, 4);

	CHECK(a->get_id_path(1, 4).is_empty());
	CHECK(a->get_point_path(1, 4).is_empty());
}

TEST_CASE("[AStar2D] Partial path returns closest reachable point") {
	Ref<AStar2D> a;
	a.instantiate();
	a->add_point(1, Vector2(0, 0));
	a->add_point(2, Vector2(1, 0));
	a->add_point(3, Vector2(10, 10));
	a->connect_points(1, 2);
	// Point 3 is unreachable from 1.

	CHECK(a->get_id_path(1, 3, false).is_empty());

	Vector<int64_t> partial = a->get_id_path(1, 3, true);
	REQUIRE(partial.size() == 2);
	CHECK(partial[0] == 1);
	CHECK(partial[1] == 2);

	Vector<Vector2> partial_points = a->get_point_path(1, 3, true);
	REQUIRE(partial_points.size() == 2);
	CHECK(partial_points[0] == Vector2(0, 0));
	CHECK(partial_points[1] == Vector2(1, 0));
}

TEST_CASE("[AStar2D] Path through disabled intermediate point is skipped") {
	// A direct route 0 -> 1 -> 3 exists, plus a detour 0 -> 2 -> 3.
	// Disabling point 1 should force the detour through 2.
	Ref<AStar2D> a;
	a.instantiate();
	a->add_point(0, Vector2(0, 0));
	a->add_point(1, Vector2(1, 0));
	a->add_point(2, Vector2(1, 5));
	a->add_point(3, Vector2(2, 0));
	a->connect_points(0, 1);
	a->connect_points(1, 3);
	a->connect_points(0, 2);
	a->connect_points(2, 3);

	a->set_point_disabled(1, true);

	Vector<int64_t> path = a->get_id_path(0, 3);
	REQUIRE(path.size() == 3);
	CHECK(path[0] == 0);
	CHECK(path[1] == 2);
	CHECK(path[2] == 3);
}

TEST_CASE("[AStar2D] Neighbor filter toggle") {
	Ref<AStar2D> a;
	a.instantiate();
	CHECK_FALSE(a->is_neighbor_filter_enabled());
	a->set_neighbor_filter_enabled(true);
	CHECK(a->is_neighbor_filter_enabled());
	a->set_neighbor_filter_enabled(false);
	CHECK_FALSE(a->is_neighbor_filter_enabled());
}

TEST_CASE("[AStar2D] Invalid input is rejected without modifying state") {
	Ref<AStar2D> a;
	a.instantiate();
	a->add_point(1, Vector2(0, 0));
	a->add_point(2, Vector2(1, 0));
	a->connect_points(1, 2);

	ERR_PRINT_OFF;

	// Negative id is rejected by add_point.
	a->add_point(-1, Vector2(5, 5));
	CHECK(a->get_point_count() == 2);
	CHECK_FALSE(a->has_point(-1));

	// Negative weight scale is rejected by add_point.
	a->add_point(3, Vector2(2, 0), -1.0);
	CHECK_FALSE(a->has_point(3));
	CHECK(a->get_point_count() == 2);

	// Negative weight scale is rejected by set_point_weight_scale; existing scale is preserved.
	a->set_point_weight_scale(1, -2.5);
	CHECK(a->get_point_weight_scale(1) == doctest::Approx(1.0));

	// Operations on non-existent points return safe defaults / no-op.
	CHECK(a->get_point_position(999) == Vector2());
	CHECK(a->get_point_weight_scale(999) == doctest::Approx(0.0));
	CHECK(a->get_point_connections(999).is_empty());
	CHECK_FALSE(a->is_point_disabled(999));

	a->set_point_position(999, Vector2(7, 7)); // No-op.
	a->set_point_weight_scale(999, 5.0); // No-op.
	a->set_point_disabled(999, true); // No-op.
	a->remove_point(999); // No-op.
	CHECK(a->get_point_count() == 2);

	// Self-connection is rejected.
	a->connect_points(1, 1);
	CHECK_FALSE(a->are_points_connected(1, 1, false));

	// Connecting/disconnecting with a non-existent point is rejected; existing edge survives.
	a->connect_points(1, 999);
	a->disconnect_points(1, 999);
	CHECK(a->are_points_connected(1, 2));

	// Pathfinding with invalid endpoints returns an empty result.
	CHECK(a->get_id_path(1, 999).is_empty());
	CHECK(a->get_id_path(999, 2).is_empty());
	CHECK(a->get_point_path(1, 999).is_empty());
	CHECK(a->get_point_path(999, 2).is_empty());

	// reserve_space rejects non-positive capacity (capacity unchanged).
	int64_t cap = a->get_point_capacity();
	a->reserve_space(0);
	a->reserve_space(-10);
	CHECK(a->get_point_capacity() == cap);

	ERR_PRINT_ON;
}

} // namespace TestAStar2D
