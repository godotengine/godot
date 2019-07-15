/*************************************************************************/
/*  test_astar.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "test_astar.h"

#include "core/math/a_star.h"
#include "core/os/os.h"

#include <stdio.h>

namespace TestAStar {

class ABCX : public AStar {
public:
	enum { A,
		B,
		C,
		X };

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

	// Disable heuristic completely
	float _compute_cost(int p_from, int p_to) {
		if (p_from == A && p_to == C) {
			return 1000;
		}
		return 100;
	}
};

bool test_abc() {
	ABCX abcx;
	PoolVector<int> path = abcx.get_id_path(ABCX::A, ABCX::C);
	bool ok = path.size() == 3;
	int i = 0;
	ok = ok && path[i++] == ABCX::A;
	ok = ok && path[i++] == ABCX::B;
	ok = ok && path[i++] == ABCX::C;
	return ok;
}

bool test_abcx() {
	ABCX abcx;
	PoolVector<int> path = abcx.get_id_path(ABCX::X, ABCX::C);
	bool ok = path.size() == 4;
	int i = 0;
	ok = ok && path[i++] == ABCX::X;
	ok = ok && path[i++] == ABCX::A;
	ok = ok && path[i++] == ABCX::B;
	ok = ok && path[i++] == ABCX::C;
	return ok;
}

bool test_add_remove() {
	AStar a;
	bool ok = true;

	// Manual tests
	a.add_point(1, Vector3(0, 0, 0));
	a.add_point(2, Vector3(0, 1, 0));
	a.add_point(3, Vector3(1, 1, 0));
	a.add_point(4, Vector3(2, 0, 0));
	a.connect_points(1, 2, true);
	a.connect_points(1, 3, true);
	a.connect_points(1, 4, false);

	ok = ok && (a.are_points_connected(2, 1) == true);
	ok = ok && (a.are_points_connected(4, 1) == true);
	ok = ok && (a.are_points_connected(2, 1, false) == true);
	ok = ok && (a.are_points_connected(4, 1, false) == false);

	a.disconnect_points(1, 2, true);
	ok = ok && (a.get_point_connections(1).size() == 2); // 3, 4
	ok = ok && (a.get_point_connections(2).size() == 0);

	a.disconnect_points(4, 1, false);
	ok = ok && (a.get_point_connections(1).size() == 2); // 3, 4
	ok = ok && (a.get_point_connections(4).size() == 0);

	a.disconnect_points(4, 1, true);
	ok = ok && (a.get_point_connections(1).size() == 1); // 3
	ok = ok && (a.get_point_connections(4).size() == 0);

	a.connect_points(2, 3, false);
	ok = ok && (a.get_point_connections(2).size() == 1); // 3
	ok = ok && (a.get_point_connections(3).size() == 1); // 1

	a.connect_points(2, 3, true);
	ok = ok && (a.get_point_connections(2).size() == 1); // 3
	ok = ok && (a.get_point_connections(3).size() == 2); // 1, 2

	a.disconnect_points(2, 3, false);
	ok = ok && (a.get_point_connections(2).size() == 0);
	ok = ok && (a.get_point_connections(3).size() == 2); // 1, 2

	a.connect_points(4, 3, true);
	ok = ok && (a.get_point_connections(3).size() == 3); // 1, 2, 4
	ok = ok && (a.get_point_connections(4).size() == 1); // 3

	a.disconnect_points(3, 4, false);
	ok = ok && (a.get_point_connections(3).size() == 2); // 1, 2
	ok = ok && (a.get_point_connections(4).size() == 1); // 3

	a.remove_point(3);
	ok = ok && (a.get_point_connections(1).size() == 0);
	ok = ok && (a.get_point_connections(2).size() == 0);
	ok = ok && (a.get_point_connections(4).size() == 0);

	a.add_point(0, Vector3(0, -1, 0));
	a.add_point(3, Vector3(2, 1, 0));
	// 0: (0, -1)
	// 1: (0, 0)
	// 2: (0, 1)
	// 3: (2, 1)
	// 4: (2, 0)

	// Tests for get_closest_position_in_segment
	a.connect_points(2, 3);
	ok = ok && (a.get_closest_position_in_segment(Vector3(0.5, 0.5, 0)) == Vector3(0.5, 1, 0));

	a.connect_points(3, 4);
	a.connect_points(0, 3);
	a.connect_points(1, 4);
	a.disconnect_points(1, 4, false);
	a.disconnect_points(4, 3, false);
	a.disconnect_points(3, 4, false);
	// Remaining edges: <2, 3>, <0, 3>, <1, 4> (directed)
	ok = ok && (a.get_closest_position_in_segment(Vector3(2, 0.5, 0)) == Vector3(1.75, 0.75, 0));
	ok = ok && (a.get_closest_position_in_segment(Vector3(-1, 0.2, 0)) == Vector3(0, 0, 0));
	ok = ok && (a.get_closest_position_in_segment(Vector3(3, 2, 0)) == Vector3(2, 1, 0));

	int seed = 0;

	// Random tests for connectivity checks
	for (int i = 0; i < 20000; i++) {
		seed = (seed * 1103515245 + 12345) & 0x7fffffff;
		int u = (seed / 5) % 5;
		int v = seed % 5;
		if (u == v) {
			i--;
			continue;
		}
		if (seed % 2 == 1) {
			// Add a (possibly existing) directed edge and confirm connectivity
			a.connect_points(u, v, false);
			ok = ok && (a.are_points_connected(u, v, false) == true);
		} else {
			// Remove a (possibly nonexistent) directed edge and confirm disconnectivity
			a.disconnect_points(u, v, false);
			ok = ok && (a.are_points_connected(u, v, false) == false);
		}
	}

	// Random tests for point removal
	for (int i = 0; i < 20000; i++) {
		a.clear();
		for (int j = 0; j < 5; j++)
			a.add_point(j, Vector3(0, 0, 0));

		// Add or remove random edges
		for (int j = 0; j < 10; j++) {
			seed = (seed * 1103515245 + 12345) & 0x7fffffff;
			int u = (seed / 5) % 5;
			int v = seed % 5;
			if (u == v) {
				j--;
				continue;
			}
			if (seed % 2 == 1)
				a.connect_points(u, v, false);
			else
				a.disconnect_points(u, v, false);
		}

		// Remove point 0
		a.remove_point(0);
		// White box: this will check all edges remaining in the segments set
		for (int j = 1; j < 5; j++) {
			ok = ok && (a.are_points_connected(0, j, true) == false);
		}
	}

	// It's been great work, cheers \(^ ^)/
	return ok;
}

typedef bool (*TestFunc)(void);

TestFunc test_funcs[] = {
	test_abc,
	test_abcx,
	test_add_remove,
	NULL
};

MainLoop *test() {
	int count = 0;
	int passed = 0;

	while (true) {
		if (!test_funcs[count])
			break;
		bool pass = test_funcs[count]();
		if (pass)
			passed++;
		OS::get_singleton()->print("\t%s\n", pass ? "PASS" : "FAILED");

		count++;
	}
	OS::get_singleton()->print("\n");
	OS::get_singleton()->print("Passed %i of %i tests\n", passed, count);
	return NULL;
}

} // namespace TestAStar
