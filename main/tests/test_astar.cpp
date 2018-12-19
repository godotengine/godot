/*************************************************************************/
/*  test_astar.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

typedef bool (*TestFunc)(void);

TestFunc test_funcs[] = {
	test_abc,
	test_abcx,
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
