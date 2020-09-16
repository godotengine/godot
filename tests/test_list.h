/*************************************************************************/
/*  test_list.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_LIST_H
#define TEST_LIST_H

#include "core/list.h"

#include "tests/test_macros.h"

namespace TestList {

static void populate_integers(List<int> &p_list, List<int>::Element *r_elements[], int num_elements) {
	p_list.clear();
	for (int i = 0; i < num_elements; ++i) {
		List<int>::Element *n = p_list.push_back(i);
		r_elements[i] = n;
	}
}

TEST_CASE("[List] Swap adjacent front and back") {
	List<int> list;
	List<int>::Element *n[2];
	populate_integers(list, n, 2);

	list.swap(list.front(), list.back());

	CHECK(list.front()->prev() == nullptr);
	CHECK(list.front() != list.front()->next());

	CHECK(list.front() == n[1]);
	CHECK(list.back() == n[0]);

	CHECK(list.back()->next() == nullptr);
	CHECK(list.back() != list.back()->prev());
}

TEST_CASE("[List] Swap first adjacent pair") {
	List<int> list;
	List<int>::Element *n[4];
	populate_integers(list, n, 4);

	list.swap(n[0], n[1]);

	CHECK(list.front()->prev() == nullptr);
	CHECK(list.front() != list.front()->next());

	CHECK(list.front() == n[1]);
	CHECK(list.front()->next() == n[0]);
	CHECK(list.back()->prev() == n[2]);
	CHECK(list.back() == n[3]);

	CHECK(list.back()->next() == nullptr);
	CHECK(list.back() != list.back()->prev());
}

TEST_CASE("[List] Swap middle adjacent pair") {
	List<int> list;
	List<int>::Element *n[4];
	populate_integers(list, n, 4);

	list.swap(n[1], n[2]);

	CHECK(list.front()->prev() == nullptr);

	CHECK(list.front() == n[0]);
	CHECK(list.front()->next() == n[2]);
	CHECK(list.back()->prev() == n[1]);
	CHECK(list.back() == n[3]);

	CHECK(list.back()->next() == nullptr);
}

TEST_CASE("[List] Swap last adjacent pair") {
	List<int> list;
	List<int>::Element *n[4];
	populate_integers(list, n, 4);

	list.swap(n[2], n[3]);

	CHECK(list.front()->prev() == nullptr);

	CHECK(list.front() == n[0]);
	CHECK(list.front()->next() == n[1]);
	CHECK(list.back()->prev() == n[3]);
	CHECK(list.back() == n[2]);

	CHECK(list.back()->next() == nullptr);
}

TEST_CASE("[List] Swap first cross pair") {
	List<int> list;
	List<int>::Element *n[4];
	populate_integers(list, n, 4);

	list.swap(n[0], n[2]);

	CHECK(list.front()->prev() == nullptr);

	CHECK(list.front() == n[2]);
	CHECK(list.front()->next() == n[1]);
	CHECK(list.back()->prev() == n[0]);
	CHECK(list.back() == n[3]);

	CHECK(list.back()->next() == nullptr);
}

TEST_CASE("[List] Swap last cross pair") {
	List<int> list;
	List<int>::Element *n[4];
	populate_integers(list, n, 4);

	list.swap(n[1], n[3]);

	CHECK(list.front()->prev() == nullptr);

	CHECK(list.front() == n[0]);
	CHECK(list.front()->next() == n[3]);
	CHECK(list.back()->prev() == n[2]);
	CHECK(list.back() == n[1]);

	CHECK(list.back()->next() == nullptr);
}

TEST_CASE("[List] Swap edges") {
	List<int> list;
	List<int>::Element *n[4];
	populate_integers(list, n, 4);

	list.swap(n[1], n[3]);

	CHECK(list.front()->prev() == nullptr);

	CHECK(list.front() == n[0]);
	CHECK(list.front()->next() == n[3]);
	CHECK(list.back()->prev() == n[2]);
	CHECK(list.back() == n[1]);

	CHECK(list.back()->next() == nullptr);
}

TEST_CASE("[List] Swap middle (values check)") {
	List<String> list;
	List<String>::Element *n_str1 = list.push_back("Still");
	List<String>::Element *n_str2 = list.push_back("waiting");
	List<String>::Element *n_str3 = list.push_back("for");
	List<String>::Element *n_str4 = list.push_back("Godot.");

	CHECK(n_str1->get() == "Still");
	CHECK(n_str4->get() == "Godot.");

	CHECK(list.front()->get() == "Still");
	CHECK(list.front()->next()->get() == "waiting");
	CHECK(list.back()->prev()->get() == "for");
	CHECK(list.back()->get() == "Godot.");

	list.swap(n_str2, n_str3);

	CHECK(list.front()->next()->get() == "for");
	CHECK(list.back()->prev()->get() == "waiting");
}

TEST_CASE("[List] Swap front and back (values check)") {
	List<Variant> list;
	Variant str = "Godot";
	List<Variant>::Element *n_str = list.push_back(str);
	Variant color = Color(0, 0, 1);
	List<Variant>::Element *n_color = list.push_back(color);

	CHECK(list.front()->get() == "Godot");
	CHECK(list.back()->get() == Color(0, 0, 1));

	list.swap(n_str, n_color);

	CHECK(list.front()->get() == Color(0, 0, 1));
	CHECK(list.back()->get() == "Godot");
}

TEST_CASE("[List] Swap adjacent back and front (reverse order of elements)") {
	List<int> list;
	List<int>::Element *n[2];
	populate_integers(list, n, 2);

	list.swap(n[1], n[0]);

	List<int>::Element *it = list.front();
	while (it) {
		List<int>::Element *prev_it = it;
		it = it->next();
		if (it == prev_it) {
			FAIL_CHECK("Infinite loop detected.");
			break;
		}
	}
}

static void swap_random(List<int> &p_list, List<int>::Element *r_elements[], size_t p_size, size_t p_iterations) {
	Math::seed(0);

	for (size_t test_i = 0; test_i < p_iterations; ++test_i) {
		// A and B elements have corresponding indices as values.
		const int a_idx = static_cast<int>(Math::rand() % p_size);
		const int b_idx = static_cast<int>(Math::rand() % p_size);
		List<int>::Element *a = p_list.find(a_idx); // via find.
		List<int>::Element *b = r_elements[b_idx]; // via pointer.

		int va = a->get();
		int vb = b->get();

		p_list.swap(a, b);

		CHECK(va == a->get());
		CHECK(vb == b->get());

		size_t element_count = 0;

		// Fully traversable after swap?
		List<int>::Element *it = p_list.front();
		while (it) {
			element_count += 1;
			List<int>::Element *prev_it = it;
			it = it->next();
			if (it == prev_it) {
				FAIL_CHECK("Infinite loop detected.");
				break;
			}
		}
		// We should not lose anything in the process.
		if (element_count != p_size) {
			FAIL_CHECK("Element count mismatch.");
			break;
		}
	}
}

TEST_CASE("[Stress][List] Swap random 100 elements, 500 iterations.") {
	List<int> list;
	List<int>::Element *n[100];
	populate_integers(list, n, 100);
	swap_random(list, n, 100, 500);
}

TEST_CASE("[Stress][List] Swap random 10 elements, 1000 iterations.") {
	List<int> list;
	List<int>::Element *n[10];
	populate_integers(list, n, 10);
	swap_random(list, n, 10, 1000);
}

} // namespace TestList

#endif // TEST_LIST_H
