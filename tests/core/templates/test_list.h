/**************************************************************************/
/*  test_list.h                                                           */
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

#include "core/templates/list.h"

#include "tests/test_macros.h"

namespace TestList {

static void populate_integers(List<int> &p_list, List<int>::Element *r_elements[], int num_elements) {
	p_list.clear();
	for (int i = 0; i < num_elements; ++i) {
		List<int>::Element *n = p_list.push_back(i);
		r_elements[i] = n;
	}
}

TEST_CASE("[List] List initialization") {
	List<int> list{ 0, 1, 2, 3, 4 };

	CHECK(list.size() == 5);
	CHECK(list.get(0) == 0);
	CHECK(list.get(1) == 1);
	CHECK(list.get(2) == 2);
	CHECK(list.get(3) == 3);
	CHECK(list.get(4) == 4);
}

TEST_CASE("[List] Push/pop back") {
	List<String> list;

	List<String>::Element *n;
	n = list.push_back("A");
	CHECK(n->get() == "A");
	n = list.push_back("B");
	CHECK(n->get() == "B");
	n = list.push_back("C");
	CHECK(n->get() == "C");

	CHECK(list.size() == 3);
	CHECK(!list.is_empty());

	String v;
	v = list.back()->get();
	list.pop_back();
	CHECK(v == "C");
	v = list.back()->get();
	list.pop_back();
	CHECK(v == "B");
	v = list.back()->get();
	list.pop_back();
	CHECK(v == "A");

	CHECK(list.size() == 0);
	CHECK(list.is_empty());

	CHECK(list.back() == nullptr);
	CHECK(list.front() == nullptr);
}

TEST_CASE("[List] Push/pop front") {
	List<String> list;

	List<String>::Element *n;
	n = list.push_front("A");
	CHECK(n->get() == "A");
	n = list.push_front("B");
	CHECK(n->get() == "B");
	n = list.push_front("C");
	CHECK(n->get() == "C");

	CHECK(list.size() == 3);
	CHECK(!list.is_empty());

	String v;
	v = list.front()->get();
	list.pop_front();
	CHECK(v == "C");
	v = list.front()->get();
	list.pop_front();
	CHECK(v == "B");
	v = list.front()->get();
	list.pop_front();
	CHECK(v == "A");

	CHECK(list.size() == 0);
	CHECK(list.is_empty());

	CHECK(list.back() == nullptr);
	CHECK(list.front() == nullptr);
}

TEST_CASE("[List] Set and get") {
	List<String> list;
	list.push_back("A");

	List<String>::Element *n = list.front();
	CHECK(n->get() == "A");

	n->set("X");
	CHECK(n->get() == "X");
}

TEST_CASE("[List] Insert before") {
	List<String> list;
	List<String>::Element *a = list.push_back("A");
	List<String>::Element *b = list.push_back("B");
	List<String>::Element *c = list.push_back("C");

	list.insert_before(b, "I");

	CHECK(a->next()->get() == "I");
	CHECK(c->prev()->prev()->get() == "I");
	CHECK(list.front()->next()->get() == "I");
	CHECK(list.back()->prev()->prev()->get() == "I");
}

TEST_CASE("[List] Insert after") {
	List<String> list;
	List<String>::Element *a = list.push_back("A");
	List<String>::Element *b = list.push_back("B");
	List<String>::Element *c = list.push_back("C");

	list.insert_after(b, "I");

	CHECK(a->next()->next()->get() == "I");
	CHECK(c->prev()->get() == "I");
	CHECK(list.front()->next()->next()->get() == "I");
	CHECK(list.back()->prev()->get() == "I");
}

TEST_CASE("[List] Insert before null") {
	List<String> list;
	List<String>::Element *a = list.push_back("A");
	List<String>::Element *b = list.push_back("B");
	List<String>::Element *c = list.push_back("C");

	list.insert_before(nullptr, "I");

	CHECK(a->next()->get() == "B");
	CHECK(b->get() == "B");
	CHECK(c->prev()->prev()->get() == "A");
	CHECK(list.front()->next()->get() == "B");
	CHECK(list.back()->prev()->prev()->get() == "B");
	CHECK(list.back()->get() == "I");
}

TEST_CASE("[List] Insert after null") {
	List<String> list;
	List<String>::Element *a = list.push_back("A");
	List<String>::Element *b = list.push_back("B");
	List<String>::Element *c = list.push_back("C");

	list.insert_after(nullptr, "I");

	CHECK(a->next()->get() == "B");
	CHECK(b->get() == "B");
	CHECK(c->prev()->prev()->get() == "A");
	CHECK(list.front()->next()->get() == "B");
	CHECK(list.back()->prev()->prev()->get() == "B");
	CHECK(list.back()->get() == "I");
}

TEST_CASE("[List] Find") {
	List<int> list;
	List<int>::Element *n[10];
	// Indices match values.
	populate_integers(list, n, 10);

	for (int i = 0; i < 10; ++i) {
		CHECK(n[i]->get() == list.find(i)->get());
	}
}

TEST_CASE("[List] Erase (by value)") {
	List<int> list;
	List<int>::Element *n[4];
	// Indices match values.
	populate_integers(list, n, 4);

	CHECK(list.front()->next()->next()->get() == 2);
	bool erased = list.erase(2); // 0, 1, 3.
	CHECK(erased);
	CHECK(list.size() == 3);

	// The pointer n[2] points to the freed memory which is not reset to zero,
	// so the below assertion may pass, but this relies on undefined behavior.
	// CHECK(n[2]->get() == 2);

	CHECK(list.front()->get() == 0);
	CHECK(list.front()->next()->next()->get() == 3);
	CHECK(list.back()->get() == 3);
	CHECK(list.back()->prev()->get() == 1);

	CHECK(n[1]->next()->get() == 3);
	CHECK(n[3]->prev()->get() == 1);

	erased = list.erase(9000); // Doesn't exist.
	CHECK(!erased);
}

TEST_CASE("[List] Erase (by element)") {
	List<int> list;
	List<int>::Element *n[4];
	// Indices match values.
	populate_integers(list, n, 4);

	bool erased = list.erase(n[2]);
	CHECK(erased);
	CHECK(list.size() == 3);
	CHECK(n[1]->next()->get() == 3);
	CHECK(n[3]->prev()->get() == 1);
}

TEST_CASE("[List] Element erase") {
	List<int> list;
	List<int>::Element *n[4];
	// Indices match values.
	populate_integers(list, n, 4);

	n[2]->erase();

	CHECK(list.size() == 3);
	CHECK(n[1]->next()->get() == 3);
	CHECK(n[3]->prev()->get() == 1);
}

TEST_CASE("[List] Clear") {
	List<int> list;
	List<int>::Element *n[100];
	populate_integers(list, n, 100);

	list.clear();

	CHECK(list.size() == 0);
	CHECK(list.is_empty());
}

TEST_CASE("[List] Invert") {
	List<int> list;
	List<int>::Element *n[4];
	populate_integers(list, n, 4);

	list.reverse();

	CHECK(list.front()->get() == 3);
	CHECK(list.front()->next()->get() == 2);
	CHECK(list.back()->prev()->get() == 1);
	CHECK(list.back()->get() == 0);
}

TEST_CASE("[List] Move to front") {
	List<int> list;
	List<int>::Element *n[4];
	populate_integers(list, n, 4);

	list.move_to_front(n[3]);

	CHECK(list.front()->get() == 3);
	CHECK(list.back()->get() == 2);
}

TEST_CASE("[List] Move to back") {
	List<int> list;
	List<int>::Element *n[4];
	populate_integers(list, n, 4);

	list.move_to_back(n[0]);

	CHECK(list.back()->get() == 0);
	CHECK(list.front()->get() == 1);
}

TEST_CASE("[List] Move before") {
	List<int> list;
	List<int>::Element *n[4];
	populate_integers(list, n, 4);

	list.move_before(n[3], n[1]);

	CHECK(list.front()->next()->get() == n[3]->get());
}

template <typename T>
static void compare_lists(const List<T> &p_result, const List<T> &p_expected) {
	CHECK_EQ(p_result.size(), p_expected.size());
	const typename List<T>::Element *result_it = p_result.front();
	const typename List<T>::Element *expected_it = p_expected.front();
	for (int i = 0; i < p_result.size(); i++) {
		CHECK(result_it);
		CHECK(expected_it);
		CHECK_EQ(result_it->get(), expected_it->get());
		result_it = result_it->next();
		expected_it = expected_it->next();
	}
	CHECK(!result_it);
	CHECK(!expected_it);

	result_it = p_result.back();
	expected_it = p_expected.back();
	for (int i = 0; i < p_result.size(); i++) {
		CHECK(result_it);
		CHECK(expected_it);
		CHECK_EQ(result_it->get(), expected_it->get());
		result_it = result_it->prev();
		expected_it = expected_it->prev();
	}
	CHECK(!result_it);
	CHECK(!expected_it);
}

TEST_CASE("[List] Sort") {
	List<String> result{ "D", "B", "A", "C" };
	result.sort();
	List<String> expected{ "A", "B", "C", "D" };
	compare_lists(result, expected);

	List<int> empty_result{};
	empty_result.sort();
	List<int> empty_expected{};
	compare_lists(empty_result, empty_expected);

	List<int> one_result{ 1 };
	one_result.sort();
	List<int> one_expected{ 1 };
	compare_lists(one_result, one_expected);

	List<float> reversed_result{ 2.0, 1.5, 1.0 };
	reversed_result.sort();
	List<float> reversed_expected{ 1.0, 1.5, 2.0 };
	compare_lists(reversed_result, reversed_expected);

	List<int> already_sorted_result{ 1, 2, 3, 4, 5 };
	already_sorted_result.sort();
	List<int> already_sorted_expected{ 1, 2, 3, 4, 5 };
	compare_lists(already_sorted_result, already_sorted_expected);

	List<int> with_duplicates_result{ 1, 2, 3, 1, 2, 3 };
	with_duplicates_result.sort();
	List<int> with_duplicates_expected{ 1, 1, 2, 2, 3, 3 };
	compare_lists(with_duplicates_result, with_duplicates_expected);
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
} // namespace TestList
