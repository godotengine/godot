/**************************************************************************/
/*  test_disjoint_set.cpp                                                 */
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

TEST_FORCE_LINK(test_disjoint_set)

#include "core/math/disjoint_set.h"

namespace TestDisjointSet {

TEST_CASE("[DisjointSet] Insert single element") {
	DisjointSet<int> ds;
	ds.insert(1);

	Vector<int> reps;
	ds.get_representatives(reps);
	CHECK_MESSAGE(
			reps.size() == 1,
			"A single inserted element should produce exactly 1 representative.");
	CHECK_MESSAGE(
			reps[0] == 1,
			"The representative should be the inserted element.");
}

TEST_CASE("[DisjointSet] Insert multiple elements") {
	DisjointSet<int> ds;
	ds.insert(1);
	ds.insert(2);
	ds.insert(3);

	Vector<int> reps;
	ds.get_representatives(reps);
	CHECK_MESSAGE(
			reps.size() == 3,
			"Three inserted elements without unions should produce 3 representatives.");
}

TEST_CASE("[DisjointSet] Duplicate insert is idempotent") {
	DisjointSet<int> ds;
	ds.insert(1);
	ds.insert(1);

	Vector<int> reps;
	ds.get_representatives(reps);
	CHECK_MESSAGE(
			reps.size() == 1,
			"Inserting the same element twice should still produce 1 representative.");
}

TEST_CASE("[DisjointSet] Union two elements") {
	DisjointSet<int> ds;
	ds.create_union(1, 2);

	Vector<int> reps;
	ds.get_representatives(reps);
	CHECK_MESSAGE(
			reps.size() == 1,
			"Two elements united should produce 1 representative.");
}

TEST_CASE("[DisjointSet] Union is transitive") {
	DisjointSet<int> ds;
	ds.create_union(1, 2);
	ds.create_union(2, 3);

	Vector<int> reps;
	ds.get_representatives(reps);
	CHECK_MESSAGE(
			reps.size() == 1,
			"Transitive unions (1-2, 2-3) should produce 1 representative.");

	Vector<int> members;
	ds.get_members(members, reps[0]);
	CHECK_MESSAGE(
			members.size() == 3,
			"All 3 elements should be members of the same set.");
}

TEST_CASE("[DisjointSet] Union of already-united elements") {
	DisjointSet<int> ds;
	ds.create_union(1, 2);
	ds.create_union(1, 2);

	Vector<int> reps;
	ds.get_representatives(reps);
	CHECK_MESSAGE(
			reps.size() == 1,
			"Redundant union should not change the number of representatives.");
}

TEST_CASE("[DisjointSet] Multiple disjoint sets") {
	DisjointSet<int> ds;
	ds.create_union(1, 2);
	ds.create_union(3, 4);

	Vector<int> reps;
	ds.get_representatives(reps);
	CHECK_MESSAGE(
			reps.size() == 2,
			"Two separate unions should produce 2 representatives.");
}

TEST_CASE("[DisjointSet] Merging two groups") {
	DisjointSet<int> ds;
	ds.create_union(1, 2);
	ds.create_union(3, 4);
	ds.create_union(2, 3);

	Vector<int> reps;
	ds.get_representatives(reps);
	CHECK_MESSAGE(
			reps.size() == 1,
			"Bridging two groups should merge them into 1 representative.");

	Vector<int> members;
	ds.get_members(members, reps[0]);
	CHECK_MESSAGE(
			members.size() == 4,
			"All 4 elements should be in the merged set.");
}

TEST_CASE("[DisjointSet] Members of a single-element set") {
	DisjointSet<int> ds;
	ds.insert(42);

	Vector<int> members;
	ds.get_members(members, 42);
	CHECK_MESSAGE(
			members.size() == 1,
			"A lone element's set should contain exactly itself.");
	CHECK_MESSAGE(
			members[0] == 42,
			"The sole member should be the element itself.");
}

TEST_CASE("[DisjointSet] Members after union") {
	DisjointSet<int> ds;
	ds.create_union(10, 20);

	Vector<int> reps;
	ds.get_representatives(reps);
	REQUIRE(reps.size() == 1);

	Vector<int> members;
	ds.get_members(members, reps[0]);
	CHECK_MESSAGE(
			members.size() == 2,
			"United set should have 2 members.");
	CHECK_MESSAGE(
			members.find(10) != -1,
			"Members should include 10.");
	CHECK_MESSAGE(
			members.find(20) != -1,
			"Members should include 20.");
}

TEST_CASE("[DisjointSet] Members of multiple disjoint sets") {
	DisjointSet<int> ds;
	ds.create_union(1, 2);
	ds.create_union(3, 4);

	Vector<int> reps;
	ds.get_representatives(reps);
	REQUIRE(reps.size() == 2);

	for (int i = 0; i < reps.size(); i++) {
		Vector<int> members;
		ds.get_members(members, reps[i]);
		CHECK_MESSAGE(
				members.size() == 2,
				"Each disjoint set should have exactly 2 members.");
	}
}

TEST_CASE("[DisjointSet] get_members with non-representative fails gracefully") {
	DisjointSet<int> ds;
	ds.create_union(1, 2);

	// Find the non-representative element.
	Vector<int> reps;
	ds.get_representatives(reps);
	REQUIRE(reps.size() == 1);
	int non_rep = (reps[0] == 1) ? 2 : 1;

	Vector<int> members;
	ERR_PRINT_OFF;
	ds.get_members(members, non_rep);
	ERR_PRINT_ON;
	CHECK_MESSAGE(
			members.size() == 0,
			"get_members with a non-representative should produce no output.");
}

TEST_CASE("[DisjointSet] get_members with non-existent element fails gracefully") {
	DisjointSet<int> ds;
	ds.insert(1);

	Vector<int> members;
	ERR_PRINT_OFF;
	ds.get_members(members, 999);
	ERR_PRINT_ON;
	CHECK_MESSAGE(
			members.size() == 0,
			"get_members with a non-existent element should produce no output.");
}

TEST_CASE("[DisjointSet] Chain of unions preserves correctness") {
	DisjointSet<int> ds;
	// Build pairs: {0,1}, {2,3}, {4,5}
	ds.create_union(0, 1);
	ds.create_union(2, 3);
	ds.create_union(4, 5);

	Vector<int> reps;
	ds.get_representatives(reps);
	CHECK_MESSAGE(
			reps.size() == 3,
			"Three separate pairs should have 3 representatives.");

	// Merge pairs together: {0,1,2,3}, {4,5}
	ds.create_union(0, 2);
	reps.clear();
	ds.get_representatives(reps);
	CHECK_MESSAGE(
			reps.size() == 2,
			"After merging two pairs, should have 2 representatives.");

	// Merge all: {0,1,2,3,4,5}
	ds.create_union(0, 4);
	reps.clear();
	ds.get_representatives(reps);
	CHECK_MESSAGE(
			reps.size() == 1,
			"After merging all pairs, should have 1 representative.");

	Vector<int> members;
	ds.get_members(members, reps[0]);
	CHECK_MESSAGE(
			members.size() == 6,
			"All 6 elements should be in the final merged set.");
}

} // namespace TestDisjointSet
