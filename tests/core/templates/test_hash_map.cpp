/**************************************************************************/
/*  test_hash_map.cpp                                                     */
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

TEST_FORCE_LINK(test_hash_map)

#include "core/templates/hash_map.h"

#include <unordered_map>

namespace TestHashMap {

// A hasher that funnels every key into the same H1 group (low bits all zero)
// so that we exercise quadratic probing and adversarial-collision behavior.
struct SameGroupHasher {
	static _FORCE_INLINE_ uint32_t hash(int p_key) {
		// Spread fingerprints across the high bits so h2 still varies, but
		// keep the low bits zero so h1 is constant and every probe starts at
		// group 0.
		return (static_cast<uint32_t>(p_key) << 7);
	}
};

// A hasher whose H2 fingerprint is also constant -- every key collides on
// both group AND fingerprint, forcing full key compares all the way through.
struct SameSlotHasher {
	static _FORCE_INLINE_ uint32_t hash(int /*p_key*/) {
		return 0;
	}
};

TEST_CASE("[HashMap] List initialization") {
	HashMap<int, String> map{ { 0, "A" }, { 1, "B" }, { 2, "C" }, { 3, "D" }, { 4, "E" } };

	CHECK(map.size() == 5);
	CHECK(map[0] == "A");
	CHECK(map[1] == "B");
	CHECK(map[2] == "C");
	CHECK(map[3] == "D");
	CHECK(map[4] == "E");
}

TEST_CASE("[HashMap] List initialization with existing elements") {
	HashMap<int, String> map{ { 0, "A" }, { 0, "B" }, { 0, "C" }, { 0, "D" }, { 0, "E" } };

	CHECK(map.size() == 1);
	CHECK(map[0] == "E");
}

TEST_CASE("[HashMap] Insert element") {
	HashMap<int, int> map;
	HashMap<int, int>::Iterator e = map.insert(42, 84);

	CHECK(e);
	CHECK(e->key == 42);
	CHECK(e->value == 84);
	CHECK(map[42] == 84);
	CHECK(map.has(42));
	CHECK(map.find(42));
}

TEST_CASE("[HashMap] Overwrite element") {
	HashMap<int, int> map;
	map.insert(42, 84);
	map.insert(42, 1234);

	CHECK(map[42] == 1234);
}

TEST_CASE("[HashMap] Erase via element") {
	HashMap<int, int> map;
	HashMap<int, int>::Iterator e = map.insert(42, 84);
	map.remove(e);
	CHECK(!map.has(42));
	CHECK(!map.find(42));
}

TEST_CASE("[HashMap] Erase via key") {
	HashMap<int, int> map;
	map.insert(42, 84);
	map.erase(42);
	CHECK(!map.has(42));
	CHECK(!map.find(42));
}

TEST_CASE("[HashMap] Size") {
	HashMap<int, int> map;
	map.insert(42, 84);
	map.insert(123, 84);
	map.insert(123, 84);
	map.insert(0, 84);
	map.insert(123485, 84);

	CHECK(map.size() == 4);
}

TEST_CASE("[HashMap] Iteration") {
	HashMap<int, int> map;
	map.insert(42, 84);
	map.insert(123, 12385);
	map.insert(0, 12934);
	map.insert(123485, 1238888);
	map.insert(123, 111111);

	Vector<Pair<int, int>> expected;
	expected.push_back(Pair<int, int>(42, 84));
	expected.push_back(Pair<int, int>(123, 111111));
	expected.push_back(Pair<int, int>(0, 12934));
	expected.push_back(Pair<int, int>(123485, 1238888));

	int idx = 0;
	for (const KeyValue<int, int> &E : map) {
		CHECK(expected[idx] == Pair<int, int>(E.key, E.value));
		++idx;
	}
}

TEST_CASE("[HashMap] Const iteration") {
	HashMap<int, int> map;
	map.insert(42, 84);
	map.insert(123, 12385);
	map.insert(0, 12934);
	map.insert(123485, 1238888);
	map.insert(123, 111111);

	const HashMap<int, int> const_map(map);

	Vector<Pair<int, int>> expected;
	expected.push_back(Pair<int, int>(42, 84));
	expected.push_back(Pair<int, int>(123, 111111));
	expected.push_back(Pair<int, int>(0, 12934));
	expected.push_back(Pair<int, int>(123485, 1238888));
	expected.push_back(Pair<int, int>(123, 111111));

	int idx = 0;
	for (const KeyValue<int, int> &E : const_map) {
		CHECK(expected[idx] == Pair<int, int>(E.key, E.value));
		++idx;
	}
}

TEST_CASE("[HashMap] Sort") {
	HashMap<int, int> hashmap;
	int shuffled_ints[]{ 6, 1, 9, 8, 3, 0, 4, 5, 7, 2 };

	for (int i : shuffled_ints) {
		hashmap[i] = i;
	}
	hashmap.sort();

	int i = 0;
	for (const KeyValue<int, int> &kv : hashmap) {
		CHECK_EQ(kv.key, i);
		i++;
	}

	struct ReverseSort {
		bool operator()(const KeyValue<int, int> &p_a, const KeyValue<int, int> &p_b) {
			return p_a.key > p_b.key;
		}
	};
	hashmap.sort_custom<ReverseSort>();

	for (const KeyValue<int, int> &kv : hashmap) {
		i--;
		CHECK_EQ(kv.key, i);
	}
}

TEST_CASE("[HashMap] Insertion order preserved across erases") {
	HashMap<int, int> map;
	for (int i = 0; i < 32; i++) {
		map.insert(i, i * 10);
	}
	// Erase every other key.
	for (int i = 0; i < 32; i += 2) {
		map.erase(i);
	}
	int expected = 1;
	for (const KeyValue<int, int> &kv : map) {
		CHECK(kv.key == expected);
		CHECK(kv.value == expected * 10);
		expected += 2;
	}
	CHECK(expected == 33);
}

TEST_CASE("[HashMap] front_insert prepends to iteration order") {
	HashMap<int, int> map;
	map.insert(2, 20);
	map.insert(3, 30);
	map.insert(1, 10, /*p_front_insert=*/true);
	map.insert(0, 0, /*p_front_insert=*/true);

	int expected[] = { 0, 1, 2, 3 };
	int idx = 0;
	for (const KeyValue<int, int> &kv : map) {
		CHECK(kv.key == expected[idx]);
		idx++;
	}
	CHECK(idx == 4);
}

TEST_CASE("[HashMap] Pointer stability across many inserts and erases") {
	HashMap<int, int> map;
	const int N = 256;
	for (int i = 0; i < N; i++) {
		map.insert(i, i + 1000);
	}
	// Capture pointers to a subset of values.
	int *anchors[N];
	for (int i = 0; i < N; i++) {
		anchors[i] = map.getptr(i);
		REQUIRE(anchors[i] != nullptr);
		CHECK(*anchors[i] == i + 1000);
	}
	// Insert a bunch more entries; existing pointers must remain valid.
	for (int i = N; i < N * 4; i++) {
		map.insert(i, i + 1000);
	}
	for (int i = 0; i < N; i++) {
		CHECK(*anchors[i] == i + 1000);
		CHECK(map.getptr(i) == anchors[i]);
	}
	// Erase a disjoint set; remaining anchored pointers must remain valid.
	for (int i = N; i < N * 4; i += 3) {
		map.erase(i);
	}
	for (int i = 0; i < N; i++) {
		CHECK(*anchors[i] == i + 1000);
		CHECK(map.getptr(i) == anchors[i]);
	}
}

TEST_CASE("[HashMap] Adversarial same-group collisions") {
	HashMap<int, int, SameGroupHasher> map;
	const int N = 200;
	for (int i = 0; i < N; i++) {
		map.insert(i, i * 7);
	}
	CHECK(map.size() == (uint32_t)N);
	for (int i = 0; i < N; i++) {
		REQUIRE(map.has(i));
		CHECK(map[i] == i * 7);
	}
	// Erase odd keys, then look up everything.
	for (int i = 1; i < N; i += 2) {
		map.erase(i);
	}
	for (int i = 0; i < N; i++) {
		if (i & 1) {
			CHECK(!map.has(i));
		} else {
			CHECK(map[i] == i * 7);
		}
	}
}

TEST_CASE("[HashMap] Adversarial same-slot collisions (h1 and h2 collide)") {
	HashMap<int, int, SameSlotHasher> map;
	const int N = 64;
	for (int i = 0; i < N; i++) {
		map.insert(i, i * 3);
	}
	for (int i = 0; i < N; i++) {
		CHECK(map[i] == i * 3);
	}
	for (int i = 0; i < N; i++) {
		map.erase(i);
		CHECK(!map.has(i));
	}
	CHECK(map.size() == 0);
}

TEST_CASE("[HashMap] std::unordered_map oracle fuzz") {
	HashMap<int, int> map;
	std::unordered_map<int, int> oracle;
	// Deterministic pseudo-random sequence (xorshift) so the test reproduces.
	uint32_t rng = 0xCAFEBABEu;
	auto next = [&]() {
		rng ^= rng << 13;
		rng ^= rng >> 17;
		rng ^= rng << 5;
		return rng;
	};
	for (int step = 0; step < 4000; step++) {
		const int key = static_cast<int>(next() % 200);
		const uint32_t op = next() % 4;
		if (op == 0) {
			const int v = static_cast<int>(next());
			map.insert(key, v);
			oracle[key] = v;
		} else if (op == 1) {
			map.erase(key);
			oracle.erase(key);
		} else if (op == 2) {
			const int *p = map.getptr(key);
			auto it = oracle.find(key);
			if (it == oracle.end()) {
				CHECK(p == nullptr);
			} else {
				REQUIRE(p != nullptr);
				CHECK(*p == it->second);
			}
		} else {
			CHECK(map.has(key) == (oracle.count(key) > 0));
		}
		CHECK(map.size() == (uint32_t)oracle.size());
	}
	// Final equality check: every oracle entry must be present with the
	// expected value, and the map must contain no extras.
	for (const auto &kv : oracle) {
		const int *p = map.getptr(kv.first);
		REQUIRE(p != nullptr);
		CHECK(*p == kv.second);
	}
	uint32_t counted = 0;
	for (const KeyValue<int, int> &kv : map) {
		auto it = oracle.find(kv.key);
		REQUIRE(it != oracle.end());
		CHECK(it->second == kv.value);
		counted++;
	}
	CHECK(counted == (uint32_t)oracle.size());
}

TEST_CASE("[HashMap] Memory recycling across churn") {
	HashMap<int, int> map;
	const int N = 1024;
	// Fill, drain, refill several times. Verifies that erase + insert reuses
	// slots without leaking state across cycles.
	for (int cycle = 0; cycle < 5; cycle++) {
		for (int i = 0; i < N; i++) {
			map.insert(i, i + cycle);
		}
		CHECK(map.size() == (uint32_t)N);
		for (int i = 0; i < N; i++) {
			CHECK(map[i] == i + cycle);
		}
		for (int i = 0; i < N; i++) {
			map.erase(i);
		}
		CHECK(map.size() == 0);
	}
	// Re-fill one more time and walk insertion order.
	for (int i = 0; i < N; i++) {
		map.insert(i, i);
	}
	int expected = 0;
	for (const KeyValue<int, int> &kv : map) {
		CHECK(kv.key == expected);
		expected++;
	}
	CHECK(expected == N);
}

TEST_CASE("[HashMap] replace_key preserves iteration position and pointer") {
	HashMap<int, int> map;
	for (int i = 0; i < 8; i++) {
		map.insert(i, i * 11);
	}
	int *p = map.getptr(3);
	REQUIRE(p != nullptr);
	CHECK(map.replace_key(3, 100));
	CHECK(!map.has(3));
	CHECK(map.has(100));
	CHECK(map[100] == 33);
	// Pointer to the old slot should now read as the same entry under new key
	// (KeyValue at the same address; only the key was mutated).
	CHECK(map.getptr(100) == p);

	// Iteration order: 0, 1, 2, 100, 4, 5, 6, 7
	int expected[] = { 0, 1, 2, 100, 4, 5, 6, 7 };
	int idx = 0;
	for (const KeyValue<int, int> &kv : map) {
		CHECK(kv.key == expected[idx]);
		idx++;
	}
	CHECK(idx == 8);
}

TEST_CASE("[HashMap] Reserve avoids subsequent rehash") {
	HashMap<int, int> map;
	map.reserve(1000);
	// Reservation is allowed to be lazy; force the index table to actually
	// materialize by inserting a single element, then snapshot the capacity.
	map.insert(0, 0);
	const uint32_t cap_before = map.get_capacity();
	for (int i = 1; i < 700; i++) {
		map.insert(i, i);
	}
	CHECK(map.get_capacity() == cap_before);
	for (int i = 0; i < 700; i++) {
		CHECK(map[i] == i);
	}
}

TEST_CASE("[HashMap] Reserve tiers keep ordered semantics under churn") {
	{
		HashMap<int, int> small_map;
		small_map.reserve(200);
		for (int i = 0; i < 200; i++) {
			small_map.insert(i, i * 2);
		}
		for (int i = 0; i < 200; i++) {
			CHECK(small_map[i] == i * 2);
		}
	}

	HashMap<int, int> map;
	map.reserve(300);
	for (int i = 0; i < 300; i++) {
		map.insert(i, i * 10);
	}

	int *replaced_ptr = map.getptr(151);
	REQUIRE(replaced_ptr != nullptr);

	for (int i = 0; i < 300; i += 3) {
		CHECK(map.erase(i));
	}

	for (int i = 1000; i < 1032; i++) {
		map.insert(i, i, /*p_front_insert=*/true);
	}

	CHECK(map.replace_key(151, 1151));
	CHECK(map.getptr(1151) == replaced_ptr);
	CHECK(map[1151] == 1510);

	int idx = 0;
	for (const KeyValue<int, int> &kv : map) {
		if (idx < 32) {
			CHECK(kv.key == 1031 - idx);
		} else {
			int original = 1;
			int live_index = idx - 32;
			while ((original % 3) == 0) {
				original++;
			}
			for (int skip = 0; skip < live_index; skip++) {
				original++;
				while ((original % 3) == 0) {
					original++;
				}
			}
			CHECK(kv.key == (original == 151 ? 1151 : original));
		}
		idx++;
	}
	CHECK(idx == 232);
}

} // namespace TestHashMap
