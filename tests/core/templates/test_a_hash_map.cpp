/**************************************************************************/
/*  test_a_hash_map.cpp                                                   */
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

TEST_FORCE_LINK(test_a_hash_map)

#include "core/templates/a_hash_map.h"

#include <unordered_map>

namespace TestAHashMap {

// Hasher that funnels every key into the same H1 group (low bits all zero)
// to exercise quadratic probing under adversarial collisions.
struct AHM_SameGroupHasher {
	static _FORCE_INLINE_ uint32_t hash(int p_key) {
		return (static_cast<uint32_t>(p_key) << 7);
	}
};

// Hasher whose H1 AND H2 are constant -- forces full key compares for every
// probe step.
struct AHM_SameSlotHasher {
	static _FORCE_INLINE_ uint32_t hash(int /*p_key*/) {
		return 0;
	}
};

TEST_CASE("[AHashMap] List initialization") {
	AHashMap<int, String> map{ { 0, "A" }, { 1, "B" }, { 2, "C" }, { 3, "D" }, { 4, "E" } };

	CHECK(map.size() == 5);
	CHECK(map[0] == "A");
	CHECK(map[1] == "B");
	CHECK(map[2] == "C");
	CHECK(map[3] == "D");
	CHECK(map[4] == "E");
}

TEST_CASE("[AHashMap] List initialization with existing elements") {
	AHashMap<int, String> map{ { 0, "A" }, { 0, "B" }, { 0, "C" }, { 0, "D" }, { 0, "E" } };

	CHECK(map.size() == 1);
	CHECK(map[0] == "E");
}

TEST_CASE("[AHashMap] Insert element") {
	AHashMap<int, int> map;
	AHashMap<int, int>::Iterator e = map.insert(42, 84);

	CHECK(e);
	CHECK(e->key == 42);
	CHECK(e->value == 84);
	CHECK(map[42] == 84);
	CHECK(map.has(42));
	CHECK(map.find(42));
}

TEST_CASE("[AHashMap] Overwrite element") {
	AHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(42, 1234);

	CHECK(map[42] == 1234);
}

TEST_CASE("[AHashMap] Erase via element") {
	AHashMap<int, int> map;
	AHashMap<int, int>::Iterator e = map.insert(42, 84);
	map.remove(e);
	CHECK(!map.has(42));
	CHECK(!map.find(42));
}

TEST_CASE("[AHashMap] Erase via key") {
	AHashMap<int, int> map;
	map.insert(42, 84);
	map.erase(42);
	CHECK(!map.has(42));
	CHECK(!map.find(42));
}

TEST_CASE("[AHashMap] Size") {
	AHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(123, 84);
	map.insert(123, 84);
	map.insert(0, 84);
	map.insert(123485, 84);

	CHECK(map.size() == 4);
}

TEST_CASE("[AHashMap] Iteration") {
	AHashMap<int, int> map;

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
		idx++;
	}

	idx--;
	for (AHashMap<int, int>::Iterator it = map.last(); it; --it) {
		CHECK(expected[idx] == Pair<int, int>(it->key, it->value));
		idx--;
	}
}

TEST_CASE("[AHashMap] Const iteration") {
	AHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(123, 12385);
	map.insert(0, 12934);
	map.insert(123485, 1238888);
	map.insert(123, 111111);

	const AHashMap<int, int> const_map(map);

	Vector<Pair<int, int>> expected;
	expected.push_back(Pair<int, int>(42, 84));
	expected.push_back(Pair<int, int>(123, 111111));
	expected.push_back(Pair<int, int>(0, 12934));
	expected.push_back(Pair<int, int>(123485, 1238888));
	expected.push_back(Pair<int, int>(123, 111111));

	int idx = 0;
	for (const KeyValue<int, int> &E : const_map) {
		CHECK(expected[idx] == Pair<int, int>(E.key, E.value));
		idx++;
	}

	idx--;
	for (AHashMap<int, int>::ConstIterator it = const_map.last(); it; --it) {
		CHECK(expected[idx] == Pair<int, int>(it->key, it->value));
		idx--;
	}
}

TEST_CASE("[AHashMap] Replace key") {
	AHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(0, 12934);
	CHECK(map.replace_key(0, 1));
	CHECK(map.has(1));
	CHECK(map[1] == 12934);
}

TEST_CASE("[AHashMap] Clear") {
	AHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(123, 12385);
	map.insert(0, 12934);

	map.clear();
	CHECK(!map.has(42));
	CHECK(map.size() == 0);
	CHECK(map.is_empty());
}

TEST_CASE("[AHashMap] Get") {
	AHashMap<int, int> map;
	map.insert(42, 84);
	map.insert(123, 12385);
	map.insert(0, 12934);

	CHECK(map.get(123) == 12385);
	map.get(123) = 10;
	CHECK(map.get(123) == 10);

	CHECK(*map.getptr(0) == 12934);
	*map.getptr(0) = 1;
	CHECK(*map.getptr(0) == 1);

	CHECK(map.get(42) == 84);
	CHECK(map.getptr(-10) == nullptr);
}

TEST_CASE("[AHashMap] Insert, iterate and remove many elements") {
	const int elem_max = 1234;
	AHashMap<int, int> map;
	for (int i = 0; i < elem_max; i++) {
		map.insert(i, i);
	}

	//insert order should have been kept
	int idx = 0;
	for (const KeyValue<int, int> &K : map) {
		CHECK(idx == K.key);
		CHECK(idx == K.value);
		CHECK(map.has(idx));
		idx++;
	}

	Vector<int> elems_still_valid;

	for (int i = 0; i < elem_max; i++) {
		if ((i % 5) == 0) {
			map.erase(i);
		} else {
			elems_still_valid.push_back(i);
		}
	}

	CHECK(elems_still_valid.size() == map.size());

	for (int i = 0; i < elems_still_valid.size(); i++) {
		CHECK(map.has(elems_still_valid[i]));
	}
}

TEST_CASE("[AHashMap] Insert, iterate and remove many strings") {
	const int elem_max = 432;
	AHashMap<String, String> map;

	// To not print WARNING: Excessive collision count (NN), is the right hash function being used?
	ERR_PRINT_OFF;
	for (int i = 0; i < elem_max; i++) {
		map.insert(itos(i), itos(i));
	}
	ERR_PRINT_ON;

	//insert order should have been kept
	int idx = 0;
	for (auto &K : map) {
		CHECK(itos(idx) == K.key);
		CHECK(itos(idx) == K.value);
		CHECK(map.has(itos(idx)));
		idx++;
	}

	Vector<String> elems_still_valid;

	for (int i = 0; i < elem_max; i++) {
		if ((i % 5) == 0) {
			map.erase(itos(i));
		} else {
			elems_still_valid.push_back(itos(i));
		}
	}

	CHECK(elems_still_valid.size() == map.size());

	for (int i = 0; i < elems_still_valid.size(); i++) {
		CHECK(map.has(elems_still_valid[i]));
	}

	elems_still_valid.clear();
}

TEST_CASE("[AHashMap] Copy constructor") {
	AHashMap<int, int> map0;
	const uint32_t count = 5;
	for (uint32_t i = 0; i < count; i++) {
		map0.insert(i, i);
	}
	AHashMap<int, int> map1(map0);
	CHECK(map0.size() == map1.size());
	CHECK(map0.get_capacity() == map1.get_capacity());
	CHECK(*map0.getptr(0) == *map1.getptr(0));
}

TEST_CASE("[AHashMap] Operator =") {
	AHashMap<int, int> map0;
	AHashMap<int, int> map1;
	const uint32_t count = 5;
	map1.insert(1234, 1234);
	for (uint32_t i = 0; i < count; i++) {
		map0.insert(i, i);
	}
	map1 = map0;
	CHECK(map0.size() == map1.size());
	CHECK(map0.get_capacity() == map1.get_capacity());
	CHECK(*map0.getptr(0) == *map1.getptr(0));
}

TEST_CASE("[AHashMap] Array methods") {
	AHashMap<int, int> map;
	for (int i = 0; i < 100; i++) {
		map.insert(100 - i, i);
	}
	for (int i = 0; i < 100; i++) {
		CHECK(map.get_by_index(i).value == i);
	}
	int index = map.get_index(1);
	CHECK(map.get_by_index(index).value == 99);
	CHECK(map.erase_by_index(index));
	CHECK(!map.erase_by_index(index));
	CHECK(map.get_index(1) == -1);
}

TEST_CASE("[AHashMap] Swap-on-erase keeps tail key reachable") {
	AHashMap<int, int> map;
	for (int i = 0; i < 32; i++) {
		map.insert(i, i + 1000);
	}
	// Erase from the middle and head; check that all the surviving keys are
	// still reachable and that the dense entries array stays packed.
	map.erase(0);
	map.erase(15);
	map.erase(31);
	CHECK(map.size() == 29);
	for (int i = 1; i < 31; i++) {
		if (i == 15) {
			continue;
		}
		REQUIRE(map.has(i));
		CHECK(map[i] == i + 1000);
	}

	// All entries returned by get_by_index() must reflect what get() returns
	// for that key (i.e. the swap-on-erase didn't desync the slot mapping).
	for (uint32_t i = 0; i < map.size(); i++) {
		const KeyValue<int, int> &kv = map.get_by_index(i);
		CHECK(map.get(kv.key) == kv.value);
		CHECK((uint32_t)map.get_index(kv.key) == i);
	}
}

TEST_CASE("[AHashMap] Adversarial same-group collisions") {
	AHashMap<int, int, AHM_SameGroupHasher> map;
	const int N = 200;
	for (int i = 0; i < N; i++) {
		map.insert(i, i * 7);
	}
	CHECK(map.size() == (uint32_t)N);
	for (int i = 0; i < N; i++) {
		REQUIRE(map.has(i));
		CHECK(map[i] == i * 7);
	}
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

TEST_CASE("[AHashMap] Adversarial same-slot collisions") {
	AHashMap<int, int, AHM_SameSlotHasher> map;
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

TEST_CASE("[AHashMap] std::unordered_map oracle fuzz") {
	AHashMap<int, int> map;
	std::unordered_map<int, int> oracle;
	uint32_t rng = 0xDEADBEEFu;
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
	for (const auto &kv : oracle) {
		const int *p = map.getptr(kv.first);
		REQUIRE(p != nullptr);
		CHECK(*p == kv.second);
	}
}

TEST_CASE("[AHashMap] Memory recycling across churn") {
	AHashMap<int, int> map;
	const int N = 1024;
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
}

TEST_CASE("[AHashMap] get_elements_ptr stays packed after erase") {
	AHashMap<int, int> map;
	const int N = 50;
	for (int i = 0; i < N; i++) {
		map.insert(i, i);
	}
	for (int i = 0; i < N; i += 3) {
		map.erase(i);
	}
	const uint32_t live = map.size();
	const KeyValue<int, int> *raw = map.get_elements_ptr();
	REQUIRE(raw != nullptr);
	for (uint32_t i = 0; i < live; i++) {
		// Each entry must be reachable by both index AND key, and they must
		// agree.
		CHECK(raw[i].value == map.get(raw[i].key));
		CHECK((uint32_t)map.get_index(raw[i].key) == i);
	}
}

} // namespace TestAHashMap
