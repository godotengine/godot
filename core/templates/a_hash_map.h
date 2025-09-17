/**************************************************************************/
/*  a_hash_map.h                                                          */
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

#include "core/templates/hash_map.h"

struct HashSplit {
	uint8_t tag_hash;
	uint32_t pos_hash;

	HashSplit(uint8_t tag, uint32_t pos) :
			tag_hash(tag), pos_hash(pos) {}
};

static_assert(sizeof(HashSplit) == 8);

/**
 * An array-based implementation of a hash map. It is very efficient in terms of performance and
 * memory usage. Works like a dynamic array, adding elements to the end of the array, and
 * allows you to access array elements by their index by using `get_by_index` method.
 * Example:
 * ```
 *  AHashMap<int, Object *> map;
 *
 *  int get_object_id_by_number(int p_number) {
 *		int id = map.get_index(p_number);
 *		return id;
 *  }
 *
 *  Object *get_object_by_id(int p_id) {
 *		map.get_by_index(p_id).value;
 *  }
 * ```
 * Still, don`t erase the elements because ID can break.
 *
 * When an element erase, its place is taken by the element from the end.
 *
 *        <-------------
 *      |               |
 *  6 8 X 9 32 -1 5 -10 7 X X X
 *  6 8 7 9 32 -1 5 -10 X X X X
 *
 *
 * Use RBMap if you need to iterate over sorted elements.
 *
 * Use HashMap if:
 *   - You need to keep an iterator or const pointer to Key and you intend to add/remove elements in the meantime.
 *   - You need to preserve the insertion order when using erase.
 *
 * It is recommended to use `HashMap` if `KeyValue` size is very large.
 */
template <typename TKey, typename TValue,
		typename Hasher = HashMapHasherDefault,
		typename Comparator = HashMapComparatorDefault<TKey>>
class AHashMap {
public:
	// Must be a power of two.
	static constexpr uint32_t INITIAL_CAPACITY = 16;
	static constexpr uint32_t EMPTY_HASH = 0;
	static constexpr uint8_t DELETED_HASH = 1;
	static constexpr uint8_t HASH_MASK = 0b11111110;
	static_assert(EMPTY_HASH == 0, "EMPTY_HASH must always be 0 for the memcpy() optimization.");
	static_assert(DELETED_HASH == 1, "DELETED_HASH must always be 1");
	static_assert(((EMPTY_HASH & HASH_MASK) == 0 && (DELETED_HASH & HASH_MASK) == 0), "HASH_MASK should mask EMPTY_HASH and DELETED_HASH");

private:
	typedef KeyValue<TKey, TValue> MapKeyValue;
	MapKeyValue *elements = nullptr;
	uint8_t *hash_vec = nullptr;
	uint32_t *pos_vec = nullptr;

	// Due to optimization, this is `capacity - 1`. Use + 1 to get normal capacity.
	uint32_t capacity = 0;
	uint32_t num_elements = 0;
	uint32_t occupied_slots = 0;

	uint32_t _hash(const TKey &p_key) const {
		return Hasher::hash(p_key);
	}

	static _FORCE_INLINE_ uint32_t _get_resize_count(uint32_t p_capacity) {
		return p_capacity ^ (p_capacity + 1) >> 2; // = get_capacity() * 0.75 - 1; Works only if p_capacity = 2^n - 1.
	}

	static _FORCE_INLINE_ bool _valid_hash_mask(uint8_t hash) {
		return hash & HASH_MASK;
	}

	_FORCE_INLINE_ uint32_t _hash_to_pos(uint32_t p_hash) const {
		return _hash_to_pos(p_hash, capacity);
	}

	_FORCE_INLINE_ static uint32_t _hash_to_pos(uint32_t p_hash, uint32_t capacity) {
		return p_hash & capacity;
	}

	_FORCE_INLINE_ static uint8_t _hash_to_tag(uint32_t p_hash) {
		return static_cast<uint8_t>(p_hash >> (sizeof(p_hash) - sizeof(uint8_t)));
	}

	_FORCE_INLINE_ HashSplit _hash_to_split(uint32_t p_hash) const {
		uint8_t tag_hash = _hash_to_tag(p_hash);

		if (unlikely(!_valid_hash_mask(tag_hash))) {
			tag_hash = DELETED_HASH + 1;
		}

		return HashSplit(tag_hash, _hash_to_pos(p_hash));
	}

	_FORCE_INLINE_ bool _lookup_pos(const TKey &p_key, uint32_t &r_pos, uint32_t &r_hash_pos) const {
		if (unlikely(elements == nullptr)) {
			return false; // Failed lookups, no elements.
		}
		return _lookup_pos_with_hash(p_key, r_pos, r_hash_pos, _hash_to_split(_hash(p_key)));
	}

	bool _lookup_pos_with_hash(const TKey &p_key, uint32_t &r_pos, uint32_t &r_hash_pos, HashSplit split) const {
		if (unlikely(elements == nullptr)) {
			return false; // Failed lookups, no elements.
		}

		uint32_t pos = split.pos_hash;
		uint8_t tag = split.tag_hash;
		bool found_invalid = false;

		while (true) {
			uint8_t data_hash = hash_vec[pos];
			if (data_hash == tag) {
				uint32_t data_key = pos_vec[pos];
				if (Comparator::compare(elements[data_key].key, p_key)) {
					r_pos = data_key;
					r_hash_pos = pos;
					return true;
				}
			} else if (!found_invalid && !_valid_hash_mask(data_hash)) {
				found_invalid = true;
				r_hash_pos = pos;
			}

			if (data_hash == EMPTY_HASH) {
				return false;
			}

			pos = (pos + 1) & capacity;
		}
	}

	uint32_t _insert_with_hash(HashSplit split, uint32_t p_index) {
		uint32_t pos = split.pos_hash;
		uint8_t tag = split.tag_hash;

#ifdef DEV_ENABLED
		uint32_t distance = 0;
#endif
		while (true) {
			if (!_valid_hash_mask(hash_vec[pos])) {
#ifdef DEV_ENABLED
				if (unlikely(distance > 12)) {
					WARN_PRINT("Excessive collision count (" +
							itos(distance) + "), is the right hash function being used?");
				}
#endif
				hash_vec[pos] = tag;
				pos_vec[pos] = p_index;

				if (hash_vec[pos] != DELETED_HASH) {
					occupied_slots++;
				}

				return pos;
			}

			pos = (pos + 1) & capacity;
#ifdef DEV_ENABLED
			distance++;
#endif
		}
	}

	void _resize_and_rehash(uint32_t p_new_capacity) {
		// Capacity can't be 0 and must be 2^n - 1.
		capacity = MAX(4u, p_new_capacity);
		uint32_t real_capacity = next_power_of_2(capacity);
		capacity = real_capacity - 1;

		uint8_t *old_hash_vec = hash_vec;
		uint32_t *old_pos_vec = pos_vec;

		Memory::free_static(old_hash_vec);
		Memory::free_static(old_pos_vec);

		hash_vec = reinterpret_cast<uint8_t *>(Memory::alloc_static_zeroed(sizeof(uint8_t) * real_capacity));
		pos_vec = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * real_capacity));
		elements = reinterpret_cast<MapKeyValue *>(Memory::realloc_static(elements, sizeof(MapKeyValue) * (_get_resize_count(capacity) + 1)));

		occupied_slots = 0;
		for (uint32_t i = 0; i < num_elements; i++) {
			const TKey &key = elements[i].key;
			uint32_t hash = _hash(key);
			HashSplit split = _hash_to_split(hash);

			_insert_with_hash(split, i);
		}
	}

	int32_t _insert_element(const TKey &p_key, const TValue &p_value, HashSplit split) {
		if (unlikely(elements == nullptr)) {
			// Allocate on demand to save memory.

			uint32_t real_capacity = capacity + 1;
			hash_vec = reinterpret_cast<uint8_t *>(Memory::alloc_static_zeroed(sizeof(uint8_t) * real_capacity));
			pos_vec = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * real_capacity));
			elements = reinterpret_cast<MapKeyValue *>(Memory::alloc_static(sizeof(MapKeyValue) * (_get_resize_count(capacity) + 1)));
		}

		if (unlikely(occupied_slots > _get_resize_count(capacity))) {
			_resize_and_rehash(capacity * 4);
			// `_resize_and_rehash` causes a capacity change, thus causing a need to resplit
			split = _hash_to_split(_hash(p_key));
		}

		memnew_placement(&elements[num_elements], MapKeyValue(p_key, p_value));

		_insert_with_hash(split, num_elements);
		num_elements++;
		return num_elements - 1;
	}

	int32_t _insert_element_to_index(const TKey &p_key, const TValue &p_value, HashSplit split, uint32_t pos) {
		if (unlikely(elements == nullptr)) {
			return _insert_element(p_key, p_value, split);
		}

		if (unlikely(occupied_slots > _get_resize_count(capacity))) {
			return _insert_element(p_key, p_value, split);
		}

		if (hash_vec[pos] == EMPTY_HASH) {
			occupied_slots++;
		}

		memnew_placement(&elements[num_elements], MapKeyValue(p_key, p_value));
		hash_vec[pos] = split.tag_hash;
		pos_vec[pos] = num_elements;
		num_elements++;
		return num_elements - 1;
	}

	void _init_from(const AHashMap &p_other) {
		capacity = p_other.capacity;
		uint32_t real_capacity = capacity + 1;
		num_elements = p_other.num_elements;
		occupied_slots = p_other.occupied_slots;

		if (p_other.num_elements == 0) {
			return;
		}

		hash_vec = reinterpret_cast<uint8_t *>(Memory::alloc_static(sizeof(uint8_t) * real_capacity));
		pos_vec = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * real_capacity));
		elements = reinterpret_cast<MapKeyValue *>(Memory::alloc_static(sizeof(MapKeyValue) * (_get_resize_count(capacity) + 1)));

		if constexpr (std::is_trivially_copyable_v<TKey> && std::is_trivially_copyable_v<TValue>) {
			void *destination = elements;
			const void *source = p_other.elements;
			memcpy(destination, source, sizeof(MapKeyValue) * num_elements);
		} else {
			for (uint32_t i = 0; i < num_elements; i++) {
				memnew_placement(&elements[i], MapKeyValue(p_other.elements[i]));
			}
		}

		memcpy(hash_vec, p_other.hash_vec, sizeof(uint8_t) * real_capacity);
		memcpy(pos_vec, p_other.pos_vec, sizeof(uint32_t) * real_capacity);
	}

public:
	/* Standard Godot Container API */

	_FORCE_INLINE_ uint32_t get_capacity() const { return capacity + 1; }
	_FORCE_INLINE_ uint32_t size() const { return num_elements; }

	_FORCE_INLINE_ bool is_empty() const {
		return num_elements == 0;
	}

	void clear() {
		if (elements == nullptr || occupied_slots == 0) {
			return;
		}

		memset(hash_vec, EMPTY_HASH, (capacity + 1) * sizeof(uint8_t));
		if constexpr (!(std::is_trivially_destructible_v<TKey> && std::is_trivially_destructible_v<TValue>)) {
			for (uint32_t i = 0; i < num_elements; i++) {
				elements[i].key.~TKey();
				elements[i].value.~TValue();
			}
		}

		num_elements = 0;
		occupied_slots = 0;
	}

	TValue &get(const TKey &p_key) {
		uint32_t pos = 0;
		uint32_t hash_pos = 0;
		bool exists = _lookup_pos(p_key, pos, hash_pos);
		CRASH_COND_MSG(!exists, "AHashMap key not found.");
		return elements[pos].value;
	}

	const TValue &get(const TKey &p_key) const {
		uint32_t pos = 0;
		uint32_t hash_pos = 0;
		bool exists = _lookup_pos(p_key, pos, hash_pos);
		CRASH_COND_MSG(!exists, "AHashMap key not found.");
		return elements[pos].value;
	}

	const TValue *getptr(const TKey &p_key) const {
		uint32_t pos = 0;
		uint32_t hash_pos = 0;
		bool exists = _lookup_pos(p_key, pos, hash_pos);

		if (exists) {
			return &elements[pos].value;
		}
		return nullptr;
	}

	TValue *getptr(const TKey &p_key) {
		uint32_t pos = 0;
		uint32_t hash_pos = 0;
		bool exists = _lookup_pos(p_key, pos, hash_pos);

		if (exists) {
			return &elements[pos].value;
		}
		return nullptr;
	}

	bool has(const TKey &p_key) const {
		uint32_t _pos = 0;
		uint32_t h_pos = 0;
		return _lookup_pos(p_key, _pos, h_pos);
	}

	bool erase(const TKey &p_key) {
		uint32_t pos = 0;
		uint32_t element_pos = 0;
		bool exists = _lookup_pos(p_key, element_pos, pos);

		if (!exists) {
			return false;
		}

		hash_vec[pos] = DELETED_HASH;
		elements[element_pos].key.~TKey();
		elements[element_pos].value.~TValue();
		num_elements--;

		if (element_pos < num_elements) {
			void *destination = &elements[element_pos];
			const void *source = &elements[num_elements];
			memcpy(destination, source, sizeof(MapKeyValue));
			uint32_t h_pos = 0;
			_lookup_pos(elements[num_elements].key, pos, h_pos);
			pos_vec[h_pos] = element_pos;
		}

		return true;
	}

	// Replace the key of an entry in-place, without invalidating iterators or changing the entries position during iteration.
	// p_old_key must exist in the map and p_new_key must not, unless it is equal to p_old_key.
	bool replace_key(const TKey &p_old_key, const TKey &p_new_key) {
		if (p_old_key == p_new_key) {
			return true;
		}
		uint32_t pos = 0;
		uint32_t element_pos = 0;
		ERR_FAIL_COND_V(_lookup_pos(p_new_key, element_pos, pos), false);
		ERR_FAIL_COND_V(!_lookup_pos(p_old_key, element_pos, pos), false);
		MapKeyValue &element = elements[element_pos];
		const_cast<TKey &>(element.key) = p_new_key;

		uint32_t next_pos = (pos + 1) & capacity;
		while (hash_vec[next_pos] != EMPTY_HASH) {
			SWAP(hash_vec[next_pos], hash_vec[pos]);
			SWAP(pos_vec[next_pos], pos_vec[pos]);

			pos = next_pos;
			next_pos = (next_pos + 1) & capacity;
		}

		hash_vec[pos] = EMPTY_HASH;

		uint32_t hash = _hash(p_new_key);
		_insert_with_hash(_hash_to_split(hash), element_pos);

		return true;
	}

	// Reserves space for a number of elements, useful to avoid many resizes and rehashes.
	// If adding a known (possibly large) number of elements at once, must be larger than old capacity.
	void reserve(uint32_t p_new_capacity) {
		ERR_FAIL_COND_MSG(p_new_capacity < size(), "reserve() called with a capacity smaller than the current size. This is likely a mistake.");
		if (elements == nullptr) {
			capacity = MAX(4u, p_new_capacity);
			capacity = next_power_of_2(capacity) - 1;
			return; // Unallocated yet.
		}
		if (p_new_capacity <= get_capacity()) {
			return;
		}
		_resize_and_rehash(p_new_capacity);
	}

	/** Iterator API **/

	struct ConstIterator {
		_FORCE_INLINE_ const MapKeyValue &operator*() const {
			return *pair;
		}
		_FORCE_INLINE_ const MapKeyValue *operator->() const {
			return pair;
		}
		_FORCE_INLINE_ ConstIterator &operator++() {
			pair++;
			return *this;
		}

		_FORCE_INLINE_ ConstIterator &operator--() {
			pair--;
			if (pair < begin) {
				pair = end;
			}
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const ConstIterator &b) const { return pair == b.pair; }
		_FORCE_INLINE_ bool operator!=(const ConstIterator &b) const { return pair != b.pair; }

		_FORCE_INLINE_ explicit operator bool() const {
			return pair != end;
		}

		_FORCE_INLINE_ ConstIterator(MapKeyValue *p_key, MapKeyValue *p_begin, MapKeyValue *p_end) {
			pair = p_key;
			begin = p_begin;
			end = p_end;
		}
		_FORCE_INLINE_ ConstIterator() {}
		_FORCE_INLINE_ ConstIterator(const ConstIterator &p_it) {
			pair = p_it.pair;
			begin = p_it.begin;
			end = p_it.end;
		}
		_FORCE_INLINE_ void operator=(const ConstIterator &p_it) {
			pair = p_it.pair;
			begin = p_it.begin;
			end = p_it.end;
		}

	private:
		MapKeyValue *pair = nullptr;
		MapKeyValue *begin = nullptr;
		MapKeyValue *end = nullptr;
	};

	struct Iterator {
		_FORCE_INLINE_ MapKeyValue &operator*() const {
			return *pair;
		}
		_FORCE_INLINE_ MapKeyValue *operator->() const {
			return pair;
		}
		_FORCE_INLINE_ Iterator &operator++() {
			pair++;
			return *this;
		}
		_FORCE_INLINE_ Iterator &operator--() {
			pair--;
			if (pair < begin) {
				pair = end;
			}
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const Iterator &b) const { return pair == b.pair; }
		_FORCE_INLINE_ bool operator!=(const Iterator &b) const { return pair != b.pair; }

		_FORCE_INLINE_ explicit operator bool() const {
			return pair != end;
		}

		_FORCE_INLINE_ Iterator(MapKeyValue *p_key, MapKeyValue *p_begin, MapKeyValue *p_end) {
			pair = p_key;
			begin = p_begin;
			end = p_end;
		}
		_FORCE_INLINE_ Iterator() {}
		_FORCE_INLINE_ Iterator(const Iterator &p_it) {
			pair = p_it.pair;
			begin = p_it.begin;
			end = p_it.end;
		}
		_FORCE_INLINE_ void operator=(const Iterator &p_it) {
			pair = p_it.pair;
			begin = p_it.begin;
			end = p_it.end;
		}

		operator ConstIterator() const {
			return ConstIterator(pair, begin, end);
		}

	private:
		MapKeyValue *pair = nullptr;
		MapKeyValue *begin = nullptr;
		MapKeyValue *end = nullptr;
	};

	_FORCE_INLINE_ Iterator begin() {
		return Iterator(elements, elements, elements + num_elements);
	}
	_FORCE_INLINE_ Iterator end() {
		return Iterator(elements + num_elements, elements, elements + num_elements);
	}
	_FORCE_INLINE_ Iterator last() {
		if (unlikely(num_elements == 0)) {
			return Iterator(nullptr, nullptr, nullptr);
		}
		return Iterator(elements + num_elements - 1, elements, elements + num_elements);
	}

	Iterator find(const TKey &p_key) {
		uint32_t pos = 0;
		uint32_t h_pos = 0;
		bool exists = _lookup_pos(p_key, pos, h_pos);
		if (!exists) {
			return end();
		}
		return Iterator(elements + pos, elements, elements + num_elements);
	}

	void remove(const Iterator &p_iter) {
		if (p_iter) {
			erase(p_iter->key);
		}
	}

	_FORCE_INLINE_ ConstIterator begin() const {
		return ConstIterator(elements, elements, elements + num_elements);
	}
	_FORCE_INLINE_ ConstIterator end() const {
		return ConstIterator(elements + num_elements, elements, elements + num_elements);
	}
	_FORCE_INLINE_ ConstIterator last() const {
		if (unlikely(num_elements == 0)) {
			return ConstIterator(nullptr, nullptr, nullptr);
		}
		return ConstIterator(elements + num_elements - 1, elements, elements + num_elements);
	}

	ConstIterator find(const TKey &p_key) const {
		uint32_t pos = 0;
		uint32_t h_pos = 0;
		bool exists = _lookup_pos(p_key, pos, h_pos);
		if (!exists) {
			return end();
		}
		return ConstIterator(elements + pos, elements, elements + num_elements);
	}

	/* Indexing */

	const TValue &operator[](const TKey &p_key) const {
		uint32_t pos = 0;
		uint32_t h_pos = 0;
		bool exists = _lookup_pos(p_key, pos, h_pos);
		CRASH_COND(!exists);
		return elements[pos].value;
	}

	TValue &operator[](const TKey &p_key) {
		uint32_t pos = 0;
		uint32_t h_pos = 0;
		uint32_t hash = _hash(p_key);
		HashSplit split = _hash_to_split(hash);
		bool exists = _lookup_pos_with_hash(p_key, pos, h_pos, split);

		if (exists) {
			return elements[pos].value;
		} else {
			pos = _insert_element(p_key, TValue(), split);
			return elements[pos].value;
		}
	}

	/* Insert */

	Iterator insert(const TKey &p_key, const TValue &p_value) {
		uint32_t pos = 0;
		uint32_t h_pos = 0;
		uint32_t hash = _hash(p_key);
		HashSplit split = _hash_to_split(hash);
		bool exists = _lookup_pos_with_hash(p_key, pos, h_pos, split);

		if (!exists) {
			pos = _insert_element_to_index(p_key, p_value, split, h_pos);
		} else {
			elements[pos].value = p_value;
		}
		return Iterator(elements + pos, elements, elements + num_elements);
	}

	// Inserts an element without checking if it already exists.
	Iterator insert_new(const TKey &p_key, const TValue &p_value) {
		DEV_ASSERT(!has(p_key));
		uint32_t hash = _hash(p_key);
		uint32_t pos = _insert_element(p_key, p_value, _hash_to_split(hash));
		return Iterator(elements + pos, elements, elements + num_elements);
	}

	/* Array methods. */

	// Unsafe. Changing keys and going outside the bounds of an array can lead to undefined behavior.
	KeyValue<TKey, TValue> *get_elements_ptr() {
		return elements;
	}

	// Returns the element index. If not found, returns -1.
	int get_index(const TKey &p_key) {
		uint32_t pos = 0;
		uint32_t h_pos = 0;
		bool exists = _lookup_pos(p_key, pos, h_pos);
		if (!exists) {
			return -1;
		}
		return pos;
	}

	KeyValue<TKey, TValue> &get_by_index(uint32_t p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, num_elements);
		return elements[p_index];
	}

	bool erase_by_index(uint32_t p_index) {
		if (p_index >= size()) {
			return false;
		}
		return erase(elements[p_index].key);
	}

	/* Constructors */

	AHashMap(const AHashMap &p_other) {
		_init_from(p_other);
	}

	AHashMap(const HashMap<TKey, TValue> &p_other) {
		reserve(p_other.size());
		for (const KeyValue<TKey, TValue> &E : p_other) {
			uint32_t hash = _hash(E.key);
			_insert_element(E.key, E.value, _hash_to_split(hash));
		}
	}

	void operator=(const AHashMap &p_other) {
		if (this == &p_other) {
			return; // Ignore self assignment.
		}

		reset();

		_init_from(p_other);
	}

	void operator=(const HashMap<TKey, TValue> &p_other) {
		reset();
		reserve(p_other.size());
		for (const KeyValue<TKey, TValue> &E : p_other) {
			uint32_t hash = _hash(E.key);
			_insert_element(E.key, E.value, _hash_to_split(hash));
		}
	}

	AHashMap(uint32_t p_initial_capacity) {
		// Capacity can't be 0 and must be 2^n - 1.
		capacity = MAX(4u, p_initial_capacity);
		capacity = next_power_of_2(capacity) - 1;
	}
	AHashMap() :
			capacity(INITIAL_CAPACITY - 1) {
	}

	AHashMap(std::initializer_list<KeyValue<TKey, TValue>> p_init) {
		reserve(p_init.size());
		for (const KeyValue<TKey, TValue> &E : p_init) {
			insert(E.key, E.value);
		}
	}

	void reset() {
		if (elements != nullptr) {
			if constexpr (!(std::is_trivially_destructible_v<TKey> && std::is_trivially_destructible_v<TValue>)) {
				for (uint32_t i = 0; i < num_elements; i++) {
					elements[i].key.~TKey();
					elements[i].value.~TValue();
				}
			}
			Memory::free_static(elements);
			Memory::free_static(hash_vec);
			Memory::free_static(pos_vec);
			elements = nullptr;
		}
		capacity = INITIAL_CAPACITY - 1;
		num_elements = 0;
		occupied_slots = 0;
	}

	~AHashMap() {
		reset();
	}
};

extern template class AHashMap<int, int>;
extern template class AHashMap<String, int>;
extern template class AHashMap<StringName, StringName>;
extern template class AHashMap<StringName, Variant>;
extern template class AHashMap<StringName, int>;
