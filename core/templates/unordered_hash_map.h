/*************************************************************************/
/*  unordered_hash_map.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef UNORDERED_HASH_MAP_H
#define UNORDERED_HASH_MAP_H

#include "core/math/math_funcs.h"
#include "core/os/memory.h"
#include "core/templates/hash_map.h"
#include "core/templates/hashfuncs.h"
#include "core/templates/paged_allocator.h"

/**
 * This version is similar to the one in HashMap (see description there)
 * but it allocates KeyValues in a flat buffer.
 * Use this instead of UnorderedHashMap only if all the following conditions are met:
 *
 * - Usage is performance critical.
 * - Iteration order does not matter.
 * - Iterator or pointer references to TValue are not saved.
 * - If Value has a copy constructor, it must be cheap.
 *
 */

template <class TKey, class TValue,
		class Hasher = HashMapHasherDefault,
		class Comparator = HashMapComparatorDefault<TKey>>
class UnorderedHashMap {
public:
	static constexpr uint32_t MIN_CAPACITY_INDEX = 2; // Use a prime.
	static constexpr float MAX_OCCUPANCY = 0.75;
	static constexpr uint32_t EMPTY_HASH = 0;

private:
	KeyValue<TKey, TValue> *elements = nullptr;
	uint32_t *hashes = nullptr;

	uint32_t capacity_index = 0;
	uint32_t num_elements = 0;

	_FORCE_INLINE_ uint32_t _hash(const TKey &p_key) const {
		uint32_t hash = Hasher::hash(p_key);

		if (unlikely(hash == EMPTY_HASH)) {
			hash = EMPTY_HASH + 1;
		}

		return hash;
	}

	_FORCE_INLINE_ uint32_t _get_probe_length(uint32_t p_pos, uint32_t p_hash, uint32_t p_capacity) const {
		uint32_t original_pos = p_hash % p_capacity;
		return (p_pos - original_pos + p_capacity) % p_capacity;
	}

	bool _lookup_pos(const TKey &p_key, uint32_t &r_pos) const {
		if (elements == nullptr) {
			return false; // Failed lookups, no elements
		}

		uint32_t capacity = hash_table_size_primes[capacity_index];
		uint32_t hash = _hash(p_key);
		uint32_t pos = hash % capacity;
		uint32_t distance = 0;

		while (true) {
			if (hashes[pos] == EMPTY_HASH) {
				return false;
			}

			if (distance > _get_probe_length(pos, hashes[pos], capacity)) {
				return false;
			}

			if (hashes[pos] == hash && Comparator::compare(elements[pos].key, p_key)) {
				r_pos = pos;
				return true;
			}

			pos = (pos + 1) % capacity;
			distance++;
		}
	}

	uint32_t _insert_with_hash(uint32_t p_hash, const TKey &p_key, const TValue &p_value) {
		uint32_t capacity = hash_table_size_primes[capacity_index];
		uint32_t hash = p_hash;
		TKey key = p_key;
		TValue value = p_value;
		uint32_t distance = 0;
		uint32_t pos = hash % capacity;

		while (true) {
			if (hashes[pos] == EMPTY_HASH) {
				memnew_placement(&elements[pos], KeyValue(key, value));
				hashes[pos] = hash;
				num_elements++;
				return pos;
			}

			// Not an empty slot, let's check the probing length of the existing one.
			uint32_t existing_probe_len = _get_probe_length(pos, hashes[pos], capacity);
			if (existing_probe_len < distance) {
				SWAP(hash, hashes[pos]);
				KeyValue<TKey, TValue> temp(elements[pos]);
				elements[pos].~KeyValue<TKey, TValue>();
				memnew_placement(&elements[pos], KeyValue(key, value));
				key = temp.key;
				value = temp.value;
				distance = existing_probe_len;
			}

			pos = (pos + 1) % capacity;
			distance++;
		}
	}

	void _resize_and_rehash(uint32_t p_new_capacity_index) {
		uint32_t old_capacity = hash_table_size_primes[capacity_index];

		// Capacity can't be 0.
		capacity_index = MAX((uint32_t)MIN_CAPACITY_INDEX, p_new_capacity_index);

		uint32_t capacity = hash_table_size_primes[capacity_index];

		KeyValue<TKey, TValue> *old_elements = elements;
		uint32_t *old_hashes = hashes;

		num_elements = 0;
		hashes = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * capacity));
		elements = reinterpret_cast<KeyValue<TKey, TValue> *>(Memory::alloc_static(sizeof(KeyValue<TKey, TValue>) * capacity));

		for (uint32_t i = 0; i < capacity; i++) {
			hashes[i] = 0;
		}

		if (old_capacity == 0) {
			// Nothing to do.
			return;
		}

		for (uint32_t i = 0; i < old_capacity; i++) {
			if (old_hashes[i] == EMPTY_HASH) {
				continue;
			}

			_insert_with_hash(old_hashes[i], old_elements[i].key, old_elements[i].value);
			old_elements[i].~KeyValue<TKey, TValue>();
		}

		Memory::free_static(old_elements);
		Memory::free_static(old_hashes);
	}

	_FORCE_INLINE_ int32_t _insert(const TKey &p_key, const TValue &p_value, bool p_front_insert = false) {
		uint32_t capacity = hash_table_size_primes[capacity_index];
		if (unlikely(elements == nullptr)) {
			// Allocate on demand to save memory.

			hashes = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * capacity));
			elements = reinterpret_cast<KeyValue<TKey, TValue> *>(Memory::alloc_static(sizeof(KeyValue<TKey, TValue>) * capacity));

			for (uint32_t i = 0; i < capacity; i++) {
				hashes[i] = EMPTY_HASH;
			}
		}

		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (exists) {
			elements[pos].value = p_value;
			return pos;
		} else {
			if (num_elements + 1 > MAX_OCCUPANCY * capacity) {
				ERR_FAIL_COND_V_MSG(capacity_index + 1 == HASH_TABLE_SIZE_MAX, -1, "Hash table maximum capacity reached, aborting insertion.");
				_resize_and_rehash(capacity_index + 1);
			}

			uint32_t hash = _hash(p_key);
			return _insert_with_hash(hash, p_key, p_value);
		}
	}

public:
	_FORCE_INLINE_ uint32_t get_capacity() const { return hash_table_size_primes[capacity_index]; }
	_FORCE_INLINE_ uint32_t size() const { return num_elements; }

	/* Standard Godot Container API */

	bool is_empty() const {
		return num_elements == 0;
	}

	void clear() {
		if (elements == nullptr) {
			return;
		}
		uint32_t capacity = hash_table_size_primes[capacity_index];
		for (uint32_t i = 0; i < capacity; i++) {
			if (hashes[i] == EMPTY_HASH) {
				continue;
			}

			hashes[i] = EMPTY_HASH;
			elements[i].~KeyValue<TKey, TValue>();
		}

		num_elements = 0;
	}

	TValue &get(const TKey &p_key) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);
		CRASH_COND_MSG(!exists, "UnorderedHashMap key not found.");
		return elements[pos].value;
	}

	const TValue &get(const TKey &p_key) const {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);
		CRASH_COND_MSG(!exists, "UnorderedHashMap key not found.");
		return elements[pos].value;
	}

	const TValue *getptr(const TKey &p_key) const {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (exists) {
			return &elements[pos].value;
		}
		return nullptr;
	}

	TValue *getptr(const TKey &p_key) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (exists) {
			return &elements[pos].value;
		}
		return nullptr;
	}

	_FORCE_INLINE_ bool has(const TKey &p_key) const {
		uint32_t _pos = 0;
		return _lookup_pos(p_key, _pos);
	}

	bool erase(const TKey &p_key) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (!exists) {
			return false;
		}

		uint32_t capacity = hash_table_size_primes[capacity_index];
		uint32_t next_pos = (pos + 1) % capacity;
		while (hashes[next_pos] != EMPTY_HASH && _get_probe_length(next_pos, hashes[next_pos], capacity) != 0) {
			SWAP(hashes[next_pos], hashes[pos]);
			SWAP(elements[next_pos], elements[pos]);
			pos = next_pos;
			next_pos = (pos + 1) % capacity;
		}

		hashes[pos] = EMPTY_HASH;
		elements[pos].~KeyValue<TKey, TValue>();

		num_elements--;
		return true;
	}

	// Reserves space for a number of elements, useful to avoid many resizes and rehashes.
	// If adding a known (possibly large) number of elements at once, must be larger than old capacity.
	void reserve(uint32_t p_new_capacity) {
		uint32_t new_index = capacity_index;

		while (hash_table_size_primes[new_index] < p_new_capacity) {
			ERR_FAIL_COND_MSG(new_index + 1 == (uint32_t)HASH_TABLE_SIZE_MAX, nullptr);
			new_index++;
		}

		if (new_index == capacity_index) {
			return;
		}

		if (elements == nullptr) {
			capacity_index = new_index;
			return; // Unallocated yet.
		}
		_resize_and_rehash(new_index);
	}

	/** Iterator API **/

	struct ConstIterator {
		_FORCE_INLINE_ const KeyValue<TKey, TValue> &operator*() const {
			return elements[index];
		}
		_FORCE_INLINE_ const KeyValue<TKey, TValue> *operator->() const {
			return &elements[index];
		}
		_FORCE_INLINE_ ConstIterator &operator++() {
			if (elements) {
				while (true) {
					index++;
					if (unlikely((uint32_t)index == capacity)) {
						elements = nullptr;
						hashes = nullptr;
						capacity = 0;
						index = -1;
						return *this;
					}
					if (hashes[index] != EMPTY_HASH) {
						break;
					}
				}
			}
			return *this;
		}
		_FORCE_INLINE_ ConstIterator &operator--() {
			if (elements) {
				while (true) {
					index--;
					if (unlikely(index == -1)) {
						elements = nullptr;
						hashes = nullptr;
						capacity = 0;
						return *this;
					}
					if (hashes[index] != EMPTY_HASH) {
						break;
					}
				}
			}
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const ConstIterator &b) const { return elements == b.elements && index == b.index; }
		_FORCE_INLINE_ bool operator!=(const ConstIterator &b) const { return elements != b.elements || index != b.index; }

		_FORCE_INLINE_ explicit operator bool() const {
			return elements != nullptr;
		}

		_FORCE_INLINE_ ConstIterator(const KeyValue<TKey, TValue> *p_elements, const uint32_t *p_hashes, uint32_t p_capacity, uint32_t p_index = 0) {
			elements = p_elements;
			hashes = p_hashes;
			capacity = p_capacity;
			index = p_index;
			if (p_capacity != 0) {
				// Ensure it sits on a valid hash
				while (hashes[index] == EMPTY_HASH) {
					index++;
					if (unlikely((uint32_t)index == capacity)) {
						elements = nullptr;
						hashes = nullptr;
						capacity = 0;
						index = -1;
						return;
					}
				}
			}
		}
		_FORCE_INLINE_ ConstIterator() {}
		_FORCE_INLINE_ ConstIterator(const ConstIterator &p_it) {
			hashes = p_it.hashes;
			elements = p_it.elements;
			capacity = p_it.capacity;
			index = p_it.index;
		}
		_FORCE_INLINE_ void operator=(const ConstIterator &p_it) {
			hashes = p_it.hashes;
			elements = p_it.elements;
			capacity = p_it.capacity;
			index = p_it.index;
		}

	private:
		const KeyValue<TKey, TValue> *elements = nullptr;
		const uint32_t *hashes = nullptr;
		uint32_t capacity = 0;
		int32_t index = -1;
	};

	struct Iterator {
		_FORCE_INLINE_ const KeyValue<TKey, TValue> &operator*() {
			return elements[index];
		}
		_FORCE_INLINE_ const KeyValue<TKey, TValue> *operator->() {
			return &elements[index];
		}
		_FORCE_INLINE_ Iterator &operator++() {
			if (elements) {
				while (true) {
					index++;
					if (unlikely((uint32_t)index == capacity)) {
						elements = nullptr;
						hashes = nullptr;
						capacity = 0;
						index = -1;
						return *this;
					}
					if (hashes[index] != EMPTY_HASH) {
						break;
					}
				}
			}
			return *this;
		}
		_FORCE_INLINE_ Iterator &operator--() {
			if (elements) {
				while (true) {
					index--;
					if (unlikely(index == -1)) {
						elements = nullptr;
						hashes = nullptr;
						capacity = 0;
						return *this;
					}
					if (hashes[index] != EMPTY_HASH) {
						break;
					}
				}
			}
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const Iterator &b) const { return elements == b.elements && index == b.index; }
		_FORCE_INLINE_ bool operator!=(const Iterator &b) const { return elements != b.elements || index != b.index; }

		_FORCE_INLINE_ explicit operator bool() const {
			return elements != nullptr;
		}

		_FORCE_INLINE_ Iterator(KeyValue<TKey, TValue> *p_elements, const uint32_t *p_hashes, uint32_t p_capacity, uint32_t p_index = 0) {
			elements = p_elements;
			hashes = p_hashes;
			capacity = p_capacity;
			index = p_index;
			if (p_capacity != 0) {
				// Ensure it sits on a valid hash
				while (hashes[index] == EMPTY_HASH) {
					index++;
					if (unlikely((uint32_t)index == capacity)) {
						elements = nullptr;
						hashes = nullptr;
						capacity = 0;
						index = -1;
						return;
					}
				}
			}
		}
		_FORCE_INLINE_ operator ConstIterator() {
			return ConstIterator(elements, hashes, capacity, index);
		}
		_FORCE_INLINE_ Iterator() {}
		_FORCE_INLINE_ Iterator(const Iterator &p_it) {
			hashes = p_it.hashes;
			elements = p_it.elements;
			capacity = p_it.capacity;
			index = p_it.index;
		}
		_FORCE_INLINE_ void operator=(const Iterator &p_it) {
			hashes = p_it.hashes;
			elements = p_it.elements;
			capacity = p_it.capacity;
			index = p_it.index;
		}

	private:
		KeyValue<TKey, TValue> *elements = nullptr;
		const uint32_t *hashes = nullptr;
		uint32_t capacity = 0;
		int32_t index = -1;
	};

	_FORCE_INLINE_ Iterator begin() {
		return num_elements ? Iterator(elements, hashes, get_capacity(), 0) : Iterator();
	}
	_FORCE_INLINE_ Iterator end() {
		return Iterator();
	}
	_FORCE_INLINE_ Iterator last() {
		if (num_elements == 0) {
			return Iterator();
		}
		uint32_t index = get_capacity() - 1;
		while (hashes[index] != EMPTY_HASH) {
			index--;
		}
		return Iterator(elements, hashes, get_capacity(), index);
	}

	_FORCE_INLINE_ Iterator find(const TKey &p_key) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);
		if (!exists) {
			return end();
		}
		return Iterator(elements, hashes, get_capacity(), pos);
	}

	_FORCE_INLINE_ void remove(const Iterator &p_iter) {
		if (p_iter) {
			erase(p_iter->key);
		}
	}

	_FORCE_INLINE_ ConstIterator begin() const {
		return num_elements ? ConstIterator(elements, hashes, get_capacity(), 0) : ConstIterator();
	}
	_FORCE_INLINE_ ConstIterator end() const {
		return ConstIterator();
	}
	_FORCE_INLINE_ ConstIterator last() const {
		if (num_elements == 0) {
			return ConstIterator();
		}
		uint32_t index = get_capacity() - 1;
		while (hashes[index] != EMPTY_HASH) {
			index--;
		}
		return ConstIterator(elements, hashes, get_capacity(), index);
	}

	_FORCE_INLINE_ ConstIterator find(const TKey &p_key) const {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);
		if (!exists) {
			return end();
		}
		return Iterator(elements, hashes, get_capacity(), pos);
	}

	/* Indexing */

	const TValue &operator[](const TKey &p_key) const {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);
		CRASH_COND(!exists);
		return elements[pos].value;
	}

	TValue &operator[](const TKey &p_key) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);
		if (!exists) {
			return elements[_insert(p_key, TValue())].value;
		} else {
			return elements[pos].value;
		}
	}

	/* Insert */

	Iterator insert(const TKey &p_key, const TValue &p_value, bool p_front_insert = false) {
		uint32_t pos = _insert(p_key, p_value, p_front_insert);
		return Iterator(elements, hashes, get_capacity(), pos);
	}

	/* Constructors */

	UnorderedHashMap(const UnorderedHashMap &p_other) {
		reserve(hash_table_size_primes[p_other.capacity_index]);

		if (p_other.num_elements == 0) {
			return;
		}

		for (const KeyValue<TKey, TValue> &E : p_other) {
			insert(E.key, E.value);
		}
	}

	void operator=(const UnorderedHashMap &p_other) {
		if (this == &p_other) {
			return; // Ignore self assignment.
		}
		if (num_elements != 0) {
			clear();
		}

		reserve(hash_table_size_primes[p_other.capacity_index]);

		if (p_other.elements == nullptr) {
			return; // Nothing to copy.
		}

		for (const KeyValue<TKey, TValue> &E : p_other) {
			insert(E.key, E.value);
		}
	}

	UnorderedHashMap(uint32_t p_initial_capacity) {
		// Capacity can't be 0.
		capacity_index = 0;
		reserve(p_initial_capacity);
	}
	UnorderedHashMap() {
		capacity_index = MIN_CAPACITY_INDEX;
	}

	uint32_t debug_get_hash(uint32_t p_index) {
		if (num_elements == 0) {
			return 0;
		}
		ERR_FAIL_INDEX_V(p_index, get_capacity(), 0);
		return hashes[p_index];
	}
	Iterator debug_get_element(uint32_t p_index) {
		if (num_elements == 0) {
			return Iterator();
		}
		ERR_FAIL_INDEX_V(p_index, get_capacity(), Iterator());
		return Iterator(elements[p_index]);
	}

	~UnorderedHashMap() {
		clear();

		if (elements != nullptr) {
			Memory::free_static(elements);
			Memory::free_static(hashes);
		}
	}
};

#endif // UNORDERED_HASH_MAP_H
