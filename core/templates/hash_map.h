/**************************************************************************/
/*  hash_map.h                                                            */
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

#include "core/os/memory.h"
#include "core/string/print_string.h"
#include "core/templates/hashfuncs.h"
#include "core/templates/pair.h"
#include "core/templates/sort_list.h"

#include <initializer_list>

/**
 * A HashMap implementation that uses open addressing with Robin Hood hashing.
 * Robin Hood hashing swaps out entries that have a smaller probing distance
 * than the to-be-inserted entry, that evens out the average probing distance
 * and enables faster lookups. Backward shift deletion is employed to further
 * improve the performance and to avoid infinite loops in rare cases.
 *
 * Keys and values are stored in a double linked list by insertion order. This
 * has a slight performance overhead on lookup, which can be mostly compensated
 * using a paged allocator if required.
 *
 * The assignment operator copy the pairs from one map to the other.
 */

template <typename TKey, typename TValue>
struct HashMapElement {
	HashMapElement *next = nullptr;
	HashMapElement *prev = nullptr;
	KeyValue<TKey, TValue> data;
	HashMapElement() {}
	HashMapElement(const TKey &p_key, const TValue &p_value) :
			data(p_key, p_value) {}
};

template <typename TKey, typename TValue,
		typename Hasher = HashMapHasherDefault,
		typename Comparator = HashMapComparatorDefault<TKey>,
		typename Allocator = DefaultTypedAllocator<HashMapElement<TKey, TValue>>>
class HashMap : private Allocator {
public:
	static constexpr uint32_t MIN_CAPACITY_INDEX = 2; // Use a prime.
	static constexpr float MAX_OCCUPANCY = 0.75;
	static constexpr uint32_t EMPTY_HASH = 0;
	using KV = KeyValue<TKey, TValue>; // Type alias for easier access to KeyValue.

private:
	HashMapElement<TKey, TValue> **_elements = nullptr;
	uint32_t *_hashes = nullptr;
	HashMapElement<TKey, TValue> *_head_element = nullptr;
	HashMapElement<TKey, TValue> *_tail_element = nullptr;

	uint32_t _capacity_idx = 0;
	uint32_t _size = 0;

	_FORCE_INLINE_ static uint32_t _hash(const TKey &p_key) {
		uint32_t hash = Hasher::hash(p_key);

		if (unlikely(hash == EMPTY_HASH)) {
			hash = EMPTY_HASH + 1;
		}

		return hash;
	}

	_FORCE_INLINE_ static constexpr void _increment_mod(uint32_t &r_idx, const uint32_t p_capacity) {
		r_idx++;
		// `if` is faster than both fastmod and mod.
		if (unlikely(r_idx == p_capacity)) {
			r_idx = 0;
		}
	}

	static _FORCE_INLINE_ uint32_t _get_probe_length(const uint32_t p_idx, const uint32_t p_hash, const uint32_t p_capacity, const uint64_t p_capacity_inv) {
		const uint32_t original_idx = fastmod(p_hash, p_capacity_inv, p_capacity);
		const uint32_t distance_idx = p_idx - original_idx + p_capacity;
		// At most p_capacity over 0, so we can use an if (faster than fastmod).
		return distance_idx >= p_capacity ? distance_idx - p_capacity : distance_idx;
	}

	bool _lookup_idx(const TKey &p_key, uint32_t &r_idx) const {
		return _elements != nullptr && _size > 0 && _lookup_idx_unchecked(p_key, _hash(p_key), r_idx);
	}

	/// Note: Assumes that _elements != nullptr
	bool _lookup_idx_unchecked(const TKey &p_key, uint32_t p_hash, uint32_t &r_idx) const {
		const uint32_t capacity = hash_table_size_primes[_capacity_idx];
		const uint64_t capacity_inv = hash_table_size_primes_inv[_capacity_idx];
		uint32_t idx = fastmod(p_hash, capacity_inv, capacity);
		uint32_t distance = 0;

		while (true) {
			if (_hashes[idx] == EMPTY_HASH) {
				return false;
			}

			if (distance > _get_probe_length(idx, _hashes[idx], capacity, capacity_inv)) {
				return false;
			}

			if (_hashes[idx] == p_hash && Comparator::compare(_elements[idx]->data.key, p_key)) {
				r_idx = idx;
				return true;
			}

			_increment_mod(idx, capacity);
			distance++;
		}
	}

	void _insert_element(uint32_t p_hash, HashMapElement<TKey, TValue> *p_value) {
		const uint32_t capacity = hash_table_size_primes[_capacity_idx];
		const uint64_t capacity_inv = hash_table_size_primes_inv[_capacity_idx];
		uint32_t hash = p_hash;
		HashMapElement<TKey, TValue> *value = p_value;
		uint32_t distance = 0;
		uint32_t idx = fastmod(hash, capacity_inv, capacity);

		while (true) {
			if (_hashes[idx] == EMPTY_HASH) {
				_elements[idx] = value;
				_hashes[idx] = hash;

				_size++;

				return;
			}

			// Not an empty slot, let's check the probing length of the existing one.
			uint32_t existing_probe_len = _get_probe_length(idx, _hashes[idx], capacity, capacity_inv);
			if (existing_probe_len < distance) {
				SWAP(hash, _hashes[idx]);
				SWAP(value, _elements[idx]);
				distance = existing_probe_len;
			}

			_increment_mod(idx, capacity);
			distance++;
		}
	}

	void _resize_and_rehash(uint32_t p_new_capacity_idx) {
		uint32_t old_capacity = hash_table_size_primes[_capacity_idx];

		// Capacity can't be 0.
		_capacity_idx = MAX((uint32_t)MIN_CAPACITY_INDEX, p_new_capacity_idx);

		uint32_t capacity = hash_table_size_primes[_capacity_idx];

		HashMapElement<TKey, TValue> **old_elements = _elements;
		uint32_t *old_hashes = _hashes;

		_size = 0;
		static_assert(EMPTY_HASH == 0, "Assuming EMPTY_HASH = 0 for alloc_static_zeroed call");

		_hashes = reinterpret_cast<uint32_t *>(Memory::alloc_static_zeroed(sizeof(uint32_t) * capacity));
		_elements = reinterpret_cast<HashMapElement<TKey, TValue> **>(Memory::alloc_static(sizeof(HashMapElement<TKey, TValue> *) * capacity));

		if (old_capacity == 0) {
			// Nothing to do.
			return;
		}

		for (uint32_t i = 0; i < old_capacity; i++) {
			if (old_hashes[i] == EMPTY_HASH) {
				continue;
			}

			_insert_element(old_hashes[i], old_elements[i]);
		}

		Memory::free_static(old_elements);
		Memory::free_static(old_hashes);
	}

	_FORCE_INLINE_ HashMapElement<TKey, TValue> *_insert(const TKey &p_key, const TValue &p_value, uint32_t p_hash, bool p_front_insert = false) {
		uint32_t capacity = hash_table_size_primes[_capacity_idx];
		if (unlikely(_elements == nullptr)) {
			// Allocate on demand to save memory.

			static_assert(EMPTY_HASH == 0, "Assuming EMPTY_HASH = 0 for alloc_static_zeroed call");
			_hashes = reinterpret_cast<uint32_t *>(Memory::alloc_static_zeroed(sizeof(uint32_t) * capacity));
			_elements = reinterpret_cast<HashMapElement<TKey, TValue> **>(Memory::alloc_static(sizeof(HashMapElement<TKey, TValue> *) * capacity));
		}

		if (_size + 1 > MAX_OCCUPANCY * capacity) {
			ERR_FAIL_COND_V_MSG(_capacity_idx + 1 == HASH_TABLE_SIZE_MAX, nullptr, "Hash table maximum capacity reached, aborting insertion.");
			_resize_and_rehash(_capacity_idx + 1);
		}

		HashMapElement<TKey, TValue> *elem = Allocator::new_allocation(HashMapElement<TKey, TValue>(p_key, p_value));

		if (_tail_element == nullptr) {
			_head_element = elem;
			_tail_element = elem;
		} else if (p_front_insert) {
			_head_element->prev = elem;
			elem->next = _head_element;
			_head_element = elem;
		} else {
			_tail_element->next = elem;
			elem->prev = _tail_element;
			_tail_element = elem;
		}

		_insert_element(p_hash, elem);
		return elem;
	}

	void _clear_data() {
		HashMapElement<TKey, TValue> *current = _tail_element;
		while (current != nullptr) {
			HashMapElement<TKey, TValue> *prev = current->prev;
			Allocator::delete_allocation(current);
			current = prev;
		}
	}

public:
	_FORCE_INLINE_ uint32_t get_capacity() const { return hash_table_size_primes[_capacity_idx]; }
	_FORCE_INLINE_ uint32_t size() const { return _size; }

	/* Standard Godot Container API */

	bool is_empty() const {
		return _size == 0;
	}

	void clear() {
		if (_elements == nullptr || _size == 0) {
			return;
		}

		_clear_data();
		memset(_hashes, EMPTY_HASH, get_capacity() * sizeof(uint32_t));

		_tail_element = nullptr;
		_head_element = nullptr;
		_size = 0;
	}

	void sort() {
		sort_custom<KeyValueSort<TKey, TValue>>();
	}

	template <typename C>
	void sort_custom() {
		if (size() < 2) {
			return;
		}

		using E = HashMapElement<TKey, TValue>;
		SortList<E, KeyValue<TKey, TValue>, &E::data, &E::prev, &E::next, C> sorter;
		sorter.sort(_head_element, _tail_element);
	}

	TValue &get(const TKey &p_key) {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);
		CRASH_COND_MSG(!exists, "HashMap key not found.");
		return _elements[idx]->data.value;
	}

	const TValue &get(const TKey &p_key) const {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);
		CRASH_COND_MSG(!exists, "HashMap key not found.");
		return _elements[idx]->data.value;
	}

	const TValue *getptr(const TKey &p_key) const {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);

		if (exists) {
			return &_elements[idx]->data.value;
		}
		return nullptr;
	}

	TValue *getptr(const TKey &p_key) {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);

		if (exists) {
			return &_elements[idx]->data.value;
		}
		return nullptr;
	}

	TValue &get(const TKey &p_key, const TValue &p_default) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);
		if (unlikely(!exists)) {
			return _insert(p_key, p_default)->data.value;
		}
		return elements[pos]->data.value;
	}

	TValue *getptr(const TKey &p_key, const TValue &p_default) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (likely(exists)) {
			return &elements[pos]->data.value;
		}
		return &_insert(p_key, p_default)->data.value;
	}

	_FORCE_INLINE_ bool has(const TKey &p_key) const {
		uint32_t _idx = 0;
		return _lookup_idx(p_key, _idx);
	}

	bool erase(const TKey &p_key) {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);

		if (!exists) {
			return false;
		}

		const uint32_t capacity = hash_table_size_primes[_capacity_idx];
		const uint64_t capacity_inv = hash_table_size_primes_inv[_capacity_idx];
		uint32_t next_idx = fastmod((idx + 1), capacity_inv, capacity);
		while (_hashes[next_idx] != EMPTY_HASH && _get_probe_length(next_idx, _hashes[next_idx], capacity, capacity_inv) != 0) {
			SWAP(_hashes[next_idx], _hashes[idx]);
			SWAP(_elements[next_idx], _elements[idx]);
			idx = next_idx;
			_increment_mod(next_idx, capacity);
		}

		_hashes[idx] = EMPTY_HASH;

		if (_head_element == _elements[idx]) {
			_head_element = _elements[idx]->next;
		}

		if (_tail_element == _elements[idx]) {
			_tail_element = _elements[idx]->prev;
		}

		if (_elements[idx]->prev) {
			_elements[idx]->prev->next = _elements[idx]->next;
		}

		if (_elements[idx]->next) {
			_elements[idx]->next->prev = _elements[idx]->prev;
		}

		Allocator::delete_allocation(_elements[idx]);

		_size--;
		return true;
	}

	// Replace the key of an entry in-place, without invalidating iterators or changing the entries position during iteration.
	// p_old_key must exist in the map and p_new_key must not, unless it is equal to p_old_key.
	bool replace_key(const TKey &p_old_key, const TKey &p_new_key) {
		ERR_FAIL_COND_V(_elements == nullptr || _size == 0, false);
		if (p_old_key == p_new_key) {
			return true;
		}
		const uint32_t new_hash = _hash(p_new_key);
		uint32_t idx = 0;
		ERR_FAIL_COND_V(_lookup_idx_unchecked(p_new_key, new_hash, idx), false);
		ERR_FAIL_COND_V(!_lookup_idx(p_old_key, idx), false);
		HashMapElement<TKey, TValue> *element = _elements[idx];

		// Delete the old entries in _hashes and _elements.
		const uint32_t capacity = hash_table_size_primes[_capacity_idx];
		const uint64_t capacity_inv = hash_table_size_primes_inv[_capacity_idx];
		uint32_t next_idx = fastmod((idx + 1), capacity_inv, capacity);
		while (_hashes[next_idx] != EMPTY_HASH && _get_probe_length(next_idx, _hashes[next_idx], capacity, capacity_inv) != 0) {
			SWAP(_hashes[next_idx], _hashes[idx]);
			SWAP(_elements[next_idx], _elements[idx]);
			idx = next_idx;
			_increment_mod(next_idx, capacity);
		}

		_hashes[idx] = EMPTY_HASH;

		// _insert_element will increment this again.
		_size--;

		// Update the HashMapElement with the new key and reinsert it.
		const_cast<TKey &>(element->data.key) = p_new_key;
		_insert_element(new_hash, element);

		return true;
	}

	// Reserves space for a number of elements, useful to avoid many resizes and rehashes.
	// If adding a known (possibly large) number of elements at once, must be larger than old capacity.
	void reserve(uint32_t p_new_capacity) {
		uint32_t new_idx = _capacity_idx;

		while (hash_table_size_primes[new_idx] < p_new_capacity) {
			ERR_FAIL_COND_MSG(new_idx + 1 == (uint32_t)HASH_TABLE_SIZE_MAX, nullptr);
			new_idx++;
		}

		if (new_idx == _capacity_idx) {
			if (p_new_capacity < _size) {
				WARN_VERBOSE("reserve() called with a capacity smaller than the current size. This is likely a mistake.");
			}
			return;
		}

		if (_elements == nullptr) {
			_capacity_idx = new_idx;
			return; // Unallocated yet.
		}
		_resize_and_rehash(new_idx);
	}

	/** Iterator API **/

	struct ConstIterator {
		_FORCE_INLINE_ const KeyValue<TKey, TValue> &operator*() const {
			return E->data;
		}
		_FORCE_INLINE_ const KeyValue<TKey, TValue> *operator->() const { return &E->data; }
		_FORCE_INLINE_ ConstIterator &operator++() {
			if (E) {
				E = E->next;
			}
			return *this;
		}
		_FORCE_INLINE_ ConstIterator &operator--() {
			if (E) {
				E = E->prev;
			}
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const ConstIterator &b) const { return E == b.E; }
		_FORCE_INLINE_ bool operator!=(const ConstIterator &b) const { return E != b.E; }

		_FORCE_INLINE_ explicit operator bool() const {
			return E != nullptr;
		}

		_FORCE_INLINE_ ConstIterator(const HashMapElement<TKey, TValue> *p_E) { E = p_E; }
		_FORCE_INLINE_ ConstIterator() {}
		_FORCE_INLINE_ ConstIterator(const ConstIterator &p_it) { E = p_it.E; }
		_FORCE_INLINE_ void operator=(const ConstIterator &p_it) {
			E = p_it.E;
		}

	private:
		const HashMapElement<TKey, TValue> *E = nullptr;
	};

	struct Iterator {
		_FORCE_INLINE_ KeyValue<TKey, TValue> &operator*() const {
			return E->data;
		}
		_FORCE_INLINE_ KeyValue<TKey, TValue> *operator->() const { return &E->data; }
		_FORCE_INLINE_ Iterator &operator++() {
			if (E) {
				E = E->next;
			}
			return *this;
		}
		_FORCE_INLINE_ Iterator &operator--() {
			if (E) {
				E = E->prev;
			}
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const Iterator &b) const { return E == b.E; }
		_FORCE_INLINE_ bool operator!=(const Iterator &b) const { return E != b.E; }

		_FORCE_INLINE_ explicit operator bool() const {
			return E != nullptr;
		}

		_FORCE_INLINE_ Iterator(HashMapElement<TKey, TValue> *p_E) { E = p_E; }
		_FORCE_INLINE_ Iterator() {}
		_FORCE_INLINE_ Iterator(const Iterator &p_it) { E = p_it.E; }
		_FORCE_INLINE_ void operator=(const Iterator &p_it) {
			E = p_it.E;
		}

		operator ConstIterator() const {
			return ConstIterator(E);
		}

	private:
		HashMapElement<TKey, TValue> *E = nullptr;
	};

	_FORCE_INLINE_ Iterator begin() {
		return Iterator(_head_element);
	}
	_FORCE_INLINE_ Iterator end() {
		return Iterator(nullptr);
	}
	_FORCE_INLINE_ Iterator last() {
		return Iterator(_tail_element);
	}

	_FORCE_INLINE_ Iterator find(const TKey &p_key) {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);
		if (!exists) {
			return end();
		}
		return Iterator(_elements[idx]);
	}

	_FORCE_INLINE_ void remove(const Iterator &p_iter) {
		if (p_iter) {
			erase(p_iter->key);
		}
	}

	_FORCE_INLINE_ ConstIterator begin() const {
		return ConstIterator(_head_element);
	}
	_FORCE_INLINE_ ConstIterator end() const {
		return ConstIterator(nullptr);
	}
	_FORCE_INLINE_ ConstIterator last() const {
		return ConstIterator(_tail_element);
	}

	_FORCE_INLINE_ ConstIterator find(const TKey &p_key) const {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);
		if (!exists) {
			return end();
		}
		return ConstIterator(_elements[idx]);
	}

	/* Indexing */

	const TValue &operator[](const TKey &p_key) const {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);
		CRASH_COND(!exists);
		return _elements[idx]->data.value;
	}

	TValue &operator[](const TKey &p_key) {
		const uint32_t hash = _hash(p_key);
		uint32_t idx = 0;
		bool exists = _elements && _size > 0 && _lookup_idx_unchecked(p_key, hash, idx);
		if (!exists) {
			return _insert(p_key, TValue(), hash)->data.value;
		} else {
			return _elements[idx]->data.value;
		}
	}

	/* Insert */

	Iterator insert(const TKey &p_key, const TValue &p_value, bool p_front_insert = false) {
		const uint32_t hash = _hash(p_key);
		uint32_t idx = 0;
		bool exists = _elements && _size > 0 && _lookup_idx_unchecked(p_key, hash, idx);
		if (!exists) {
			return Iterator(_insert(p_key, p_value, hash, p_front_insert));
		} else {
			_elements[idx]->data.value = p_value;
			return Iterator(_elements[idx]);
		}
	}

	/* Constructors */

	HashMap(const HashMap &p_other) {
		reserve(hash_table_size_primes[p_other._capacity_idx]);

		if (p_other._size == 0) {
			return;
		}

		for (const KeyValue<TKey, TValue> &E : p_other) {
			insert(E.key, E.value);
		}
	}

	HashMap(HashMap &&p_other) {
		_elements = p_other._elements;
		_hashes = p_other._hashes;
		_head_element = p_other._head_element;
		_tail_element = p_other._tail_element;
		_capacity_idx = p_other._capacity_idx;
		_size = p_other._size;

		p_other._elements = nullptr;
		p_other._hashes = nullptr;
		p_other._head_element = nullptr;
		p_other._tail_element = nullptr;
		p_other._capacity_idx = MIN_CAPACITY_INDEX;
		p_other._size = 0;
	}

	void operator=(const HashMap &p_other) {
		if (this == &p_other) {
			return; // Ignore self assignment.
		}
		if (_size != 0) {
			clear();
		}

		reserve(hash_table_size_primes[p_other._capacity_idx]);

		if (p_other._elements == nullptr) {
			return; // Nothing to copy.
		}

		for (const KeyValue<TKey, TValue> &E : p_other) {
			insert(E.key, E.value);
		}
	}

	HashMap &operator=(HashMap &&p_other) {
		if (this == &p_other) {
			return *this;
		}

		if (_size != 0) {
			clear();
		}
		if (_elements != nullptr) {
			Memory::free_static(_elements);
			Memory::free_static(_hashes);
		}

		_elements = p_other._elements;
		_hashes = p_other._hashes;
		_head_element = p_other._head_element;
		_tail_element = p_other._tail_element;
		_capacity_idx = p_other._capacity_idx;
		_size = p_other._size;

		p_other._elements = nullptr;
		p_other._hashes = nullptr;
		p_other._head_element = nullptr;
		p_other._tail_element = nullptr;
		p_other._capacity_idx = MIN_CAPACITY_INDEX;
		p_other._size = 0;

		return *this;
	}

	HashMap(uint32_t p_initial_capacity) {
		// Capacity can't be 0.
		_capacity_idx = 0;
		reserve(p_initial_capacity);
	}
	HashMap() {
		_capacity_idx = MIN_CAPACITY_INDEX;
	}

	HashMap(std::initializer_list<KeyValue<TKey, TValue>> p_init) {
		reserve(p_init.size());
		for (const KeyValue<TKey, TValue> &E : p_init) {
			insert(E.key, E.value);
		}
	}

	uint32_t debug_get_hash(uint32_t p_idx) {
		if (_size == 0) {
			return 0;
		}
		ERR_FAIL_INDEX_V(p_idx, get_capacity(), 0);
		return _hashes[p_idx];
	}
	Iterator debug_get_element(uint32_t p_idx) {
		if (_size == 0) {
			return Iterator();
		}
		ERR_FAIL_INDEX_V(p_idx, get_capacity(), Iterator());
		return Iterator(_elements[p_idx]);
	}

	~HashMap() {
		_clear_data();

		if (_elements != nullptr) {
			Memory::free_static(_elements);
			Memory::free_static(_hashes);
		}
	}
};
