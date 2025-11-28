/**************************************************************************/
/*  a_hash_set.h                                                          */
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

#include "core/string/print_string.h"
#include "core/templates/raw_a_hash_table.h"

#include <initializer_list>

/**
 * An array-based implementation of a hash set. See parent class RawAHashTable for details.
 *
 * Use RBSet if you need to iterate over sorted elements.
 *
 */
template <typename TKey,
		typename Hasher = HashMapHasherDefault,
		typename Comparator = HashMapComparatorDefault<TKey>>
class AHashSet final : public RawAHashTable<TKey, Hasher, Comparator> {
	using Base = RawAHashTable<TKey, Hasher, Comparator>;

	using Base::_capacity_mask;
	using Base::_get_probe_length;
	using Base::_get_resize_count;
	using Base::_hash;
	using Base::_insert_metadata;
	using Base::_lookup_idx;
	using Base::_lookup_idx_with_hash;
	using Base::_metadata;
	using Base::_resize_and_rehash;
	using Base::_size;
	using Base::_clear_metadata;
	using Base::EMPTY_HASH;
	using Base::INITIAL_CAPACITY;

protected:
	virtual const TKey &_get_key(uint32_t p_idx) const override {
		return _elements[p_idx];
	}

	virtual void _resize_elements(uint32_t p_new_capacity) override {
		_elements = reinterpret_cast<TKey *>(Memory::realloc_static(_elements, sizeof(TKey) * p_new_capacity));
	}

	virtual bool _is_elements_valid() const override {
		return _elements != nullptr;
	}

private:
	TKey *_elements = nullptr;

	int32_t _insert_element(const TKey &p_key, uint32_t p_hash) {
		if (unlikely(_elements == nullptr)) {
			// Allocate on demand to save memory.

			uint32_t real_capacity = _capacity_mask + 1;
			_metadata = reinterpret_cast<RawAHashTableMetadata *>(Memory::alloc_static_zeroed(sizeof(RawAHashTableMetadata) * real_capacity));
			_elements = reinterpret_cast<TKey *>(Memory::alloc_static(sizeof(TKey) * (_get_resize_count(_capacity_mask) + 1)));
		}

		if (unlikely(_size > _get_resize_count(_capacity_mask))) {
			_resize_and_rehash(_capacity_mask * 2);
		}

		memnew_placement(&_elements[_size], TKey(p_key));

		_insert_metadata(p_hash, _size);
		_size++;
		return _size - 1;
	}

	void _init_from(const AHashSet &p_other) {
		_capacity_mask = p_other._capacity_mask;
		uint32_t real_capacity = _capacity_mask + 1;
		_size = p_other._size;

		if (p_other._size == 0) {
			return;
		}

		_metadata = reinterpret_cast<RawAHashTableMetadata *>(Memory::alloc_static(sizeof(RawAHashTableMetadata) * real_capacity));
		_elements = reinterpret_cast<TKey *>(Memory::alloc_static(sizeof(TKey) * (_get_resize_count(_capacity_mask) + 1)));

		if constexpr (std::is_trivially_copyable_v<TKey>) {
			void *destination = _elements;
			const void *source = p_other._elements;
			memcpy(destination, source, sizeof(TKey) * _size);
		} else {
			for (uint32_t i = 0; i < _size; i++) {
				memnew_placement(&_elements[i], TKey(p_other._elements[i]));
			}
		}

		memcpy(_metadata, p_other._metadata, sizeof(RawAHashTableMetadata) * real_capacity);
	}

public:
	/* Standard Godot Container API */

	_FORCE_INLINE_ uint32_t get_capacity() const { return _capacity_mask + 1; }
	_FORCE_INLINE_ uint32_t size() const { return _size; }

	_FORCE_INLINE_ bool is_empty() const {
		return _size == 0;
	}

	void clear() {
		if (_elements == nullptr || _size == 0) {
			return;
		}

		_clear_metadata();
		if constexpr (!std::is_trivially_destructible_v<TKey>) {
			for (uint32_t i = 0; i < _size; i++) {
				_elements[i].~TKey();
			}
		}

		_size = 0;
	}

	bool has(const TKey &p_key) const {
		uint32_t _idx = 0;
		uint32_t meta_idx = 0;
		return _lookup_idx(p_key, _idx, meta_idx);
	}

	bool erase(const TKey &p_key) {
		uint32_t meta_idx = 0;
		uint32_t element_idx = 0;
		bool exists = _lookup_idx(p_key, element_idx, meta_idx);

		if (!exists) {
			return false;
		}

		uint32_t next_meta_idx = (meta_idx + 1) & _capacity_mask;
		while (_metadata[next_meta_idx].hash != EMPTY_HASH && _get_probe_length(next_meta_idx, _metadata[next_meta_idx].hash, _capacity_mask) != 0) {
			SWAP(_metadata[next_meta_idx], _metadata[meta_idx]);

			meta_idx = next_meta_idx;
			next_meta_idx = (next_meta_idx + 1) & _capacity_mask;
		}

		_metadata[meta_idx].hash = EMPTY_HASH;
		_elements[element_idx].~TKey();
		_size--;

		if (element_idx < _size) {
			memcpy((void *)&_elements[element_idx], (const void *)&_elements[_size], sizeof(TKey));
			uint32_t moved_element_idx = 0;
			uint32_t moved_meta_idx = 0;
			_lookup_idx(_elements[_size], moved_element_idx, moved_meta_idx);
			_metadata[moved_meta_idx].element_idx = element_idx;
		}

		return true;
	}

	// Replace the key of an entry in-place, without invalidating iterators or changing the entries position during iteration.
	// p_old_key must exist in the map and p_new_key must not, unless it is equal to p_old_key.
	bool replace_key(const TKey &p_old_key, const TKey &p_new_key) {
		if (p_old_key == p_new_key) {
			return true;
		}
		uint32_t meta_idx = 0;
		uint32_t element_idx = 0;
		ERR_FAIL_COND_V(_lookup_idx(p_new_key, element_idx, meta_idx), false);
		ERR_FAIL_COND_V(!_lookup_idx(p_old_key, element_idx, meta_idx), false);
		TKey &element = _elements[element_idx];
		const_cast<TKey &>(element) = p_new_key;

		uint32_t next_meta_idx = (meta_idx + 1) & _capacity_mask;
		while (_metadata[next_meta_idx].hash != EMPTY_HASH && _get_probe_length(next_meta_idx, _metadata[next_meta_idx].hash, _capacity_mask) != 0) {
			SWAP(_metadata[next_meta_idx], _metadata[meta_idx]);

			meta_idx = next_meta_idx;
			next_meta_idx = (next_meta_idx + 1) & _capacity_mask;
		}

		_metadata[meta_idx].hash = EMPTY_HASH;

		uint32_t hash = _hash(p_new_key);
		_insert_metadata(hash, element_idx);

		return true;
	}

	// Reserves space for a number of elements, useful to avoid many resizes and rehashes.
	// If adding a known (possibly large) number of elements at once, must be larger than old capacity.
	void reserve(uint32_t p_new_capacity) {
		if (_elements == nullptr) {
			_capacity_mask = MAX(4u, p_new_capacity);
			_capacity_mask = next_power_of_2(_capacity_mask) - 1;
			return; // Unallocated yet.
		}
		if (p_new_capacity <= get_capacity()) {
			if (p_new_capacity < size()) {
				WARN_VERBOSE("reserve() called with a capacity smaller than the current size. This is likely a mistake.");
			}
			return;
		}
		_resize_and_rehash(p_new_capacity);
	}

	/** Iterator API **/

	struct Iterator {
		_FORCE_INLINE_ const TKey &operator*() const {
			return *pair;
		}
		_FORCE_INLINE_ const TKey *operator->() const {
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

		_FORCE_INLINE_ Iterator(TKey *p_key, TKey *p_begin, TKey *p_end) {
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

	private:
		TKey *pair = nullptr;
		TKey *begin = nullptr;
		TKey *end = nullptr;
	};

	void remove(const Iterator &p_iter) {
		if (p_iter) {
			erase(*p_iter);
		}
	}

	_FORCE_INLINE_ Iterator begin() const {
		return Iterator(_elements, _elements, _elements + _size);
	}
	_FORCE_INLINE_ Iterator end() const {
		return Iterator(_elements + _size, _elements, _elements + _size);
	}
	_FORCE_INLINE_ Iterator last() const {
		if (unlikely(_size == 0)) {
			return Iterator(nullptr, nullptr, nullptr);
		}
		return Iterator(_elements + _size - 1, _elements, _elements + _size);
	}

	Iterator find(const TKey &p_key) const {
		uint32_t element_idx = 0;
		uint32_t meta_idx = 0;
		bool exists = _lookup_idx(p_key, element_idx, meta_idx);
		if (!exists) {
			return end();
		}
		return Iterator(_elements + element_idx, _elements, _elements + _size);
	}

	/* Insert */

	Iterator insert(const TKey &p_key) {
		uint32_t element_idx = 0;
		uint32_t meta_idx = 0;
		uint32_t hash = _hash(p_key);
		bool exists = _lookup_idx_with_hash(p_key, element_idx, meta_idx, hash);

		if (!exists) {
			element_idx = _insert_element(p_key, hash);
		} else {
			_elements[element_idx] = p_key;
		}
		return Iterator(_elements + element_idx, _elements, _elements + _size);
	}

	// Inserts an element without checking if it already exists.
	//
	// SAFETY: In dev builds, the insertions are checked and causes a crash on bad use.
	// In release builds, the insertions are not checked, but bad use will cause duplicate
	// keys, which also affect iterators. Bad use does not cause undefined behavior.
	Iterator insert_new(const TKey &p_key) {
		DEV_ASSERT(!has(p_key));
		uint32_t hash = _hash(p_key);
		uint32_t element_idx = _insert_element(p_key, hash);
		return Iterator(_elements + element_idx, _elements, _elements + _size);
	}

	/* Array methods. */

	// Unsafe. Changing keys and going outside the bounds of an array can lead to undefined behavior.
	TKey *get_elements_ptr() {
		return _elements;
	}

	// Returns the element index. If not found, returns -1.
	int get_index(const TKey &p_key) {
		uint32_t element_idx = 0;
		uint32_t meta_idx = 0;
		bool exists = _lookup_idx(p_key, element_idx, meta_idx);
		if (!exists) {
			return -1;
		}
		return element_idx;
	}

	TKey &get_by_index(uint32_t p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, _size);
		return _elements[p_index];
	}

	bool erase_by_index(uint32_t p_index) {
		if (p_index >= size()) {
			return false;
		}
		return erase(_elements[p_index]);
	}

	/* Constructors */

	AHashSet(AHashSet &&p_other) {
		_elements = p_other._elements;
		_metadata = p_other._metadata;
		_capacity_mask = p_other._capacity_mask;
		_size = p_other._size;

		p_other._elements = nullptr;
		p_other._metadata = nullptr;
		p_other._capacity_mask = 0;
		p_other._size = 0;
	}

	AHashSet(const AHashSet &p_other) {
		_init_from(p_other);
	}

	void operator=(const AHashSet &p_other) {
		if (this == &p_other) {
			return; // Ignore self assignment.
		}

		reset();

		_init_from(p_other);
	}

	bool operator==(const AHashSet &p_other) const {
		if (_size != p_other._size) {
			return false;
		}
		for (uint32_t i = 0; i < _size; i++) {
			if (!p_other.has(_elements[i])) {
				return false;
			}
		}
		return true;
	}
	bool operator!=(const AHashSet &p_other) const {
		return !(*this == p_other);
	}

	AHashSet(uint32_t p_initial_capacity) {
		// Capacity can't be 0 and must be 2^n - 1.
		_capacity_mask = MAX(4u, p_initial_capacity);
		_capacity_mask = next_power_of_2(_capacity_mask) - 1;
	}
	AHashSet() {
		_capacity_mask = (INITIAL_CAPACITY - 1);
	}

	AHashSet(std::initializer_list<TKey> p_init) {
		reserve(p_init.size());
		for (const TKey &E : p_init) {
			insert(E);
		}
	}

	void reset() {
		if (_elements != nullptr) {
			if constexpr (!std::is_trivially_destructible_v<TKey>) {
				for (uint32_t i = 0; i < _size; i++) {
					_elements[i].~TKey();
				}
			}
			Memory::free_static(_elements);
			Memory::free_static(_metadata);
			_elements = nullptr;
		}
		_capacity_mask = INITIAL_CAPACITY - 1;
		_size = 0;
	}

	virtual ~AHashSet() override {
		reset();
	}
};
