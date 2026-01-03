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
class AHashSet final : public RawAHashTable<AHashSet<TKey, Hasher, Comparator>, TKey, Hasher, Comparator> {
	using Base = RawAHashTable<AHashSet<TKey, Hasher, Comparator>, TKey, Hasher, Comparator>;

	using Base::_capacity_mask;
	using Base::_clear_metadata;
	using Base::_get_probe_length;
	using Base::_get_resize_count;
	using Base::_hash;
	using Base::_insert_metadata;
	using Base::_lookup_idx;
	using Base::_lookup_idx_with_hash;
	using Base::_metadata;
	using Base::_resize_and_rehash;
	using Base::_size;

public:
	using Base::clear;
	using Base::EMPTY_HASH;
	using Base::get_capacity;
	using Base::has;
	using Base::INITIAL_CAPACITY;
	using Base::is_empty;
	using Base::reserve;
	using Base::reset;
	using Base::size;

protected:
	// `friend` declaration is needed, otherwise RawAHashTable was not able to read encapsulated methods.
	// Type aliases seem to be broken with friend declarations.
	friend class RawAHashTable<AHashSet<TKey, Hasher, Comparator>, TKey, Hasher, Comparator>;

	const TKey &_get_key(uint32_t p_idx) const {
		return _elements[p_idx];
	}

	void _resize_elements(uint32_t p_new_capacity) {
		_elements = reinterpret_cast<TKey *>(Memory::realloc_static(_elements, sizeof(TKey) * p_new_capacity));
	}

	bool _is_elements_valid() const {
		return _elements != nullptr;
	}

	void _clear_elements() {
		if constexpr (!std::is_trivially_destructible_v<TKey>) {
			for (uint32_t i = 0; i < _size; i++) {
				_elements[i].~TKey();
			}
		}
	}

	void _free_elements() {
		Memory::free_static(_elements);
		_elements = nullptr;
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

	virtual ~AHashSet() override {
		reset();
	}
};
