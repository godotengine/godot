/**************************************************************************/
/*  raw_a_hash_table.h                                                    */
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
#include "core/templates/hashfuncs.h"

struct RawAHashTableMetadata {
	uint32_t hash;
	uint32_t element_idx;
};
static_assert(sizeof(RawAHashTableMetadata) == 8);

/**
 * The abstract class behind the AHashMap and AHashSet containers.
 *
 * An array-based implementation of a hash table. It is very efficient in terms of performance and
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
 * IDs are not stable as they can break in the case of a deletion.
 *
 * When an element erase, its place is taken by the element from the end.
 *
 *        <-------------
 *      |               |
 *  6 8 X 9 32 -1 5 -10 7 X X X
 *  6 8 7 9 32 -1 5 -10 X X X X
 *
 *	Element pointers are not stable as they can break in the case of a deletion or rehash.
 *
 */
template <
		typename Derived,
		typename TKey,
		typename Hasher,
		typename Comparator>
class RawAHashTable {
public:
	// Must be a power of two.
	static constexpr uint32_t INITIAL_CAPACITY = 16;
	static constexpr uint32_t EMPTY_HASH = 0;
	static_assert(EMPTY_HASH == 0, "EMPTY_HASH must always be 0 for the memcpy() optimization.");

protected:
	RawAHashTableMetadata *_metadata = nullptr;

	// Due to optimization, this is `capacity - 1`. Use + 1 to get normal capacity.
	uint32_t _capacity_mask = 0;
	uint32_t _size = 0;

	static _FORCE_INLINE_ uint32_t _get_resize_count(uint32_t p_capacity_mask) {
		return p_capacity_mask ^ (p_capacity_mask + 1) >> 2; // = get_capacity() * 0.75 - 1; Works only if p_capacity_mask = 2^n - 1.
	}

	static _FORCE_INLINE_ uint32_t _get_probe_length(uint32_t p_meta_idx, uint32_t p_hash, uint32_t p_capacity) {
		const uint32_t original_idx = p_hash & p_capacity;
		return (p_meta_idx - original_idx + p_capacity + 1) & p_capacity;
	}

	// CRTP (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
	// The following methods should be implemented by the derived class without the `_v` prefix.
	_FORCE_INLINE_ const TKey &_v_get_key(uint32_t p_idx) const {
		return static_cast<const Derived *>(this)->_get_key(p_idx);
	}
	_FORCE_INLINE_ void _v_resize_elements(uint32_t p_new_capacity) {
		static_cast<Derived *>(this)->_resize_elements(p_new_capacity);
	}
	_FORCE_INLINE_ bool _v_is_elements_valid() const {
		return static_cast<const Derived *>(this)->_is_elements_valid();
	}
	_FORCE_INLINE_ void _v_clear_elements() {
		static_cast<Derived *>(this)->_clear_elements();
	}
	_FORCE_INLINE_ void _v_free_elements() {
		static_cast<Derived *>(this)->_free_elements();
	}

	_FORCE_INLINE_ uint32_t _hash(const TKey &p_key) const {
		uint32_t hash = Hasher::hash(p_key);

		if (unlikely(hash == EMPTY_HASH)) {
			hash = EMPTY_HASH + 1;
		}

		return hash;
	}

	_FORCE_INLINE_ bool _eq(const TKey &a, const TKey &b) const {
		return Comparator::compare(a, b);
	}

	_FORCE_INLINE_ bool _lookup_idx(const TKey &p_key, uint32_t &r_element_idx, uint32_t &r_meta_idx) const {
		if (unlikely(!_v_is_elements_valid())) {
			return false; // Failed lookups, no _elements.
		}
		return _lookup_idx_with_hash(p_key, r_element_idx, r_meta_idx, _hash(p_key));
	}

	bool _lookup_idx_with_hash(const TKey &p_key, uint32_t &r_element_idx, uint32_t &r_meta_idx, uint32_t p_hash) const {
		if (unlikely(!_v_is_elements_valid())) {
			return false; // Failed lookups, no _elements.
		}

		uint32_t meta_idx = p_hash & _capacity_mask;
		RawAHashTableMetadata metadata = _metadata[meta_idx];
		if (metadata.hash == p_hash && _eq(_v_get_key(metadata.element_idx), p_key)) {
			r_element_idx = metadata.element_idx;
			r_meta_idx = meta_idx;
			return true;
		}

		if (metadata.hash == EMPTY_HASH) {
			return false;
		}

		// A collision occurred.
		meta_idx = (meta_idx + 1) & _capacity_mask;
		uint32_t distance = 1;
		while (true) {
			metadata = _metadata[meta_idx];
			if (metadata.hash == p_hash && _eq(_v_get_key(metadata.element_idx), p_key)) {
				r_element_idx = metadata.element_idx;
				r_meta_idx = meta_idx;
				return true;
			}

			if (metadata.hash == EMPTY_HASH) {
				return false;
			}

			if (distance > _get_probe_length(meta_idx, metadata.hash, _capacity_mask)) {
				return false;
			}

			meta_idx = (meta_idx + 1) & _capacity_mask;
			distance++;
		}
	}

	uint32_t _insert_metadata(uint32_t p_hash, uint32_t p_element_idx) {
		uint32_t meta_idx = p_hash & _capacity_mask;

		if (_metadata[meta_idx].hash == EMPTY_HASH) {
			_metadata[meta_idx] = RawAHashTableMetadata{ p_hash, p_element_idx };
			return meta_idx;
		}

		uint32_t distance = 1;
		meta_idx = (meta_idx + 1) & _capacity_mask;
		RawAHashTableMetadata metadata;
		metadata.hash = p_hash;
		metadata.element_idx = p_element_idx;

		while (true) {
			if (_metadata[meta_idx].hash == EMPTY_HASH) {
#ifdef DEV_ENABLED
				if (unlikely(distance > 12)) {
					WARN_PRINT("Excessive collision count, is the right hash function being used?");
				}
#endif
				_metadata[meta_idx] = metadata;
				return meta_idx;
			}

			// Not an empty slot, let's check the probing length of the existing one.
			uint32_t existing_probe_len = _get_probe_length(meta_idx, _metadata[meta_idx].hash, _capacity_mask);
			if (existing_probe_len < distance) {
				SWAP(metadata, _metadata[meta_idx]);
				distance = existing_probe_len;
			}

			meta_idx = (meta_idx + 1) & _capacity_mask;
			distance++;
		}
	}

	void _resize_and_rehash(uint32_t p_new_capacity) {
		uint32_t real_old_capacity = _capacity_mask + 1;
		// Capacity can't be 0 and must be 2^n - 1.
		_capacity_mask = MAX(4u, p_new_capacity);
		uint32_t real_capacity = next_power_of_2(_capacity_mask);
		_capacity_mask = real_capacity - 1;

		RawAHashTableMetadata *old_map_data = _metadata;

		_metadata = reinterpret_cast<RawAHashTableMetadata *>(Memory::alloc_static_zeroed(sizeof(RawAHashTableMetadata) * real_capacity));
		_v_resize_elements(_get_resize_count(_capacity_mask) + 1);

		if (_size != 0) {
			for (uint32_t i = 0; i < real_old_capacity; i++) {
				RawAHashTableMetadata metadata = old_map_data[i];
				if (metadata.hash != EMPTY_HASH) {
					_insert_metadata(metadata.hash, metadata.element_idx);
				}
			}
		}

		Memory::free_static(old_map_data);
	}

	void _clear_metadata() {
		memset(_metadata, EMPTY_HASH, (_capacity_mask + 1) * sizeof(RawAHashTableMetadata));
	}

public:
	_FORCE_INLINE_ uint32_t get_capacity() const { return _capacity_mask + 1; }
	_FORCE_INLINE_ uint32_t size() const { return _size; }

	_FORCE_INLINE_ bool is_empty() const {
		return size() == 0;
	}

	void clear() {
		if (!_v_is_elements_valid() || _size == 0) {
			return;
		}

		_clear_metadata();
		_v_clear_elements();

		_size = 0;
	}

	bool has(const TKey &p_key) const {
		uint32_t _idx = 0;
		uint32_t meta_idx = 0;
		return _lookup_idx(p_key, _idx, meta_idx);
	}

	// Reserves space for a number of elements, useful to avoid many resizes and rehashes.
	// If adding a known (possibly large) number of elements at once, must be larger than old capacity.
	void reserve(uint32_t p_new_capacity) {
		if (!_v_is_elements_valid()) {
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

	void reset() {
		if (_v_is_elements_valid()) {
			_v_clear_elements();
			_v_free_elements();
			Memory::free_static(_metadata);
		}
		_capacity_mask = INITIAL_CAPACITY - 1;
		_size = 0;
	}

	virtual ~RawAHashTable() {}
};
