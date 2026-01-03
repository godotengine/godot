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

#include "core/os/memory.h"
#include "core/string/print_string.h"
#include "core/templates/hashfuncs.h"
#include "core/templates/pair.h"

#include <initializer_list>

class String;
class StringName;
class Variant;

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
	static_assert(EMPTY_HASH == 0, "EMPTY_HASH must always be 0 for the memcpy() optimization.");

private:
	struct Metadata {
		uint32_t hash;
		uint32_t element_idx;
	};

	static_assert(sizeof(Metadata) == 8);

	typedef KeyValue<TKey, TValue> MapKeyValue;
	MapKeyValue *_elements = nullptr;
	Metadata *_metadata = nullptr;

	// Due to optimization, this is `capacity - 1`. Use + 1 to get normal capacity.
	uint32_t _capacity_mask = 0;
	uint32_t _size = 0;

	uint32_t _hash(const TKey &p_key) const {
		uint32_t hash = Hasher::hash(p_key);

		if (unlikely(hash == EMPTY_HASH)) {
			hash = EMPTY_HASH + 1;
		}

		return hash;
	}

	static _FORCE_INLINE_ uint32_t _get_resize_count(uint32_t p_capacity_mask) {
		return p_capacity_mask ^ (p_capacity_mask + 1) >> 2; // = get_capacity() * 0.75 - 1; Works only if p_capacity_mask = 2^n - 1.
	}

	static _FORCE_INLINE_ uint32_t _get_probe_length(uint32_t p_meta_idx, uint32_t p_hash, uint32_t p_capacity) {
		const uint32_t original_idx = p_hash & p_capacity;
		return (p_meta_idx - original_idx + p_capacity + 1) & p_capacity;
	}

	bool _lookup_idx(const TKey &p_key, uint32_t &r_element_idx, uint32_t &r_meta_idx) const {
		if (unlikely(_elements == nullptr)) {
			return false; // Failed lookups, no _elements.
		}
		return _lookup_idx_with_hash(p_key, r_element_idx, r_meta_idx, _hash(p_key));
	}

	bool _lookup_idx_with_hash(const TKey &p_key, uint32_t &r_element_idx, uint32_t &r_meta_idx, uint32_t p_hash) const {
		if (unlikely(_elements == nullptr)) {
			return false; // Failed lookups, no _elements.
		}

		uint32_t meta_idx = p_hash & _capacity_mask;
		Metadata metadata = _metadata[meta_idx];
		if (metadata.hash == p_hash && Comparator::compare(_elements[metadata.element_idx].key, p_key)) {
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
			if (metadata.hash == p_hash && Comparator::compare(_elements[metadata.element_idx].key, p_key)) {
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
			_metadata[meta_idx] = Metadata{ p_hash, p_element_idx };
			return meta_idx;
		}

		uint32_t distance = 1;
		meta_idx = (meta_idx + 1) & _capacity_mask;
		Metadata metadata;
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

		Metadata *old_map_data = _metadata;

		_metadata = reinterpret_cast<Metadata *>(Memory::alloc_static_zeroed(sizeof(Metadata) * real_capacity));
		_elements = reinterpret_cast<MapKeyValue *>(Memory::realloc_static(_elements, sizeof(MapKeyValue) * (_get_resize_count(_capacity_mask) + 1)));

		if (_size != 0) {
			for (uint32_t i = 0; i < real_old_capacity; i++) {
				Metadata metadata = old_map_data[i];
				if (metadata.hash != EMPTY_HASH) {
					_insert_metadata(metadata.hash, metadata.element_idx);
				}
			}
		}

		Memory::free_static(old_map_data);
	}

	int32_t _insert_element(const TKey &p_key, const TValue &p_value, uint32_t p_hash) {
		if (unlikely(_elements == nullptr)) {
			// Allocate on demand to save memory.

			uint32_t real_capacity = _capacity_mask + 1;
			_metadata = reinterpret_cast<Metadata *>(Memory::alloc_static_zeroed(sizeof(Metadata) * real_capacity));
			_elements = reinterpret_cast<MapKeyValue *>(Memory::alloc_static(sizeof(MapKeyValue) * (_get_resize_count(_capacity_mask) + 1)));
		}

		if (unlikely(_size > _get_resize_count(_capacity_mask))) {
			_resize_and_rehash(_capacity_mask * 2);
		}

		memnew_placement(&_elements[_size], MapKeyValue(p_key, p_value));

		_insert_metadata(p_hash, _size);
		_size++;
		return _size - 1;
	}

	void _init_from(const AHashMap &p_other) {
		_capacity_mask = p_other._capacity_mask;
		uint32_t real_capacity = _capacity_mask + 1;
		_size = p_other._size;

		if (p_other._size == 0) {
			return;
		}

		_metadata = reinterpret_cast<Metadata *>(Memory::alloc_static(sizeof(Metadata) * real_capacity));
		_elements = reinterpret_cast<MapKeyValue *>(Memory::alloc_static(sizeof(MapKeyValue) * (_get_resize_count(_capacity_mask) + 1)));

		if constexpr (std::is_trivially_copyable_v<TKey> && std::is_trivially_copyable_v<TValue>) {
			void *destination = _elements;
			const void *source = p_other._elements;
			memcpy(destination, source, sizeof(MapKeyValue) * _size);
		} else {
			for (uint32_t i = 0; i < _size; i++) {
				memnew_placement(&_elements[i], MapKeyValue(p_other._elements[i]));
			}
		}

		memcpy(_metadata, p_other._metadata, sizeof(Metadata) * real_capacity);
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

		memset(_metadata, EMPTY_HASH, (_capacity_mask + 1) * sizeof(Metadata));
		if constexpr (!(std::is_trivially_destructible_v<TKey> && std::is_trivially_destructible_v<TValue>)) {
			for (uint32_t i = 0; i < _size; i++) {
				_elements[i].key.~TKey();
				_elements[i].value.~TValue();
			}
		}

		_size = 0;
	}

	TValue &get(const TKey &p_key) {
		uint32_t element_idx = 0;
		uint32_t meta_idx = 0;
		bool exists = _lookup_idx(p_key, element_idx, meta_idx);
		CRASH_COND_MSG(!exists, "AHashMap key not found.");
		return _elements[element_idx].value;
	}

	const TValue &get(const TKey &p_key) const {
		uint32_t element_idx = 0;
		uint32_t meta_idx = 0;
		bool exists = _lookup_idx(p_key, element_idx, meta_idx);
		CRASH_COND_MSG(!exists, "AHashMap key not found.");
		return _elements[element_idx].value;
	}

	const TValue *getptr(const TKey &p_key) const {
		uint32_t element_idx = 0;
		uint32_t meta_idx = 0;
		bool exists = _lookup_idx(p_key, element_idx, meta_idx);

		if (exists) {
			return &_elements[element_idx].value;
		}
		return nullptr;
	}

	TValue *getptr(const TKey &p_key) {
		uint32_t element_idx = 0;
		uint32_t meta_idx = 0;
		bool exists = _lookup_idx(p_key, element_idx, meta_idx);

		if (exists) {
			return &_elements[element_idx].value;
		}
		return nullptr;
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
		_elements[element_idx].key.~TKey();
		_elements[element_idx].value.~TValue();
		_size--;

		if (element_idx < _size) {
			memcpy((void *)&_elements[element_idx], (const void *)&_elements[_size], sizeof(MapKeyValue));
			uint32_t moved_element_idx = 0;
			uint32_t moved_meta_idx = 0;
			_lookup_idx(_elements[_size].key, moved_element_idx, moved_meta_idx);
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
		MapKeyValue &element = _elements[element_idx];
		const_cast<TKey &>(element.key) = p_new_key;

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
		return Iterator(_elements, _elements, _elements + _size);
	}
	_FORCE_INLINE_ Iterator end() {
		return Iterator(_elements + _size, _elements, _elements + _size);
	}
	_FORCE_INLINE_ Iterator last() {
		if (unlikely(_size == 0)) {
			return Iterator(nullptr, nullptr, nullptr);
		}
		return Iterator(_elements + _size - 1, _elements, _elements + _size);
	}

	Iterator find(const TKey &p_key) {
		uint32_t meta_idx = 0;
		uint32_t element_idx = 0;
		bool exists = _lookup_idx(p_key, element_idx, meta_idx);
		if (!exists) {
			return end();
		}
		return Iterator(_elements + element_idx, _elements, _elements + _size);
	}

	void remove(const Iterator &p_iter) {
		if (p_iter) {
			erase(p_iter->key);
		}
	}

	_FORCE_INLINE_ ConstIterator begin() const {
		return ConstIterator(_elements, _elements, _elements + _size);
	}
	_FORCE_INLINE_ ConstIterator end() const {
		return ConstIterator(_elements + _size, _elements, _elements + _size);
	}
	_FORCE_INLINE_ ConstIterator last() const {
		if (unlikely(_size == 0)) {
			return ConstIterator(nullptr, nullptr, nullptr);
		}
		return ConstIterator(_elements + _size - 1, _elements, _elements + _size);
	}

	ConstIterator find(const TKey &p_key) const {
		uint32_t element_idx = 0;
		uint32_t meta_idx = 0;
		bool exists = _lookup_idx(p_key, element_idx, meta_idx);
		if (!exists) {
			return end();
		}
		return ConstIterator(_elements + element_idx, _elements, _elements + _size);
	}

	/* Indexing */

	const TValue &operator[](const TKey &p_key) const {
		uint32_t element_idx = 0;
		uint32_t meta_idx = 0;
		bool exists = _lookup_idx(p_key, element_idx, meta_idx);
		CRASH_COND(!exists);
		return _elements[element_idx].value;
	}

	TValue &operator[](const TKey &p_key) {
		uint32_t element_idx = 0;
		uint32_t meta_idx = 0;
		uint32_t hash = _hash(p_key);
		bool exists = _lookup_idx_with_hash(p_key, element_idx, meta_idx, hash);

		if (exists) {
			return _elements[element_idx].value;
		} else {
			element_idx = _insert_element(p_key, TValue(), hash);
			return _elements[element_idx].value;
		}
	}

	/* Insert */

	Iterator insert(const TKey &p_key, const TValue &p_value) {
		uint32_t element_idx = 0;
		uint32_t meta_idx = 0;
		uint32_t hash = _hash(p_key);
		bool exists = _lookup_idx_with_hash(p_key, element_idx, meta_idx, hash);

		if (!exists) {
			element_idx = _insert_element(p_key, p_value, hash);
		} else {
			_elements[element_idx].value = p_value;
		}
		return Iterator(_elements + element_idx, _elements, _elements + _size);
	}

	// Inserts an element without checking if it already exists.
	Iterator insert_new(const TKey &p_key, const TValue &p_value) {
		DEV_ASSERT(!has(p_key));
		uint32_t hash = _hash(p_key);
		uint32_t element_idx = _insert_element(p_key, p_value, hash);
		return Iterator(_elements + element_idx, _elements, _elements + _size);
	}

	/* Array methods. */

	// Unsafe. Changing keys and going outside the bounds of an array can lead to undefined behavior.
	KeyValue<TKey, TValue> *get_elements_ptr() {
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

	KeyValue<TKey, TValue> &get_by_index(uint32_t p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, _size);
		return _elements[p_index];
	}

	bool erase_by_index(uint32_t p_index) {
		if (p_index >= size()) {
			return false;
		}
		return erase(_elements[p_index].key);
	}

	/* Constructors */

	AHashMap(AHashMap &&p_other) {
		_elements = p_other._elements;
		_metadata = p_other._metadata;
		_capacity_mask = p_other._capacity_mask;
		_size = p_other._size;

		p_other._elements = nullptr;
		p_other._metadata = nullptr;
		p_other._capacity_mask = 0;
		p_other._size = 0;
	}

	AHashMap(const AHashMap &p_other) {
		_init_from(p_other);
	}

	void operator=(const AHashMap &p_other) {
		if (this == &p_other) {
			return; // Ignore self assignment.
		}

		reset();

		_init_from(p_other);
	}

	explicit AHashMap(uint32_t p_initial_capacity) {
		// Capacity can't be 0 and must be 2^n - 1.
		_capacity_mask = MAX(4u, p_initial_capacity);
		_capacity_mask = next_power_of_2(_capacity_mask) - 1;
	}
	AHashMap() :
			_capacity_mask(INITIAL_CAPACITY - 1) {
	}

	AHashMap(std::initializer_list<KeyValue<TKey, TValue>> p_init) {
		reserve(p_init.size());
		for (const KeyValue<TKey, TValue> &E : p_init) {
			insert(E.key, E.value);
		}
	}

	void reset() {
		if (_elements != nullptr) {
			if constexpr (!(std::is_trivially_destructible_v<TKey> && std::is_trivially_destructible_v<TValue>)) {
				for (uint32_t i = 0; i < _size; i++) {
					_elements[i].key.~TKey();
					_elements[i].value.~TValue();
				}
			}
			Memory::free_static(_elements);
			Memory::free_static(_metadata);
			_elements = nullptr;
		}
		_capacity_mask = INITIAL_CAPACITY - 1;
		_size = 0;
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
