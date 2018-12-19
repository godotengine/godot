/*************************************************************************/
/*  oa_hash_map.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef OA_HASH_MAP_H
#define OA_HASH_MAP_H

#include "core/hashfuncs.h"
#include "core/math/math_funcs.h"
#include "core/os/copymem.h"
#include "core/os/memory.h"

/**
 * A HashMap implementation that uses open addressing with robinhood hashing.
 * Robinhood hashing swaps out entries that have a smaller probing distance
 * than the to-be-inserted entry, that evens out the average probing distance
 * and enables faster lookups.
 *
 * The entries are stored inplace, so huge keys or values might fill cache lines
 * a lot faster.
 */
template <class TKey, class TValue,
		class Hasher = HashMapHasherDefault,
		class Comparator = HashMapComparatorDefault<TKey> >
class OAHashMap {

private:
	TValue *values;
	TKey *keys;
	uint32_t *hashes;

	uint32_t capacity;

	uint32_t num_elements;

	static const uint32_t EMPTY_HASH = 0;
	static const uint32_t DELETED_HASH_BIT = 1 << 31;

	_FORCE_INLINE_ uint32_t _hash(const TKey &p_key) {
		uint32_t hash = Hasher::hash(p_key);

		if (hash == EMPTY_HASH) {
			hash = EMPTY_HASH + 1;
		} else if (hash & DELETED_HASH_BIT) {
			hash &= ~DELETED_HASH_BIT;
		}

		return hash;
	}

	_FORCE_INLINE_ uint32_t _get_probe_length(uint32_t p_pos, uint32_t p_hash) {
		p_hash = p_hash & ~DELETED_HASH_BIT; // we don't care if it was deleted or not

		uint32_t original_pos = p_hash % capacity;

		return p_pos - original_pos;
	}

	_FORCE_INLINE_ void _construct(uint32_t p_pos, uint32_t p_hash, const TKey &p_key, const TValue &p_value) {
		memnew_placement(&keys[p_pos], TKey(p_key));
		memnew_placement(&values[p_pos], TValue(p_value));
		hashes[p_pos] = p_hash;

		num_elements++;
	}

	bool _lookup_pos(const TKey &p_key, uint32_t &r_pos) {
		uint32_t hash = _hash(p_key);
		uint32_t pos = hash % capacity;
		uint32_t distance = 0;

		while (42) {
			if (hashes[pos] == EMPTY_HASH) {
				return false;
			}

			if (distance > _get_probe_length(pos, hashes[pos])) {
				return false;
			}

			if (hashes[pos] == hash && Comparator::compare(keys[pos], p_key)) {
				r_pos = pos;
				return true;
			}

			pos = (pos + 1) % capacity;
			distance++;
		}
	}

	void _insert_with_hash(uint32_t p_hash, const TKey &p_key, const TValue &p_value) {

		uint32_t hash = p_hash;
		uint32_t distance = 0;
		uint32_t pos = hash % capacity;

		TKey key = p_key;
		TValue value = p_value;

		while (42) {
			if (hashes[pos] == EMPTY_HASH) {
				_construct(pos, hash, key, value);

				return;
			}

			// not an empty slot, let's check the probing length of the existing one
			uint32_t existing_probe_len = _get_probe_length(pos, hashes[pos]);
			if (existing_probe_len < distance) {

				if (hashes[pos] & DELETED_HASH_BIT) {
					// we found a place where we can fit in!
					_construct(pos, hash, key, value);

					return;
				}

				SWAP(hash, hashes[pos]);
				SWAP(key, keys[pos]);
				SWAP(value, values[pos]);
				distance = existing_probe_len;
			}

			pos = (pos + 1) % capacity;
			distance++;
		}
	}
	void _resize_and_rehash() {

		TKey *old_keys = keys;
		TValue *old_values = values;
		uint32_t *old_hashes = hashes;

		uint32_t old_capacity = capacity;

		capacity = old_capacity * 2;
		num_elements = 0;

		keys = memnew_arr(TKey, capacity);
		values = memnew_arr(TValue, capacity);
		hashes = memnew_arr(uint32_t, capacity);

		for (uint32_t i = 0; i < capacity; i++) {
			hashes[i] = 0;
		}

		for (uint32_t i = 0; i < old_capacity; i++) {
			if (old_hashes[i] == EMPTY_HASH) {
				continue;
			}
			if (old_hashes[i] & DELETED_HASH_BIT) {
				continue;
			}

			_insert_with_hash(old_hashes[i], old_keys[i], old_values[i]);
		}

		memdelete_arr(old_keys);
		memdelete_arr(old_values);
		memdelete_arr(old_hashes);
	}

public:
	_FORCE_INLINE_ uint32_t get_capacity() const { return capacity; }
	_FORCE_INLINE_ uint32_t get_num_elements() const { return num_elements; }

	void insert(const TKey &p_key, const TValue &p_value) {

		if ((float)num_elements / (float)capacity > 0.9) {
			_resize_and_rehash();
		}

		uint32_t hash = _hash(p_key);

		_insert_with_hash(hash, p_key, p_value);
	}

	void set(const TKey &p_key, const TValue &p_data) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (exists) {
			values[pos].~TValue();
			memnew_placement(&values[pos], TValue(p_data));
		} else {
			insert(p_key, p_data);
		}
	}

	/**
	 * returns true if the value was found, false otherwise.
	 *
	 * if r_data is not NULL then the value will be written to the object
	 * it points to.
	 */
	bool lookup(const TKey &p_key, TValue &r_data) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (exists) {
			r_data.~TValue();
			memnew_placement(&r_data, TValue(values[pos]));
			return true;
		}

		return false;
	}

	_FORCE_INLINE_ bool has(const TKey &p_key) {
		uint32_t _pos = 0;
		return _lookup_pos(p_key, _pos);
	}

	void remove(const TKey &p_key) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (!exists) {
			return;
		}

		hashes[pos] |= DELETED_HASH_BIT;
		values[pos].~TValue();
		keys[pos].~TKey();
		num_elements--;
	}

	struct Iterator {
		bool valid;

		const TKey *key;
		const TValue *value;

	private:
		uint32_t pos;
		friend class OAHashMap;
	};

	Iterator iter() const {
		Iterator it;

		it.valid = true;
		it.pos = 0;

		return next_iter(it);
	}

	Iterator next_iter(const Iterator &p_iter) const {

		if (!p_iter.valid) {
			return p_iter;
		}

		Iterator it;
		it.valid = false;
		it.pos = p_iter.pos;
		it.key = NULL;
		it.value = NULL;

		for (uint32_t i = it.pos; i < capacity; i++) {
			it.pos = i + 1;

			if (hashes[i] == EMPTY_HASH) {
				continue;
			}
			if (hashes[i] & DELETED_HASH_BIT) {
				continue;
			}

			it.valid = true;
			it.key = &keys[i];
			it.value = &values[i];
			return it;
		}

		return it;
	}

	OAHashMap(uint32_t p_initial_capacity = 64) {

		capacity = p_initial_capacity;
		num_elements = 0;

		keys = memnew_arr(TKey, p_initial_capacity);
		values = memnew_arr(TValue, p_initial_capacity);
		hashes = memnew_arr(uint32_t, p_initial_capacity);

		for (uint32_t i = 0; i < p_initial_capacity; i++) {
			hashes[i] = 0;
		}
	}

	~OAHashMap() {

		memdelete_arr(keys);
		memdelete_arr(values);
		memdelete_arr(hashes);
	}
};

#endif
