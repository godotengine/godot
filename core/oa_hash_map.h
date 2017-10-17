/*************************************************************************/
/*  oa_hash_map.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "hashfuncs.h"
#include "math_funcs.h"
#include "os/copymem.h"
#include "os/memory.h"

// uncomment this to disable intial local storage.
#define OA_HASH_MAP_INITIAL_LOCAL_STORAGE

/**
 * This class implements a hash map datastructure that uses open addressing with
 * local probing.
 *
 * It can give huge performance improvements over a chained HashMap because of
 * the increased data locality.
 *
 * Because of that locality property it's important to not use "large" value
 * types as the "TData" type. If TData values are too big it can cause more
 * cache misses then chaining. If larger values are needed then storing those
 * in a separate array and using pointers or indices to reference them is the
 * better solution.
 *
 * This hash map also implements real-time incremental rehashing.
 *
 */
template <class TKey, class TData,
		uint16_t INITIAL_NUM_ELEMENTS = 64,
		class Hasher = HashMapHasherDefault,
		class Comparator = HashMapComparatorDefault<TKey> >
class OAHashMap {

private:
#ifdef OA_HASH_MAP_INITIAL_LOCAL_STORAGE
	TData local_data[INITIAL_NUM_ELEMENTS];
	TKey local_keys[INITIAL_NUM_ELEMENTS];
	uint32_t local_hashes[INITIAL_NUM_ELEMENTS];
	uint8_t local_flags[INITIAL_NUM_ELEMENTS / 4 + (INITIAL_NUM_ELEMENTS % 4 != 0 ? 1 : 0)];
#endif

	struct {
		TData *data;
		TKey *keys;
		uint32_t *hashes;

		// This is actually an array of bits, 4 bit pairs per octet.
		// | ba ba ba ba | ba ba ba ba | ....
		//
		// if a is set it means that there is an element present.
		// if b is set it means that an element was deleted. This is needed for
		//   the local probing to work without relocating any succeeding and
		//    colliding entries.
		uint8_t *flags;

		uint32_t capacity;
	} table, old_table;

	bool is_rehashing;
	uint32_t rehash_position;
	uint32_t rehash_amount;

	uint32_t elements;

	/* Methods */

	// returns true if the value already existed, false if it's a new entry
	bool _raw_set_with_hash(uint32_t p_hash, const TKey &p_key, const TData &p_data) {
		for (int i = 0; i < table.capacity; i++) {

			int pos = (p_hash + i) % table.capacity;

			int flags_pos = pos / 4;
			int flags_pos_offset = pos % 4;

			bool is_filled_flag = table.flags[flags_pos] & (1 << (2 * flags_pos_offset));
			bool is_deleted_flag = table.flags[flags_pos] & (1 << (2 * flags_pos_offset + 1));

			if (is_filled_flag) {
				if (table.hashes[pos] == p_hash && Comparator::compare(table.keys[pos], p_key)) {
					table.data[pos] = p_data;
					return true;
				}
				continue;
			}

			table.keys[pos] = p_key;
			table.data[pos] = p_data;
			table.hashes[pos] = p_hash;

			table.flags[flags_pos] |= (1 << (2 * flags_pos_offset));
			table.flags[flags_pos] &= ~(1 << (2 * flags_pos_offset + 1));

			return false;
		}
		return false;
	}

public:
	_FORCE_INLINE_ uint32_t get_capacity() const { return table.capacity; }
	_FORCE_INLINE_ uint32_t get_num_elements() const { return elements; }

	void set(const TKey &p_key, const TData &p_data) {

		uint32_t hash = Hasher::hash(p_key);

		// We don't progress the rehashing if the table just got resized
		// to keep the cost of this function low.
		if (is_rehashing) {

			// rehash progress

			for (int i = 0; i <= rehash_amount && rehash_position < old_table.capacity; rehash_position++) {

				int flags_pos = rehash_position / 4;
				int flags_pos_offset = rehash_position % 4;

				bool is_filled_flag = (old_table.flags[flags_pos] & (1 << (2 * flags_pos_offset))) > 0;
				bool is_deleted_flag = (old_table.flags[flags_pos] & (1 << (2 * flags_pos_offset + 1))) > 0;

				if (is_filled_flag) {
					_raw_set_with_hash(old_table.hashes[rehash_position], old_table.keys[rehash_position], old_table.data[rehash_position]);

					old_table.keys[rehash_position].~TKey();
					old_table.data[rehash_position].~TData();

					memnew_placement(&old_table.keys[rehash_position], TKey);
					memnew_placement(&old_table.data[rehash_position], TData);

					old_table.flags[flags_pos] &= ~(1 << (2 * flags_pos_offset));
					old_table.flags[flags_pos] |= (1 << (2 * flags_pos_offset + 1));
				}
			}

			if (rehash_position >= old_table.capacity) {

				// wohooo, we can get rid of the old table.
				is_rehashing = false;

#ifdef OA_HASH_MAP_INITIAL_LOCAL_STORAGE
				if (old_table.data == local_data) {
					// Everything is local, so no cleanup :P
				} else
#endif
				{
					memdelete_arr(old_table.data);
					memdelete_arr(old_table.keys);
					memdelete_arr(old_table.hashes);
					memdelete_arr(old_table.flags);
				}
			}
		}

		// Table is almost full, resize and start rehashing process.
		if (elements >= table.capacity * 0.7) {

			old_table.capacity = table.capacity;
			old_table.data = table.data;
			old_table.flags = table.flags;
			old_table.hashes = table.hashes;
			old_table.keys = table.keys;

			table.capacity = old_table.capacity * 2;

			table.data = memnew_arr(TData, table.capacity);
			table.flags = memnew_arr(uint8_t, table.capacity / 4 + (table.capacity % 4 != 0 ? 1 : 0));
			table.hashes = memnew_arr(uint32_t, table.capacity);
			table.keys = memnew_arr(TKey, table.capacity);

			zeromem(table.flags, table.capacity / 4 + (table.capacity % 4 != 0 ? 1 : 0));

			is_rehashing = true;
			rehash_position = 0;
			rehash_amount = (elements * 2) / (table.capacity * 0.7 - old_table.capacity);
		}

		if (!_raw_set_with_hash(hash, p_key, p_data))
			elements++;
	}

	/**
	 * returns true if the value was found, false otherwise.
	 *
	 * if r_data is not NULL then the value will be written to the object
	 * it points to.
	 */
	bool lookup(const TKey &p_key, TData *r_data) {

		uint32_t hash = Hasher::hash(p_key);

		bool check_old_table = is_rehashing;
		bool check_new_table = true;

		// search for the key and return the value associated with it
		//
		// if we're rehashing we need to check both the old and the
		// current table. If we find a value in the old table we still
		// need to continue searching in the new table as it might have
		// been added after

		TData *value = NULL;

		for (int i = 0; i < table.capacity; i++) {

			if (!check_new_table && !check_old_table) {

				break;
			}

			// if we're rehashing check the old table
			if (check_old_table && i < old_table.capacity) {

				int pos = (hash + i) % old_table.capacity;

				int flags_pos = pos / 4;
				int flags_pos_offset = pos % 4;

				bool is_filled_flag = (old_table.flags[flags_pos] & (1 << (2 * flags_pos_offset))) > 0;
				bool is_deleted_flag = (old_table.flags[flags_pos] & (1 << (2 * flags_pos_offset + 1))) > 0;

				if (is_filled_flag) {
					// found our entry?
					if (old_table.hashes[pos] == hash && Comparator::compare(old_table.keys[pos], p_key)) {
						value = &old_table.data[pos];
						check_old_table = false;
					}
				} else if (!is_deleted_flag) {

					// we hit an empty field here, we don't
					// need to further check this old table
					// because we know it's not in here.

					check_old_table = false;
				}
			}

			if (check_new_table) {

				int pos = (hash + i) % table.capacity;

				int flags_pos = pos / 4;
				int flags_pos_offset = pos % 4;

				bool is_filled_flag = (table.flags[flags_pos] & (1 << (2 * flags_pos_offset))) > 0;
				bool is_deleted_flag = (table.flags[flags_pos] & (1 << (2 * flags_pos_offset + 1))) > 0;

				if (is_filled_flag) {
					// found our entry?
					if (table.hashes[pos] == hash && Comparator::compare(table.keys[pos], p_key)) {
						if (r_data != NULL)
							*r_data = table.data[pos];
						return true;
					}
					continue;
				} else if (is_deleted_flag) {
					continue;
				} else if (value != NULL) {

					// We found a value in the old table
					if (r_data != NULL)
						*r_data = *value;
					return true;
				} else {
					check_new_table = false;
				}
			}
		}

		if (value != NULL) {
			if (r_data != NULL)
				*r_data = *value;
			return true;
		}
		return false;
	}

	_FORCE_INLINE_ bool has(const TKey &p_key) {
		return lookup(p_key, NULL);
	}

	void remove(const TKey &p_key) {
		uint32_t hash = Hasher::hash(p_key);

		bool check_old_table = is_rehashing;
		bool check_new_table = true;

		for (int i = 0; i < table.capacity; i++) {

			if (!check_new_table && !check_old_table) {
				return;
			}

			// if we're rehashing check the old table
			if (check_old_table && i < old_table.capacity) {

				int pos = (hash + i) % old_table.capacity;

				int flags_pos = pos / 4;
				int flags_pos_offset = pos % 4;

				bool is_filled_flag = (old_table.flags[flags_pos] & (1 << (2 * flags_pos_offset))) > 0;
				bool is_deleted_flag = (old_table.flags[flags_pos] & (1 << (2 * flags_pos_offset + 1))) > 0;

				if (is_filled_flag) {
					// found our entry?
					if (old_table.hashes[pos] == hash && Comparator::compare(old_table.keys[pos], p_key)) {
						old_table.keys[pos].~TKey();
						old_table.data[pos].~TData();

						memnew_placement(&old_table.keys[pos], TKey);
						memnew_placement(&old_table.data[pos], TData);

						old_table.flags[flags_pos] &= ~(1 << (2 * flags_pos_offset));
						old_table.flags[flags_pos] |= (1 << (2 * flags_pos_offset + 1));

						elements--;
						return;
					}
				} else if (!is_deleted_flag) {

					// we hit an empty field here, we don't
					// need to further check this old table
					// because we know it's not in here.

					check_old_table = false;
				}
			}

			if (check_new_table) {

				int pos = (hash + i) % table.capacity;

				int flags_pos = pos / 4;
				int flags_pos_offset = pos % 4;

				bool is_filled_flag = (table.flags[flags_pos] & (1 << (2 * flags_pos_offset))) > 0;
				bool is_deleted_flag = (table.flags[flags_pos] & (1 << (2 * flags_pos_offset + 1))) > 0;

				if (is_filled_flag) {
					// found our entry?
					if (table.hashes[pos] == hash && Comparator::compare(table.keys[pos], p_key)) {
						table.keys[pos].~TKey();
						table.data[pos].~TData();

						memnew_placement(&table.keys[pos], TKey);
						memnew_placement(&table.data[pos], TData);

						table.flags[flags_pos] &= ~(1 << (2 * flags_pos_offset));
						table.flags[flags_pos] |= (1 << (2 * flags_pos_offset + 1));

						// don't return here, this value might still be in the old table
						// if it was already relocated.

						elements--;
						return;
					}
					continue;
				} else if (is_deleted_flag) {
					continue;
				} else {
					check_new_table = false;
				}
			}
		}
	}

	struct Iterator {
		bool valid;

		uint32_t hash;

		const TKey *key;
		const TData *data;

	private:
		friend class OAHashMap;
		bool was_from_old_table;
	};

	Iterator iter() const {
		Iterator it;

		it.valid = false;
		it.was_from_old_table = false;

		bool check_old_table = is_rehashing;

		for (int i = 0; i < table.capacity; i++) {

			// if we're rehashing check the old table first
			if (check_old_table && i < old_table.capacity) {

				int pos = i;

				int flags_pos = pos / 4;
				int flags_pos_offset = pos % 4;

				bool is_filled_flag = (old_table.flags[flags_pos] & (1 << (2 * flags_pos_offset))) > 0;

				if (is_filled_flag) {
					it.valid = true;
					it.hash = old_table.hashes[pos];
					it.data = &old_table.data[pos];
					it.key = &old_table.keys[pos];

					it.was_from_old_table = true;

					return it;
				}
			}

			{

				int pos = i;

				int flags_pos = pos / 4;
				int flags_pos_offset = pos % 4;

				bool is_filled_flag = (table.flags[flags_pos] & (1 << (2 * flags_pos_offset))) > 0;

				if (is_filled_flag) {
					it.valid = true;
					it.hash = table.hashes[pos];
					it.data = &table.data[pos];
					it.key = &table.keys[pos];

					return it;
				}
			}
		}

		return it;
	}

	Iterator next_iter(const Iterator &p_iter) const {
		if (!p_iter.valid) {
			return p_iter;
		}

		Iterator it;

		it.valid = false;
		it.was_from_old_table = false;

		bool check_old_table = is_rehashing;

		// we use this to skip the first check or not
		bool was_from_old_table = p_iter.was_from_old_table;

		int prev_index = (p_iter.data - (p_iter.was_from_old_table ? old_table.data : table.data));

		if (!was_from_old_table) {
			prev_index++;
		}

		for (int i = prev_index; i < table.capacity; i++) {

			// if we're rehashing check the old table first
			if (check_old_table && i < old_table.capacity && !was_from_old_table) {

				int pos = i;

				int flags_pos = pos / 4;
				int flags_pos_offset = pos % 4;

				bool is_filled_flag = (old_table.flags[flags_pos] & (1 << (2 * flags_pos_offset))) > 0;

				if (is_filled_flag) {
					it.valid = true;
					it.hash = old_table.hashes[pos];
					it.data = &old_table.data[pos];
					it.key = &old_table.keys[pos];

					it.was_from_old_table = true;

					return it;
				}
			}

			was_from_old_table = false;

			{
				int pos = i;

				int flags_pos = pos / 4;
				int flags_pos_offset = pos % 4;

				bool is_filled_flag = (table.flags[flags_pos] & (1 << (2 * flags_pos_offset))) > 0;

				if (is_filled_flag) {
					it.valid = true;
					it.hash = table.hashes[pos];
					it.data = &table.data[pos];
					it.key = &table.keys[pos];

					return it;
				}
			}
		}

		return it;
	}

	OAHashMap(uint32_t p_initial_capacity = INITIAL_NUM_ELEMENTS) {

#ifdef OA_HASH_MAP_INITIAL_LOCAL_STORAGE

		if (p_initial_capacity <= INITIAL_NUM_ELEMENTS) {
			table.data = local_data;
			table.keys = local_keys;
			table.hashes = local_hashes;
			table.flags = local_flags;

			zeromem(table.flags, INITIAL_NUM_ELEMENTS / 4 + (INITIAL_NUM_ELEMENTS % 4 != 0 ? 1 : 0));

			table.capacity = INITIAL_NUM_ELEMENTS;
			elements = 0;
		} else
#endif
		{
			table.data = memnew_arr(TData, p_initial_capacity);
			table.keys = memnew_arr(TKey, p_initial_capacity);
			table.hashes = memnew_arr(uint32_t, p_initial_capacity);
			table.flags = memnew_arr(uint8_t, p_initial_capacity / 4 + (p_initial_capacity % 4 != 0 ? 1 : 0));

			zeromem(table.flags, p_initial_capacity / 4 + (p_initial_capacity % 4 != 0 ? 1 : 0));

			table.capacity = p_initial_capacity;
			elements = 0;
		}

		is_rehashing = false;
		rehash_position = 0;
	}

	~OAHashMap() {
#ifdef OA_HASH_MAP_INITIAL_LOCAL_STORAGE
		if (table.capacity <= INITIAL_NUM_ELEMENTS) {
			return; // Everything is local, so no cleanup :P
		}
#endif
		if (is_rehashing) {

#ifdef OA_HASH_MAP_INITIAL_LOCAL_STORAGE
			if (old_table.data == local_data) {
				// Everything is local, so no cleanup :P
			} else
#endif
			{
				memdelete_arr(old_table.data);
				memdelete_arr(old_table.keys);
				memdelete_arr(old_table.hashes);
				memdelete_arr(old_table.flags);
			}
		}

		memdelete_arr(table.data);
		memdelete_arr(table.keys);
		memdelete_arr(table.hashes);
		memdelete_arr(table.flags);
	}
};

#endif
