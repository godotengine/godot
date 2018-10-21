/*************************************************************************/
/*  hash_map.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef HASH_MAP_H
#define HASH_MAP_H

#include "core/error_macros.h"
#include "core/hashfuncs.h"
#include "core/list.h"
#include "core/math/math_funcs.h"
#include "core/os/memory.h"
#include "core/ustring.h"

/**
 * @class HashMap
 * @author Juan Linietsky <reduzio@gmail.com>
 *
 * Implementation of a standard Hashing HashMap, for quick lookups of Data associated with a Key.
 * The implementation provides hashers for the default types, if you need a special kind of hasher, provide
 * your own.
 * @param TKey  Key, search is based on it, needs to be hasheable. It is unique in this container.
 * @param TData Data, data associated with the key
 * @param Hasher Hasher object, needs to provide a valid static hash function for TKey
 * @param Comparator comparator object, needs to be able to safely compare two TKey values. It needs to ensure that x == x for any items inserted in the map. Bear in mind that nan != nan when implementing an equality check.
 * @param MIN_HASH_TABLE_POWER Miminum size of the hash table, as a power of two. You rarely need to change this parameter.
 * @param RELATIONSHIP Relationship at which the hash table is resized. if amount of elements is RELATIONSHIP
 * times bigger than the hash table, table is resized to solve this condition. if RELATIONSHIP is zero, table is always MIN_HASH_TABLE_POWER.
*/

template <class TKey, class TData, class Hasher = HashMapHasherDefault, class Comparator = HashMapComparatorDefault<TKey>, uint8_t MIN_HASH_TABLE_POWER = 3, uint8_t RELATIONSHIP = 8>
class HashMap {
public:
	struct Pair {
		TKey key;
		TData data;

		Pair() {}
		Pair(const TKey &p_key, const TData &p_data) :
				key(p_key),
				data(p_data) {
		}
	};

	struct Element {
	private:
		friend class HashMap;

		uint32_t hash;
		Element *next;
		Element() { next = 0; }
		Pair pair;

	public:
		const TKey &key() const {
			return pair.key;
		}

		TData &value() {
			return pair.data;
		}

		const TData &value() const {
			return pair.data;
		}
	};

private:
	Element **hash_table;
	uint8_t hash_table_power;
	uint32_t elements;

	void _make_hash_table() {
		ERR_FAIL_COND(hash_table);

		uint32_t hash_table_size = (uint32_t)1 << MIN_HASH_TABLE_POWER;
		hash_table = memnew_arr(Element *, hash_table_size);
		for (uint32_t i = 0; i < hash_table_size; i++)
			hash_table[i] = NULL;

		hash_table_power = MIN_HASH_TABLE_POWER;
		elements = 0;
	}

	// Used to update the hash table when elements are added or removed
	void _update_hash_table() {
		uint8_t new_hash_table_power = hash_table_power;
		uint32_t hash_table_size = (uint32_t)1 << hash_table_power;

		if (elements > (hash_table_size * RELATIONSHIP)) {
			new_hash_table_power++;

		} else if ((hash_table_power > MIN_HASH_TABLE_POWER) && (elements < ((hash_table_size >> 1) * RELATIONSHIP))) {
			new_hash_table_power--;

		} else {
			return; // Hash table doesn't need updating (i.e. resizing)
		}

		uint32_t new_hash_table_size = (uint32_t)1 << new_hash_table_power;
		Element **new_hash_table = memnew_arr(Element *, new_hash_table_size);
		if (!new_hash_table) {
			ERR_PRINT("Out of Memory");
			return;
		}

		for (uint32_t i = 0; i < new_hash_table_size; i++) {
			new_hash_table[i] = NULL;
		}

		for (uint32_t i = 0; i < hash_table_size; i++) {
			while (hash_table[i]) {
				Element *se = hash_table[i];
				hash_table[i] = se->next;

				uint32_t new_index = se->hash & (new_hash_table_size - 1);
				se->next = new_hash_table[new_index];
				new_hash_table[new_index] = se;
			}
		}

		memdelete_arr(hash_table);
		hash_table = new_hash_table;
		hash_table_power = new_hash_table_power;
	}

	_FORCE_INLINE_ const Element *_get_element(const TKey &p_key) const {
		uint32_t hash = Hasher::hash(p_key);
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		Element *e = hash_table[index];
		while (e) {
			// Checking hash first avoids comparing key, which may take longer
			if (e->hash == hash && Comparator::compare(e->pair.key, p_key))
				return e; // Pair found

			e = e->next;
		}

		return NULL; // Pair not found
	}

	Element *_find_or_create_element(const TKey &p_key) {
		Element *e = NULL;
		if (!hash_table)
			_make_hash_table();
		else
			e = const_cast<Element *>(_get_element(p_key));

		if (!e) { // Couldn't find element, so create a new element
			e = memnew(Element);
			CRASH_COND(!e); // Out of memory

			e->hash = Hasher::hash(p_key);
			e->pair.key = p_key;

			uint32_t index = e->hash & ((1 << hash_table_power) - 1);
			e->next = hash_table[index];

			hash_table[index] = e;
			elements++;
			_update_hash_table();
		}

		return e;
	}

	void _copy_from(const HashMap &p_t) {
		if (&p_t == this)
			return; // Already initialised with p_t

		clear();

		if (!p_t.elements)
			return; // Hash table is empty

		uint32_t hash_table_size = (uint32_t)1 << p_t.hash_table_power;
		hash_table = memnew_arr(Element *, hash_table_size);
		hash_table_power = p_t.hash_table_power;
		elements = p_t.elements;

		for (uint32_t i = 0; i < hash_table_size; i++) {
			hash_table[i] = NULL;
			const Element *e = p_t.hash_table[i];

			while (e) {
				// Copy element
				Element *le = memnew(Element);
				*le = *e;

				// Add to list
				le->next = hash_table[i];
				hash_table[i] = le;

				e = e->next;
			}
		}
	}

public:
	Element *set(const TKey &p_key, const TData &p_data) {
		return set(Pair(p_key, p_data));
	}

	Element *set(const Pair &p_pair) {
		Element *e = _find_or_create_element(p_pair.key);
		e->pair.data = p_pair.data;
		return e;
	}

	bool has(const TKey &p_key) const {
		return getptr(p_key) != NULL;
	}

	// WARNING: 'get' doesn't check errors,
	// use either getptr directly and check NULL, or check first with has(key)
	const TData &get(const TKey &p_key) const {
		const TData *res = getptr(p_key);
		ERR_FAIL_COND_V(!res, *res);
		return *res;
	}

	TData &get(const TKey &p_key) {
		TData *res = getptr(p_key);
		ERR_FAIL_COND_V(!res, *res);
		return *res;
	}

	// Returns a pointer to the data or NULL if not found
	_FORCE_INLINE_ TData *getptr(const TKey &p_key) {
		if (!elements)
			return NULL;

		Element *e = const_cast<Element *>(_get_element(p_key));
		if (e)
			return &e->pair.data;

		return NULL;
	}

	_FORCE_INLINE_ const TData *getptr(const TKey &p_key) const {
		if (!elements)
			return NULL;

		const Element *e = _get_element(p_key);
		if (e)
			return &e->pair.data;

		return NULL;
	}

	// This version will take a custom hash and a custom key, that should support operator==()
	template <class C>
	_FORCE_INLINE_ TData *custom_getptr(C p_custom_key, uint32_t p_custom_hash) {
		if (!elements)
			return NULL;

		uint32_t hash = p_custom_hash;
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		Element *e = hash_table[index];
		while (e) {
			// Checking hash first avoids comparing key, which may take longer
			if (e->hash == hash && Comparator::compare(e->pair.key, p_custom_key))
				return &e->pair.data; // Pair found

			e = e->next;
		}

		return NULL; // Pair not found
	}

	template <class C>
	_FORCE_INLINE_ const TData *custom_getptr(C p_custom_key, uint32_t p_custom_hash) const {
		if (!elements)
			return NULL;

		uint32_t hash = p_custom_hash;
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		const Element *e = hash_table[index];
		while (e) {
			// Checking hash first avoids comparing key, which may take longer
			if (e->hash == hash && Comparator::compare(e->pair.key, p_custom_key))
				return &e->pair.data; // Pair found

			e = e->next;
		}

		return NULL; // Pair not found
	}

	// Erase an item, return true if erasing was successful
	bool erase(const TKey &p_key) {
		if (!elements)
			return false;

		uint32_t hash = Hasher::hash(p_key);
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		Element *e = hash_table[index];
		Element *p = NULL;
		while (e) {
			// Checking hash first avoids comparing key, which may take longer
			if (e->hash == hash && Comparator::compare(e->pair.key, p_key)) {
				if (p) {
					p->next = e->next;
				} else { // Element is first at hash index
					hash_table[index] = e->next;
				}

				memdelete(e);
				elements--;

				if (elements == 0)
					clear();
				else
					_update_hash_table();

				return true;
			}

			p = e;
			e = e->next;
		}

		return false;
	}

	inline const TData &operator[](const TKey &p_key) const {
		return get(p_key);
	}

	inline TData &operator[](const TKey &p_key) {
		Element *e = _find_or_create_element(p_key);
		return e->pair.data;
	}

	// Get the next key to p_key, and the first key if p_key is null.
	// Returns a pointer to the next key if found, NULL otherwise.
	// Use next if memory space is limited else consider using:
	// get_key_list or get_key_value_ptr_array for performance
	const TKey *next(const TKey *p_key) const {
		if (!elements)
			return NULL;

		if (!p_key) { // Return the first key of the hash table
			uint32_t hash_table_size = (uint32_t)(1 << hash_table_power);
			for (uint32_t i = 0; i < hash_table_size; i++) {
				if (hash_table[i])
					return &hash_table[i]->pair.key;
			}

		} else { // Return the next key after p_key
			const Element *e = _get_element(*p_key);
			ERR_FAIL_COND_V(!e, NULL); // Invalid key supplied

			if (e->next) {
				// If there is a "next" in the list, return that
				return &e->next->pair.key;
			} else {
				// Element is at end of list at current index, return element at next non-NULL index
				uint32_t hash_table_size = (uint32_t)(1 << hash_table_power);
				uint32_t index = e->hash & (hash_table_size - 1);
				index++;
				for (uint32_t i = index; i < hash_table_size; i++) {
					if (hash_table[i])
						return &hash_table[i]->pair.key;
				}
			}
		}

		return NULL; // Key is last, so next element doesn't exist
	}

	inline unsigned int size() const {
		return elements;
	}

	inline bool empty() const {
		return elements == 0;
	}

	void clear() {
		if (elements) {
			uint32_t hash_table_size = (uint32_t)(1 << hash_table_power);
			for (uint32_t i = 0; i < hash_table_size; i++) {
				while (hash_table[i]) {
					Element *e = hash_table[i];
					hash_table[i] = e->next;
					memdelete(e);
				}
			}
			elements = 0;
		}

		if (hash_table) {
			memdelete_arr(hash_table);
			hash_table = NULL;
			hash_table_power = 0;
		}
	}

	void operator=(const HashMap &p_table) {
		_copy_from(p_table);
	}

	HashMap() {
		hash_table = NULL;
		hash_table_power = 0;
		elements = 0;
	}

	// Get a pointer to a pair pointer array of size elements
	const Pair **get_key_value_ptr_array() const {
		if (!elements)
			return NULL;

		const Pair **ptr_array = memnew_arr(Pair *, elements);
		uint32_t index = 0;
		uint32_t hash_table_size = (uint32_t)(1 << hash_table_power);
		for (uint32_t i = 0; i < hash_table_size; i++) {
			Element *e = hash_table[i];
			while (e) {
				*ptr_array[index] = &(e->pair);
				index++;
				e = e->next;
			}
		}

		return ptr_array;
	}

	void get_key_list(List<TKey> *p_keys) const {
		if (!elements)
			return;

		uint32_t hash_table_size = (uint32_t)(1 << hash_table_power);
		for (uint32_t i = 0; i < hash_table_size; i++) {
			Element *e = hash_table[i];
			while (e) {
				p_keys->push_back(e->pair.key);
				e = e->next;
			}
		}
	}

	HashMap(const HashMap &p_table) {
		hash_table = NULL;
		hash_table_power = 0;
		elements = 0;

		_copy_from(p_table);
	}

	~HashMap() {
		clear();
	}
};

#endif // HASH_MAP_H
