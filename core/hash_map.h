/*************************************************************************/
/*  hash_map.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
 *
 */

template <class TKey, class TData, class Hasher = HashMapHasherDefault, class Comparator = HashMapComparatorDefault<TKey>, uint8_t MIN_HASH_TABLE_POWER = 3, uint8_t RELATIONSHIP = 8>
class HashMap {
public:
	struct Pair {
		TKey key;
		TData data;

		Pair(const TKey &p_key) :
				key(p_key),
				data() {}
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
		Element() { next = nullptr; }
		Pair pair;

	public:
		const TKey &key() const {
			return pair.key;
		}

		TData &value() {
			return pair.data;
		}

		const TData &value() const {
			return pair.value();
		}

		Element(const TKey &p_key) :
				pair(p_key) {}
		Element(const Element &p_other) :
				hash(p_other.hash),
				pair(p_other.pair.key, p_other.pair.data) {}
	};

private:
	Element **hash_table;
	uint8_t hash_table_power;
	uint32_t elements;

	void make_hash_table() {
		ERR_FAIL_COND(hash_table);

		hash_table = memnew_arr(Element *, (1 << MIN_HASH_TABLE_POWER));

		hash_table_power = MIN_HASH_TABLE_POWER;
		elements = 0;
		for (int i = 0; i < (1 << MIN_HASH_TABLE_POWER); i++) {
			hash_table[i] = nullptr;
		}
	}

	void erase_hash_table() {
		ERR_FAIL_COND_MSG(elements, "Cannot erase hash table if there are still elements inside.");

		memdelete_arr(hash_table);
		hash_table = nullptr;
		hash_table_power = 0;
		elements = 0;
	}

	void check_hash_table() {
		int new_hash_table_power = -1;

		if ((int)elements > ((1 << hash_table_power) * RELATIONSHIP)) {
			/* rehash up */
			new_hash_table_power = hash_table_power + 1;

			while ((int)elements > ((1 << new_hash_table_power) * RELATIONSHIP)) {
				new_hash_table_power++;
			}

		} else if ((hash_table_power > (int)MIN_HASH_TABLE_POWER) && ((int)elements < ((1 << (hash_table_power - 1)) * RELATIONSHIP))) {
			/* rehash down */
			new_hash_table_power = hash_table_power - 1;

			while ((int)elements < ((1 << (new_hash_table_power - 1)) * RELATIONSHIP)) {
				new_hash_table_power--;
			}

			if (new_hash_table_power < (int)MIN_HASH_TABLE_POWER) {
				new_hash_table_power = MIN_HASH_TABLE_POWER;
			}
		}

		if (new_hash_table_power == -1) {
			return;
		}

		Element **new_hash_table = memnew_arr(Element *, ((uint64_t)1 << new_hash_table_power));
		ERR_FAIL_COND_MSG(!new_hash_table, "Out of memory.");

		for (int i = 0; i < (1 << new_hash_table_power); i++) {
			new_hash_table[i] = nullptr;
		}

		if (hash_table) {
			for (int i = 0; i < (1 << hash_table_power); i++) {
				while (hash_table[i]) {
					Element *se = hash_table[i];
					hash_table[i] = se->next;
					int new_pos = se->hash & ((1 << new_hash_table_power) - 1);
					se->next = new_hash_table[new_pos];
					new_hash_table[new_pos] = se;
				}
			}

			memdelete_arr(hash_table);
		}
		hash_table = new_hash_table;
		hash_table_power = new_hash_table_power;
	}

	/* I want to have only one function.. */
	_FORCE_INLINE_ const Element *get_element(const TKey &p_key) const {
		uint32_t hash = Hasher::hash(p_key);
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		Element *e = hash_table[index];

		while (e) {
			/* checking hash first avoids comparing key, which may take longer */
			if (e->hash == hash && Comparator::compare(e->pair.key, p_key)) {
				/* the pair exists in this hashtable, so just update data */
				return e;
			}

			e = e->next;
		}

		return nullptr;
	}

	Element *create_element(const TKey &p_key) {
		/* if element doesn't exist, create it */
		Element *e = memnew(Element(p_key));
		ERR_FAIL_COND_V_MSG(!e, nullptr, "Out of memory.");
		uint32_t hash = Hasher::hash(p_key);
		uint32_t index = hash & ((1 << hash_table_power) - 1);
		e->next = hash_table[index];
		e->hash = hash;

		hash_table[index] = e;
		elements++;

		return e;
	}

	void copy_from(const HashMap &p_t) {
		if (&p_t == this) {
			return; /* much less bother with that */
		}

		clear();

		if (!p_t.hash_table || p_t.hash_table_power == 0) {
			return; /* not copying from empty table */
		}

		hash_table = memnew_arr(Element *, (uint64_t)1 << p_t.hash_table_power);
		hash_table_power = p_t.hash_table_power;
		elements = p_t.elements;

		for (int i = 0; i < (1 << p_t.hash_table_power); i++) {
			hash_table[i] = nullptr;

			const Element *e = p_t.hash_table[i];

			while (e) {
				Element *le = memnew(Element(*e)); /* local element */

				/* add to list and reassign pointers */
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
		Element *e = nullptr;
		if (!hash_table) {
			make_hash_table(); // if no table, make one
		} else {
			e = const_cast<Element *>(get_element(p_pair.key));
		}

		/* if we made it up to here, the pair doesn't exist, create and assign */

		if (!e) {
			e = create_element(p_pair.key);
			if (!e) {
				return nullptr;
			}
			check_hash_table(); // perform mantenience routine
		}

		e->pair.data = p_pair.data;
		return e;
	}

	bool has(const TKey &p_key) const {
		return getptr(p_key) != nullptr;
	}

	/**
	 * Get a key from data, return a const reference.
	 * WARNING: this doesn't check errors, use either getptr and check NULL, or check
	 * first with has(key)
	 */

	const TData &get(const TKey &p_key) const {
		const TData *res = getptr(p_key);
		CRASH_COND_MSG(!res, "Map key not found.");
		return *res;
	}

	TData &get(const TKey &p_key) {
		TData *res = getptr(p_key);
		CRASH_COND_MSG(!res, "Map key not found.");
		return *res;
	}

	/**
	 * Same as get, except it can return NULL when item was not found.
	 * This is mainly used for speed purposes.
	 */

	_FORCE_INLINE_ TData *getptr(const TKey &p_key) {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		Element *e = const_cast<Element *>(get_element(p_key));

		if (e) {
			return &e->pair.data;
		}

		return nullptr;
	}

	_FORCE_INLINE_ const TData *getptr(const TKey &p_key) const {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		const Element *e = const_cast<Element *>(get_element(p_key));

		if (e) {
			return &e->pair.data;
		}

		return nullptr;
	}

	/**
	 * Same as get, except it can return NULL when item was not found.
	 * This version is custom, will take a hash and a custom key (that should support operator==()
	 */

	template <class C>
	_FORCE_INLINE_ TData *custom_getptr(C p_custom_key, uint32_t p_custom_hash) {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		uint32_t hash = p_custom_hash;
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		Element *e = hash_table[index];

		while (e) {
			/* checking hash first avoids comparing key, which may take longer */
			if (e->hash == hash && Comparator::compare(e->pair.key, p_custom_key)) {
				/* the pair exists in this hashtable, so just update data */
				return &e->pair.data;
			}

			e = e->next;
		}

		return nullptr;
	}

	template <class C>
	_FORCE_INLINE_ const TData *custom_getptr(C p_custom_key, uint32_t p_custom_hash) const {
		if (unlikely(!hash_table)) {
			return NULL;
		}

		uint32_t hash = p_custom_hash;
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		const Element *e = hash_table[index];

		while (e) {
			/* checking hash first avoids comparing key, which may take longer */
			if (e->hash == hash && Comparator::compare(e->pair.key, p_custom_key)) {
				/* the pair exists in this hashtable, so just update data */
				return &e->pair.data;
			}

			e = e->next;
		}

		return NULL;
	}

	/**
	 * Erase an item, return true if erasing was successful
	 */

	bool erase(const TKey &p_key) {
		if (unlikely(!hash_table)) {
			return false;
		}

		uint32_t hash = Hasher::hash(p_key);
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		Element *e = hash_table[index];
		Element *p = nullptr;
		while (e) {
			/* checking hash first avoids comparing key, which may take longer */
			if (e->hash == hash && Comparator::compare(e->pair.key, p_key)) {
				if (p) {
					p->next = e->next;
				} else {
					//begin of list
					hash_table[index] = e->next;
				}

				memdelete(e);
				elements--;

				if (elements == 0) {
					erase_hash_table();
				} else {
					check_hash_table();
				}
				return true;
			}

			p = e;
			e = e->next;
		}

		return false;
	}

	inline const TData &operator[](const TKey &p_key) const { //constref

		return get(p_key);
	}
	inline TData &operator[](const TKey &p_key) { //assignment

		Element *e = nullptr;
		if (!hash_table) {
			make_hash_table(); // if no table, make one
		} else {
			e = const_cast<Element *>(get_element(p_key));
		}

		/* if we made it up to here, the pair doesn't exist, create */
		if (!e) {
			e = create_element(p_key);
			CRASH_COND(!e);
			check_hash_table(); // perform mantenience routine
		}

		return e->pair.data;
	}

	/**
	 * Get the next key to p_key, and the first key if p_key is null.
	 * Returns a pointer to the next key if found, NULL otherwise.
	 * Adding/Removing elements while iterating will, of course, have unexpected results, don't do it.
	 *
	 * Example:
	 *
	 * 	const TKey *k=NULL;
	 *
	 * 	while( (k=table.next(k)) ) {
	 *
	 * 		print( *k );
	 * 	}
	 *
	 */
	const TKey *next(const TKey *p_key) const {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		if (!p_key) { /* get the first key */

			for (int i = 0; i < (1 << hash_table_power); i++) {
				if (hash_table[i]) {
					return &hash_table[i]->pair.key;
				}
			}

		} else { /* get the next key */

			const Element *e = get_element(*p_key);
			ERR_FAIL_COND_V_MSG(!e, nullptr, "Invalid key supplied.");
			if (e->next) {
				/* if there is a "next" in the list, return that */
				return &e->next->pair.key;
			} else {
				/* go to next elements */
				uint32_t index = e->hash & ((1 << hash_table_power) - 1);
				index++;
				for (int i = index; i < (1 << hash_table_power); i++) {
					if (hash_table[i]) {
						return &hash_table[i]->pair.key;
					}
				}
			}

			/* nothing found, was at end */
		}

		return nullptr; /* nothing found */
	}

	inline unsigned int size() const {
		return elements;
	}

	inline bool empty() const {
		return elements == 0;
	}

	void clear() {
		/* clean up */
		if (hash_table) {
			for (int i = 0; i < (1 << hash_table_power); i++) {
				while (hash_table[i]) {
					Element *e = hash_table[i];
					hash_table[i] = e->next;
					memdelete(e);
				}
			}

			memdelete_arr(hash_table);
		}

		hash_table = nullptr;
		hash_table_power = 0;
		elements = 0;
	}

	void operator=(const HashMap &p_table) {
		copy_from(p_table);
	}

	HashMap() {
		hash_table = nullptr;
		elements = 0;
		hash_table_power = 0;
	}

	void get_key_value_ptr_array(const Pair **p_pairs) const {
		if (unlikely(!hash_table)) {
			return;
		}
		for (int i = 0; i < (1 << hash_table_power); i++) {
			Element *e = hash_table[i];
			while (e) {
				*p_pairs = &e->pair;
				p_pairs++;
				e = e->next;
			}
		}
	}

	void get_key_list(List<TKey> *p_keys) const {
		if (unlikely(!hash_table)) {
			return;
		}
		for (int i = 0; i < (1 << hash_table_power); i++) {
			Element *e = hash_table[i];
			while (e) {
				p_keys->push_back(e->pair.key);
				e = e->next;
			}
		}
	}

	HashMap(const HashMap &p_table) {
		hash_table = nullptr;
		elements = 0;
		hash_table_power = 0;

		copy_from(p_table);
	}

	~HashMap() {
		clear();
	}
};

#endif
