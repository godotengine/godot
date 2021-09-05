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

#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/os/memory.h"
#include "core/string/ustring.h"
#include "core/templates/hashfuncs.h"
#include "core/templates/list.h"
#include "core/templates/pair.h"

/**
 * @class HashMap
 * @author Juan Linietsky <reduzio@gmail.com>
 *
 * Implementation of a standard Hashing HashMap, for quick lookups of Data associated with a Key.
 * The implementation provides hashers for the default types, if you need a special kind of hasher, provide
 * your own.
 * @param K Key, search is based on it, needs to be hasheable. It is unique in this container.
 * @param V Value, data associated with the key
 * @param Hasher Hasher object, needs to provide a valid static hash function for K
 * @param Comparator comparator object, needs to be able to safely compare two K values. It needs to ensure that x == x for any items inserted in the map. Bear in mind that nan != nan when implementing an equality check.
 * @param MIN_HASH_TABLE_POWER Miminum size of the hash table, as a power of two. You rarely need to change this parameter.
 * @param RELATIONSHIP Relationship at which the hash table is resized. if amount of elements is RELATIONSHIP
 * times bigger than the hash table, table is resized to solve this condition. if RELATIONSHIP is zero, table is always MIN_HASH_TABLE_POWER.
 *
 */

template <class K, class V, class Hasher = HashMapHasherDefault, class Comparator = HashMapComparatorDefault<K>, uint8_t MIN_HASH_TABLE_POWER = 3, uint8_t RELATIONSHIP = 8>
class HashMap {
public:
	struct Element {
	private:
		friend class HashMap;

		uint32_t hash = 0;
		Element *_next = nullptr;
		KeyValue<K, V> _data;

	public:
		KeyValue<K, V> &key_value() { return _data; }
		const KeyValue<K, V> &key_value() const { return _data; }

		const Element *next() const {
			return _next;
		}
		Element *next() {
			return _next;
		}
		const K &key() const {
			return _data.key;
		}
		V &value() {
			return _data.value;
		}
		const V &value() const {
			return _data.value;
		}
		V &get() {
			return _data.value;
		}
		const V &get() const {
			return _data.value;
		}
		Element(const KeyValue<K, V> &p_data) :
				_data(p_data) {}
		Element(const Element &p_other) :
				hash(p_other.hash),
				_data(p_other._data) {}
	};

	typedef KeyValue<K, V> ValueType;

private:
	Element **hash_table = nullptr;
	uint8_t hash_table_power = 0;
	uint32_t elements = 0;

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
					hash_table[i] = se->_next;
					int new_pos = se->hash & ((1 << new_hash_table_power) - 1);
					se->_next = new_hash_table[new_pos];
					new_hash_table[new_pos] = se;
				}
			}

			memdelete_arr(hash_table);
		}
		hash_table = new_hash_table;
		hash_table_power = new_hash_table_power;
	}

	/* I want to have only one function.. */
	_FORCE_INLINE_ const Element *get_element(const K &p_key) const {
		uint32_t hash = Hasher::hash(p_key);
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		Element *e = hash_table[index];

		while (e) {
			/* checking hash first avoids comparing key, which may take longer */
			if (e->hash == hash && Comparator::compare(e->_data.key, p_key)) {
				/* the KeyValue exists in this hashtable, so just update data */
				return e;
			}

			e = e->_next;
		}

		return nullptr;
	}

	Element *create_element(const K &p_key) {
		/* if element doesn't exist, create it */
		Element *e = memnew(Element(KeyValue<K, V>(p_key, V())));
		ERR_FAIL_COND_V_MSG(!e, nullptr, "Out of memory.");
		uint32_t hash = Hasher::hash(p_key);
		uint32_t index = hash & ((1 << hash_table_power) - 1);
		e->_next = hash_table[index];
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
				Element *le = memnew(Element(KeyValue(e->_data.key, e->_data.value))); /* local element */

				/* add to list and reassign pointers */
				le->_next = hash_table[i];
				hash_table[i] = le;

				e = e->_next;
			}
		}
	}

public:
	Element *set(const K &p_key, const V &p_data) {
		return set(KeyValue(p_key, p_data));
	}

	Element *set(const KeyValue<K, V> &p_key_value) {
		Element *e = nullptr;
		if (!hash_table) {
			make_hash_table(); // if no table, make one
		} else {
			e = const_cast<Element *>(get_element(p_key_value.key));
		}

		/* if we made it up to here, the KeyValue doesn't exist, create and assign */

		if (!e) {
			e = create_element(p_key_value.key);
			if (!e) {
				return nullptr;
			}
			check_hash_table(); // perform mantenience routine
		}

		e->_data.value = p_key_value.value;
		return e;
	}

	bool has(const K &p_key) const {
		return getptr(p_key) != nullptr;
	}

	/**
	 * Get a key from data, return a const reference.
	 * WARNING: this doesn't check errors, use either getptr and check nullptr, or check
	 * first with has(key)
	 */

	const V &get(const K &p_key) const {
		const V *res = getptr(p_key);
		CRASH_COND_MSG(!res, "Map key not found.");
		return *res;
	}

	V &get(const K &p_key) {
		V *res = getptr(p_key);
		CRASH_COND_MSG(!res, "Map key not found.");
		return *res;
	}

	/**
	 * Same as get, except it can return nullptr when item was not found.
	 * This is mainly used for speed purposes.
	 */

	_FORCE_INLINE_ V *getptr(const K &p_key) {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		Element *e = const_cast<Element *>(get_element(p_key));

		if (e) {
			return &e->_data.value;
		}

		return nullptr;
	}

	_FORCE_INLINE_ const V *getptr(const K &p_key) const {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		const Element *e = const_cast<Element *>(get_element(p_key));

		if (e) {
			return &e->_data.value;
		}

		return nullptr;
	}

	/**
	 * Same as get, except it can return nullptr when item was not found.
	 * This version is custom, will take a hash and a custom key (that should support operator==()
	 */

	template <class C>
	_FORCE_INLINE_ V *custom_getptr(C p_custom_key, uint32_t p_custom_hash) {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		uint32_t hash = p_custom_hash;
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		Element *e = hash_table[index];

		while (e) {
			/* checking hash first avoids comparing key, which may take longer */
			if (e->hash == hash && Comparator::compare(e->_data.key, p_custom_key)) {
				/* the KeyValue exists in this hashtable, so just update data */
				return &e->_data.value;
			}

			e = e->_next;
		}

		return nullptr;
	}

	template <class C>
	_FORCE_INLINE_ const V *custom_getptr(C p_custom_key, uint32_t p_custom_hash) const {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		uint32_t hash = p_custom_hash;
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		const Element *e = hash_table[index];

		while (e) {
			/* checking hash first avoids comparing key, which may take longer */
			if (e->hash == hash && Comparator::compare(e->_data.key, p_custom_key)) {
				/* the KeyValue exists in this hashtable, so just update data */
				return &e->_data.value;
			}

			e = e->_next;
		}

		return nullptr;
	}

	/**
	 * Erase an item, return true if erasing was successful
	 */

	bool erase(const K &p_key) {
		if (unlikely(!hash_table)) {
			return false;
		}

		uint32_t hash = Hasher::hash(p_key);
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		Element *e = hash_table[index];
		Element *p = nullptr;
		while (e) {
			/* checking hash first avoids comparing key, which may take longer */
			if (e->hash == hash && Comparator::compare(e->_data.key, p_key)) {
				if (p) {
					p->_next = e->_next;
				} else {
					//begin of list
					hash_table[index] = e->_next;
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
			e = e->_next;
		}

		return false;
	}

	inline const V &operator[](const K &p_key) const { //constref

		return get(p_key);
	}
	inline V &operator[](const K &p_key) { //assignment

		Element *e = nullptr;
		if (!hash_table) {
			make_hash_table(); // if no table, make one
		} else {
			e = const_cast<Element *>(get_element(p_key));
		}

		/* if we made it up to here, the KeyValue doesn't exist, create */
		if (!e) {
			e = create_element(p_key);
			CRASH_COND(!e);
			check_hash_table(); // perform mantenience routine
		}

		return e->_data.value;
	}

	/**
	 * Get the next key to p_key, and the first key if p_key is null.
	 * Returns a pointer to the next key if found, nullptr otherwise.
	 * Adding/Removing elements while iterating will, of course, have unexpected results, don't do it.
	 *
	 * Example:
	 *
	 * 	const K *k=nullptr;
	 *
	 * 	while( (k=table.next(k)) ) {
	 *
	 * 		print( *k );
	 * 	}
	 *
	 */
	const K *next(const K *p_key) const {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		if (!p_key) { /* get the first key */

			for (int i = 0; i < (1 << hash_table_power); i++) {
				if (hash_table[i]) {
					return &hash_table[i]->_data.key;
				}
			}

		} else { /* get the next key */

			const Element *e = get_element(*p_key);
			ERR_FAIL_COND_V_MSG(!e, nullptr, "Invalid key supplied.");
			if (e->_next) {
				/* if there is a "next" in the list, return that */
				return &e->_next->_data.key;
			} else {
				/* go to next elements */
				uint32_t index = e->hash & ((1 << hash_table_power) - 1);
				index++;
				for (int i = index; i < (1 << hash_table_power); i++) {
					if (hash_table[i]) {
						return &hash_table[i]->_data.key;
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

	inline bool is_empty() const {
		return elements == 0;
	}

	void clear() {
		/* clean up */
		if (hash_table) {
			for (int i = 0; i < (1 << hash_table_power); i++) {
				while (hash_table[i]) {
					Element *e = hash_table[i];
					hash_table[i] = e->_next;
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

	void get_key_list(List<K> *r_keys) const {
		if (unlikely(!hash_table)) {
			return;
		}
		for (int i = 0; i < (1 << hash_table_power); i++) {
			Element *e = hash_table[i];
			while (e) {
				r_keys->push_back(e->_data.key);
				e = e->_next;
			}
		}
	}

	HashMap() {}

	HashMap(const HashMap &p_table) {
		copy_from(p_table);
	}

	~HashMap() {
		clear();
	}
};

#endif // HASH_MAP_H
