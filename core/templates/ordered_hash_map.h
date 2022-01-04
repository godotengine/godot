/*************************************************************************/
/*  ordered_hash_map.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef ORDERED_HASH_MAP_H
#define ORDERED_HASH_MAP_H

#include "core/templates/hash_map.h"
#include "core/templates/list.h"
#include "core/templates/pair.h"

/**
 * A hash map which allows to iterate elements in insertion order.
 * Insertion, lookup, deletion have O(1) complexity.
 * The API aims to be consistent with Map rather than HashMap, because the
 * former is more frequently used and is more coherent with the rest of the
 * codebase.
 * Deletion during iteration is safe and will preserve the order.
 */
template <class K, class V, class Hasher = HashMapHasherDefault, class Comparator = HashMapComparatorDefault<K>, uint8_t MIN_HASH_TABLE_POWER = 3, uint8_t RELATIONSHIP = 8>
class OrderedHashMap {
	typedef List<Pair<const K *, V>> InternalList;
	typedef HashMap<K, typename InternalList::Element *, Hasher, Comparator, MIN_HASH_TABLE_POWER, RELATIONSHIP> InternalMap;

	InternalList list;
	InternalMap map;

public:
	class Element {
		friend class OrderedHashMap<K, V, Hasher, Comparator, MIN_HASH_TABLE_POWER, RELATIONSHIP>;

		typename InternalList::Element *list_element = nullptr;
		typename InternalList::Element *prev_element = nullptr;
		typename InternalList::Element *next_element = nullptr;

		Element(typename InternalList::Element *p_element) {
			list_element = p_element;

			if (list_element) {
				next_element = list_element->next();
				prev_element = list_element->prev();
			}
		}

	public:
		_FORCE_INLINE_ Element() {}

		Element next() const {
			return Element(next_element);
		}

		Element prev() const {
			return Element(prev_element);
		}

		Element(const Element &other) :
				list_element(other.list_element),
				prev_element(other.prev_element),
				next_element(other.next_element) {
		}

		void operator=(const Element &other) {
			list_element = other.list_element;
			next_element = other.next_element;
			prev_element = other.prev_element;
		}

		_FORCE_INLINE_ bool operator==(const Element &p_other) const {
			return this->list_element == p_other.list_element;
		}
		_FORCE_INLINE_ bool operator!=(const Element &p_other) const {
			return this->list_element != p_other.list_element;
		}

		operator bool() const {
			return (list_element != nullptr);
		}

		const K &key() const {
			CRASH_COND(!list_element);
			return *(list_element->get().first);
		}

		V &value() {
			CRASH_COND(!list_element);
			return list_element->get().second;
		}

		const V &value() const {
			CRASH_COND(!list_element);
			return list_element->get().second;
		}

		V &get() {
			CRASH_COND(!list_element);
			return list_element->get().second;
		}

		const V &get() const {
			CRASH_COND(!list_element);
			return list_element->get().second;
		}
	};

	class ConstElement {
		friend class OrderedHashMap<K, V, Hasher, Comparator, MIN_HASH_TABLE_POWER, RELATIONSHIP>;

		const typename InternalList::Element *list_element = nullptr;

		ConstElement(const typename InternalList::Element *p_element) :
				list_element(p_element) {
		}

	public:
		_FORCE_INLINE_ ConstElement() {}

		ConstElement(const ConstElement &other) :
				list_element(other.list_element) {
		}

		void operator=(const ConstElement &other) {
			list_element = other.list_element;
		}

		ConstElement next() const {
			return ConstElement(list_element ? list_element->next() : nullptr);
		}

		ConstElement prev() const {
			return ConstElement(list_element ? list_element->prev() : nullptr);
		}

		_FORCE_INLINE_ bool operator==(const ConstElement &p_other) const {
			return this->list_element == p_other.list_element;
		}
		_FORCE_INLINE_ bool operator!=(const ConstElement &p_other) const {
			return this->list_element != p_other.list_element;
		}

		operator bool() const {
			return (list_element != nullptr);
		}

		const K &key() const {
			CRASH_COND(!list_element);
			return *(list_element->get().first);
		}

		const V &value() const {
			CRASH_COND(!list_element);
			return list_element->get().second;
		}

		const V &get() const {
			CRASH_COND(!list_element);
			return list_element->get().second;
		}
	};

	ConstElement find(const K &p_key) const {
		typename InternalList::Element *const *list_element = map.getptr(p_key);
		if (list_element) {
			return ConstElement(*list_element);
		}
		return ConstElement(nullptr);
	}

	Element find(const K &p_key) {
		typename InternalList::Element **list_element = map.getptr(p_key);
		if (list_element) {
			return Element(*list_element);
		}
		return Element(nullptr);
	}

	Element insert(const K &p_key, const V &p_value) {
		typename InternalList::Element **list_element = map.getptr(p_key);
		if (list_element) {
			(*list_element)->get().second = p_value;
			return Element(*list_element);
		}
		// Incorrectly set the first value of the pair with a value that will
		// be invalid as soon as we leave this function...
		typename InternalList::Element *new_element = list.push_back(Pair<const K *, V>(&p_key, p_value));
		// ...this is needed here in case the hashmap recursively reference itself...
		typename InternalMap::Element *e = map.set(p_key, new_element);
		// ...now we can set the right value !
		new_element->get().first = &e->key();

		return Element(new_element);
	}

	void erase(Element &p_element) {
		map.erase(p_element.key());
		list.erase(p_element.list_element);
		p_element.list_element = nullptr;
	}

	bool erase(const K &p_key) {
		typename InternalList::Element **list_element = map.getptr(p_key);
		if (list_element) {
			list.erase(*list_element);
			map.erase(p_key);
			return true;
		}
		return false;
	}

	inline bool has(const K &p_key) const {
		return map.has(p_key);
	}

	const V &operator[](const K &p_key) const {
		ConstElement e = find(p_key);
		CRASH_COND(!e);
		return e.value();
	}

	V &operator[](const K &p_key) {
		Element e = find(p_key);
		if (!e) {
			// consistent with Map behaviour
			e = insert(p_key, V());
		}
		return e.value();
	}

	inline Element front() {
		return Element(list.front());
	}

	inline Element back() {
		return Element(list.back());
	}

	inline ConstElement front() const {
		return ConstElement(list.front());
	}

	inline ConstElement back() const {
		return ConstElement(list.back());
	}

	inline bool is_empty() const { return list.is_empty(); }
	inline int size() const { return list.size(); }

	const void *id() const {
		return list.id();
	}

	void clear() {
		map.clear();
		list.clear();
	}

private:
	void _copy_from(const OrderedHashMap &p_map) {
		for (ConstElement E = p_map.front(); E; E = E.next()) {
			insert(E.key(), E.value());
		}
	}

public:
	void operator=(const OrderedHashMap &p_map) {
		_copy_from(p_map);
	}

	OrderedHashMap(const OrderedHashMap &p_map) {
		_copy_from(p_map);
	}

	_FORCE_INLINE_ OrderedHashMap() {}
};

#endif // ORDERED_HASH_MAP_H
