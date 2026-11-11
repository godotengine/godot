/**************************************************************************/
/*  disjoint_set.h                                                        */
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

#include "core/templates/hash_map.h"
#include "core/templates/vector.h"

/* This DisjointSet class uses Find with path compression and Union by rank */
template <typename T, typename H = HashMapHasherDefault, typename C = HashMapComparatorDefault<T>, typename AL = DefaultAllocator>
class DisjointSet {
	struct Element {
		T object;
		Element *parent = nullptr;
		int rank = 0;
	};

	typedef HashMap<T, Element *, H, C> MapT;

	MapT elements;

	Element *get_parent(Element *r_element);

	_FORCE_INLINE_ Element *insert_or_get(T p_object);

public:
	~DisjointSet();

	_FORCE_INLINE_ void insert(T p_object) { (void)insert_or_get(p_object); }

	void create_union(T p_left, T p_right);

	void get_representatives(Vector<T> &r_out_roots);

	void get_members(Vector<T> &r_out_members, T p_representative);
};

/* FUNCTIONS */

template <typename T, typename H, typename C, typename AL>
DisjointSet<T, H, C, AL>::~DisjointSet() {
	for (KeyValue<T, Element *> &E : elements) {
		memdelete_allocator<Element, AL>(E.value);
	}
}

template <typename T, typename H, typename C, typename AL>
typename DisjointSet<T, H, C, AL>::Element *DisjointSet<T, H, C, AL>::get_parent(Element *r_element) {
	if (r_element->parent != r_element) {
		r_element->parent = get_parent(r_element->parent);
	}

	return r_element->parent;
}

template <typename T, typename H, typename C, typename AL>
typename DisjointSet<T, H, C, AL>::Element *DisjointSet<T, H, C, AL>::insert_or_get(T p_object) {
	typename MapT::Iterator itr = elements.find(p_object);
	if (itr != nullptr) {
		return itr->value;
	}

	Element *new_element = memnew_allocator(Element, AL);
	new_element->object = p_object;
	new_element->parent = new_element;
	elements.insert(p_object, new_element);

	return new_element;
}

template <typename T, typename H, typename C, typename AL>
void DisjointSet<T, H, C, AL>::create_union(T p_left, T p_right) {
	Element *x = insert_or_get(p_left);
	Element *y = insert_or_get(p_right);

	Element *x_root = get_parent(x);
	Element *y_root = get_parent(y);

	// Already in the same set
	if (x_root == y_root) {
		return;
	}

	// Not in the same set, merge
	if (x_root->rank < y_root->rank) {
		SWAP(x_root, y_root);
	}

	// Merge y_root into x_root
	y_root->parent = x_root;
	if (x_root->rank == y_root->rank) {
		++x_root->rank;
	}
}

template <typename T, typename H, typename C, typename AL>
void DisjointSet<T, H, C, AL>::get_representatives(Vector<T> &r_out_representatives) {
	for (KeyValue<T, Element *> &E : elements) {
		Element *element = E.value;
		if (element->parent == element) {
			r_out_representatives.push_back(element->object);
		}
	}
}

template <typename T, typename H, typename C, typename AL>
void DisjointSet<T, H, C, AL>::get_members(Vector<T> &r_out_members, T p_representative) {
	typename MapT::Iterator rep_itr = elements.find(p_representative);
	ERR_FAIL_NULL(rep_itr);

	Element *rep_element = rep_itr->value;
	ERR_FAIL_COND(rep_element->parent != rep_element);

	for (KeyValue<T, Element *> &E : elements) {
		Element *parent = get_parent(E.value);
		if (parent == rep_element) {
			r_out_members.push_back(E.key);
		}
	}
}
