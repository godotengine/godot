/**************************************************************************/
/*  set.cpp                                                               */
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

#include "set.h"

#include "core/object/class_db.h"
#include "core/variant/variant_internal.h"

struct SetPrivate {
	SafeRefCount refcount;
	Variant *read_only = nullptr; // If enabled, a pointer is used to a temporary value that is used to return read-only values.
	HashSet<Variant, VariantHasher, StringLikeVariantComparator> variant_set;
};

Set::Iterator &Set::Iterator::operator++() {
	HashSet<Variant, VariantHasher, StringLikeVariantComparator>::Iterator E = private_data->variant_set.find(*element_ptr);
	++E;
	element_ptr = &*E;
	return *this;
}

Set::Iterator Set::begin() const {
	return Iterator(&*_p->variant_set.begin(), _p);
}

Set::Iterator Set::end() const {
	return Iterator(&*_p->variant_set.end(), _p);
}

int Set::size() const {
	return _p->variant_set.size();
}

bool Set::is_empty() const {
	return !_p->variant_set.size();
}

bool Set::has(const Variant &p_value) const {
	return _p->variant_set.has(p_value);
}

bool Set::has_all(const Array &p_values) const {
	for (int i = 0; i < p_values.size(); i++) {
		if (!has(p_values[i])) {
			return false;
		}
	}
	return true;
}

bool Set::erase(const Variant &p_value) {
	ERR_FAIL_COND_V_MSG(_p->read_only, false, "Set is in read-only state.");
	return _p->variant_set.erase(p_value);
}

void Set::insert(const Variant &p_value) {
	ERR_FAIL_COND_MSG(_p->read_only, "Set is in read-only state.");
	_p->variant_set.insert(p_value);
}

bool Set::operator==(const Set &p_set) const {
	return recursive_equal(p_set, 0);
}

bool Set::operator!=(const Set &p_set) const {
	return !recursive_equal(p_set, 0);
}

bool Set::recursive_equal(const Set &p_set, int p_recursion_count) const {
	// Cheap checks
	if (_p == p_set._p) {
		return true;
	}
	if (_p->variant_set.size() != p_set._p->variant_set.size()) {
		return false;
	}

	// Heavy O(n) check
	if (p_recursion_count > MAX_RECURSION) {
		ERR_PRINT("Max recursion reached");
		return true;
	}
	p_recursion_count++;
	for (const Variant &this_E : _p->variant_set) {
		if (!p_set._p->variant_set.has(this_E)) {
			return false;
		}
	}
	return true;
}

void Set::_ref(const Set &p_from) const {
	//make a copy first (thread safe)
	if (!p_from._p->refcount.ref()) {
		return; // couldn't copy
	}

	//if this is the same, unreference the other one
	if (p_from._p == _p) {
		_p->refcount.unref();
		return;
	}
	if (_p) {
		_unref();
	}
	_p = p_from._p;
}

void Set::clear() {
	ERR_FAIL_COND_MSG(_p->read_only, "Set is in read-only state.");
	_p->variant_set.clear();
}

bool Set::is_overlapping(const Set &p_set) const {
	for (const Variant &E : p_set._p->variant_set) {
		if (has(E)) {
			return true;
		}
	}
	return false;
}

void Set::merge(const Set &p_set) {
	ERR_FAIL_COND_MSG(_p->read_only, "Set is in read-only state.");
	for (const Variant &E : p_set._p->variant_set) {
		if (!has(E)) {
			insert(E);
		}
	}
}

Set Set::merged(const Set &p_set) const {
	Set ret = duplicate();
	ret.merge(p_set);
	return ret;
}

void Set::difference(const Set &p_set) {
	ERR_FAIL_COND_MSG(_p->read_only, "Set is in read-only state.");
	for (const Variant &E : p_set._p->variant_set) {
		HashSet<Variant, VariantHasher, StringLikeVariantComparator>::Iterator it = _p->variant_set.find(E);
		if (it != _p->variant_set.end()) {
			_p->variant_set.remove(it);
		}
	}
}

Set Set::differentiated(const Set &p_set) const {
	Set ret = duplicate();
	ret.difference(p_set);
	return ret;
}

void Set::intersect(const Set &p_set) {
	ERR_FAIL_COND_MSG(_p->read_only, "Set is in read-only state.");
	/*
	Vector<const Variant *> to_remove;
	for (const Variant &E : _p->variant_set) {
		if (!p_set.has(E)) {
			to_remove.append(&E);
		}
	}
	for (const Variant *E : to_remove) {
		remove(*E);
	}
	*/
	*this = differentiated(differentiated(p_set)); // Intersection doesn't work. For now, use this.
}

Set Set::intersected(const Set &p_set) const {
	Set ret = duplicate();
	ret.intersect(p_set);
	return ret;
}

void Set::symmetric_difference(const Set &p_set) {
	ERR_FAIL_COND_MSG(_p->read_only, "Set is in read-only state.");
	for (const Variant &E : p_set._p->variant_set) {
		HashSet<Variant, VariantHasher, StringLikeVariantComparator>::Iterator it = _p->variant_set.find(E);
		if (it != _p->variant_set.end()) {
			_p->variant_set.remove(it);
		} else {
			_p->variant_set.insert(E);
		}
	}
}

Set Set::symmetric_differentiated(const Set &p_set) const {
	Set ret = duplicate();
	ret.difference(p_set);
	return ret;
}

bool Set::includes(const Set &p_set) const {
	for (const Variant &E : p_set) {
		if (!has(E)) {
			return false;
		}
	}
	return true;
}

void Set::_unref() const {
	ERR_FAIL_NULL(_p);
	if (_p->refcount.unref()) {
		if (_p->read_only) {
			memdelete(_p->read_only);
		}
		memdelete(_p);
	}
	_p = nullptr;
}

uint32_t Set::hash() const {
	return recursive_hash(0);
}

uint32_t Set::recursive_hash(int p_recursion_count) const {
	if (p_recursion_count > MAX_RECURSION) {
		ERR_PRINT("Max recursion reached");
		return 0;
	}

	uint32_t h = hash_murmur3_one_32(Variant::DICTIONARY);

	p_recursion_count++;
	for (const Variant &E : _p->variant_set) {
		h = hash_murmur3_one_32(E.recursive_hash(p_recursion_count), h);
	}

	return hash_fmix32(h);
}

Array Set::values() const {
	Array varr;
	if (_p->variant_set.is_empty()) {
		return varr;
	}

	varr.resize(size());

	int i = 0;
	for (const Variant &E : _p->variant_set) {
		varr[i] = E;
		i++;
	}

	return varr;
}

Variant Set::get_value_at_index(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, size(), Variant::NIL);
	Iterator it = begin();
	for (int i = 0; i < p_idx; i++) {
		++it;
	}
	return *it;
}

const Variant *Set::next(const Variant *p_value) const {
	if (p_value == nullptr) {
		// caller wants to get the first element
		if (_p->variant_set.begin()) {
			return &*_p->variant_set.begin();
		}
		return nullptr;
	}
	HashSet<Variant, VariantHasher, StringLikeVariantComparator>::Iterator E = _p->variant_set.find(*p_value);

	if (!E) {
		return nullptr;
	}

	++E;

	if (E) {
		return &*E;
	}

	return nullptr;
}

Set Set::duplicate(bool p_deep) const {
	return recursive_duplicate(p_deep, 0);
}

void Set::make_read_only() {
	if (_p->read_only == nullptr) {
		_p->read_only = memnew(Variant);
	}
}
bool Set::is_read_only() const {
	return _p->read_only != nullptr;
}

Set Set::recursive_duplicate(bool p_deep, int recursion_count) const {
	Set n;

	if (recursion_count > MAX_RECURSION) {
		ERR_PRINT("Max recursion reached");
		return n;
	}

	if (p_deep) {
		recursion_count++;
		for (const Variant &E : _p->variant_set) {
			n.add(E.recursive_duplicate(true, recursion_count));
		}
	} else {
		for (const Variant &E : _p->variant_set) {
			n.add(E);
		}
	}

	return n;
}

void Set::operator=(const Set &p_set) {
	if (this == &p_set) {
		return;
	}
	_ref(p_set);
}

const void *Set::id() const {
	return _p;
}

Set::Set(const Set &p_from) {
	_p = nullptr;
	_ref(p_from);
}

Set::Set() {
	_p = memnew(SetPrivate);
	_p->refcount.init();
}

Set::~Set() {
	_unref();
}
