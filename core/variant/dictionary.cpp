/*************************************************************************/
/*  dictionary.cpp                                                       */
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

#include "dictionary.h"

#include "core/templates/ordered_hash_map.h"
#include "core/templates/safe_refcount.h"
#include "core/variant/variant.h"
// required in this order by VariantInternal, do not remove this comment.
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/variant/type_info.h"
#include "core/variant/variant_internal.h"

struct DictionaryPrivate {
	SafeRefCount refcount;
	OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator> variant_map;
};

void Dictionary::get_key_list(List<Variant> *p_keys) const {
	if (_p->variant_map.is_empty()) {
		return;
	}

	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		p_keys->push_back(E.key());
	}
}

Variant Dictionary::get_key_at_index(int p_index) const {
	int index = 0;
	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		if (index == p_index) {
			return E.key();
		}
		index++;
	}

	return Variant();
}

Variant Dictionary::get_value_at_index(int p_index) const {
	int index = 0;
	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		if (index == p_index) {
			return E.value();
		}
		index++;
	}

	return Variant();
}

Variant &Dictionary::operator[](const Variant &p_key) {
	if (p_key.get_type() == Variant::STRING_NAME) {
		const StringName *sn = VariantInternal::get_string_name(&p_key);
		return _p->variant_map[sn->operator String()];
	} else {
		return _p->variant_map[p_key];
	}
}

const Variant &Dictionary::operator[](const Variant &p_key) const {
	if (p_key.get_type() == Variant::STRING_NAME) {
		const StringName *sn = VariantInternal::get_string_name(&p_key);
		return _p->variant_map[sn->operator String()];
	} else {
		return _p->variant_map[p_key];
	}
}

const Variant *Dictionary::getptr(const Variant &p_key) const {
	OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::ConstElement E;

	if (p_key.get_type() == Variant::STRING_NAME) {
		const StringName *sn = VariantInternal::get_string_name(&p_key);
		E = ((const OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator> *)&_p->variant_map)->find(sn->operator String());
	} else {
		E = ((const OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator> *)&_p->variant_map)->find(p_key);
	}

	if (!E) {
		return nullptr;
	}
	return &E.get();
}

Variant *Dictionary::getptr(const Variant &p_key) {
	OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E;

	if (p_key.get_type() == Variant::STRING_NAME) {
		const StringName *sn = VariantInternal::get_string_name(&p_key);
		E = ((OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator> *)&_p->variant_map)->find(sn->operator String());
	} else {
		E = ((OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator> *)&_p->variant_map)->find(p_key);
	}
	if (!E) {
		return nullptr;
	}
	return &E.get();
}

Variant Dictionary::get_valid(const Variant &p_key) const {
	OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::ConstElement E;

	if (p_key.get_type() == Variant::STRING_NAME) {
		const StringName *sn = VariantInternal::get_string_name(&p_key);
		E = ((const OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator> *)&_p->variant_map)->find(sn->operator String());
	} else {
		E = ((const OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator> *)&_p->variant_map)->find(p_key);
	}

	if (!E) {
		return Variant();
	}
	return E.get();
}

Variant Dictionary::get(const Variant &p_key, const Variant &p_default) const {
	const Variant *result = getptr(p_key);
	if (!result) {
		return p_default;
	}

	return *result;
}

int Dictionary::size() const {
	return _p->variant_map.size();
}

bool Dictionary::is_empty() const {
	return !_p->variant_map.size();
}

bool Dictionary::has(const Variant &p_key) const {
	if (p_key.get_type() == Variant::STRING_NAME) {
		const StringName *sn = VariantInternal::get_string_name(&p_key);
		return _p->variant_map.has(sn->operator String());
	} else {
		return _p->variant_map.has(p_key);
	}
}

bool Dictionary::has_all(const Array &p_keys) const {
	for (int i = 0; i < p_keys.size(); i++) {
		if (!has(p_keys[i])) {
			return false;
		}
	}
	return true;
}

bool Dictionary::erase(const Variant &p_key) {
	if (p_key.get_type() == Variant::STRING_NAME) {
		const StringName *sn = VariantInternal::get_string_name(&p_key);
		return _p->variant_map.erase(sn->operator String());
	} else {
		return _p->variant_map.erase(p_key);
	}
}

bool Dictionary::operator==(const Dictionary &p_dictionary) const {
	return recursive_equal(p_dictionary, 0);
}

bool Dictionary::operator!=(const Dictionary &p_dictionary) const {
	return !recursive_equal(p_dictionary, 0);
}

bool Dictionary::recursive_equal(const Dictionary &p_dictionary, int recursion_count) const {
	// Cheap checks
	if (_p == p_dictionary._p) {
		return true;
	}
	if (_p->variant_map.size() != p_dictionary._p->variant_map.size()) {
		return false;
	}

	// Heavy O(n) check
	if (recursion_count > MAX_RECURSION) {
		ERR_PRINT("Max recursion reached");
		return true;
	}
	recursion_count++;
	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::ConstElement this_E = ((const OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator> *)&_p->variant_map)->front(); this_E; this_E = this_E.next()) {
		OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::ConstElement other_E = ((const OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator> *)&p_dictionary._p->variant_map)->find(this_E.key());
		if (!other_E || !this_E.value().hash_compare(other_E.value(), recursion_count)) {
			return false;
		}
	}
	return true;
}

void Dictionary::_ref(const Dictionary &p_from) const {
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

void Dictionary::clear() {
	_p->variant_map.clear();
}

void Dictionary::_unref() const {
	ERR_FAIL_COND(!_p);
	if (_p->refcount.unref()) {
		memdelete(_p);
	}
	_p = nullptr;
}

uint32_t Dictionary::hash() const {
	return recursive_hash(0);
}

uint32_t Dictionary::recursive_hash(int recursion_count) const {
	if (recursion_count > MAX_RECURSION) {
		ERR_PRINT("Max recursion reached");
		return 0;
	}

	uint32_t h = hash_djb2_one_32(Variant::DICTIONARY);

	recursion_count++;
	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		h = hash_djb2_one_32(E.key().recursive_hash(recursion_count), h);
		h = hash_djb2_one_32(E.value().recursive_hash(recursion_count), h);
	}

	return h;
}

Array Dictionary::keys() const {
	Array varr;
	if (_p->variant_map.is_empty()) {
		return varr;
	}

	varr.resize(size());

	int i = 0;
	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		varr[i] = E.key();
		i++;
	}

	return varr;
}

Array Dictionary::values() const {
	Array varr;
	if (_p->variant_map.is_empty()) {
		return varr;
	}

	varr.resize(size());

	int i = 0;
	for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
		varr[i] = E.get();
		i++;
	}

	return varr;
}

const Variant *Dictionary::next(const Variant *p_key) const {
	if (p_key == nullptr) {
		// caller wants to get the first element
		if (_p->variant_map.front()) {
			return &_p->variant_map.front().key();
		}
		return nullptr;
	}
	OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.find(*p_key);

	if (E && E.next()) {
		return &E.next().key();
	}
	return nullptr;
}

Dictionary Dictionary::duplicate(bool p_deep) const {
	return recursive_duplicate(p_deep, 0);
}

Dictionary Dictionary::recursive_duplicate(bool p_deep, int recursion_count) const {
	Dictionary n;

	if (recursion_count > MAX_RECURSION) {
		ERR_PRINT("Max recursion reached");
		return n;
	}

	if (p_deep) {
		recursion_count++;
		for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
			n[E.key().recursive_duplicate(true, recursion_count)] = E.value().recursive_duplicate(true, recursion_count);
		}
	} else {
		for (OrderedHashMap<Variant, Variant, VariantHasher, VariantComparator>::Element E = _p->variant_map.front(); E; E = E.next()) {
			n[E.key()] = E.value();
		}
	}

	return n;
}

void Dictionary::operator=(const Dictionary &p_dictionary) {
	_ref(p_dictionary);
}

const void *Dictionary::id() const {
	return _p->variant_map.id();
}

Dictionary::Dictionary(const Dictionary &p_from) {
	_p = nullptr;
	_ref(p_from);
}

Dictionary::Dictionary() {
	_p = memnew(DictionaryPrivate);
	_p->refcount.init();
}

Dictionary::~Dictionary() {
	_unref();
}
