/**************************************************************************/
/*  dictionary.cpp                                                        */
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

#include "dictionary.h"

#include "core/templates/hash_map.h"
#include "core/templates/safe_refcount.h"
#include "core/variant/variant.h"
// required in this order by VariantInternal, do not remove this comment.
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/variant/type_info.h"
#include "core/variant/variant_internal.h"

struct DictionaryPrivate {
	SafeRefCount refcount;
	Variant *read_only = nullptr; // If enabled, a pointer is used to a temporary value that is used to return read-only values.
	HashMap<Variant, Variant, VariantHasher, StringLikeVariantComparator> variant_map;
};

void Dictionary::get_key_list(List<Variant> *p_keys) const {
	if (_p->variant_map.is_empty()) {
		return;
	}

	for (const KeyValue<Variant, Variant> &E : _p->variant_map) {
		p_keys->push_back(E.key);
	}
}

Variant Dictionary::get_key_at_index(int p_index) const {
	int index = 0;
	for (const KeyValue<Variant, Variant> &E : _p->variant_map) {
		if (index == p_index) {
			return E.key;
		}
		index++;
	}

	return Variant();
}

Variant Dictionary::get_value_at_index(int p_index) const {
	int index = 0;
	for (const KeyValue<Variant, Variant> &E : _p->variant_map) {
		if (index == p_index) {
			return E.value;
		}
		index++;
	}

	return Variant();
}

Variant &Dictionary::operator[](const Variant &p_key) {
	if (unlikely(_p->read_only)) {
		if (p_key.get_type() == Variant::STRING_NAME) {
			const StringName *sn = VariantInternal::get_string_name(&p_key);
			const String &key = sn->operator String();
			if (likely(_p->variant_map.has(key))) {
				*_p->read_only = _p->variant_map[key];
			} else {
				*_p->read_only = Variant();
			}
		} else if (likely(_p->variant_map.has(p_key))) {
			*_p->read_only = _p->variant_map[p_key];
		} else {
			*_p->read_only = Variant();
		}

		return *_p->read_only;
	} else {
		if (p_key.get_type() == Variant::STRING_NAME) {
			const StringName *sn = VariantInternal::get_string_name(&p_key);
			return _p->variant_map[sn->operator String()];
		} else {
			return _p->variant_map[p_key];
		}
	}
}

const Variant &Dictionary::operator[](const Variant &p_key) const {
	// Will not insert key, so no conversion is necessary.
	return _p->variant_map[p_key];
}

const Variant *Dictionary::getptr(const Variant &p_key) const {
	HashMap<Variant, Variant, VariantHasher, StringLikeVariantComparator>::ConstIterator E(_p->variant_map.find(p_key));
	if (!E) {
		return nullptr;
	}
	return &E->value;
}

Variant *Dictionary::getptr(const Variant &p_key) {
	HashMap<Variant, Variant, VariantHasher, StringLikeVariantComparator>::Iterator E(_p->variant_map.find(p_key));
	if (!E) {
		return nullptr;
	}
	if (unlikely(_p->read_only != nullptr)) {
		*_p->read_only = E->value;
		return _p->read_only;
	} else {
		return &E->value;
	}
}

Variant Dictionary::get_valid(const Variant &p_key) const {
	HashMap<Variant, Variant, VariantHasher, StringLikeVariantComparator>::ConstIterator E(_p->variant_map.find(p_key));

	if (!E) {
		return Variant();
	}
	return E->value;
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
	return _p->variant_map.has(p_key);
}

bool Dictionary::has_all(const Array &p_keys) const {
	for (int i = 0; i < p_keys.size(); i++) {
		if (!has(p_keys[i])) {
			return false;
		}
	}
	return true;
}

Variant Dictionary::find_key(const Variant &p_value) const {
	for (const KeyValue<Variant, Variant> &E : _p->variant_map) {
		if (E.value == p_value) {
			return E.key;
		}
	}
	return Variant();
}

bool Dictionary::erase(const Variant &p_key) {
	ERR_FAIL_COND_V_MSG(_p->read_only, false, "Dictionary is in read-only state.");
	return _p->variant_map.erase(p_key);
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
	for (const KeyValue<Variant, Variant> &this_E : _p->variant_map) {
		HashMap<Variant, Variant, VariantHasher, StringLikeVariantComparator>::ConstIterator other_E(p_dictionary._p->variant_map.find(this_E.key));
		if (!other_E || !this_E.value.hash_compare(other_E->value, recursion_count, false)) {
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
	ERR_FAIL_COND_MSG(_p->read_only, "Dictionary is in read-only state.");
	_p->variant_map.clear();
}

void Dictionary::merge(const Dictionary &p_dictionary, bool p_overwrite) {
	for (const KeyValue<Variant, Variant> &E : p_dictionary._p->variant_map) {
		if (p_overwrite || !has(E.key)) {
			this->operator[](E.key) = E.value;
		}
	}
}

void Dictionary::_unref() const {
	ERR_FAIL_NULL(_p);
	if (_p->refcount.unref()) {
		if (_p->read_only) {
			memdelete(_p->read_only);
		}
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

	uint32_t h = hash_murmur3_one_32(Variant::DICTIONARY);

	recursion_count++;
	for (const KeyValue<Variant, Variant> &E : _p->variant_map) {
		h = hash_murmur3_one_32(E.key.recursive_hash(recursion_count), h);
		h = hash_murmur3_one_32(E.value.recursive_hash(recursion_count), h);
	}

	return hash_fmix32(h);
}

Array Dictionary::keys() const {
	Array varr;
	if (_p->variant_map.is_empty()) {
		return varr;
	}

	varr.resize(size());

	int i = 0;
	for (const KeyValue<Variant, Variant> &E : _p->variant_map) {
		varr[i] = E.key;
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
	for (const KeyValue<Variant, Variant> &E : _p->variant_map) {
		varr[i] = E.value;
		i++;
	}

	return varr;
}

const Variant *Dictionary::next(const Variant *p_key) const {
	if (p_key == nullptr) {
		// caller wants to get the first element
		if (_p->variant_map.begin()) {
			return &_p->variant_map.begin()->key;
		}
		return nullptr;
	}
	HashMap<Variant, Variant, VariantHasher, StringLikeVariantComparator>::Iterator E = _p->variant_map.find(*p_key);

	if (!E) {
		return nullptr;
	}

	++E;

	if (E) {
		return &E->key;
	}

	return nullptr;
}

Dictionary Dictionary::duplicate(bool p_deep) const {
	return recursive_duplicate(p_deep, 0);
}

void Dictionary::make_read_only() {
	if (_p->read_only == nullptr) {
		_p->read_only = memnew(Variant);
	}
}
bool Dictionary::is_read_only() const {
	return _p->read_only != nullptr;
}

Dictionary Dictionary::recursive_duplicate(bool p_deep, int recursion_count) const {
	Dictionary n;

	if (recursion_count > MAX_RECURSION) {
		ERR_PRINT("Max recursion reached");
		return n;
	}

	if (p_deep) {
		recursion_count++;
		for (const KeyValue<Variant, Variant> &E : _p->variant_map) {
			n[E.key.recursive_duplicate(true, recursion_count)] = E.value.recursive_duplicate(true, recursion_count);
		}
	} else {
		for (const KeyValue<Variant, Variant> &E : _p->variant_map) {
			n[E.key] = E.value;
		}
	}

	return n;
}

void Dictionary::operator=(const Dictionary &p_dictionary) {
	if (this == &p_dictionary) {
		return;
	}
	_ref(p_dictionary);
}

const void *Dictionary::id() const {
	return _p;
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
