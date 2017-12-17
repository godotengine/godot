/*************************************************************************/
/*  dictionary.cpp                                                       */
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
#include "dictionary.h"

#include "ordered_hash_map.h"
#include "safe_refcount.h"
#include "variant.h"

struct _DictionaryVariantHash {

	static _FORCE_INLINE_ uint32_t hash(const Variant &p_variant) { return p_variant.hash(); }
};

struct DictionaryPrivate {

	SafeRefCount refcount;
	OrderedHashMap<Variant, Variant, _DictionaryVariantHash> variant_map;
};

void Dictionary::get_key_list(List<Variant> *p_keys) const {

	if (_p->variant_map.empty())
		return;

	for (OrderedHashMap<Variant, Variant, _DictionaryVariantHash>::Element E = _p->variant_map.front(); E; E = E.next()) {
		p_keys->push_back(E.key());
	}
}

Variant &Dictionary::operator[](const Variant &p_key) {

	return _p->variant_map[p_key];
}

const Variant &Dictionary::operator[](const Variant &p_key) const {

	return _p->variant_map[p_key];
}
const Variant *Dictionary::getptr(const Variant &p_key) const {

	OrderedHashMap<Variant, Variant, _DictionaryVariantHash>::ConstElement E = ((const OrderedHashMap<Variant, Variant, _DictionaryVariantHash> *)&_p->variant_map)->find(p_key);

	if (!E)
		return NULL;
	return &E.get();
}

Variant *Dictionary::getptr(const Variant &p_key) {

	OrderedHashMap<Variant, Variant, _DictionaryVariantHash>::Element E = _p->variant_map.find(p_key);

	if (!E)
		return NULL;
	return &E.get();
}

Variant Dictionary::get_valid(const Variant &p_key) const {

	OrderedHashMap<Variant, Variant, _DictionaryVariantHash>::ConstElement E = ((const OrderedHashMap<Variant, Variant, _DictionaryVariantHash> *)&_p->variant_map)->find(p_key);

	if (!E)
		return Variant();
	return E.get();
}

int Dictionary::size() const {

	return _p->variant_map.size();
}
bool Dictionary::empty() const {

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

void Dictionary::erase(const Variant &p_key) {

	_p->variant_map.erase(p_key);
}

bool Dictionary::operator==(const Dictionary &p_dictionary) const {

	return _p == p_dictionary._p;
}

void Dictionary::_ref(const Dictionary &p_from) const {

	//make a copy first (thread safe)
	if (!p_from._p->refcount.ref())
		return; // couldn't copy

	//if this is the same, unreference the other one
	if (p_from._p == _p) {
		_p->refcount.unref();
		return;
	}
	if (_p)
		_unref();
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
	_p = NULL;
}
uint32_t Dictionary::hash() const {

	uint32_t h = hash_djb2_one_32(Variant::DICTIONARY);

	List<Variant> keys;
	get_key_list(&keys);

	for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {

		h = hash_djb2_one_32(E->get().hash(), h);
		h = hash_djb2_one_32(operator[](E->get()).hash(), h);
	}

	return h;
}

Array Dictionary::keys() const {

	Array varr;
	varr.resize(size());
	if (_p->variant_map.empty())
		return varr;

	int i = 0;
	for (OrderedHashMap<Variant, Variant, _DictionaryVariantHash>::Element E = _p->variant_map.front(); E; E = E.next()) {
		varr[i] = E.key();
		i++;
	}

	return varr;
}

Array Dictionary::values() const {

	Array varr;
	varr.resize(size());
	if (_p->variant_map.empty())
		return varr;

	int i = 0;
	for (OrderedHashMap<Variant, Variant, _DictionaryVariantHash>::Element E = _p->variant_map.front(); E; E = E.next()) {
		varr[i] = E.get();
		i++;
	}

	return varr;
}

const Variant *Dictionary::next(const Variant *p_key) const {

	if (p_key == NULL) {
		// caller wants to get the first element
		if (_p->variant_map.front())
			return &_p->variant_map.front().key();
		return NULL;
	}
	OrderedHashMap<Variant, Variant, _DictionaryVariantHash>::Element E = _p->variant_map.find(*p_key);

	if (E && E.next())
		return &E.next().key();
	return NULL;
}

Dictionary Dictionary::duplicate() const {

	Dictionary n;

	List<Variant> keys;
	get_key_list(&keys);

	for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
		n[E->get()] = operator[](E->get());
	}

	return n;
}

void Dictionary::operator=(const Dictionary &p_dictionary) {

	_ref(p_dictionary);
}

Dictionary::Dictionary(const Dictionary &p_from) {
	_p = NULL;
	_ref(p_from);
}

Dictionary::Dictionary() {

	_p = memnew(DictionaryPrivate);
	_p->refcount.init();
}
Dictionary::~Dictionary() {

	_unref();
}
