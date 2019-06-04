/*************************************************************************/
/*  dictionary.cpp                                                       */
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
#include "dictionary.h"
#include "io/json.h"
#include "safe_refcount.h"
#include "variant.h"

struct _DictionaryVariantHash {

	static _FORCE_INLINE_ uint32_t hash(const Variant &p_variant) { return p_variant.hash(); }
};

struct DictionaryPrivate {

	SafeRefCount refcount;
	HashMap<Variant, Variant, _DictionaryVariantHash> variant_map;
	bool shared;
};

void Dictionary::get_key_list(List<Variant> *p_keys) const {

	_p->variant_map.get_key_list(p_keys);
}

void Dictionary::_copy_on_write() const {

	//make a copy of what we have
	if (_p->shared)
		return;

	DictionaryPrivate *p = memnew(DictionaryPrivate);
	p->shared = _p->shared;
	p->variant_map = _p->variant_map;
	p->refcount.init();
	_unref();
	_p = p;
}

Variant &Dictionary::operator[](const Variant &p_key) {

	_copy_on_write();

	return _p->variant_map[p_key];
}

const Variant &Dictionary::operator[](const Variant &p_key) const {

	return _p->variant_map[p_key];
}
const Variant *Dictionary::getptr(const Variant &p_key) const {

	return _p->variant_map.getptr(p_key);
}
Variant *Dictionary::getptr(const Variant &p_key) {

	_copy_on_write();
	return _p->variant_map.getptr(p_key);
}

Variant Dictionary::get_valid(const Variant &p_key) const {

	const Variant *v = getptr(p_key);
	if (!v)
		return Variant();
	return *v;
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
	_copy_on_write();
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

	_copy_on_write();
	_p->variant_map.clear();
}

bool Dictionary::is_shared() const {

	return _p->shared;
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

	Array karr;
	karr.resize(size());
	const Variant *K = NULL;
	int idx = 0;
	while ((K = next(K))) {
		karr[idx++] = (*K);
	}
	return karr;
}

Array Dictionary::values() const {

	Array varr;
	varr.resize(size());
	const Variant *key = NULL;
	int i = 0;
	while ((key = next(key))) {
		varr[i++] = _p->variant_map[*key];
	}
	return varr;
}

const Variant *Dictionary::next(const Variant *p_key) const {

	return _p->variant_map.next(p_key);
}

Error Dictionary::parse_json(const String &p_json) {

	String errstr;
	int errline = 0;
	if (p_json != "") {
		Error err = JSON::parse(p_json, *this, errstr, errline);
		if (err != OK) {
			ERR_EXPLAIN("Error parsing JSON: " + errstr + " at line: " + itos(errline));
			ERR_FAIL_COND_V(err != OK, err);
		}
	}

	return OK;
}

String Dictionary::to_json() const {

	return JSON::print(*this);
}

void Dictionary::operator=(const Dictionary &p_dictionary) {

	_ref(p_dictionary);
}

Dictionary::Dictionary(const Dictionary &p_from) {
	_p = NULL;
	_ref(p_from);
}

Dictionary::Dictionary(bool p_shared) {

	_p = memnew(DictionaryPrivate);
	_p->refcount.init();
	_p->shared = p_shared;
}
Dictionary::~Dictionary() {

	_unref();
}
