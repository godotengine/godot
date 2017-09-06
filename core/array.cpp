/*************************************************************************/
/*  array.cpp                                                            */
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
#include "array.h"

#include "hashfuncs.h"
#include "object.h"
#include "variant.h"
#include "vector.h"

struct ArrayPrivate {

	SafeRefCount refcount;
	Vector<Variant> array;
};

void Array::_ref(const Array &p_from) const {

	ArrayPrivate *_fp = p_from._p;

	ERR_FAIL_COND(!_fp); // should NOT happen.

	if (_fp == _p)
		return; //wathever it is, nothing to do here move along

	bool success = _fp->refcount.ref();

	ERR_FAIL_COND(!success); //should really not happen either

	_unref();

	_p = p_from._p;
}

void Array::_unref() const {

	if (!_p)
		return;

	if (_p->refcount.unref()) {
		memdelete(_p);
	}
	_p = NULL;
}

Variant &Array::operator[](int p_idx) {

	return _p->array[p_idx];
}

const Variant &Array::operator[](int p_idx) const {

	return _p->array[p_idx];
}

int Array::size() const {

	return _p->array.size();
}
bool Array::empty() const {

	return _p->array.empty();
}
void Array::clear() {

	_p->array.clear();
}

bool Array::operator==(const Array &p_array) const {

	return _p == p_array._p;
}

uint32_t Array::hash() const {

	uint32_t h = hash_djb2_one_32(0);

	for (int i = 0; i < _p->array.size(); i++) {

		h = hash_djb2_one_32(_p->array[i].hash(), h);
	}
	return h;
}
void Array::operator=(const Array &p_array) {

	_ref(p_array);
}
void Array::push_back(const Variant &p_value) {

	_p->array.push_back(p_value);
}

Error Array::resize(int p_new_size) {

	return _p->array.resize(p_new_size);
}

void Array::insert(int p_pos, const Variant &p_value) {

	_p->array.insert(p_pos, p_value);
}

void Array::erase(const Variant &p_value) {

	_p->array.erase(p_value);
}

Variant Array::front() const {
	ERR_FAIL_COND_V(_p->array.size() == 0, Variant());
	return operator[](0);
}

Variant Array::back() const {
	ERR_FAIL_COND_V(_p->array.size() == 0, Variant());
	return operator[](_p->array.size() - 1);
}

int Array::find(const Variant &p_value, int p_from) const {

	return _p->array.find(p_value, p_from);
}

int Array::rfind(const Variant &p_value, int p_from) const {

	if (_p->array.size() == 0)
		return -1;

	if (p_from < 0) {
		// Relative offset from the end
		p_from = _p->array.size() + p_from;
	}
	if (p_from < 0 || p_from >= _p->array.size()) {
		// Limit to array boundaries
		p_from = _p->array.size() - 1;
	}

	for (int i = p_from; i >= 0; i--) {

		if (_p->array[i] == p_value) {
			return i;
		};
	};

	return -1;
}

int Array::find_last(const Variant &p_value) const {

	return rfind(p_value);
}

int Array::count(const Variant &p_value) const {

	if (_p->array.size() == 0)
		return 0;

	int amount = 0;
	for (int i = 0; i < _p->array.size(); i++) {

		if (_p->array[i] == p_value) {
			amount++;
		};
	};

	return amount;
}

bool Array::has(const Variant &p_value) const {
	return _p->array.find(p_value, 0) != -1;
}

void Array::remove(int p_pos) {

	_p->array.remove(p_pos);
}

void Array::set(int p_idx, const Variant &p_value) {

	operator[](p_idx) = p_value;
}

const Variant &Array::get(int p_idx) const {

	return operator[](p_idx);
}

Array Array::duplicate() const {

	Array new_arr;
	int element_count = size();
	new_arr.resize(element_count);
	for (int i = 0; i < element_count; i++) {
		new_arr[i] = get(i);
	}

	return new_arr;
}
struct _ArrayVariantSort {

	_FORCE_INLINE_ bool operator()(const Variant &p_l, const Variant &p_r) const {
		bool valid = false;
		Variant res;
		Variant::evaluate(Variant::OP_LESS, p_l, p_r, res, valid);
		if (!valid)
			res = false;
		return res;
	}
};

void Array::sort() {

	_p->array.sort_custom<_ArrayVariantSort>();
}

struct _ArrayVariantSortCustom {

	Object *obj;
	StringName func;

	_FORCE_INLINE_ bool operator()(const Variant &p_l, const Variant &p_r) const {

		const Variant *args[2] = { &p_l, &p_r };
		Variant::CallError err;
		bool res = obj->call(func, args, 2, err);
		if (err.error != Variant::CallError::CALL_OK)
			res = false;
		return res;
	}
};
void Array::sort_custom(Object *p_obj, const StringName &p_function) {

	ERR_FAIL_NULL(p_obj);

	SortArray<Variant, _ArrayVariantSortCustom> avs;
	avs.compare.obj = p_obj;
	avs.compare.func = p_function;
	avs.sort(_p->array.ptr(), _p->array.size());
}

void Array::invert() {

	_p->array.invert();
}

void Array::push_front(const Variant &p_value) {

	_p->array.insert(0, p_value);
}

Variant Array::pop_back() {

	if (!_p->array.empty()) {
		int n = _p->array.size() - 1;
		Variant ret = _p->array.get(n);
		_p->array.resize(n);
		return ret;
	}
	return Variant();
}

Variant Array::pop_front() {

	if (!_p->array.empty()) {
		Variant ret = _p->array.get(0);
		_p->array.remove(0);
		return ret;
	}
	return Variant();
}

Array::Array(const Array &p_from) {

	_p = NULL;
	_ref(p_from);
}
Array::Array() {

	_p = memnew(ArrayPrivate);
	_p->refcount.init();
}
Array::~Array() {

	_unref();
}
