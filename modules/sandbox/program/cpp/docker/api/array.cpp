/**************************************************************************/
/*  array.cpp                                                             */
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

#include "array.hpp"

#include "syscalls.h"

MAKE_SYSCALL(ECALL_ARRAY_OPS, void, sys_array_ops, Array_Op, unsigned, int, Variant *);
MAKE_SYSCALL(ECALL_ARRAY_AT, void, sys_array_at, unsigned, int, Variant *);
MAKE_SYSCALL(ECALL_ARRAY_SIZE, int, sys_array_size, unsigned);
EXTERN_SYSCALL(unsigned, sys_vassign, unsigned, unsigned);

Array &Array::operator=(const std::vector<Variant> &values) {
	Variant v = Variant::from_array(values);
	this->m_idx = sys_vassign(m_idx, v.get_internal_index());
	return *this;
}

Array &Array::operator=(const Array &other) {
	this->m_idx = sys_vassign(this->m_idx, other.m_idx);
	return *this;
}

void Array::push_back(const Variant &value) {
	sys_array_ops(Array_Op::PUSH_BACK, m_idx, 0, (Variant *)&value);
}

void Array::push_front(const Variant &value) {
	sys_array_ops(Array_Op::PUSH_FRONT, m_idx, 0, (Variant *)&value);
}

void Array::pop_at(int idx) {
	sys_array_ops(Array_Op::POP_AT, m_idx, idx, nullptr);
}

void Array::pop_back() {
	sys_array_ops(Array_Op::POP_BACK, m_idx, 0, nullptr);
}

void Array::pop_front() {
	sys_array_ops(Array_Op::POP_FRONT, m_idx, 0, nullptr);
}

void Array::insert(int idx, const Variant &value) {
	sys_array_ops(Array_Op::INSERT, m_idx, idx, (Variant *)&value);
}

void Array::erase(const Variant &value) {
	sys_array_ops(Array_Op::ERASE, m_idx, 0, (Variant *)&value);
}

void Array::resize(int size) {
	sys_array_ops(Array_Op::RESIZE, m_idx, size, nullptr);
}

void Array::clear() {
	sys_array_ops(Array_Op::CLEAR, m_idx, 0, nullptr);
}

void Array::sort() {
	sys_array_ops(Array_Op::SORT, m_idx, 0, nullptr);
}

int Array::size() const {
	return sys_array_size(m_idx);
}

bool Array::has(const Variant &value) const {
	Variant v = value;
	sys_array_ops(Array_Op::HAS, m_idx, 0, &v);
	return v;
}

Array::Array(unsigned size) {
	Variant v;
	sys_array_ops(Array_Op::CREATE, size, 0, &v);
	this->m_idx = v.get_internal_index();
}

Array::Array(const std::vector<Variant> &values) {
	Variant v = Variant::from_array(values);
	this->m_idx = v.get_internal_index();
}

std::vector<Variant> Array::to_vector() const {
	std::vector<Variant> result;
	sys_array_ops(Array_Op::FETCH_TO_VECTOR, m_idx, 0, (Variant *)&result);
	return result;
}

ArrayProxy &ArrayProxy::operator=(const Variant &value) { // set
	const int set_idx = -this->m_idx - 1;
	sys_array_at(this->m_array.get_variant_index(), set_idx, (Variant *)&value);
	return *this;
}

Variant ArrayProxy::get() const { // get
	Variant v;
	sys_array_at(this->m_array.get_variant_index(), this->m_idx, &v);
	return v;
}
