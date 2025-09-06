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

#include "dictionary.hpp"

#include "syscalls.h"

EXTERN_SYSCALL(void, sys_vcreate, Variant *, int, int, ...);
MAKE_SYSCALL(ECALL_DICTIONARY_OPS, int, sys_dict_ops, Dictionary_Op, unsigned, ...);
EXTERN_SYSCALL(unsigned, sys_vassign, unsigned, unsigned);

Dictionary &Dictionary::operator=(const Dictionary &other) {
	this->m_idx = sys_vassign(this->m_idx, other.m_idx);
	return *this;
}

void Dictionary::clear() {
	(void)sys_dict_ops(Dictionary_Op::CLEAR, m_idx);
}

void Dictionary::erase(const Variant &key) {
	(void)sys_dict_ops(Dictionary_Op::ERASE, m_idx, &key);
}

bool Dictionary::has(const Variant &key) const {
	return sys_dict_ops(Dictionary_Op::HAS, m_idx, &key);
}

int Dictionary::size() const {
	return sys_dict_ops(Dictionary_Op::GET_SIZE, m_idx);
}

Variant Dictionary::get(const Variant &key) const {
	Variant v;
	(void)sys_dict_ops(Dictionary_Op::GET, m_idx, &key, &v);
	return v;
}
void Dictionary::set(const Variant &key, const Variant &value) {
	(void)sys_dict_ops(Dictionary_Op::SET, m_idx, &key, &value);
}
Variant Dictionary::get_or_add(const Variant &key, const Variant &default_value) {
	Variant v;
	(void)sys_dict_ops(Dictionary_Op::GET_OR_ADD, m_idx, &key, &v, &default_value);
	return v;
}

void Dictionary::merge(const Dictionary &other) {
	Variant v(other);
	(void)sys_dict_ops(Dictionary_Op::MERGE, m_idx, &v);
}

Dictionary Dictionary::Create() {
	Variant v;
	sys_vcreate(&v, Variant::DICTIONARY, 0);
	Dictionary d;
	d.m_idx = v.get_internal_index();
	return d;
}
