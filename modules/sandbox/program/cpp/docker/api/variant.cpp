/**************************************************************************/
/*  variant.cpp                                                           */
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

#include "variant.hpp"

#include "callable.hpp"
#include "syscalls.h"

MAKE_SYSCALL(ECALL_VCALL, void, sys_vcall, Variant *, const char *, size_t, const Variant *, size_t, Variant &);
MAKE_SYSCALL(ECALL_VEVAL, bool, sys_veval, int, const Variant *, const Variant *, Variant *);
MAKE_SYSCALL(ECALL_VASSIGN, unsigned, sys_vassign, unsigned, Variant *);

MAKE_SYSCALL(ECALL_VCREATE, void, sys_vcreate, Variant *, int, int, ...);
MAKE_SYSCALL(ECALL_VFETCH, void, sys_vfetch, unsigned, void *, int);
MAKE_SYSCALL(ECALL_VCLONE, void, sys_vclone, const Variant *, Variant *);
MAKE_SYSCALL(ECALL_VSTORE, void, sys_vstore, unsigned *, const void *, size_t);

MAKE_SYSCALL(ECALL_CALLABLE_CREATE, unsigned, sys_callable_create, void (*)(), const Variant *, const void *, size_t);

Variant Variant::new_array() {
	Variant v;
	sys_vcreate(&v, ARRAY, 0, nullptr);
	return v;
}

Variant Variant::from_array(const std::vector<Variant> &values) {
	Variant v;
	sys_vcreate(&v, ARRAY, -1, &values);
	return v;
}
Variant Variant::from_array(std::span<const Variant> array) {
	Variant v;
	sys_vcreate(&v, ARRAY, array.size(), array.data());
	return v;
}

Variant Variant::new_dictionary() {
	Variant v;
	sys_vcreate(&v, DICTIONARY, 0, nullptr);
	return v;
}

void Variant::evaluate(const Operator &op, const Variant &a, const Variant &b, Variant &r_ret, bool &r_valid) {
	r_valid = sys_veval(op, &a, &b, &r_ret);
}

void Variant::internal_create_string(Type type, const std::string &value) {
	if constexpr (sizeof(std::string) == 32) {
		sys_vcreate(this, type, 0, &value);
	} else {
		struct Buffer {
			const char *data;
			size_t size;
		} buffer = { value.data(), value.size() };
		sys_vcreate(this, type, 1, &buffer);
	}
}

void Variant::internal_create_u32string(Type type, const std::u32string &value) {
	sys_vcreate(this, type, 2, &value);
}

std::string Variant::internal_fetch_string() const {
	std::string result;
	if constexpr (sizeof(std::string) == 32) {
		sys_vfetch(this->v.i, &result, 0); // Fetch as std::string
	} else {
		struct Buffer {
			char *data;
			size_t size;
		} buffer;
		sys_vfetch(this->v.i, &buffer, 1); // Fetch as const char*, size_t struct
		result.assign(buffer.data, buffer.size);
		delete[] buffer.data;
	}
	return result;
}

std::u32string Variant::internal_fetch_u32string() const {
	std::u32string result;
	sys_vfetch(this->v.i, &result, 2); // Fetch as std::u32string
	return result;
}

void Variant::internal_clone(const Variant &other) {
	sys_vclone(&other, this);
}

Variant Variant::duplicate() const {
	Variant v;
	sys_vclone(this, &v);
	return v;
}

void Variant::clear() {
	// TODO: If the Variant is a reference, we should clear the reference.
	this->m_type = NIL;
}

Variant &Variant::make_permanent() {
	sys_vclone(this, nullptr);
	return *this;
}

bool Variant::is_permanent() const noexcept {
	return int32_t(uint32_t(this->v.i)) < 0;
}
