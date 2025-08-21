/**************************************************************************/
/*  string.cpp                                                            */
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

#include "string.hpp"

#include "syscalls.h"

MAKE_SYSCALL(ECALL_STRING_CREATE, unsigned, sys_string_create, const char *, size_t);
MAKE_SYSCALL(ECALL_STRING_OPS, int, sys_string_ops, String_Op, unsigned, int, ...);
MAKE_SYSCALL(ECALL_STRING_AT, unsigned, sys_string_at, unsigned, int);
MAKE_SYSCALL(ECALL_STRING_SIZE, int, sys_string_size, unsigned);
MAKE_SYSCALL(ECALL_STRING_APPEND, void, sys_string_append, unsigned, const char *, size_t);
EXTERN_SYSCALL(unsigned, sys_vassign, unsigned, unsigned);

String &String::operator=(const String &value) {
	this->m_idx = sys_vassign(m_idx, value.m_idx);
	return *this;
}

void String::append(const String &value) {
	(void)sys_string_ops(String_Op::APPEND, m_idx, 0, &value);
}

void String::insert(int idx, const String &value) {
	(void)sys_string_ops(String_Op::INSERT, m_idx, 0, &value);
}

void String::append(std::string_view value) {
	(void)sys_string_append(m_idx, value.data(), value.size());
}

void String::erase(int idx, int count) {
	sys_string_ops(String_Op::ERASE, m_idx, idx, count);
}

int String::find(const String &value) const {
	return sys_string_ops(String_Op::FIND, m_idx, 0, &value);
}

String String::operator[](int idx) const {
	unsigned new_stridx = sys_string_at(m_idx, idx);
	return String::from_variant_index(new_stridx);
}

int String::size() const {
	return sys_string_size(m_idx);
}

unsigned String::Create(const char *data, size_t size) {
	return sys_string_create(data, size);
}

std::string String::utf8() const {
	std::string str;
	if constexpr (sizeof(std::string) == 32) {
		sys_string_ops(String_Op::TO_STD_STRING, m_idx, 0, &str);
	} else {
		// Guesstimate that the string is less than 32 bytes.
		str.resize_and_overwrite(32, [idx = m_idx](char *data, size_t size) -> std::size_t {
			struct Buffer {
				char *data;
				size_t size;
			} buffer;
			buffer.data = data;
			buffer.size = size;
			// This syscall will either copy to the existing buffer or allocate a new one,
			// and then update the buffer struct with the new data, freeing the old buffer.
			sys_string_ops(String_Op::TO_STD_STRING, idx, 1, &buffer);
			return buffer.size;
		});
	}
	return str;
}

std::u32string String::utf32() const {
	std::u32string str;
	sys_string_ops(String_Op::TO_STD_STRING, m_idx, 2, &str);
	return str;
}

bool String::operator==(const String &other) const {
	return sys_string_ops(String_Op::COMPARE, m_idx, 0, &other);
}

bool String::operator==(const char *other) const {
	return sys_string_ops(String_Op::COMPARE_CSTR, m_idx, 0, other);
}
