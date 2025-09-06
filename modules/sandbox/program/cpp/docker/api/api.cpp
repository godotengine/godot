/**************************************************************************/
/*  api.cpp                                                               */
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

#include "api.hpp"

#include "syscalls.h"
#include <cstring>
#include <exception>

/* void sys_print(const Variant *, size_t) */
MAKE_SYSCALL(ECALL_PRINT, void, sys_print, const Variant *, size_t);
MAKE_SYSCALL(ECALL_THROW, void, sys_throw, const char *, size_t, const char *, size_t, ...);
EXTERN_SYSCALL(uint64_t, sys_node_create, Node_Create_Shortlist, const char *, size_t, const char *, size_t);
MAKE_SYSCALL(ECALL_LOAD, void, sys_load, const char *, size_t, Variant *);
MAKE_SYSCALL(ECALL_SANDBOX_ADD, void, sys_sandbox_add, int, ...);

/* Default main: Do nothing */
__attribute__((weak)) int main() {
	halt(); // Prevent closing pipes, calling global destructors etc.
}

/* fast_exit */
extern "C" __attribute__((used, retain, noreturn)) void fast_exit() {
	asm(".insn i SYSTEM, 0, x0, x0, 0x7ff");
	__builtin_unreachable();
}

// ClassDB::instantiate
Object ClassDB::instantiate(std::string_view class_name, std::string_view name) {
	return Object(sys_node_create(Node_Create_Shortlist::CREATE_CLASSDB, class_name.data(), class_name.size(), name.data(), name.size()));
}

// Resource loader
Variant loadv(std::string_view path) {
	Variant result;
	sys_load(path.data(), path.size(), &result);
	return result;
}

__attribute__((constructor, used)) void setup_native_stuff() {
	/* Set exit address to fast_exit */
	sys_sandbox_add(2, &fast_exit);
	/* Handle uncaught C++ exceptions */
	std::set_terminate([] {
		try {
			std::rethrow_exception(std::current_exception());
		} catch (const std::exception &e) {
			const auto *name = typeid(e).name();
			sys_throw(name, strlen(name), e.what(), strlen(e.what()), nullptr);
			__builtin_unreachable();
		}
	});
}

// Use Godot-Sandbox Math system calls for some math functions.
// 32-bit floating point math functions:
extern "C" __attribute__((used)) float sinf(float x) {
	return Math::sinf(x);
}
extern "C" __attribute__((used)) float cosf(float x) {
	return Math::cosf(x);
}
extern "C" __attribute__((used)) float tanf(float x) {
	return Math::tanf(x);
}
extern "C" __attribute__((used)) float asinf(float x) {
	return Math::asinf(x);
}
extern "C" __attribute__((used)) float acosf(float x) {
	return Math::acosf(x);
}
extern "C" __attribute__((used)) float atanf(float x) {
	return Math::atanf(x);
}
extern "C" __attribute__((used)) float atan2f(float y, float x) {
	return Math::atan2f(y, x);
}
extern "C" __attribute__((used)) float powf(float x, float y) {
	return Math::powf(x, y);
}
// 64-bit floating point math functions:
extern "C" __attribute__((used)) double sin(double x) {
	return Math::sin(x);
}
extern "C" __attribute__((used)) double cos(double x) {
	return Math::cos(x);
}
extern "C" __attribute__((used)) double tan(double x) {
	return Math::tan(x);
}
extern "C" __attribute__((used)) double asin(double x) {
	return Math::asin(x);
}
extern "C" __attribute__((used)) double acos(double x) {
	return Math::acos(x);
}
extern "C" __attribute__((used)) double atan(double x) {
	return Math::atan(x);
}
extern "C" __attribute__((used)) double atan2(double y, double x) {
	return Math::atan2(y, x);
}
extern "C" __attribute__((used)) double pow(double x, double y) {
	return Math::pow(x, y);
}

// clang-format off
#define STR2(x) #x
#define STR(x) STR2(x)
__asm__(".pushsection .comment\n\t"
		".string \"Godot C++ API v" STR(VERSION) "\"\n\t"
		".popsection");
// clang-format on
