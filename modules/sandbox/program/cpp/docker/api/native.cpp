/**************************************************************************/
/*  native.cpp                                                            */
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

#include "syscalls.h"
#include <array>
#include <cstddef>
#include <cstdint>

#define NATIVE_MEM_FUNCATTR /* */
#define NATIVE_SYSCALLS_BASE 480 /* libc starts at 480 */

#define SYSCALL_MALLOC (NATIVE_SYSCALLS_BASE + 0)
#define SYSCALL_CALLOC (NATIVE_SYSCALLS_BASE + 1)
#define SYSCALL_REALLOC (NATIVE_SYSCALLS_BASE + 2)
#define SYSCALL_FREE (NATIVE_SYSCALLS_BASE + 3)
#define SYSCALL_MEMINFO (NATIVE_SYSCALLS_BASE + 4)

#define SYSCALL_MEMCPY (NATIVE_SYSCALLS_BASE + 5)
#define SYSCALL_MEMSET (NATIVE_SYSCALLS_BASE + 6)
#define SYSCALL_MEMMOVE (NATIVE_SYSCALLS_BASE + 7)
#define SYSCALL_MEMCMP (NATIVE_SYSCALLS_BASE + 8)

#define SYSCALL_STRLEN (NATIVE_SYSCALLS_BASE + 10)
#define SYSCALL_STRCMP (NATIVE_SYSCALLS_BASE + 11)

#define SYSCALL_BACKTRACE (NATIVE_SYSCALLS_BASE + 19)

#define STR1(x) #x
#define STR(x) STR1(x)

// clang-format off
#define CREATE_SYSCALL(name, syscall_id)                 \
	__asm__(".pushsection .text\n"                  \
			".global " #name "\n"            \
			".type " #name ", @function\n"   \
			"" #name ":\n"                   \
			"	li a7, " STR(syscall_id) "\n"       \
			"	ecall\n" \
			"	ret\n"   \
			".popsection .text\n")
#define CREATE_SYSCALL_STRCMP(name, syscall_id) \
	__asm__(".pushsection .text\n"              \
			".global " #name "\n"            \
			".type " #name ", @function\n"   \
			"" #name ":\n"                   \
			"   li a2, 4096\n"               \
			"	li a7, " STR(syscall_id) "\n"       \
			"	ecall\n" \
			"	ret\n"   \
			".popsection .text\n")

// clang-format on

#ifdef ZIG_COMPILER
#define WRAP_FANCY 0
#else
#define WRAP_FANCY 1
#endif

#if !WRAP_FANCY
CREATE_SYSCALL(malloc, SYSCALL_MALLOC);
CREATE_SYSCALL(calloc, SYSCALL_CALLOC);
CREATE_SYSCALL(realloc, SYSCALL_REALLOC);
CREATE_SYSCALL(free, SYSCALL_FREE);
CREATE_SYSCALL(memset, SYSCALL_MEMSET);
CREATE_SYSCALL(memcpy, SYSCALL_MEMCPY);
CREATE_SYSCALL(memmove, SYSCALL_MEMMOVE);
CREATE_SYSCALL(memcmp, SYSCALL_MEMCMP);
CREATE_SYSCALL(strlen, SYSCALL_STRLEN);
CREATE_SYSCALL_STRCMP(strcmp, SYSCALL_STRCMP);
CREATE_SYSCALL(strncmp, SYSCALL_STRCMP);

extern "C" void *__wrap_malloc(size_t size) {
	register void *ret __asm__("a0");
	register size_t a0 __asm__("a0") = size;
	register long syscall_id __asm__("a7") = SYSCALL_MALLOC;

	asm volatile("ecall"
			: "=m"(*(char (*)[size])ret), "=r"(ret)
			: "r"(a0), "r"(syscall_id));
	return ret;
}
extern "C" void __wrap_free(void *ptr) {
	register void *a0 __asm__("a0") = ptr;
	register long syscall_id __asm__("a7") = SYSCALL_FREE;

	asm volatile("ecall"
			:
			: "r"(a0), "r"(syscall_id));
}
#else // WRAP_FANCY

CREATE_SYSCALL(__wrap_malloc, SYSCALL_MALLOC);
CREATE_SYSCALL(__wrap_calloc, SYSCALL_CALLOC);
CREATE_SYSCALL(__wrap_realloc, SYSCALL_REALLOC);
CREATE_SYSCALL(__wrap_free, SYSCALL_FREE);
CREATE_SYSCALL(__wrap_memset, SYSCALL_MEMSET);
CREATE_SYSCALL(__wrap_memcpy, SYSCALL_MEMCPY);
CREATE_SYSCALL(__wrap_memmove, SYSCALL_MEMMOVE);
CREATE_SYSCALL(__wrap_memcmp, SYSCALL_MEMCMP);
CREATE_SYSCALL(__wrap_strlen, SYSCALL_STRLEN);
CREATE_SYSCALL_STRCMP(__wrap_strcmp, SYSCALL_STRCMP);
CREATE_SYSCALL_STRCMP(__wrap_strncmp, SYSCALL_STRCMP);

extern "C" void *__wrap_malloc(size_t size);
extern "C" void __wrap_free(void *ptr);
#endif // WRAP_FANCY

// extern "C" void *__wrap_memset(void *vdest, const int ch, size_t size) {
// 	register char *a0 __asm__("a0") = (char *)vdest;
// 	register int a1 __asm__("a1") = ch;
// 	register size_t a2 __asm__("a2") = size;
// 	register long syscall_id __asm__("a7") = SYSCALL_MEMSET;

// 	asm volatile("ecall"
// 				 : "=m"(*(char(*)[size])a0)
// 				 : "r"(a0), "r"(a1), "r"(a2), "r"(syscall_id));
// 	return vdest;
// }
// extern "C" void *__wrap_memcpy(void *vdest, const void *vsrc, size_t size) {
// 	register char *a0 __asm__("a0") = (char *)vdest;
// 	register const char *a1 __asm__("a1") = (const char *)vsrc;
// 	register size_t a2 __asm__("a2") = size;
// 	register long syscall_id __asm__("a7") = SYSCALL_MEMCPY;

// 	asm volatile("ecall"
// 				 : "=m"(*(char(*)[size])a0), "+r"(a0)
// 				 : "r"(a1), "m"(*(const char(*)[size])a1),
// 				 "r"(a2), "r"(syscall_id));
// 	return vdest;
// }
// extern "C" void *__wrap_memmove(void *vdest, const void *vsrc, size_t size) {
// 	// An assumption is being made here that since vsrc might be
// 	// inside vdest, we cannot assume that vsrc is const anymore.
// 	register char *a0 __asm__("a0") = (char *)vdest;
// 	register char *a1 __asm__("a1") = (char *)vsrc;
// 	register size_t a2 __asm__("a2") = size;
// 	register long syscall_id __asm__("a7") = SYSCALL_MEMMOVE;

// 	asm volatile("ecall"
// 				 : "=m"(*(char(*)[size])a0), "=m"(*(char(*)[size])a1)
// 				 : "r"(a0), "r"(a1), "r"(a2), "r"(syscall_id));
// 	return vdest;
// }
// extern "C" int __wrap_memcmp(const void *m1, const void *m2, size_t size) {
// 	register const char *a0 __asm__("a0") = (const char *)m1;
// 	register const char *a1 __asm__("a1") = (const char *)m2;
// 	register size_t a2 __asm__("a2") = size;
// 	register long syscall_id __asm__("a7") = SYSCALL_MEMCMP;
// 	register int a0_out __asm__("a0");

// 	asm volatile("ecall"
// 				 : "=r"(a0_out)
// 				 : "r"(a0), "m"(*(const char(*)[size])a0),
// 				 "r"(a1), "m"(*(const char(*)[size])a1),
// 				 "r"(a2), "r"(syscall_id));
// 	return a0_out;
// }
// extern "C" size_t __wrap_strlen(const char *str) {
// 	register const char *a0 __asm__("a0") = str;
// 	register size_t a0_out __asm__("a0");
// 	register long syscall_id __asm__("a7") = SYSCALL_STRLEN;

// 	asm volatile("ecall"
// 				 : "=r"(a0_out)
// 				 : "r"(a0), "m"(*(const char(*)[4096])a0), "r"(syscall_id));
// 	return a0_out;
// }
// extern "C" int __wrap_strcmp(const char *str1, const char *str2) {
// 	register const char *a0 __asm__("a0") = str1;
// 	register const char *a1 __asm__("a1") = str2;
// 	register size_t a2 __asm__("a2") = 4096;
// 	register size_t a0_out __asm__("a0");
// 	register long syscall_id __asm__("a7") = SYSCALL_STRCMP;

// 	asm volatile("ecall"
// 				 : "=r"(a0_out)
// 				 : "r"(a0), "m"(*(const char(*)[4096])a0),
// 				 "r"(a1), "m"(*(const char(*)[4096])a1),
// 				 "r"(a2), "r"(syscall_id));
// 	return a0_out;
// }
// extern "C" int __wrap_strncmp(const char *str1, const char *str2, size_t maxlen) {
// 	register const char *a0 __asm__("a0") = str1;
// 	register const char *a1 __asm__("a1") = str2;
// 	register size_t a2 __asm__("a2") = maxlen;
// 	register size_t a0_out __asm__("a0");
// 	register long syscall_id __asm__("a7") = SYSCALL_STRCMP;

// 	asm volatile("ecall"
// 				 : "=r"(a0_out)
// 				 : "r"(a0), "m"(*(const char(*)[maxlen])a0),
// 				 "r"(a1), "m"(*(const char(*)[maxlen])a1),
// 				 "r"(a2), "r"(syscall_id));
// 	return a0_out;
// }

// Fallback implementation of aligned allocations
extern "C" void *memalign(size_t alignment, size_t size) {
	if (alignment <= 16) {
		return __wrap_malloc(size);
	}

	std::array<void *, 16> list;
	size_t i = 0;
	void *result = nullptr;
	for (i = 0; i < list.size(); i++) {
		result = __wrap_malloc(size);
		list[i] = result;
		const bool aligned = ((uintptr_t)result % alignment) == 0;
		if (result && aligned) {
			break;
		} else if (result) {
			__wrap_free(result);
			list[i] = __wrap_malloc(16);
		} else {
			result = nullptr;
			break;
		}
	}
	for (size_t j = 0; j < i; j++) {
		__wrap_free(list[j]);
	}
	return result;
}
extern "C" void *aligned_alloc(size_t alignment, size_t size) {
	return memalign(alignment, size);
}
extern "C" int posix_memalign(void **memptr, size_t alignment, size_t size) {
	void *result = memalign(alignment, size);
	if (result) {
		*memptr = result;
		return 0;
	}
	return 1;
}

void *operator new(size_t size) noexcept(false) {
	return __wrap_malloc(size);
}
void *operator new[](size_t size) noexcept(false) {
	return __wrap_malloc(size);
}
void operator delete(void *ptr) noexcept(true) {
	__wrap_free(ptr);
}
void operator delete[](void *ptr) noexcept(true) {
	__wrap_free(ptr);
}
void *operator new(size_t size, size_t alignment) noexcept(false) {
	return memalign(alignment, size);
}
void *operator new[](size_t size, size_t alignment) noexcept(false) {
	return memalign(alignment, size);
}
void operator delete(void *ptr, size_t alignment) noexcept(true) {
	__wrap_free(ptr);
}
void operator delete[](void *ptr, size_t alignment) noexcept(true) {
	__wrap_free(ptr);
}
