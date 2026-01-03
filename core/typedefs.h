/**************************************************************************/
/*  typedefs.h                                                            */
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

#pragma once

/**
 * Basic definitions and simple functions to be used everywhere.
 */

// IWYU pragma: always_keep

// Ensure that C++ standard is at least C++17.
// If on MSVC, also ensures that the `Zc:__cplusplus` flag is present.
static_assert(__cplusplus >= 201703L, "Minimum of C++17 required.");

// IWYU pragma: begin_exports

// Include first in case the platform needs to pre-define/include some things.
#include "platform_config.h"

// Should be available everywhere.
#include "core/error/error_list.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

// IWYU pragma: end_exports

#if defined(__has_feature)
#define GD_HAS_FEATURE(m_feature) __has_feature(m_feature)
#else
#define GD_HAS_FEATURE(m_feature) 0
#endif // defined(__has_feature)

#if GD_HAS_FEATURE(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#define ASAN_ENABLED
#endif // GD_HAS_FEATURE(address_sanitizer) || defined(__SANITIZE_ADDRESS__)

#if GD_HAS_FEATURE(leak_sanitizer) || defined(__SANITIZE_LEAKS__)
#define LSAN_ENABLED
#endif // GD_HAS_FEATURE(leak_sanitizer) || defined(__SANITIZE_LEAKS__)

#if GD_HAS_FEATURE(memory_sanitizer) || defined(__SANITIZE_MEMORY__)
#define MSAN_ENABLED
#endif // GD_HAS_FEATURE(memory_sanitizer) || defined(__SANITIZE_MEMORY__)

#if GD_HAS_FEATURE(thread_sanitizer) || defined(__SANITIZE_THREAD__)
#define TSAN_ENABLED
#endif // GD_HAS_FEATURE(thread_sanitizer) || defined(__SANITIZE_THREAD__)

#if GD_HAS_FEATURE(undefined_behavior_sanitizer) || defined(__UNDEFINED_SANITIZER__)
#define UBSAN_ENABLED
#endif // GD_HAS_FEATURE(undefined_behavior_sanitizer) || defined(__UNDEFINED_SANITIZER__)

// Turn argument to string constant:
// https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html#Stringizing
#ifndef _STR
#define _STR(m_x) #m_x
#define _MKSTR(m_x) _STR(m_x)
#endif

// Should always inline no matter what.
#ifndef _ALWAYS_INLINE_
#if defined(__GNUC__)
#define _ALWAYS_INLINE_ __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define _ALWAYS_INLINE_ __forceinline
#else
#define _ALWAYS_INLINE_ inline
#endif
#endif

// Should always inline, except in dev builds because it makes debugging harder,
// or `size_enabled` builds where inlining is actively avoided.
#ifndef _FORCE_INLINE_
#if defined(DEV_ENABLED) || defined(SIZE_EXTRA)
#define _FORCE_INLINE_ inline
#else
#define _FORCE_INLINE_ _ALWAYS_INLINE_
#endif
#endif

// Should never inline.
#ifndef _NO_INLINE_
#if defined(__GNUC__)
#define _NO_INLINE_ __attribute__((noinline))
#elif defined(_MSC_VER)
#define _NO_INLINE_ __declspec(noinline)
#else
#define _NO_INLINE_
#endif
#endif

// In some cases [[nodiscard]] will get false positives,
// we can prevent the warning in specific cases by preceding the call with a cast.
#ifndef _ALLOW_DISCARD_
#define _ALLOW_DISCARD_ (void)
#endif

// Windows badly defines a lot of stuff we'll never use. Undefine it.
#ifdef _WIN32
#undef min // override standard definition
#undef max // override standard definition
#undef ERROR // override (really stupid) wingdi.h standard definition
#undef DELETE // override (another really stupid) winnt.h standard definition
#undef MessageBox // override winuser.h standard definition
#undef Error
#undef OK
#undef CONNECT_DEFERRED // override from Windows SDK, clashes with Object enum
#undef MONO_FONT
#endif

// Make room for our constexpr's below by overriding potential system-specific macros.
#undef SIGN
#undef MIN
#undef MAX
#undef CLAMP

template <typename T>
constexpr const T SIGN(const T m_v) {
	return m_v > 0 ? +1.0f : (m_v < 0 ? -1.0f : 0.0f);
}

template <typename T, typename T2>
constexpr auto MIN(const T m_a, const T2 m_b) {
	return m_a < m_b ? m_a : m_b;
}

template <typename T, typename T2>
constexpr auto MAX(const T m_a, const T2 m_b) {
	return m_a > m_b ? m_a : m_b;
}

template <typename T, typename T2, typename T3>
constexpr auto CLAMP(const T m_a, const T2 m_min, const T3 m_max) {
	return m_a < m_min ? m_min : (m_a > m_max ? m_max : m_a);
}

// Generic swap template.
#ifndef SWAP
#define SWAP(m_x, m_y) std::swap((m_x), (m_y))
#endif // SWAP

// Like std::size, but without requiring any additional includes.
template <typename T, size_t SIZE>
constexpr size_t std_size(const T (&)[SIZE]) {
	return SIZE;
}

/* Functions to handle powers of 2 and shifting. */

// Returns `true` if a positive integer is a power of 2, `false` otherwise.
template <typename T>
inline bool is_power_of_2(const T x) {
	return x && ((x & (x - 1)) == 0);
}

// Function to find the next power of 2 to an integer.
constexpr uint64_t next_power_of_2(uint64_t p_number) {
	if (p_number == 0) {
		return 0;
	}

	--p_number;
	p_number |= p_number >> 1;
	p_number |= p_number >> 2;
	p_number |= p_number >> 4;
	p_number |= p_number >> 8;
	p_number |= p_number >> 16;
	p_number |= p_number >> 32;

	return ++p_number;
}

constexpr uint32_t next_power_of_2(uint32_t p_number) {
	if (p_number == 0) {
		return 0;
	}

	--p_number;
	p_number |= p_number >> 1;
	p_number |= p_number >> 2;
	p_number |= p_number >> 4;
	p_number |= p_number >> 8;
	p_number |= p_number >> 16;

	return ++p_number;
}

// Function to find the previous power of 2 to an integer.
constexpr uint64_t previous_power_of_2(uint64_t p_number) {
	p_number |= p_number >> 1;
	p_number |= p_number >> 2;
	p_number |= p_number >> 4;
	p_number |= p_number >> 8;
	p_number |= p_number >> 16;
	p_number |= p_number >> 32;
	return p_number - (p_number >> 1);
}

constexpr uint32_t previous_power_of_2(uint32_t p_number) {
	p_number |= p_number >> 1;
	p_number |= p_number >> 2;
	p_number |= p_number >> 4;
	p_number |= p_number >> 8;
	p_number |= p_number >> 16;
	return p_number - (p_number >> 1);
}

// Function to find the closest power of 2 to an integer.
constexpr uint64_t closest_power_of_2(uint64_t p_number) {
	uint64_t nx = next_power_of_2(p_number);
	uint64_t px = previous_power_of_2(p_number);
	return (nx - p_number) > (p_number - px) ? px : nx;
}

constexpr uint32_t closest_power_of_2(uint32_t p_number) {
	uint32_t nx = next_power_of_2(p_number);
	uint32_t px = previous_power_of_2(p_number);
	return (nx - p_number) > (p_number - px) ? px : nx;
}

// Get a shift value from a power of 2.
constexpr int32_t get_shift_from_power_of_2(uint64_t p_bits) {
	for (uint64_t i = 0; i < (uint64_t)64; i++) {
		if (p_bits == (uint64_t)((uint64_t)1 << i)) {
			return i;
		}
	}

	return -1;
}

constexpr int32_t get_shift_from_power_of_2(uint32_t p_bits) {
	for (uint32_t i = 0; i < (uint32_t)32; i++) {
		if (p_bits == (uint32_t)((uint32_t)1 << i)) {
			return i;
		}
	}

	return -1;
}

template <typename T>
static _FORCE_INLINE_ T nearest_power_of_2_templated(T p_number) {
	--p_number;

	// The number of operations on x is the base two logarithm
	// of the number of bits in the type. Add three to account
	// for sizeof(T) being in bytes.
	constexpr size_t shift_steps = get_shift_from_power_of_2((uint64_t)sizeof(T)) + 3;

	// If the compiler is smart, it unrolls this loop.
	// If it's dumb, this is a bit slow.
	for (size_t i = 0; i < shift_steps; i++) {
		p_number |= p_number >> (1 << i);
	}

	return ++p_number;
}

// Function to find the nearest (bigger) power of 2 to an integer.
constexpr uint64_t nearest_shift(uint64_t p_number) {
	uint64_t i = 63;
	do {
		i--;
		if (p_number & ((uint64_t)1 << i)) {
			return i + (uint64_t)1;
		}
	} while (i != 0);

	return 0;
}

constexpr uint32_t nearest_shift(uint32_t p_number) {
	uint32_t i = 31;
	do {
		i--;
		if (p_number & ((uint32_t)1 << i)) {
			return i + (uint32_t)1;
		}
	} while (i != 0);

	return 0;
}

// constexpr function to find the floored log2 of a number
template <typename T>
constexpr T floor_log2(T x) {
	return x < 2 ? x : 1 + floor_log2(x >> 1);
}

// Get the number of bits needed to represent the number.
// IE, if you pass in 8, you will get 4.
// If you want to know how many bits are needed to store 8 values however, pass in (8 - 1).
template <typename T>
constexpr T get_num_bits(T x) {
	return floor_log2(x);
}

// Swap 16, 32 and 64 bits value for endianness.
#if defined(__GNUC__)
#define BSWAP16(x) __builtin_bswap16(x)
#define BSWAP32(x) __builtin_bswap32(x)
#define BSWAP64(x) __builtin_bswap64(x)
#elif defined(_MSC_VER)
#define BSWAP16(x) _byteswap_ushort(x)
#define BSWAP32(x) _byteswap_ulong(x)
#define BSWAP64(x) _byteswap_uint64(x)
#else
static inline uint16_t BSWAP16(uint16_t x) {
	return (x >> 8) | (x << 8);
}

static inline uint32_t BSWAP32(uint32_t x) {
	return ((x << 24) | ((x << 8) & 0x00FF0000) | ((x >> 8) & 0x0000FF00) | (x >> 24));
}

static inline uint64_t BSWAP64(uint64_t x) {
	x = (x & 0x00000000FFFFFFFF) << 32 | (x & 0xFFFFFFFF00000000) >> 32;
	x = (x & 0x0000FFFF0000FFFF) << 16 | (x & 0xFFFF0000FFFF0000) >> 16;
	x = (x & 0x00FF00FF00FF00FF) << 8 | (x & 0xFF00FF00FF00FF00) >> 8;
	return x;
}
#endif

// Generic comparator used in Map, List, etc.
template <typename T>
struct Comparator {
	_ALWAYS_INLINE_ bool operator()(const T &p_a, const T &p_b) const { return (p_a < p_b); }
};

// Global lock macro, relies on the static Mutex::_global_mutex.
void _global_lock();
void _global_unlock();

struct _GlobalLock {
	_GlobalLock() { _global_lock(); }
	~_GlobalLock() { _global_unlock(); }
};

#define GLOBAL_LOCK_FUNCTION _GlobalLock _global_lock_;

#if defined(__GNUC__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) x
#define unlikely(x) x
#endif

#if defined(__GNUC__)
#define _PRINTF_FORMAT_ATTRIBUTE_2_0 __attribute__((format(printf, 2, 0)))
#define _PRINTF_FORMAT_ATTRIBUTE_2_3 __attribute__((format(printf, 2, 3)))
#else
#define _PRINTF_FORMAT_ATTRIBUTE_2_0
#define _PRINTF_FORMAT_ATTRIBUTE_2_3
#endif

// This is needed due to a strange OpenGL API that expects a pointer
// type for an argument that is actually an offset.
#define CAST_INT_TO_UCHAR_PTR(ptr) ((uint8_t *)(uintptr_t)(ptr))

// Home-made index sequence trick, so it can be used everywhere without the costly include of std::tuple.
// https://stackoverflow.com/questions/15014096/c-index-of-type-during-variadic-template-expansion
template <size_t... Is>
struct IndexSequence {};

template <size_t N, size_t... Is>
struct BuildIndexSequence : BuildIndexSequence<N - 1, N - 1, Is...> {};

template <size_t... Is>
struct BuildIndexSequence<0, Is...> : IndexSequence<Is...> {};

// Limit the depth of recursive algorithms when dealing with Array/Dictionary
#define MAX_RECURSION 100

// Macro GD_IS_DEFINED() allows to check if a macro is defined. It needs to be defined to anything (say 1) to work.
#define __GDARG_PLACEHOLDER_1 false,
#define __gd_take_second_arg(__ignored, val, ...) val
#define ____gd_is_defined(arg1_or_junk) __gd_take_second_arg(arg1_or_junk true, false)
#define ___gd_is_defined(val) ____gd_is_defined(__GDARG_PLACEHOLDER_##val)
#define GD_IS_DEFINED(x) ___gd_is_defined(x)

// Whether the default value of a type is just all-0 bytes.
// This can most commonly be exploited by using memset for these types instead of loop-construct.
// Trivially constructible types are also zero-constructible.
template <typename T>
struct is_zero_constructible : std::is_trivially_constructible<T> {};

template <typename T>
struct is_zero_constructible<const T> : is_zero_constructible<T> {};

template <typename T>
struct is_zero_constructible<volatile T> : is_zero_constructible<T> {};

template <typename T>
struct is_zero_constructible<const volatile T> : is_zero_constructible<T> {};

template <typename T>
inline constexpr bool is_zero_constructible_v = is_zero_constructible<T>::value;

// Warning suppression helper macros.
#if defined(__clang__)
#define GODOT_CLANG_PRAGMA(m_content) _Pragma(#m_content)
#define GODOT_CLANG_WARNING_PUSH GODOT_CLANG_PRAGMA(clang diagnostic push)
#define GODOT_CLANG_WARNING_IGNORE(m_warning) GODOT_CLANG_PRAGMA(clang diagnostic ignored m_warning)
#define GODOT_CLANG_WARNING_POP GODOT_CLANG_PRAGMA(clang diagnostic pop)
#define GODOT_CLANG_WARNING_PUSH_AND_IGNORE(m_warning) GODOT_CLANG_WARNING_PUSH GODOT_CLANG_WARNING_IGNORE(m_warning)
#else
#define GODOT_CLANG_PRAGMA(m_content)
#define GODOT_CLANG_WARNING_PUSH
#define GODOT_CLANG_WARNING_IGNORE(m_warning)
#define GODOT_CLANG_WARNING_POP
#define GODOT_CLANG_WARNING_PUSH_AND_IGNORE(m_warning)
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define GODOT_GCC_PRAGMA(m_content) _Pragma(#m_content)
#define GODOT_GCC_WARNING_PUSH GODOT_GCC_PRAGMA(GCC diagnostic push)
#define GODOT_GCC_WARNING_IGNORE(m_warning) GODOT_GCC_PRAGMA(GCC diagnostic ignored m_warning)
#define GODOT_GCC_WARNING_POP GODOT_GCC_PRAGMA(GCC diagnostic pop)
#define GODOT_GCC_WARNING_PUSH_AND_IGNORE(m_warning) GODOT_GCC_WARNING_PUSH GODOT_GCC_WARNING_IGNORE(m_warning)
#else
#define GODOT_GCC_PRAGMA(m_content)
#define GODOT_GCC_WARNING_PUSH
#define GODOT_GCC_WARNING_IGNORE(m_warning)
#define GODOT_GCC_WARNING_POP
#define GODOT_GCC_WARNING_PUSH_AND_IGNORE(m_warning)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define GODOT_MSVC_PRAGMA(m_command) __pragma(m_command)
#define GODOT_MSVC_WARNING_PUSH GODOT_MSVC_PRAGMA(warning(push))
#define GODOT_MSVC_WARNING_IGNORE(m_warning) GODOT_MSVC_PRAGMA(warning(disable : m_warning))
#define GODOT_MSVC_WARNING_POP GODOT_MSVC_PRAGMA(warning(pop))
#define GODOT_MSVC_WARNING_PUSH_AND_IGNORE(m_warning) GODOT_MSVC_WARNING_PUSH GODOT_MSVC_WARNING_IGNORE(m_warning)
#else
#define GODOT_MSVC_PRAGMA(m_command)
#define GODOT_MSVC_WARNING_PUSH
#define GODOT_MSVC_WARNING_IGNORE(m_warning)
#define GODOT_MSVC_WARNING_POP
#define GODOT_MSVC_WARNING_PUSH_AND_IGNORE(m_warning)
#endif

template <typename T, typename = void>
struct is_fully_defined : std::false_type {};

template <typename T>
struct is_fully_defined<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

template <typename T>
constexpr bool is_fully_defined_v = is_fully_defined<T>::value;

#ifndef SCU_BUILD_ENABLED
/// Enforces the requirement that a class is not fully defined.
/// This can be used to reduce include coupling and keep compile times low.
/// The check must be made at the top of the corresponding .cpp file of a header.
#define STATIC_ASSERT_INCOMPLETE_TYPE(m_keyword, m_type) \
	m_keyword m_type;                                    \
	static_assert(!is_fully_defined_v<m_type>, #m_type " was unexpectedly fully defined. Please check the include hierarchy of '" __FILE__ "' and remove includes that resolve the " #m_keyword ".");
#else
#define STATIC_ASSERT_INCOMPLETE_TYPE(m_keyword, m_type)
#endif

#define _GD_VARNAME_CONCAT_B_(m_ignore, m_name) m_name
#define _GD_VARNAME_CONCAT_A_(m_a, m_b, m_c) _GD_VARNAME_CONCAT_B_(hello there, m_a##m_b##m_c)
#define _GD_VARNAME_CONCAT_(m_a, m_b, m_c) _GD_VARNAME_CONCAT_A_(m_a, m_b, m_c)
#define GD_UNIQUE_NAME(m_name) _GD_VARNAME_CONCAT_(m_name, _, __COUNTER__)
