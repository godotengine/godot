/**************************************************************************/
/*  hashfuncs.h                                                           */
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

#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/typedefs.h"

#ifdef _MSC_VER
#include <intrin.h> // Needed for `__umulh` below.
#endif

template <typename F, typename S>
struct Pair;

/**
 * Hashing functions
 */

/**
 * DJB2 Hash function
 * @param C String
 * @return 32-bits hashcode
 */
_FORCE_INLINE_ uint32_t hash_djb2(const char *p_cstr) {
	const unsigned char *chr = (const unsigned char *)p_cstr;
	uint32_t hash = 5381;
	uint32_t c = *chr++;

	while (c) {
		hash = ((hash << 5) + hash) ^ c; /* hash * 33 ^ c */
		c = *chr++;
	}

	return hash;
}

_FORCE_INLINE_ uint32_t hash_djb2_buffer(const uint8_t *p_buff, int p_len, uint32_t p_prev = 5381) {
	uint32_t hash = p_prev;

	for (int i = 0; i < p_len; i++) {
		hash = ((hash << 5) + hash) ^ p_buff[i]; /* hash * 33 + c */
	}

	return hash;
}

_FORCE_INLINE_ uint32_t hash_djb2_one_32(uint32_t p_in, uint32_t p_prev = 5381) {
	return ((p_prev << 5) + p_prev) ^ p_in;
}

/**
 * Thomas Wang's 64-bit to 32-bit Hash function:
 * https://web.archive.org/web/20071223173210/https:/www.concentric.net/~Ttwang/tech/inthash.htm
 *
 * @param p_int - 64-bit unsigned integer key to be hashed
 * @return unsigned 32-bit value representing hashcode
 */
_FORCE_INLINE_ uint32_t hash_one_uint64(const uint64_t p_int) {
	uint64_t v = p_int;
	v = (~v) + (v << 18); // v = (v << 18) - v - 1;
	v = v ^ (v >> 31);
	v = v * 21; // v = (v + (v << 2)) + (v << 4);
	v = v ^ (v >> 11);
	v = v + (v << 6);
	v = v ^ (v >> 22);
	return uint32_t(v);
}

_FORCE_INLINE_ uint64_t hash64_murmur3_64(uint64_t key, uint64_t seed) {
	key ^= seed;
	key ^= key >> 33;
	key *= 0xff51afd7ed558ccd;
	key ^= key >> 33;
	key *= 0xc4ceb9fe1a85ec53;
	key ^= key >> 33;
	return key;
}

#define HASH_MURMUR3_SEED 0x7F07C65
// Murmurhash3 32-bit version.
// All MurmurHash versions are public domain software, and the author disclaims all copyright to their code.

_FORCE_INLINE_ uint32_t hash_murmur3_one_32(uint32_t p_in, uint32_t p_seed = HASH_MURMUR3_SEED) {
	p_in *= 0xcc9e2d51;
	p_in = (p_in << 15) | (p_in >> 17);
	p_in *= 0x1b873593;

	p_seed ^= p_in;
	p_seed = (p_seed << 13) | (p_seed >> 19);
	p_seed = p_seed * 5 + 0xe6546b64;

	return p_seed;
}

_FORCE_INLINE_ uint32_t hash_murmur3_one_64(uint64_t p_in, uint32_t p_seed = HASH_MURMUR3_SEED) {
	p_seed = hash_murmur3_one_32(p_in & 0xFFFFFFFF, p_seed);
	return hash_murmur3_one_32(p_in >> 32, p_seed);
}

uint32_t hash_murmur3_one_float(float p_in, uint32_t p_seed = HASH_MURMUR3_SEED);
uint32_t hash_murmur3_one_double(double p_in, uint32_t p_seed = HASH_MURMUR3_SEED);

_FORCE_INLINE_ uint32_t hash_murmur3_one_real(real_t p_in, uint32_t p_seed = HASH_MURMUR3_SEED) {
#ifdef REAL_T_IS_DOUBLE
	return hash_murmur3_one_double(p_in, p_seed);
#else
	return hash_murmur3_one_float(p_in, p_seed);
#endif
}

_FORCE_INLINE_ uint32_t hash_rotl32(uint32_t x, int8_t r) {
	return (x << r) | (x >> (32 - r));
}

_FORCE_INLINE_ uint32_t hash_fmix32(uint32_t h) {
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;

	return h;
}

uint32_t hash_murmur3_buffer(const void *key, int length, uint32_t seed = HASH_MURMUR3_SEED);

uint32_t hash_djb2_one_float(double p_in, uint32_t p_prev = 5381);
uint64_t hash_djb2_one_float_64(double p_in, uint64_t p_prev = 5381);

_FORCE_INLINE_ uint64_t hash_djb2_one_64(uint64_t p_in, uint64_t p_prev = 5381) {
	return ((p_prev << 5) + p_prev) ^ p_in;
}

template <typename, typename = std::void_t<>>
struct has_hash_method : std::false_type {};

template <typename T>
struct has_hash_method<T, std::void_t<std::is_same<decltype(std::declval<const T>().hash()), uint32_t>>> : std::true_type {};

template <typename T>
constexpr bool has_hash_method_v = has_hash_method<T>::value;

template <typename T, typename = void>
struct HashMapHasherDefaultImpl {
};

struct HashMapHasherDefault {
	template <typename T>
	static _FORCE_INLINE_ uint32_t hash(const T &p_type) {
		return HashMapHasherDefaultImpl<std::decay_t<T>>::hash(p_type);
	}
};

template <typename T>
struct HashMapHasherDefaultImpl<T, std::enable_if_t<has_hash_method_v<T>>> {
	// For self hashing types.
	static _FORCE_INLINE_ uint32_t hash(const T &p_value) {
		return p_value.hash();
	}
};

template <typename T>
struct HashMapHasherDefaultImpl<T *> {
	// For pointer types.
	static _FORCE_INLINE_ uint32_t hash(const T *p_pointer) { return hash_one_uint64((uint64_t)p_pointer); }
};

template <typename T>
struct HashMapHasherDefaultImpl<T, std::enable_if_t<std::is_enum_v<T>>> {
	// For all enums.
	static _FORCE_INLINE_ uint32_t hash(T p_value) {
		return HashMapHasherDefaultImpl<std::underlying_type_t<T>>::hash(static_cast<std::underlying_type_t<T>>(p_value));
	}
};

template <>
struct HashMapHasherDefaultImpl<char *> {
	static _FORCE_INLINE_ uint32_t hash(const char *p_cstr) { return hash_djb2(p_cstr); }
};

template <>
struct HashMapHasherDefaultImpl<wchar_t> {
	static _FORCE_INLINE_ uint32_t hash(const wchar_t p_wchar) { return hash_fmix32(uint32_t(p_wchar)); }
};

template <>
struct HashMapHasherDefaultImpl<char16_t> {
	static _FORCE_INLINE_ uint32_t hash(const char16_t p_uchar) { return hash_fmix32(uint32_t(p_uchar)); }
};

template <>
struct HashMapHasherDefaultImpl<char32_t> {
	static _FORCE_INLINE_ uint32_t hash(const char32_t p_uchar) { return hash_fmix32(uint32_t(p_uchar)); }
};

template <>
struct HashMapHasherDefaultImpl<uint64_t> {
	static _FORCE_INLINE_ uint32_t hash(const uint64_t p_int) { return hash_one_uint64(p_int); }
};

template <>
struct HashMapHasherDefaultImpl<int64_t> {
	static _FORCE_INLINE_ uint32_t hash(const int64_t p_int) { return hash_one_uint64(uint64_t(p_int)); }
};

template <>
struct HashMapHasherDefaultImpl<float> {
	static _FORCE_INLINE_ uint32_t hash(const float p_float) { return hash_murmur3_one_float(p_float); }
};

template <>
struct HashMapHasherDefaultImpl<double> {
	static _FORCE_INLINE_ uint32_t hash(const double p_double) { return hash_murmur3_one_double(p_double); }
};

template <>
struct HashMapHasherDefaultImpl<uint32_t> {
	static _FORCE_INLINE_ uint32_t hash(const uint32_t p_int) { return hash_fmix32(p_int); }
};

template <>
struct HashMapHasherDefaultImpl<int32_t> {
	static _FORCE_INLINE_ uint32_t hash(const int32_t p_int) { return hash_fmix32(uint32_t(p_int)); }
};

template <>
struct HashMapHasherDefaultImpl<uint16_t> {
	static _FORCE_INLINE_ uint32_t hash(const uint16_t p_int) { return hash_fmix32(uint32_t(p_int)); }
};

template <>
struct HashMapHasherDefaultImpl<int16_t> {
	static _FORCE_INLINE_ uint32_t hash(const int16_t p_int) { return hash_fmix32(uint32_t(p_int)); }
};

template <>
struct HashMapHasherDefaultImpl<uint8_t> {
	static _FORCE_INLINE_ uint32_t hash(const uint8_t p_int) { return hash_fmix32(uint32_t(p_int)); }
};

template <>
struct HashMapHasherDefaultImpl<int8_t> {
	static _FORCE_INLINE_ uint32_t hash(const int8_t p_int) { return hash_fmix32(uint32_t(p_int)); }
};

template <typename F, typename S>
struct HashMapHasherDefaultImpl<Pair<F, S>> {
	static _FORCE_INLINE_ uint32_t hash(const Pair<F, S> &p_pair) {
		uint64_t h1 = HashMapHasherDefault::hash(p_pair.first);
		uint64_t h2 = HashMapHasherDefault::hash(p_pair.second);
		return hash_one_uint64((h1 << 32) | h2);
	}
};

template <typename, typename = std::void_t<>>
struct has_is_same_method : std::false_type {};

template <typename T>
struct has_is_same_method<T, std::void_t<std::is_same<decltype(std::declval<const T>().is_same(std::declval<const T>())), uint32_t>>> : std::true_type {};

template <typename T>
constexpr bool has_is_same_method_v = has_is_same_method<T>::value;

struct HashHasher {
	static _FORCE_INLINE_ uint32_t hash(const int32_t hash) { return hash; }
	static _FORCE_INLINE_ uint32_t hash(const uint32_t hash) { return hash; }
	static _FORCE_INLINE_ uint64_t hash(const int64_t hash) { return hash; }
	static _FORCE_INLINE_ uint64_t hash(const uint64_t hash) { return hash; }
};

template <typename T, typename = void>
struct HashMapComparatorDefault {
	static bool compare(const T &p_lhs, const T &p_rhs) {
		return p_lhs == p_rhs;
	}
};

template <>
struct HashMapComparatorDefault<float> {
	static bool compare(const float &p_lhs, const float &p_rhs) {
		return Math::is_same(p_lhs, p_rhs);
	}
};

template <>
struct HashMapComparatorDefault<double> {
	static bool compare(const double &p_lhs, const double &p_rhs) {
		return Math::is_same(p_lhs, p_rhs);
	}
};

template <typename T>
struct HashMapComparatorDefault<T, std::enable_if_t<has_is_same_method_v<T>>> {
	// For self comparing types.
	static bool compare(const T &p_lhs, const T &p_rhs) {
		return p_lhs.is_same(p_rhs);
	}
};

constexpr uint32_t HASH_TABLE_SIZE_MAX = 29;

inline constexpr uint32_t hash_table_size_primes[HASH_TABLE_SIZE_MAX] = {
	5,
	13,
	23,
	47,
	97,
	193,
	389,
	769,
	1543,
	3079,
	6151,
	12289,
	24593,
	49157,
	98317,
	196613,
	393241,
	786433,
	1572869,
	3145739,
	6291469,
	12582917,
	25165843,
	50331653,
	100663319,
	201326611,
	402653189,
	805306457,
	1610612741,
};

// Computed with elem_i = UINT64_C (0 x FFFFFFFF FFFFFFFF ) / d_i + 1, where d_i is the i-th element of the above array.
inline constexpr uint64_t hash_table_size_primes_inv[HASH_TABLE_SIZE_MAX] = {
	3689348814741910324,
	1418980313362273202,
	802032351030850071,
	392483916461905354,
	190172619316593316,
	95578984837873325,
	47420935922132524,
	23987963684927896,
	11955116055547344,
	5991147799191151,
	2998982941588287,
	1501077717772769,
	750081082979285,
	375261795343686,
	187625172388393,
	93822606204624,
	46909513691883,
	23456218233098,
	11728086747027,
	5864041509391,
	2932024948977,
	1466014921160,
	733007198436,
	366503839517,
	183251896093,
	91625960335,
	45812983922,
	22906489714,
	11453246088
};

/**
 * Fastmod computes ( n mod d ) given the precomputed c much faster than n % d.
 * The implementation of fastmod is based on the following paper by Daniel Lemire et al.
 * Faster Remainder by Direct Computation: Applications to Compilers and Software Libraries
 * https://arxiv.org/abs/1902.01961
 */
static _FORCE_INLINE_ uint32_t fastmod(const uint32_t n, const uint64_t c, const uint32_t d) {
#if defined(_MSC_VER)
	// Returns the upper 64 bits of the product of two 64-bit unsigned integers.
	// This intrinsic function is required since MSVC does not support unsigned 128-bit integers.
#if defined(_M_X64) || defined(_M_ARM64)
	return __umulh(c * n, d);
#else
	// Fallback to the slower method for 32-bit platforms.
	return n % d;
#endif // _M_X64 || _M_ARM64
#else
#ifdef __SIZEOF_INT128__
	// Prevent compiler warning, because we know what we are doing.
	uint64_t lowbits = c * n;
	__extension__ typedef unsigned __int128 uint128;
	return static_cast<uint64_t>(((uint128)lowbits * d) >> 64);
#else
	// Fallback to the slower method if no 128-bit unsigned integer type is available.
	return n % d;
#endif // __SIZEOF_INT128__
#endif // _MSC_VER
}
