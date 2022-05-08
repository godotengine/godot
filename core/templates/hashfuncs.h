/*************************************************************************/
/*  hashfuncs.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef HASHFUNCS_H
#define HASHFUNCS_H

#include "core/math/aabb.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/math/rect2.h"
#include "core/math/rect2i.h"
#include "core/math/vector2.h"
#include "core/math/vector2i.h"
#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/object/object_id.h"
#include "core/string/node_path.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/rid.h"
#include "core/typedefs.h"

/**
 * Hashing functions
 */

/**
 * DJB2 Hash function
 * @param C String
 * @return 32-bits hashcode
 */
static inline uint32_t hash_djb2(const char *p_cstr) {
	const unsigned char *chr = (const unsigned char *)p_cstr;
	uint32_t hash = 5381;
	uint32_t c;

	while ((c = *chr++)) {
		hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
	}

	return hash;
}

static inline uint32_t hash_djb2_buffer(const uint8_t *p_buff, int p_len, uint32_t p_prev = 5381) {
	uint32_t hash = p_prev;

	for (int i = 0; i < p_len; i++) {
		hash = ((hash << 5) + hash) + p_buff[i]; /* hash * 33 + c */
	}

	return hash;
}

static inline uint32_t hash_djb2_one_32(uint32_t p_in, uint32_t p_prev = 5381) {
	return ((p_prev << 5) + p_prev) + p_in;
}

/**
 * Thomas Wang's 64-bit to 32-bit Hash function:
 * https://web.archive.org/web/20071223173210/https:/www.concentric.net/~Ttwang/tech/inthash.htm
 *
 * @param p_int - 64-bit unsigned integer key to be hashed
 * @return unsigned 32-bit value representing hashcode
 */
static inline uint32_t hash_one_uint64(const uint64_t p_int) {
	uint64_t v = p_int;
	v = (~v) + (v << 18); // v = (v << 18) - v - 1;
	v = v ^ (v >> 31);
	v = v * 21; // v = (v + (v << 2)) + (v << 4);
	v = v ^ (v >> 11);
	v = v + (v << 6);
	v = v ^ (v >> 22);
	return uint32_t(v);
}

static inline uint32_t hash_djb2_one_float(double p_in, uint32_t p_prev = 5381) {
	union {
		double d;
		uint64_t i;
	} u;

	// Normalize +/- 0.0 and NaN values so they hash the same.
	if (p_in == 0.0f) {
		u.d = 0.0;
	} else if (Math::is_nan(p_in)) {
		u.d = NAN;
	} else {
		u.d = p_in;
	}

	return ((p_prev << 5) + p_prev) + hash_one_uint64(u.i);
}

template <class T>
static inline uint32_t make_uint32_t(T p_in) {
	union {
		T t;
		uint32_t _u32;
	} _u;
	_u._u32 = 0;
	_u.t = p_in;
	return _u._u32;
}

static inline uint64_t hash_djb2_one_float_64(double p_in, uint64_t p_prev = 5381) {
	union {
		double d;
		uint64_t i;
	} u;

	// Normalize +/- 0.0 and NaN values so they hash the same.
	if (p_in == 0.0f) {
		u.d = 0.0;
	} else if (Math::is_nan(p_in)) {
		u.d = NAN;
	} else {
		u.d = p_in;
	}

	return ((p_prev << 5) + p_prev) + u.i;
}

static inline uint64_t hash_djb2_one_64(uint64_t p_in, uint64_t p_prev = 5381) {
	return ((p_prev << 5) + p_prev) + p_in;
}

template <class T>
static inline uint64_t make_uint64_t(T p_in) {
	union {
		T t;
		uint64_t _u64;
	} _u;
	_u._u64 = 0; // in case p_in is smaller

	_u.t = p_in;
	return _u._u64;
}

struct HashMapHasherDefault {
	static _FORCE_INLINE_ uint32_t hash(const String &p_string) { return p_string.hash(); }
	static _FORCE_INLINE_ uint32_t hash(const char *p_cstr) { return hash_djb2(p_cstr); }
	static _FORCE_INLINE_ uint32_t hash(const uint64_t p_int) { return hash_one_uint64(p_int); }
	static _FORCE_INLINE_ uint32_t hash(const ObjectID &p_id) { return hash_one_uint64(p_id); }

	static _FORCE_INLINE_ uint32_t hash(const int64_t p_int) { return hash(uint64_t(p_int)); }
	static _FORCE_INLINE_ uint32_t hash(const float p_float) { return hash_djb2_one_float(p_float); }
	static _FORCE_INLINE_ uint32_t hash(const double p_double) { return hash_djb2_one_float(p_double); }
	static _FORCE_INLINE_ uint32_t hash(const uint32_t p_int) { return p_int; }
	static _FORCE_INLINE_ uint32_t hash(const int32_t p_int) { return (uint32_t)p_int; }
	static _FORCE_INLINE_ uint32_t hash(const uint16_t p_int) { return p_int; }
	static _FORCE_INLINE_ uint32_t hash(const int16_t p_int) { return (uint32_t)p_int; }
	static _FORCE_INLINE_ uint32_t hash(const uint8_t p_int) { return p_int; }
	static _FORCE_INLINE_ uint32_t hash(const int8_t p_int) { return (uint32_t)p_int; }
	static _FORCE_INLINE_ uint32_t hash(const wchar_t p_wchar) { return (uint32_t)p_wchar; }
	static _FORCE_INLINE_ uint32_t hash(const char16_t p_uchar) { return (uint32_t)p_uchar; }
	static _FORCE_INLINE_ uint32_t hash(const char32_t p_uchar) { return (uint32_t)p_uchar; }
	static _FORCE_INLINE_ uint32_t hash(const RID &p_rid) { return hash_one_uint64(p_rid.get_id()); }

	static _FORCE_INLINE_ uint32_t hash(const StringName &p_string_name) { return p_string_name.hash(); }
	static _FORCE_INLINE_ uint32_t hash(const NodePath &p_path) { return p_path.hash(); }

	static _FORCE_INLINE_ uint32_t hash(const Vector2i &p_vec) {
		uint32_t h = hash_djb2_one_32(p_vec.x);
		return hash_djb2_one_32(p_vec.y, h);
	}
	static _FORCE_INLINE_ uint32_t hash(const Vector3i &p_vec) {
		uint32_t h = hash_djb2_one_32(p_vec.x);
		h = hash_djb2_one_32(p_vec.y, h);
		return hash_djb2_one_32(p_vec.z, h);
	}

	static _FORCE_INLINE_ uint32_t hash(const Vector2 &p_vec) {
		uint32_t h = hash_djb2_one_float(p_vec.x);
		return hash_djb2_one_float(p_vec.y, h);
	}
	static _FORCE_INLINE_ uint32_t hash(const Vector3 &p_vec) {
		uint32_t h = hash_djb2_one_float(p_vec.x);
		h = hash_djb2_one_float(p_vec.y, h);
		return hash_djb2_one_float(p_vec.z, h);
	}

	static _FORCE_INLINE_ uint32_t hash(const Rect2i &p_rect) {
		uint32_t h = hash_djb2_one_32(p_rect.position.x);
		h = hash_djb2_one_32(p_rect.position.y, h);
		h = hash_djb2_one_32(p_rect.size.x, h);
		return hash_djb2_one_32(p_rect.size.y, h);
	}

	static _FORCE_INLINE_ uint32_t hash(const Rect2 &p_rect) {
		uint32_t h = hash_djb2_one_float(p_rect.position.x);
		h = hash_djb2_one_float(p_rect.position.y, h);
		h = hash_djb2_one_float(p_rect.size.x, h);
		return hash_djb2_one_float(p_rect.size.y, h);
	}

	static _FORCE_INLINE_ uint32_t hash(const AABB &p_aabb) {
		uint32_t h = hash_djb2_one_float(p_aabb.position.x);
		h = hash_djb2_one_float(p_aabb.position.y, h);
		h = hash_djb2_one_float(p_aabb.position.z, h);
		h = hash_djb2_one_float(p_aabb.size.x, h);
		h = hash_djb2_one_float(p_aabb.size.y, h);
		return hash_djb2_one_float(p_aabb.size.z, h);
	}

	//static _FORCE_INLINE_ uint32_t hash(const void* p_ptr)  { return uint32_t(uint64_t(p_ptr))*(0x9e3779b1L); }
};

template <typename T>
struct HashMapComparatorDefault {
	static bool compare(const T &p_lhs, const T &p_rhs) {
		return p_lhs == p_rhs;
	}

	bool compare(const float &p_lhs, const float &p_rhs) {
		return (p_lhs == p_rhs) || (Math::is_nan(p_lhs) && Math::is_nan(p_rhs));
	}

	bool compare(const double &p_lhs, const double &p_rhs) {
		return (p_lhs == p_rhs) || (Math::is_nan(p_lhs) && Math::is_nan(p_rhs));
	}
};

constexpr uint32_t HASH_TABLE_SIZE_MAX = 29;

const uint32_t hash_table_size_primes[HASH_TABLE_SIZE_MAX] = {
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

#endif // HASHFUNCS_H
