/*************************************************************************/
/*  hashfuncs.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef HASHFUNCS_H
#define HASHFUNCS_H

#include "math_defs.h"
#include "math_funcs.h"
#include "typedefs.h"

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

	while ((c = *chr++))
		hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

	return hash;
}

static inline uint32_t hash_djb2_buffer(const uint8_t *p_buff, int p_len, uint32_t p_prev = 5381) {

	uint32_t hash = p_prev;

	for (int i = 0; i < p_len; i++)
		hash = ((hash << 5) + hash) + p_buff[i]; /* hash * 33 + c */

	return hash;
}

static inline uint32_t hash_djb2_one_32(uint32_t p_in, uint32_t p_prev = 5381) {

	return ((p_prev << 5) + p_prev) + p_in;
}

static inline uint32_t hash_one_uint64(const uint64_t p_int) {
	uint64_t v = p_int;
	v = (~v) + (v << 18); // v = (v << 18) - v - 1;
	v = v ^ (v >> 31);
	v = v * 21; // v = (v + (v << 2)) + (v << 4);
	v = v ^ (v >> 11);
	v = v + (v << 6);
	v = v ^ (v >> 22);
	return (int)v;
}

static inline uint32_t hash_djb2_one_float(float p_in, uint32_t p_prev = 5381) {
	union {
		float f;
		uint32_t i;
	} u;

	// Normalize +/- 0.0 and NaN values so they hash the same.
	if (p_in == 0.0f)
		u.f = 0.0;
	else if (Math::is_nan(p_in))
		u.f = Math_NAN;
	else
		u.f = p_in;

	return ((p_prev << 5) + p_prev) + u.i;
}

// Overload for real_t size changes
static inline uint32_t hash_djb2_one_float(double p_in, uint32_t p_prev = 5381) {
	union {
		double d;
		uint64_t i;
	} u;

	// Normalize +/- 0.0 and NaN values so they hash the same.
	if (p_in == 0.0f)
		u.d = 0.0;
	else if (Math::is_nan(p_in))
		u.d = Math_NAN;
	else
		u.d = p_in;

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

#endif
