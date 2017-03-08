/*************************************************************************/
/*  mt19937_64.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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

#include <limits>
#include "mt19937_64.h"

// This implementation is based on the implemention from the Boost Libraries
// 1.63.0. This is the copyright information for the original Boost code:
//
// Copyright Jens Maurer 2000-2001
// Copyright Steven Watanabe 2010
//
// Distributed under the Boost Software License, Version 1.0. (See
// http://www.boost.org/LICENSE_1_0.txt)


RandMT19937_64::RandMT19937_64() {
	seed(5489); // Default seed used by Yakuji Nishimura and Makoto Matsumoto
}


RandMT19937_64::~RandMT19937_64() {
	// Nothing here.
}


uint64_t RandMT19937_64::random() {
	if(i == n)
		twist();

	uint64_t z = x[i];
	++i;
	z ^= ((z >> u) & d);
	z ^= ((z << s) & b);
	z ^= ((z << t) & c);
	z ^= (z >> l);

	return z;
}


uint64_t RandMT19937_64::max_random() {
	return std::numeric_limits<uint64_t>::max();
}


void RandMT19937_64::seed(uint64_t p_seed) {
	const uint64_t mask = max_random();
	x[0] = p_seed & mask;
	for (i = 1; i < n; i++) {
		x[i] = (f * (x[i-1] ^ (x[i-1] >> (w-2))) + i) & mask;
	}

	// Normalize state
	const uint64_t upper_mask = (~static_cast<uint64_t>(0)) << r;
	const uint64_t lower_mask = ~upper_mask;
	uint64_t y0 = x[m-1] ^ x[n-1];
	if(y0 & (static_cast<uint64_t>(1) << (w-1))) {
		y0 = ((y0 ^ a) << 1) | 1;
	} else {
		y0 = y0 << 1;
	}
	x[0] = (x[0] & upper_mask) | (y0 & lower_mask);

	// fix up the state if it's all zeroes.
	for (size_t j = 0; j < n; ++j) {
		if(x[j] != 0) return;
	}
	x[0] = static_cast<uint64_t>(1) << (w-1);
}


void RandMT19937_64::twist() {
	const uint64_t upper_mask = (~static_cast<uint64_t>(0)) << r;
	const uint64_t lower_mask = ~upper_mask;

	const uint64_t unroll_factor = 6;
	const uint64_t unroll_extra1 = (n-m) % unroll_factor;
	const uint64_t unroll_extra2 = (m-1) % unroll_factor;

	// split loop to avoid costly modulo operations
	for (size_t j = 0; j < n-m-unroll_extra1; j++) {
		uint64_t y = (x[j] & upper_mask) | (x[j+1] & lower_mask);
		x[j] = x[j+m] ^ (y >> 1) ^ ((x[j+1]&1) * a);
	}

	for (std::size_t j = n-m-unroll_extra1; j < n-m; j++) {
		uint64_t y = (x[j] & upper_mask) | (x[j+1] & lower_mask);
		x[j] = x[j+m] ^ (y >> 1) ^ ((x[j+1]&1) * a);
	}

	for (std::size_t j = n-m; j < n-1-unroll_extra2; j++) {
		uint64_t y = (x[j] & upper_mask) | (x[j+1] & lower_mask);
		x[j] = x[j-(n-m)] ^ (y >> 1) ^ ((x[j+1]&1) * a);
	}

	for (std::size_t j = n-1-unroll_extra2; j < n-1; j++) {
		uint64_t y = (x[j] & upper_mask) | (x[j+1] & lower_mask);
		x[j] = x[j-(n-m)] ^ (y >> 1) ^ ((x[j+1]&1) * a);
	}

	// last iteration
	uint64_t y = (x[n-1] & upper_mask) | (x[0] & lower_mask);
	x[n-1] = x[m-1] ^ (y >> 1) ^ ((x[0]&1) * a);
	i = 0;
}

void RandMT19937_64::_bind_methods() {
	// All exported methods are declared in the superclass.
}
