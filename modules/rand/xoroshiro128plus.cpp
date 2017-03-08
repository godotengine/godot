/*************************************************************************/
/*  xoroshiro128plus.cpp                                                 */
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
#include "xoroshiro128plus.h"
#include "splitmix64.h"

// This code is derivative of the public domain C code by David Blackman and
// Sebastiano Vigna, available at
// http://xoroshiro.di.unimi.it/xoroshiro128plus.c.

namespace {
	// Rotates x left by k bits.
	uint64_t rotl(uint64_t x, unsigned k ) {
		return (x << k) | (x >> (64 - k));
	}
}


RandXoroshiro128Plus::RandXoroshiro128Plus() {
	seed(0x537A73BC);
}


RandXoroshiro128Plus::~RandXoroshiro128Plus() {
	// Nothing here.
}


uint64_t RandXoroshiro128Plus::random() {
	uint64_t result = state[0] + state[1];

	state[1] ^= state[0];
	state[0] = rotl(state[0], 55) ^ state[1] ^ (state[1] << 14);
	state[1] = rotl(state[1], 36);

	return result;
}


uint64_t RandXoroshiro128Plus::max_random() {
	return std::numeric_limits<uint64_t>::max();
}


void RandXoroshiro128Plus::seed(uint64_t p_seed) {
	// Use a SplitMix64 RNG to generate two seed values from `p_seed`
	RandSplitMix64 rng;
	rng.seed(p_seed);
	state[0] = rng.random();
	state[1] = rng.random();
}


void RandXoroshiro128Plus::_bind_methods() {
	// All exported methods are declared in the superclass.
}
