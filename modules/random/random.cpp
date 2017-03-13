/*************************************************************************/
/*  random.cpp                                                           */
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

#include "random.h"


Random::Random() {
	pcg64_srandom_r(&rng, PCG_128BIT_CONSTANT(0ULL, 42ULL),
		PCG_128BIT_CONSTANT(0ULL, 54ULL));
}


Random::~Random() {
	// Nothing here.
}


int64_t Random::random() {
	return pcg64_random_r(&rng) >> 1;
}


int64_t Random::max_random() {
	return (1ULL << 63) - 1;
}


void Random::seed(int64_t p_seed) {
	seed_2(p_seed, 0x6d1f1ce5ca5cadedULL);
}


// PCG-specific seeding, just in case someone wants to use it.
void Random::seed_2(int64_t p_state, int64_t p_seq) {
	pcg64_srandom_r(&rng, PCG_128BIT_CONSTANT(0ULL, p_state),
		PCG_128BIT_CONSTANT(0ULL, p_seq));
}

void Random::_bind_methods() {
	ClassDB::bind_method(D_METHOD("seed_2", "state", "seq"), &Random::seed_2);
}
