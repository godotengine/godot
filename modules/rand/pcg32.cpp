/*************************************************************************/
/*  pcg32.cpp                                                            */
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

//
// The code in this file is derived from the C implementation originally found
// at https://github.com/imneme/pcg-c-basic/
//
// That code is Copyright 2014 Melissa O'Neill <oneill@pcg-random.org> and was
// originally licensed under the Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0).
//
// For additional information about the PCG random number generation scheme,
// including its license and other licensing options, visit
//
//     http://www.pcg-random.org
//

#include "pcg32.h"
#include <limits>


RandPCG32::RandPCG32()
	: state(0x853c49e6748fea9bULL), inc(0xda3e39cb94b95bdbULL) {
	// Nothing here.
}


RandPCG32::~RandPCG32() {
	// Nothing here.
}


uint64_t RandPCG32::random() {
	const uint64_t oldstate = state;
	state = oldstate * 6364136223846793005ULL + inc;
	const uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	const uint32_t rot = oldstate >> 59u;
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}


uint64_t RandPCG32::max_random() {
	return std::numeric_limits<uint32_t>::max();
}


void RandPCG32::seed(uint64_t p_seed) {
	seed_2(p_seed, 0x6d1f1ce5ca5cadedULL);
}


// PCG-specific seeding, just in case someone wants to use it.
void RandPCG32::seed_2(uint64_t p_state, uint64_t p_seq) {
	state = 0U;
    inc = (p_seq << 1u) | 1u;
    random();
    state += p_state;
	random();
}

void RandPCG32::_bind_methods() {
	ClassDB::bind_method(D_METHOD("seed_2", "state", "seq"), &RandPCG32::seed_2);
}
