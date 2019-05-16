/*************************************************************************/
/*  random_pcg.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef RANDOM_PCG_H
#define RANDOM_PCG_H

#include <math.h>

#include "core/math/math_defs.h"

#include "thirdparty/misc/pcg.h"

class RandomPCG {
	pcg32_random_t pcg;
	uint64_t current_seed; // seed with this to get the same state
	uint64_t current_inc;

public:
	static const uint64_t DEFAULT_SEED = 12047754176567800795U;
	static const uint64_t DEFAULT_INC = PCG_DEFAULT_INC_64;
	static const uint64_t RANDOM_MAX = 0xFFFFFFFF;

	RandomPCG(uint64_t p_seed = DEFAULT_SEED, uint64_t p_inc = DEFAULT_INC);

	_FORCE_INLINE_ void seed(uint64_t p_seed) {
		current_seed = p_seed;
		pcg32_srandom_r(&pcg, current_seed, current_inc);
	}
	_FORCE_INLINE_ uint64_t get_seed() { return current_seed; }

	void randomize();
	_FORCE_INLINE_ uint32_t rand() {
		current_seed = pcg.state;
		return pcg32_random_r(&pcg);
	}
	_FORCE_INLINE_ double randd() { return (double)rand() / (double)RANDOM_MAX; }
	_FORCE_INLINE_ float randf() { return (float)rand() / (float)RANDOM_MAX; }

	_FORCE_INLINE_ double randfn(double p_mean, double p_deviation) {
		return p_mean + p_deviation * (cos(Math_TAU * randd()) * sqrt(-2.0 * log(randd()))); // Box-Muller transform
	}
	_FORCE_INLINE_ float randfn(float p_mean, float p_deviation) {
		return p_mean + p_deviation * (cos(Math_TAU * randf()) * sqrt(-2.0 * log(randf()))); // Box-Muller transform
	}

	double random(double p_from, double p_to);
	float random(float p_from, float p_to);
	real_t random(int p_from, int p_to) { return (real_t)random((real_t)p_from, (real_t)p_to); }
};

#endif // RANDOM_PCG_H
