/*************************************************************************/
/*  random_pcg.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/math/math_defs.h"

#include "thirdparty/misc/pcg.h"

class RandomPCG {
	pcg32_random_t pcg;

public:
	static const uint64_t DEFAULT_SEED = 12047754176567800795ULL;
	static const uint64_t DEFAULT_INC = PCG_DEFAULT_INC_64;
	static const uint64_t RANDOM_MAX = 4294967295;

	RandomPCG(uint64_t seed = DEFAULT_SEED, uint64_t inc = PCG_DEFAULT_INC_64);

	_FORCE_INLINE_ void seed(uint64_t seed) { pcg.state = seed; }
	_FORCE_INLINE_ uint64_t get_seed() { return pcg.state; }

	void randomize();
	_FORCE_INLINE_ uint32_t rand() { return pcg32_random_r(&pcg); }
	_FORCE_INLINE_ double randf() { return (double)rand() / (double)RANDOM_MAX; }
	_FORCE_INLINE_ float randd() { return (float)rand() / (float)RANDOM_MAX; }

	double random(double from, double to);
	float random(float from, float to);
	real_t random(int from, int to) { return (real_t)random((real_t)from, (real_t)to); }
};

#endif // RANDOM_PCG_H
