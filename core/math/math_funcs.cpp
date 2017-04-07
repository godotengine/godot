/*************************************************************************/
/*  math_funcs.cpp                                                       */
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
#include "math_funcs.h"
#include "core/os/os.h"

pcg32_random_t Math::default_pcg = { 1, PCG_DEFAULT_INC_64 };

#define PHI 0x9e3779b9

#if 0
static uint32_t Q[4096];
#endif

// TODO: we should eventually expose pcg.inc too
uint32_t Math::rand_from_seed(uint64_t *seed) {
	pcg32_random_t pcg = { *seed, PCG_DEFAULT_INC_64 };
	uint32_t r = pcg32_random_r(&pcg);
	*seed = pcg.state;
	return r;
}

void Math::seed(uint64_t x) {
	default_pcg.state = x;
}

void Math::randomize() {

	OS::Time time = OS::get_singleton()->get_time();
	seed(OS::get_singleton()->get_ticks_usec() * (time.hour + 1) * (time.min + 1) * (time.sec + 1) * rand()); // TODO: can be simplified.
}

uint32_t Math::rand() {
	return pcg32_random_r(&default_pcg);
}

int Math::step_decimals(double p_step) {
	static const int maxn = 9;
	static const double sd[maxn] = {
		0.9999, // somehow compensate for floating point error
		0.09999,
		0.009999,
		0.0009999,
		0.00009999,
		0.000009999,
		0.0000009999,
		0.00000009999,
		0.000000009999
	};

	double as = Math::abs(p_step);
	for (int i = 0; i < maxn; i++) {
		if (as >= sd[i]) {
			return i;
		}
	}

	return maxn;
}

double Math::dectime(double p_value, double p_amount, double p_step) {
	double sgn = p_value < 0 ? -1.0 : 1.0;
	double val = Math::abs(p_value);
	val -= p_amount * p_step;
	if (val < 0.0)
		val = 0.0;
	return val * sgn;
}

double Math::ease(double p_x, double p_c) {
	if (p_x < 0)
		p_x = 0;
	else if (p_x > 1.0)
		p_x = 1.0;
	if (p_c > 0) {
		if (p_c < 1.0) {
			return 1.0 - Math::pow(1.0 - p_x, 1.0 / p_c);
		} else {
			return Math::pow(p_x, p_c);
		}
	} else if (p_c < 0) {
		//inout ease

		if (p_x < 0.5) {
			return Math::pow(p_x * 2.0, -p_c) * 0.5;
		} else {
			return (1.0 - Math::pow(1.0 - (p_x - 0.5) * 2.0, -p_c)) * 0.5 + 0.5;
		}
	} else
		return 0; // no ease (raw)
}

double Math::stepify(double p_value, double p_step) {
	if (p_step != 0) {
		p_value = Math::floor(p_value / p_step + 0.5) * p_step;
	}
	return p_value;
}

uint32_t Math::larger_prime(uint32_t p_val) {

	static const uint32_t primes[] = {
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
		0,
	};

	int idx = 0;
	while (true) {

		ERR_FAIL_COND_V(primes[idx] == 0, 0);
		if (primes[idx] > p_val)
			return primes[idx];
		idx++;
	}

	return 0;
}

double Math::random(double from, double to) {
	unsigned int r = Math::rand();
	double ret = (double)r / (double)RANDOM_MAX;
	return (ret) * (to - from) + from;
}

float Math::random(float from, float to) {
	unsigned int r = Math::rand();
	float ret = (float)r / (float)RANDOM_MAX;
	return (ret) * (to - from) + from;
}
