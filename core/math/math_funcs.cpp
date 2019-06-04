/*************************************************************************/
/*  math_funcs.cpp                                                       */
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
#include "math_funcs.h"

#include "core/os/os.h"
#include "float.h"
#include <math.h>
uint32_t Math::default_seed = 1;

#define PHI 0x9e3779b9

#if 0
static uint32_t Q[4096];
#endif

uint32_t Math::rand_from_seed(uint32_t *seed) {

#if 1
	uint32_t k;
	uint32_t s = (*seed);
	if (s == 0)
		s = 0x12345987;
	k = s / 127773;
	s = 16807 * (s - k * 127773) - 2836 * k;
	//	if (s < 0)
	//		s += 2147483647;
	(*seed) = s;
	return (s & Math::RANDOM_MAX);
#else
	*seed = *seed * 1103515245 + 12345;
	return (*seed % ((unsigned int)RANDOM_MAX + 1));
#endif
}

void Math::seed(uint32_t x) {
#if 0
	int i;

	Q[0] = x;
	Q[1] = x + PHI;
	Q[2] = x + PHI + PHI;

	for (i = 3; i < 4096; i++)
		Q[i] = Q[i - 3] ^ Q[i - 2] ^ PHI ^ i;
#else
	default_seed = x;
#endif
}

void Math::randomize() {

	OS::Time time = OS::get_singleton()->get_time();
	seed(OS::get_singleton()->get_ticks_usec() * (time.hour + 1) * (time.min + 1) * (time.sec + 1) * rand()); /* *OS::get_singleton()->get_time().sec); // windows doesn't have get_time(), returns always 0 */
}

uint32_t Math::rand() {

	return rand_from_seed(&default_seed) & 0x7FFFFFFF;
}

double Math::randf() {

	return (double)rand() / (double)RANDOM_MAX;
}

double Math::sin(double p_x) {

	return ::sin(p_x);
}

double Math::cos(double p_x) {

	return ::cos(p_x);
}

double Math::tan(double p_x) {

	return ::tan(p_x);
}
double Math::sinh(double p_x) {

	return ::sinh(p_x);
}

double Math::cosh(double p_x) {

	return ::cosh(p_x);
}

double Math::tanh(double p_x) {

	return ::tanh(p_x);
}

double Math::deg2rad(double p_y) {

	return p_y * Math_PI / 180.0;
}

double Math::rad2deg(double p_y) {

	return p_y * 180.0 / Math_PI;
}

double Math::round(double p_val) {

	if (p_val >= 0) {
		return ::floor(p_val + 0.5);
	} else {
		p_val = -p_val;
		return -::floor(p_val + 0.5);
	}
}

double Math::asin(double p_x) {

	return ::asin(p_x);
}

double Math::acos(double p_x) {

	return ::acos(p_x);
}

double Math::atan(double p_x) {

	return ::atan(p_x);
}

double Math::dectime(double p_value, double p_amount, double p_step) {

	float sgn = p_value < 0 ? -1.0 : 1.0;
	float val = absf(p_value);
	val -= p_amount * p_step;
	if (val < 0.0)
		val = 0.0;
	return val * sgn;
}

double Math::atan2(double p_y, double p_x) {

	return ::atan2(p_y, p_x);
}
double Math::sqrt(double p_x) {

	return ::sqrt(p_x);
}

double Math::fmod(double p_x, double p_y) {

	return ::fmod(p_x, p_y);
}

double Math::fposmod(double p_x, double p_y) {

	if (p_x >= 0) {

		return Math::fmod(p_x, p_y);

	} else {

		return p_y - Math::fmod(-p_x, p_y);
	}
}
double Math::floor(double p_x) {

	return ::floor(p_x);
}

double Math::ceil(double p_x) {

	return ::ceil(p_x);
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

	double as = absf(p_step);
	for (int i = 0; i < maxn; i++) {
		if (as >= sd[i]) {
			return i;
		}
	}

	return maxn;
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

		p_value = floor(p_value / p_step + 0.5) * p_step;
	}
	return p_value;
}

bool Math::is_nan(double p_val) {

	return (p_val != p_val);
}

bool Math::is_inf(double p_val) {

#ifdef _MSC_VER
	return !_finite(p_val);
#else
	return isinf(p_val);
#endif
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

double Math::pow(double x, double y) {

	return ::pow(x, y);
}

double Math::log(double x) {

	return ::log(x);
}
double Math::exp(double x) {

	return ::exp(x);
}
