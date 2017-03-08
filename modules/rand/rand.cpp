/*************************************************************************/
/*  rand.cpp                                                             */
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

#include <algorithm>
#include <cassert>
#include "rand.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"


// Including the address of the RNG in the seed makes sure that different RNGs
// get different seeds even in the unlikely event of they being seeded at the
// same time. Regardless of that, for systems with address space layout
// randomization (ASLR), `this` is also a good source of entropy.
void Rand::randomize() {
	seed(OS::get_singleton()->get_ticks_usec()
		^ reinterpret_cast<uint64_t>(this));
}


// The traditional way to do this, using modulo, is biased (especially for
// larger ranges).  We can be more uniform without much extra cost. This
// implementation is based on the one provided with the PCG family of RNGs.
// See https://github.com/imneme/pcg-c-basic/blob/master/pcg_basic.c#L79
int64_t Rand::uniform_int(int64_t p_a, int64_t p_b) {
	if (p_a > p_b)
		std::swap(p_a, p_b);

	const uint64_t bound = p_b - p_a + 1;

	if (bound > max_random())
		ERR_FAIL_V(0);

	const uint64_t threshold = -bound % bound;

	for (;;) {
        uint64_t r = random();
        if (r >= threshold)
            return p_a + r % bound;
	}
}


// Arguably, this is not the best way to generate random floats (see, for example,
// https://readings.owlfolio.org/2007/generating-pseudorandom-floating-point-values/),
// but it should be good enough for our ludic purposes. No need to complicate
// things any further.
double Rand::uniform_float(double p_a, double p_b) {

	if (isnan(p_b)) {
		if (isnan(p_a)) {
			p_a = 0.0;
			p_b = 1.0;
		}
		else {
			p_b = p_a;
			p_a = 0.0;
		}
	}

	if (p_a > p_b)
		std::swap(p_a, p_b);

	const double range = p_b - p_a;
	return (static_cast<double>(random()) / max_random()) * range + p_a;
}


bool Rand::boolean(double p_p) {
	return static_cast<double>(random()) < static_cast<double>(max_random()) * p_p;
}


// This implements an algorithm by Peter John Acklam for computing an
// approximation of the inverse normal cumulative distribution function. See
// http://home.online.no/~pjacklam/notes/invnorm/ (or, if this is not online,
// try the Wayback Machine:
// https://web.archive.org/web/20151110174102/http://home.online.no/~pjacklam/notes/invnorm)
double Rand::normal(double p_mean, double p_std_dev) {
	// Be nice and give reasonable results even for negative standard deviation
	if (p_std_dev < 0)
		p_std_dev = 0;


	// Coefficients in rational approximations
	const double a1 = -3.969683028665376e+01;
	const double a2 =  2.209460984245205e+02;
	const double a3 = -2.759285104469687e+02;
	const double a4 =  1.383577518672690e+02;
	const double a5 = -3.066479806614716e+01;
	const double a6 =  2.506628277459239e+00;

	const double b1 = -5.447609879822406e+01;
	const double b2 =  1.615858368580409e+02;
	const double b3 = -1.556989798598866e+02;
	const double b4 =  6.680131188771972e+01;
	const double b5 = -1.328068155288572e+01;

	const double c1 = -7.784894002430293e-03;
	const double c2 = -3.223964580411365e-01;
	const double c3 = -2.400758277161838e+00;
	const double c4 = -2.549732539343734e+00;
	const double c5 =  4.374664141464968e+00;
	const double c6 =  2.938163982698783e+00;

	const double d1 =  7.784695709041462e-03;
	const double d2 =  3.224671290700398e-01;
	const double d3 =  2.445134137142996e+00;
	const double d4 =  3.754408661907416e+00;

	// Break-points
	const double p_low  = 0.02425;
	const double p_high = 1 - p_low;

    // Input and output variables
    double p = 0.0;
    while (p <= 0.0 || p >= 1.0)
        p = static_cast<double>(random()) / static_cast<double>(max_random());

    double x;

    // Rational approximation for lower region
    if (p < p_low)
    {
        const double q = sqrt(-2 * log(p));
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
    }

    // Rational approximation for central region
    else if (p <= p_high)
    {
       const double q = p - 0.5;
       const double r = q * q;
       x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
    }

    // Rational approximation for upper region
    else
    {
        assert(p > p_high);
        const double q = sqrt(-2 * log(1-p));
        x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
    }

    // There we are
	return x * p_std_dev + p_mean;
}

double Rand::exponential(double p_mean)
{
    const double r01 = static_cast<double>(random()) / static_cast<double>(max_random());
    return -p_mean * log(r01);
}


void Rand::_bind_methods() {
	ClassDB::bind_method(D_METHOD("seed", "seed"), &Rand::seed);
	ClassDB::bind_method("randomize", &Rand::randomize);

	ClassDB::bind_method("random", &Rand::random);
	ClassDB::bind_method("max_random", &Rand::max_random);

	ClassDB::bind_method(D_METHOD("uniform_int", "a", "b"), &Rand::uniform_int);
	ClassDB::bind_method(D_METHOD("uniform_float", "a", "b"), &Rand::uniform_float, DEFVAL(NAN), DEFVAL(NAN));
	ClassDB::bind_method(D_METHOD("boolean", "p"), &Rand::boolean, DEFVAL(0.5));
	ClassDB::bind_method(D_METHOD("normal", "mean", "std_dev"), &Rand::normal, DEFVAL(0.0), DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("exponential", "mean"), &Rand::exponential);
}


Rand::~Rand() { }
