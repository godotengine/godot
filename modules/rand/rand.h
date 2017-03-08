/*************************************************************************/
/*  rand.h                                                               */
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
#ifndef RAND_H
#define RAND_H

#include "core/reference.h"


// This is the base class for all random number generators in the Rand module.
// Its main purposes are to (1) define a common interface for all random number
// generators, and (2) provide the common code to generate random numbers
// according to different distributions.
class Rand: public Reference {
	GDCLASS(Rand, Reference);
public:
	virtual ~Rand();

	// Returns the next random number in the sequence. It must be a number
	// between zero and `max_random()`.
	virtual uint64_t random() = 0;

	// Seeds the random number generator.
	virtual void seed(uint64_t seed) = 0;

	// Seeds the RNG with some "unpredictable" value.
	void randomize();

	//
	// Distributions
	//

	// Generates integer numbers between `p_a` and `p_b` (closed interval
	// at both ends).
	int64_t uniform_int(int64_t p_a, int64_t p_b);

	// Generates floating point numbers between `p_a` and `p_b` (closed interval
	// at both ends). Defaults to [0,1]. If only `p_a` is passed, the interval
	// used is [0, p_a].
	double uniform_float(double p_a = NAN, double p_b = NAN);

	// Generates a Boolean with a probability `p_p` of being true (AKA Bernoulli
	// distribution). Defaults to 0.5 (a fair coin toss).
	bool boolean(double p_p = 0.5);

	// Generates a floating point number from a normal (Gaussian) distribution
	// with mean `p_mean` and standart deviation `p_std_dev`.
	double normal(double p_mean = 0.0, double p_std_dev = 1.0);

	// Generates a floating point number from an exponential distribution with
	// mean `p_mean`.
	double exponential(double p_mean);

protected:
	static void _bind_methods();

	// Returns the highest value `random()` will ever return.
	virtual uint64_t max_random() = 0;
};

#endif // RAND_H
