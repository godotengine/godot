/*************************************************************************/
/*  mt19937_64.h                                                         */
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

#ifndef RAND_MT19937_64_H
#define RAND_MT19937_64_H

#include "rand.h"

class RandMT19937_64: public Rand {
	GDCLASS(RandMT19937_64, Rand);

public:
	RandMT19937_64();
	virtual ~RandMT19937_64();
	virtual uint64_t random();
	virtual uint64_t max_random();
	virtual void seed(uint64_t p_seed);

protected:
	static void _bind_methods();

private:
	static const uint64_t w = 64;
	static const uint64_t n = 312;
	static const uint64_t m = 156;
	static const uint64_t r = 31;
	static const uint64_t a = 0xB5026F5AA96619E9;
	static const uint64_t u = 29;
	static const uint64_t d = 0x5555555555555555;
	static const uint64_t s = 17;
	static const uint64_t b = 0x71D67FFFEDA60000;
	static const uint64_t t = 37;
	static const uint64_t c = 0xFFF7EEE000000000;
	static const uint64_t l = 43;
	static const uint64_t f = 6364136223846793005;
	void twist();
	uint64_t x[n]; // the state itself
	uint64_t i;    // index into `x`
};

#endif // RAND_MT19937_64_H
