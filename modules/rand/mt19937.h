/*************************************************************************/
/*  mt19937.h                                                            */
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

#ifndef RAND_MT19937_H
#define RAND_MT19937_H

#include "rand.h"

class RandMT19937: public Rand {
	GDCLASS(RandMT19937, Rand);
public:
	RandMT19937();
	virtual ~RandMT19937();
	virtual uint64_t random();
	virtual uint64_t max_random();
	virtual void seed(uint64_t p_seed);

protected:
	static void _bind_methods();

private:
	static const uint32_t w = 32;
	static const uint32_t n = 624;
	static const uint32_t m = 397;
	static const uint32_t r = 31;
	static const uint32_t a = 0X9908B0DF;
	static const uint32_t u = 11;
	static const uint32_t d = 0XFFFFFFFF;
	static const uint32_t s = 7;
	static const uint32_t b = 0X9D2C5680;
	static const uint32_t t = 15;
	static const uint32_t c = 0XEFC60000;
	static const uint32_t l = 18;
	static const uint32_t f = 1812433253;
	void twist();
	uint32_t x[n]; // the state itself
	uint32_t i;    // index into `x`

};

#endif // RAND_MT19937_H
