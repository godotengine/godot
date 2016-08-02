/*************************************************************************/
/*  math_funcs.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef MATH_FUNCS_H
#define MATH_FUNCS_H

#include "typedefs.h"
#include "math_defs.h"

#ifndef NO_MATH_H
#include "math.h"
#endif

class Math {


	static uint32_t default_seed;
public:
	Math() {}; // useless to instance

	enum {
		RANDOM_MAX=2147483647L
	};

	static double sin(double p_x);
	static double cos(double p_x);
	static double tan(double p_x);
	static double sinh(double p_x);
	static double cosh(double p_x);
	static double tanh(double p_x);
	static double asin(double p_x);
	static double acos(double p_x);
	static double atan(double p_x);
	static double atan2(double p_y, double p_x);
	static double deg2rad(double p_y);
	static double rad2deg(double p_y);
	static double sqrt(double p_x);
	static double fmod(double p_x,double p_y);
	static double fposmod(double p_x,double p_y);
	static uint32_t rand_from_seed(uint32_t *seed);
	static double floor(double p_x);
	static double ceil(double p_x);
	static double ease(double p_x, double p_c);
	static int step_decimals(double p_step);
	static double stepify(double p_value,double p_step);
	static void seed(uint32_t x=0);
	static void randomize();
	static uint32_t larger_prime(uint32_t p_val);
	static double dectime(double p_value,double p_amount, double p_step);


	static inline double linear2db(double p_linear) {

		return Math::log( p_linear ) * 8.6858896380650365530225783783321;
	}

	static inline double db2linear(double p_db) {

		return Math::exp( p_db * 0.11512925464970228420089957273422 );
	}

	static bool is_nan(double p_val);
	static bool is_inf(double p_val);



	static uint32_t rand();
	static double randf();

	static double round(double p_val);

	static double random(double from, double to);


	static _FORCE_INLINE_ real_t abs(real_t g) {

#ifdef REAL_T_IS_DOUBLE

		return absd(g);
#else

		return absf(g);
#endif
	}

	static _FORCE_INLINE_ float absf(float g) {

		union {
			float f;
			uint32_t i;
		} u;

		u.f=g;
		u.i&=2147483647u;
		return u.f;
	}

	static _FORCE_INLINE_ double absd(double g) {

		union {
			double d;
			uint64_t i;
		} u;
		u.d=g;
		u.i&=(uint64_t)9223372036854775807ll;
		return u.d;
	}

	//this function should be as fast as possible and rounding mode should not matter
	static _FORCE_INLINE_ int fast_ftoi(float a) {

		static int b;

#if (defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0603) || WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP // windows 8 phone?
		b = (int)((a>0.0f) ? (a + 0.5f):(a -0.5f));

#elif defined(_MSC_VER) && _MSC_VER < 1800
		__asm fld a
		__asm fistp b
/*#elif defined( __GNUC__ ) && ( defined( __i386__ ) || defined( __x86_64__ ) )
		// use AT&T inline assembly style, document that
		// we use memory as output (=m) and input (m)
		__asm__ __volatile__ (
		"flds %1        \n\t"
		"fistpl %0      \n\t"
		: "=m" (b)
		: "m" (a));*/

#else
		b=lrintf(a); //assuming everything but msvc 2012 or earlier has lrint
#endif
		return	b;
	}


#if defined(__GNUC__)

	static _FORCE_INLINE_ int64_t dtoll(double p_double) { return (int64_t)p_double; } ///@TODO OPTIMIZE
#else

	static _FORCE_INLINE_ int64_t dtoll(double p_double) { return (int64_t)p_double; } ///@TODO OPTIMIZE
#endif

	static _FORCE_INLINE_ float lerp(float a, float b, float c) {

		return a+(b-a)*c;
	}

	static double pow(double x, double y);
	static double log(double x);
	static double exp(double x);

};


#define Math_PI 3.14159265358979323846
#define Math_SQRT12 0.7071067811865475244008443621048490

#endif // MATH_FUNCS_H
