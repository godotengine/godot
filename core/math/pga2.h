/**************************************************************************/
/*  pga2.h                                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef PGA2_H
#define PGA2_H

#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/string/ustring.h"

struct _NO_DISCARD_ PGAVector2 {
	static const int AXIS_COUNT = 8;

	enum GradeMask {
		GRADE_MASK_0 = 0b0001,
		GRADE_MASK_1 = 0b0010,
		GRADE_MASK_2 = 0b0100,
		GRADE_MASK_3 = 0b1000,

		// Presets
		GRADE_MASK_SCALAR = GRADE_MASK_0,
		GRADE_MASK_LINE = GRADE_MASK_1,
		GRADE_MASK_POINT = GRADE_MASK_2,
		GRADE_MASK_MOTOR = GRADE_MASK_0 | GRADE_MASK_2
	};
	
    union {
		struct {
			real_t e;
			real_t e0;
			real_t e1;
			real_t e2;
			real_t e01;
			real_t e02;
			real_t e12;
			real_t e012;
		};

		real_t coord[AXIS_COUNT] = { 0 };
	};

	_FORCE_INLINE_ const real_t &operator[](int p_axis) const {
		DEV_ASSERT((unsigned int)p_axis < AXIS_COUNT);
		return coord[p_axis];
	}

	_FORCE_INLINE_ real_t &operator[](int p_axis) {
		DEV_ASSERT((unsigned int)p_axis < AXIS_COUNT);
		return coord[p_axis];
	}

	// Products
	PGAVector2 gp(const PGAVector2 &p_mv) const; // Geometric
	PGAVector2 op(const PGAVector2 &p_mv) const; // Outer
	PGAVector2 ip(const PGAVector2 &p_mv) const; // Inner
	PGAVector2 rp(const PGAVector2 &p_mv) const; // Regressive
	PGAVector2 cp(const PGAVector2 &p_mv) const; // Commutator
	real_t sp(const PGAVector2 &p_mv) const; // Scalar

	// Following the convention of https://kingdon.readthedocs.io/en/latest/usage.html
	PGAVector2 operator*(const PGAVector2 &p_mv) const { // Geometric
        return gp(p_mv);
	}
	PGAVector2 operator^(const PGAVector2 &p_mv) const { // Outer
        return op(p_mv);
	}
	PGAVector2 operator|(const PGAVector2 &p_mv) const { // Inner
        return ip(p_mv);
	}
	PGAVector2 operator&(const PGAVector2 &p_mv) const { // Regressive
        return rp(p_mv);
	}

	_FORCE_INLINE_ PGAVector2 sw(const PGAVector2 &p_mv) const { // Sandwich
		return (*this * p_mv) * reverse();
	}
	_FORCE_INLINE_ PGAVector2 proj(const PGAVector2 &p_mv) const { // Projective
		return (*this | p_mv) * reverse();
	}

	void operator+=(const PGAVector2 &p_mv);
	void operator-=(const PGAVector2 &p_mv);
	void operator*=(const real_t p_s);
	void operator/=(const real_t p_s);

	_FORCE_INLINE_ PGAVector2 operator+(const PGAVector2 &p_mv) const {
		PGAVector2 r = *this;
		r += p_mv;
		return r;
	}
	_FORCE_INLINE_ PGAVector2 operator-(const PGAVector2 &p_mv) const {
		PGAVector2 r = *this;
		r -= p_mv;
		return r;
	}
	_FORCE_INLINE_ PGAVector2 operator*(const real_t p_s) const {
		PGAVector2 r = *this;
		r *= p_s;
		return r;
	}
	_FORCE_INLINE_ PGAVector2 operator/(const real_t p_s) const {
		PGAVector2 r = *this;
		r /= p_s;
		return r;
	}

	PGAVector2 reverse() const;
	PGAVector2 dual() const;
	real_t norm_square() const {
        return gp(reverse()).e;
    }

	_FORCE_INLINE_ real_t norm() const {
		return Math::sqrt(norm_square());
	}

	PGAVector2 grade(GradeMask mask) const;

	// Convertor to subtype

    // Constructor
};


struct _NO_DISCARD_ PGALine2 {
	static const int AXIS_COUNT = 3;

	union {
		struct {
			real_t e0;
			real_t e1;
			real_t e2;
		};

		real_t coord[AXIS_COUNT] = { 0 };
	};

	_FORCE_INLINE_ const real_t &operator[](int p_axis) const {
		DEV_ASSERT((unsigned int)p_axis < AXIS_COUNT);
		return coord[p_axis];
	}

	_FORCE_INLINE_ real_t &operator[](int p_axis) {
		DEV_ASSERT((unsigned int)p_axis < AXIS_COUNT);
		return coord[p_axis];
	}
};

struct _NO_DISCARD_ PGAPoint2 {
	static const int AXIS_COUNT = 3;

	union {
		struct {
			real_t e01;
			real_t e02;
			real_t e12;
		};

		real_t coord[AXIS_COUNT] = { 0 };
	};

	_FORCE_INLINE_ const real_t &operator[](int p_axis) const {
		DEV_ASSERT((unsigned int)p_axis < AXIS_COUNT);
		return coord[p_axis];
	}

	_FORCE_INLINE_ real_t &operator[](int p_axis) {
		DEV_ASSERT((unsigned int)p_axis < AXIS_COUNT);
		return coord[p_axis];
	}
};

struct _NO_DISCARD_ PGAMotor2 {
	static const int AXIS_COUNT = 4;

	union {
		struct {
			real_t e;
            real_t e01;
			real_t e02;
			real_t e12;
		};

		real_t coord[AXIS_COUNT] = { 0 };
	};

	_FORCE_INLINE_ const real_t &operator[](int p_axis) const {
		DEV_ASSERT((unsigned int)p_axis < AXIS_COUNT);
		return coord[p_axis];
	}

	_FORCE_INLINE_ real_t &operator[](int p_axis) {
		DEV_ASSERT((unsigned int)p_axis < AXIS_COUNT);
		return coord[p_axis];
	}
};


#endif // PGA2_H