/*************************************************************************/
/*  polynomial.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include "core/templates/vector.h"
#include "math_defs.h"

class Polynomial {
	// Based on David Eberly's Polynomial class.

	// NOTE: In this implementation the zero polynomial has degree zero,
	// which violates the equality (p1*p2).degree() == p1.degree() + p2.degree(),
	// but on the other hand it does simplify the code.

	Vector<real_t> coefficients;

public:
	Polynomial() :
			coefficients({ 0.0 }) {}

	Polynomial(const Polynomial &other) {
		coefficients.resize_zeroed(other.degree() + 1);
		for (size_t i = 0; i <= other.degree(); i++) {
			coefficients.ptrw()[i] = other[i];
		}
	}

	Polynomial(size_t degree) {
		coefficients.resize_zeroed(degree + 1);
	}

	Polynomial(Vector<real_t> p_coefficients) :
			coefficients(p_coefficients) {
		eliminate_leading_zero_coefficients();
	}

	void eliminate_leading_zero_coefficients() {
		size_t size = coefficients.size();
		if (size > 1) {
			size_t leading = size - 1;
			while (leading > 0 && coefficients[leading] == 0.0) {
				leading--;
			}
			coefficients.resize(leading + 1);
		}
	}

	Vector<real_t> coefficients_decreasing() {
		Vector<real_t> coeffs = coefficients.duplicate();
		coeffs.reverse();
		return coeffs;
	}

	inline size_t degree() const {
		return coefficients.size() - 1; // By design, coefficients.size() > 0, so no typecasting is required for the subtraction.
	}

	inline real_t &operator[](size_t p_index) {
		return coefficients.ptrw()[p_index];
	}

	inline real_t const &operator[](size_t p_index) const {
		return coefficients[p_index];
	}

	// Polynomial evaluation.

	real_t operator()(real_t const &p_input) {
		return operator()(&p_input);
	}

	real_t operator()(real_t const *p_input) {
		// By design, coefficients.size() > 0, so no typecasting is required for the subtraction to compute jmax.
		size_t jmax = coefficients.size() - 1;
		real_t output = coefficients[jmax];
		if (jmax > 0) {
			// In this block, the initial i-value is guaranteed to be nonnegative.
			for (size_t j = 0, i = jmax - 1; j < jmax; ++j, --i) {
				output *= *p_input;
				output += coefficients[i];
			}
		}
		return output;
	}
};

// Unary operations

Polynomial operator+(Polynomial const &p_poly) {
	return p_poly;
}

Polynomial operator-(Polynomial const &p_poly) {
	Polynomial result = p_poly;
	for (size_t i = 0; i <= p_poly.degree(); ++i) {
		result[i] = -result[i];
	}
	return result;
}

// Linear algebraic operations

Polynomial operator+(Polynomial const &p_poly, real_t const &p_scalar) {
	Polynomial result = p_poly;
	result[0] += p_scalar;
	return result;
}

Polynomial operator+(real_t const &p_scalar, Polynomial const &p_poly) {
	Polynomial result = p_poly;
	result[0] += p_scalar;
	return result;
}

Polynomial operator+(Polynomial const &p_poly0, Polynomial const &p_poly1) {
	size_t const p0degree = p_poly0.degree();
	size_t const p1degree = p_poly1.degree();
	if (p0degree >= p1degree) {
		Polynomial result = p_poly0;
		for (size_t i = 0; i <= p1degree; i++) {
			result[i] += p_poly1[i];
		}
		result.eliminate_leading_zero_coefficients();
		return result;
	} else {
		Polynomial result = p_poly1;
		for (size_t i = 0; i <= p0degree; i++) {
			result[i] += p_poly0[i];
		}
		result.eliminate_leading_zero_coefficients();
		return result;
	}
}

Polynomial operator-(Polynomial const &p_poly, real_t const &p_scalar) {
	Polynomial result = p_poly;
	result[0] -= p_scalar;
	return result;
}

Polynomial operator-(real_t const &p_scalar, Polynomial const &p_poly) {
	Polynomial result = -p_poly;
	result[0] += p_scalar;
	return result;
}

Polynomial operator-(Polynomial const &p_poly0, Polynomial const &p_poly1) {
	size_t const p0degree = p_poly0.degree();
	size_t const p1degree = p_poly1.degree();
	if (p0degree >= p1degree) {
		Polynomial result = p_poly0;
		for (size_t i = 0; i <= p1degree; i++) {
			result[i] -= p_poly1[i];
		}
		result.eliminate_leading_zero_coefficients();
		return result;
	} else {
		Polynomial result = -p_poly1;
		for (size_t i = 0; i <= p0degree; i++) {
			result[i] += p_poly0[i];
		}
		result.eliminate_leading_zero_coefficients();
		return result;
	}
}

// Polynomial multiplication.

Polynomial operator*(real_t const &p_scalar, Polynomial const &p_poly) {
	Polynomial result = p_poly;
	for (size_t i = 0; i <= p_poly.degree(); i++) {
		result[i] *= p_scalar;
	}
	return result;
}

Polynomial operator*(Polynomial const &p_poly, real_t const &p_scalar) {
	Polynomial result = p_poly;
	for (size_t i = 0; i <= p_poly.degree(); i++) {
		result[i] *= p_scalar;
	}
	return result;
}

Polynomial operator*(Polynomial const &p_poly0, Polynomial const &p_poly1) {
	size_t const p0degree = p_poly0.degree();
	size_t const p1degree = p_poly1.degree();
	Polynomial result(p0degree + p1degree); // Initialized to zero.
	for (size_t i = 0; i <= p0degree; i++) {
		for (size_t j = 0; j <= p1degree; j++) {
			result[i + j] += p_poly0[i] * p_poly1[j];
		}
	}
	return result;
}

#endif // POLYNOMIAL_H
