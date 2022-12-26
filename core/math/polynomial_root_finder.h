/*************************************************************************/
/*  polynomial_root_finder.h                                             */
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

#ifndef POLYNOMIAL_ROOT_FINDER_H
#define POLYNOMIAL_ROOT_FINDER_H

#include "thirdparty/misc/rpoly.hpp"

int real_roots_of_nonconstant_polynomial(int p_degree, const real_t *p_coefficients, real_t *r_roots) {
	// Convenient wrapper for RPOLY.
	// The array of coefficients should be given in decreasing order of powers, and the leading coefficient should be nonzero.
	ERR_FAIL_COND_V_MSG(p_degree <= 0, -1, "The polynomial must be non-constant.");
#ifdef REAL_T_IS_DOUBLE
	double *coeffs = const_cast<double *>(p_coefficients);
#else
	double *coeffs = (double *)alloca(sizeof(double) * (p_degree + 1));
	for (int i = 0; i < p_degree + 1; i++) {
		coeffs[i] = (double)p_coefficients[i];
	}
#endif
	double *roots_real = (double *)alloca(sizeof(double) * p_degree);
	double *roots_imag = (double *)alloca(sizeof(double) * p_degree);
	int num_roots = RPOLY::rpoly(coeffs, p_degree, roots_real, roots_imag);
	ERR_FAIL_COND_V_MSG(num_roots == -1, -1, "The leading coefficient of the polynomial must be nonzero.");
	for (int i = 0; i < num_roots; i++) {
		if (Math::is_zero_approx(roots_imag[i])) {
			r_roots[i] = (real_t)roots_real[i];
		}
	}
	return num_roots;
}

#endif // POLYNOMIAL_ROOT_FINDER_H
