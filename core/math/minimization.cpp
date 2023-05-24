/**************************************************************************/
/*  minimization.cpp                                                      */
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

#include "minimization.h"

#include "core/error/error_macros.h"
#include "core/typedefs.h"
#include "math_funcs.h"

#define SIGN2(a, b) ((b) >= 0.0 ? Math::abs(a) : -Math::abs(a))

/*
 * Find a bracketing triplet (ax, bx, cx) such that bx is between ax and cx (so ax < bx < cx or cx < bx < ax) and f(bx) is less than both f(ax) and f(cx).
 */

constexpr real_t GOLD = 1.618034; // The default ratio by which successive intervals are magnified.
constexpr real_t GLIMIT = 100.0; // The maximum magnification allowed for a parabolic-fit step.
constexpr real_t TINY = 1.0e-20; // Used to prevent any possible division by zero.
#define SHFT(a, b, c, d) \
	(a) = (b);           \
	(b) = (c);           \
	(c) = (d);

void Minimization::bracketing_triplet_from_interval(void *data, real_function *f, real_t *ax, real_t *bx, real_t *cx, real_t *fa, real_t *fb, real_t *fc) {
	// Given a function f, and given distinct initial points ax and bx, this routine searches in
	// the downhill direction (defined by the function as evaluated at the initial points) and returns
	// new points ax, bx, cx that bracket a minimum of the function. Also returned are the function
	// values at the three points, fa, fb, and fc.

	real_t ulim, u, r, q, fu, dum;
	*fa = (*f)(data, *ax);
	*fb = (*f)(data, *bx);
	if (*fb > *fa) { // Switch roles of a and b so that we can go downhill
		SHFT(dum, *ax, *bx, dum) // in the direction from a to b.
		SHFT(dum, *fb, *fa, dum)
	}
	*cx = (*bx) + GOLD * (*bx - *ax); // First guess for c.
	*fc = (*f)(data, *cx);
	while (*fb > *fc) { // Keep returning here until we bracket.
		// Compute u by parabolic extrapolation from a; b; c.
		r = (*bx - *ax) * (*fb - *fc);
		q = (*bx - *cx) * (*fb - *fa);
		u = (*bx) - ((*bx - *cx) * q - (*bx - *ax) * r) / (2.0 * SIGN2(MAX(Math::abs(q - r), TINY), q - r));
		ulim = (*bx) + GLIMIT * (*cx - *bx);
		// We won't go farther than this. Test various possibilities:
		if ((*bx - u) * (u - *cx) > 0.0) { // Parabolic u is between b and c: try it.
			fu = (*f)(data, u);
			if (fu < *fc) { // Got a minimum between b and c.
				*ax = (*bx);
				*bx = u;
				*fa = (*fb);
				*fb = fu;
				return;
			} else if (fu > *fb) { // Got a minimum between between a and u.
				*cx = u;
				*fc = fu;
				return;
			}
			u = (*cx) + GOLD * (*cx - *bx); // Parabolic fit was no use. Use default magnification.
			fu = (*f)(data, u);
		} else if ((*cx - u) * (u - ulim) > 0.0) { // Parabolic fit is between c and its allowed limit.
			fu = (*f)(data, u);
			if (fu < *fc) {
				SHFT(*bx, *cx, u, *cx + GOLD * (*cx - *bx))
				SHFT(*fb, *fc, fu, (*f)(data, u))
			}
		} else if ((u - ulim) * (ulim - *cx) >= 0.0) { // Limit parabolic u to maximum allowed value.
			u = ulim;
			fu = (*f)(data, u);
		} else { // Reject parabolic u, use default magnification.
			u = (*cx) + GOLD * (*cx - *bx);
			fu = (*f)(data, u);
		}
		SHFT(*ax, *bx, *cx, u) // Eliminate oldest point and continue.
		SHFT(*fa, *fb, *fc, fu)
	}
}

/*
 * Find a local minimum of a differentiable real-valued function using a modification of Richard P. Brent's method, making use of the derivative.
 */

constexpr int ITMAX = 100;
constexpr real_t ZEPS = 1e-6;
#define MOV3(a, b, c, d, e, f) \
	(a) = (d);                 \
	(b) = (e);                 \
	(c) = (f);

real_t Minimization::get_local_minimum(void *data, real_function *f, real_function *df, real_t ax, real_t bx, real_t cx, real_t tol, real_t *xmin) {
	// Given a function f and its derivative function df, and given a bracketing triplet of abscissas ax,
	// bx, cx [such that bx is between ax and cx, and f(bx) is less than both f(ax) and f(cx)],
	// this routine isolates the minimum to a fractional precision of about tol using a modification of
	// Brent's method that uses derivatives. The abscissa of the minimum is returned as xmin, and
	// the minimum function value is returned as min, the returned function value.

	int iter;
	bool ok1, ok2; // Will be used as flags for whether proposed steps are acceptable or not.
	real_t a, b, d = 0.0, d1, d2, du, dv, dw, dx, e = 0.0;
	real_t fu, fv, fw, fx, olde = 0.0, tol1, tol2, u, u1, u2, v, w, x, xm;

	a = (ax < cx ? ax : cx);
	b = (ax > cx ? ax : cx);
	x = w = v = bx;
	fw = fv = fx = (*f)(data, x);
	dw = dv = dx = (*df)(data, x);
	for (iter = 1; iter <= ITMAX; iter++) {
		xm = 0.5 * (a + b);
		tol1 = tol * Math::abs(x) + ZEPS;
		tol2 = 2.0 * tol1;
		if (Math::abs(x - xm) <= (tol2 - 0.5 * (b - a))) {
			*xmin = x;
			return fx;
		}
		if (Math::abs(e) > tol1) {
			// Initialize these d's to an out-of-bracket value.
			d1 = 2.0 * (b - a);
			d2 = d1;
			if (dw != dx) {
				d1 = (w - x) * dx / (dx - dw); // Secant method with one point.
			}
			if (dv != dx) {
				d2 = (v - x) * dx / (dx - dv); // And the other.
			}
			// Which of these two estimates of d shall we take? We will insist that they be within the bracket, and on the side pointed to by the derivative at x.
			u1 = x + d1;
			u2 = x + d2;
			ok1 = (a - u1) * (u1 - b) > 0.0 && dx * d1 <= 0.0;
			ok2 = (a - u2) * (u2 - b) > 0.0 && dx * d2 <= 0.0;
			olde = e; // Movement on the step before last.
			e = d;
			if (ok1 || ok2) { // Take only an acceptable d,
				if (ok1 && ok2) { // and if both are acceptable, then take the smallest one.
					d = (Math::abs(d1) < Math::abs(d2) ? d1 : d2);
				} else if (ok1) {
					d = d1;
				} else {
					d = d2;
				}
				if (Math::abs(d) <= Math::abs(0.5 * olde)) {
					u = x + d;
					if (u - a < tol2 || b - u < tol2) {
						d = SIGN2(tol1, xm - x);
					}
				} else { // Bisect, not golden section.
					d = 0.5 * (e = (dx >= 0.0 ? a - x : b - x)); // Decide which segment by the sign of the derivative.
				}
			} else {
				d = 0.5 * (e = (dx >= 0.0 ? a - x : b - x));
			}
		} else {
			d = 0.5 * (e = (dx >= 0.0 ? a - x : b - x));
		}
		if (Math::abs(d) >= tol1) {
			u = x + d;
			fu = (*f)(data, u);
		} else {
			u = x + SIGN2(tol1, d);
			fu = (*f)(data, u);
			if (fu > fx) { // If the minimum step in the downhill direction takes us uphill, then we are done.
				*xmin = x;
				return fx;
			}
		}
		// Now all the housekeeping.
		du = (*df)(data, u);
		if (fu <= fx) {
			if (u >= x) {
				a = x;
			} else {
				b = x;
			}
			MOV3(v, fv, dv, w, fw, dw)
			MOV3(w, fw, dw, x, fx, dx)
			MOV3(x, fx, dx, u, fu, du)
		} else {
			if (u < x) {
				a = u;
			} else {
				b = u;
			}
			if (fu <= fw || w == x) {
				MOV3(v, fv, dv, w, fw, dw)
				MOV3(w, fw, dw, u, fu, du)
			} else if (fu < fv || v == x || v == w) {
				MOV3(v, fv, dv, u, fu, du)
			}
		}
	}
	ERR_FAIL_V_MSG(0.0, "get_local_minimum failed to converge.");
}
