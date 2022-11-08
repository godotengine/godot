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

#include "math_funcs.h" // The RPOLY implementation below was modified to use this instead of system's math.h.

int rpoly(double *op, int degree, double *zeror, double *zeroi); // See below.

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
	int num_roots = rpoly(coeffs, p_degree, roots_real, roots_imag);
	ERR_FAIL_COND_V_MSG(num_roots == -1, -1, "The leading coefficient of the polynomial must be nonzero.");
	for (int i = 0; i < num_roots; i++) {
		if (Math::is_zero_approx(roots_imag[i])) {
			r_roots[i] = (real_t)roots_real[i];
		}
	}
	return num_roots;
}

/*      rpoly.cpp -- Jenkins-Traub real polynomial root finder.
 *
 *      Author: C. Bond (2000).
 *      This is free software released into the public domain.
 *
 *      Translation of TOMS493 from FORTRAN to C. This
 *      implementation of Jenkins-Traub partially adapts
 *      the original code to a C environment by restruction
 *      many of the 'goto' controls to better fit a block
 *      structured form. It also eliminates the global memory
 *      allocation in favor of local, dynamic memory management.
 *
 *      The calling conventions are slightly modified to return
 *      the number of roots found as the function value.
 *
 *      INPUT:
 *      op - double precision vector of coefficients in order of
 *              decreasing powers.
 *      degree - integer degree of polynomial
 *
 *      OUTPUT:
 *      zeror,zeroi - output double precision vectors of the
 *              real and imaginary parts of the zeros.
 *
 *      RETURN:
 *      returnval:   -1 if leading coefficient is zero, otherwise
 *                  number of roots found.
 */

void quad(double a, double b1, double c, double *sr, double *si,
		double *lr, double *li);
void fxshfr(int l2, int *nz);
void quadit(double *uu, double *vv, int *nz);
void realit(double sss, int *nz, int *iflag);
void calcsc(int *type);
void nextk(int *type);
void newest(int type, double *uu, double *vv);
void quadsd(int n, double *u, double *v, double *p, double *q,
		double *a, double *b);
double *p, *qp, *k, *qk, *svk;
double sr, si, u, v, a, b, c, d, a1, a2;
double a3, a6, a7, e, f, g, h, szr, szi, lzr, lzi;
double eta, are, mre;
int n, nn, nmi, zerok;

int rpoly(double *op, int degree, double *zeror, double *zeroi) {
	double t, aa, bb, cc, *temp, factor, rot;
	double *pt;
	double lo, max, min, xx, yy, cosr, sinr, xxx, x, sc, bnd;
	double xm, ff, df, dx, infin, smalno, base;
	int cnt, nz, i, j, jj, l, nm1, zerok;
	/*  The following statements set machine constants. */
	base = 2.0;
	eta = 2.22e-16;
	infin = 3.4e38;
	smalno = 1.2e-38;

	are = eta;
	mre = eta;
	lo = smalno / eta;
	/*  Initialization of constants for shift rotation. */
	xx = Math::sqrt(0.5);
	yy = -xx;
	rot = 94.0;
	rot *= 0.017453293;
	cosr = Math::cos(rot);
	sinr = Math::sin(rot);
	n = degree;
	/*  Algorithm fails of the leading coefficient is zero. */
	if (op[0] == 0.0)
		return -1;
	/*  Remove the zeros at the origin, if any. */
	while (op[n] == 0.0) {
		j = degree - n;
		zeror[j] = 0.0;
		zeroi[j] = 0.0;
		n--;
	}
	if (n < 1)
		return -1;
	/*
	 *  Allocate memory here
	 */
	temp = new double[degree + 1];
	pt = new double[degree + 1];
	p = new double[degree + 1];
	qp = new double[degree + 1];
	k = new double[degree + 1];
	qk = new double[degree + 1];
	svk = new double[degree + 1];
	/*  Make a copy of the coefficients. */
	for (i = 0; i <= n; i++)
		p[i] = op[i];
/*  Start the algorithm for one zero. */
_40:
	if (n == 1) {
		zeror[degree - 1] = -p[1] / p[0];
		zeroi[degree - 1] = 0.0;
		n -= 1;
		goto _99;
	}
	/*  Calculate the final zero or pair of zeros. */
	if (n == 2) {
		quad(p[0], p[1], p[2], &zeror[degree - 2], &zeroi[degree - 2],
				&zeror[degree - 1], &zeroi[degree - 1]);
		n -= 2;
		goto _99;
	}
	/*  Find largest and smallest moduli of coefficients. */
	max = 0.0;
	min = infin;
	for (i = 0; i <= n; i++) {
		x = Math::abs(p[i]);
		if (x > max)
			max = x;
		if (x != 0.0 && x < min)
			min = x;
	}
	/*  Scale if there are large or very small coefficients.
	 *  Computes a scale factor to multiply the coefficients of the
	 *  polynomial. The scaling si done to avoid overflow and to
	 *  avoid undetected underflow interfering with the convergence
	 *  criterion. The factor is a power of the base.
	 */
	sc = lo / min;
	if (sc > 1.0 && infin / sc < max)
		goto _110;
	if (sc <= 1.0) {
		if (max < 10.0)
			goto _110;
		if (sc == 0.0)
			sc = smalno;
	}
	l = (int)(Math::log(sc) / Math::log(base) + 0.5);
	factor = Math::pow(base * 1.0, l);
	if (factor != 1.0) {
		for (i = 0; i <= n; i++)
			p[i] = factor * p[i]; /* Scale polynomial. */
	}
_110:
	/*  Compute lower bound on moduli of roots. */
	for (i = 0; i <= n; i++) {
		pt[i] = (Math::abs(p[i]));
	}
	pt[n] = -pt[n];
	/*  Compute upper estimate of bound. */
	x = Math::exp((Math::log(-pt[n]) - Math::log(pt[0])) / (double)n);
	/*  If Newton step at the origin is better, use it. */
	if (pt[n - 1] != 0.0) {
		xm = -pt[n] / pt[n - 1];
		if (xm < x)
			x = xm;
	}
	/*  Chop the interval (0,x) until ff <= 0 */
	while (1) {
		xm = x * 0.1;
		ff = pt[0];
		for (i = 1; i <= n; i++)
			ff = ff * xm + pt[i];
		if (ff <= 0.0)
			break;
		x = xm;
	}
	dx = x;
	/*  Do Newton interation until x converges to two
	 *  decimal places.
	 */
	while (Math::abs(dx / x) > 0.005) {
		ff = pt[0];
		df = ff;
		for (i = 1; i < n; i++) {
			ff = ff * x + pt[i];
			df = df * x + ff;
		}
		ff = ff * x + pt[n];
		dx = ff / df;
		x -= dx;
	}
	bnd = x;
	/*  Compute the derivative as the initial k polynomial
	 *  and do 5 steps with no shift.
	 */
	nm1 = n - 1;
	for (i = 1; i < n; i++)
		k[i] = (double)(n - i) * p[i] / (double)n;
	k[0] = p[0];
	aa = p[n];
	bb = p[n - 1];
	zerok = (k[n - 1] == 0);
	for (jj = 0; jj < 5; jj++) {
		cc = k[n - 1];
		if (!zerok) {
			/*  Use a scaled form of recurrence if value of k at 0 is nonzero. */
			t = -aa / cc;
			for (i = 0; i < nm1; i++) {
				j = n - i - 1;
				k[j] = t * k[j - 1] + p[j];
			}
			k[0] = p[0];
			zerok = (Math::abs(k[n - 1]) <= Math::abs(bb) * eta * 10.0);
		} else {
			/*  Use unscaled form of recurrence. */
			for (i = 0; i < nm1; i++) {
				j = n - i - 1;
				k[j] = k[j - 1];
			}
			k[0] = 0.0;
			zerok = (k[n - 1] == 0.0);
		}
	}
	/*  Save k for restarts with new shifts. */
	for (i = 0; i < n; i++)
		temp[i] = k[i];
	/*  Loop to select the quadratic corresponding to each new shift. */
	for (cnt = 0; cnt < 20; cnt++) {
		/*  Quadratic corresponds to a double shift to a
		 *  non-real point and its complex conjugate. The point
		 *  has modulus bnd and amplitude rotated by 94 degrees
		 *  from the previous shift.
		 */
		xxx = cosr * xx - sinr * yy;
		yy = sinr * xx + cosr * yy;
		xx = xxx;
		sr = bnd * xx;
		si = bnd * yy;
		u = -2.0 * sr;
		v = bnd;
		fxshfr(20 * (cnt + 1), &nz);
		if (nz != 0) {
			/*  The second stage jumps directly to one of the third
			 *  stage iterations and returns here if successful.
			 *  Deflate the polynomial, store the zero or zeros and
			 *  return to the main algorithm.
			 */
			j = degree - n;
			zeror[j] = szr;
			zeroi[j] = szi;
			n -= nz;
			for (i = 0; i <= n; i++)
				p[i] = qp[i];
			if (nz != 1) {
				zeror[j + 1] = lzr;
				zeroi[j + 1] = lzi;
			}
			goto _40;
		}
		/*  If the iteration is unsuccessful another quadratic
		 *  is chosen after restoring k.
		 */
		for (i = 0; i < n; i++) {
			k[i] = temp[i];
		}
	}
/*  Return with failure if no convergence with 20 shifts. */
_99:
	delete[] svk;
	delete[] qk;
	delete[] k;
	delete[] qp;
	delete[] p;
	delete[] pt;
	delete[] temp;

	return degree - n;
}
/*  Computes up to L2 fixed shift k-polynomials,
 *  testing for convergence in the linear or quadratic
 *  case. Initiates one of the variable shift
 *  iterations and returns with the number of zeros
 *  found.
 */
void fxshfr(int l2, int *nz) {
	double svu, svv, ui, vi, s;
	double betas, betav, oss, ovv, ss, vv, ts, tv;
	double ots, otv, tvv, tss;
	int type, i, j, iflag, vpass, spass, vtry, stry;

	*nz = 0;
	betav = 0.25;
	betas = 0.25;
	oss = sr;
	ovv = v;
	/*  Evaluate polynomial by synthetic division. */
	quadsd(n, &u, &v, p, qp, &a, &b);
	calcsc(&type);
	for (j = 0; j < l2; j++) {
		/*  Calculate next k polynomial and estimate v. */
		nextk(&type);
		calcsc(&type);
		newest(type, &ui, &vi);
		vv = vi;
		/*  Estimate s. */
		ss = 0.0;
		if (k[n - 1] != 0.0)
			ss = -p[n] / k[n - 1];
		tv = 1.0;
		ts = 1.0;
		if (j == 0 || type == 3)
			goto _70;
		/*  Compute relative measures of convergence of s and v sequences. */
		if (vv != 0.0)
			tv = Math::abs((vv - ovv) / vv);
		if (ss != 0.0)
			ts = Math::abs((ss - oss) / ss);
		/*  If decreasing, multiply two most recent convergence measures. */
		tvv = 1.0;
		if (tv < otv)
			tvv = tv * otv;
		tss = 1.0;
		if (ts < ots)
			tss = ts * ots;
		/*  Compare with convergence criteria. */
		vpass = (tvv < betav);
		spass = (tss < betas);
		if (!(spass || vpass))
			goto _70;
		/*  At least one sequence has passed the convergence test.
		 *  Store variables before iterating.
		 */
		svu = u;
		svv = v;
		for (i = 0; i < n; i++) {
			svk[i] = k[i];
		}
		s = ss;
		/*  Choose iteration according to the fastest converging
		 *  sequence.
		 */
		vtry = 0;
		stry = 0;
		if ((spass && (!vpass)) || tss < tvv)
			goto _40;
	_20:
		quadit(&ui, &vi, nz);
		if (*nz > 0)
			return;
		/*  Quadratic iteration has failed. Flag that it has
		 *  been tried and decrease the convergence criterion.
		 */
		vtry = 1;
		betav *= 0.25;
		/*  Try linear iteration if it has not been tried and
		 *  the S sequence is converging.
		 */
		if (stry || !spass)
			goto _50;
		for (i = 0; i < n; i++) {
			k[i] = svk[i];
		}
	_40:
		realit(s, nz, &iflag);
		if (*nz > 0)
			return;
		/*  Linear iteration has failed. Flag that it has been
		 *  tried and decrease the convergence criterion.
		 */
		stry = 1;
		betas *= 0.25;
		if (iflag == 0)
			goto _50;
		/*  If linear iteration signals an almost double real
		 *  zero attempt quadratic iteration.
		 */
		ui = -(s + s);
		vi = s * s;
		goto _20;
	/*  Restore variables. */
	_50:
		u = svu;
		v = svv;
		for (i = 0; i < n; i++) {
			k[i] = svk[i];
		}
		/*  Try quadratic iteration if it has not been tried
		 *  and the V sequence is convergin.
		 */
		if (vpass && !vtry)
			goto _20;
		/*  Recompute QP and scalar values to continue the
		 *  second stage.
		 */
		quadsd(n, &u, &v, p, qp, &a, &b);
		calcsc(&type);
	_70:
		ovv = vv;
		oss = ss;
		otv = tv;
		ots = ts;
	}
}
/*  Variable-shift k-polynomial iteration for a
 *  quadratic factor converges only if the zeros are
 *  equimodular or nearly so.
 *  uu, vv - coefficients of starting quadratic.
 *  nz - number of zeros found.
 */
void quadit(double *uu, double *vv, int *nz) {
	double ui, vi;
	double mp, omp, ee, relstp, t, zm;
	int type, i, j, tried;

	*nz = 0;
	tried = 0;
	u = *uu;
	v = *vv;
	j = 0;
/*  Main loop. */
_10:
	quad(1.0, u, v, &szr, &szi, &lzr, &lzi);
	/*  Return if roots of the quadratic are real and not
	 *  close to multiple or nearly equal and of opposite
	 *  sign.
	 */
	if (Math::abs(Math::abs(szr) - Math::abs(lzr)) > 0.01 * Math::abs(lzr))
		return;
	/*  Evaluate polynomial by quadratic synthetic division. */
	quadsd(n, &u, &v, p, qp, &a, &b);
	mp = Math::abs(a - szr * b) + Math::abs(szi * b);
	/*  Compute a rigorous bound on the rounding error in
	 *  evaluating p.
	 */
	zm = Math::sqrt(Math::abs(v));
	ee = 2.0 * Math::abs(qp[0]);
	t = -szr * b;
	for (i = 1; i < n; i++) {
		ee = ee * zm + Math::abs(qp[i]);
	}
	ee = ee * zm + Math::abs(a + t);
	ee *= (5.0 * mre + 4.0 * are);
	ee = ee - (5.0 * mre + 2.0 * are) * (Math::abs(a + t) + Math::abs(b * zm));
	ee = ee + 2.0 * are * Math::abs(t);
	/*  Iteration has converged sufficiently if the
	 *  polynomial value is less than 20 times this bound.
	 */
	if (mp <= 20.0 * ee) {
		*nz = 2;
		return;
	}
	//_30:
	j++;
	/*  Stop iteration after 20 steps. */
	if (j > 20)
		return;
	if (j < 2)
		goto _50;
	if (relstp > 0.01 || mp < omp || tried)
		goto _50;
	/*  A cluster appears to be stalling the convergence.
	 *  Five fixed shift steps are taken with a u,v close
	 *  to the cluster.
	 */
	if (relstp < eta)
		relstp = eta;
	relstp = Math::sqrt(relstp);
	u = u - u * relstp;
	v = v + v * relstp;
	quadsd(n, &u, &v, p, qp, &a, &b);
	for (i = 0; i < 5; i++) {
		calcsc(&type);
		nextk(&type);
	}
	tried = 1;
	j = 0;
_50:
	omp = mp;
	/*  Calculate next k polynomial and new u and v. */
	calcsc(&type);
	nextk(&type);
	calcsc(&type);
	newest(type, &ui, &vi);
	/*  If vi is zero the iteration is not converging. */
	if (vi == 0.0)
		return;
	relstp = Math::abs((vi - v) / vi);
	u = ui;
	v = vi;
	goto _10;
}
/*  Variable-shift H polynomial iteration for a real zero.
 *  sss - starting iterate
 *  nz  - number of zeros found
 *  iflag - flag to indicate a pair of zeros near real axis.
 */
void realit(double sss, int *nz, int *iflag) {
	double pv, kv, t, s;
	double ms, mp, omp, ee;
	int i, j;

	*nz = 0;
	s = sss;
	*iflag = 0;
	j = 0;
	/*  Main loop */
	while (1) {
		pv = p[0];
		/*  Evaluate p at s. */
		qp[0] = pv;
		for (i = 1; i <= n; i++) {
			pv = pv * s + p[i];
			qp[i] = pv;
		}
		mp = Math::abs(pv);
		/*  Compute a rigorous bound on the error in evaluating p. */
		ms = Math::abs(s);
		ee = (mre / (are + mre)) * Math::abs(qp[0]);
		for (i = 1; i <= n; i++) {
			ee = ee * ms + Math::abs(qp[i]);
		}
		/*  Iteration has converged sufficiently if the polynomial
		 *  value is less than 20 times this bound.
		 */
		if (mp <= 20.0 * ((are + mre) * ee - mre * mp)) {
			*nz = 1;
			szr = s;
			szi = 0.0;
		}
		j++;
		/*  Stop iteration after 10 steps. */
		if (j > 10)
			return;
		if (j < 2)
			goto _50;
		if (Math::abs(t) > 0.001 * Math::abs(s - t) || mp < omp)
			goto _50;
		/*  A cluster of zeros near the real axis has been
		 *  encountered. Return with iflag set to initiate a
		 *  quadratic iteration.
		 */
		*iflag = 1;
		return;
	/*  Return if the polynomial value has increased significantly. */
	_50:
		omp = mp;
		/*  Compute t, the next polynomial, and the new iterate. */
		kv = k[0];
		qk[0] = kv;
		for (i = 1; i < n; i++) {
			kv = kv * s + k[i];
			qk[i] = kv;
		}
		if (Math::abs(kv) <= Math::abs(k[n]) * 10.0 * eta) {
			/*  Use unscaled form. */
			k[0] = 0.0;
			for (i = 1; i < n; i++) {
				k[i] = qk[i - 1];
			}
		} else {
			/*  Use the scaled form of the recurrence if the value
			 *  of k at s is nonzero.
			 */
			t = -pv / kv;
			k[0] = qp[0];
			for (i = 1; i < n; i++) {
				k[i] = t * qk[i - 1] + qp[i];
			}
		}
		kv = k[0];
		for (i = 1; i < n; i++) {
			kv = kv * s + k[i];
		}
		t = 0.0;
		if (Math::abs(kv) > Math::abs(k[n - 1] * 10.0 * eta))
			t = -pv / kv;
		s += t;
	}
}

/*  This routine calculates scalar quantities used to
 *  compute the next k polynomial and new estimates of
 *  the quadratic coefficients.
 *  type - integer variable set here indicating how the
 *  calculations are normalized to avoid overflow.
 */
void calcsc(int *type) {
	/*  Synthetic division of k by the quadratic 1,u,v */
	quadsd(n - 1, &u, &v, k, qk, &c, &d);
	if (Math::abs(c) > Math::abs(k[n - 1] * 100.0 * eta))
		goto _10;
	if (Math::abs(d) > Math::abs(k[n - 2] * 100.0 * eta))
		goto _10;
	*type = 3;
	/*  Type=3 indicates the quadratic is almost a factor of k. */
	return;
_10:
	if (Math::abs(d) < Math::abs(c)) {
		*type = 1;
		/*  Type=1 indicates that all formulas are divided by c. */
		e = a / c;
		f = d / c;
		g = u * e;
		h = v * b;
		a3 = a * e + (h / c + g) * b;
		a1 = b - a * (d / c);
		a7 = a + g * d + h * f;
		return;
	}
	*type = 2;
	/*  Type=2 indicates that all formulas are divided by d. */
	e = a / d;
	f = c / d;
	g = u * b;
	h = v * b;
	a3 = (a + g) * e + h * (b / d);
	a1 = b * f - a;
	a7 = (f + u) * a + h;
}
/*  Computes the next k polynomials using scalars
 *  computed in calcsc.
 */
void nextk(int *type) {
	double temp;
	int i;

	if (*type == 3) {
		/*  Use unscaled form of the recurrence if type is 3. */
		k[0] = 0.0;
		k[1] = 0.0;
		for (i = 2; i < n; i++) {
			k[i] = qk[i - 2];
		}
		return;
	}
	temp = a;
	if (*type == 1)
		temp = b;
	if (Math::abs(a1) <= Math::abs(temp) * eta * 10.0) {
		/*  If a1 is nearly zero then use a special form of the
		 *  recurrence.
		 */
		k[0] = 0.0;
		k[1] = -a7 * qp[0];
		for (i = 2; i < n; i++) {
			k[i] = a3 * qk[i - 2] - a7 * qp[i - 1];
		}
	}
	/*  Use scaled form of the recurrence. */
	a7 /= a1;
	a3 /= a1;
	k[0] = qp[0];
	k[1] = qp[1] - a7 * qp[0];
	for (i = 2; i < n; i++) {
		k[i] = a3 * qk[i - 2] - a7 * qp[i - 1] + qp[i];
	}
}
/*  Compute new estimates of the quadratic coefficients
 *  using the scalars computed in calcsc.
 */
void newest(int type, double *uu, double *vv) {
	double a4, a5, b1, b2, c1, c2, c3, c4, temp;

	/* Use formulas appropriate to setting of type. */
	if (type == 3) {
		/*  If type=3 the quadratic is zeroed. */
		*uu = 0.0;
		*vv = 0.0;
		return;
	}
	if (type == 2) {
		a4 = (a + g) * f + h;
		a5 = (f + u) * c + v * d;
	} else {
		a4 = a + u * b + h * f;
		a5 = c + (u + v * f) * d;
	}
	/*  Evaluate new quadratic coefficients. */
	b1 = -k[n - 1] / p[n];
	b2 = -(k[n - 2] + b1 * p[n - 1]) / p[n];
	c1 = v * b2 * a1;
	c2 = b1 * a7;
	c3 = b1 * b1 * a3;
	c4 = c1 - c2 - c3;
	temp = a5 + b1 * a4 - c4;
	if (temp == 0.0) {
		*uu = 0.0;
		*vv = 0.0;
		return;
	}
	*uu = u - (u * (c3 + c2) + v * (b1 * a1 + b2 * a7)) / temp;
	*vv = v * (1.0 + c4 / temp);
	return;
}

/*  Divides p by the quadratic 1,u,v placing the quotient
 *  in q and the remainder in a,b.
 */
void quadsd(int nn, double *u, double *v, double *p, double *q,
		double *a, double *b) {
	double c;
	int i;
	*b = p[0];
	q[0] = *b;
	*a = p[1] - (*b) * (*u);
	q[1] = *a;
	for (i = 2; i <= nn; i++) {
		c = p[i] - (*a) * (*u) - (*b) * (*v);
		q[i] = c;
		*b = *a;
		*a = c;
	}
}
/*  Calculate the zeros of the quadratic a*z^2 + b1*z + c.
 *  The quadratic formula, modified to avoid overflow, is used
 *  to find the larger zero if the zeros are real and both
 *  are complex. The smaller real zero is found directly from
 *  the product of the zeros c/a.
 */
void quad(double a, double b1, double c, double *sr, double *si,
		double *lr, double *li) {
	double b, d, e;

	if (a == 0.0) { /* less than two roots */
		if (b1 != 0.0)
			*sr = -c / b1;
		else
			*sr = 0.0;
		*lr = 0.0;
		*si = 0.0;
		*li = 0.0;
		return;
	}
	if (c == 0.0) { /* one real root, one zero root */
		*sr = 0.0;
		*lr = -b1 / a;
		*si = 0.0;
		*li = 0.0;
		return;
	}
	/* Compute discriminant avoiding overflow. */
	b = b1 / 2.0;
	if (Math::abs(b) < Math::abs(c)) {
		if (c < 0.0)
			e = -a;
		else
			e = a;
		e = b * (b / Math::abs(c)) - e;
		d = Math::sqrt(Math::abs(e)) * Math::sqrt(Math::abs(c));
	} else {
		e = 1.0 - (a / b) * (c / b);
		d = Math::sqrt(Math::abs(e)) * Math::abs(b);
	}
	if (e < 0.0) { /* complex conjugate zeros */
		*sr = -b / a;
		*lr = *sr;
		*si = Math::abs(d / a);
		*li = -(*si);
	} else {
		if (b >= 0.0) /* real zeros. */
			d = -d;
		*lr = (-b + d) / a;
		*sr = 0.0;
		if (*lr != 0.0)
			*sr = (c / *lr) / a;
		*si = 0.0;
		*li = 0.0;
	}
}

// end rpoly.cpp

#endif // POLYNOMIAL_ROOT_FINDER_H
