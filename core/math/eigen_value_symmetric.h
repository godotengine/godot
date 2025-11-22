/**************************************************************************/
/*  eigen_value_symmetric.h                                               */
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

#pragma once

#include "core/math/basis.h"

// This code is simply a re-implementation of the same function from Jolt Physics,
// but using Godot's Vector3 and Basis structures
bool eigen_value_symmetric(const Basis &p_matrix, Basis &r_eig_vec, Vector3 &r_eig_val) {
	const int MAX_SWEEPS = 50;
	const int N = 3;

	Basis a = p_matrix;

	Vector3 b;
	Vector3 z;

	for (int ip = 0; ip < N; ip++) {
		// Initialize b and output to diagonal of a
		b[ip] = a[ip][ip];
		r_eig_val[ip] = a[ip][ip];

		// reset z
		z[ip] = 0.0;
	}

	for (int sweep = 0; sweep < MAX_SWEEPS; sweep++) {
		// sum the off-diagonal elements of a
		real_t sm = 0.0;
		for (int ip = 0; ip < N; ip++) {
			for (int iq = ip + 1; iq < N; iq++) {
				sm += Math::abs(a[ip][iq]);
			}
		}
		real_t avg_sm = sm / (N * N);

		// Normal return
		if (avg_sm < CMP_EPSILON2) {
			return true;
		}

		real_t thresh = sweep < 4 ? 0.2 * avg_sm : CMP_EPSILON2;

		for (int ip = 0; ip < N - 1; ip++) {
			for (int iq = ip + 1; iq < N; iq++) {
				real_t &a_pq = a[ip][iq];
				real_t &eigval_p = r_eig_val[ip];
				real_t &eigval_q = r_eig_val[iq];

				real_t abs_a_pq = Math::abs(a_pq);
				real_t g = 100.0 * abs_a_pq;

				// After four sweeps, skip the rotation if the off-diagonal element is small
				if (sweep > 4 && Math::abs(eigval_p) + g == Math::abs(eigval_p) && Math::abs(eigval_q) + g == Math::abs(eigval_q)) {
					a_pq = 0.0;
				} else if (abs_a_pq > thresh) {
					real_t h = eigval_q - eigval_p;
					real_t abs_h = Math::abs(h);

					real_t t;
					if (abs_h + g == abs_h) {
						t = a_pq / h;
					} else {
						real_t theta = 0.5 * h / a_pq; // Warning: Can become infinite if a(ip, iq) is very small which may trigger an invalid float exception
						t = 1.0 / (Math::abs(theta) + Math::sqrt(1.0 + theta * theta)); // If theta becomes inf, t will be 0 so the infinite is not a problem for the algorithm
						if (theta < 0.0f) {
							t = -t;
						}
					}
					real_t c = 1.0 / Math::sqrt(1.0 + t * t);
					real_t s = t * c;
					real_t tau = s / (1.0 + c);
					h = t * a_pq;

					a_pq = 0.0;

					z[ip] -= h;
					z[iq] += h;

					eigval_p -= h;
					eigval_q += h;

#define GODOT_EVS_ROTATE(m_a, m_i, m_j, m_k, m_l) \
	g = m_a[m_i][m_j];                            \
	h = m_a[m_k][m_l];                            \
	m_a[m_i][m_j] = g - s * (h + g * tau);        \
	m_a[m_k][m_l] = h + s * (g - h * tau);

					int j;
					for (j = 0; j < ip; j++) {
						GODOT_EVS_ROTATE(a, j, ip, j, iq);
					}
					for (j = ip + 1; j < iq; j++) {
						GODOT_EVS_ROTATE(a, ip, j, j, iq);
					}
					for (j = iq + 1; j < N; j++) {
						GODOT_EVS_ROTATE(a, ip, j, iq, j);
					}
					for (j = 0; j < N; j++) {
						GODOT_EVS_ROTATE(r_eig_vec, j, ip, j, iq);
					}

#undef GODOT_EVS_ROTATE
				}
			}
		}
		// Update eigenvalues with the sum of ta_pq and reinitialize z
		for (int ip = 0; ip < N; ip++) {
			b[ip] += z[ip];
			r_eig_val[ip] = b[ip];
			z[ip] = 0.0;
		}
	}

	// Too many iterations
	return false;
}
