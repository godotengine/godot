// This is new code produced by Erik Scott in the 'products of inertia' branch
// Should this be integrated into the Godot Engine, please add appropriate 
// copyright and license information consistent with other source files
// License: MIT

// This code is simply a re-implementation of the same function from Jolt Physics, 
// but using Godot's Vector3 and Basis structures

#pragma once

#include "core/math/vector3.h"
#include "core/math/basis.h"

bool eigen_value_symmetric(const Basis &in_matrix, Basis &out_eig_vec, Vector3 &out_eig_val) {

    const int MAX_SWEEPS = 50;
    const int N = 3;

    Basis a = in_matrix;

    Vector3 b, z;

    for (int ip = 0; ip < N; ++ip) {
        // Initialize b and output to diagonal of a
        b[ip] = a[ip][ip];
        out_eig_val[ip] = a[ip][ip];

        // reset z
        z[ip] = 0.0;
    }

    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {

        // sum the off-diagonal elements of a
        real_t sm = 0.0;
        for (int ip = 0; ip < N; ++ip){
            for (int iq = ip +1; iq < N; ++iq){
                sm += abs(a[ip][iq]);
            }
        }
        real_t avg_sm = sm/(N*N);

        // Normal return
        if (avg_sm < CMP_EPSILON2) {
            return true;
        }

        real_t thresh = sweep < 4? 0.2*avg_sm : CMP_EPSILON2;

        for (int ip = 0; ip < N-1; ++ip) {
            for (int iq = ip + 1; iq < N; ++iq) {

                real_t &a_pq = a[ip][iq];
                real_t &eigval_p = out_eig_val[ip];
                real_t &eigval_q = out_eig_val[iq];

                real_t abs_a_pq = abs(a_pq);
                real_t g = 100.0 * abs_a_pq;

                // After four sweeps, skip the rotation if the off-diagonal element is small
				if (sweep > 4
					&& abs(eigval_p) + g == abs(eigval_p)
					&& abs(eigval_q) + g == abs(eigval_q)) {
					a_pq = 0.0;
				} else if (abs_a_pq > thresh) {
					real_t h = eigval_q - eigval_p;
					real_t abs_h = abs(h);

                    real_t t;
                    if (abs_h + g == abs_h)	{
						t = a_pq / h;
					} else {
						real_t theta = 0.5 * h / a_pq; // Warning: Can become infinite if a(ip, iq) is very small which may trigger an invalid float exception
						t = 1.0 / (abs(theta) + sqrt(1.0 + theta * theta)); // If theta becomes inf, t will be 0 so the infinite is not a problem for the algorithm
						if (theta < 0.0f) {
                            t = -t;
                        }
					}
                    real_t c = 1.0 / sqrt(1.0 + t * t);
					real_t s = t * c;
					real_t tau = s / (1.0 + c);
					h = t * a_pq;

                    a_pq = 0.0;

					z[ip] -= h;
					z[iq] += h;

					eigval_p -= h;
					eigval_q += h;

                    #define JPH_EVS_ROTATE(a, i, j, k, l)		\
						g = a[i][j],							\
						h = a[k][l],							\
						a[i][j] = g - s * (h + g * tau),		\
						a[k][l] = h + s * (g - h * tau)
                    
                    int j;
					for (j = 0; j < ip; ++j)		JPH_EVS_ROTATE(a, j, ip, j, iq);
					for (j = ip + 1; j < iq; ++j)	JPH_EVS_ROTATE(a, ip, j, j, iq);
					for (j = iq + 1; j < N; ++j)	JPH_EVS_ROTATE(a, ip, j, iq, j);
					for (j = 0; j < N; ++j)			JPH_EVS_ROTATE(out_eig_vec, j, ip, j, iq);

					#undef JPH_EVS_ROTATE

                }
            }
        }
        // Update eigenvalues with the sum of ta_pq and reinitialize z
		for (int ip = 0; ip < N; ++ip)
		{
			b[ip] += z[ip];
			out_eig_val[ip] = b[ip];
			z[ip] = 0.0;
		}

    }

    // Too many iterations
    return false;

}