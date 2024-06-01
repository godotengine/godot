// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Core/FPFlushDenormals.h>

JPH_NAMESPACE_BEGIN

/// Function to determine the eigen vectors and values of a N x N real symmetric matrix
/// by Jacobi transformations. This method is most suitable for N < 10.
///
/// Taken and adapted from Numerical Recipies paragraph 11.1
///
/// An eigen vector is a vector v for which \f$A \: v = \lambda \: v\f$
///
/// Where:
/// A: A square matrix.
/// \f$\lambda\f$: a non-zero constant value.
///
/// @see https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
///
/// Matrix is a matrix type, which has dimensions N x N.
/// @param inMatrix is the matrix of which to return the eigenvalues and vectors
/// @param outEigVec will contain a matrix whose columns contain the normalized eigenvectors (must be identity before call)
/// @param outEigVal will contain the eigenvalues
template <class Vector, class Matrix>
bool EigenValueSymmetric(const Matrix &inMatrix, Matrix &outEigVec, Vector &outEigVal)
{
	// This algorithm works with very small numbers and can trigger invalid float exceptions when not flushing denormals
	FPFlushDenormals flush_denormals;
	(void)flush_denormals;

	// Maximum number of sweeps to make
	const int cMaxSweeps = 50;

	// Get problem dimension
	const uint n = inMatrix.GetRows();

	// Make sure the dimensions are right
	JPH_ASSERT(inMatrix.GetRows() == n);
	JPH_ASSERT(inMatrix.GetCols() == n);
	JPH_ASSERT(outEigVec.GetRows() == n);
	JPH_ASSERT(outEigVec.GetCols() == n);
	JPH_ASSERT(outEigVal.GetRows() == n);
	JPH_ASSERT(outEigVec.IsIdentity());

	// Get the matrix in a so we can mess with it
	Matrix a = inMatrix;

	Vector b, z;

	for (uint ip = 0; ip < n; ++ip)
	{
		// Initialize b to diagonal of a
		b[ip] = a(ip, ip);

		// Initialize output to diagonal of a
		outEigVal[ip] = a(ip, ip);

		// Reset z
		z[ip] = 0.0f;
	}

	for (int sweep = 0; sweep < cMaxSweeps; ++sweep)
	{
		// Get the sum of the off-diagonal elements of a
		float sm = 0.0f;
		for (uint ip = 0; ip < n - 1; ++ip)
			for (uint iq = ip + 1; iq < n; ++iq)
				sm += abs(a(ip, iq));

		// Normal return, convergence to machine underflow
		if (sm == 0.0f)
		{
			// Sanity checks
			#ifdef JPH_ENABLE_ASSERTS
				for (uint c = 0; c < n; ++c)
				{
					// Check if the eigenvector is normalized
					JPH_ASSERT(outEigVec.GetColumn(c).IsNormalized());

					// Check if inMatrix * eigen_vector = eigen_value * eigen_vector
					Vector mat_eigvec = inMatrix * outEigVec.GetColumn(c);
					Vector eigval_eigvec = outEigVal[c] * outEigVec.GetColumn(c);
					JPH_ASSERT(mat_eigvec.IsClose(eigval_eigvec, max(mat_eigvec.LengthSq(), eigval_eigvec.LengthSq()) * 1.0e-6f));
				}
			#endif

			// Success
			return true;
		}

		// On the first three sweeps use a fraction of the sum of the off diagonal elements as threshold
		float tresh = sweep < 4? 0.2f * sm / Square(n) : 0.0f;

		for (uint ip = 0; ip < n - 1; ++ip)
			for (uint iq = ip + 1; iq < n; ++iq)
			{
				float g = 100.0f * abs(a(ip, iq));

				// After four sweeps, skip the rotation if the off-diagonal element is small
				if (sweep > 4
					&& abs(outEigVal[ip]) + g == abs(outEigVal[ip])
					&& abs(outEigVal[iq]) + g == abs(outEigVal[iq]))
				{
					a(ip, iq) = 0.0f;
				}
				else if (abs(a(ip, iq)) > tresh)
				{
					float h = outEigVal[iq] - outEigVal[ip];

					float t;
					if (abs(h) + g == abs(h))
					{
						t = a(ip, iq) / h;
					}
					else
					{
						float theta = 0.5f * h / a(ip, iq); // Warning: Can become inf if a(ip, iq) too small
						t = 1.0f / (abs(theta) + sqrt(1.0f + theta * theta)); // Warning: Squaring large value can make it inf
						if (theta < 0.0f) t = -t;
					}

					float c = 1.0f / sqrt(1.0f + t * t);
					float s = t * c;
					float tau = s / (1.0f + c);
					h = t * a(ip, iq);

					a(ip, iq) = 0.0f;

					// !Modification from Numerical Recipes!
					// h can become infinite due to numerical overflow, this only happens when a(ip, iq) is very small
					// so we can safely set a(ip, iq) to zero and skip the rotation, see lines marked with 'Warning' above.
					if (!isnan(h))
					{
						z[ip] -= h;
						z[iq] += h;

						outEigVal[ip] -= h;
						outEigVal[iq] += h;

						#define JPH_EVS_ROTATE(a, i, j, k, l)		\
							g = a(i, j),							\
							h = a(k, l),							\
							a(i, j) = g - s * (h + g * tau),		\
							a(k, l) = h + s * (g - h * tau)

						uint j;
						for (j = 0; j < ip; ++j)		JPH_EVS_ROTATE(a, j, ip, j, iq);
						for (j = ip + 1; j < iq; ++j)	JPH_EVS_ROTATE(a, ip, j, j, iq);
						for (j = iq + 1; j < n; ++j)	JPH_EVS_ROTATE(a, ip, j, iq, j);
						for (j = 0; j < n; ++j)			JPH_EVS_ROTATE(outEigVec, j, ip, j, iq);

						#undef JPH_EVS_ROTATE
					}
				}
			}

		// Update eigenvalues with the sum of ta_pq and reinitialize z
		for (uint ip = 0; ip < n; ++ip)
		{
			b[ip] += z[ip];
			outEigVal[ip] = b[ip];
			z[ip] = 0.0f;
		}
	}

	// Failure
	JPH_ASSERT(false, "Too many iterations");
	return false;
}

JPH_NAMESPACE_END
