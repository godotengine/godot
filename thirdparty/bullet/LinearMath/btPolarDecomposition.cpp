#include "btPolarDecomposition.h"
#include "btMinMax.h"

namespace
{
btScalar abs_column_sum(const btMatrix3x3& a, int i)
{
	return btFabs(a[0][i]) + btFabs(a[1][i]) + btFabs(a[2][i]);
}

btScalar abs_row_sum(const btMatrix3x3& a, int i)
{
	return btFabs(a[i][0]) + btFabs(a[i][1]) + btFabs(a[i][2]);
}

btScalar p1_norm(const btMatrix3x3& a)
{
	const btScalar sum0 = abs_column_sum(a, 0);
	const btScalar sum1 = abs_column_sum(a, 1);
	const btScalar sum2 = abs_column_sum(a, 2);
	return btMax(btMax(sum0, sum1), sum2);
}

btScalar pinf_norm(const btMatrix3x3& a)
{
	const btScalar sum0 = abs_row_sum(a, 0);
	const btScalar sum1 = abs_row_sum(a, 1);
	const btScalar sum2 = abs_row_sum(a, 2);
	return btMax(btMax(sum0, sum1), sum2);
}
}  // namespace

btPolarDecomposition::btPolarDecomposition(btScalar tolerance, unsigned int maxIterations)
	: m_tolerance(tolerance), m_maxIterations(maxIterations)
{
}

unsigned int btPolarDecomposition::decompose(const btMatrix3x3& a, btMatrix3x3& u, btMatrix3x3& h) const
{
	// Use the 'u' and 'h' matrices for intermediate calculations
	u = a;
	h = a.inverse();

	for (unsigned int i = 0; i < m_maxIterations; ++i)
	{
		const btScalar h_1 = p1_norm(h);
		const btScalar h_inf = pinf_norm(h);
		const btScalar u_1 = p1_norm(u);
		const btScalar u_inf = pinf_norm(u);

		const btScalar h_norm = h_1 * h_inf;
		const btScalar u_norm = u_1 * u_inf;

		// The matrix is effectively singular so we cannot invert it
		if (btFuzzyZero(h_norm) || btFuzzyZero(u_norm))
			break;

		const btScalar gamma = btPow(h_norm / u_norm, 0.25f);
		const btScalar inv_gamma = btScalar(1.0) / gamma;

		// Determine the delta to 'u'
		const btMatrix3x3 delta = (u * (gamma - btScalar(2.0)) + h.transpose() * inv_gamma) * btScalar(0.5);

		// Update the matrices
		u += delta;
		h = u.inverse();

		// Check for convergence
		if (p1_norm(delta) <= m_tolerance * u_1)
		{
			h = u.transpose() * a;
			h = (h + h.transpose()) * 0.5;
			return i;
		}
	}

	// The algorithm has failed to converge to the specified tolerance, but we
	// want to make sure that the matrices returned are in the right form.
	h = u.transpose() * a;
	h = (h + h.transpose()) * 0.5;

	return m_maxIterations;
}

unsigned int btPolarDecomposition::maxIterations() const
{
	return m_maxIterations;
}

unsigned int polarDecompose(const btMatrix3x3& a, btMatrix3x3& u, btMatrix3x3& h)
{
	static btPolarDecomposition polar;
	return polar.decompose(a, u, h);
}
