/*
Written by Xuchen Han <xuchenhan2015@u.northwestern.edu>

Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2019 Google Inc. http://bulletphysics.org
This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:
1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_NEOHOOKEAN_H
#define BT_NEOHOOKEAN_H

#include "btDeformableLagrangianForce.h"
#include "LinearMath/btQuickprof.h"
#include "LinearMath/btImplicitQRSVD.h"
// This energy is as described in https://graphics.pixar.com/library/StableElasticity/paper.pdf
class btDeformableNeoHookeanForce : public btDeformableLagrangianForce
{
public:
	typedef btAlignedObjectArray<btVector3> TVStack;
	btScalar m_mu, m_lambda;  // Lame Parameters
	btScalar m_E, m_nu;       // Young's modulus and Poisson ratio
	btScalar m_mu_damp, m_lambda_damp;
	btDeformableNeoHookeanForce() : m_mu(1), m_lambda(1)
	{
		btScalar damping = 0.05;
		m_mu_damp = damping * m_mu;
		m_lambda_damp = damping * m_lambda;
		updateYoungsModulusAndPoissonRatio();
	}

	btDeformableNeoHookeanForce(btScalar mu, btScalar lambda, btScalar damping = 0.05) : m_mu(mu), m_lambda(lambda)
	{
		m_mu_damp = damping * m_mu;
		m_lambda_damp = damping * m_lambda;
		updateYoungsModulusAndPoissonRatio();
	}

	void updateYoungsModulusAndPoissonRatio()
	{
		// conversion from Lame Parameters to Young's modulus and Poisson ratio
		// https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
		m_E = m_mu * (3 * m_lambda + 2 * m_mu) / (m_lambda + m_mu);
		m_nu = m_lambda * 0.5 / (m_mu + m_lambda);
	}

	void updateLameParameters()
	{
		// conversion from Young's modulus and Poisson ratio to Lame Parameters
		// https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
		m_mu = m_E * 0.5 / (1 + m_nu);
		m_lambda = m_E * m_nu / ((1 + m_nu) * (1 - 2 * m_nu));
	}

	void setYoungsModulus(btScalar E)
	{
		m_E = E;
		updateLameParameters();
	}

	void setPoissonRatio(btScalar nu)
	{
		m_nu = nu;
		updateLameParameters();
	}

	void setDamping(btScalar damping)
	{
		m_mu_damp = damping * m_mu;
		m_lambda_damp = damping * m_lambda;
	}

	void setLameParameters(btScalar mu, btScalar lambda)
	{
		m_mu = mu;
		m_lambda = lambda;
		updateYoungsModulusAndPoissonRatio();
	}

	virtual void addScaledForces(btScalar scale, TVStack& force)
	{
		addScaledDampingForce(scale, force);
		addScaledElasticForce(scale, force);
	}

	virtual void addScaledExplicitForce(btScalar scale, TVStack& force)
	{
		addScaledElasticForce(scale, force);
	}

	// The damping matrix is calculated using the time n state as described in https://www.math.ucla.edu/~jteran/papers/GSSJT15.pdf to allow line search
	virtual void addScaledDampingForce(btScalar scale, TVStack& force)
	{
		if (m_mu_damp == 0 && m_lambda_damp == 0)
			return;
		int numNodes = getNumNodes();
		btAssert(numNodes <= force.size());
		btVector3 grad_N_hat_1st_col = btVector3(-1, -1, -1);
		for (int i = 0; i < m_softBodies.size(); ++i)
		{
			btSoftBody* psb = m_softBodies[i];
			if (!psb->isActive())
			{
				continue;
			}
			for (int j = 0; j < psb->m_tetras.size(); ++j)
			{
				btSoftBody::Tetra& tetra = psb->m_tetras[j];
				btSoftBody::Node* node0 = tetra.m_n[0];
				btSoftBody::Node* node1 = tetra.m_n[1];
				btSoftBody::Node* node2 = tetra.m_n[2];
				btSoftBody::Node* node3 = tetra.m_n[3];
				size_t id0 = node0->index;
				size_t id1 = node1->index;
				size_t id2 = node2->index;
				size_t id3 = node3->index;
				btMatrix3x3 dF = DsFromVelocity(node0, node1, node2, node3) * tetra.m_Dm_inverse;
				btMatrix3x3 I;
				I.setIdentity();
				btMatrix3x3 dP = (dF + dF.transpose()) * m_mu_damp + I * (dF[0][0] + dF[1][1] + dF[2][2]) * m_lambda_damp;
				//                firstPiolaDampingDifferential(psb->m_tetraScratchesTn[j], dF, dP);
				btVector3 df_on_node0 = dP * (tetra.m_Dm_inverse.transpose() * grad_N_hat_1st_col);
				btMatrix3x3 df_on_node123 = dP * tetra.m_Dm_inverse.transpose();

				// damping force differential
				btScalar scale1 = scale * tetra.m_element_measure;
				force[id0] -= scale1 * df_on_node0;
				force[id1] -= scale1 * df_on_node123.getColumn(0);
				force[id2] -= scale1 * df_on_node123.getColumn(1);
				force[id3] -= scale1 * df_on_node123.getColumn(2);
			}
		}
	}

	virtual double totalElasticEnergy(btScalar dt)
	{
		double energy = 0;
		for (int i = 0; i < m_softBodies.size(); ++i)
		{
			btSoftBody* psb = m_softBodies[i];
			if (!psb->isActive())
			{
				continue;
			}
			for (int j = 0; j < psb->m_tetraScratches.size(); ++j)
			{
				btSoftBody::Tetra& tetra = psb->m_tetras[j];
				btSoftBody::TetraScratch& s = psb->m_tetraScratches[j];
				energy += tetra.m_element_measure * elasticEnergyDensity(s);
			}
		}
		return energy;
	}

	// The damping energy is formulated as in https://www.math.ucla.edu/~jteran/papers/GSSJT15.pdf to allow line search
	virtual double totalDampingEnergy(btScalar dt)
	{
		double energy = 0;
		int sz = 0;
		for (int i = 0; i < m_softBodies.size(); ++i)
		{
			btSoftBody* psb = m_softBodies[i];
			if (!psb->isActive())
			{
				continue;
			}
			for (int j = 0; j < psb->m_nodes.size(); ++j)
			{
				sz = btMax(sz, psb->m_nodes[j].index);
			}
		}
		TVStack dampingForce;
		dampingForce.resize(sz + 1);
		for (int i = 0; i < dampingForce.size(); ++i)
			dampingForce[i].setZero();
		addScaledDampingForce(0.5, dampingForce);
		for (int i = 0; i < m_softBodies.size(); ++i)
		{
			btSoftBody* psb = m_softBodies[i];
			for (int j = 0; j < psb->m_nodes.size(); ++j)
			{
				const btSoftBody::Node& node = psb->m_nodes[j];
				energy -= dampingForce[node.index].dot(node.m_v) / dt;
			}
		}
		return energy;
	}

	double elasticEnergyDensity(const btSoftBody::TetraScratch& s)
	{
		double density = 0;
		density += m_mu * 0.5 * (s.m_trace - 3.);
		density += m_lambda * 0.5 * (s.m_J - 1. - 0.75 * m_mu / m_lambda) * (s.m_J - 1. - 0.75 * m_mu / m_lambda);
		density -= m_mu * 0.5 * log(s.m_trace + 1);
		return density;
	}

	virtual void addScaledElasticForce(btScalar scale, TVStack& force)
	{
		int numNodes = getNumNodes();
		btAssert(numNodes <= force.size());
		btVector3 grad_N_hat_1st_col = btVector3(-1, -1, -1);
		for (int i = 0; i < m_softBodies.size(); ++i)
		{
			btSoftBody* psb = m_softBodies[i];
			if (!psb->isActive())
			{
				continue;
			}
			btScalar max_p = psb->m_cfg.m_maxStress;
			for (int j = 0; j < psb->m_tetras.size(); ++j)
			{
				btSoftBody::Tetra& tetra = psb->m_tetras[j];
				btMatrix3x3 P;
				firstPiola(psb->m_tetraScratches[j], P);
#ifdef USE_SVD
				if (max_p > 0)
				{
					// since we want to clamp the principal stress to max_p, we only need to
					// calculate SVD when sigma_0^2 + sigma_1^2 + sigma_2^2 > max_p * max_p
					btScalar trPTP = (P[0].length2() + P[1].length2() + P[2].length2());
					if (trPTP > max_p * max_p)
					{
						btMatrix3x3 U, V;
						btVector3 sigma;
						singularValueDecomposition(P, U, sigma, V);
						sigma[0] = btMin(sigma[0], max_p);
						sigma[1] = btMin(sigma[1], max_p);
						sigma[2] = btMin(sigma[2], max_p);
						sigma[0] = btMax(sigma[0], -max_p);
						sigma[1] = btMax(sigma[1], -max_p);
						sigma[2] = btMax(sigma[2], -max_p);
						btMatrix3x3 Sigma;
						Sigma.setIdentity();
						Sigma[0][0] = sigma[0];
						Sigma[1][1] = sigma[1];
						Sigma[2][2] = sigma[2];
						P = U * Sigma * V.transpose();
					}
				}
#endif
				//                btVector3 force_on_node0 = P * (tetra.m_Dm_inverse.transpose()*grad_N_hat_1st_col);
				btMatrix3x3 force_on_node123 = P * tetra.m_Dm_inverse.transpose();
				btVector3 force_on_node0 = force_on_node123 * grad_N_hat_1st_col;

				btSoftBody::Node* node0 = tetra.m_n[0];
				btSoftBody::Node* node1 = tetra.m_n[1];
				btSoftBody::Node* node2 = tetra.m_n[2];
				btSoftBody::Node* node3 = tetra.m_n[3];
				size_t id0 = node0->index;
				size_t id1 = node1->index;
				size_t id2 = node2->index;
				size_t id3 = node3->index;

				// elastic force
				btScalar scale1 = scale * tetra.m_element_measure;
				force[id0] -= scale1 * force_on_node0;
				force[id1] -= scale1 * force_on_node123.getColumn(0);
				force[id2] -= scale1 * force_on_node123.getColumn(1);
				force[id3] -= scale1 * force_on_node123.getColumn(2);
			}
		}
	}

	// The damping matrix is calculated using the time n state as described in https://www.math.ucla.edu/~jteran/papers/GSSJT15.pdf to allow line search
	virtual void addScaledDampingForceDifferential(btScalar scale, const TVStack& dv, TVStack& df)
	{
		if (m_mu_damp == 0 && m_lambda_damp == 0)
			return;
		int numNodes = getNumNodes();
		btAssert(numNodes <= df.size());
		btVector3 grad_N_hat_1st_col = btVector3(-1, -1, -1);
		for (int i = 0; i < m_softBodies.size(); ++i)
		{
			btSoftBody* psb = m_softBodies[i];
			if (!psb->isActive())
			{
				continue;
			}
			for (int j = 0; j < psb->m_tetras.size(); ++j)
			{
				btSoftBody::Tetra& tetra = psb->m_tetras[j];
				btSoftBody::Node* node0 = tetra.m_n[0];
				btSoftBody::Node* node1 = tetra.m_n[1];
				btSoftBody::Node* node2 = tetra.m_n[2];
				btSoftBody::Node* node3 = tetra.m_n[3];
				size_t id0 = node0->index;
				size_t id1 = node1->index;
				size_t id2 = node2->index;
				size_t id3 = node3->index;
				btMatrix3x3 dF = Ds(id0, id1, id2, id3, dv) * tetra.m_Dm_inverse;
				btMatrix3x3 I;
				I.setIdentity();
				btMatrix3x3 dP = (dF + dF.transpose()) * m_mu_damp + I * (dF[0][0] + dF[1][1] + dF[2][2]) * m_lambda_damp;
				//                firstPiolaDampingDifferential(psb->m_tetraScratchesTn[j], dF, dP);
				//                btVector3 df_on_node0 = dP * (tetra.m_Dm_inverse.transpose()*grad_N_hat_1st_col);
				btMatrix3x3 df_on_node123 = dP * tetra.m_Dm_inverse.transpose();
				btVector3 df_on_node0 = df_on_node123 * grad_N_hat_1st_col;

				// damping force differential
				btScalar scale1 = scale * tetra.m_element_measure;
				df[id0] -= scale1 * df_on_node0;
				df[id1] -= scale1 * df_on_node123.getColumn(0);
				df[id2] -= scale1 * df_on_node123.getColumn(1);
				df[id3] -= scale1 * df_on_node123.getColumn(2);
			}
		}
	}

	virtual void buildDampingForceDifferentialDiagonal(btScalar scale, TVStack& diagA) {}

	virtual void addScaledElasticForceDifferential(btScalar scale, const TVStack& dx, TVStack& df)
	{
		int numNodes = getNumNodes();
		btAssert(numNodes <= df.size());
		btVector3 grad_N_hat_1st_col = btVector3(-1, -1, -1);
		for (int i = 0; i < m_softBodies.size(); ++i)
		{
			btSoftBody* psb = m_softBodies[i];
			if (!psb->isActive())
			{
				continue;
			}
			for (int j = 0; j < psb->m_tetras.size(); ++j)
			{
				btSoftBody::Tetra& tetra = psb->m_tetras[j];
				btSoftBody::Node* node0 = tetra.m_n[0];
				btSoftBody::Node* node1 = tetra.m_n[1];
				btSoftBody::Node* node2 = tetra.m_n[2];
				btSoftBody::Node* node3 = tetra.m_n[3];
				size_t id0 = node0->index;
				size_t id1 = node1->index;
				size_t id2 = node2->index;
				size_t id3 = node3->index;
				btMatrix3x3 dF = Ds(id0, id1, id2, id3, dx) * tetra.m_Dm_inverse;
				btMatrix3x3 dP;
				firstPiolaDifferential(psb->m_tetraScratches[j], dF, dP);
				//                btVector3 df_on_node0 = dP * (tetra.m_Dm_inverse.transpose()*grad_N_hat_1st_col);
				btMatrix3x3 df_on_node123 = dP * tetra.m_Dm_inverse.transpose();
				btVector3 df_on_node0 = df_on_node123 * grad_N_hat_1st_col;

				// elastic force differential
				btScalar scale1 = scale * tetra.m_element_measure;
				df[id0] -= scale1 * df_on_node0;
				df[id1] -= scale1 * df_on_node123.getColumn(0);
				df[id2] -= scale1 * df_on_node123.getColumn(1);
				df[id3] -= scale1 * df_on_node123.getColumn(2);
			}
		}
	}

	void firstPiola(const btSoftBody::TetraScratch& s, btMatrix3x3& P)
	{
		btScalar c1 = (m_mu * (1. - 1. / (s.m_trace + 1.)));
		btScalar c2 = (m_lambda * (s.m_J - 1.) - 0.75 * m_mu);
		P = s.m_F * c1 + s.m_cofF * c2;
	}

	// Let P be the first piola stress.
	// This function calculates the dP = dP/dF * dF
	void firstPiolaDifferential(const btSoftBody::TetraScratch& s, const btMatrix3x3& dF, btMatrix3x3& dP)
	{
		btScalar c1 = m_mu * (1. - 1. / (s.m_trace + 1.));
		btScalar c2 = (2. * m_mu) * DotProduct(s.m_F, dF) * (1. / ((1. + s.m_trace) * (1. + s.m_trace)));
		btScalar c3 = (m_lambda * DotProduct(s.m_cofF, dF));
		dP = dF * c1 + s.m_F * c2;
		addScaledCofactorMatrixDifferential(s.m_F, dF, m_lambda * (s.m_J - 1.) - 0.75 * m_mu, dP);
		dP += s.m_cofF * c3;
	}

	// Let Q be the damping stress.
	// This function calculates the dP = dQ/dF * dF
	void firstPiolaDampingDifferential(const btSoftBody::TetraScratch& s, const btMatrix3x3& dF, btMatrix3x3& dP)
	{
		btScalar c1 = (m_mu_damp * (1. - 1. / (s.m_trace + 1.)));
		btScalar c2 = ((2. * m_mu_damp) * DotProduct(s.m_F, dF) * (1. / ((1. + s.m_trace) * (1. + s.m_trace))));
		btScalar c3 = (m_lambda_damp * DotProduct(s.m_cofF, dF));
		dP = dF * c1 + s.m_F * c2;
		addScaledCofactorMatrixDifferential(s.m_F, dF, m_lambda_damp * (s.m_J - 1.) - 0.75 * m_mu_damp, dP);
		dP += s.m_cofF * c3;
	}

	btScalar DotProduct(const btMatrix3x3& A, const btMatrix3x3& B)
	{
		btScalar ans = 0;
		for (int i = 0; i < 3; ++i)
		{
			ans += A[i].dot(B[i]);
		}
		return ans;
	}

	// Let C(A) be the cofactor of the matrix A
	// Let H = the derivative of C(A) with respect to A evaluated at F = A
	// This function calculates H*dF
	void addScaledCofactorMatrixDifferential(const btMatrix3x3& F, const btMatrix3x3& dF, btScalar scale, btMatrix3x3& M)
	{
		M[0][0] += scale * (dF[1][1] * F[2][2] + F[1][1] * dF[2][2] - dF[2][1] * F[1][2] - F[2][1] * dF[1][2]);
		M[1][0] += scale * (dF[2][1] * F[0][2] + F[2][1] * dF[0][2] - dF[0][1] * F[2][2] - F[0][1] * dF[2][2]);
		M[2][0] += scale * (dF[0][1] * F[1][2] + F[0][1] * dF[1][2] - dF[1][1] * F[0][2] - F[1][1] * dF[0][2]);
		M[0][1] += scale * (dF[2][0] * F[1][2] + F[2][0] * dF[1][2] - dF[1][0] * F[2][2] - F[1][0] * dF[2][2]);
		M[1][1] += scale * (dF[0][0] * F[2][2] + F[0][0] * dF[2][2] - dF[2][0] * F[0][2] - F[2][0] * dF[0][2]);
		M[2][1] += scale * (dF[1][0] * F[0][2] + F[1][0] * dF[0][2] - dF[0][0] * F[1][2] - F[0][0] * dF[1][2]);
		M[0][2] += scale * (dF[1][0] * F[2][1] + F[1][0] * dF[2][1] - dF[2][0] * F[1][1] - F[2][0] * dF[1][1]);
		M[1][2] += scale * (dF[2][0] * F[0][1] + F[2][0] * dF[0][1] - dF[0][0] * F[2][1] - F[0][0] * dF[2][1]);
		M[2][2] += scale * (dF[0][0] * F[1][1] + F[0][0] * dF[1][1] - dF[1][0] * F[0][1] - F[1][0] * dF[0][1]);
	}

	virtual btDeformableLagrangianForceType getForceType()
	{
		return BT_NEOHOOKEAN_FORCE;
	}
};
#endif /* BT_NEOHOOKEAN_H */
