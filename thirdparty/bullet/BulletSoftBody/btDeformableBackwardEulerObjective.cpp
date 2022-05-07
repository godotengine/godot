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

#include "btDeformableBackwardEulerObjective.h"
#include "btPreconditioner.h"
#include "LinearMath/btQuickprof.h"

btDeformableBackwardEulerObjective::btDeformableBackwardEulerObjective(btAlignedObjectArray<btSoftBody*>& softBodies, const TVStack& backup_v)
	: m_softBodies(softBodies), m_projection(softBodies), m_backupVelocity(backup_v), m_implicit(false)
{
	m_massPreconditioner = new MassPreconditioner(m_softBodies);
	m_KKTPreconditioner = new KKTPreconditioner(m_softBodies, m_projection, m_lf, m_dt, m_implicit);
	m_preconditioner = m_KKTPreconditioner;
}

btDeformableBackwardEulerObjective::~btDeformableBackwardEulerObjective()
{
	delete m_KKTPreconditioner;
	delete m_massPreconditioner;
}

void btDeformableBackwardEulerObjective::reinitialize(bool nodeUpdated, btScalar dt)
{
	BT_PROFILE("reinitialize");
	if (dt > 0)
	{
		setDt(dt);
	}
	if (nodeUpdated)
	{
		updateId();
	}
	for (int i = 0; i < m_lf.size(); ++i)
	{
		m_lf[i]->reinitialize(nodeUpdated);
	}
	btMatrix3x3 I;
	I.setIdentity();
	for (int i = 0; i < m_softBodies.size(); ++i)
	{
		btSoftBody* psb = m_softBodies[i];
		for (int j = 0; j < psb->m_nodes.size(); ++j)
		{
			if (psb->m_nodes[j].m_im > 0)
				psb->m_nodes[j].m_effectiveMass = I * (1.0 / psb->m_nodes[j].m_im);
		}
	}
	m_projection.reinitialize(nodeUpdated);
	//    m_preconditioner->reinitialize(nodeUpdated);
}

void btDeformableBackwardEulerObjective::setDt(btScalar dt)
{
	m_dt = dt;
}

void btDeformableBackwardEulerObjective::multiply(const TVStack& x, TVStack& b) const
{
	BT_PROFILE("multiply");
	// add in the mass term
	size_t counter = 0;
	for (int i = 0; i < m_softBodies.size(); ++i)
	{
		btSoftBody* psb = m_softBodies[i];
		for (int j = 0; j < psb->m_nodes.size(); ++j)
		{
			const btSoftBody::Node& node = psb->m_nodes[j];
			b[counter] = (node.m_im == 0) ? btVector3(0, 0, 0) : x[counter] / node.m_im;
			++counter;
		}
	}

	for (int i = 0; i < m_lf.size(); ++i)
	{
		// add damping matrix
		m_lf[i]->addScaledDampingForceDifferential(-m_dt, x, b);
        // Always integrate picking force implicitly for stability.
        if (m_implicit || m_lf[i]->getForceType() == BT_MOUSE_PICKING_FORCE)
		{
			m_lf[i]->addScaledElasticForceDifferential(-m_dt * m_dt, x, b);
		}
	}
	int offset = m_nodes.size();
	for (int i = offset; i < b.size(); ++i)
	{
		b[i].setZero();
	}
	// add in the lagrange multiplier terms

	for (int c = 0; c < m_projection.m_lagrangeMultipliers.size(); ++c)
	{
		// C^T * lambda
		const LagrangeMultiplier& lm = m_projection.m_lagrangeMultipliers[c];
		for (int i = 0; i < lm.m_num_nodes; ++i)
		{
			for (int j = 0; j < lm.m_num_constraints; ++j)
			{
				b[lm.m_indices[i]] += x[offset + c][j] * lm.m_weights[i] * lm.m_dirs[j];
			}
		}
		// C * x
		for (int d = 0; d < lm.m_num_constraints; ++d)
		{
			for (int i = 0; i < lm.m_num_nodes; ++i)
			{
				b[offset + c][d] += lm.m_weights[i] * x[lm.m_indices[i]].dot(lm.m_dirs[d]);
			}
		}
	}
}

void btDeformableBackwardEulerObjective::updateVelocity(const TVStack& dv)
{
	for (int i = 0; i < m_softBodies.size(); ++i)
	{
		btSoftBody* psb = m_softBodies[i];
		for (int j = 0; j < psb->m_nodes.size(); ++j)
		{
			btSoftBody::Node& node = psb->m_nodes[j];
			node.m_v = m_backupVelocity[node.index] + dv[node.index];
		}
	}
}

void btDeformableBackwardEulerObjective::applyForce(TVStack& force, bool setZero)
{
	size_t counter = 0;
	for (int i = 0; i < m_softBodies.size(); ++i)
	{
		btSoftBody* psb = m_softBodies[i];
		if (!psb->isActive())
		{
			counter += psb->m_nodes.size();
			continue;
		}
		if (m_implicit)
		{
			for (int j = 0; j < psb->m_nodes.size(); ++j)
			{
				if (psb->m_nodes[j].m_im != 0)
				{
					psb->m_nodes[j].m_v += psb->m_nodes[j].m_effectiveMass_inv * force[counter++];
				}
			}
		}
		else
		{
			for (int j = 0; j < psb->m_nodes.size(); ++j)
			{
				btScalar one_over_mass = (psb->m_nodes[j].m_im == 0) ? 0 : psb->m_nodes[j].m_im;
				psb->m_nodes[j].m_v += one_over_mass * force[counter++];
			}
		}
	}
	if (setZero)
	{
		for (int i = 0; i < force.size(); ++i)
			force[i].setZero();
	}
}

void btDeformableBackwardEulerObjective::computeResidual(btScalar dt, TVStack& residual)
{
	BT_PROFILE("computeResidual");
	// add implicit force
	for (int i = 0; i < m_lf.size(); ++i)
	{
        // Always integrate picking force implicitly for stability.
		if (m_implicit || m_lf[i]->getForceType() == BT_MOUSE_PICKING_FORCE)
		{
			m_lf[i]->addScaledForces(dt, residual);
		}
		else
		{
			m_lf[i]->addScaledDampingForce(dt, residual);
		}
	}
	//    m_projection.project(residual);
}

btScalar btDeformableBackwardEulerObjective::computeNorm(const TVStack& residual) const
{
	btScalar mag = 0;
	for (int i = 0; i < residual.size(); ++i)
	{
		mag += residual[i].length2();
	}
	return std::sqrt(mag);
}

btScalar btDeformableBackwardEulerObjective::totalEnergy(btScalar dt)
{
	btScalar e = 0;
	for (int i = 0; i < m_lf.size(); ++i)
	{
		e += m_lf[i]->totalEnergy(dt);
	}
	return e;
}

void btDeformableBackwardEulerObjective::applyExplicitForce(TVStack& force)
{
	for (int i = 0; i < m_softBodies.size(); ++i)
	{
		m_softBodies[i]->advanceDeformation();
	}
	if (m_implicit)
	{
		// apply forces except gravity force
		btVector3 gravity;
		for (int i = 0; i < m_lf.size(); ++i)
		{
			if (m_lf[i]->getForceType() == BT_GRAVITY_FORCE)
			{
				gravity = static_cast<btDeformableGravityForce*>(m_lf[i])->m_gravity;
			}
			else
			{
				m_lf[i]->addScaledForces(m_dt, force);
			}
		}
		for (int i = 0; i < m_lf.size(); ++i)
		{
			m_lf[i]->addScaledHessian(m_dt);
		}
		for (int i = 0; i < m_softBodies.size(); ++i)
		{
			btSoftBody* psb = m_softBodies[i];
			if (psb->isActive())
			{
				for (int j = 0; j < psb->m_nodes.size(); ++j)
				{
					// add gravity explicitly
					psb->m_nodes[j].m_v += m_dt * psb->m_gravityFactor * gravity;
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < m_lf.size(); ++i)
		{
			m_lf[i]->addScaledExplicitForce(m_dt, force);
		}
	}
	// calculate inverse mass matrix for all nodes
	for (int i = 0; i < m_softBodies.size(); ++i)
	{
		btSoftBody* psb = m_softBodies[i];
		if (psb->isActive())
		{
			for (int j = 0; j < psb->m_nodes.size(); ++j)
			{
				if (psb->m_nodes[j].m_im > 0)
				{
					psb->m_nodes[j].m_effectiveMass_inv = psb->m_nodes[j].m_effectiveMass.inverse();
				}
			}
		}
	}
	applyForce(force, true);
}

void btDeformableBackwardEulerObjective::initialGuess(TVStack& dv, const TVStack& residual)
{
	size_t counter = 0;
	for (int i = 0; i < m_softBodies.size(); ++i)
	{
		btSoftBody* psb = m_softBodies[i];
		for (int j = 0; j < psb->m_nodes.size(); ++j)
		{
			dv[counter] = psb->m_nodes[j].m_im * residual[counter];
			++counter;
		}
	}
}

//set constraints as projections
void btDeformableBackwardEulerObjective::setConstraints(const btContactSolverInfo& infoGlobal)
{
	m_projection.setConstraints(infoGlobal);
}

void btDeformableBackwardEulerObjective::applyDynamicFriction(TVStack& r)
{
	m_projection.applyDynamicFriction(r);
}
