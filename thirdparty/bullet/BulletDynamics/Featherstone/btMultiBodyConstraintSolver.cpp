/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2013 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "btMultiBodyConstraintSolver.h"
#include "BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"
#include "btMultiBodyLinkCollider.h"

#include "BulletDynamics/ConstraintSolver/btSolverBody.h"
#include "btMultiBodyConstraint.h"
#include "BulletDynamics/ConstraintSolver/btContactSolverInfo.h"

#include "LinearMath/btQuickprof.h"
#include "BulletDynamics/Featherstone/btMultiBodySolverConstraint.h"
#include "LinearMath/btScalar.h"

btScalar btMultiBodyConstraintSolver::solveSingleIteration(int iteration, btCollisionObject** bodies, int numBodies, btPersistentManifold** manifoldPtr, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* debugDrawer)
{
	btScalar leastSquaredResidual = btSequentialImpulseConstraintSolver::solveSingleIteration(iteration, bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints, infoGlobal, debugDrawer);

	//solve featherstone non-contact constraints
	btScalar nonContactResidual = 0;
	//printf("m_multiBodyNonContactConstraints = %d\n",m_multiBodyNonContactConstraints.size());
	for (int i = 0; i < infoGlobal.m_numNonContactInnerIterations; ++i)
	{
		// reset the nonContactResdual to 0 at start of each inner iteration
		nonContactResidual = 0;
		for (int j = 0; j < m_multiBodyNonContactConstraints.size(); j++)
		{
			int index = iteration & 1 ? j : m_multiBodyNonContactConstraints.size() - 1 - j;

			btMultiBodySolverConstraint& constraint = m_multiBodyNonContactConstraints[index];

			btScalar residual = resolveSingleConstraintRowGeneric(constraint);
			nonContactResidual = btMax(nonContactResidual, residual * residual);

			if (constraint.m_multiBodyA)
				constraint.m_multiBodyA->setPosUpdated(false);
			if (constraint.m_multiBodyB)
				constraint.m_multiBodyB->setPosUpdated(false);
		}
	}
	leastSquaredResidual = btMax(leastSquaredResidual, nonContactResidual);

	//solve featherstone normal contact
	for (int j0 = 0; j0 < m_multiBodyNormalContactConstraints.size(); j0++)
	{
		int index = j0;  //iteration&1? j0 : m_multiBodyNormalContactConstraints.size()-1-j0;

		btMultiBodySolverConstraint& constraint = m_multiBodyNormalContactConstraints[index];
		btScalar residual = 0.f;

		if (iteration < infoGlobal.m_numIterations)
		{
			residual = resolveSingleConstraintRowGeneric(constraint);
		}

		leastSquaredResidual = btMax(leastSquaredResidual, residual * residual);

		if (constraint.m_multiBodyA)
			constraint.m_multiBodyA->setPosUpdated(false);
		if (constraint.m_multiBodyB)
			constraint.m_multiBodyB->setPosUpdated(false);
	}

	//solve featherstone frictional contact
	if (infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS && ((infoGlobal.m_solverMode & SOLVER_DISABLE_IMPLICIT_CONE_FRICTION) == 0))
	{
		for (int j1 = 0; j1 < this->m_multiBodySpinningFrictionContactConstraints.size(); j1++)
		{
			if (iteration < infoGlobal.m_numIterations)
			{
				int index = j1;

				btMultiBodySolverConstraint& frictionConstraint = m_multiBodySpinningFrictionContactConstraints[index];
				btScalar totalImpulse = m_multiBodyNormalContactConstraints[frictionConstraint.m_frictionIndex].m_appliedImpulse;
				//adjust friction limits here
				if (totalImpulse > btScalar(0))
				{
					frictionConstraint.m_lowerLimit = -(frictionConstraint.m_friction * totalImpulse);
					frictionConstraint.m_upperLimit = frictionConstraint.m_friction * totalImpulse;
					btScalar residual = resolveSingleConstraintRowGeneric(frictionConstraint);
					leastSquaredResidual = btMax(leastSquaredResidual, residual * residual);

					if (frictionConstraint.m_multiBodyA)
						frictionConstraint.m_multiBodyA->setPosUpdated(false);
					if (frictionConstraint.m_multiBodyB)
						frictionConstraint.m_multiBodyB->setPosUpdated(false);
				}
			}
		}

		for (int j1 = 0; j1 < this->m_multiBodyTorsionalFrictionContactConstraints.size(); j1++)
		{
			if (iteration < infoGlobal.m_numIterations)
			{
				int index = j1;  //iteration&1? j1 : m_multiBodyTorsionalFrictionContactConstraints.size()-1-j1;

				btMultiBodySolverConstraint& frictionConstraint = m_multiBodyTorsionalFrictionContactConstraints[index];
				btScalar totalImpulse = m_multiBodyNormalContactConstraints[frictionConstraint.m_frictionIndex].m_appliedImpulse;
				j1++;
				int index2 = j1;
				btMultiBodySolverConstraint& frictionConstraintB = m_multiBodyTorsionalFrictionContactConstraints[index2];
				//adjust friction limits here
				if (totalImpulse > btScalar(0) && frictionConstraint.m_frictionIndex == frictionConstraintB.m_frictionIndex)
				{
					frictionConstraint.m_lowerLimit = -(frictionConstraint.m_friction * totalImpulse);
					frictionConstraint.m_upperLimit = frictionConstraint.m_friction * totalImpulse;
					frictionConstraintB.m_lowerLimit = -(frictionConstraintB.m_friction * totalImpulse);
					frictionConstraintB.m_upperLimit = frictionConstraintB.m_friction * totalImpulse;

					btScalar residual = resolveConeFrictionConstraintRows(frictionConstraint, frictionConstraintB);
					leastSquaredResidual = btMax(leastSquaredResidual, residual * residual);

					if (frictionConstraint.m_multiBodyA)
						frictionConstraint.m_multiBodyA->setPosUpdated(false);
					if (frictionConstraint.m_multiBodyB)
						frictionConstraint.m_multiBodyB->setPosUpdated(false);

					if (frictionConstraintB.m_multiBodyA)
						frictionConstraintB.m_multiBodyA->setPosUpdated(false);
					if (frictionConstraintB.m_multiBodyB)
						frictionConstraintB.m_multiBodyB->setPosUpdated(false);
				}
			}
		}

		for (int j1 = 0; j1 < this->m_multiBodyFrictionContactConstraints.size(); j1++)
		{
			if (iteration < infoGlobal.m_numIterations)
			{
				int index = j1;  //iteration&1? j1 : m_multiBodyFrictionContactConstraints.size()-1-j1;
				btMultiBodySolverConstraint& frictionConstraint = m_multiBodyFrictionContactConstraints[index];

				btScalar totalImpulse = m_multiBodyNormalContactConstraints[frictionConstraint.m_frictionIndex].m_appliedImpulse;
				j1++;
				int index2 = j1;  //iteration&1? j1 : m_multiBodyFrictionContactConstraints.size()-1-j1;
				btMultiBodySolverConstraint& frictionConstraintB = m_multiBodyFrictionContactConstraints[index2];
				btAssert(frictionConstraint.m_frictionIndex == frictionConstraintB.m_frictionIndex);

				if (frictionConstraint.m_frictionIndex == frictionConstraintB.m_frictionIndex)
				{
					frictionConstraint.m_lowerLimit = -(frictionConstraint.m_friction * totalImpulse);
					frictionConstraint.m_upperLimit = frictionConstraint.m_friction * totalImpulse;
					frictionConstraintB.m_lowerLimit = -(frictionConstraintB.m_friction * totalImpulse);
					frictionConstraintB.m_upperLimit = frictionConstraintB.m_friction * totalImpulse;
					btScalar residual = resolveConeFrictionConstraintRows(frictionConstraint, frictionConstraintB);
					leastSquaredResidual = btMax(leastSquaredResidual, residual * residual);

					if (frictionConstraintB.m_multiBodyA)
						frictionConstraintB.m_multiBodyA->setPosUpdated(false);
					if (frictionConstraintB.m_multiBodyB)
						frictionConstraintB.m_multiBodyB->setPosUpdated(false);

					if (frictionConstraint.m_multiBodyA)
						frictionConstraint.m_multiBodyA->setPosUpdated(false);
					if (frictionConstraint.m_multiBodyB)
						frictionConstraint.m_multiBodyB->setPosUpdated(false);
				}
			}
		}
	}
	else
	{
		for (int j1 = 0; j1 < this->m_multiBodyFrictionContactConstraints.size(); j1++)
		{
			if (iteration < infoGlobal.m_numIterations)
			{
				int index = j1;  //iteration&1? j1 : m_multiBodyFrictionContactConstraints.size()-1-j1;

				btMultiBodySolverConstraint& frictionConstraint = m_multiBodyFrictionContactConstraints[index];
				btScalar totalImpulse = m_multiBodyNormalContactConstraints[frictionConstraint.m_frictionIndex].m_appliedImpulse;
				//adjust friction limits here
				if (totalImpulse > btScalar(0))
				{
					frictionConstraint.m_lowerLimit = -(frictionConstraint.m_friction * totalImpulse);
					frictionConstraint.m_upperLimit = frictionConstraint.m_friction * totalImpulse;
					btScalar residual = resolveSingleConstraintRowGeneric(frictionConstraint);
					leastSquaredResidual = btMax(leastSquaredResidual, residual * residual);

					if (frictionConstraint.m_multiBodyA)
						frictionConstraint.m_multiBodyA->setPosUpdated(false);
					if (frictionConstraint.m_multiBodyB)
						frictionConstraint.m_multiBodyB->setPosUpdated(false);
				}
			}
		}
	}
	return leastSquaredResidual;
}

btScalar btMultiBodyConstraintSolver::solveGroupCacheFriendlySetup(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifoldPtr, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* debugDrawer)
{
	m_multiBodyNonContactConstraints.resize(0);
	m_multiBodyNormalContactConstraints.resize(0);
	m_multiBodyFrictionContactConstraints.resize(0);
	m_multiBodyTorsionalFrictionContactConstraints.resize(0);
	m_multiBodySpinningFrictionContactConstraints.resize(0);

	m_data.m_jacobians.resize(0);
	m_data.m_deltaVelocitiesUnitImpulse.resize(0);
	m_data.m_deltaVelocities.resize(0);

	for (int i = 0; i < numBodies; i++)
	{
		const btMultiBodyLinkCollider* fcA = btMultiBodyLinkCollider::upcast(bodies[i]);
		if (fcA)
		{
			fcA->m_multiBody->setCompanionId(-1);
		}
	}

	btScalar val = btSequentialImpulseConstraintSolver::solveGroupCacheFriendlySetup(bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints, infoGlobal, debugDrawer);

	return val;
}

void btMultiBodyConstraintSolver::applyDeltaVee(btScalar* delta_vee, btScalar impulse, int velocityIndex, int ndof)
{
	for (int i = 0; i < ndof; ++i)
		m_data.m_deltaVelocities[velocityIndex + i] += delta_vee[i] * impulse;
}

btScalar btMultiBodyConstraintSolver::resolveSingleConstraintRowGeneric(const btMultiBodySolverConstraint& c)
{
	btScalar deltaImpulse = c.m_rhs - btScalar(c.m_appliedImpulse) * c.m_cfm;
	btScalar deltaVelADotn = 0;
	btScalar deltaVelBDotn = 0;
	btSolverBody* bodyA = 0;
	btSolverBody* bodyB = 0;
	int ndofA = 0;
	int ndofB = 0;

	if (c.m_multiBodyA)
	{
		ndofA = c.m_multiBodyA->getNumDofs() + 6;
		for (int i = 0; i < ndofA; ++i)
			deltaVelADotn += m_data.m_jacobians[c.m_jacAindex + i] * m_data.m_deltaVelocities[c.m_deltaVelAindex + i];
	}
	else if (c.m_solverBodyIdA >= 0)
	{
		bodyA = &m_tmpSolverBodyPool[c.m_solverBodyIdA];
		deltaVelADotn += c.m_contactNormal1.dot(bodyA->internalGetDeltaLinearVelocity()) + c.m_relpos1CrossNormal.dot(bodyA->internalGetDeltaAngularVelocity());
	}

	if (c.m_multiBodyB)
	{
		ndofB = c.m_multiBodyB->getNumDofs() + 6;
		for (int i = 0; i < ndofB; ++i)
			deltaVelBDotn += m_data.m_jacobians[c.m_jacBindex + i] * m_data.m_deltaVelocities[c.m_deltaVelBindex + i];
	}
	else if (c.m_solverBodyIdB >= 0)
	{
		bodyB = &m_tmpSolverBodyPool[c.m_solverBodyIdB];
		deltaVelBDotn += c.m_contactNormal2.dot(bodyB->internalGetDeltaLinearVelocity()) + c.m_relpos2CrossNormal.dot(bodyB->internalGetDeltaAngularVelocity());
	}

	deltaImpulse -= deltaVelADotn * c.m_jacDiagABInv;  //m_jacDiagABInv = 1./denom
	deltaImpulse -= deltaVelBDotn * c.m_jacDiagABInv;
	const btScalar sum = btScalar(c.m_appliedImpulse) + deltaImpulse;

	if (sum < c.m_lowerLimit)
	{
		deltaImpulse = c.m_lowerLimit - c.m_appliedImpulse;
		c.m_appliedImpulse = c.m_lowerLimit;
	}
	else if (sum > c.m_upperLimit)
	{
		deltaImpulse = c.m_upperLimit - c.m_appliedImpulse;
		c.m_appliedImpulse = c.m_upperLimit;
	}
	else
	{
		c.m_appliedImpulse = sum;
	}

	if (c.m_multiBodyA)
	{
		applyDeltaVee(&m_data.m_deltaVelocitiesUnitImpulse[c.m_jacAindex], deltaImpulse, c.m_deltaVelAindex, ndofA);
#ifdef DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
		//note: update of the actual velocities (below) in the multibody does not have to happen now since m_deltaVelocities can be applied after all iterations
		//it would make the multibody solver more like the regular one with m_deltaVelocities being equivalent to btSolverBody::m_deltaLinearVelocity/m_deltaAngularVelocity
		c.m_multiBodyA->applyDeltaVeeMultiDof2(&m_data.m_deltaVelocitiesUnitImpulse[c.m_jacAindex], deltaImpulse);
#endif  //DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
	}
	else if (c.m_solverBodyIdA >= 0)
	{
		bodyA->internalApplyImpulse(c.m_contactNormal1 * bodyA->internalGetInvMass(), c.m_angularComponentA, deltaImpulse);
	}
	if (c.m_multiBodyB)
	{
		applyDeltaVee(&m_data.m_deltaVelocitiesUnitImpulse[c.m_jacBindex], deltaImpulse, c.m_deltaVelBindex, ndofB);
#ifdef DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
		//note: update of the actual velocities (below) in the multibody does not have to happen now since m_deltaVelocities can be applied after all iterations
		//it would make the multibody solver more like the regular one with m_deltaVelocities being equivalent to btSolverBody::m_deltaLinearVelocity/m_deltaAngularVelocity
		c.m_multiBodyB->applyDeltaVeeMultiDof2(&m_data.m_deltaVelocitiesUnitImpulse[c.m_jacBindex], deltaImpulse);
#endif  //DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
	}
	else if (c.m_solverBodyIdB >= 0)
	{
		bodyB->internalApplyImpulse(c.m_contactNormal2 * bodyB->internalGetInvMass(), c.m_angularComponentB, deltaImpulse);
	}
	btScalar deltaVel = deltaImpulse / c.m_jacDiagABInv;
	return deltaVel;
}

btScalar btMultiBodyConstraintSolver::resolveConeFrictionConstraintRows(const btMultiBodySolverConstraint& cA1, const btMultiBodySolverConstraint& cB)
{
	int ndofA = 0;
	int ndofB = 0;
	btSolverBody* bodyA = 0;
	btSolverBody* bodyB = 0;
	btScalar deltaImpulseB = 0.f;
	btScalar sumB = 0.f;
	{
		deltaImpulseB = cB.m_rhs - btScalar(cB.m_appliedImpulse) * cB.m_cfm;
		btScalar deltaVelADotn = 0;
		btScalar deltaVelBDotn = 0;
		if (cB.m_multiBodyA)
		{
			ndofA = cB.m_multiBodyA->getNumDofs() + 6;
			for (int i = 0; i < ndofA; ++i)
				deltaVelADotn += m_data.m_jacobians[cB.m_jacAindex + i] * m_data.m_deltaVelocities[cB.m_deltaVelAindex + i];
		}
		else if (cB.m_solverBodyIdA >= 0)
		{
			bodyA = &m_tmpSolverBodyPool[cB.m_solverBodyIdA];
			deltaVelADotn += cB.m_contactNormal1.dot(bodyA->internalGetDeltaLinearVelocity()) + cB.m_relpos1CrossNormal.dot(bodyA->internalGetDeltaAngularVelocity());
		}

		if (cB.m_multiBodyB)
		{
			ndofB = cB.m_multiBodyB->getNumDofs() + 6;
			for (int i = 0; i < ndofB; ++i)
				deltaVelBDotn += m_data.m_jacobians[cB.m_jacBindex + i] * m_data.m_deltaVelocities[cB.m_deltaVelBindex + i];
		}
		else if (cB.m_solverBodyIdB >= 0)
		{
			bodyB = &m_tmpSolverBodyPool[cB.m_solverBodyIdB];
			deltaVelBDotn += cB.m_contactNormal2.dot(bodyB->internalGetDeltaLinearVelocity()) + cB.m_relpos2CrossNormal.dot(bodyB->internalGetDeltaAngularVelocity());
		}

		deltaImpulseB -= deltaVelADotn * cB.m_jacDiagABInv;  //m_jacDiagABInv = 1./denom
		deltaImpulseB -= deltaVelBDotn * cB.m_jacDiagABInv;
		sumB = btScalar(cB.m_appliedImpulse) + deltaImpulseB;
	}

	btScalar deltaImpulseA = 0.f;
	btScalar sumA = 0.f;
	const btMultiBodySolverConstraint& cA = cA1;
	{
		{
			deltaImpulseA = cA.m_rhs - btScalar(cA.m_appliedImpulse) * cA.m_cfm;
			btScalar deltaVelADotn = 0;
			btScalar deltaVelBDotn = 0;
			if (cA.m_multiBodyA)
			{
				ndofA = cA.m_multiBodyA->getNumDofs() + 6;
				for (int i = 0; i < ndofA; ++i)
					deltaVelADotn += m_data.m_jacobians[cA.m_jacAindex + i] * m_data.m_deltaVelocities[cA.m_deltaVelAindex + i];
			}
			else if (cA.m_solverBodyIdA >= 0)
			{
				bodyA = &m_tmpSolverBodyPool[cA.m_solverBodyIdA];
				deltaVelADotn += cA.m_contactNormal1.dot(bodyA->internalGetDeltaLinearVelocity()) + cA.m_relpos1CrossNormal.dot(bodyA->internalGetDeltaAngularVelocity());
			}

			if (cA.m_multiBodyB)
			{
				ndofB = cA.m_multiBodyB->getNumDofs() + 6;
				for (int i = 0; i < ndofB; ++i)
					deltaVelBDotn += m_data.m_jacobians[cA.m_jacBindex + i] * m_data.m_deltaVelocities[cA.m_deltaVelBindex + i];
			}
			else if (cA.m_solverBodyIdB >= 0)
			{
				bodyB = &m_tmpSolverBodyPool[cA.m_solverBodyIdB];
				deltaVelBDotn += cA.m_contactNormal2.dot(bodyB->internalGetDeltaLinearVelocity()) + cA.m_relpos2CrossNormal.dot(bodyB->internalGetDeltaAngularVelocity());
			}

			deltaImpulseA -= deltaVelADotn * cA.m_jacDiagABInv;  //m_jacDiagABInv = 1./denom
			deltaImpulseA -= deltaVelBDotn * cA.m_jacDiagABInv;
			sumA = btScalar(cA.m_appliedImpulse) + deltaImpulseA;
		}
	}

	if (sumA * sumA + sumB * sumB >= cA.m_lowerLimit * cB.m_lowerLimit)
	{
		btScalar angle = btAtan2(sumA, sumB);
		btScalar sumAclipped = btFabs(cA.m_lowerLimit * btSin(angle));
		btScalar sumBclipped = btFabs(cB.m_lowerLimit * btCos(angle));

		if (sumA < -sumAclipped)
		{
			deltaImpulseA = -sumAclipped - cA.m_appliedImpulse;
			cA.m_appliedImpulse = -sumAclipped;
		}
		else if (sumA > sumAclipped)
		{
			deltaImpulseA = sumAclipped - cA.m_appliedImpulse;
			cA.m_appliedImpulse = sumAclipped;
		}
		else
		{
			cA.m_appliedImpulse = sumA;
		}

		if (sumB < -sumBclipped)
		{
			deltaImpulseB = -sumBclipped - cB.m_appliedImpulse;
			cB.m_appliedImpulse = -sumBclipped;
		}
		else if (sumB > sumBclipped)
		{
			deltaImpulseB = sumBclipped - cB.m_appliedImpulse;
			cB.m_appliedImpulse = sumBclipped;
		}
		else
		{
			cB.m_appliedImpulse = sumB;
		}
		//deltaImpulseA = sumAclipped-cA.m_appliedImpulse;
		//cA.m_appliedImpulse = sumAclipped;
		//deltaImpulseB = sumBclipped-cB.m_appliedImpulse;
		//cB.m_appliedImpulse = sumBclipped;
	}
	else
	{
		cA.m_appliedImpulse = sumA;
		cB.m_appliedImpulse = sumB;
	}

	if (cA.m_multiBodyA)
	{
		applyDeltaVee(&m_data.m_deltaVelocitiesUnitImpulse[cA.m_jacAindex], deltaImpulseA, cA.m_deltaVelAindex, ndofA);
#ifdef DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
		//note: update of the actual velocities (below) in the multibody does not have to happen now since m_deltaVelocities can be applied after all iterations
		//it would make the multibody solver more like the regular one with m_deltaVelocities being equivalent to btSolverBody::m_deltaLinearVelocity/m_deltaAngularVelocity
		cA.m_multiBodyA->applyDeltaVeeMultiDof2(&m_data.m_deltaVelocitiesUnitImpulse[cA.m_jacAindex], deltaImpulseA);
#endif  //DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
	}
	else if (cA.m_solverBodyIdA >= 0)
	{
		bodyA->internalApplyImpulse(cA.m_contactNormal1 * bodyA->internalGetInvMass(), cA.m_angularComponentA, deltaImpulseA);
	}
	if (cA.m_multiBodyB)
	{
		applyDeltaVee(&m_data.m_deltaVelocitiesUnitImpulse[cA.m_jacBindex], deltaImpulseA, cA.m_deltaVelBindex, ndofB);
#ifdef DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
		//note: update of the actual velocities (below) in the multibody does not have to happen now since m_deltaVelocities can be applied after all iterations
		//it would make the multibody solver more like the regular one with m_deltaVelocities being equivalent to btSolverBody::m_deltaLinearVelocity/m_deltaAngularVelocity
		cA.m_multiBodyB->applyDeltaVeeMultiDof2(&m_data.m_deltaVelocitiesUnitImpulse[cA.m_jacBindex], deltaImpulseA);
#endif  //DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
	}
	else if (cA.m_solverBodyIdB >= 0)
	{
		bodyB->internalApplyImpulse(cA.m_contactNormal2 * bodyB->internalGetInvMass(), cA.m_angularComponentB, deltaImpulseA);
	}

	if (cB.m_multiBodyA)
	{
		applyDeltaVee(&m_data.m_deltaVelocitiesUnitImpulse[cB.m_jacAindex], deltaImpulseB, cB.m_deltaVelAindex, ndofA);
#ifdef DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
		//note: update of the actual velocities (below) in the multibody does not have to happen now since m_deltaVelocities can be applied after all iterations
		//it would make the multibody solver more like the regular one with m_deltaVelocities being equivalent to btSolverBody::m_deltaLinearVelocity/m_deltaAngularVelocity
		cB.m_multiBodyA->applyDeltaVeeMultiDof2(&m_data.m_deltaVelocitiesUnitImpulse[cB.m_jacAindex], deltaImpulseB);
#endif  //DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
	}
	else if (cB.m_solverBodyIdA >= 0)
	{
		bodyA->internalApplyImpulse(cB.m_contactNormal1 * bodyA->internalGetInvMass(), cB.m_angularComponentA, deltaImpulseB);
	}
	if (cB.m_multiBodyB)
	{
		applyDeltaVee(&m_data.m_deltaVelocitiesUnitImpulse[cB.m_jacBindex], deltaImpulseB, cB.m_deltaVelBindex, ndofB);
#ifdef DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
		//note: update of the actual velocities (below) in the multibody does not have to happen now since m_deltaVelocities can be applied after all iterations
		//it would make the multibody solver more like the regular one with m_deltaVelocities being equivalent to btSolverBody::m_deltaLinearVelocity/m_deltaAngularVelocity
		cB.m_multiBodyB->applyDeltaVeeMultiDof2(&m_data.m_deltaVelocitiesUnitImpulse[cB.m_jacBindex], deltaImpulseB);
#endif  //DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
	}
	else if (cB.m_solverBodyIdB >= 0)
	{
		bodyB->internalApplyImpulse(cB.m_contactNormal2 * bodyB->internalGetInvMass(), cB.m_angularComponentB, deltaImpulseB);
	}

	btScalar deltaVel = deltaImpulseA / cA.m_jacDiagABInv + deltaImpulseB / cB.m_jacDiagABInv;
	return deltaVel;
}

void btMultiBodyConstraintSolver::setupMultiBodyContactConstraint(btMultiBodySolverConstraint& solverConstraint, const btVector3& contactNormal, const btScalar& appliedImpulse, btManifoldPoint& cp, const btContactSolverInfo& infoGlobal, btScalar& relaxation, bool isFriction, btScalar desiredVelocity, btScalar cfmSlip)
{
	BT_PROFILE("setupMultiBodyContactConstraint");
	btVector3 rel_pos1;
	btVector3 rel_pos2;

	btMultiBody* multiBodyA = solverConstraint.m_multiBodyA;
	btMultiBody* multiBodyB = solverConstraint.m_multiBodyB;

	const btVector3& pos1 = cp.getPositionWorldOnA();
	const btVector3& pos2 = cp.getPositionWorldOnB();

	btSolverBody* bodyA = multiBodyA ? 0 : &m_tmpSolverBodyPool[solverConstraint.m_solverBodyIdA];
	btSolverBody* bodyB = multiBodyB ? 0 : &m_tmpSolverBodyPool[solverConstraint.m_solverBodyIdB];

	btRigidBody* rb0 = multiBodyA ? 0 : bodyA->m_originalBody;
	btRigidBody* rb1 = multiBodyB ? 0 : bodyB->m_originalBody;

	if (bodyA)
		rel_pos1 = pos1 - bodyA->getWorldTransform().getOrigin();
	if (bodyB)
		rel_pos2 = pos2 - bodyB->getWorldTransform().getOrigin();

	relaxation = infoGlobal.m_sor;

	btScalar invTimeStep = btScalar(1) / infoGlobal.m_timeStep;

	//cfm = 1 /       ( dt * kp + kd )
	//erp = dt * kp / ( dt * kp + kd )

	btScalar cfm;
	btScalar erp;
	if (isFriction)
	{
		cfm = infoGlobal.m_frictionCFM;
		erp = infoGlobal.m_frictionERP;
	}
	else
	{
		cfm = infoGlobal.m_globalCfm;
		erp = infoGlobal.m_erp2;

		if ((cp.m_contactPointFlags & BT_CONTACT_FLAG_HAS_CONTACT_CFM) || (cp.m_contactPointFlags & BT_CONTACT_FLAG_HAS_CONTACT_ERP))
		{
			if (cp.m_contactPointFlags & BT_CONTACT_FLAG_HAS_CONTACT_CFM)
				cfm = cp.m_contactCFM;
			if (cp.m_contactPointFlags & BT_CONTACT_FLAG_HAS_CONTACT_ERP)
				erp = cp.m_contactERP;
		}
		else
		{
			if (cp.m_contactPointFlags & BT_CONTACT_FLAG_CONTACT_STIFFNESS_DAMPING)
			{
				btScalar denom = (infoGlobal.m_timeStep * cp.m_combinedContactStiffness1 + cp.m_combinedContactDamping1);
				if (denom < SIMD_EPSILON)
				{
					denom = SIMD_EPSILON;
				}
				cfm = btScalar(1) / denom;
				erp = (infoGlobal.m_timeStep * cp.m_combinedContactStiffness1) / denom;
			}
		}
	}

	cfm *= invTimeStep;

	if (multiBodyA)
	{
		if (solverConstraint.m_linkA < 0)
		{
			rel_pos1 = pos1 - multiBodyA->getBasePos();
		}
		else
		{
			rel_pos1 = pos1 - multiBodyA->getLink(solverConstraint.m_linkA).m_cachedWorldTransform.getOrigin();
		}
		const int ndofA = multiBodyA->getNumDofs() + 6;

		solverConstraint.m_deltaVelAindex = multiBodyA->getCompanionId();

		if (solverConstraint.m_deltaVelAindex < 0)
		{
			solverConstraint.m_deltaVelAindex = m_data.m_deltaVelocities.size();
			multiBodyA->setCompanionId(solverConstraint.m_deltaVelAindex);
			m_data.m_deltaVelocities.resize(m_data.m_deltaVelocities.size() + ndofA);
		}
		else
		{
			btAssert(m_data.m_deltaVelocities.size() >= solverConstraint.m_deltaVelAindex + ndofA);
		}

		solverConstraint.m_jacAindex = m_data.m_jacobians.size();
		m_data.m_jacobians.resize(m_data.m_jacobians.size() + ndofA);
		m_data.m_deltaVelocitiesUnitImpulse.resize(m_data.m_deltaVelocitiesUnitImpulse.size() + ndofA);
		btAssert(m_data.m_jacobians.size() == m_data.m_deltaVelocitiesUnitImpulse.size());

		btScalar* jac1 = &m_data.m_jacobians[solverConstraint.m_jacAindex];
		multiBodyA->fillContactJacobianMultiDof(solverConstraint.m_linkA, cp.getPositionWorldOnA(), contactNormal, jac1, m_data.scratch_r, m_data.scratch_v, m_data.scratch_m);
		btScalar* delta = &m_data.m_deltaVelocitiesUnitImpulse[solverConstraint.m_jacAindex];
		multiBodyA->calcAccelerationDeltasMultiDof(&m_data.m_jacobians[solverConstraint.m_jacAindex], delta, m_data.scratch_r, m_data.scratch_v);

		btVector3 torqueAxis0 = rel_pos1.cross(contactNormal);
		solverConstraint.m_relpos1CrossNormal = torqueAxis0;
		solverConstraint.m_contactNormal1 = contactNormal;
	}
	else
	{
		btVector3 torqueAxis0 = rel_pos1.cross(contactNormal);
		solverConstraint.m_relpos1CrossNormal = torqueAxis0;
		solverConstraint.m_contactNormal1 = contactNormal;
		solverConstraint.m_angularComponentA = rb0 ? rb0->getInvInertiaTensorWorld() * torqueAxis0 * rb0->getAngularFactor() : btVector3(0, 0, 0);
	}

	if (multiBodyB)
	{
		if (solverConstraint.m_linkB < 0)
		{
			rel_pos2 = pos2 - multiBodyB->getBasePos();
		}
		else
		{
			rel_pos2 = pos2 - multiBodyB->getLink(solverConstraint.m_linkB).m_cachedWorldTransform.getOrigin();
		}

		const int ndofB = multiBodyB->getNumDofs() + 6;

		solverConstraint.m_deltaVelBindex = multiBodyB->getCompanionId();
		if (solverConstraint.m_deltaVelBindex < 0)
		{
			solverConstraint.m_deltaVelBindex = m_data.m_deltaVelocities.size();
			multiBodyB->setCompanionId(solverConstraint.m_deltaVelBindex);
			m_data.m_deltaVelocities.resize(m_data.m_deltaVelocities.size() + ndofB);
		}

		solverConstraint.m_jacBindex = m_data.m_jacobians.size();

		m_data.m_jacobians.resize(m_data.m_jacobians.size() + ndofB);
		m_data.m_deltaVelocitiesUnitImpulse.resize(m_data.m_deltaVelocitiesUnitImpulse.size() + ndofB);
		btAssert(m_data.m_jacobians.size() == m_data.m_deltaVelocitiesUnitImpulse.size());

		multiBodyB->fillContactJacobianMultiDof(solverConstraint.m_linkB, cp.getPositionWorldOnB(), -contactNormal, &m_data.m_jacobians[solverConstraint.m_jacBindex], m_data.scratch_r, m_data.scratch_v, m_data.scratch_m);
		multiBodyB->calcAccelerationDeltasMultiDof(&m_data.m_jacobians[solverConstraint.m_jacBindex], &m_data.m_deltaVelocitiesUnitImpulse[solverConstraint.m_jacBindex], m_data.scratch_r, m_data.scratch_v);

		btVector3 torqueAxis1 = rel_pos2.cross(contactNormal);
		solverConstraint.m_relpos2CrossNormal = -torqueAxis1;
		solverConstraint.m_contactNormal2 = -contactNormal;
	}
	else
	{
		btVector3 torqueAxis1 = rel_pos2.cross(contactNormal);
		solverConstraint.m_relpos2CrossNormal = -torqueAxis1;
		solverConstraint.m_contactNormal2 = -contactNormal;

		solverConstraint.m_angularComponentB = rb1 ? rb1->getInvInertiaTensorWorld() * -torqueAxis1 * rb1->getAngularFactor() : btVector3(0, 0, 0);
	}

	{
		btVector3 vec;
		btScalar denom0 = 0.f;
		btScalar denom1 = 0.f;
		btScalar* jacB = 0;
		btScalar* jacA = 0;
		btScalar* lambdaA = 0;
		btScalar* lambdaB = 0;
		int ndofA = 0;
		if (multiBodyA)
		{
			ndofA = multiBodyA->getNumDofs() + 6;
			jacA = &m_data.m_jacobians[solverConstraint.m_jacAindex];
			lambdaA = &m_data.m_deltaVelocitiesUnitImpulse[solverConstraint.m_jacAindex];
			for (int i = 0; i < ndofA; ++i)
			{
				btScalar j = jacA[i];
				btScalar l = lambdaA[i];
				denom0 += j * l;
			}
		}
		else
		{
			if (rb0)
			{
				vec = (solverConstraint.m_angularComponentA).cross(rel_pos1);
				denom0 = rb0->getInvMass() + contactNormal.dot(vec);
			}
		}
		if (multiBodyB)
		{
			const int ndofB = multiBodyB->getNumDofs() + 6;
			jacB = &m_data.m_jacobians[solverConstraint.m_jacBindex];
			lambdaB = &m_data.m_deltaVelocitiesUnitImpulse[solverConstraint.m_jacBindex];
			for (int i = 0; i < ndofB; ++i)
			{
				btScalar j = jacB[i];
				btScalar l = lambdaB[i];
				denom1 += j * l;
			}
		}
		else
		{
			if (rb1)
			{
				vec = (-solverConstraint.m_angularComponentB).cross(rel_pos2);
				denom1 = rb1->getInvMass() + contactNormal.dot(vec);
			}
		}

		btScalar d = denom0 + denom1 + cfm;
		if (d > SIMD_EPSILON)
		{
			solverConstraint.m_jacDiagABInv = relaxation / (d);
		}
		else
		{
			//disable the constraint row to handle singularity/redundant constraint
			solverConstraint.m_jacDiagABInv = 0.f;
		}
	}

	//compute rhs and remaining solverConstraint fields

	btScalar restitution = 0.f;
	btScalar distance = 0;
	if (!isFriction)
	{
		distance = cp.getDistance() + infoGlobal.m_linearSlop;
	}
	else
	{
		if (cp.m_contactPointFlags & BT_CONTACT_FLAG_FRICTION_ANCHOR)
		{
			distance = (cp.getPositionWorldOnA() - cp.getPositionWorldOnB()).dot(contactNormal);
		}
	}

	btScalar rel_vel = 0.f;
	int ndofA = 0;
	int ndofB = 0;
	{
		btVector3 vel1, vel2;
		if (multiBodyA)
		{
			ndofA = multiBodyA->getNumDofs() + 6;
			btScalar* jacA = &m_data.m_jacobians[solverConstraint.m_jacAindex];
			for (int i = 0; i < ndofA; ++i)
				rel_vel += multiBodyA->getVelocityVector()[i] * jacA[i];
		}
		else
		{
			if (rb0)
			{
				rel_vel += (rb0->getVelocityInLocalPoint(rel_pos1) +
							(rb0->getTotalTorque() * rb0->getInvInertiaTensorWorld() * infoGlobal.m_timeStep).cross(rel_pos1) +
							rb0->getTotalForce() * rb0->getInvMass() * infoGlobal.m_timeStep)
							   .dot(solverConstraint.m_contactNormal1);
			}
		}
		if (multiBodyB)
		{
			ndofB = multiBodyB->getNumDofs() + 6;
			btScalar* jacB = &m_data.m_jacobians[solverConstraint.m_jacBindex];
			for (int i = 0; i < ndofB; ++i)
				rel_vel += multiBodyB->getVelocityVector()[i] * jacB[i];
		}
		else
		{
			if (rb1)
			{
				rel_vel += (rb1->getVelocityInLocalPoint(rel_pos2) +
							(rb1->getTotalTorque() * rb1->getInvInertiaTensorWorld() * infoGlobal.m_timeStep).cross(rel_pos2) +
							rb1->getTotalForce() * rb1->getInvMass() * infoGlobal.m_timeStep)
							   .dot(solverConstraint.m_contactNormal2);
			}
		}

		solverConstraint.m_friction = cp.m_combinedFriction;

		if (!isFriction)
		{
			restitution = restitutionCurve(rel_vel, cp.m_combinedRestitution, infoGlobal.m_restitutionVelocityThreshold);
			if (restitution <= btScalar(0.))
			{
				restitution = 0.f;
			}
		}
	}

	{
		btScalar positionalError = 0.f;
		btScalar velocityError = restitution - rel_vel;  // * damping;	//note for friction restitution is always set to 0 (check above) so it is acutally velocityError = -rel_vel for friction
		if (isFriction)
		{
			positionalError = -distance * erp / infoGlobal.m_timeStep;
		}
		else
		{
			if (distance > 0)
			{
				positionalError = 0;
				velocityError -= distance / infoGlobal.m_timeStep;
			}
			else
			{
				positionalError = -distance * erp / infoGlobal.m_timeStep;
			}
		}

		btScalar penetrationImpulse = positionalError * solverConstraint.m_jacDiagABInv;
		btScalar velocityImpulse = velocityError * solverConstraint.m_jacDiagABInv;

		if (!isFriction)
		{
			//	if (!infoGlobal.m_splitImpulse || (penetration > infoGlobal.m_splitImpulsePenetrationThreshold))
			{
				//combine position and velocity into rhs
				solverConstraint.m_rhs = penetrationImpulse + velocityImpulse;
				solverConstraint.m_rhsPenetration = 0.f;
			}
			/*else
			{
				//split position and velocity into rhs and m_rhsPenetration
				solverConstraint.m_rhs = velocityImpulse;
				solverConstraint.m_rhsPenetration = penetrationImpulse;
			}
			*/
			solverConstraint.m_lowerLimit = 0;
			solverConstraint.m_upperLimit = 1e10f;
		}
		else
		{
			solverConstraint.m_rhs = penetrationImpulse + velocityImpulse;
			solverConstraint.m_rhsPenetration = 0.f;
			solverConstraint.m_lowerLimit = -solverConstraint.m_friction;
			solverConstraint.m_upperLimit = solverConstraint.m_friction;
		}

		solverConstraint.m_cfm = cfm * solverConstraint.m_jacDiagABInv;
	}
        
	if (infoGlobal.m_solverMode & SOLVER_USE_ARTICULATED_WARMSTARTING)
	{
		if (btFabs(cp.m_prevRHS) > 1e-5 && cp.m_prevRHS < 2* solverConstraint.m_rhs && solverConstraint.m_rhs < 2*cp.m_prevRHS)
		{
			solverConstraint.m_appliedImpulse = isFriction ? 0 : cp.m_appliedImpulse / cp.m_prevRHS * solverConstraint.m_rhs * infoGlobal.m_articulatedWarmstartingFactor;
			if (solverConstraint.m_appliedImpulse < 0)
				solverConstraint.m_appliedImpulse = 0;
		}
		else
		{
			solverConstraint.m_appliedImpulse = 0.f;
		}

		if (solverConstraint.m_appliedImpulse)
		{
			if (multiBodyA)
			{
				btScalar impulse = solverConstraint.m_appliedImpulse;
				btScalar* deltaV = &m_data.m_deltaVelocitiesUnitImpulse[solverConstraint.m_jacAindex];
				multiBodyA->applyDeltaVeeMultiDof2(deltaV, impulse);

				applyDeltaVee(deltaV, impulse, solverConstraint.m_deltaVelAindex, ndofA);
			}
			else
			{
				if (rb0)
					bodyA->internalApplyImpulse(solverConstraint.m_contactNormal1 * bodyA->internalGetInvMass() * rb0->getLinearFactor(), solverConstraint.m_angularComponentA, solverConstraint.m_appliedImpulse);
			}
			if (multiBodyB)
			{
				btScalar impulse = solverConstraint.m_appliedImpulse;
				btScalar* deltaV = &m_data.m_deltaVelocitiesUnitImpulse[solverConstraint.m_jacBindex];
				multiBodyB->applyDeltaVeeMultiDof2(deltaV, impulse);
				applyDeltaVee(deltaV, impulse, solverConstraint.m_deltaVelBindex, ndofB);
			}
			else
			{
				if (rb1)
					bodyB->internalApplyImpulse(-solverConstraint.m_contactNormal2 * bodyB->internalGetInvMass() * rb1->getLinearFactor(), -solverConstraint.m_angularComponentB, -(btScalar)solverConstraint.m_appliedImpulse);
			}
		}
	}
	else
	{
		solverConstraint.m_appliedImpulse = 0.f;
   	solverConstraint.m_appliedPushImpulse = 0.f;
	}
}

void btMultiBodyConstraintSolver::setupMultiBodyTorsionalFrictionConstraint(btMultiBodySolverConstraint& solverConstraint,
																			const btVector3& constraintNormal,
																			btManifoldPoint& cp,
																			btScalar combinedTorsionalFriction,
																			const btContactSolverInfo& infoGlobal,
																			btScalar& relaxation,
																			bool isFriction, btScalar desiredVelocity, btScalar cfmSlip)
{
	BT_PROFILE("setupMultiBodyRollingFrictionConstraint");
	btVector3 rel_pos1;
	btVector3 rel_pos2;

	btMultiBody* multiBodyA = solverConstraint.m_multiBodyA;
	btMultiBody* multiBodyB = solverConstraint.m_multiBodyB;

	const btVector3& pos1 = cp.getPositionWorldOnA();
	const btVector3& pos2 = cp.getPositionWorldOnB();

	btSolverBody* bodyA = multiBodyA ? 0 : &m_tmpSolverBodyPool[solverConstraint.m_solverBodyIdA];
	btSolverBody* bodyB = multiBodyB ? 0 : &m_tmpSolverBodyPool[solverConstraint.m_solverBodyIdB];

	btRigidBody* rb0 = multiBodyA ? 0 : bodyA->m_originalBody;
	btRigidBody* rb1 = multiBodyB ? 0 : bodyB->m_originalBody;

	if (bodyA)
		rel_pos1 = pos1 - bodyA->getWorldTransform().getOrigin();
	if (bodyB)
		rel_pos2 = pos2 - bodyB->getWorldTransform().getOrigin();

	relaxation = infoGlobal.m_sor;

	// btScalar invTimeStep = btScalar(1)/infoGlobal.m_timeStep;

	if (multiBodyA)
	{
		if (solverConstraint.m_linkA < 0)
		{
			rel_pos1 = pos1 - multiBodyA->getBasePos();
		}
		else
		{
			rel_pos1 = pos1 - multiBodyA->getLink(solverConstraint.m_linkA).m_cachedWorldTransform.getOrigin();
		}
		const int ndofA = multiBodyA->getNumDofs() + 6;

		solverConstraint.m_deltaVelAindex = multiBodyA->getCompanionId();

		if (solverConstraint.m_deltaVelAindex < 0)
		{
			solverConstraint.m_deltaVelAindex = m_data.m_deltaVelocities.size();
			multiBodyA->setCompanionId(solverConstraint.m_deltaVelAindex);
			m_data.m_deltaVelocities.resize(m_data.m_deltaVelocities.size() + ndofA);
		}
		else
		{
			btAssert(m_data.m_deltaVelocities.size() >= solverConstraint.m_deltaVelAindex + ndofA);
		}

		solverConstraint.m_jacAindex = m_data.m_jacobians.size();
		m_data.m_jacobians.resize(m_data.m_jacobians.size() + ndofA);
		m_data.m_deltaVelocitiesUnitImpulse.resize(m_data.m_deltaVelocitiesUnitImpulse.size() + ndofA);
		btAssert(m_data.m_jacobians.size() == m_data.m_deltaVelocitiesUnitImpulse.size());

		btScalar* jac1 = &m_data.m_jacobians[solverConstraint.m_jacAindex];
		multiBodyA->fillConstraintJacobianMultiDof(solverConstraint.m_linkA, cp.getPositionWorldOnA(), constraintNormal, btVector3(0, 0, 0), jac1, m_data.scratch_r, m_data.scratch_v, m_data.scratch_m);
		btScalar* delta = &m_data.m_deltaVelocitiesUnitImpulse[solverConstraint.m_jacAindex];
		multiBodyA->calcAccelerationDeltasMultiDof(&m_data.m_jacobians[solverConstraint.m_jacAindex], delta, m_data.scratch_r, m_data.scratch_v);

		btVector3 torqueAxis0 = constraintNormal;
		solverConstraint.m_relpos1CrossNormal = torqueAxis0;
		solverConstraint.m_contactNormal1 = btVector3(0, 0, 0);
	}
	else
	{
		btVector3 torqueAxis0 = constraintNormal;
		solverConstraint.m_relpos1CrossNormal = torqueAxis0;
		solverConstraint.m_contactNormal1 = btVector3(0, 0, 0);
		solverConstraint.m_angularComponentA = rb0 ? rb0->getInvInertiaTensorWorld() * torqueAxis0 * rb0->getAngularFactor() : btVector3(0, 0, 0);
	}

	if (multiBodyB)
	{
		if (solverConstraint.m_linkB < 0)
		{
			rel_pos2 = pos2 - multiBodyB->getBasePos();
		}
		else
		{
			rel_pos2 = pos2 - multiBodyB->getLink(solverConstraint.m_linkB).m_cachedWorldTransform.getOrigin();
		}

		const int ndofB = multiBodyB->getNumDofs() + 6;

		solverConstraint.m_deltaVelBindex = multiBodyB->getCompanionId();
		if (solverConstraint.m_deltaVelBindex < 0)
		{
			solverConstraint.m_deltaVelBindex = m_data.m_deltaVelocities.size();
			multiBodyB->setCompanionId(solverConstraint.m_deltaVelBindex);
			m_data.m_deltaVelocities.resize(m_data.m_deltaVelocities.size() + ndofB);
		}

		solverConstraint.m_jacBindex = m_data.m_jacobians.size();

		m_data.m_jacobians.resize(m_data.m_jacobians.size() + ndofB);
		m_data.m_deltaVelocitiesUnitImpulse.resize(m_data.m_deltaVelocitiesUnitImpulse.size() + ndofB);
		btAssert(m_data.m_jacobians.size() == m_data.m_deltaVelocitiesUnitImpulse.size());

		multiBodyB->fillConstraintJacobianMultiDof(solverConstraint.m_linkB, cp.getPositionWorldOnB(), -constraintNormal, btVector3(0, 0, 0), &m_data.m_jacobians[solverConstraint.m_jacBindex], m_data.scratch_r, m_data.scratch_v, m_data.scratch_m);
		multiBodyB->calcAccelerationDeltasMultiDof(&m_data.m_jacobians[solverConstraint.m_jacBindex], &m_data.m_deltaVelocitiesUnitImpulse[solverConstraint.m_jacBindex], m_data.scratch_r, m_data.scratch_v);

		btVector3 torqueAxis1 = -constraintNormal;
		solverConstraint.m_relpos2CrossNormal = torqueAxis1;
		solverConstraint.m_contactNormal2 = -btVector3(0, 0, 0);
	}
	else
	{
		btVector3 torqueAxis1 = -constraintNormal;
		solverConstraint.m_relpos2CrossNormal = torqueAxis1;
		solverConstraint.m_contactNormal2 = -btVector3(0, 0, 0);

		solverConstraint.m_angularComponentB = rb1 ? rb1->getInvInertiaTensorWorld() * torqueAxis1 * rb1->getAngularFactor() : btVector3(0, 0, 0);
	}

	{
		btScalar denom0 = 0.f;
		btScalar denom1 = 0.f;
		btScalar* jacB = 0;
		btScalar* jacA = 0;
		btScalar* lambdaA = 0;
		btScalar* lambdaB = 0;
		int ndofA = 0;
		if (multiBodyA)
		{
			ndofA = multiBodyA->getNumDofs() + 6;
			jacA = &m_data.m_jacobians[solverConstraint.m_jacAindex];
			lambdaA = &m_data.m_deltaVelocitiesUnitImpulse[solverConstraint.m_jacAindex];
			for (int i = 0; i < ndofA; ++i)
			{
				btScalar j = jacA[i];
				btScalar l = lambdaA[i];
				denom0 += j * l;
			}
		}
		else
		{
			if (rb0)
			{
				btVector3 iMJaA = rb0 ? rb0->getInvInertiaTensorWorld() * solverConstraint.m_relpos1CrossNormal : btVector3(0, 0, 0);
				denom0 = iMJaA.dot(solverConstraint.m_relpos1CrossNormal);
			}
		}
		if (multiBodyB)
		{
			const int ndofB = multiBodyB->getNumDofs() + 6;
			jacB = &m_data.m_jacobians[solverConstraint.m_jacBindex];
			lambdaB = &m_data.m_deltaVelocitiesUnitImpulse[solverConstraint.m_jacBindex];
			for (int i = 0; i < ndofB; ++i)
			{
				btScalar j = jacB[i];
				btScalar l = lambdaB[i];
				denom1 += j * l;
			}
		}
		else
		{
			if (rb1)
			{
				btVector3 iMJaB = rb1 ? rb1->getInvInertiaTensorWorld() * solverConstraint.m_relpos2CrossNormal : btVector3(0, 0, 0);
				denom1 = iMJaB.dot(solverConstraint.m_relpos2CrossNormal);
			}
		}

		btScalar d = denom0 + denom1 + infoGlobal.m_globalCfm;
		if (d > SIMD_EPSILON)
		{
			solverConstraint.m_jacDiagABInv = relaxation / (d);
		}
		else
		{
			//disable the constraint row to handle singularity/redundant constraint
			solverConstraint.m_jacDiagABInv = 0.f;
		}
	}

	//compute rhs and remaining solverConstraint fields

	btScalar restitution = 0.f;
	btScalar penetration = isFriction ? 0 : cp.getDistance();

	btScalar rel_vel = 0.f;
	int ndofA = 0;
	int ndofB = 0;
	{
		btVector3 vel1, vel2;
		if (multiBodyA)
		{
			ndofA = multiBodyA->getNumDofs() + 6;
			btScalar* jacA = &m_data.m_jacobians[solverConstraint.m_jacAindex];
			for (int i = 0; i < ndofA; ++i)
				rel_vel += multiBodyA->getVelocityVector()[i] * jacA[i];
		}
		else
		{
			if (rb0)
			{
				btSolverBody* solverBodyA = &m_tmpSolverBodyPool[solverConstraint.m_solverBodyIdA];
				rel_vel += solverConstraint.m_contactNormal1.dot(rb0 ? solverBodyA->m_linearVelocity + solverBodyA->m_externalForceImpulse : btVector3(0, 0, 0)) + solverConstraint.m_relpos1CrossNormal.dot(rb0 ? solverBodyA->m_angularVelocity : btVector3(0, 0, 0));
			}
		}
		if (multiBodyB)
		{
			ndofB = multiBodyB->getNumDofs() + 6;
			btScalar* jacB = &m_data.m_jacobians[solverConstraint.m_jacBindex];
			for (int i = 0; i < ndofB; ++i)
				rel_vel += multiBodyB->getVelocityVector()[i] * jacB[i];
		}
		else
		{
			if (rb1)
			{
				btSolverBody* solverBodyB = &m_tmpSolverBodyPool[solverConstraint.m_solverBodyIdB];
				rel_vel += solverConstraint.m_contactNormal2.dot(rb1 ? solverBodyB->m_linearVelocity + solverBodyB->m_externalForceImpulse : btVector3(0, 0, 0)) + solverConstraint.m_relpos2CrossNormal.dot(rb1 ? solverBodyB->m_angularVelocity : btVector3(0, 0, 0));
			}
		}

		solverConstraint.m_friction = combinedTorsionalFriction;

		if (!isFriction)
		{
			restitution = restitutionCurve(rel_vel, cp.m_combinedRestitution, infoGlobal.m_restitutionVelocityThreshold);
			if (restitution <= btScalar(0.))
			{
				restitution = 0.f;
			}
		}
	}

	solverConstraint.m_appliedImpulse = 0.f;
	solverConstraint.m_appliedPushImpulse = 0.f;

	{
		btScalar velocityError = 0 - rel_vel;  // * damping;	//note for friction restitution is always set to 0 (check above) so it is acutally velocityError = -rel_vel for friction

		btScalar velocityImpulse = velocityError * solverConstraint.m_jacDiagABInv;

		solverConstraint.m_rhs = velocityImpulse;
		solverConstraint.m_rhsPenetration = 0.f;
		solverConstraint.m_lowerLimit = -solverConstraint.m_friction;
		solverConstraint.m_upperLimit = solverConstraint.m_friction;

		solverConstraint.m_cfm = infoGlobal.m_globalCfm * solverConstraint.m_jacDiagABInv;
	}
}

btMultiBodySolverConstraint& btMultiBodyConstraintSolver::addMultiBodyFrictionConstraint(const btVector3& normalAxis, const btScalar& appliedImpulse, btPersistentManifold* manifold, int frictionIndex, btManifoldPoint& cp, btCollisionObject* colObj0, btCollisionObject* colObj1, btScalar relaxation, const btContactSolverInfo& infoGlobal, btScalar desiredVelocity, btScalar cfmSlip)
{
	BT_PROFILE("addMultiBodyFrictionConstraint");
	btMultiBodySolverConstraint& solverConstraint = m_multiBodyFrictionContactConstraints.expandNonInitializing();
	solverConstraint.m_orgConstraint = 0;
	solverConstraint.m_orgDofIndex = -1;

	solverConstraint.m_frictionIndex = frictionIndex;
	bool isFriction = true;

	const btMultiBodyLinkCollider* fcA = btMultiBodyLinkCollider::upcast(manifold->getBody0());
	const btMultiBodyLinkCollider* fcB = btMultiBodyLinkCollider::upcast(manifold->getBody1());

	btMultiBody* mbA = fcA ? fcA->m_multiBody : 0;
	btMultiBody* mbB = fcB ? fcB->m_multiBody : 0;

	int solverBodyIdA = mbA ? -1 : getOrInitSolverBody(*colObj0, infoGlobal.m_timeStep);
	int solverBodyIdB = mbB ? -1 : getOrInitSolverBody(*colObj1, infoGlobal.m_timeStep);

	solverConstraint.m_solverBodyIdA = solverBodyIdA;
	solverConstraint.m_solverBodyIdB = solverBodyIdB;
	solverConstraint.m_multiBodyA = mbA;
	if (mbA)
		solverConstraint.m_linkA = fcA->m_link;

	solverConstraint.m_multiBodyB = mbB;
	if (mbB)
		solverConstraint.m_linkB = fcB->m_link;

	solverConstraint.m_originalContactPoint = &cp;

	setupMultiBodyContactConstraint(solverConstraint, normalAxis, 0, cp, infoGlobal, relaxation, isFriction, desiredVelocity, cfmSlip);
	return solverConstraint;
}

btMultiBodySolverConstraint& btMultiBodyConstraintSolver::addMultiBodyTorsionalFrictionConstraint(const btVector3& normalAxis, btPersistentManifold* manifold, int frictionIndex, btManifoldPoint& cp,
																								  btScalar combinedTorsionalFriction,
																								  btCollisionObject* colObj0, btCollisionObject* colObj1, btScalar relaxation, const btContactSolverInfo& infoGlobal, btScalar desiredVelocity, btScalar cfmSlip)
{
	BT_PROFILE("addMultiBodyRollingFrictionConstraint");

	bool useTorsionalAndConeFriction = (infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS && ((infoGlobal.m_solverMode & SOLVER_DISABLE_IMPLICIT_CONE_FRICTION) == 0));

	btMultiBodySolverConstraint& solverConstraint = useTorsionalAndConeFriction ? m_multiBodyTorsionalFrictionContactConstraints.expandNonInitializing() : m_multiBodyFrictionContactConstraints.expandNonInitializing();
	solverConstraint.m_orgConstraint = 0;
	solverConstraint.m_orgDofIndex = -1;

	solverConstraint.m_frictionIndex = frictionIndex;
	bool isFriction = true;

	const btMultiBodyLinkCollider* fcA = btMultiBodyLinkCollider::upcast(manifold->getBody0());
	const btMultiBodyLinkCollider* fcB = btMultiBodyLinkCollider::upcast(manifold->getBody1());

	btMultiBody* mbA = fcA ? fcA->m_multiBody : 0;
	btMultiBody* mbB = fcB ? fcB->m_multiBody : 0;

	int solverBodyIdA = mbA ? -1 : getOrInitSolverBody(*colObj0, infoGlobal.m_timeStep);
	int solverBodyIdB = mbB ? -1 : getOrInitSolverBody(*colObj1, infoGlobal.m_timeStep);

	solverConstraint.m_solverBodyIdA = solverBodyIdA;
	solverConstraint.m_solverBodyIdB = solverBodyIdB;
	solverConstraint.m_multiBodyA = mbA;
	if (mbA)
		solverConstraint.m_linkA = fcA->m_link;

	solverConstraint.m_multiBodyB = mbB;
	if (mbB)
		solverConstraint.m_linkB = fcB->m_link;

	solverConstraint.m_originalContactPoint = &cp;

	setupMultiBodyTorsionalFrictionConstraint(solverConstraint, normalAxis, cp, combinedTorsionalFriction, infoGlobal, relaxation, isFriction, desiredVelocity, cfmSlip);
	return solverConstraint;
}

btMultiBodySolverConstraint& btMultiBodyConstraintSolver::addMultiBodySpinningFrictionConstraint(const btVector3& normalAxis, btPersistentManifold* manifold, int frictionIndex, btManifoldPoint& cp,
	btScalar combinedTorsionalFriction,
	btCollisionObject* colObj0, btCollisionObject* colObj1, btScalar relaxation, const btContactSolverInfo& infoGlobal, btScalar desiredVelocity, btScalar cfmSlip)
{
	BT_PROFILE("addMultiBodyRollingFrictionConstraint");

	btMultiBodySolverConstraint& solverConstraint = m_multiBodySpinningFrictionContactConstraints.expandNonInitializing();
	solverConstraint.m_orgConstraint = 0;
	solverConstraint.m_orgDofIndex = -1;

	solverConstraint.m_frictionIndex = frictionIndex;
	bool isFriction = true;

	const btMultiBodyLinkCollider* fcA = btMultiBodyLinkCollider::upcast(manifold->getBody0());
	const btMultiBodyLinkCollider* fcB = btMultiBodyLinkCollider::upcast(manifold->getBody1());

	btMultiBody* mbA = fcA ? fcA->m_multiBody : 0;
	btMultiBody* mbB = fcB ? fcB->m_multiBody : 0;

	int solverBodyIdA = mbA ? -1 : getOrInitSolverBody(*colObj0, infoGlobal.m_timeStep);
	int solverBodyIdB = mbB ? -1 : getOrInitSolverBody(*colObj1, infoGlobal.m_timeStep);

	solverConstraint.m_solverBodyIdA = solverBodyIdA;
	solverConstraint.m_solverBodyIdB = solverBodyIdB;
	solverConstraint.m_multiBodyA = mbA;
	if (mbA)
		solverConstraint.m_linkA = fcA->m_link;

	solverConstraint.m_multiBodyB = mbB;
	if (mbB)
		solverConstraint.m_linkB = fcB->m_link;

	solverConstraint.m_originalContactPoint = &cp;

	setupMultiBodyTorsionalFrictionConstraint(solverConstraint, normalAxis, cp, combinedTorsionalFriction, infoGlobal, relaxation, isFriction, desiredVelocity, cfmSlip);
	return solverConstraint;
}
void btMultiBodyConstraintSolver::convertMultiBodyContact(btPersistentManifold* manifold, const btContactSolverInfo& infoGlobal)
{
	const btMultiBodyLinkCollider* fcA = btMultiBodyLinkCollider::upcast(manifold->getBody0());
	const btMultiBodyLinkCollider* fcB = btMultiBodyLinkCollider::upcast(manifold->getBody1());
	
	btMultiBody* mbA = fcA ? fcA->m_multiBody : 0;
	btMultiBody* mbB = fcB ? fcB->m_multiBody : 0;

	btCollisionObject *colObj0 = 0, *colObj1 = 0;

	colObj0 = (btCollisionObject*)manifold->getBody0();
	colObj1 = (btCollisionObject*)manifold->getBody1();

	int solverBodyIdA = mbA ? -1 : getOrInitSolverBody(*colObj0, infoGlobal.m_timeStep);
	int solverBodyIdB = mbB ? -1 : getOrInitSolverBody(*colObj1, infoGlobal.m_timeStep);

	//	btSolverBody* solverBodyA = mbA ? 0 : &m_tmpSolverBodyPool[solverBodyIdA];
	//	btSolverBody* solverBodyB = mbB ? 0 : &m_tmpSolverBodyPool[solverBodyIdB];

	///avoid collision response between two static objects
	//	if (!solverBodyA || (solverBodyA->m_invMass.isZero() && (!solverBodyB || solverBodyB->m_invMass.isZero())))
	//	return;

	//only a single rollingFriction per manifold
	int rollingFriction = 4;

	for (int j = 0; j < manifold->getNumContacts(); j++)
	{
		btManifoldPoint& cp = manifold->getContactPoint(j);

		if (cp.getDistance() <= manifold->getContactProcessingThreshold())
		{
			btScalar relaxation;

			int frictionIndex = m_multiBodyNormalContactConstraints.size();

			btMultiBodySolverConstraint& solverConstraint = m_multiBodyNormalContactConstraints.expandNonInitializing();

			//		btRigidBody* rb0 = btRigidBody::upcast(colObj0);
			//		btRigidBody* rb1 = btRigidBody::upcast(colObj1);
			solverConstraint.m_orgConstraint = 0;
			solverConstraint.m_orgDofIndex = -1;
			solverConstraint.m_solverBodyIdA = solverBodyIdA;
			solverConstraint.m_solverBodyIdB = solverBodyIdB;
			solverConstraint.m_multiBodyA = mbA;
			if (mbA)
				solverConstraint.m_linkA = fcA->m_link;

			solverConstraint.m_multiBodyB = mbB;
			if (mbB)
				solverConstraint.m_linkB = fcB->m_link;

			solverConstraint.m_originalContactPoint = &cp;

			bool isFriction = false;
			setupMultiBodyContactConstraint(solverConstraint, cp.m_normalWorldOnB, cp.m_appliedImpulse, cp, infoGlobal, relaxation, isFriction);

			//			const btVector3& pos1 = cp.getPositionWorldOnA();
			//			const btVector3& pos2 = cp.getPositionWorldOnB();

			/////setup the friction constraints
#define ENABLE_FRICTION
#ifdef ENABLE_FRICTION
			solverConstraint.m_frictionIndex = m_multiBodyFrictionContactConstraints.size();

			///Bullet has several options to set the friction directions
			///By default, each contact has only a single friction direction that is recomputed automatically every frame
			///based on the relative linear velocity.
			///If the relative velocity is zero, it will automatically compute a friction direction.

			///You can also enable two friction directions, using the SOLVER_USE_2_FRICTION_DIRECTIONS.
			///In that case, the second friction direction will be orthogonal to both contact normal and first friction direction.
			///
			///If you choose SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION, then the friction will be independent from the relative projected velocity.
			///
			///The user can manually override the friction directions for certain contacts using a contact callback,
			///and set the cp.m_lateralFrictionInitialized to true
			///In that case, you can set the target relative motion in each friction direction (cp.m_contactMotion1 and cp.m_contactMotion2)
			///this will give a conveyor belt effect
			///

			btPlaneSpace1(cp.m_normalWorldOnB, cp.m_lateralFrictionDir1, cp.m_lateralFrictionDir2);
			cp.m_lateralFrictionDir1.normalize();
			cp.m_lateralFrictionDir2.normalize();

			if (rollingFriction > 0)
			{
				if (cp.m_combinedSpinningFriction > 0)
				{
					addMultiBodySpinningFrictionConstraint(cp.m_normalWorldOnB, manifold, frictionIndex, cp, cp.m_combinedSpinningFriction, colObj0, colObj1, relaxation, infoGlobal);
				}
				if (cp.m_combinedRollingFriction > 0)
				{
					applyAnisotropicFriction(colObj0, cp.m_lateralFrictionDir1, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
					applyAnisotropicFriction(colObj1, cp.m_lateralFrictionDir1, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
					applyAnisotropicFriction(colObj0, cp.m_lateralFrictionDir2, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
					applyAnisotropicFriction(colObj1, cp.m_lateralFrictionDir2, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);

					addMultiBodyTorsionalFrictionConstraint(cp.m_lateralFrictionDir1, manifold, frictionIndex, cp, cp.m_combinedRollingFriction, colObj0, colObj1, relaxation, infoGlobal);
					addMultiBodyTorsionalFrictionConstraint(cp.m_lateralFrictionDir2, manifold, frictionIndex, cp, cp.m_combinedRollingFriction, colObj0, colObj1, relaxation, infoGlobal);
				}
				rollingFriction--;
			}
			if (!(infoGlobal.m_solverMode & SOLVER_ENABLE_FRICTION_DIRECTION_CACHING) || !(cp.m_contactPointFlags & BT_CONTACT_FLAG_LATERAL_FRICTION_INITIALIZED))
			{ /*
				cp.m_lateralFrictionDir1 = vel - cp.m_normalWorldOnB * rel_vel;
				btScalar lat_rel_vel = cp.m_lateralFrictionDir1.length2();
				if (!(infoGlobal.m_solverMode & SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION) && lat_rel_vel > SIMD_EPSILON)
				{
					cp.m_lateralFrictionDir1 *= 1.f/btSqrt(lat_rel_vel);
					if((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
					{
						cp.m_lateralFrictionDir2 = cp.m_lateralFrictionDir1.cross(cp.m_normalWorldOnB);
						cp.m_lateralFrictionDir2.normalize();//??
						applyAnisotropicFriction(colObj0,cp.m_lateralFrictionDir2,btCollisionObject::CF_ANISOTROPIC_FRICTION);
						applyAnisotropicFriction(colObj1,cp.m_lateralFrictionDir2,btCollisionObject::CF_ANISOTROPIC_FRICTION);
						addMultiBodyFrictionConstraint(cp.m_lateralFrictionDir2,solverBodyIdA,solverBodyIdB,frictionIndex,cp,rel_pos1,rel_pos2,colObj0,colObj1, relaxation);

					}

					applyAnisotropicFriction(colObj0,cp.m_lateralFrictionDir1,btCollisionObject::CF_ANISOTROPIC_FRICTION);
					applyAnisotropicFriction(colObj1,cp.m_lateralFrictionDir1,btCollisionObject::CF_ANISOTROPIC_FRICTION);
					addMultiBodyFrictionConstraint(cp.m_lateralFrictionDir1,solverBodyIdA,solverBodyIdB,frictionIndex,cp,rel_pos1,rel_pos2,colObj0,colObj1, relaxation);

				} else
				*/
				{
					applyAnisotropicFriction(colObj0, cp.m_lateralFrictionDir1, btCollisionObject::CF_ANISOTROPIC_FRICTION);
					applyAnisotropicFriction(colObj1, cp.m_lateralFrictionDir1, btCollisionObject::CF_ANISOTROPIC_FRICTION);
					addMultiBodyFrictionConstraint(cp.m_lateralFrictionDir1, cp.m_appliedImpulseLateral1, manifold, frictionIndex, cp, colObj0, colObj1, relaxation, infoGlobal);

					if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
					{
						applyAnisotropicFriction(colObj0, cp.m_lateralFrictionDir2, btCollisionObject::CF_ANISOTROPIC_FRICTION);
						applyAnisotropicFriction(colObj1, cp.m_lateralFrictionDir2, btCollisionObject::CF_ANISOTROPIC_FRICTION);
					  addMultiBodyFrictionConstraint(cp.m_lateralFrictionDir2, cp.m_appliedImpulseLateral2, manifold, frictionIndex, cp, colObj0, colObj1, relaxation, infoGlobal);
					}

					if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS) && (infoGlobal.m_solverMode & SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION))
					{
						cp.m_contactPointFlags |= BT_CONTACT_FLAG_LATERAL_FRICTION_INITIALIZED;
					}
				}
			}
			else
			{
				addMultiBodyFrictionConstraint(cp.m_lateralFrictionDir1, cp.m_appliedImpulseLateral1, manifold, frictionIndex, cp, colObj0, colObj1, relaxation, infoGlobal, cp.m_contactMotion1, cp.m_frictionCFM);

				if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
					addMultiBodyFrictionConstraint(cp.m_lateralFrictionDir2, cp.m_appliedImpulseLateral2, manifold, frictionIndex, cp, colObj0, colObj1, relaxation, infoGlobal, cp.m_contactMotion2, cp.m_frictionCFM);
				solverConstraint.m_appliedImpulse = 0.f;
				solverConstraint.m_appliedPushImpulse = 0.f;
      }

#endif  //ENABLE_FRICTION
		}
		else
		{
			// Reset quantities related to warmstart as 0.
			cp.m_appliedImpulse = 0;
			cp.m_prevRHS = 0;
		}
	}
}

void btMultiBodyConstraintSolver::convertContacts(btPersistentManifold** manifoldPtr, int numManifolds, const btContactSolverInfo& infoGlobal)
{
	for (int i = 0; i < numManifolds; i++)
	{
		btPersistentManifold* manifold = manifoldPtr[i];
		const btMultiBodyLinkCollider* fcA = btMultiBodyLinkCollider::upcast(manifold->getBody0());
		const btMultiBodyLinkCollider* fcB = btMultiBodyLinkCollider::upcast(manifold->getBody1());
		if (!fcA && !fcB)
		{
			//the contact doesn't involve any Featherstone btMultiBody, so deal with the regular btRigidBody/btCollisionObject case
			convertContact(manifold, infoGlobal);
		}
		else
		{
			convertMultiBodyContact(manifold, infoGlobal);
		}
	}

	//also convert the multibody constraints, if any

	for (int i = 0; i < m_tmpNumMultiBodyConstraints; i++)
	{
		btMultiBodyConstraint* c = m_tmpMultiBodyConstraints[i];
		m_data.m_solverBodyPool = &m_tmpSolverBodyPool;
		m_data.m_fixedBodyId = m_fixedBodyId;

		c->createConstraintRows(m_multiBodyNonContactConstraints, m_data, infoGlobal);
	}

	// Warmstart for noncontact constraints
	if (infoGlobal.m_solverMode & SOLVER_USE_ARTICULATED_WARMSTARTING)
	{
		for (int i = 0; i < m_multiBodyNonContactConstraints.size(); i++)
		{
			btMultiBodySolverConstraint& solverConstraint =
				m_multiBodyNonContactConstraints[i];
			solverConstraint.m_appliedImpulse =
				solverConstraint.m_orgConstraint->getAppliedImpulse(solverConstraint.m_orgDofIndex) *
				infoGlobal.m_articulatedWarmstartingFactor;

			btMultiBody* multiBodyA = solverConstraint.m_multiBodyA;
			btMultiBody* multiBodyB = solverConstraint.m_multiBodyB;
			if (solverConstraint.m_appliedImpulse)
			{
				if (multiBodyA)
				{
					int ndofA = multiBodyA->getNumDofs() + 6;
					btScalar* deltaV =
						&m_data.m_deltaVelocitiesUnitImpulse[solverConstraint.m_jacAindex];
					btScalar impulse = solverConstraint.m_appliedImpulse;
					multiBodyA->applyDeltaVeeMultiDof2(deltaV, impulse);
					applyDeltaVee(deltaV, impulse, solverConstraint.m_deltaVelAindex, ndofA);
				}
				if (multiBodyB)
				{
					int ndofB = multiBodyB->getNumDofs() + 6;
					btScalar* deltaV =
						&m_data.m_deltaVelocitiesUnitImpulse[solverConstraint.m_jacBindex];
					btScalar impulse = solverConstraint.m_appliedImpulse;
					multiBodyB->applyDeltaVeeMultiDof2(deltaV, impulse);
					applyDeltaVee(deltaV, impulse, solverConstraint.m_deltaVelBindex, ndofB);
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < m_multiBodyNonContactConstraints.size(); i++)
		{
			btMultiBodySolverConstraint& solverConstraint = m_multiBodyNonContactConstraints[i];
			solverConstraint.m_appliedImpulse = 0;
		}
	}
}

btScalar btMultiBodyConstraintSolver::solveGroup(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifold, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& info, btIDebugDraw* debugDrawer, btDispatcher* dispatcher)
{
	//printf("btMultiBodyConstraintSolver::solveGroup: numBodies=%d, numConstraints=%d\n", numBodies, numConstraints);
	return btSequentialImpulseConstraintSolver::solveGroup(bodies, numBodies, manifold, numManifolds, constraints, numConstraints, info, debugDrawer, dispatcher);
}

#if 0
static void applyJointFeedback(btMultiBodyJacobianData& data, const btMultiBodySolverConstraint& solverConstraint, int jacIndex, btMultiBody* mb, btScalar appliedImpulse)
{
	if (appliedImpulse!=0 && mb->internalNeedsJointFeedback())
	{
		//todo: get rid of those temporary memory allocations for the joint feedback
		btAlignedObjectArray<btScalar> forceVector;
		int numDofsPlusBase = 6+mb->getNumDofs();
		forceVector.resize(numDofsPlusBase);
		for (int i=0;i<numDofsPlusBase;i++)
		{
			forceVector[i] = data.m_jacobians[jacIndex+i]*appliedImpulse;
		}
		btAlignedObjectArray<btScalar> output;
		output.resize(numDofsPlusBase);
		bool applyJointFeedback = true;
		mb->calcAccelerationDeltasMultiDof(&forceVector[0],&output[0],data.scratch_r,data.scratch_v,applyJointFeedback);
	}
}
#endif

void btMultiBodyConstraintSolver::writeBackSolverBodyToMultiBody(btMultiBodySolverConstraint& c, btScalar deltaTime)
{
#if 1

	//bod->addBaseForce(m_gravity * bod->getBaseMass());
	//bod->addLinkForce(j, m_gravity * bod->getLinkMass(j));

	if (c.m_orgConstraint)
	{
		c.m_orgConstraint->internalSetAppliedImpulse(c.m_orgDofIndex, c.m_appliedImpulse);
	}

	if (c.m_multiBodyA)
	{
		c.m_multiBodyA->setCompanionId(-1);
		btVector3 force = c.m_contactNormal1 * (c.m_appliedImpulse / deltaTime);
		btVector3 torque = c.m_relpos1CrossNormal * (c.m_appliedImpulse / deltaTime);
		if (c.m_linkA < 0)
		{
			c.m_multiBodyA->addBaseConstraintForce(force);
			c.m_multiBodyA->addBaseConstraintTorque(torque);
		}
		else
		{
			c.m_multiBodyA->addLinkConstraintForce(c.m_linkA, force);
			//b3Printf("force = %f,%f,%f\n",force[0],force[1],force[2]);//[0],torque[1],torque[2]);
			c.m_multiBodyA->addLinkConstraintTorque(c.m_linkA, torque);
		}
	}

	if (c.m_multiBodyB)
	{
		{
			c.m_multiBodyB->setCompanionId(-1);
			btVector3 force = c.m_contactNormal2 * (c.m_appliedImpulse / deltaTime);
			btVector3 torque = c.m_relpos2CrossNormal * (c.m_appliedImpulse / deltaTime);
			if (c.m_linkB < 0)
			{
				c.m_multiBodyB->addBaseConstraintForce(force);
				c.m_multiBodyB->addBaseConstraintTorque(torque);
			}
			else
			{
				{
					c.m_multiBodyB->addLinkConstraintForce(c.m_linkB, force);
					//b3Printf("t = %f,%f,%f\n",force[0],force[1],force[2]);//[0],torque[1],torque[2]);
					c.m_multiBodyB->addLinkConstraintTorque(c.m_linkB, torque);
				}
			}
		}
	}
#endif

#ifndef DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS

	if (c.m_multiBodyA)
	{
		c.m_multiBodyA->applyDeltaVeeMultiDof(&m_data.m_deltaVelocitiesUnitImpulse[c.m_jacAindex], c.m_appliedImpulse);
	}

	if (c.m_multiBodyB)
	{
		c.m_multiBodyB->applyDeltaVeeMultiDof(&m_data.m_deltaVelocitiesUnitImpulse[c.m_jacBindex], c.m_appliedImpulse);
	}
#endif
}

btScalar btMultiBodyConstraintSolver::solveGroupCacheFriendlyFinish(btCollisionObject** bodies, int numBodies, const btContactSolverInfo& infoGlobal)
{
	BT_PROFILE("btMultiBodyConstraintSolver::solveGroupCacheFriendlyFinish");
	int numPoolConstraints = m_multiBodyNormalContactConstraints.size();

	//write back the delta v to the multi bodies, either as applied impulse (direct velocity change)
	//or as applied force, so we can measure the joint reaction forces easier
	for (int i = 0; i < numPoolConstraints; i++)
	{
		btMultiBodySolverConstraint& solverConstraint = m_multiBodyNormalContactConstraints[i];
		writeBackSolverBodyToMultiBody(solverConstraint, infoGlobal.m_timeStep);

		writeBackSolverBodyToMultiBody(m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex], infoGlobal.m_timeStep);

		if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
		{
			writeBackSolverBodyToMultiBody(m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex + 1], infoGlobal.m_timeStep);
		}
	}

	for (int i = 0; i < m_multiBodyNonContactConstraints.size(); i++)
	{
		btMultiBodySolverConstraint& solverConstraint = m_multiBodyNonContactConstraints[i];
		writeBackSolverBodyToMultiBody(solverConstraint, infoGlobal.m_timeStep);
	}


	{
		BT_PROFILE("warm starting write back");
		for (int j = 0; j < numPoolConstraints; j++)
		{
			const btMultiBodySolverConstraint& solverConstraint = m_multiBodyNormalContactConstraints[j];
			btManifoldPoint* pt = (btManifoldPoint*)solverConstraint.m_originalContactPoint;
			btAssert(pt);
			pt->m_appliedImpulse = solverConstraint.m_appliedImpulse;
 		  pt->m_prevRHS = solverConstraint.m_rhs;
			pt->m_appliedImpulseLateral1 = m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex].m_appliedImpulse;

			//printf("pt->m_appliedImpulseLateral1 = %f\n", pt->m_appliedImpulseLateral1);
			if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
			{
				pt->m_appliedImpulseLateral2 = m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex + 1].m_appliedImpulse;
			} else
			{
				pt->m_appliedImpulseLateral2 = 0;
			}
		}
	}

#if 0
	//multibody joint feedback
	{
		BT_PROFILE("multi body joint feedback");
		for (int j=0;j<numPoolConstraints;j++)
		{
			const btMultiBodySolverConstraint& solverConstraint = m_multiBodyNormalContactConstraints[j];
		
			//apply the joint feedback into all links of the btMultiBody
			//todo: double-check the signs of the applied impulse

			if(solverConstraint.m_multiBodyA && solverConstraint.m_multiBodyA->isMultiDof())
			{
				applyJointFeedback(m_data,solverConstraint, solverConstraint.m_jacAindex,solverConstraint.m_multiBodyA, solverConstraint.m_appliedImpulse*btSimdScalar(1./infoGlobal.m_timeStep));
			}
			if(solverConstraint.m_multiBodyB && solverConstraint.m_multiBodyB->isMultiDof())
			{
				applyJointFeedback(m_data,solverConstraint, solverConstraint.m_jacBindex,solverConstraint.m_multiBodyB,solverConstraint.m_appliedImpulse*btSimdScalar(-1./infoGlobal.m_timeStep));
			}
#if 0
			if (m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex].m_multiBodyA && m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex].m_multiBodyA->isMultiDof())
			{
				applyJointFeedback(m_data,m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex],
					m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex].m_jacAindex,
					m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex].m_multiBodyA,
					m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex].m_appliedImpulse*btSimdScalar(1./infoGlobal.m_timeStep));

			}
			if (m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex].m_multiBodyB && m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex].m_multiBodyB->isMultiDof())
			{
				applyJointFeedback(m_data,m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex],
					m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex].m_jacBindex,
					m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex].m_multiBodyB,
					m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex].m_appliedImpulse*btSimdScalar(-1./infoGlobal.m_timeStep));
			}
		
			if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
			{
				if (m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex+1].m_multiBodyA && m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex+1].m_multiBodyA->isMultiDof())
				{
					applyJointFeedback(m_data,m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex+1],
						m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex+1].m_jacAindex,
						m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex+1].m_multiBodyA,
						m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex+1].m_appliedImpulse*btSimdScalar(1./infoGlobal.m_timeStep));
				}

				if (m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex+1].m_multiBodyB && m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex+1].m_multiBodyB->isMultiDof())
				{
					applyJointFeedback(m_data,m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex+1],
						m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex+1].m_jacBindex,
						m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex+1].m_multiBodyB,
						m_multiBodyFrictionContactConstraints[solverConstraint.m_frictionIndex+1].m_appliedImpulse*btSimdScalar(-1./infoGlobal.m_timeStep));
				}
			}
#endif
		}
	
		for (int i=0;i<m_multiBodyNonContactConstraints.size();i++)
		{
			const btMultiBodySolverConstraint& solverConstraint = m_multiBodyNonContactConstraints[i];
			if(solverConstraint.m_multiBodyA && solverConstraint.m_multiBodyA->isMultiDof())
			{
				applyJointFeedback(m_data,solverConstraint, solverConstraint.m_jacAindex,solverConstraint.m_multiBodyA, solverConstraint.m_appliedImpulse*btSimdScalar(1./infoGlobal.m_timeStep));
			}
			if(solverConstraint.m_multiBodyB && solverConstraint.m_multiBodyB->isMultiDof())
			{
				applyJointFeedback(m_data,solverConstraint, solverConstraint.m_jacBindex,solverConstraint.m_multiBodyB,solverConstraint.m_appliedImpulse*btSimdScalar(1./infoGlobal.m_timeStep));
			}
		}
	}

	numPoolConstraints = m_multiBodyNonContactConstraints.size();

#if 0
	//@todo: m_originalContactPoint is not initialized for btMultiBodySolverConstraint
	for (int i=0;i<numPoolConstraints;i++)
	{
		const btMultiBodySolverConstraint& c = m_multiBodyNonContactConstraints[i];

		btTypedConstraint* constr = (btTypedConstraint*)c.m_originalContactPoint;
		btJointFeedback* fb = constr->getJointFeedback();
		if (fb)
		{
			fb->m_appliedForceBodyA += c.m_contactNormal1*c.m_appliedImpulse*constr->getRigidBodyA().getLinearFactor()/infoGlobal.m_timeStep;
			fb->m_appliedForceBodyB += c.m_contactNormal2*c.m_appliedImpulse*constr->getRigidBodyB().getLinearFactor()/infoGlobal.m_timeStep;
			fb->m_appliedTorqueBodyA += c.m_relpos1CrossNormal* constr->getRigidBodyA().getAngularFactor()*c.m_appliedImpulse/infoGlobal.m_timeStep;
			fb->m_appliedTorqueBodyB += c.m_relpos2CrossNormal* constr->getRigidBodyB().getAngularFactor()*c.m_appliedImpulse/infoGlobal.m_timeStep; /*RGM ???? */
			
		}

		constr->internalSetAppliedImpulse(c.m_appliedImpulse);
		if (btFabs(c.m_appliedImpulse)>=constr->getBreakingImpulseThreshold())
		{
			constr->setEnabled(false);
		}

	}
#endif
#endif

	return btSequentialImpulseConstraintSolver::solveGroupCacheFriendlyFinish(bodies, numBodies, infoGlobal);
}

void btMultiBodyConstraintSolver::solveMultiBodyGroup(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifold, int numManifolds, btTypedConstraint** constraints, int numConstraints, btMultiBodyConstraint** multiBodyConstraints, int numMultiBodyConstraints, const btContactSolverInfo& info, btIDebugDraw* debugDrawer, btDispatcher* dispatcher)
{
	//printf("solveMultiBodyGroup: numBodies=%d, numConstraints=%d, numManifolds=%d, numMultiBodyConstraints=%d\n", numBodies, numConstraints, numManifolds, numMultiBodyConstraints);

	//printf("solveMultiBodyGroup start\n");
	m_tmpMultiBodyConstraints = multiBodyConstraints;
	m_tmpNumMultiBodyConstraints = numMultiBodyConstraints;

	btSequentialImpulseConstraintSolver::solveGroup(bodies, numBodies, manifold, numManifolds, constraints, numConstraints, info, debugDrawer, dispatcher);

	m_tmpMultiBodyConstraints = 0;
	m_tmpNumMultiBodyConstraints = 0;
}
