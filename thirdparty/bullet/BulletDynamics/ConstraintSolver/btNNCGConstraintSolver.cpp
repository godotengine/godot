/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "btNNCGConstraintSolver.h"

btScalar btNNCGConstraintSolver::solveGroupCacheFriendlySetup(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifoldPtr, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* debugDrawer)
{
	btScalar val = btSequentialImpulseConstraintSolver::solveGroupCacheFriendlySetup(bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints, infoGlobal, debugDrawer);

	m_pNC.resizeNoInitialize(m_tmpSolverNonContactConstraintPool.size());
	m_pC.resizeNoInitialize(m_tmpSolverContactConstraintPool.size());
	m_pCF.resizeNoInitialize(m_tmpSolverContactFrictionConstraintPool.size());
	m_pCRF.resizeNoInitialize(m_tmpSolverContactRollingFrictionConstraintPool.size());

	m_deltafNC.resizeNoInitialize(m_tmpSolverNonContactConstraintPool.size());
	m_deltafC.resizeNoInitialize(m_tmpSolverContactConstraintPool.size());
	m_deltafCF.resizeNoInitialize(m_tmpSolverContactFrictionConstraintPool.size());
	m_deltafCRF.resizeNoInitialize(m_tmpSolverContactRollingFrictionConstraintPool.size());

	return val;
}

btScalar btNNCGConstraintSolver::solveSingleIteration(int iteration, btCollisionObject** /*bodies */, int /*numBodies*/, btPersistentManifold** /*manifoldPtr*/, int /*numManifolds*/, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* /*debugDrawer*/)
{
	int numNonContactPool = m_tmpSolverNonContactConstraintPool.size();
	int numConstraintPool = m_tmpSolverContactConstraintPool.size();
	int numFrictionPool = m_tmpSolverContactFrictionConstraintPool.size();

	if (infoGlobal.m_solverMode & SOLVER_RANDMIZE_ORDER)
	{
		if (1)  // uncomment this for a bit less random ((iteration & 7) == 0)
		{
			for (int j = 0; j < numNonContactPool; ++j)
			{
				int tmp = m_orderNonContactConstraintPool[j];
				int swapi = btRandInt2(j + 1);
				m_orderNonContactConstraintPool[j] = m_orderNonContactConstraintPool[swapi];
				m_orderNonContactConstraintPool[swapi] = tmp;
			}

			//contact/friction constraints are not solved more than
			if (iteration < infoGlobal.m_numIterations)
			{
				for (int j = 0; j < numConstraintPool; ++j)
				{
					int tmp = m_orderTmpConstraintPool[j];
					int swapi = btRandInt2(j + 1);
					m_orderTmpConstraintPool[j] = m_orderTmpConstraintPool[swapi];
					m_orderTmpConstraintPool[swapi] = tmp;
				}

				for (int j = 0; j < numFrictionPool; ++j)
				{
					int tmp = m_orderFrictionConstraintPool[j];
					int swapi = btRandInt2(j + 1);
					m_orderFrictionConstraintPool[j] = m_orderFrictionConstraintPool[swapi];
					m_orderFrictionConstraintPool[swapi] = tmp;
				}
			}
		}
	}

	btScalar deltaflengthsqr = 0;
	{
		for (int j = 0; j < m_tmpSolverNonContactConstraintPool.size(); j++)
		{
			btSolverConstraint& constraint = m_tmpSolverNonContactConstraintPool[m_orderNonContactConstraintPool[j]];
			if (iteration < constraint.m_overrideNumSolverIterations)
			{
				btScalar deltaf = resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[constraint.m_solverBodyIdA], m_tmpSolverBodyPool[constraint.m_solverBodyIdB], constraint);
				m_deltafNC[j] = deltaf;
				deltaflengthsqr += deltaf * deltaf;
			}
		}
	}

	if (m_onlyForNoneContact)
	{
		if (iteration == 0)
		{
			for (int j = 0; j < m_tmpSolverNonContactConstraintPool.size(); j++) m_pNC[j] = m_deltafNC[j];
		}
		else
		{
			// deltaflengthsqrprev can be 0 only if the solver solved the problem exactly in the previous iteration. In this case we should have quit, but mainly for debug reason with this 'hack' it is now allowed to continue the calculation
			btScalar beta = m_deltafLengthSqrPrev > 0 ? deltaflengthsqr / m_deltafLengthSqrPrev : 2;
			if (beta > 1)
			{
				for (int j = 0; j < m_tmpSolverNonContactConstraintPool.size(); j++) m_pNC[j] = 0;
			}
			else
			{
				for (int j = 0; j < m_tmpSolverNonContactConstraintPool.size(); j++)
				{
					btSolverConstraint& constraint = m_tmpSolverNonContactConstraintPool[m_orderNonContactConstraintPool[j]];
					if (iteration < constraint.m_overrideNumSolverIterations)
					{
						btScalar additionaldeltaimpulse = beta * m_pNC[j];
						constraint.m_appliedImpulse = btScalar(constraint.m_appliedImpulse) + additionaldeltaimpulse;
						m_pNC[j] = beta * m_pNC[j] + m_deltafNC[j];
						btSolverBody& body1 = m_tmpSolverBodyPool[constraint.m_solverBodyIdA];
						btSolverBody& body2 = m_tmpSolverBodyPool[constraint.m_solverBodyIdB];
						const btSolverConstraint& c = constraint;
						body1.internalApplyImpulse(c.m_contactNormal1 * body1.internalGetInvMass(), c.m_angularComponentA, additionaldeltaimpulse);
						body2.internalApplyImpulse(c.m_contactNormal2 * body2.internalGetInvMass(), c.m_angularComponentB, additionaldeltaimpulse);
					}
				}
			}
		}
		m_deltafLengthSqrPrev = deltaflengthsqr;
	}

	{
		if (iteration < infoGlobal.m_numIterations)
		{
			for (int j = 0; j < numConstraints; j++)
			{
				if (constraints[j]->isEnabled())
				{
					int bodyAid = getOrInitSolverBody(constraints[j]->getRigidBodyA(), infoGlobal.m_timeStep);
					int bodyBid = getOrInitSolverBody(constraints[j]->getRigidBodyB(), infoGlobal.m_timeStep);
					btSolverBody& bodyA = m_tmpSolverBodyPool[bodyAid];
					btSolverBody& bodyB = m_tmpSolverBodyPool[bodyBid];
					constraints[j]->solveConstraintObsolete(bodyA, bodyB, infoGlobal.m_timeStep);
				}
			}

			///solve all contact constraints
			if (infoGlobal.m_solverMode & SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS)
			{
				int numPoolConstraints = m_tmpSolverContactConstraintPool.size();
				int multiplier = (infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS) ? 2 : 1;

				for (int c = 0; c < numPoolConstraints; c++)
				{
					btScalar totalImpulse = 0;

					{
						const btSolverConstraint& solveManifold = m_tmpSolverContactConstraintPool[m_orderTmpConstraintPool[c]];
						btScalar deltaf = resolveSingleConstraintRowLowerLimit(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
						m_deltafC[c] = deltaf;
						deltaflengthsqr += deltaf * deltaf;
						totalImpulse = solveManifold.m_appliedImpulse;
					}
					bool applyFriction = true;
					if (applyFriction)
					{
						{
							btSolverConstraint& solveManifold = m_tmpSolverContactFrictionConstraintPool[m_orderFrictionConstraintPool[c * multiplier]];

							if (totalImpulse > btScalar(0))
							{
								solveManifold.m_lowerLimit = -(solveManifold.m_friction * totalImpulse);
								solveManifold.m_upperLimit = solveManifold.m_friction * totalImpulse;
								btScalar deltaf = resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
								m_deltafCF[c * multiplier] = deltaf;
								deltaflengthsqr += deltaf * deltaf;
							}
							else
							{
								m_deltafCF[c * multiplier] = 0;
							}
						}

						if (infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS)
						{
							btSolverConstraint& solveManifold = m_tmpSolverContactFrictionConstraintPool[m_orderFrictionConstraintPool[c * multiplier + 1]];

							if (totalImpulse > btScalar(0))
							{
								solveManifold.m_lowerLimit = -(solveManifold.m_friction * totalImpulse);
								solveManifold.m_upperLimit = solveManifold.m_friction * totalImpulse;
								btScalar deltaf = resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
								m_deltafCF[c * multiplier + 1] = deltaf;
								deltaflengthsqr += deltaf * deltaf;
							}
							else
							{
								m_deltafCF[c * multiplier + 1] = 0;
							}
						}
					}
				}
			}
			else  //SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS
			{
				//solve the friction constraints after all contact constraints, don't interleave them
				int numPoolConstraints = m_tmpSolverContactConstraintPool.size();
				int j;

				for (j = 0; j < numPoolConstraints; j++)
				{
					const btSolverConstraint& solveManifold = m_tmpSolverContactConstraintPool[m_orderTmpConstraintPool[j]];
					btScalar deltaf = resolveSingleConstraintRowLowerLimit(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
					m_deltafC[j] = deltaf;
					deltaflengthsqr += deltaf * deltaf;
				}

				///solve all friction constraints

				int numFrictionPoolConstraints = m_tmpSolverContactFrictionConstraintPool.size();
				for (j = 0; j < numFrictionPoolConstraints; j++)
				{
					btSolverConstraint& solveManifold = m_tmpSolverContactFrictionConstraintPool[m_orderFrictionConstraintPool[j]];
					btScalar totalImpulse = m_tmpSolverContactConstraintPool[solveManifold.m_frictionIndex].m_appliedImpulse;

					if (totalImpulse > btScalar(0))
					{
						solveManifold.m_lowerLimit = -(solveManifold.m_friction * totalImpulse);
						solveManifold.m_upperLimit = solveManifold.m_friction * totalImpulse;

						btScalar deltaf = resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
						m_deltafCF[j] = deltaf;
						deltaflengthsqr += deltaf * deltaf;
					}
					else
					{
						m_deltafCF[j] = 0;
					}
				}
			}

			{
				int numRollingFrictionPoolConstraints = m_tmpSolverContactRollingFrictionConstraintPool.size();
				for (int j = 0; j < numRollingFrictionPoolConstraints; j++)
				{
					btSolverConstraint& rollingFrictionConstraint = m_tmpSolverContactRollingFrictionConstraintPool[j];
					btScalar totalImpulse = m_tmpSolverContactConstraintPool[rollingFrictionConstraint.m_frictionIndex].m_appliedImpulse;
					if (totalImpulse > btScalar(0))
					{
						btScalar rollingFrictionMagnitude = rollingFrictionConstraint.m_friction * totalImpulse;
						if (rollingFrictionMagnitude > rollingFrictionConstraint.m_friction)
							rollingFrictionMagnitude = rollingFrictionConstraint.m_friction;

						rollingFrictionConstraint.m_lowerLimit = -rollingFrictionMagnitude;
						rollingFrictionConstraint.m_upperLimit = rollingFrictionMagnitude;

						btScalar deltaf = resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[rollingFrictionConstraint.m_solverBodyIdA], m_tmpSolverBodyPool[rollingFrictionConstraint.m_solverBodyIdB], rollingFrictionConstraint);
						m_deltafCRF[j] = deltaf;
						deltaflengthsqr += deltaf * deltaf;
					}
					else
					{
						m_deltafCRF[j] = 0;
					}
				}
			}
		}
	}

	if (!m_onlyForNoneContact)
	{
		if (iteration == 0)
		{
			for (int j = 0; j < m_tmpSolverNonContactConstraintPool.size(); j++) m_pNC[j] = m_deltafNC[j];
			for (int j = 0; j < m_tmpSolverContactConstraintPool.size(); j++) m_pC[j] = m_deltafC[j];
			for (int j = 0; j < m_tmpSolverContactFrictionConstraintPool.size(); j++) m_pCF[j] = m_deltafCF[j];
			for (int j = 0; j < m_tmpSolverContactRollingFrictionConstraintPool.size(); j++) m_pCRF[j] = m_deltafCRF[j];
		}
		else
		{
			// deltaflengthsqrprev can be 0 only if the solver solved the problem exactly in the previous iteration. In this case we should have quit, but mainly for debug reason with this 'hack' it is now allowed to continue the calculation
			btScalar beta = m_deltafLengthSqrPrev > 0 ? deltaflengthsqr / m_deltafLengthSqrPrev : 2;
			if (beta > 1)
			{
				for (int j = 0; j < m_tmpSolverNonContactConstraintPool.size(); j++) m_pNC[j] = 0;
				for (int j = 0; j < m_tmpSolverContactConstraintPool.size(); j++) m_pC[j] = 0;
				for (int j = 0; j < m_tmpSolverContactFrictionConstraintPool.size(); j++) m_pCF[j] = 0;
				for (int j = 0; j < m_tmpSolverContactRollingFrictionConstraintPool.size(); j++) m_pCRF[j] = 0;
			}
			else
			{
				for (int j = 0; j < m_tmpSolverNonContactConstraintPool.size(); j++)
				{
					btSolverConstraint& constraint = m_tmpSolverNonContactConstraintPool[m_orderNonContactConstraintPool[j]];
					if (iteration < constraint.m_overrideNumSolverIterations)
					{
						btScalar additionaldeltaimpulse = beta * m_pNC[j];
						constraint.m_appliedImpulse = btScalar(constraint.m_appliedImpulse) + additionaldeltaimpulse;
						m_pNC[j] = beta * m_pNC[j] + m_deltafNC[j];
						btSolverBody& body1 = m_tmpSolverBodyPool[constraint.m_solverBodyIdA];
						btSolverBody& body2 = m_tmpSolverBodyPool[constraint.m_solverBodyIdB];
						const btSolverConstraint& c = constraint;
						body1.internalApplyImpulse(c.m_contactNormal1 * body1.internalGetInvMass(), c.m_angularComponentA, additionaldeltaimpulse);
						body2.internalApplyImpulse(c.m_contactNormal2 * body2.internalGetInvMass(), c.m_angularComponentB, additionaldeltaimpulse);
					}
				}
				for (int j = 0; j < m_tmpSolverContactConstraintPool.size(); j++)
				{
					btSolverConstraint& constraint = m_tmpSolverContactConstraintPool[m_orderTmpConstraintPool[j]];
					if (iteration < infoGlobal.m_numIterations)
					{
						btScalar additionaldeltaimpulse = beta * m_pC[j];
						constraint.m_appliedImpulse = btScalar(constraint.m_appliedImpulse) + additionaldeltaimpulse;
						m_pC[j] = beta * m_pC[j] + m_deltafC[j];
						btSolverBody& body1 = m_tmpSolverBodyPool[constraint.m_solverBodyIdA];
						btSolverBody& body2 = m_tmpSolverBodyPool[constraint.m_solverBodyIdB];
						const btSolverConstraint& c = constraint;
						body1.internalApplyImpulse(c.m_contactNormal1 * body1.internalGetInvMass(), c.m_angularComponentA, additionaldeltaimpulse);
						body2.internalApplyImpulse(c.m_contactNormal2 * body2.internalGetInvMass(), c.m_angularComponentB, additionaldeltaimpulse);
					}
				}
				for (int j = 0; j < m_tmpSolverContactFrictionConstraintPool.size(); j++)
				{
					btSolverConstraint& constraint = m_tmpSolverContactFrictionConstraintPool[m_orderFrictionConstraintPool[j]];
					if (iteration < infoGlobal.m_numIterations)
					{
						btScalar additionaldeltaimpulse = beta * m_pCF[j];
						constraint.m_appliedImpulse = btScalar(constraint.m_appliedImpulse) + additionaldeltaimpulse;
						m_pCF[j] = beta * m_pCF[j] + m_deltafCF[j];
						btSolverBody& body1 = m_tmpSolverBodyPool[constraint.m_solverBodyIdA];
						btSolverBody& body2 = m_tmpSolverBodyPool[constraint.m_solverBodyIdB];
						const btSolverConstraint& c = constraint;
						body1.internalApplyImpulse(c.m_contactNormal1 * body1.internalGetInvMass(), c.m_angularComponentA, additionaldeltaimpulse);
						body2.internalApplyImpulse(c.m_contactNormal2 * body2.internalGetInvMass(), c.m_angularComponentB, additionaldeltaimpulse);
					}
				}
				{
					for (int j = 0; j < m_tmpSolverContactRollingFrictionConstraintPool.size(); j++)
					{
						btSolverConstraint& constraint = m_tmpSolverContactRollingFrictionConstraintPool[j];
						if (iteration < infoGlobal.m_numIterations)
						{
							btScalar additionaldeltaimpulse = beta * m_pCRF[j];
							constraint.m_appliedImpulse = btScalar(constraint.m_appliedImpulse) + additionaldeltaimpulse;
							m_pCRF[j] = beta * m_pCRF[j] + m_deltafCRF[j];
							btSolverBody& body1 = m_tmpSolverBodyPool[constraint.m_solverBodyIdA];
							btSolverBody& body2 = m_tmpSolverBodyPool[constraint.m_solverBodyIdB];
							const btSolverConstraint& c = constraint;
							body1.internalApplyImpulse(c.m_contactNormal1 * body1.internalGetInvMass(), c.m_angularComponentA, additionaldeltaimpulse);
							body2.internalApplyImpulse(c.m_contactNormal2 * body2.internalGetInvMass(), c.m_angularComponentB, additionaldeltaimpulse);
						}
					}
				}
			}
		}
		m_deltafLengthSqrPrev = deltaflengthsqr;
	}

	return deltaflengthsqr;
}

btScalar btNNCGConstraintSolver::solveGroupCacheFriendlyFinish(btCollisionObject** bodies, int numBodies, const btContactSolverInfo& infoGlobal)
{
	m_pNC.resizeNoInitialize(0);
	m_pC.resizeNoInitialize(0);
	m_pCF.resizeNoInitialize(0);
	m_pCRF.resizeNoInitialize(0);

	m_deltafNC.resizeNoInitialize(0);
	m_deltafC.resizeNoInitialize(0);
	m_deltafCF.resizeNoInitialize(0);
	m_deltafCRF.resizeNoInitialize(0);

	return btSequentialImpulseConstraintSolver::solveGroupCacheFriendlyFinish(bodies, numBodies, infoGlobal);
}
