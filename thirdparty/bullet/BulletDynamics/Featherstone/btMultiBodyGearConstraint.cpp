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

///This file was written by Erwin Coumans

#include "btMultiBodyGearConstraint.h"
#include "btMultiBody.h"
#include "btMultiBodyLinkCollider.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"

btMultiBodyGearConstraint::btMultiBodyGearConstraint(btMultiBody* bodyA, int linkA, btMultiBody* bodyB, int linkB, const btVector3& pivotInA, const btVector3& pivotInB, const btMatrix3x3& frameInA, const btMatrix3x3& frameInB)
	: btMultiBodyConstraint(bodyA, bodyB, linkA, linkB, 1, false, MULTIBODY_CONSTRAINT_GEAR),
	  m_gearRatio(1),
	  m_gearAuxLink(-1),
	  m_erp(0),
	  m_relativePositionTarget(0)
{
}

void btMultiBodyGearConstraint::finalizeMultiDof()
{
	allocateJacobiansMultiDof();

	m_numDofsFinalized = m_jacSizeBoth;
}

btMultiBodyGearConstraint::~btMultiBodyGearConstraint()
{
}

int btMultiBodyGearConstraint::getIslandIdA() const
{
	if (m_bodyA)
	{
		if (m_linkA < 0)
		{
			btMultiBodyLinkCollider* col = m_bodyA->getBaseCollider();
			if (col)
				return col->getIslandTag();
		}
		else
		{
			if (m_bodyA->getLink(m_linkA).m_collider)
				return m_bodyA->getLink(m_linkA).m_collider->getIslandTag();
		}
	}
	return -1;
}

int btMultiBodyGearConstraint::getIslandIdB() const
{
	if (m_bodyB)
	{
		if (m_linkB < 0)
		{
			btMultiBodyLinkCollider* col = m_bodyB->getBaseCollider();
			if (col)
				return col->getIslandTag();
		}
		else
		{
			if (m_bodyB->getLink(m_linkB).m_collider)
				return m_bodyB->getLink(m_linkB).m_collider->getIslandTag();
		}
	}
	return -1;
}

void btMultiBodyGearConstraint::createConstraintRows(btMultiBodyConstraintArray& constraintRows,
													 btMultiBodyJacobianData& data,
													 const btContactSolverInfo& infoGlobal)
{
	// only positions need to be updated -- data.m_jacobians and force
	// directions were set in the ctor and never change.

	if (m_numDofsFinalized != m_jacSizeBoth)
	{
		finalizeMultiDof();
	}

	//don't crash
	if (m_numDofsFinalized != m_jacSizeBoth)
		return;

	if (m_maxAppliedImpulse == 0.f)
		return;

	// note: we rely on the fact that data.m_jacobians are
	// always initialized to zero by the Constraint ctor
	int linkDoF = 0;
	unsigned int offsetA = 6 + (m_bodyA->getLink(m_linkA).m_dofOffset + linkDoF);
	unsigned int offsetB = 6 + (m_bodyB->getLink(m_linkB).m_dofOffset + linkDoF);

	// row 0: the lower bound
	jacobianA(0)[offsetA] = 1;
	jacobianB(0)[offsetB] = m_gearRatio;

	btScalar posError = 0;
	const btVector3 dummy(0, 0, 0);

	btScalar kp = 1;
	btScalar kd = 1;
	int numRows = getNumRows();

	for (int row = 0; row < numRows; row++)
	{
		btMultiBodySolverConstraint& constraintRow = constraintRows.expandNonInitializing();

		int dof = 0;
		btScalar currentPosition = m_bodyA->getJointPosMultiDof(m_linkA)[dof];
		btScalar currentVelocity = m_bodyA->getJointVelMultiDof(m_linkA)[dof];
		btScalar auxVel = 0;

		if (m_gearAuxLink >= 0)
		{
			auxVel = m_bodyA->getJointVelMultiDof(m_gearAuxLink)[dof];
		}
		currentVelocity += auxVel;
		if (m_erp != 0)
		{
			btScalar currentPositionA = m_bodyA->getJointPosMultiDof(m_linkA)[dof];
			if (m_gearAuxLink >= 0)
			{
				currentPositionA -= m_bodyA->getJointPosMultiDof(m_gearAuxLink)[dof];
			}
			btScalar currentPositionB = m_gearRatio * m_bodyA->getJointPosMultiDof(m_linkB)[dof];
			btScalar diff = currentPositionB + currentPositionA;
			btScalar desiredPositionDiff = this->m_relativePositionTarget;
			posError = -m_erp * (desiredPositionDiff - diff);
		}

		btScalar desiredRelativeVelocity = auxVel;

		fillMultiBodyConstraint(constraintRow, data, jacobianA(row), jacobianB(row), dummy, dummy, dummy, dummy, posError, infoGlobal, -m_maxAppliedImpulse, m_maxAppliedImpulse, false, 1, false, desiredRelativeVelocity);

		constraintRow.m_orgConstraint = this;
		constraintRow.m_orgDofIndex = row;
		{
			//expect either prismatic or revolute joint type for now
			btAssert((m_bodyA->getLink(m_linkA).m_jointType == btMultibodyLink::eRevolute) || (m_bodyA->getLink(m_linkA).m_jointType == btMultibodyLink::ePrismatic));
			switch (m_bodyA->getLink(m_linkA).m_jointType)
			{
				case btMultibodyLink::eRevolute:
				{
					constraintRow.m_contactNormal1.setZero();
					constraintRow.m_contactNormal2.setZero();
					btVector3 revoluteAxisInWorld = quatRotate(m_bodyA->getLink(m_linkA).m_cachedWorldTransform.getRotation(), m_bodyA->getLink(m_linkA).m_axes[0].m_topVec);
					constraintRow.m_relpos1CrossNormal = revoluteAxisInWorld;
					constraintRow.m_relpos2CrossNormal = -revoluteAxisInWorld;

					break;
				}
				case btMultibodyLink::ePrismatic:
				{
					btVector3 prismaticAxisInWorld = quatRotate(m_bodyA->getLink(m_linkA).m_cachedWorldTransform.getRotation(), m_bodyA->getLink(m_linkA).m_axes[0].m_bottomVec);
					constraintRow.m_contactNormal1 = prismaticAxisInWorld;
					constraintRow.m_contactNormal2 = -prismaticAxisInWorld;
					constraintRow.m_relpos1CrossNormal.setZero();
					constraintRow.m_relpos2CrossNormal.setZero();
					break;
				}
				default:
				{
					btAssert(0);
				}
			};
		}
	}
}
