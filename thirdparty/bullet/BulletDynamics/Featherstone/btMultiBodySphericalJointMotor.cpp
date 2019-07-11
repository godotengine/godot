/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2018 Erwin Coumans  http://bulletphysics.org

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

#include "btMultiBodySphericalJointMotor.h"
#include "btMultiBody.h"
#include "btMultiBodyLinkCollider.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "LinearMath/btTransformUtil.h"
#include "BulletDynamics/ConstraintSolver/btGeneric6DofSpring2Constraint.h"

btMultiBodySphericalJointMotor::btMultiBodySphericalJointMotor(btMultiBody* body, int link, btScalar maxMotorImpulse)
	: btMultiBodyConstraint(body, body, link, body->getLink(link).m_parent, 3, true),
	m_desiredVelocity(0, 0, 0),
	m_desiredPosition(0,0,0,1),
	m_kd(1.),
	m_kp(0.2),
	m_erp(1),
	m_rhsClamp(SIMD_INFINITY)
{

	m_maxAppliedImpulse = maxMotorImpulse;
}


void btMultiBodySphericalJointMotor::finalizeMultiDof()
{
	allocateJacobiansMultiDof();
	// note: we rely on the fact that data.m_jacobians are
	// always initialized to zero by the Constraint ctor
	int linkDoF = 0;
	unsigned int offset = 6 + (m_bodyA->getLink(m_linkA).m_dofOffset + linkDoF);

	// row 0: the lower bound
	// row 0: the lower bound
	jacobianA(0)[offset] = 1;

	m_numDofsFinalized = m_jacSizeBoth;
}


btMultiBodySphericalJointMotor::~btMultiBodySphericalJointMotor()
{
}

int btMultiBodySphericalJointMotor::getIslandIdA() const
{
	if (this->m_linkA < 0)
	{
		btMultiBodyLinkCollider* col = m_bodyA->getBaseCollider();
		if (col)
			return col->getIslandTag();
	}
	else
	{
		if (m_bodyA->getLink(m_linkA).m_collider)
		{
			return m_bodyA->getLink(m_linkA).m_collider->getIslandTag();
		}
	}
	return -1;
}

int btMultiBodySphericalJointMotor::getIslandIdB() const
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
		{
			return m_bodyB->getLink(m_linkB).m_collider->getIslandTag();
		}
	}
	return -1;
}

void btMultiBodySphericalJointMotor::createConstraintRows(btMultiBodyConstraintArray& constraintRows,
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

	const btScalar posError = 0;
	const btVector3 dummy(0, 0, 0);

	
	btVector3 axis[3] = { btVector3(1, 0, 0), btVector3(0, 1, 0), btVector3(0, 0, 1) };
	
	btQuaternion desiredQuat = m_desiredPosition;
	btQuaternion currentQuat(m_bodyA->getJointPosMultiDof(m_linkA)[0],
		m_bodyA->getJointPosMultiDof(m_linkA)[1],
		m_bodyA->getJointPosMultiDof(m_linkA)[2],
		m_bodyA->getJointPosMultiDof(m_linkA)[3]);

btQuaternion relRot = currentQuat.inverse() * desiredQuat;
	btVector3 angleDiff;
	btGeneric6DofSpring2Constraint::matrixToEulerXYZ(btMatrix3x3(relRot), angleDiff);



	for (int row = 0; row < getNumRows(); row++)
	{
		btMultiBodySolverConstraint& constraintRow = constraintRows.expandNonInitializing();

		int dof = row;
		
		btScalar currentVelocity = m_bodyA->getJointVelMultiDof(m_linkA)[dof];
		btScalar desiredVelocity = this->m_desiredVelocity[row];
		
		btScalar velocityError = desiredVelocity - currentVelocity;

		btMatrix3x3 frameAworld;
		frameAworld.setIdentity();
		frameAworld = m_bodyA->localFrameToWorld(m_linkA, frameAworld);
		btScalar posError = 0;
		{
			btAssert(m_bodyA->getLink(m_linkA).m_jointType == btMultibodyLink::eSpherical);
			switch (m_bodyA->getLink(m_linkA).m_jointType)
			{
				case btMultibodyLink::eSpherical:
				{
					btVector3 constraintNormalAng = frameAworld.getColumn(row % 3);
					posError = m_kp*angleDiff[row % 3];
					fillMultiBodyConstraint(constraintRow, data, 0, 0, constraintNormalAng,
						btVector3(0,0,0), dummy, dummy,
						posError,
						infoGlobal,
						-m_maxAppliedImpulse, m_maxAppliedImpulse, true);
					constraintRow.m_orgConstraint = this;
					constraintRow.m_orgDofIndex = row;
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
