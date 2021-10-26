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

#ifndef BT_MULTIBODY_SPHERICAL_JOINT_MOTOR_H
#define BT_MULTIBODY_SPHERICAL_JOINT_MOTOR_H

#include "btMultiBodyConstraint.h"
struct btSolverInfo;

class btMultiBodySphericalJointMotor : public btMultiBodyConstraint
{
protected:
	btVector3 m_desiredVelocity;
	btQuaternion m_desiredPosition;
	bool m_use_multi_dof_params;
	btVector3 m_kd;
	btVector3 m_kp;
	btScalar m_erp;
	btScalar m_rhsClamp;  //maximum error
	btVector3 m_maxAppliedImpulseMultiDof;
	btVector3 m_damping;

public:
	btMultiBodySphericalJointMotor(btMultiBody* body, int link, btScalar maxMotorImpulse);
	
	virtual ~btMultiBodySphericalJointMotor();
	virtual void finalizeMultiDof();

	virtual int getIslandIdA() const;
	virtual int getIslandIdB() const;

	virtual void createConstraintRows(btMultiBodyConstraintArray& constraintRows,
									  btMultiBodyJacobianData& data,
									  const btContactSolverInfo& infoGlobal);

	virtual void setVelocityTarget(const btVector3& velTarget, btScalar kd = 1.0)
	{
		m_desiredVelocity = velTarget;
		m_kd = btVector3(kd, kd, kd);
		m_use_multi_dof_params = false;
	}

	virtual void setVelocityTargetMultiDof(const btVector3& velTarget, const btVector3& kd = btVector3(1.0, 1.0, 1.0))
	{
		m_desiredVelocity = velTarget;
		m_kd = kd;
		m_use_multi_dof_params = true;
	}

	virtual void setPositionTarget(const btQuaternion& posTarget, btScalar kp =1.f)
	{
		m_desiredPosition = posTarget;
		m_kp = btVector3(kp, kp, kp);
		m_use_multi_dof_params = false;
	}

	virtual void setPositionTargetMultiDof(const btQuaternion& posTarget, const btVector3& kp = btVector3(1.f, 1.f, 1.f))
	{
		m_desiredPosition = posTarget;
		m_kp = kp;
		m_use_multi_dof_params = true;
	}

	virtual void setErp(btScalar erp)
	{
		m_erp = erp;
	}
	virtual btScalar getErp() const
	{
		return m_erp;
	}
	virtual void setRhsClamp(btScalar rhsClamp)
	{
		m_rhsClamp = rhsClamp;
	}

	btScalar getMaxAppliedImpulseMultiDof(int i) const
	{
		return m_maxAppliedImpulseMultiDof[i];
	}

	void setMaxAppliedImpulseMultiDof(const btVector3& maxImp)
	{
		m_maxAppliedImpulseMultiDof = maxImp;
		m_use_multi_dof_params = true;
	}

	btScalar getDamping(int i) const
	{
		return m_damping[i];
	}

	void setDamping(const btVector3& damping)
	{
		m_damping = damping;
	}

	virtual void debugDraw(class btIDebugDraw* drawer)
	{
		//todo(erwincoumans)
	}
};

#endif  //BT_MULTIBODY_SPHERICAL_JOINT_MOTOR_H
