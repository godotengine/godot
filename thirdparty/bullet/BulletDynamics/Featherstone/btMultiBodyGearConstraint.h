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

#ifndef BT_MULTIBODY_GEAR_CONSTRAINT_H
#define BT_MULTIBODY_GEAR_CONSTRAINT_H

#include "btMultiBodyConstraint.h"

class btMultiBodyGearConstraint : public btMultiBodyConstraint
{
protected:
	btRigidBody* m_rigidBodyA;
	btRigidBody* m_rigidBodyB;
	btVector3 m_pivotInA;
	btVector3 m_pivotInB;
	btMatrix3x3 m_frameInA;
	btMatrix3x3 m_frameInB;
	btScalar m_gearRatio;
	int m_gearAuxLink;
	btScalar m_erp;
	btScalar m_relativePositionTarget;

public:
	//btMultiBodyGearConstraint(btMultiBody* body, int link, btRigidBody* bodyB, const btVector3& pivotInA, const btVector3& pivotInB, const btMatrix3x3& frameInA, const btMatrix3x3& frameInB);
	btMultiBodyGearConstraint(btMultiBody* bodyA, int linkA, btMultiBody* bodyB, int linkB, const btVector3& pivotInA, const btVector3& pivotInB, const btMatrix3x3& frameInA, const btMatrix3x3& frameInB);

	virtual ~btMultiBodyGearConstraint();

	virtual void finalizeMultiDof();

	virtual int getIslandIdA() const;
	virtual int getIslandIdB() const;

	virtual void createConstraintRows(btMultiBodyConstraintArray& constraintRows,
									  btMultiBodyJacobianData& data,
									  const btContactSolverInfo& infoGlobal);

	const btVector3& getPivotInA() const
	{
		return m_pivotInA;
	}

	void setPivotInA(const btVector3& pivotInA)
	{
		m_pivotInA = pivotInA;
	}

	const btVector3& getPivotInB() const
	{
		return m_pivotInB;
	}

	virtual void setPivotInB(const btVector3& pivotInB)
	{
		m_pivotInB = pivotInB;
	}

	const btMatrix3x3& getFrameInA() const
	{
		return m_frameInA;
	}

	void setFrameInA(const btMatrix3x3& frameInA)
	{
		m_frameInA = frameInA;
	}

	const btMatrix3x3& getFrameInB() const
	{
		return m_frameInB;
	}

	virtual void setFrameInB(const btMatrix3x3& frameInB)
	{
		m_frameInB = frameInB;
	}

	virtual void debugDraw(class btIDebugDraw* drawer)
	{
		//todo(erwincoumans)
	}

	virtual void setGearRatio(btScalar gearRatio)
	{
		m_gearRatio = gearRatio;
	}
	virtual void setGearAuxLink(int gearAuxLink)
	{
		m_gearAuxLink = gearAuxLink;
	}
	virtual void setRelativePositionTarget(btScalar relPosTarget)
	{
		m_relativePositionTarget = relPosTarget;
	}
	virtual void setErp(btScalar erp)
	{
		m_erp = erp;
	}
};

#endif  //BT_MULTIBODY_GEAR_CONSTRAINT_H
