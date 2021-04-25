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

#ifndef BT_MULTIBODY_CONSTRAINT_H
#define BT_MULTIBODY_CONSTRAINT_H

#include "LinearMath/btScalar.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "btMultiBody.h"


//Don't change any of the existing enum values, so add enum types at the end for serialization compatibility
enum btTypedMultiBodyConstraintType
{
	MULTIBODY_CONSTRAINT_LIMIT=3,
	MULTIBODY_CONSTRAINT_1DOF_JOINT_MOTOR,
	MULTIBODY_CONSTRAINT_GEAR,
	MULTIBODY_CONSTRAINT_POINT_TO_POINT,
	MULTIBODY_CONSTRAINT_SLIDER,
	MULTIBODY_CONSTRAINT_SPHERICAL_MOTOR,
	MULTIBODY_CONSTRAINT_FIXED,
	
	MAX_MULTIBODY_CONSTRAINT_TYPE,
};

class btMultiBody;
struct btSolverInfo;

#include "btMultiBodySolverConstraint.h"

struct btMultiBodyJacobianData
{
	btAlignedObjectArray<btScalar> m_jacobians;
	btAlignedObjectArray<btScalar> m_deltaVelocitiesUnitImpulse;  //holds the joint-space response of the corresp. tree to the test impulse in each constraint space dimension
	btAlignedObjectArray<btScalar> m_deltaVelocities;             //holds joint-space vectors of all the constrained trees accumulating the effect of corrective impulses applied in SI
	btAlignedObjectArray<btScalar> scratch_r;
	btAlignedObjectArray<btVector3> scratch_v;
	btAlignedObjectArray<btMatrix3x3> scratch_m;
	btAlignedObjectArray<btSolverBody>* m_solverBodyPool;
	int m_fixedBodyId;
};

ATTRIBUTE_ALIGNED16(class)
btMultiBodyConstraint
{
protected:
	btMultiBody* m_bodyA;
	btMultiBody* m_bodyB;
	int m_linkA;
	int m_linkB;

	int m_type; //btTypedMultiBodyConstraintType

	int m_numRows;
	int m_jacSizeA;
	int m_jacSizeBoth;
	int m_posOffset;

	bool m_isUnilateral;
	int m_numDofsFinalized;
	btScalar m_maxAppliedImpulse;

	// warning: the data block lay out is not consistent for all constraints
	// data block laid out as follows:
	// cached impulses. (one per row.)
	// jacobians. (interleaved, row1 body1 then row1 body2 then row2 body 1 etc)
	// positions. (one per row.)
	btAlignedObjectArray<btScalar> m_data;

	void applyDeltaVee(btMultiBodyJacobianData & data, btScalar * delta_vee, btScalar impulse, int velocityIndex, int ndof);

	btScalar fillMultiBodyConstraint(btMultiBodySolverConstraint & solverConstraint,
									 btMultiBodyJacobianData & data,
									 btScalar * jacOrgA, btScalar * jacOrgB,
									 const btVector3& constraintNormalAng,

									 const btVector3& constraintNormalLin,
									 const btVector3& posAworld, const btVector3& posBworld,
									 btScalar posError,
									 const btContactSolverInfo& infoGlobal,
									 btScalar lowerLimit, btScalar upperLimit,
									 bool angConstraint = false,

									 btScalar relaxation = 1.f,
									 bool isFriction = false, btScalar desiredVelocity = 0, btScalar cfmSlip = 0);

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	btMultiBodyConstraint(btMultiBody * bodyA, btMultiBody * bodyB, int linkA, int linkB, int numRows, bool isUnilateral, int type);
	virtual ~btMultiBodyConstraint();

	void updateJacobianSizes();
	void allocateJacobiansMultiDof();

	int getConstraintType() const
	{
		return m_type;
	}
	//many constraints have setFrameInB/setPivotInB. Will use 'getConstraintType' later.
	virtual void setFrameInB(const btMatrix3x3& frameInB) {}
	virtual void setPivotInB(const btVector3& pivotInB) {}

	virtual void finalizeMultiDof() = 0;

	virtual int getIslandIdA() const = 0;
	virtual int getIslandIdB() const = 0;

	virtual void createConstraintRows(btMultiBodyConstraintArray & constraintRows,
									  btMultiBodyJacobianData & data,
									  const btContactSolverInfo& infoGlobal) = 0;

	int getNumRows() const
	{
		return m_numRows;
	}

	btMultiBody* getMultiBodyA()
	{
		return m_bodyA;
	}
	btMultiBody* getMultiBodyB()
	{
		return m_bodyB;
	}

	int getLinkA() const
	{
		return m_linkA;
	}
	int getLinkB() const
	{
		return m_linkB;
	}
	void internalSetAppliedImpulse(int dof, btScalar appliedImpulse)
	{
		btAssert(dof >= 0);
		btAssert(dof < getNumRows());
		m_data[dof] = appliedImpulse;
	}

	btScalar getAppliedImpulse(int dof)
	{
		btAssert(dof >= 0);
		btAssert(dof < getNumRows());
		return m_data[dof];
	}
	// current constraint position
	// constraint is pos >= 0 for unilateral, or pos = 0 for bilateral
	// NOTE: ignored position for friction rows.
	btScalar getPosition(int row) const
	{
		return m_data[m_posOffset + row];
	}

	void setPosition(int row, btScalar pos)
	{
		m_data[m_posOffset + row] = pos;
	}

	bool isUnilateral() const
	{
		return m_isUnilateral;
	}

	// jacobian blocks.
	// each of size 6 + num_links. (jacobian2 is null if no body2.)
	// format: 3 'omega' coefficients, 3 'v' coefficients, then the 'qdot' coefficients.
	btScalar* jacobianA(int row)
	{
		return &m_data[m_numRows + row * m_jacSizeBoth];
	}
	const btScalar* jacobianA(int row) const
	{
		return &m_data[m_numRows + (row * m_jacSizeBoth)];
	}
	btScalar* jacobianB(int row)
	{
		return &m_data[m_numRows + (row * m_jacSizeBoth) + m_jacSizeA];
	}
	const btScalar* jacobianB(int row) const
	{
		return &m_data[m_numRows + (row * m_jacSizeBoth) + m_jacSizeA];
	}

	btScalar getMaxAppliedImpulse() const
	{
		return m_maxAppliedImpulse;
	}
	void setMaxAppliedImpulse(btScalar maxImp)
	{
		m_maxAppliedImpulse = maxImp;
	}

	virtual void debugDraw(class btIDebugDraw * drawer) = 0;

	virtual void setGearRatio(btScalar ratio) {}
	virtual void setGearAuxLink(int gearAuxLink) {}
	virtual void setRelativePositionTarget(btScalar relPosTarget) {}
	virtual void setErp(btScalar erp) {}
};

#endif  //BT_MULTIBODY_CONSTRAINT_H
