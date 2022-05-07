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

#ifndef BT_POINT2POINTCONSTRAINT_H
#define BT_POINT2POINTCONSTRAINT_H

#include "LinearMath/btVector3.h"
#include "btJacobianEntry.h"
#include "btTypedConstraint.h"

class btRigidBody;

#ifdef BT_USE_DOUBLE_PRECISION
#define btPoint2PointConstraintData2 btPoint2PointConstraintDoubleData2
#define btPoint2PointConstraintDataName "btPoint2PointConstraintDoubleData2"
#else
#define btPoint2PointConstraintData2 btPoint2PointConstraintFloatData
#define btPoint2PointConstraintDataName "btPoint2PointConstraintFloatData"
#endif  //BT_USE_DOUBLE_PRECISION

struct btConstraintSetting
{
	btConstraintSetting() : m_tau(btScalar(0.3)),
							m_damping(btScalar(1.)),
							m_impulseClamp(btScalar(0.))
	{
	}
	btScalar m_tau;
	btScalar m_damping;
	btScalar m_impulseClamp;
};

enum btPoint2PointFlags
{
	BT_P2P_FLAGS_ERP = 1,
	BT_P2P_FLAGS_CFM = 2
};

/// point to point constraint between two rigidbodies each with a pivotpoint that descibes the 'ballsocket' location in local space
ATTRIBUTE_ALIGNED16(class)
btPoint2PointConstraint : public btTypedConstraint
{
#ifdef IN_PARALLELL_SOLVER
public:
#endif
	btJacobianEntry m_jac[3];  //3 orthogonal linear constraints

	btVector3 m_pivotInA;
	btVector3 m_pivotInB;

	int m_flags;
	btScalar m_erp;
	btScalar m_cfm;

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	///for backwards compatibility during the transition to 'getInfo/getInfo2'
	bool m_useSolveConstraintObsolete;

	btConstraintSetting m_setting;

	btPoint2PointConstraint(btRigidBody & rbA, btRigidBody & rbB, const btVector3& pivotInA, const btVector3& pivotInB);

	btPoint2PointConstraint(btRigidBody & rbA, const btVector3& pivotInA);

	virtual void buildJacobian();

	virtual void getInfo1(btConstraintInfo1 * info);

	void getInfo1NonVirtual(btConstraintInfo1 * info);

	virtual void getInfo2(btConstraintInfo2 * info);

	void getInfo2NonVirtual(btConstraintInfo2 * info, const btTransform& body0_trans, const btTransform& body1_trans);

	void updateRHS(btScalar timeStep);

	void setPivotA(const btVector3& pivotA)
	{
		m_pivotInA = pivotA;
	}

	void setPivotB(const btVector3& pivotB)
	{
		m_pivotInB = pivotB;
	}

	const btVector3& getPivotInA() const
	{
		return m_pivotInA;
	}

	const btVector3& getPivotInB() const
	{
		return m_pivotInB;
	}

	///override the default global value of a parameter (such as ERP or CFM), optionally provide the axis (0..5).
	///If no axis is provided, it uses the default axis for this constraint.
	virtual void setParam(int num, btScalar value, int axis = -1);
	///return the local value of parameter
	virtual btScalar getParam(int num, int axis = -1) const;

	virtual int getFlags() const
	{
		return m_flags;
	}

	virtual int calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual const char* serialize(void* dataBuffer, btSerializer* serializer) const;
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btPoint2PointConstraintFloatData
{
	btTypedConstraintData m_typeConstraintData;
	btVector3FloatData m_pivotInA;
	btVector3FloatData m_pivotInB;
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btPoint2PointConstraintDoubleData2
{
	btTypedConstraintDoubleData m_typeConstraintData;
	btVector3DoubleData m_pivotInA;
	btVector3DoubleData m_pivotInB;
};

#ifdef BT_BACKWARDS_COMPATIBLE_SERIALIZATION
///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
///this structure is not used, except for loading pre-2.82 .bullet files
///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btPoint2PointConstraintDoubleData
{
	btTypedConstraintData m_typeConstraintData;
	btVector3DoubleData m_pivotInA;
	btVector3DoubleData m_pivotInB;
};
#endif  //BT_BACKWARDS_COMPATIBLE_SERIALIZATION

SIMD_FORCE_INLINE int btPoint2PointConstraint::calculateSerializeBufferSize() const
{
	return sizeof(btPoint2PointConstraintData2);
}

///fills the dataBuffer and returns the struct name (and 0 on failure)
SIMD_FORCE_INLINE const char* btPoint2PointConstraint::serialize(void* dataBuffer, btSerializer* serializer) const
{
	btPoint2PointConstraintData2* p2pData = (btPoint2PointConstraintData2*)dataBuffer;

	btTypedConstraint::serialize(&p2pData->m_typeConstraintData, serializer);
	m_pivotInA.serialize(p2pData->m_pivotInA);
	m_pivotInB.serialize(p2pData->m_pivotInB);

	return btPoint2PointConstraintDataName;
}

#endif  //BT_POINT2POINTCONSTRAINT_H
