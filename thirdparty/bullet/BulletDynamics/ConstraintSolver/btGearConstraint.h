/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2012 Advanced Micro Devices, Inc.  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/



#ifndef BT_GEAR_CONSTRAINT_H
#define BT_GEAR_CONSTRAINT_H

#include "BulletDynamics/ConstraintSolver/btTypedConstraint.h"


#ifdef BT_USE_DOUBLE_PRECISION
#define btGearConstraintData	btGearConstraintDoubleData
#define btGearConstraintDataName	"btGearConstraintDoubleData"
#else
#define btGearConstraintData	btGearConstraintFloatData
#define btGearConstraintDataName	"btGearConstraintFloatData"
#endif //BT_USE_DOUBLE_PRECISION



///The btGeatConstraint will couple the angular velocity for two bodies around given local axis and ratio.
///See Bullet/Demos/ConstraintDemo for an example use.
class btGearConstraint : public btTypedConstraint
{
protected:
	btVector3	m_axisInA;
	btVector3	m_axisInB;
	bool		m_useFrameA;
	btScalar	m_ratio;

public:
	btGearConstraint(btRigidBody& rbA, btRigidBody& rbB, const btVector3& axisInA,const btVector3& axisInB, btScalar ratio=1.f);
	virtual ~btGearConstraint ();

	///internal method used by the constraint solver, don't use them directly
	virtual void getInfo1 (btConstraintInfo1* info);

	///internal method used by the constraint solver, don't use them directly
	virtual void getInfo2 (btConstraintInfo2* info);

	void setAxisA(btVector3& axisA) 
	{
		m_axisInA = axisA;
	}
	void setAxisB(btVector3& axisB)
	{
		m_axisInB = axisB;
	}
	void setRatio(btScalar ratio)
	{
		m_ratio = ratio;
	}
	const btVector3& getAxisA() const
	{
		return m_axisInA;
	}
	const btVector3& getAxisB() const
	{
		return m_axisInB;
	}
	btScalar getRatio() const
	{
		return m_ratio;
	}


	virtual	void	setParam(int num, btScalar value, int axis = -1) 
	{
		(void) num;
		(void) value;
		(void) axis;
		btAssert(0);
	}

	///return the local value of parameter
	virtual	btScalar getParam(int num, int axis = -1) const 
	{ 
		(void) num;
		(void) axis;
		btAssert(0);
		return 0.f;
	}

	virtual	int	calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual	const char*	serialize(void* dataBuffer, btSerializer* serializer) const;
};




///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btGearConstraintFloatData
{
	btTypedConstraintFloatData	m_typeConstraintData;

	btVector3FloatData			m_axisInA;
	btVector3FloatData			m_axisInB;

	float							m_ratio;
	char							m_padding[4];
};

struct btGearConstraintDoubleData
{
	btTypedConstraintDoubleData	m_typeConstraintData;

	btVector3DoubleData			m_axisInA;
	btVector3DoubleData			m_axisInB;

	double						m_ratio;
};

SIMD_FORCE_INLINE	int	btGearConstraint::calculateSerializeBufferSize() const
{
	return sizeof(btGearConstraintData);
}

	///fills the dataBuffer and returns the struct name (and 0 on failure)
SIMD_FORCE_INLINE	const char*	btGearConstraint::serialize(void* dataBuffer, btSerializer* serializer) const
{
	btGearConstraintData* gear = (btGearConstraintData*)dataBuffer;
	btTypedConstraint::serialize(&gear->m_typeConstraintData,serializer);

	m_axisInA.serialize( gear->m_axisInA );
	m_axisInB.serialize( gear->m_axisInB );

	gear->m_ratio = m_ratio;

	// Fill padding with zeros to appease msan.
#ifndef BT_USE_DOUBLE_PRECISION
	gear->m_padding[0] = 0;
	gear->m_padding[1] = 0;
	gear->m_padding[2] = 0;
	gear->m_padding[3] = 0;
#endif

	return btGearConstraintDataName;
}






#endif //BT_GEAR_CONSTRAINT_H
