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

/// Implemented by Erwin Coumans. The idea for the constraint comes from Dimitris Papavasiliou.

#include "btGearConstraint.h"

btGearConstraint::btGearConstraint(btRigidBody& rbA, btRigidBody& rbB, const btVector3& axisInA,const btVector3& axisInB, btScalar ratio)
:btTypedConstraint(GEAR_CONSTRAINT_TYPE,rbA,rbB),
m_axisInA(axisInA),
m_axisInB(axisInB),
m_ratio(ratio)
{
}

btGearConstraint::~btGearConstraint ()
{
}

void btGearConstraint::getInfo1 (btConstraintInfo1* info)
{
	info->m_numConstraintRows = 1;
	info->nub = 1;
}

void btGearConstraint::getInfo2 (btConstraintInfo2* info)
{
	btVector3 globalAxisA, globalAxisB;

	globalAxisA = m_rbA.getWorldTransform().getBasis()*this->m_axisInA;
	globalAxisB = m_rbB.getWorldTransform().getBasis()*this->m_axisInB;

	info->m_J1angularAxis[0] = globalAxisA[0];
	info->m_J1angularAxis[1] = globalAxisA[1];
	info->m_J1angularAxis[2] = globalAxisA[2];

	info->m_J2angularAxis[0] = m_ratio*globalAxisB[0];
	info->m_J2angularAxis[1] = m_ratio*globalAxisB[1];
	info->m_J2angularAxis[2] = m_ratio*globalAxisB[2];

}

