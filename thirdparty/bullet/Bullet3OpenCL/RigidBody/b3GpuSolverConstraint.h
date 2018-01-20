/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2013 Erwin Coumans http://github.com/erwincoumans/bullet3

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#ifndef B3_GPU_SOLVER_CONSTRAINT_H
#define B3_GPU_SOLVER_CONSTRAINT_H


#include "Bullet3Common/b3Vector3.h"
#include "Bullet3Common/b3Matrix3x3.h"
//#include "b3JacobianEntry.h"
#include "Bullet3Common/b3AlignedObjectArray.h"

//#define NO_FRICTION_TANGENTIALS 1



///1D constraint along a normal axis between bodyA and bodyB. It can be combined to solve contact and friction constraints.
B3_ATTRIBUTE_ALIGNED16 (struct)	b3GpuSolverConstraint
{
	B3_DECLARE_ALIGNED_ALLOCATOR();

	b3Vector3		m_relpos1CrossNormal;
	b3Vector3		m_contactNormal;

	b3Vector3		m_relpos2CrossNormal;
	//b3Vector3		m_contactNormal2;//usually m_contactNormal2 == -m_contactNormal

	b3Vector3		m_angularComponentA;
	b3Vector3		m_angularComponentB;
	
	mutable b3Scalar	m_appliedPushImpulse;
	mutable b3Scalar	m_appliedImpulse;
	int m_padding1;
	int m_padding2;
	b3Scalar	m_friction;
	b3Scalar	m_jacDiagABInv;
	b3Scalar		m_rhs;
	b3Scalar		m_cfm;
	
    b3Scalar		m_lowerLimit;
	b3Scalar		m_upperLimit;
	b3Scalar		m_rhsPenetration;
    union
	{
		void*		m_originalContactPoint;
		int		m_originalConstraintIndex;
		b3Scalar	m_unusedPadding4;
	};

	int	m_overrideNumSolverIterations;
    int			m_frictionIndex;
	int m_solverBodyIdA;
	int m_solverBodyIdB;

    
	enum		b3SolverConstraintType
	{
		B3_SOLVER_CONTACT_1D = 0,
		B3_SOLVER_FRICTION_1D
	};
};

typedef b3AlignedObjectArray<b3GpuSolverConstraint>	b3GpuConstraintArray;


#endif //B3_GPU_SOLVER_CONSTRAINT_H



