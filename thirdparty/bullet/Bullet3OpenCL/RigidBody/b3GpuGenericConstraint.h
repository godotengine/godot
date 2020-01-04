/*
Copyright (c) 2013 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Erwin Coumans

#ifndef B3_GPU_GENERIC_CONSTRAINT_H
#define B3_GPU_GENERIC_CONSTRAINT_H

#include "Bullet3Common/b3Quaternion.h"
struct b3RigidBodyData;
enum B3_CONSTRAINT_FLAGS
{
	B3_CONSTRAINT_FLAG_ENABLED = 1,
};

enum b3GpuGenericConstraintType
{
	B3_GPU_POINT2POINT_CONSTRAINT_TYPE = 3,
	B3_GPU_FIXED_CONSTRAINT_TYPE = 4,
	//	B3_HINGE_CONSTRAINT_TYPE,
	//	B3_CONETWIST_CONSTRAINT_TYPE,
	//	B3_D6_CONSTRAINT_TYPE,
	//	B3_SLIDER_CONSTRAINT_TYPE,
	//	B3_CONTACT_CONSTRAINT_TYPE,
	//	B3_D6_SPRING_CONSTRAINT_TYPE,
	//	B3_GEAR_CONSTRAINT_TYPE,

	B3_GPU_MAX_CONSTRAINT_TYPE
};

struct b3GpuConstraintInfo2
{
	// integrator parameters: frames per second (1/stepsize), default error
	// reduction parameter (0..1).
	b3Scalar fps, erp;

	// for the first and second body, pointers to two (linear and angular)
	// n*3 jacobian sub matrices, stored by rows. these matrices will have
	// been initialized to 0 on entry. if the second body is zero then the
	// J2xx pointers may be 0.
	b3Scalar *m_J1linearAxis, *m_J1angularAxis, *m_J2linearAxis, *m_J2angularAxis;

	// elements to jump from one row to the next in J's
	int rowskip;

	// right hand sides of the equation J*v = c + cfm * lambda. cfm is the
	// "constraint force mixing" vector. c is set to zero on entry, cfm is
	// set to a constant value (typically very small or zero) value on entry.
	b3Scalar *m_constraintError, *cfm;

	// lo and hi limits for variables (set to -/+ infinity on entry).
	b3Scalar *m_lowerLimit, *m_upperLimit;

	// findex vector for variables. see the LCP solver interface for a
	// description of what this does. this is set to -1 on entry.
	// note that the returned indexes are relative to the first index of
	// the constraint.
	int* findex;
	// number of solver iterations
	int m_numIterations;

	//damping of the velocity
	b3Scalar m_damping;
};

B3_ATTRIBUTE_ALIGNED16(struct)
b3GpuGenericConstraint
{
	int m_constraintType;
	int m_rbA;
	int m_rbB;
	float m_breakingImpulseThreshold;

	b3Vector3 m_pivotInA;
	b3Vector3 m_pivotInB;
	b3Quaternion m_relTargetAB;

	int m_flags;
	int m_uid;
	int m_padding[2];

	int getRigidBodyA() const
	{
		return m_rbA;
	}
	int getRigidBodyB() const
	{
		return m_rbB;
	}

	const b3Vector3& getPivotInA() const
	{
		return m_pivotInA;
	}

	const b3Vector3& getPivotInB() const
	{
		return m_pivotInB;
	}

	int isEnabled() const
	{
		return m_flags & B3_CONSTRAINT_FLAG_ENABLED;
	}

	float getBreakingImpulseThreshold() const
	{
		return m_breakingImpulseThreshold;
	}

	///internal method used by the constraint solver, don't use them directly
	void getInfo1(unsigned int* info, const b3RigidBodyData* bodies);

	///internal method used by the constraint solver, don't use them directly
	void getInfo2(b3GpuConstraintInfo2 * info, const b3RigidBodyData* bodies);
};

#endif  //B3_GPU_GENERIC_CONSTRAINT_H