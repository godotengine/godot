/*
Copyright (c) 2012 Advanced Micro Devices, Inc.  

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

#include "b3GpuGenericConstraint.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

#include <new>
#include "Bullet3Common/b3Transform.h"

void b3GpuGenericConstraint::getInfo1(unsigned int* info, const b3RigidBodyData* bodies)
{
	switch (m_constraintType)
	{
		case B3_GPU_POINT2POINT_CONSTRAINT_TYPE:
		{
			*info = 3;
			break;
		};
		default:
		{
			b3Assert(0);
		}
	};
}

void getInfo2Point2Point(b3GpuGenericConstraint* constraint, b3GpuConstraintInfo2* info, const b3RigidBodyData* bodies)
{
	b3Transform trA;
	trA.setIdentity();
	trA.setOrigin(bodies[constraint->m_rbA].m_pos);
	trA.setRotation(bodies[constraint->m_rbA].m_quat);

	b3Transform trB;
	trB.setIdentity();
	trB.setOrigin(bodies[constraint->m_rbB].m_pos);
	trB.setRotation(bodies[constraint->m_rbB].m_quat);

	// anchor points in global coordinates with respect to body PORs.

	// set jacobian
	info->m_J1linearAxis[0] = 1;
	info->m_J1linearAxis[info->rowskip + 1] = 1;
	info->m_J1linearAxis[2 * info->rowskip + 2] = 1;

	b3Vector3 a1 = trA.getBasis() * constraint->getPivotInA();
	//b3Vector3 a1a = b3QuatRotate(trA.getRotation(),constraint->getPivotInA());

	{
		b3Vector3* angular0 = (b3Vector3*)(info->m_J1angularAxis);
		b3Vector3* angular1 = (b3Vector3*)(info->m_J1angularAxis + info->rowskip);
		b3Vector3* angular2 = (b3Vector3*)(info->m_J1angularAxis + 2 * info->rowskip);
		b3Vector3 a1neg = -a1;
		a1neg.getSkewSymmetricMatrix(angular0, angular1, angular2);
	}

	if (info->m_J2linearAxis)
	{
		info->m_J2linearAxis[0] = -1;
		info->m_J2linearAxis[info->rowskip + 1] = -1;
		info->m_J2linearAxis[2 * info->rowskip + 2] = -1;
	}

	b3Vector3 a2 = trB.getBasis() * constraint->getPivotInB();

	{
		//	b3Vector3 a2n = -a2;
		b3Vector3* angular0 = (b3Vector3*)(info->m_J2angularAxis);
		b3Vector3* angular1 = (b3Vector3*)(info->m_J2angularAxis + info->rowskip);
		b3Vector3* angular2 = (b3Vector3*)(info->m_J2angularAxis + 2 * info->rowskip);
		a2.getSkewSymmetricMatrix(angular0, angular1, angular2);
	}

	// set right hand side
	//	b3Scalar currERP = (m_flags & B3_P2P_FLAGS_ERP) ? m_erp : info->erp;
	b3Scalar currERP = info->erp;

	b3Scalar k = info->fps * currERP;
	int j;
	for (j = 0; j < 3; j++)
	{
		info->m_constraintError[j * info->rowskip] = k * (a2[j] + trB.getOrigin()[j] - a1[j] - trA.getOrigin()[j]);
		//printf("info->m_constraintError[%d]=%f\n",j,info->m_constraintError[j]);
	}
#if 0
	if(m_flags & B3_P2P_FLAGS_CFM)
	{
		for (j=0; j<3; j++)
		{
			info->cfm[j*info->rowskip] = m_cfm;
		}
	}
#endif

#if 0
	b3Scalar impulseClamp = m_setting.m_impulseClamp;//
	for (j=0; j<3; j++)
    {
		if (m_setting.m_impulseClamp > 0)
		{
			info->m_lowerLimit[j*info->rowskip] = -impulseClamp;
			info->m_upperLimit[j*info->rowskip] = impulseClamp;
		}
	}
	info->m_damping = m_setting.m_damping;
#endif
}

void b3GpuGenericConstraint::getInfo2(b3GpuConstraintInfo2* info, const b3RigidBodyData* bodies)
{
	switch (m_constraintType)
	{
		case B3_GPU_POINT2POINT_CONSTRAINT_TYPE:
		{
			getInfo2Point2Point(this, info, bodies);
			break;
		};
		default:
		{
			b3Assert(0);
		}
	};
}
