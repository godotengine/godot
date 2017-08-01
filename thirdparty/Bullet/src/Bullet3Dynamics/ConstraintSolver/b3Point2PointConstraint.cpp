/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#include "b3Point2PointConstraint.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

#include <new>





b3Point2PointConstraint::b3Point2PointConstraint(int rbA,int rbB, const b3Vector3& pivotInA,const b3Vector3& pivotInB)
:b3TypedConstraint(B3_POINT2POINT_CONSTRAINT_TYPE,rbA,rbB),m_pivotInA(pivotInA),m_pivotInB(pivotInB),
m_flags(0)
{

}

/*
b3Point2PointConstraint::b3Point2PointConstraint(int  rbA,const b3Vector3& pivotInA)
:b3TypedConstraint(B3_POINT2POINT_CONSTRAINT_TYPE,rbA),m_pivotInA(pivotInA),m_pivotInB(rbA.getCenterOfMassTransform()(pivotInA)),
m_flags(0),
m_useSolveConstraintObsolete(false)
{
	
}
*/


void b3Point2PointConstraint::getInfo1 (b3ConstraintInfo1* info,const b3RigidBodyData* bodies)
{
	getInfo1NonVirtual(info,bodies);
}

void b3Point2PointConstraint::getInfo1NonVirtual (b3ConstraintInfo1* info,const b3RigidBodyData* bodies)
{
		info->m_numConstraintRows = 3;
		info->nub = 3;
}




void b3Point2PointConstraint::getInfo2 (b3ConstraintInfo2* info, const b3RigidBodyData* bodies)
{
	b3Transform trA;
	trA.setIdentity();
	trA.setOrigin(bodies[m_rbA].m_pos);
	trA.setRotation(bodies[m_rbA].m_quat);

	b3Transform trB;
	trB.setIdentity();
	trB.setOrigin(bodies[m_rbB].m_pos);
	trB.setRotation(bodies[m_rbB].m_quat);

	getInfo2NonVirtual(info, trA,trB);
}

void b3Point2PointConstraint::getInfo2NonVirtual (b3ConstraintInfo2* info, const b3Transform& body0_trans, const b3Transform& body1_trans)
{

	 //retrieve matrices

	// anchor points in global coordinates with respect to body PORs.
   
    // set jacobian
    info->m_J1linearAxis[0] = 1;
	info->m_J1linearAxis[info->rowskip+1] = 1;
	info->m_J1linearAxis[2*info->rowskip+2] = 1;

	b3Vector3 a1 = body0_trans.getBasis()*getPivotInA();
	//b3Vector3 a1a = b3QuatRotate(body0_trans.getRotation(),getPivotInA());

	{
		b3Vector3* angular0 = (b3Vector3*)(info->m_J1angularAxis);
		b3Vector3* angular1 = (b3Vector3*)(info->m_J1angularAxis+info->rowskip);
		b3Vector3* angular2 = (b3Vector3*)(info->m_J1angularAxis+2*info->rowskip);
		b3Vector3 a1neg = -a1;
		a1neg.getSkewSymmetricMatrix(angular0,angular1,angular2);
	}
    
	if (info->m_J2linearAxis)
	{
		info->m_J2linearAxis[0] = -1;
		info->m_J2linearAxis[info->rowskip+1] = -1;
		info->m_J2linearAxis[2*info->rowskip+2] = -1;
	}
	
	b3Vector3 a2 = body1_trans.getBasis()*getPivotInB();
   
	{
	//	b3Vector3 a2n = -a2;
		b3Vector3* angular0 = (b3Vector3*)(info->m_J2angularAxis);
		b3Vector3* angular1 = (b3Vector3*)(info->m_J2angularAxis+info->rowskip);
		b3Vector3* angular2 = (b3Vector3*)(info->m_J2angularAxis+2*info->rowskip);
		a2.getSkewSymmetricMatrix(angular0,angular1,angular2);
	}
    


    // set right hand side
	b3Scalar currERP = (m_flags & B3_P2P_FLAGS_ERP) ? m_erp : info->erp;
    b3Scalar k = info->fps * currERP;
    int j;
	for (j=0; j<3; j++)
    {
        info->m_constraintError[j*info->rowskip] = k * (a2[j] + body1_trans.getOrigin()[j] - a1[j] - body0_trans.getOrigin()[j]);
		//printf("info->m_constraintError[%d]=%f\n",j,info->m_constraintError[j]);
    }
	if(m_flags & B3_P2P_FLAGS_CFM)
	{
		for (j=0; j<3; j++)
		{
			info->cfm[j*info->rowskip] = m_cfm;
		}
	}

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
	
}



void	b3Point2PointConstraint::updateRHS(b3Scalar	timeStep)
{
	(void)timeStep;

}

///override the default global value of a parameter (such as ERP or CFM), optionally provide the axis (0..5). 
///If no axis is provided, it uses the default axis for this constraint.
void b3Point2PointConstraint::setParam(int num, b3Scalar value, int axis)
{
	if(axis != -1)
	{
		b3AssertConstrParams(0);
	}
	else
	{
		switch(num)
		{
			case B3_CONSTRAINT_ERP :
			case B3_CONSTRAINT_STOP_ERP :
				m_erp = value; 
				m_flags |= B3_P2P_FLAGS_ERP;
				break;
			case B3_CONSTRAINT_CFM :
			case B3_CONSTRAINT_STOP_CFM :
				m_cfm = value; 
				m_flags |= B3_P2P_FLAGS_CFM;
				break;
			default: 
				b3AssertConstrParams(0);
		}
	}
}

///return the local value of parameter
b3Scalar b3Point2PointConstraint::getParam(int num, int axis) const 
{
	b3Scalar retVal(B3_INFINITY);
	if(axis != -1)
	{
		b3AssertConstrParams(0);
	}
	else
	{
		switch(num)
		{
			case B3_CONSTRAINT_ERP :
			case B3_CONSTRAINT_STOP_ERP :
				b3AssertConstrParams(m_flags & B3_P2P_FLAGS_ERP);
				retVal = m_erp; 
				break;
			case B3_CONSTRAINT_CFM :
			case B3_CONSTRAINT_STOP_CFM :
				b3AssertConstrParams(m_flags & B3_P2P_FLAGS_CFM);
				retVal = m_cfm; 
				break;
			default: 
				b3AssertConstrParams(0);
		}
	}
	return retVal;
}
	
