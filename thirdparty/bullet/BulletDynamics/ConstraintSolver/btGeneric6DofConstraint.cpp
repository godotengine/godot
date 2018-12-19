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
/*
2007-09-09
Refactored by Francisco Le?n
email: projectileman@yahoo.com
http://gimpact.sf.net
*/

#include "btGeneric6DofConstraint.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"
#include "LinearMath/btTransformUtil.h"
#include "LinearMath/btTransformUtil.h"
#include <new>



#define D6_USE_OBSOLETE_METHOD false
#define D6_USE_FRAME_OFFSET true






btGeneric6DofConstraint::btGeneric6DofConstraint(btRigidBody& rbA, btRigidBody& rbB, const btTransform& frameInA, const btTransform& frameInB, bool useLinearReferenceFrameA)
: btTypedConstraint(D6_CONSTRAINT_TYPE, rbA, rbB)
, m_frameInA(frameInA)
, m_frameInB(frameInB),
m_useLinearReferenceFrameA(useLinearReferenceFrameA),
m_useOffsetForConstraintFrame(D6_USE_FRAME_OFFSET),
m_flags(0),
m_useSolveConstraintObsolete(D6_USE_OBSOLETE_METHOD)
{
	calculateTransforms();
}



btGeneric6DofConstraint::btGeneric6DofConstraint(btRigidBody& rbB, const btTransform& frameInB, bool useLinearReferenceFrameB)
        : btTypedConstraint(D6_CONSTRAINT_TYPE, getFixedBody(), rbB),
		m_frameInB(frameInB),
		m_useLinearReferenceFrameA(useLinearReferenceFrameB),
		m_useOffsetForConstraintFrame(D6_USE_FRAME_OFFSET),
		m_flags(0),
		m_useSolveConstraintObsolete(false)
{
	///not providing rigidbody A means implicitly using worldspace for body A
	m_frameInA = rbB.getCenterOfMassTransform() * m_frameInB;
	calculateTransforms();
}




#define GENERIC_D6_DISABLE_WARMSTARTING 1



btScalar btGetMatrixElem(const btMatrix3x3& mat, int index);
btScalar btGetMatrixElem(const btMatrix3x3& mat, int index)
{
	int i = index%3;
	int j = index/3;
	return mat[i][j];
}



///MatrixToEulerXYZ from http://www.geometrictools.com/LibFoundation/Mathematics/Wm4Matrix3.inl.html
bool	matrixToEulerXYZ(const btMatrix3x3& mat,btVector3& xyz);
bool	matrixToEulerXYZ(const btMatrix3x3& mat,btVector3& xyz)
{
	//	// rot =  cy*cz          -cy*sz           sy
	//	//        cz*sx*sy+cx*sz  cx*cz-sx*sy*sz -cy*sx
	//	//       -cx*cz*sy+sx*sz  cz*sx+cx*sy*sz  cx*cy
	//

	btScalar fi = btGetMatrixElem(mat,2);
	if (fi < btScalar(1.0f))
	{
		if (fi > btScalar(-1.0f))
		{
			xyz[0] = btAtan2(-btGetMatrixElem(mat,5),btGetMatrixElem(mat,8));
			xyz[1] = btAsin(btGetMatrixElem(mat,2));
			xyz[2] = btAtan2(-btGetMatrixElem(mat,1),btGetMatrixElem(mat,0));
			return true;
		}
		else
		{
			// WARNING.  Not unique.  XA - ZA = -atan2(r10,r11)
			xyz[0] = -btAtan2(btGetMatrixElem(mat,3),btGetMatrixElem(mat,4));
			xyz[1] = -SIMD_HALF_PI;
			xyz[2] = btScalar(0.0);
			return false;
		}
	}
	else
	{
		// WARNING.  Not unique.  XAngle + ZAngle = atan2(r10,r11)
		xyz[0] = btAtan2(btGetMatrixElem(mat,3),btGetMatrixElem(mat,4));
		xyz[1] = SIMD_HALF_PI;
		xyz[2] = 0.0;
	}
	return false;
}

//////////////////////////// btRotationalLimitMotor ////////////////////////////////////

int btRotationalLimitMotor::testLimitValue(btScalar test_value)
{
	if(m_loLimit>m_hiLimit)
	{
		m_currentLimit = 0;//Free from violation
		return 0;
	}
	if (test_value < m_loLimit)
	{
		m_currentLimit = 1;//low limit violation
		m_currentLimitError =  test_value - m_loLimit;
		if(m_currentLimitError>SIMD_PI) 
			m_currentLimitError-=SIMD_2_PI;
		else if(m_currentLimitError<-SIMD_PI) 
			m_currentLimitError+=SIMD_2_PI;
		return 1;
	}
	else if (test_value> m_hiLimit)
	{
		m_currentLimit = 2;//High limit violation
		m_currentLimitError = test_value - m_hiLimit;
		if(m_currentLimitError>SIMD_PI) 
			m_currentLimitError-=SIMD_2_PI;
		else if(m_currentLimitError<-SIMD_PI) 
			m_currentLimitError+=SIMD_2_PI;
		return 2;
	};

	m_currentLimit = 0;//Free from violation
	return 0;

}



btScalar btRotationalLimitMotor::solveAngularLimits(
	btScalar timeStep,btVector3& axis,btScalar jacDiagABInv,
	btRigidBody * body0, btRigidBody * body1 )
{
	if (needApplyTorques()==false) return 0.0f;

	btScalar target_velocity = m_targetVelocity;
	btScalar maxMotorForce = m_maxMotorForce;

	//current error correction
	if (m_currentLimit!=0)
	{
		target_velocity = -m_stopERP*m_currentLimitError/(timeStep);
		maxMotorForce = m_maxLimitForce;
	}

	maxMotorForce *= timeStep;

	// current velocity difference

	btVector3 angVelA = body0->getAngularVelocity();
	btVector3 angVelB = body1->getAngularVelocity();

	btVector3 vel_diff;
	vel_diff = angVelA-angVelB;



	btScalar rel_vel = axis.dot(vel_diff);

	// correction velocity
	btScalar motor_relvel = m_limitSoftness*(target_velocity  - m_damping*rel_vel);


	if ( motor_relvel < SIMD_EPSILON && motor_relvel > -SIMD_EPSILON  )
	{
		return 0.0f;//no need for applying force
	}


	// correction impulse
	btScalar unclippedMotorImpulse = (1+m_bounce)*motor_relvel*jacDiagABInv;

	// clip correction impulse
	btScalar clippedMotorImpulse;

	///@todo: should clip against accumulated impulse
	if (unclippedMotorImpulse>0.0f)
	{
		clippedMotorImpulse =  unclippedMotorImpulse > maxMotorForce? maxMotorForce: unclippedMotorImpulse;
	}
	else
	{
		clippedMotorImpulse =  unclippedMotorImpulse < -maxMotorForce ? -maxMotorForce: unclippedMotorImpulse;
	}


	// sort with accumulated impulses
	btScalar	lo = btScalar(-BT_LARGE_FLOAT);
	btScalar	hi = btScalar(BT_LARGE_FLOAT);

	btScalar oldaccumImpulse = m_accumulatedImpulse;
	btScalar sum = oldaccumImpulse + clippedMotorImpulse;
	m_accumulatedImpulse = sum > hi ? btScalar(0.) : sum < lo ? btScalar(0.) : sum;

	clippedMotorImpulse = m_accumulatedImpulse - oldaccumImpulse;

	btVector3 motorImp = clippedMotorImpulse * axis;

	body0->applyTorqueImpulse(motorImp);
	body1->applyTorqueImpulse(-motorImp);

	return clippedMotorImpulse;


}

//////////////////////////// End btRotationalLimitMotor ////////////////////////////////////




//////////////////////////// btTranslationalLimitMotor ////////////////////////////////////


int btTranslationalLimitMotor::testLimitValue(int limitIndex, btScalar test_value)
{
	btScalar loLimit = m_lowerLimit[limitIndex];
	btScalar hiLimit = m_upperLimit[limitIndex];
	if(loLimit > hiLimit)
	{
		m_currentLimit[limitIndex] = 0;//Free from violation
		m_currentLimitError[limitIndex] = btScalar(0.f);
		return 0;
	}

	if (test_value < loLimit)
	{
		m_currentLimit[limitIndex] = 2;//low limit violation
		m_currentLimitError[limitIndex] =  test_value - loLimit;
		return 2;
	}
	else if (test_value> hiLimit)
	{
		m_currentLimit[limitIndex] = 1;//High limit violation
		m_currentLimitError[limitIndex] = test_value - hiLimit;
		return 1;
	};

	m_currentLimit[limitIndex] = 0;//Free from violation
	m_currentLimitError[limitIndex] = btScalar(0.f);
	return 0;
}



btScalar btTranslationalLimitMotor::solveLinearAxis(
	btScalar timeStep,
	btScalar jacDiagABInv,
	btRigidBody& body1,const btVector3 &pointInA,
	btRigidBody& body2,const btVector3 &pointInB,
	int limit_index,
	const btVector3 & axis_normal_on_a,
	const btVector3 & anchorPos)
{

	///find relative velocity
	//    btVector3 rel_pos1 = pointInA - body1.getCenterOfMassPosition();
	//    btVector3 rel_pos2 = pointInB - body2.getCenterOfMassPosition();
	btVector3 rel_pos1 = anchorPos - body1.getCenterOfMassPosition();
	btVector3 rel_pos2 = anchorPos - body2.getCenterOfMassPosition();

	btVector3 vel1 = body1.getVelocityInLocalPoint(rel_pos1);
	btVector3 vel2 = body2.getVelocityInLocalPoint(rel_pos2);
	btVector3 vel = vel1 - vel2;

	btScalar rel_vel = axis_normal_on_a.dot(vel);



	/// apply displacement correction

	//positional error (zeroth order error)
	btScalar depth = -(pointInA - pointInB).dot(axis_normal_on_a);
	btScalar	lo = btScalar(-BT_LARGE_FLOAT);
	btScalar	hi = btScalar(BT_LARGE_FLOAT);

	btScalar minLimit = m_lowerLimit[limit_index];
	btScalar maxLimit = m_upperLimit[limit_index];

	//handle the limits
	if (minLimit < maxLimit)
	{
		{
			if (depth > maxLimit)
			{
				depth -= maxLimit;
				lo = btScalar(0.);

			}
			else
			{
				if (depth < minLimit)
				{
					depth -= minLimit;
					hi = btScalar(0.);
				}
				else
				{
					return 0.0f;
				}
			}
		}
	}

	btScalar normalImpulse= m_limitSoftness*(m_restitution*depth/timeStep - m_damping*rel_vel) * jacDiagABInv;




	btScalar oldNormalImpulse = m_accumulatedImpulse[limit_index];
	btScalar sum = oldNormalImpulse + normalImpulse;
	m_accumulatedImpulse[limit_index] = sum > hi ? btScalar(0.) : sum < lo ? btScalar(0.) : sum;
	normalImpulse = m_accumulatedImpulse[limit_index] - oldNormalImpulse;

	btVector3 impulse_vector = axis_normal_on_a * normalImpulse;
	body1.applyImpulse( impulse_vector, rel_pos1);
	body2.applyImpulse(-impulse_vector, rel_pos2);

	

	return normalImpulse;
}

//////////////////////////// btTranslationalLimitMotor ////////////////////////////////////

void btGeneric6DofConstraint::calculateAngleInfo()
{
	btMatrix3x3 relative_frame = m_calculatedTransformA.getBasis().inverse()*m_calculatedTransformB.getBasis();
	matrixToEulerXYZ(relative_frame,m_calculatedAxisAngleDiff);
	// in euler angle mode we do not actually constrain the angular velocity
	// along the axes axis[0] and axis[2] (although we do use axis[1]) :
	//
	//    to get			constrain w2-w1 along		...not
	//    ------			---------------------		------
	//    d(angle[0])/dt = 0	ax[1] x ax[2]			ax[0]
	//    d(angle[1])/dt = 0	ax[1]
	//    d(angle[2])/dt = 0	ax[0] x ax[1]			ax[2]
	//
	// constraining w2-w1 along an axis 'a' means that a'*(w2-w1)=0.
	// to prove the result for angle[0], write the expression for angle[0] from
	// GetInfo1 then take the derivative. to prove this for angle[2] it is
	// easier to take the euler rate expression for d(angle[2])/dt with respect
	// to the components of w and set that to 0.
	btVector3 axis0 = m_calculatedTransformB.getBasis().getColumn(0);
	btVector3 axis2 = m_calculatedTransformA.getBasis().getColumn(2);

	m_calculatedAxis[1] = axis2.cross(axis0);
	m_calculatedAxis[0] = m_calculatedAxis[1].cross(axis2);
	m_calculatedAxis[2] = axis0.cross(m_calculatedAxis[1]);

	m_calculatedAxis[0].normalize();
	m_calculatedAxis[1].normalize();
	m_calculatedAxis[2].normalize();

}

void btGeneric6DofConstraint::calculateTransforms()
{
	calculateTransforms(m_rbA.getCenterOfMassTransform(),m_rbB.getCenterOfMassTransform());
}

void btGeneric6DofConstraint::calculateTransforms(const btTransform& transA,const btTransform& transB)
{
	m_calculatedTransformA = transA * m_frameInA;
	m_calculatedTransformB = transB * m_frameInB;
	calculateLinearInfo();
	calculateAngleInfo();
	if(m_useOffsetForConstraintFrame)
	{	//  get weight factors depending on masses
		btScalar miA = getRigidBodyA().getInvMass();
		btScalar miB = getRigidBodyB().getInvMass();
		m_hasStaticBody = (miA < SIMD_EPSILON) || (miB < SIMD_EPSILON);
		btScalar miS = miA + miB;
		if(miS > btScalar(0.f))
		{
			m_factA = miB / miS;
		}
		else 
		{
			m_factA = btScalar(0.5f);
		}
		m_factB = btScalar(1.0f) - m_factA;
	}
}



void btGeneric6DofConstraint::buildLinearJacobian(
	btJacobianEntry & jacLinear,const btVector3 & normalWorld,
	const btVector3 & pivotAInW,const btVector3 & pivotBInW)
{
	new (&jacLinear) btJacobianEntry(
        m_rbA.getCenterOfMassTransform().getBasis().transpose(),
        m_rbB.getCenterOfMassTransform().getBasis().transpose(),
        pivotAInW - m_rbA.getCenterOfMassPosition(),
        pivotBInW - m_rbB.getCenterOfMassPosition(),
        normalWorld,
        m_rbA.getInvInertiaDiagLocal(),
        m_rbA.getInvMass(),
        m_rbB.getInvInertiaDiagLocal(),
        m_rbB.getInvMass());
}



void btGeneric6DofConstraint::buildAngularJacobian(
	btJacobianEntry & jacAngular,const btVector3 & jointAxisW)
{
	 new (&jacAngular)	btJacobianEntry(jointAxisW,
                                      m_rbA.getCenterOfMassTransform().getBasis().transpose(),
                                      m_rbB.getCenterOfMassTransform().getBasis().transpose(),
                                      m_rbA.getInvInertiaDiagLocal(),
                                      m_rbB.getInvInertiaDiagLocal());

}



bool btGeneric6DofConstraint::testAngularLimitMotor(int axis_index)
{
	btScalar angle = m_calculatedAxisAngleDiff[axis_index];
	angle = btAdjustAngleToLimits(angle, m_angularLimits[axis_index].m_loLimit, m_angularLimits[axis_index].m_hiLimit);
	m_angularLimits[axis_index].m_currentPosition = angle;
	//test limits
	m_angularLimits[axis_index].testLimitValue(angle);
	return m_angularLimits[axis_index].needApplyTorques();
}



void btGeneric6DofConstraint::buildJacobian()
{
#ifndef __SPU__
	if (m_useSolveConstraintObsolete)
	{

		// Clear accumulated impulses for the next simulation step
		m_linearLimits.m_accumulatedImpulse.setValue(btScalar(0.), btScalar(0.), btScalar(0.));
		int i;
		for(i = 0; i < 3; i++)
		{
			m_angularLimits[i].m_accumulatedImpulse = btScalar(0.);
		}
		//calculates transform
		calculateTransforms(m_rbA.getCenterOfMassTransform(),m_rbB.getCenterOfMassTransform());

		//  const btVector3& pivotAInW = m_calculatedTransformA.getOrigin();
		//  const btVector3& pivotBInW = m_calculatedTransformB.getOrigin();
		calcAnchorPos();
		btVector3 pivotAInW = m_AnchorPos;
		btVector3 pivotBInW = m_AnchorPos;

		// not used here
		//    btVector3 rel_pos1 = pivotAInW - m_rbA.getCenterOfMassPosition();
		//    btVector3 rel_pos2 = pivotBInW - m_rbB.getCenterOfMassPosition();

		btVector3 normalWorld;
		//linear part
		for (i=0;i<3;i++)
		{
			if (m_linearLimits.isLimited(i))
			{
				if (m_useLinearReferenceFrameA)
					normalWorld = m_calculatedTransformA.getBasis().getColumn(i);
				else
					normalWorld = m_calculatedTransformB.getBasis().getColumn(i);

				buildLinearJacobian(
					m_jacLinear[i],normalWorld ,
					pivotAInW,pivotBInW);

			}
		}

		// angular part
		for (i=0;i<3;i++)
		{
			//calculates error angle
			if (testAngularLimitMotor(i))
			{
				normalWorld = this->getAxis(i);
				// Create angular atom
				buildAngularJacobian(m_jacAng[i],normalWorld);
			}
		}

	}
#endif //__SPU__

}


void btGeneric6DofConstraint::getInfo1 (btConstraintInfo1* info)
{
	if (m_useSolveConstraintObsolete)
	{
		info->m_numConstraintRows = 0;
		info->nub = 0;
	} else
	{
		//prepare constraint
		calculateTransforms(m_rbA.getCenterOfMassTransform(),m_rbB.getCenterOfMassTransform());
		info->m_numConstraintRows = 0;
		info->nub = 6;
		int i;
		//test linear limits
		for(i = 0; i < 3; i++)
		{
			if(m_linearLimits.needApplyForce(i))
			{
				info->m_numConstraintRows++;
				info->nub--;
			}
		}
		//test angular limits
		for (i=0;i<3 ;i++ )
		{
			if(testAngularLimitMotor(i))
			{
				info->m_numConstraintRows++;
				info->nub--;
			}
		}
	}
}

void btGeneric6DofConstraint::getInfo1NonVirtual (btConstraintInfo1* info)
{
	if (m_useSolveConstraintObsolete)
	{
		info->m_numConstraintRows = 0;
		info->nub = 0;
	} else
	{
		//pre-allocate all 6
		info->m_numConstraintRows = 6;
		info->nub = 0;
	}
}


void btGeneric6DofConstraint::getInfo2 (btConstraintInfo2* info)
{
	btAssert(!m_useSolveConstraintObsolete);

	const btTransform& transA = m_rbA.getCenterOfMassTransform();
	const btTransform& transB = m_rbB.getCenterOfMassTransform();
	const btVector3& linVelA = m_rbA.getLinearVelocity();
	const btVector3& linVelB = m_rbB.getLinearVelocity();
	const btVector3& angVelA = m_rbA.getAngularVelocity();
	const btVector3& angVelB = m_rbB.getAngularVelocity();

	if(m_useOffsetForConstraintFrame)
	{ // for stability better to solve angular limits first
		int row = setAngularLimits(info, 0,transA,transB,linVelA,linVelB,angVelA,angVelB);
		setLinearLimits(info, row, transA,transB,linVelA,linVelB,angVelA,angVelB);
	}
	else
	{ // leave old version for compatibility
		int row = setLinearLimits(info, 0, transA,transB,linVelA,linVelB,angVelA,angVelB);
		setAngularLimits(info, row,transA,transB,linVelA,linVelB,angVelA,angVelB);
	}

}


void btGeneric6DofConstraint::getInfo2NonVirtual (btConstraintInfo2* info, const btTransform& transA,const btTransform& transB,const btVector3& linVelA,const btVector3& linVelB,const btVector3& angVelA,const btVector3& angVelB)
{
	
	btAssert(!m_useSolveConstraintObsolete);
	//prepare constraint
	calculateTransforms(transA,transB);

	int i;
	for (i=0;i<3 ;i++ )
	{
		testAngularLimitMotor(i);
	}

	if(m_useOffsetForConstraintFrame)
	{ // for stability better to solve angular limits first
		int row = setAngularLimits(info, 0,transA,transB,linVelA,linVelB,angVelA,angVelB);
		setLinearLimits(info, row, transA,transB,linVelA,linVelB,angVelA,angVelB);
	}
	else
	{ // leave old version for compatibility
		int row = setLinearLimits(info, 0, transA,transB,linVelA,linVelB,angVelA,angVelB);
		setAngularLimits(info, row,transA,transB,linVelA,linVelB,angVelA,angVelB);
	}
}



int btGeneric6DofConstraint::setLinearLimits(btConstraintInfo2* info, int row, const btTransform& transA,const btTransform& transB,const btVector3& linVelA,const btVector3& linVelB,const btVector3& angVelA,const btVector3& angVelB)
{
//	int row = 0;
	//solve linear limits
	btRotationalLimitMotor limot;
	for (int i=0;i<3 ;i++ )
	{
		if(m_linearLimits.needApplyForce(i))
		{ // re-use rotational motor code
			limot.m_bounce = btScalar(0.f);
			limot.m_currentLimit = m_linearLimits.m_currentLimit[i];
			limot.m_currentPosition = m_linearLimits.m_currentLinearDiff[i];
			limot.m_currentLimitError  = m_linearLimits.m_currentLimitError[i];
			limot.m_damping  = m_linearLimits.m_damping;
			limot.m_enableMotor  = m_linearLimits.m_enableMotor[i];
			limot.m_hiLimit  = m_linearLimits.m_upperLimit[i];
			limot.m_limitSoftness  = m_linearLimits.m_limitSoftness;
			limot.m_loLimit  = m_linearLimits.m_lowerLimit[i];
			limot.m_maxLimitForce  = btScalar(0.f);
			limot.m_maxMotorForce  = m_linearLimits.m_maxMotorForce[i];
			limot.m_targetVelocity  = m_linearLimits.m_targetVelocity[i];
			btVector3 axis = m_calculatedTransformA.getBasis().getColumn(i);
			int flags = m_flags >> (i * BT_6DOF_FLAGS_AXIS_SHIFT);
			limot.m_normalCFM	= (flags & BT_6DOF_FLAGS_CFM_NORM) ? m_linearLimits.m_normalCFM[i] : info->cfm[0];
			limot.m_stopCFM		= (flags & BT_6DOF_FLAGS_CFM_STOP) ? m_linearLimits.m_stopCFM[i] : info->cfm[0];
			limot.m_stopERP		= (flags & BT_6DOF_FLAGS_ERP_STOP) ? m_linearLimits.m_stopERP[i] : info->erp;
			if(m_useOffsetForConstraintFrame)
			{
				int indx1 = (i + 1) % 3;
				int indx2 = (i + 2) % 3;
				int rotAllowed = 1; // rotations around orthos to current axis
				if(m_angularLimits[indx1].m_currentLimit && m_angularLimits[indx2].m_currentLimit)
				{
					rotAllowed = 0;
				}
				row += get_limit_motor_info2(&limot, transA,transB,linVelA,linVelB,angVelA,angVelB, info, row, axis, 0, rotAllowed);
			}
			else
			{
				row += get_limit_motor_info2(&limot, transA,transB,linVelA,linVelB,angVelA,angVelB, info, row, axis, 0);
			}
		}
	}
	return row;
}



int btGeneric6DofConstraint::setAngularLimits(btConstraintInfo2 *info, int row_offset, const btTransform& transA,const btTransform& transB,const btVector3& linVelA,const btVector3& linVelB,const btVector3& angVelA,const btVector3& angVelB)
{
	btGeneric6DofConstraint * d6constraint = this;
	int row = row_offset;
	//solve angular limits
	for (int i=0;i<3 ;i++ )
	{
		if(d6constraint->getRotationalLimitMotor(i)->needApplyTorques())
		{
			btVector3 axis = d6constraint->getAxis(i);
			int flags = m_flags >> ((i + 3) * BT_6DOF_FLAGS_AXIS_SHIFT);
			if(!(flags & BT_6DOF_FLAGS_CFM_NORM))
			{
				m_angularLimits[i].m_normalCFM = info->cfm[0];
			}
			if(!(flags & BT_6DOF_FLAGS_CFM_STOP))
			{
				m_angularLimits[i].m_stopCFM = info->cfm[0];
			}
			if(!(flags & BT_6DOF_FLAGS_ERP_STOP))
			{
				m_angularLimits[i].m_stopERP = info->erp;
			}
			row += get_limit_motor_info2(d6constraint->getRotationalLimitMotor(i),
												transA,transB,linVelA,linVelB,angVelA,angVelB, info,row,axis,1);
		}
	}

	return row;
}




void	btGeneric6DofConstraint::updateRHS(btScalar	timeStep)
{
	(void)timeStep;

}


void btGeneric6DofConstraint::setFrames(const btTransform& frameA, const btTransform& frameB)
{
	m_frameInA = frameA;
	m_frameInB = frameB;
	buildJacobian();
	calculateTransforms();
}



btVector3 btGeneric6DofConstraint::getAxis(int axis_index) const
{
	return m_calculatedAxis[axis_index];
}


btScalar	btGeneric6DofConstraint::getRelativePivotPosition(int axisIndex) const
{
	return m_calculatedLinearDiff[axisIndex];
}


btScalar btGeneric6DofConstraint::getAngle(int axisIndex) const
{
	return m_calculatedAxisAngleDiff[axisIndex];
}



void btGeneric6DofConstraint::calcAnchorPos(void)
{
	btScalar imA = m_rbA.getInvMass();
	btScalar imB = m_rbB.getInvMass();
	btScalar weight;
	if(imB == btScalar(0.0))
	{
		weight = btScalar(1.0);
	}
	else
	{
		weight = imA / (imA + imB);
	}
	const btVector3& pA = m_calculatedTransformA.getOrigin();
	const btVector3& pB = m_calculatedTransformB.getOrigin();
	m_AnchorPos = pA * weight + pB * (btScalar(1.0) - weight);
	return;
}



void btGeneric6DofConstraint::calculateLinearInfo()
{
	m_calculatedLinearDiff = m_calculatedTransformB.getOrigin() - m_calculatedTransformA.getOrigin();
	m_calculatedLinearDiff = m_calculatedTransformA.getBasis().inverse() * m_calculatedLinearDiff;
	for(int i = 0; i < 3; i++)
	{
		m_linearLimits.m_currentLinearDiff[i] = m_calculatedLinearDiff[i];
		m_linearLimits.testLimitValue(i, m_calculatedLinearDiff[i]);
	}
}



int btGeneric6DofConstraint::get_limit_motor_info2(
	btRotationalLimitMotor * limot,
	const btTransform& transA,const btTransform& transB,const btVector3& linVelA,const btVector3& linVelB,const btVector3& angVelA,const btVector3& angVelB,
	btConstraintInfo2 *info, int row, btVector3& ax1, int rotational,int rotAllowed)
{
    int srow = row * info->rowskip;
    bool powered = limot->m_enableMotor;
    int limit = limot->m_currentLimit;
    if (powered || limit)
    {   // if the joint is powered, or has joint limits, add in the extra row
        btScalar *J1 = rotational ? info->m_J1angularAxis : info->m_J1linearAxis;
        btScalar *J2 = rotational ? info->m_J2angularAxis : info->m_J2linearAxis;
        J1[srow+0] = ax1[0];
        J1[srow+1] = ax1[1];
        J1[srow+2] = ax1[2];

        J2[srow+0] = -ax1[0];
        J2[srow+1] = -ax1[1];
        J2[srow+2] = -ax1[2];

		if((!rotational))
        {
			if (m_useOffsetForConstraintFrame)
			{
				btVector3 tmpA, tmpB, relA, relB;
				// get vector from bodyB to frameB in WCS
				relB = m_calculatedTransformB.getOrigin() - transB.getOrigin();
				// get its projection to constraint axis
				btVector3 projB = ax1 * relB.dot(ax1);
				// get vector directed from bodyB to constraint axis (and orthogonal to it)
				btVector3 orthoB = relB - projB;
				// same for bodyA
				relA = m_calculatedTransformA.getOrigin() - transA.getOrigin();
				btVector3 projA = ax1 * relA.dot(ax1);
				btVector3 orthoA = relA - projA;
				// get desired offset between frames A and B along constraint axis
				btScalar desiredOffs = limot->m_currentPosition - limot->m_currentLimitError;
				// desired vector from projection of center of bodyA to projection of center of bodyB to constraint axis
				btVector3 totalDist = projA + ax1 * desiredOffs - projB;
				// get offset vectors relA and relB
				relA = orthoA + totalDist * m_factA;
				relB = orthoB - totalDist * m_factB;
				tmpA = relA.cross(ax1);
				tmpB = relB.cross(ax1);
				if(m_hasStaticBody && (!rotAllowed))
				{
					tmpA *= m_factA;
					tmpB *= m_factB;
				}
				int i;
				for (i=0; i<3; i++) info->m_J1angularAxis[srow+i] = tmpA[i];
				for (i=0; i<3; i++) info->m_J2angularAxis[srow+i] = -tmpB[i];
			} else
			{
				btVector3 ltd;	// Linear Torque Decoupling vector
				btVector3 c = m_calculatedTransformB.getOrigin() - transA.getOrigin();
				ltd = c.cross(ax1);
				info->m_J1angularAxis[srow+0] = ltd[0];
				info->m_J1angularAxis[srow+1] = ltd[1];
				info->m_J1angularAxis[srow+2] = ltd[2];

				c = m_calculatedTransformB.getOrigin() - transB.getOrigin();
				ltd = -c.cross(ax1);
				info->m_J2angularAxis[srow+0] = ltd[0];
				info->m_J2angularAxis[srow+1] = ltd[1];
				info->m_J2angularAxis[srow+2] = ltd[2];
			}
        }
        // if we're limited low and high simultaneously, the joint motor is
        // ineffective
        if (limit && (limot->m_loLimit == limot->m_hiLimit)) powered = false;
        info->m_constraintError[srow] = btScalar(0.f);
        if (powered)
        {
			info->cfm[srow] = limot->m_normalCFM;
            if(!limit)
            {
				btScalar tag_vel = rotational ? limot->m_targetVelocity : -limot->m_targetVelocity;

				btScalar mot_fact = getMotorFactor(	limot->m_currentPosition, 
													limot->m_loLimit,
													limot->m_hiLimit, 
													tag_vel, 
													info->fps * limot->m_stopERP);
				info->m_constraintError[srow] += mot_fact * limot->m_targetVelocity;
                info->m_lowerLimit[srow] = -limot->m_maxMotorForce / info->fps;
                info->m_upperLimit[srow] = limot->m_maxMotorForce / info->fps;
            }
        }
        if(limit)
        {
            btScalar k = info->fps * limot->m_stopERP;
			if(!rotational)
			{
				info->m_constraintError[srow] += k * limot->m_currentLimitError;
			}
			else
			{
				info->m_constraintError[srow] += -k * limot->m_currentLimitError;
			}
			info->cfm[srow] = limot->m_stopCFM;
            if (limot->m_loLimit == limot->m_hiLimit)
            {   // limited low and high simultaneously
                info->m_lowerLimit[srow] = -SIMD_INFINITY;
                info->m_upperLimit[srow] = SIMD_INFINITY;
            }
            else
            {
                if (limit == 1)
                {
                    info->m_lowerLimit[srow] = 0;
                    info->m_upperLimit[srow] = SIMD_INFINITY;
                }
                else
                {
                    info->m_lowerLimit[srow] = -SIMD_INFINITY;
                    info->m_upperLimit[srow] = 0;
                }
                // deal with bounce
                if (limot->m_bounce > 0)
                {
                    // calculate joint velocity
                    btScalar vel;
                    if (rotational)
                    {
                        vel = angVelA.dot(ax1);
//make sure that if no body -> angVelB == zero vec
//                        if (body1)
                            vel -= angVelB.dot(ax1);
                    }
                    else
                    {
                        vel = linVelA.dot(ax1);
//make sure that if no body -> angVelB == zero vec
//                        if (body1)
                            vel -= linVelB.dot(ax1);
                    }
                    // only apply bounce if the velocity is incoming, and if the
                    // resulting c[] exceeds what we already have.
                    if (limit == 1)
                    {
                        if (vel < 0)
                        {
                            btScalar newc = -limot->m_bounce* vel;
                            if (newc > info->m_constraintError[srow]) 
								info->m_constraintError[srow] = newc;
                        }
                    }
                    else
                    {
                        if (vel > 0)
                        {
                            btScalar newc = -limot->m_bounce * vel;
                            if (newc < info->m_constraintError[srow]) 
								info->m_constraintError[srow] = newc;
                        }
                    }
                }
            }
        }
        return 1;
    }
    else return 0;
}






	///override the default global value of a parameter (such as ERP or CFM), optionally provide the axis (0..5). 
	///If no axis is provided, it uses the default axis for this constraint.
void btGeneric6DofConstraint::setParam(int num, btScalar value, int axis)
{
	if((axis >= 0) && (axis < 3))
	{
		switch(num)
		{
			case BT_CONSTRAINT_STOP_ERP : 
				m_linearLimits.m_stopERP[axis] = value;
				m_flags |= BT_6DOF_FLAGS_ERP_STOP << (axis * BT_6DOF_FLAGS_AXIS_SHIFT);
				break;
			case BT_CONSTRAINT_STOP_CFM : 
				m_linearLimits.m_stopCFM[axis] = value;
				m_flags |= BT_6DOF_FLAGS_CFM_STOP << (axis * BT_6DOF_FLAGS_AXIS_SHIFT);
				break;
			case BT_CONSTRAINT_CFM : 
				m_linearLimits.m_normalCFM[axis] = value;
				m_flags |= BT_6DOF_FLAGS_CFM_NORM << (axis * BT_6DOF_FLAGS_AXIS_SHIFT);
				break;
			default : 
				btAssertConstrParams(0);
		}
	}
	else if((axis >=3) && (axis < 6))
	{
		switch(num)
		{
			case BT_CONSTRAINT_STOP_ERP : 
				m_angularLimits[axis - 3].m_stopERP = value;
				m_flags |= BT_6DOF_FLAGS_ERP_STOP << (axis * BT_6DOF_FLAGS_AXIS_SHIFT);
				break;
			case BT_CONSTRAINT_STOP_CFM : 
				m_angularLimits[axis - 3].m_stopCFM = value;
				m_flags |= BT_6DOF_FLAGS_CFM_STOP << (axis * BT_6DOF_FLAGS_AXIS_SHIFT);
				break;
			case BT_CONSTRAINT_CFM : 
				m_angularLimits[axis - 3].m_normalCFM = value;
				m_flags |= BT_6DOF_FLAGS_CFM_NORM << (axis * BT_6DOF_FLAGS_AXIS_SHIFT);
				break;
			default : 
				btAssertConstrParams(0);
		}
	}
	else
	{
		btAssertConstrParams(0);
	}
}

	///return the local value of parameter
btScalar btGeneric6DofConstraint::getParam(int num, int axis) const 
{
	btScalar retVal = 0;
	if((axis >= 0) && (axis < 3))
	{
		switch(num)
		{
			case BT_CONSTRAINT_STOP_ERP : 
				btAssertConstrParams(m_flags & (BT_6DOF_FLAGS_ERP_STOP << (axis * BT_6DOF_FLAGS_AXIS_SHIFT)));
				retVal = m_linearLimits.m_stopERP[axis];
				break;
			case BT_CONSTRAINT_STOP_CFM : 
				btAssertConstrParams(m_flags & (BT_6DOF_FLAGS_CFM_STOP << (axis * BT_6DOF_FLAGS_AXIS_SHIFT)));
				retVal = m_linearLimits.m_stopCFM[axis];
				break;
			case BT_CONSTRAINT_CFM : 
				btAssertConstrParams(m_flags & (BT_6DOF_FLAGS_CFM_NORM << (axis * BT_6DOF_FLAGS_AXIS_SHIFT)));
				retVal = m_linearLimits.m_normalCFM[axis];
				break;
			default : 
				btAssertConstrParams(0);
		}
	}
	else if((axis >=3) && (axis < 6))
	{
		switch(num)
		{
			case BT_CONSTRAINT_STOP_ERP : 
				btAssertConstrParams(m_flags & (BT_6DOF_FLAGS_ERP_STOP << (axis * BT_6DOF_FLAGS_AXIS_SHIFT)));
				retVal = m_angularLimits[axis - 3].m_stopERP;
				break;
			case BT_CONSTRAINT_STOP_CFM : 
				btAssertConstrParams(m_flags & (BT_6DOF_FLAGS_CFM_STOP << (axis * BT_6DOF_FLAGS_AXIS_SHIFT)));
				retVal = m_angularLimits[axis - 3].m_stopCFM;
				break;
			case BT_CONSTRAINT_CFM : 
				btAssertConstrParams(m_flags & (BT_6DOF_FLAGS_CFM_NORM << (axis * BT_6DOF_FLAGS_AXIS_SHIFT)));
				retVal = m_angularLimits[axis - 3].m_normalCFM;
				break;
			default : 
				btAssertConstrParams(0);
		}
	}
	else
	{
		btAssertConstrParams(0);
	}
	return retVal;
}

 

void btGeneric6DofConstraint::setAxis(const btVector3& axis1,const btVector3& axis2)
{
	btVector3 zAxis = axis1.normalized();
	btVector3 yAxis = axis2.normalized();
	btVector3 xAxis = yAxis.cross(zAxis); // we want right coordinate system
	
	btTransform frameInW;
	frameInW.setIdentity();
	frameInW.getBasis().setValue(	xAxis[0], yAxis[0], zAxis[0],	
	                                xAxis[1], yAxis[1], zAxis[1],
	                               xAxis[2], yAxis[2], zAxis[2]);
	
	// now get constraint frame in local coordinate systems
	m_frameInA = m_rbA.getCenterOfMassTransform().inverse() * frameInW;
	m_frameInB = m_rbB.getCenterOfMassTransform().inverse() * frameInW;
	
	calculateTransforms();
}
