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

#include "b3Generic6DofConstraint.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

#include "Bullet3Common/b3TransformUtil.h"
#include "Bullet3Common/b3TransformUtil.h"
#include <new>



#define D6_USE_OBSOLETE_METHOD false
#define D6_USE_FRAME_OFFSET true






b3Generic6DofConstraint::b3Generic6DofConstraint(int rbA,int  rbB, const b3Transform& frameInA, const b3Transform& frameInB, bool useLinearReferenceFrameA, const b3RigidBodyData* bodies)
: b3TypedConstraint(B3_D6_CONSTRAINT_TYPE, rbA, rbB)
, m_frameInA(frameInA)
, m_frameInB(frameInB),
m_useLinearReferenceFrameA(useLinearReferenceFrameA),
m_useOffsetForConstraintFrame(D6_USE_FRAME_OFFSET),
m_flags(0)
{
	calculateTransforms(bodies);
}






#define GENERIC_D6_DISABLE_WARMSTARTING 1



b3Scalar btGetMatrixElem(const b3Matrix3x3& mat, int index);
b3Scalar btGetMatrixElem(const b3Matrix3x3& mat, int index)
{
	int i = index%3;
	int j = index/3;
	return mat[i][j];
}



///MatrixToEulerXYZ from http://www.geometrictools.com/LibFoundation/Mathematics/Wm4Matrix3.inl.html
bool	matrixToEulerXYZ(const b3Matrix3x3& mat,b3Vector3& xyz);
bool	matrixToEulerXYZ(const b3Matrix3x3& mat,b3Vector3& xyz)
{
	//	// rot =  cy*cz          -cy*sz           sy
	//	//        cz*sx*sy+cx*sz  cx*cz-sx*sy*sz -cy*sx
	//	//       -cx*cz*sy+sx*sz  cz*sx+cx*sy*sz  cx*cy
	//

	b3Scalar fi = btGetMatrixElem(mat,2);
	if (fi < b3Scalar(1.0f))
	{
		if (fi > b3Scalar(-1.0f))
		{
			xyz[0] = b3Atan2(-btGetMatrixElem(mat,5),btGetMatrixElem(mat,8));
			xyz[1] = b3Asin(btGetMatrixElem(mat,2));
			xyz[2] = b3Atan2(-btGetMatrixElem(mat,1),btGetMatrixElem(mat,0));
			return true;
		}
		else
		{
			// WARNING.  Not unique.  XA - ZA = -atan2(r10,r11)
			xyz[0] = -b3Atan2(btGetMatrixElem(mat,3),btGetMatrixElem(mat,4));
			xyz[1] = -B3_HALF_PI;
			xyz[2] = b3Scalar(0.0);
			return false;
		}
	}
	else
	{
		// WARNING.  Not unique.  XAngle + ZAngle = atan2(r10,r11)
		xyz[0] = b3Atan2(btGetMatrixElem(mat,3),btGetMatrixElem(mat,4));
		xyz[1] = B3_HALF_PI;
		xyz[2] = 0.0;
	}
	return false;
}

//////////////////////////// b3RotationalLimitMotor ////////////////////////////////////

int b3RotationalLimitMotor::testLimitValue(b3Scalar test_value)
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
		if(m_currentLimitError>B3_PI) 
			m_currentLimitError-=B3_2_PI;
		else if(m_currentLimitError<-B3_PI) 
			m_currentLimitError+=B3_2_PI;
		return 1;
	}
	else if (test_value> m_hiLimit)
	{
		m_currentLimit = 2;//High limit violation
		m_currentLimitError = test_value - m_hiLimit;
		if(m_currentLimitError>B3_PI) 
			m_currentLimitError-=B3_2_PI;
		else if(m_currentLimitError<-B3_PI) 
			m_currentLimitError+=B3_2_PI;
		return 2;
	};

	m_currentLimit = 0;//Free from violation
	return 0;

}




//////////////////////////// End b3RotationalLimitMotor ////////////////////////////////////




//////////////////////////// b3TranslationalLimitMotor ////////////////////////////////////


int b3TranslationalLimitMotor::testLimitValue(int limitIndex, b3Scalar test_value)
{
	b3Scalar loLimit = m_lowerLimit[limitIndex];
	b3Scalar hiLimit = m_upperLimit[limitIndex];
	if(loLimit > hiLimit)
	{
		m_currentLimit[limitIndex] = 0;//Free from violation
		m_currentLimitError[limitIndex] = b3Scalar(0.f);
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
	m_currentLimitError[limitIndex] = b3Scalar(0.f);
	return 0;
}



//////////////////////////// b3TranslationalLimitMotor ////////////////////////////////////

void b3Generic6DofConstraint::calculateAngleInfo()
{
	b3Matrix3x3 relative_frame = m_calculatedTransformA.getBasis().inverse()*m_calculatedTransformB.getBasis();
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
	b3Vector3 axis0 = m_calculatedTransformB.getBasis().getColumn(0);
	b3Vector3 axis2 = m_calculatedTransformA.getBasis().getColumn(2);

	m_calculatedAxis[1] = axis2.cross(axis0);
	m_calculatedAxis[0] = m_calculatedAxis[1].cross(axis2);
	m_calculatedAxis[2] = axis0.cross(m_calculatedAxis[1]);

	m_calculatedAxis[0].normalize();
	m_calculatedAxis[1].normalize();
	m_calculatedAxis[2].normalize();

}

static b3Transform getCenterOfMassTransform(const b3RigidBodyData& body)
{
	b3Transform tr(body.m_quat,body.m_pos);
	return tr;
}

void b3Generic6DofConstraint::calculateTransforms(const b3RigidBodyData* bodies)
{
	b3Transform transA;
	b3Transform transB;
	transA = getCenterOfMassTransform(bodies[m_rbA]);
	transB = getCenterOfMassTransform(bodies[m_rbB]);
	calculateTransforms(transA,transB,bodies);
}

void b3Generic6DofConstraint::calculateTransforms(const b3Transform& transA,const b3Transform& transB,const b3RigidBodyData* bodies)
{
	m_calculatedTransformA = transA * m_frameInA;
	m_calculatedTransformB = transB * m_frameInB;
	calculateLinearInfo();
	calculateAngleInfo();
	if(m_useOffsetForConstraintFrame)
	{	//  get weight factors depending on masses
		b3Scalar miA = bodies[m_rbA].m_invMass;
		b3Scalar miB = bodies[m_rbB].m_invMass;
		m_hasStaticBody = (miA < B3_EPSILON) || (miB < B3_EPSILON);
		b3Scalar miS = miA + miB;
		if(miS > b3Scalar(0.f))
		{
			m_factA = miB / miS;
		}
		else 
		{
			m_factA = b3Scalar(0.5f);
		}
		m_factB = b3Scalar(1.0f) - m_factA;
	}
}







bool b3Generic6DofConstraint::testAngularLimitMotor(int axis_index)
{
	b3Scalar angle = m_calculatedAxisAngleDiff[axis_index];
	angle = b3AdjustAngleToLimits(angle, m_angularLimits[axis_index].m_loLimit, m_angularLimits[axis_index].m_hiLimit);
	m_angularLimits[axis_index].m_currentPosition = angle;
	//test limits
	m_angularLimits[axis_index].testLimitValue(angle);
	return m_angularLimits[axis_index].needApplyTorques();
}




void b3Generic6DofConstraint::getInfo1 (b3ConstraintInfo1* info,const b3RigidBodyData* bodies)
{
	//prepare constraint
	calculateTransforms(getCenterOfMassTransform(bodies[m_rbA]),getCenterOfMassTransform(bodies[m_rbB]),bodies);
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
//	printf("info->m_numConstraintRows=%d\n",info->m_numConstraintRows);
}

void b3Generic6DofConstraint::getInfo1NonVirtual (b3ConstraintInfo1* info,const b3RigidBodyData* bodies)
{
	//pre-allocate all 6
	info->m_numConstraintRows = 6;
	info->nub = 0;
}


void b3Generic6DofConstraint::getInfo2 (b3ConstraintInfo2* info,const b3RigidBodyData* bodies)
{

	b3Transform transA = getCenterOfMassTransform(bodies[m_rbA]);
	b3Transform transB = getCenterOfMassTransform(bodies[m_rbB]);
	const b3Vector3& linVelA = bodies[m_rbA].m_linVel;
	const b3Vector3& linVelB = bodies[m_rbB].m_linVel;
	const b3Vector3& angVelA = bodies[m_rbA].m_angVel;
	const b3Vector3& angVelB = bodies[m_rbB].m_angVel;

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


void b3Generic6DofConstraint::getInfo2NonVirtual (b3ConstraintInfo2* info, const b3Transform& transA,const b3Transform& transB,const b3Vector3& linVelA,const b3Vector3& linVelB,const b3Vector3& angVelA,const b3Vector3& angVelB,const b3RigidBodyData* bodies)
{
	
	//prepare constraint
	calculateTransforms(transA,transB,bodies);

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



int b3Generic6DofConstraint::setLinearLimits(b3ConstraintInfo2* info, int row, const b3Transform& transA,const b3Transform& transB,const b3Vector3& linVelA,const b3Vector3& linVelB,const b3Vector3& angVelA,const b3Vector3& angVelB)
{
//	int row = 0;
	//solve linear limits
	b3RotationalLimitMotor limot;
	for (int i=0;i<3 ;i++ )
	{
		if(m_linearLimits.needApplyForce(i))
		{ // re-use rotational motor code
			limot.m_bounce = b3Scalar(0.f);
			limot.m_currentLimit = m_linearLimits.m_currentLimit[i];
			limot.m_currentPosition = m_linearLimits.m_currentLinearDiff[i];
			limot.m_currentLimitError  = m_linearLimits.m_currentLimitError[i];
			limot.m_damping  = m_linearLimits.m_damping;
			limot.m_enableMotor  = m_linearLimits.m_enableMotor[i];
			limot.m_hiLimit  = m_linearLimits.m_upperLimit[i];
			limot.m_limitSoftness  = m_linearLimits.m_limitSoftness;
			limot.m_loLimit  = m_linearLimits.m_lowerLimit[i];
			limot.m_maxLimitForce  = b3Scalar(0.f);
			limot.m_maxMotorForce  = m_linearLimits.m_maxMotorForce[i];
			limot.m_targetVelocity  = m_linearLimits.m_targetVelocity[i];
			b3Vector3 axis = m_calculatedTransformA.getBasis().getColumn(i);
			int flags = m_flags >> (i * B3_6DOF_FLAGS_AXIS_SHIFT);
			limot.m_normalCFM	= (flags & B3_6DOF_FLAGS_CFM_NORM) ? m_linearLimits.m_normalCFM[i] : info->cfm[0];
			limot.m_stopCFM		= (flags & B3_6DOF_FLAGS_CFM_STOP) ? m_linearLimits.m_stopCFM[i] : info->cfm[0];
			limot.m_stopERP		= (flags & B3_6DOF_FLAGS_ERP_STOP) ? m_linearLimits.m_stopERP[i] : info->erp;
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



int b3Generic6DofConstraint::setAngularLimits(b3ConstraintInfo2 *info, int row_offset, const b3Transform& transA,const b3Transform& transB,const b3Vector3& linVelA,const b3Vector3& linVelB,const b3Vector3& angVelA,const b3Vector3& angVelB)
{
	b3Generic6DofConstraint * d6constraint = this;
	int row = row_offset;
	//solve angular limits
	for (int i=0;i<3 ;i++ )
	{
		if(d6constraint->getRotationalLimitMotor(i)->needApplyTorques())
		{
			b3Vector3 axis = d6constraint->getAxis(i);
			int flags = m_flags >> ((i + 3) * B3_6DOF_FLAGS_AXIS_SHIFT);
			if(!(flags & B3_6DOF_FLAGS_CFM_NORM))
			{
				m_angularLimits[i].m_normalCFM = info->cfm[0];
			}
			if(!(flags & B3_6DOF_FLAGS_CFM_STOP))
			{
				m_angularLimits[i].m_stopCFM = info->cfm[0];
			}
			if(!(flags & B3_6DOF_FLAGS_ERP_STOP))
			{
				m_angularLimits[i].m_stopERP = info->erp;
			}
			row += get_limit_motor_info2(d6constraint->getRotationalLimitMotor(i),
												transA,transB,linVelA,linVelB,angVelA,angVelB, info,row,axis,1);
		}
	}

	return row;
}




void	b3Generic6DofConstraint::updateRHS(b3Scalar	timeStep)
{
	(void)timeStep;

}


void b3Generic6DofConstraint::setFrames(const b3Transform& frameA, const b3Transform& frameB,const b3RigidBodyData* bodies)
{
	m_frameInA = frameA;
	m_frameInB = frameB;

	calculateTransforms(bodies);
}



b3Vector3 b3Generic6DofConstraint::getAxis(int axis_index) const
{
	return m_calculatedAxis[axis_index];
}


b3Scalar	b3Generic6DofConstraint::getRelativePivotPosition(int axisIndex) const
{
	return m_calculatedLinearDiff[axisIndex];
}


b3Scalar b3Generic6DofConstraint::getAngle(int axisIndex) const
{
	return m_calculatedAxisAngleDiff[axisIndex];
}



void b3Generic6DofConstraint::calcAnchorPos(const b3RigidBodyData* bodies)
{
	b3Scalar imA = bodies[m_rbA].m_invMass;
	b3Scalar imB = bodies[m_rbB].m_invMass;
	b3Scalar weight;
	if(imB == b3Scalar(0.0))
	{
		weight = b3Scalar(1.0);
	}
	else
	{
		weight = imA / (imA + imB);
	}
	const b3Vector3& pA = m_calculatedTransformA.getOrigin();
	const b3Vector3& pB = m_calculatedTransformB.getOrigin();
	m_AnchorPos = pA * weight + pB * (b3Scalar(1.0) - weight);
	return;
}



void b3Generic6DofConstraint::calculateLinearInfo()
{
	m_calculatedLinearDiff = m_calculatedTransformB.getOrigin() - m_calculatedTransformA.getOrigin();
	m_calculatedLinearDiff = m_calculatedTransformA.getBasis().inverse() * m_calculatedLinearDiff;
	for(int i = 0; i < 3; i++)
	{
		m_linearLimits.m_currentLinearDiff[i] = m_calculatedLinearDiff[i];
		m_linearLimits.testLimitValue(i, m_calculatedLinearDiff[i]);
	}
}



int b3Generic6DofConstraint::get_limit_motor_info2(
	b3RotationalLimitMotor * limot,
	const b3Transform& transA,const b3Transform& transB,const b3Vector3& linVelA,const b3Vector3& linVelB,const b3Vector3& angVelA,const b3Vector3& angVelB,
	b3ConstraintInfo2 *info, int row, b3Vector3& ax1, int rotational,int rotAllowed)
{
    int srow = row * info->rowskip;
    bool powered = limot->m_enableMotor;
    int limit = limot->m_currentLimit;
    if (powered || limit)
    {   // if the joint is powered, or has joint limits, add in the extra row
        b3Scalar *J1 = rotational ? info->m_J1angularAxis : info->m_J1linearAxis;
        b3Scalar *J2 = rotational ? info->m_J2angularAxis : info->m_J2linearAxis;
		if (J1)
		{
			J1[srow+0] = ax1[0];
			J1[srow+1] = ax1[1];
			J1[srow+2] = ax1[2];
		}
		if (J2)
		{
			J2[srow+0] = -ax1[0];
			J2[srow+1] = -ax1[1];
			J2[srow+2] = -ax1[2];
		}
		if((!rotational))
        {
			if (m_useOffsetForConstraintFrame)
			{
				b3Vector3 tmpA, tmpB, relA, relB;
				// get vector from bodyB to frameB in WCS
				relB = m_calculatedTransformB.getOrigin() - transB.getOrigin();
				// get its projection to constraint axis
				b3Vector3 projB = ax1 * relB.dot(ax1);
				// get vector directed from bodyB to constraint axis (and orthogonal to it)
				b3Vector3 orthoB = relB - projB;
				// same for bodyA
				relA = m_calculatedTransformA.getOrigin() - transA.getOrigin();
				b3Vector3 projA = ax1 * relA.dot(ax1);
				b3Vector3 orthoA = relA - projA;
				// get desired offset between frames A and B along constraint axis
				b3Scalar desiredOffs = limot->m_currentPosition - limot->m_currentLimitError;
				// desired vector from projection of center of bodyA to projection of center of bodyB to constraint axis
				b3Vector3 totalDist = projA + ax1 * desiredOffs - projB;
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
				b3Vector3 ltd;	// Linear Torque Decoupling vector
				b3Vector3 c = m_calculatedTransformB.getOrigin() - transA.getOrigin();
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
        info->m_constraintError[srow] = b3Scalar(0.f);
        if (powered)
        {
			info->cfm[srow] = limot->m_normalCFM;
            if(!limit)
            {
				b3Scalar tag_vel = rotational ? limot->m_targetVelocity : -limot->m_targetVelocity;

				b3Scalar mot_fact = getMotorFactor(	limot->m_currentPosition, 
													limot->m_loLimit,
													limot->m_hiLimit, 
													tag_vel, 
													info->fps * limot->m_stopERP);
				info->m_constraintError[srow] += mot_fact * limot->m_targetVelocity;
                info->m_lowerLimit[srow] = -limot->m_maxMotorForce;
                info->m_upperLimit[srow] = limot->m_maxMotorForce;
            }
        }
        if(limit)
        {
            b3Scalar k = info->fps * limot->m_stopERP;
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
                info->m_lowerLimit[srow] = -B3_INFINITY;
                info->m_upperLimit[srow] = B3_INFINITY;
            }
            else
            {
                if (limit == 1)
                {
                    info->m_lowerLimit[srow] = 0;
                    info->m_upperLimit[srow] = B3_INFINITY;
                }
                else
                {
                    info->m_lowerLimit[srow] = -B3_INFINITY;
                    info->m_upperLimit[srow] = 0;
                }
                // deal with bounce
                if (limot->m_bounce > 0)
                {
                    // calculate joint velocity
                    b3Scalar vel;
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
                            b3Scalar newc = -limot->m_bounce* vel;
                            if (newc > info->m_constraintError[srow]) 
								info->m_constraintError[srow] = newc;
                        }
                    }
                    else
                    {
                        if (vel > 0)
                        {
                            b3Scalar newc = -limot->m_bounce * vel;
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
void b3Generic6DofConstraint::setParam(int num, b3Scalar value, int axis)
{
	if((axis >= 0) && (axis < 3))
	{
		switch(num)
		{
			case B3_CONSTRAINT_STOP_ERP : 
				m_linearLimits.m_stopERP[axis] = value;
				m_flags |= B3_6DOF_FLAGS_ERP_STOP << (axis * B3_6DOF_FLAGS_AXIS_SHIFT);
				break;
			case B3_CONSTRAINT_STOP_CFM : 
				m_linearLimits.m_stopCFM[axis] = value;
				m_flags |= B3_6DOF_FLAGS_CFM_STOP << (axis * B3_6DOF_FLAGS_AXIS_SHIFT);
				break;
			case B3_CONSTRAINT_CFM : 
				m_linearLimits.m_normalCFM[axis] = value;
				m_flags |= B3_6DOF_FLAGS_CFM_NORM << (axis * B3_6DOF_FLAGS_AXIS_SHIFT);
				break;
			default : 
				b3AssertConstrParams(0);
		}
	}
	else if((axis >=3) && (axis < 6))
	{
		switch(num)
		{
			case B3_CONSTRAINT_STOP_ERP : 
				m_angularLimits[axis - 3].m_stopERP = value;
				m_flags |= B3_6DOF_FLAGS_ERP_STOP << (axis * B3_6DOF_FLAGS_AXIS_SHIFT);
				break;
			case B3_CONSTRAINT_STOP_CFM : 
				m_angularLimits[axis - 3].m_stopCFM = value;
				m_flags |= B3_6DOF_FLAGS_CFM_STOP << (axis * B3_6DOF_FLAGS_AXIS_SHIFT);
				break;
			case B3_CONSTRAINT_CFM : 
				m_angularLimits[axis - 3].m_normalCFM = value;
				m_flags |= B3_6DOF_FLAGS_CFM_NORM << (axis * B3_6DOF_FLAGS_AXIS_SHIFT);
				break;
			default : 
				b3AssertConstrParams(0);
		}
	}
	else
	{
		b3AssertConstrParams(0);
	}
}

	///return the local value of parameter
b3Scalar b3Generic6DofConstraint::getParam(int num, int axis) const 
{
	b3Scalar retVal = 0;
	if((axis >= 0) && (axis < 3))
	{
		switch(num)
		{
			case B3_CONSTRAINT_STOP_ERP : 
				b3AssertConstrParams(m_flags & (B3_6DOF_FLAGS_ERP_STOP << (axis * B3_6DOF_FLAGS_AXIS_SHIFT)));
				retVal = m_linearLimits.m_stopERP[axis];
				break;
			case B3_CONSTRAINT_STOP_CFM : 
				b3AssertConstrParams(m_flags & (B3_6DOF_FLAGS_CFM_STOP << (axis * B3_6DOF_FLAGS_AXIS_SHIFT)));
				retVal = m_linearLimits.m_stopCFM[axis];
				break;
			case B3_CONSTRAINT_CFM : 
				b3AssertConstrParams(m_flags & (B3_6DOF_FLAGS_CFM_NORM << (axis * B3_6DOF_FLAGS_AXIS_SHIFT)));
				retVal = m_linearLimits.m_normalCFM[axis];
				break;
			default : 
				b3AssertConstrParams(0);
		}
	}
	else if((axis >=3) && (axis < 6))
	{
		switch(num)
		{
			case B3_CONSTRAINT_STOP_ERP : 
				b3AssertConstrParams(m_flags & (B3_6DOF_FLAGS_ERP_STOP << (axis * B3_6DOF_FLAGS_AXIS_SHIFT)));
				retVal = m_angularLimits[axis - 3].m_stopERP;
				break;
			case B3_CONSTRAINT_STOP_CFM : 
				b3AssertConstrParams(m_flags & (B3_6DOF_FLAGS_CFM_STOP << (axis * B3_6DOF_FLAGS_AXIS_SHIFT)));
				retVal = m_angularLimits[axis - 3].m_stopCFM;
				break;
			case B3_CONSTRAINT_CFM : 
				b3AssertConstrParams(m_flags & (B3_6DOF_FLAGS_CFM_NORM << (axis * B3_6DOF_FLAGS_AXIS_SHIFT)));
				retVal = m_angularLimits[axis - 3].m_normalCFM;
				break;
			default : 
				b3AssertConstrParams(0);
		}
	}
	else
	{
		b3AssertConstrParams(0);
	}
	return retVal;
}

 

void b3Generic6DofConstraint::setAxis(const b3Vector3& axis1,const b3Vector3& axis2, const b3RigidBodyData* bodies)
{
	b3Vector3 zAxis = axis1.normalized();
	b3Vector3 yAxis = axis2.normalized();
	b3Vector3 xAxis = yAxis.cross(zAxis); // we want right coordinate system
	
	b3Transform frameInW;
	frameInW.setIdentity();
	frameInW.getBasis().setValue(	xAxis[0], yAxis[0], zAxis[0],	
	                                xAxis[1], yAxis[1], zAxis[1],
	                               xAxis[2], yAxis[2], zAxis[2]);
	
	// now get constraint frame in local coordinate systems
	m_frameInA = getCenterOfMassTransform(bodies[m_rbA]).inverse() * frameInW;
	m_frameInB = getCenterOfMassTransform(bodies[m_rbB]).inverse() * frameInW;
	
	calculateTransforms(bodies);
}
