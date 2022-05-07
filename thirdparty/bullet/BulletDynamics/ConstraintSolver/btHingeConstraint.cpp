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

#include "btHingeConstraint.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"
#include "LinearMath/btTransformUtil.h"
#include "LinearMath/btMinMax.h"
#include <new>
#include "btSolverBody.h"

//#define HINGE_USE_OBSOLETE_SOLVER false
#define HINGE_USE_OBSOLETE_SOLVER false

#define HINGE_USE_FRAME_OFFSET true

#ifndef __SPU__

btHingeConstraint::btHingeConstraint(btRigidBody& rbA, btRigidBody& rbB, const btVector3& pivotInA, const btVector3& pivotInB,
									 const btVector3& axisInA, const btVector3& axisInB, bool useReferenceFrameA)
	: btTypedConstraint(HINGE_CONSTRAINT_TYPE, rbA, rbB),
#ifdef _BT_USE_CENTER_LIMIT_
	  m_limit(),
#endif
	  m_angularOnly(false),
	  m_enableAngularMotor(false),
	  m_useSolveConstraintObsolete(HINGE_USE_OBSOLETE_SOLVER),
	  m_useOffsetForConstraintFrame(HINGE_USE_FRAME_OFFSET),
	  m_useReferenceFrameA(useReferenceFrameA),
	  m_flags(0),
	  m_normalCFM(0),
	  m_normalERP(0),
	  m_stopCFM(0),
	  m_stopERP(0)
{
	m_rbAFrame.getOrigin() = pivotInA;

	// since no frame is given, assume this to be zero angle and just pick rb transform axis
	btVector3 rbAxisA1 = rbA.getCenterOfMassTransform().getBasis().getColumn(0);

	btVector3 rbAxisA2;
	btScalar projection = axisInA.dot(rbAxisA1);
	if (projection >= 1.0f - SIMD_EPSILON)
	{
		rbAxisA1 = -rbA.getCenterOfMassTransform().getBasis().getColumn(2);
		rbAxisA2 = rbA.getCenterOfMassTransform().getBasis().getColumn(1);
	}
	else if (projection <= -1.0f + SIMD_EPSILON)
	{
		rbAxisA1 = rbA.getCenterOfMassTransform().getBasis().getColumn(2);
		rbAxisA2 = rbA.getCenterOfMassTransform().getBasis().getColumn(1);
	}
	else
	{
		rbAxisA2 = axisInA.cross(rbAxisA1);
		rbAxisA1 = rbAxisA2.cross(axisInA);
	}

	m_rbAFrame.getBasis().setValue(rbAxisA1.getX(), rbAxisA2.getX(), axisInA.getX(),
								   rbAxisA1.getY(), rbAxisA2.getY(), axisInA.getY(),
								   rbAxisA1.getZ(), rbAxisA2.getZ(), axisInA.getZ());

	btQuaternion rotationArc = shortestArcQuat(axisInA, axisInB);
	btVector3 rbAxisB1 = quatRotate(rotationArc, rbAxisA1);
	btVector3 rbAxisB2 = axisInB.cross(rbAxisB1);

	m_rbBFrame.getOrigin() = pivotInB;
	m_rbBFrame.getBasis().setValue(rbAxisB1.getX(), rbAxisB2.getX(), axisInB.getX(),
								   rbAxisB1.getY(), rbAxisB2.getY(), axisInB.getY(),
								   rbAxisB1.getZ(), rbAxisB2.getZ(), axisInB.getZ());

#ifndef _BT_USE_CENTER_LIMIT_
	//start with free
	m_lowerLimit = btScalar(1.0f);
	m_upperLimit = btScalar(-1.0f);
	m_biasFactor = 0.3f;
	m_relaxationFactor = 1.0f;
	m_limitSoftness = 0.9f;
	m_solveLimit = false;
#endif
	m_referenceSign = m_useReferenceFrameA ? btScalar(-1.f) : btScalar(1.f);
}

btHingeConstraint::btHingeConstraint(btRigidBody& rbA, const btVector3& pivotInA, const btVector3& axisInA, bool useReferenceFrameA)
	: btTypedConstraint(HINGE_CONSTRAINT_TYPE, rbA),
#ifdef _BT_USE_CENTER_LIMIT_
	  m_limit(),
#endif
	  m_angularOnly(false),
	  m_enableAngularMotor(false),
	  m_useSolveConstraintObsolete(HINGE_USE_OBSOLETE_SOLVER),
	  m_useOffsetForConstraintFrame(HINGE_USE_FRAME_OFFSET),
	  m_useReferenceFrameA(useReferenceFrameA),
	  m_flags(0),
	  m_normalCFM(0),
	  m_normalERP(0),
	  m_stopCFM(0),
	  m_stopERP(0)
{
	// since no frame is given, assume this to be zero angle and just pick rb transform axis
	// fixed axis in worldspace
	btVector3 rbAxisA1, rbAxisA2;
	btPlaneSpace1(axisInA, rbAxisA1, rbAxisA2);

	m_rbAFrame.getOrigin() = pivotInA;
	m_rbAFrame.getBasis().setValue(rbAxisA1.getX(), rbAxisA2.getX(), axisInA.getX(),
								   rbAxisA1.getY(), rbAxisA2.getY(), axisInA.getY(),
								   rbAxisA1.getZ(), rbAxisA2.getZ(), axisInA.getZ());

	btVector3 axisInB = rbA.getCenterOfMassTransform().getBasis() * axisInA;

	btQuaternion rotationArc = shortestArcQuat(axisInA, axisInB);
	btVector3 rbAxisB1 = quatRotate(rotationArc, rbAxisA1);
	btVector3 rbAxisB2 = axisInB.cross(rbAxisB1);

	m_rbBFrame.getOrigin() = rbA.getCenterOfMassTransform()(pivotInA);
	m_rbBFrame.getBasis().setValue(rbAxisB1.getX(), rbAxisB2.getX(), axisInB.getX(),
								   rbAxisB1.getY(), rbAxisB2.getY(), axisInB.getY(),
								   rbAxisB1.getZ(), rbAxisB2.getZ(), axisInB.getZ());

#ifndef _BT_USE_CENTER_LIMIT_
	//start with free
	m_lowerLimit = btScalar(1.0f);
	m_upperLimit = btScalar(-1.0f);
	m_biasFactor = 0.3f;
	m_relaxationFactor = 1.0f;
	m_limitSoftness = 0.9f;
	m_solveLimit = false;
#endif
	m_referenceSign = m_useReferenceFrameA ? btScalar(-1.f) : btScalar(1.f);
}

btHingeConstraint::btHingeConstraint(btRigidBody& rbA, btRigidBody& rbB,
									 const btTransform& rbAFrame, const btTransform& rbBFrame, bool useReferenceFrameA)
	: btTypedConstraint(HINGE_CONSTRAINT_TYPE, rbA, rbB), m_rbAFrame(rbAFrame), m_rbBFrame(rbBFrame),
#ifdef _BT_USE_CENTER_LIMIT_
	  m_limit(),
#endif
	  m_angularOnly(false),
	  m_enableAngularMotor(false),
	  m_useSolveConstraintObsolete(HINGE_USE_OBSOLETE_SOLVER),
	  m_useOffsetForConstraintFrame(HINGE_USE_FRAME_OFFSET),
	  m_useReferenceFrameA(useReferenceFrameA),
	  m_flags(0),
	  m_normalCFM(0),
	  m_normalERP(0),
	  m_stopCFM(0),
	  m_stopERP(0)
{
#ifndef _BT_USE_CENTER_LIMIT_
	//start with free
	m_lowerLimit = btScalar(1.0f);
	m_upperLimit = btScalar(-1.0f);
	m_biasFactor = 0.3f;
	m_relaxationFactor = 1.0f;
	m_limitSoftness = 0.9f;
	m_solveLimit = false;
#endif
	m_referenceSign = m_useReferenceFrameA ? btScalar(-1.f) : btScalar(1.f);
}

btHingeConstraint::btHingeConstraint(btRigidBody& rbA, const btTransform& rbAFrame, bool useReferenceFrameA)
	: btTypedConstraint(HINGE_CONSTRAINT_TYPE, rbA), m_rbAFrame(rbAFrame), m_rbBFrame(rbAFrame),
#ifdef _BT_USE_CENTER_LIMIT_
	  m_limit(),
#endif
	  m_angularOnly(false),
	  m_enableAngularMotor(false),
	  m_useSolveConstraintObsolete(HINGE_USE_OBSOLETE_SOLVER),
	  m_useOffsetForConstraintFrame(HINGE_USE_FRAME_OFFSET),
	  m_useReferenceFrameA(useReferenceFrameA),
	  m_flags(0),
	  m_normalCFM(0),
	  m_normalERP(0),
	  m_stopCFM(0),
	  m_stopERP(0)
{
	///not providing rigidbody B means implicitly using worldspace for body B

	m_rbBFrame.getOrigin() = m_rbA.getCenterOfMassTransform()(m_rbAFrame.getOrigin());
#ifndef _BT_USE_CENTER_LIMIT_
	//start with free
	m_lowerLimit = btScalar(1.0f);
	m_upperLimit = btScalar(-1.0f);
	m_biasFactor = 0.3f;
	m_relaxationFactor = 1.0f;
	m_limitSoftness = 0.9f;
	m_solveLimit = false;
#endif
	m_referenceSign = m_useReferenceFrameA ? btScalar(-1.f) : btScalar(1.f);
}

void btHingeConstraint::buildJacobian()
{
	if (m_useSolveConstraintObsolete)
	{
		m_appliedImpulse = btScalar(0.);
		m_accMotorImpulse = btScalar(0.);

		if (!m_angularOnly)
		{
			btVector3 pivotAInW = m_rbA.getCenterOfMassTransform() * m_rbAFrame.getOrigin();
			btVector3 pivotBInW = m_rbB.getCenterOfMassTransform() * m_rbBFrame.getOrigin();
			btVector3 relPos = pivotBInW - pivotAInW;

			btVector3 normal[3];
			if (relPos.length2() > SIMD_EPSILON)
			{
				normal[0] = relPos.normalized();
			}
			else
			{
				normal[0].setValue(btScalar(1.0), 0, 0);
			}

			btPlaneSpace1(normal[0], normal[1], normal[2]);

			for (int i = 0; i < 3; i++)
			{
				new (&m_jac[i]) btJacobianEntry(
					m_rbA.getCenterOfMassTransform().getBasis().transpose(),
					m_rbB.getCenterOfMassTransform().getBasis().transpose(),
					pivotAInW - m_rbA.getCenterOfMassPosition(),
					pivotBInW - m_rbB.getCenterOfMassPosition(),
					normal[i],
					m_rbA.getInvInertiaDiagLocal(),
					m_rbA.getInvMass(),
					m_rbB.getInvInertiaDiagLocal(),
					m_rbB.getInvMass());
			}
		}

		//calculate two perpendicular jointAxis, orthogonal to hingeAxis
		//these two jointAxis require equal angular velocities for both bodies

		//this is unused for now, it's a todo
		btVector3 jointAxis0local;
		btVector3 jointAxis1local;

		btPlaneSpace1(m_rbAFrame.getBasis().getColumn(2), jointAxis0local, jointAxis1local);

		btVector3 jointAxis0 = getRigidBodyA().getCenterOfMassTransform().getBasis() * jointAxis0local;
		btVector3 jointAxis1 = getRigidBodyA().getCenterOfMassTransform().getBasis() * jointAxis1local;
		btVector3 hingeAxisWorld = getRigidBodyA().getCenterOfMassTransform().getBasis() * m_rbAFrame.getBasis().getColumn(2);

		new (&m_jacAng[0]) btJacobianEntry(jointAxis0,
										   m_rbA.getCenterOfMassTransform().getBasis().transpose(),
										   m_rbB.getCenterOfMassTransform().getBasis().transpose(),
										   m_rbA.getInvInertiaDiagLocal(),
										   m_rbB.getInvInertiaDiagLocal());

		new (&m_jacAng[1]) btJacobianEntry(jointAxis1,
										   m_rbA.getCenterOfMassTransform().getBasis().transpose(),
										   m_rbB.getCenterOfMassTransform().getBasis().transpose(),
										   m_rbA.getInvInertiaDiagLocal(),
										   m_rbB.getInvInertiaDiagLocal());

		new (&m_jacAng[2]) btJacobianEntry(hingeAxisWorld,
										   m_rbA.getCenterOfMassTransform().getBasis().transpose(),
										   m_rbB.getCenterOfMassTransform().getBasis().transpose(),
										   m_rbA.getInvInertiaDiagLocal(),
										   m_rbB.getInvInertiaDiagLocal());

		// clear accumulator
		m_accLimitImpulse = btScalar(0.);

		// test angular limit
		testLimit(m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform());

		//Compute K = J*W*J' for hinge axis
		btVector3 axisA = getRigidBodyA().getCenterOfMassTransform().getBasis() * m_rbAFrame.getBasis().getColumn(2);
		m_kHinge = 1.0f / (getRigidBodyA().computeAngularImpulseDenominator(axisA) +
						   getRigidBodyB().computeAngularImpulseDenominator(axisA));
	}
}

#endif  //__SPU__

static inline btScalar btNormalizeAnglePositive(btScalar angle)
{
	return btFmod(btFmod(angle, btScalar(2.0 * SIMD_PI)) + btScalar(2.0 * SIMD_PI), btScalar(2.0 * SIMD_PI));
}

static btScalar btShortestAngularDistance(btScalar accAngle, btScalar curAngle)
{
	btScalar result = btNormalizeAngle(btNormalizeAnglePositive(btNormalizeAnglePositive(curAngle) -
																btNormalizeAnglePositive(accAngle)));
	return result;
}

static btScalar btShortestAngleUpdate(btScalar accAngle, btScalar curAngle)
{
	btScalar tol(0.3);
	btScalar result = btShortestAngularDistance(accAngle, curAngle);

	if (btFabs(result) > tol)
		return curAngle;
	else
		return accAngle + result;

	return curAngle;
}

btScalar btHingeAccumulatedAngleConstraint::getAccumulatedHingeAngle()
{
	btScalar hingeAngle = getHingeAngle();
	m_accumulatedAngle = btShortestAngleUpdate(m_accumulatedAngle, hingeAngle);
	return m_accumulatedAngle;
}
void btHingeAccumulatedAngleConstraint::setAccumulatedHingeAngle(btScalar accAngle)
{
	m_accumulatedAngle = accAngle;
}

void btHingeAccumulatedAngleConstraint::getInfo1(btConstraintInfo1* info)
{
	//update m_accumulatedAngle
	btScalar curHingeAngle = getHingeAngle();
	m_accumulatedAngle = btShortestAngleUpdate(m_accumulatedAngle, curHingeAngle);

	btHingeConstraint::getInfo1(info);
}

void btHingeConstraint::getInfo1(btConstraintInfo1* info)
{
	if (m_useSolveConstraintObsolete)
	{
		info->m_numConstraintRows = 0;
		info->nub = 0;
	}
	else
	{
		info->m_numConstraintRows = 5;  // Fixed 3 linear + 2 angular
		info->nub = 1;
		//always add the row, to avoid computation (data is not available yet)
		//prepare constraint
		testLimit(m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform());
		if (getSolveLimit() || getEnableAngularMotor())
		{
			info->m_numConstraintRows++;  // limit 3rd anguar as well
			info->nub--;
		}
	}
}

void btHingeConstraint::getInfo1NonVirtual(btConstraintInfo1* info)
{
	if (m_useSolveConstraintObsolete)
	{
		info->m_numConstraintRows = 0;
		info->nub = 0;
	}
	else
	{
		//always add the 'limit' row, to avoid computation (data is not available yet)
		info->m_numConstraintRows = 6;  // Fixed 3 linear + 2 angular
		info->nub = 0;
	}
}

void btHingeConstraint::getInfo2(btConstraintInfo2* info)
{
	if (m_useOffsetForConstraintFrame)
	{
		getInfo2InternalUsingFrameOffset(info, m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform(), m_rbA.getAngularVelocity(), m_rbB.getAngularVelocity());
	}
	else
	{
		getInfo2Internal(info, m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform(), m_rbA.getAngularVelocity(), m_rbB.getAngularVelocity());
	}
}

void btHingeConstraint::getInfo2NonVirtual(btConstraintInfo2* info, const btTransform& transA, const btTransform& transB, const btVector3& angVelA, const btVector3& angVelB)
{
	///the regular (virtual) implementation getInfo2 already performs 'testLimit' during getInfo1, so we need to do it now
	testLimit(transA, transB);

	getInfo2Internal(info, transA, transB, angVelA, angVelB);
}

void btHingeConstraint::getInfo2Internal(btConstraintInfo2* info, const btTransform& transA, const btTransform& transB, const btVector3& angVelA, const btVector3& angVelB)
{
	btAssert(!m_useSolveConstraintObsolete);
	int i, skip = info->rowskip;
	// transforms in world space
	btTransform trA = transA * m_rbAFrame;
	btTransform trB = transB * m_rbBFrame;
	// pivot point
	btVector3 pivotAInW = trA.getOrigin();
	btVector3 pivotBInW = trB.getOrigin();
#if 0
	if (0)
	{
		for (i=0;i<6;i++)
		{
			info->m_J1linearAxis[i*skip]=0;
			info->m_J1linearAxis[i*skip+1]=0;
			info->m_J1linearAxis[i*skip+2]=0;

			info->m_J1angularAxis[i*skip]=0;
			info->m_J1angularAxis[i*skip+1]=0;
			info->m_J1angularAxis[i*skip+2]=0;

			info->m_J2linearAxis[i*skip]=0;
			info->m_J2linearAxis[i*skip+1]=0;
			info->m_J2linearAxis[i*skip+2]=0;

			info->m_J2angularAxis[i*skip]=0;
			info->m_J2angularAxis[i*skip+1]=0;
			info->m_J2angularAxis[i*skip+2]=0;

			info->m_constraintError[i*skip]=0.f;
		}
	}
#endif  //#if 0
	// linear (all fixed)

	if (!m_angularOnly)
	{
		info->m_J1linearAxis[0] = 1;
		info->m_J1linearAxis[skip + 1] = 1;
		info->m_J1linearAxis[2 * skip + 2] = 1;

		info->m_J2linearAxis[0] = -1;
		info->m_J2linearAxis[skip + 1] = -1;
		info->m_J2linearAxis[2 * skip + 2] = -1;
	}

	btVector3 a1 = pivotAInW - transA.getOrigin();
	{
		btVector3* angular0 = (btVector3*)(info->m_J1angularAxis);
		btVector3* angular1 = (btVector3*)(info->m_J1angularAxis + skip);
		btVector3* angular2 = (btVector3*)(info->m_J1angularAxis + 2 * skip);
		btVector3 a1neg = -a1;
		a1neg.getSkewSymmetricMatrix(angular0, angular1, angular2);
	}
	btVector3 a2 = pivotBInW - transB.getOrigin();
	{
		btVector3* angular0 = (btVector3*)(info->m_J2angularAxis);
		btVector3* angular1 = (btVector3*)(info->m_J2angularAxis + skip);
		btVector3* angular2 = (btVector3*)(info->m_J2angularAxis + 2 * skip);
		a2.getSkewSymmetricMatrix(angular0, angular1, angular2);
	}
	// linear RHS
	btScalar normalErp = (m_flags & BT_HINGE_FLAGS_ERP_NORM) ? m_normalERP : info->erp;

	btScalar k = info->fps * normalErp;
	if (!m_angularOnly)
	{
		for (i = 0; i < 3; i++)
		{
			info->m_constraintError[i * skip] = k * (pivotBInW[i] - pivotAInW[i]);
		}
	}
	// make rotations around X and Y equal
	// the hinge axis should be the only unconstrained
	// rotational axis, the angular velocity of the two bodies perpendicular to
	// the hinge axis should be equal. thus the constraint equations are
	//    p*w1 - p*w2 = 0
	//    q*w1 - q*w2 = 0
	// where p and q are unit vectors normal to the hinge axis, and w1 and w2
	// are the angular velocity vectors of the two bodies.
	// get hinge axis (Z)
	btVector3 ax1 = trA.getBasis().getColumn(2);
	// get 2 orthos to hinge axis (X, Y)
	btVector3 p = trA.getBasis().getColumn(0);
	btVector3 q = trA.getBasis().getColumn(1);
	// set the two hinge angular rows
	int s3 = 3 * info->rowskip;
	int s4 = 4 * info->rowskip;

	info->m_J1angularAxis[s3 + 0] = p[0];
	info->m_J1angularAxis[s3 + 1] = p[1];
	info->m_J1angularAxis[s3 + 2] = p[2];
	info->m_J1angularAxis[s4 + 0] = q[0];
	info->m_J1angularAxis[s4 + 1] = q[1];
	info->m_J1angularAxis[s4 + 2] = q[2];

	info->m_J2angularAxis[s3 + 0] = -p[0];
	info->m_J2angularAxis[s3 + 1] = -p[1];
	info->m_J2angularAxis[s3 + 2] = -p[2];
	info->m_J2angularAxis[s4 + 0] = -q[0];
	info->m_J2angularAxis[s4 + 1] = -q[1];
	info->m_J2angularAxis[s4 + 2] = -q[2];
	// compute the right hand side of the constraint equation. set relative
	// body velocities along p and q to bring the hinge back into alignment.
	// if ax1,ax2 are the unit length hinge axes as computed from body1 and
	// body2, we need to rotate both bodies along the axis u = (ax1 x ax2).
	// if `theta' is the angle between ax1 and ax2, we need an angular velocity
	// along u to cover angle erp*theta in one step :
	//   |angular_velocity| = angle/time = erp*theta / stepsize
	//                      = (erp*fps) * theta
	//    angular_velocity  = |angular_velocity| * (ax1 x ax2) / |ax1 x ax2|
	//                      = (erp*fps) * theta * (ax1 x ax2) / sin(theta)
	// ...as ax1 and ax2 are unit length. if theta is smallish,
	// theta ~= sin(theta), so
	//    angular_velocity  = (erp*fps) * (ax1 x ax2)
	// ax1 x ax2 is in the plane space of ax1, so we project the angular
	// velocity to p and q to find the right hand side.
	btVector3 ax2 = trB.getBasis().getColumn(2);
	btVector3 u = ax1.cross(ax2);
	info->m_constraintError[s3] = k * u.dot(p);
	info->m_constraintError[s4] = k * u.dot(q);
	// check angular limits
	int nrow = 4;  // last filled row
	int srow;
	btScalar limit_err = btScalar(0.0);
	int limit = 0;
	if (getSolveLimit())
	{
#ifdef _BT_USE_CENTER_LIMIT_
		limit_err = m_limit.getCorrection() * m_referenceSign;
#else
		limit_err = m_correction * m_referenceSign;
#endif
		limit = (limit_err > btScalar(0.0)) ? 1 : 2;
	}
	// if the hinge has joint limits or motor, add in the extra row
	bool powered = getEnableAngularMotor();
	if (limit || powered)
	{
		nrow++;
		srow = nrow * info->rowskip;
		info->m_J1angularAxis[srow + 0] = ax1[0];
		info->m_J1angularAxis[srow + 1] = ax1[1];
		info->m_J1angularAxis[srow + 2] = ax1[2];

		info->m_J2angularAxis[srow + 0] = -ax1[0];
		info->m_J2angularAxis[srow + 1] = -ax1[1];
		info->m_J2angularAxis[srow + 2] = -ax1[2];

		btScalar lostop = getLowerLimit();
		btScalar histop = getUpperLimit();
		if (limit && (lostop == histop))
		{  // the joint motor is ineffective
			powered = false;
		}
		info->m_constraintError[srow] = btScalar(0.0f);
		btScalar currERP = (m_flags & BT_HINGE_FLAGS_ERP_STOP) ? m_stopERP : normalErp;
		if (powered)
		{
			if (m_flags & BT_HINGE_FLAGS_CFM_NORM)
			{
				info->cfm[srow] = m_normalCFM;
			}
			btScalar mot_fact = getMotorFactor(m_hingeAngle, lostop, histop, m_motorTargetVelocity, info->fps * currERP);
			info->m_constraintError[srow] += mot_fact * m_motorTargetVelocity * m_referenceSign;
			info->m_lowerLimit[srow] = -m_maxMotorImpulse;
			info->m_upperLimit[srow] = m_maxMotorImpulse;
		}
		if (limit)
		{
			k = info->fps * currERP;
			info->m_constraintError[srow] += k * limit_err;
			if (m_flags & BT_HINGE_FLAGS_CFM_STOP)
			{
				info->cfm[srow] = m_stopCFM;
			}
			if (lostop == histop)
			{
				// limited low and high simultaneously
				info->m_lowerLimit[srow] = -SIMD_INFINITY;
				info->m_upperLimit[srow] = SIMD_INFINITY;
			}
			else if (limit == 1)
			{  // low limit
				info->m_lowerLimit[srow] = 0;
				info->m_upperLimit[srow] = SIMD_INFINITY;
			}
			else
			{  // high limit
				info->m_lowerLimit[srow] = -SIMD_INFINITY;
				info->m_upperLimit[srow] = 0;
			}
			// bounce (we'll use slider parameter abs(1.0 - m_dampingLimAng) for that)
#ifdef _BT_USE_CENTER_LIMIT_
			btScalar bounce = m_limit.getRelaxationFactor();
#else
			btScalar bounce = m_relaxationFactor;
#endif
			if (bounce > btScalar(0.0))
			{
				btScalar vel = angVelA.dot(ax1);
				vel -= angVelB.dot(ax1);
				// only apply bounce if the velocity is incoming, and if the
				// resulting c[] exceeds what we already have.
				if (limit == 1)
				{  // low limit
					if (vel < 0)
					{
						btScalar newc = -bounce * vel;
						if (newc > info->m_constraintError[srow])
						{
							info->m_constraintError[srow] = newc;
						}
					}
				}
				else
				{  // high limit - all those computations are reversed
					if (vel > 0)
					{
						btScalar newc = -bounce * vel;
						if (newc < info->m_constraintError[srow])
						{
							info->m_constraintError[srow] = newc;
						}
					}
				}
			}
#ifdef _BT_USE_CENTER_LIMIT_
			info->m_constraintError[srow] *= m_limit.getBiasFactor();
#else
			info->m_constraintError[srow] *= m_biasFactor;
#endif
		}  // if(limit)
	}      // if angular limit or powered
}

void btHingeConstraint::setFrames(const btTransform& frameA, const btTransform& frameB)
{
	m_rbAFrame = frameA;
	m_rbBFrame = frameB;
	buildJacobian();
}

void btHingeConstraint::updateRHS(btScalar timeStep)
{
	(void)timeStep;
}

btScalar btHingeConstraint::getHingeAngle()
{
	return getHingeAngle(m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform());
}

btScalar btHingeConstraint::getHingeAngle(const btTransform& transA, const btTransform& transB)
{
	const btVector3 refAxis0 = transA.getBasis() * m_rbAFrame.getBasis().getColumn(0);
	const btVector3 refAxis1 = transA.getBasis() * m_rbAFrame.getBasis().getColumn(1);
	const btVector3 swingAxis = transB.getBasis() * m_rbBFrame.getBasis().getColumn(1);
	//	btScalar angle = btAtan2Fast(swingAxis.dot(refAxis0), swingAxis.dot(refAxis1));
	btScalar angle = btAtan2(swingAxis.dot(refAxis0), swingAxis.dot(refAxis1));
	return m_referenceSign * angle;
}

void btHingeConstraint::testLimit(const btTransform& transA, const btTransform& transB)
{
	// Compute limit information
	m_hingeAngle = getHingeAngle(transA, transB);
#ifdef _BT_USE_CENTER_LIMIT_
	m_limit.test(m_hingeAngle);
#else
	m_correction = btScalar(0.);
	m_limitSign = btScalar(0.);
	m_solveLimit = false;
	if (m_lowerLimit <= m_upperLimit)
	{
		m_hingeAngle = btAdjustAngleToLimits(m_hingeAngle, m_lowerLimit, m_upperLimit);
		if (m_hingeAngle <= m_lowerLimit)
		{
			m_correction = (m_lowerLimit - m_hingeAngle);
			m_limitSign = 1.0f;
			m_solveLimit = true;
		}
		else if (m_hingeAngle >= m_upperLimit)
		{
			m_correction = m_upperLimit - m_hingeAngle;
			m_limitSign = -1.0f;
			m_solveLimit = true;
		}
	}
#endif
	return;
}

static btVector3 vHinge(0, 0, btScalar(1));

void btHingeConstraint::setMotorTarget(const btQuaternion& qAinB, btScalar dt)
{
	// convert target from body to constraint space
	btQuaternion qConstraint = m_rbBFrame.getRotation().inverse() * qAinB * m_rbAFrame.getRotation();
	qConstraint.normalize();

	// extract "pure" hinge component
	btVector3 vNoHinge = quatRotate(qConstraint, vHinge);
	vNoHinge.normalize();
	btQuaternion qNoHinge = shortestArcQuat(vHinge, vNoHinge);
	btQuaternion qHinge = qNoHinge.inverse() * qConstraint;
	qHinge.normalize();

	// compute angular target, clamped to limits
	btScalar targetAngle = qHinge.getAngle();
	if (targetAngle > SIMD_PI)  // long way around. flip quat and recalculate.
	{
		qHinge = -(qHinge);
		targetAngle = qHinge.getAngle();
	}
	if (qHinge.getZ() < 0)
		targetAngle = -targetAngle;

	setMotorTarget(targetAngle, dt);
}

void btHingeConstraint::setMotorTarget(btScalar targetAngle, btScalar dt)
{
#ifdef _BT_USE_CENTER_LIMIT_
	m_limit.fit(targetAngle);
#else
	if (m_lowerLimit < m_upperLimit)
	{
		if (targetAngle < m_lowerLimit)
			targetAngle = m_lowerLimit;
		else if (targetAngle > m_upperLimit)
			targetAngle = m_upperLimit;
	}
#endif
	// compute angular velocity
	btScalar curAngle = getHingeAngle(m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform());
	btScalar dAngle = targetAngle - curAngle;
	m_motorTargetVelocity = dAngle / dt;
}

void btHingeConstraint::getInfo2InternalUsingFrameOffset(btConstraintInfo2* info, const btTransform& transA, const btTransform& transB, const btVector3& angVelA, const btVector3& angVelB)
{
	btAssert(!m_useSolveConstraintObsolete);
	int i, s = info->rowskip;
	// transforms in world space
	btTransform trA = transA * m_rbAFrame;
	btTransform trB = transB * m_rbBFrame;
	// pivot point
//	btVector3 pivotAInW = trA.getOrigin();
//	btVector3 pivotBInW = trB.getOrigin();
#if 1
	// difference between frames in WCS
	btVector3 ofs = trB.getOrigin() - trA.getOrigin();
	// now get weight factors depending on masses
	btScalar miA = getRigidBodyA().getInvMass();
	btScalar miB = getRigidBodyB().getInvMass();
	bool hasStaticBody = (miA < SIMD_EPSILON) || (miB < SIMD_EPSILON);
	btScalar miS = miA + miB;
	btScalar factA, factB;
	if (miS > btScalar(0.f))
	{
		factA = miB / miS;
	}
	else
	{
		factA = btScalar(0.5f);
	}
	factB = btScalar(1.0f) - factA;
	// get the desired direction of hinge axis
	// as weighted sum of Z-orthos of frameA and frameB in WCS
	btVector3 ax1A = trA.getBasis().getColumn(2);
	btVector3 ax1B = trB.getBasis().getColumn(2);
	btVector3 ax1 = ax1A * factA + ax1B * factB;
	if (ax1.length2()<SIMD_EPSILON)
	{
		factA=0.f;
		factB=1.f;
		ax1 = ax1A * factA + ax1B * factB;
	}
	ax1.normalize();
	// fill first 3 rows
	// we want: velA + wA x relA == velB + wB x relB
	btTransform bodyA_trans = transA;
	btTransform bodyB_trans = transB;
	int s0 = 0;
	int s1 = s;
	int s2 = s * 2;
	int nrow = 2;  // last filled row
	btVector3 tmpA, tmpB, relA, relB, p, q;
	// get vector from bodyB to frameB in WCS
	relB = trB.getOrigin() - bodyB_trans.getOrigin();
	// get its projection to hinge axis
	btVector3 projB = ax1 * relB.dot(ax1);
	// get vector directed from bodyB to hinge axis (and orthogonal to it)
	btVector3 orthoB = relB - projB;
	// same for bodyA
	relA = trA.getOrigin() - bodyA_trans.getOrigin();
	btVector3 projA = ax1 * relA.dot(ax1);
	btVector3 orthoA = relA - projA;
	btVector3 totalDist = projA - projB;
	// get offset vectors relA and relB
	relA = orthoA + totalDist * factA;
	relB = orthoB - totalDist * factB;
	// now choose average ortho to hinge axis
	p = orthoB * factA + orthoA * factB;
	btScalar len2 = p.length2();
	if (len2 > SIMD_EPSILON)
	{
		p /= btSqrt(len2);
	}
	else
	{
		p = trA.getBasis().getColumn(1);
	}
	// make one more ortho
	q = ax1.cross(p);
	// fill three rows
	tmpA = relA.cross(p);
	tmpB = relB.cross(p);
	for (i = 0; i < 3; i++) info->m_J1angularAxis[s0 + i] = tmpA[i];
	for (i = 0; i < 3; i++) info->m_J2angularAxis[s0 + i] = -tmpB[i];
	tmpA = relA.cross(q);
	tmpB = relB.cross(q);
	if (hasStaticBody && getSolveLimit())
	{  // to make constraint between static and dynamic objects more rigid
		// remove wA (or wB) from equation if angular limit is hit
		tmpB *= factB;
		tmpA *= factA;
	}
	for (i = 0; i < 3; i++) info->m_J1angularAxis[s1 + i] = tmpA[i];
	for (i = 0; i < 3; i++) info->m_J2angularAxis[s1 + i] = -tmpB[i];
	tmpA = relA.cross(ax1);
	tmpB = relB.cross(ax1);
	if (hasStaticBody)
	{  // to make constraint between static and dynamic objects more rigid
		// remove wA (or wB) from equation
		tmpB *= factB;
		tmpA *= factA;
	}
	for (i = 0; i < 3; i++) info->m_J1angularAxis[s2 + i] = tmpA[i];
	for (i = 0; i < 3; i++) info->m_J2angularAxis[s2 + i] = -tmpB[i];

	btScalar normalErp = (m_flags & BT_HINGE_FLAGS_ERP_NORM) ? m_normalERP : info->erp;
	btScalar k = info->fps * normalErp;

	if (!m_angularOnly)
	{
		for (i = 0; i < 3; i++) info->m_J1linearAxis[s0 + i] = p[i];
		for (i = 0; i < 3; i++) info->m_J1linearAxis[s1 + i] = q[i];
		for (i = 0; i < 3; i++) info->m_J1linearAxis[s2 + i] = ax1[i];

		for (i = 0; i < 3; i++) info->m_J2linearAxis[s0 + i] = -p[i];
		for (i = 0; i < 3; i++) info->m_J2linearAxis[s1 + i] = -q[i];
		for (i = 0; i < 3; i++) info->m_J2linearAxis[s2 + i] = -ax1[i];

		// compute three elements of right hand side

		btScalar rhs = k * p.dot(ofs);
		info->m_constraintError[s0] = rhs;
		rhs = k * q.dot(ofs);
		info->m_constraintError[s1] = rhs;
		rhs = k * ax1.dot(ofs);
		info->m_constraintError[s2] = rhs;
	}
	// the hinge axis should be the only unconstrained
	// rotational axis, the angular velocity of the two bodies perpendicular to
	// the hinge axis should be equal. thus the constraint equations are
	//    p*w1 - p*w2 = 0
	//    q*w1 - q*w2 = 0
	// where p and q are unit vectors normal to the hinge axis, and w1 and w2
	// are the angular velocity vectors of the two bodies.
	int s3 = 3 * s;
	int s4 = 4 * s;
	info->m_J1angularAxis[s3 + 0] = p[0];
	info->m_J1angularAxis[s3 + 1] = p[1];
	info->m_J1angularAxis[s3 + 2] = p[2];
	info->m_J1angularAxis[s4 + 0] = q[0];
	info->m_J1angularAxis[s4 + 1] = q[1];
	info->m_J1angularAxis[s4 + 2] = q[2];

	info->m_J2angularAxis[s3 + 0] = -p[0];
	info->m_J2angularAxis[s3 + 1] = -p[1];
	info->m_J2angularAxis[s3 + 2] = -p[2];
	info->m_J2angularAxis[s4 + 0] = -q[0];
	info->m_J2angularAxis[s4 + 1] = -q[1];
	info->m_J2angularAxis[s4 + 2] = -q[2];
	// compute the right hand side of the constraint equation. set relative
	// body velocities along p and q to bring the hinge back into alignment.
	// if ax1A,ax1B are the unit length hinge axes as computed from bodyA and
	// bodyB, we need to rotate both bodies along the axis u = (ax1 x ax2).
	// if "theta" is the angle between ax1 and ax2, we need an angular velocity
	// along u to cover angle erp*theta in one step :
	//   |angular_velocity| = angle/time = erp*theta / stepsize
	//                      = (erp*fps) * theta
	//    angular_velocity  = |angular_velocity| * (ax1 x ax2) / |ax1 x ax2|
	//                      = (erp*fps) * theta * (ax1 x ax2) / sin(theta)
	// ...as ax1 and ax2 are unit length. if theta is smallish,
	// theta ~= sin(theta), so
	//    angular_velocity  = (erp*fps) * (ax1 x ax2)
	// ax1 x ax2 is in the plane space of ax1, so we project the angular
	// velocity to p and q to find the right hand side.
	k = info->fps * normalErp;  //??

	btVector3 u = ax1A.cross(ax1B);
	info->m_constraintError[s3] = k * u.dot(p);
	info->m_constraintError[s4] = k * u.dot(q);
#endif
	// check angular limits
	nrow = 4;  // last filled row
	int srow;
	btScalar limit_err = btScalar(0.0);
	int limit = 0;
	if (getSolveLimit())
	{
#ifdef _BT_USE_CENTER_LIMIT_
		limit_err = m_limit.getCorrection() * m_referenceSign;
#else
		limit_err = m_correction * m_referenceSign;
#endif
		limit = (limit_err > btScalar(0.0)) ? 1 : 2;
	}
	// if the hinge has joint limits or motor, add in the extra row
	bool powered = getEnableAngularMotor();
	if (limit || powered)
	{
		nrow++;
		srow = nrow * info->rowskip;
		info->m_J1angularAxis[srow + 0] = ax1[0];
		info->m_J1angularAxis[srow + 1] = ax1[1];
		info->m_J1angularAxis[srow + 2] = ax1[2];

		info->m_J2angularAxis[srow + 0] = -ax1[0];
		info->m_J2angularAxis[srow + 1] = -ax1[1];
		info->m_J2angularAxis[srow + 2] = -ax1[2];

		btScalar lostop = getLowerLimit();
		btScalar histop = getUpperLimit();
		if (limit && (lostop == histop))
		{  // the joint motor is ineffective
			powered = false;
		}
		info->m_constraintError[srow] = btScalar(0.0f);
		btScalar currERP = (m_flags & BT_HINGE_FLAGS_ERP_STOP) ? m_stopERP : normalErp;
		if (powered)
		{
			if (m_flags & BT_HINGE_FLAGS_CFM_NORM)
			{
				info->cfm[srow] = m_normalCFM;
			}
			btScalar mot_fact = getMotorFactor(m_hingeAngle, lostop, histop, m_motorTargetVelocity, info->fps * currERP);
			info->m_constraintError[srow] += mot_fact * m_motorTargetVelocity * m_referenceSign;
			info->m_lowerLimit[srow] = -m_maxMotorImpulse;
			info->m_upperLimit[srow] = m_maxMotorImpulse;
		}
		if (limit)
		{
			k = info->fps * currERP;
			info->m_constraintError[srow] += k * limit_err;
			if (m_flags & BT_HINGE_FLAGS_CFM_STOP)
			{
				info->cfm[srow] = m_stopCFM;
			}
			if (lostop == histop)
			{
				// limited low and high simultaneously
				info->m_lowerLimit[srow] = -SIMD_INFINITY;
				info->m_upperLimit[srow] = SIMD_INFINITY;
			}
			else if (limit == 1)
			{  // low limit
				info->m_lowerLimit[srow] = 0;
				info->m_upperLimit[srow] = SIMD_INFINITY;
			}
			else
			{  // high limit
				info->m_lowerLimit[srow] = -SIMD_INFINITY;
				info->m_upperLimit[srow] = 0;
			}
			// bounce (we'll use slider parameter abs(1.0 - m_dampingLimAng) for that)
#ifdef _BT_USE_CENTER_LIMIT_
			btScalar bounce = m_limit.getRelaxationFactor();
#else
			btScalar bounce = m_relaxationFactor;
#endif
			if (bounce > btScalar(0.0))
			{
				btScalar vel = angVelA.dot(ax1);
				vel -= angVelB.dot(ax1);
				// only apply bounce if the velocity is incoming, and if the
				// resulting c[] exceeds what we already have.
				if (limit == 1)
				{  // low limit
					if (vel < 0)
					{
						btScalar newc = -bounce * vel;
						if (newc > info->m_constraintError[srow])
						{
							info->m_constraintError[srow] = newc;
						}
					}
				}
				else
				{  // high limit - all those computations are reversed
					if (vel > 0)
					{
						btScalar newc = -bounce * vel;
						if (newc < info->m_constraintError[srow])
						{
							info->m_constraintError[srow] = newc;
						}
					}
				}
			}
#ifdef _BT_USE_CENTER_LIMIT_
			info->m_constraintError[srow] *= m_limit.getBiasFactor();
#else
			info->m_constraintError[srow] *= m_biasFactor;
#endif
		}  // if(limit)
	}      // if angular limit or powered
}

///override the default global value of a parameter (such as ERP or CFM), optionally provide the axis (0..5).
///If no axis is provided, it uses the default axis for this constraint.
void btHingeConstraint::setParam(int num, btScalar value, int axis)
{
	if ((axis == -1) || (axis == 5))
	{
		switch (num)
		{
			case BT_CONSTRAINT_STOP_ERP:
				m_stopERP = value;
				m_flags |= BT_HINGE_FLAGS_ERP_STOP;
				break;
			case BT_CONSTRAINT_STOP_CFM:
				m_stopCFM = value;
				m_flags |= BT_HINGE_FLAGS_CFM_STOP;
				break;
			case BT_CONSTRAINT_CFM:
				m_normalCFM = value;
				m_flags |= BT_HINGE_FLAGS_CFM_NORM;
				break;
			case BT_CONSTRAINT_ERP:
				m_normalERP = value;
				m_flags |= BT_HINGE_FLAGS_ERP_NORM;
				break;
			default:
				btAssertConstrParams(0);
		}
	}
	else
	{
		btAssertConstrParams(0);
	}
}

///return the local value of parameter
btScalar btHingeConstraint::getParam(int num, int axis) const
{
	btScalar retVal = 0;
	if ((axis == -1) || (axis == 5))
	{
		switch (num)
		{
			case BT_CONSTRAINT_STOP_ERP:
				btAssertConstrParams(m_flags & BT_HINGE_FLAGS_ERP_STOP);
				retVal = m_stopERP;
				break;
			case BT_CONSTRAINT_STOP_CFM:
				btAssertConstrParams(m_flags & BT_HINGE_FLAGS_CFM_STOP);
				retVal = m_stopCFM;
				break;
			case BT_CONSTRAINT_CFM:
				btAssertConstrParams(m_flags & BT_HINGE_FLAGS_CFM_NORM);
				retVal = m_normalCFM;
				break;
			case BT_CONSTRAINT_ERP:
				btAssertConstrParams(m_flags & BT_HINGE_FLAGS_ERP_NORM);
				retVal = m_normalERP;
				break;
			default:
				btAssertConstrParams(0);
		}
	}
	else
	{
		btAssertConstrParams(0);
	}
	return retVal;
}
