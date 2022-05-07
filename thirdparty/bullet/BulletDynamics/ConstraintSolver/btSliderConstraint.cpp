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

/*
Added by Roman Ponomarev (rponom@gmail.com)
April 04, 2008
*/

#include "btSliderConstraint.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"
#include "LinearMath/btTransformUtil.h"
#include <new>

#define USE_OFFSET_FOR_CONSTANT_FRAME true

void btSliderConstraint::initParams()
{
	m_lowerLinLimit = btScalar(1.0);
	m_upperLinLimit = btScalar(-1.0);
	m_lowerAngLimit = btScalar(0.);
	m_upperAngLimit = btScalar(0.);
	m_softnessDirLin = SLIDER_CONSTRAINT_DEF_SOFTNESS;
	m_restitutionDirLin = SLIDER_CONSTRAINT_DEF_RESTITUTION;
	m_dampingDirLin = btScalar(0.);
	m_cfmDirLin = SLIDER_CONSTRAINT_DEF_CFM;
	m_softnessDirAng = SLIDER_CONSTRAINT_DEF_SOFTNESS;
	m_restitutionDirAng = SLIDER_CONSTRAINT_DEF_RESTITUTION;
	m_dampingDirAng = btScalar(0.);
	m_cfmDirAng = SLIDER_CONSTRAINT_DEF_CFM;
	m_softnessOrthoLin = SLIDER_CONSTRAINT_DEF_SOFTNESS;
	m_restitutionOrthoLin = SLIDER_CONSTRAINT_DEF_RESTITUTION;
	m_dampingOrthoLin = SLIDER_CONSTRAINT_DEF_DAMPING;
	m_cfmOrthoLin = SLIDER_CONSTRAINT_DEF_CFM;
	m_softnessOrthoAng = SLIDER_CONSTRAINT_DEF_SOFTNESS;
	m_restitutionOrthoAng = SLIDER_CONSTRAINT_DEF_RESTITUTION;
	m_dampingOrthoAng = SLIDER_CONSTRAINT_DEF_DAMPING;
	m_cfmOrthoAng = SLIDER_CONSTRAINT_DEF_CFM;
	m_softnessLimLin = SLIDER_CONSTRAINT_DEF_SOFTNESS;
	m_restitutionLimLin = SLIDER_CONSTRAINT_DEF_RESTITUTION;
	m_dampingLimLin = SLIDER_CONSTRAINT_DEF_DAMPING;
	m_cfmLimLin = SLIDER_CONSTRAINT_DEF_CFM;
	m_softnessLimAng = SLIDER_CONSTRAINT_DEF_SOFTNESS;
	m_restitutionLimAng = SLIDER_CONSTRAINT_DEF_RESTITUTION;
	m_dampingLimAng = SLIDER_CONSTRAINT_DEF_DAMPING;
	m_cfmLimAng = SLIDER_CONSTRAINT_DEF_CFM;

	m_poweredLinMotor = false;
	m_targetLinMotorVelocity = btScalar(0.);
	m_maxLinMotorForce = btScalar(0.);
	m_accumulatedLinMotorImpulse = btScalar(0.0);

	m_poweredAngMotor = false;
	m_targetAngMotorVelocity = btScalar(0.);
	m_maxAngMotorForce = btScalar(0.);
	m_accumulatedAngMotorImpulse = btScalar(0.0);

	m_flags = 0;
	m_flags = 0;

	m_useOffsetForConstraintFrame = USE_OFFSET_FOR_CONSTANT_FRAME;

	calculateTransforms(m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform());
}

btSliderConstraint::btSliderConstraint(btRigidBody& rbA, btRigidBody& rbB, const btTransform& frameInA, const btTransform& frameInB, bool useLinearReferenceFrameA)
	: btTypedConstraint(SLIDER_CONSTRAINT_TYPE, rbA, rbB),
	  m_useSolveConstraintObsolete(false),
	  m_frameInA(frameInA),
	  m_frameInB(frameInB),
	  m_useLinearReferenceFrameA(useLinearReferenceFrameA)
{
	initParams();
}

btSliderConstraint::btSliderConstraint(btRigidBody& rbB, const btTransform& frameInB, bool useLinearReferenceFrameA)
	: btTypedConstraint(SLIDER_CONSTRAINT_TYPE, getFixedBody(), rbB),
	  m_useSolveConstraintObsolete(false),
	  m_frameInB(frameInB),
	  m_useLinearReferenceFrameA(useLinearReferenceFrameA)
{
	///not providing rigidbody A means implicitly using worldspace for body A
	m_frameInA = rbB.getCenterOfMassTransform() * m_frameInB;
	//	m_frameInA.getOrigin() = m_rbA.getCenterOfMassTransform()(m_frameInA.getOrigin());

	initParams();
}

void btSliderConstraint::getInfo1(btConstraintInfo1* info)
{
	if (m_useSolveConstraintObsolete)
	{
		info->m_numConstraintRows = 0;
		info->nub = 0;
	}
	else
	{
		info->m_numConstraintRows = 4;  // Fixed 2 linear + 2 angular
		info->nub = 2;
		//prepare constraint
		calculateTransforms(m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform());
		testAngLimits();
		testLinLimits();
		if (getSolveLinLimit() || getPoweredLinMotor())
		{
			info->m_numConstraintRows++;  // limit 3rd linear as well
			info->nub--;
		}
		if (getSolveAngLimit() || getPoweredAngMotor())
		{
			info->m_numConstraintRows++;  // limit 3rd angular as well
			info->nub--;
		}
	}
}

void btSliderConstraint::getInfo1NonVirtual(btConstraintInfo1* info)
{
	info->m_numConstraintRows = 6;  // Fixed 2 linear + 2 angular + 1 limit (even if not used)
	info->nub = 0;
}

void btSliderConstraint::getInfo2(btConstraintInfo2* info)
{
	getInfo2NonVirtual(info, m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform(), m_rbA.getLinearVelocity(), m_rbB.getLinearVelocity(), m_rbA.getInvMass(), m_rbB.getInvMass());
}

void btSliderConstraint::calculateTransforms(const btTransform& transA, const btTransform& transB)
{
	if (m_useLinearReferenceFrameA || (!m_useSolveConstraintObsolete))
	{
		m_calculatedTransformA = transA * m_frameInA;
		m_calculatedTransformB = transB * m_frameInB;
	}
	else
	{
		m_calculatedTransformA = transB * m_frameInB;
		m_calculatedTransformB = transA * m_frameInA;
	}
	m_realPivotAInW = m_calculatedTransformA.getOrigin();
	m_realPivotBInW = m_calculatedTransformB.getOrigin();
	m_sliderAxis = m_calculatedTransformA.getBasis().getColumn(0);  // along X
	if (m_useLinearReferenceFrameA || m_useSolveConstraintObsolete)
	{
		m_delta = m_realPivotBInW - m_realPivotAInW;
	}
	else
	{
		m_delta = m_realPivotAInW - m_realPivotBInW;
	}
	m_projPivotInW = m_realPivotAInW + m_sliderAxis.dot(m_delta) * m_sliderAxis;
	btVector3 normalWorld;
	int i;
	//linear part
	for (i = 0; i < 3; i++)
	{
		normalWorld = m_calculatedTransformA.getBasis().getColumn(i);
		m_depth[i] = m_delta.dot(normalWorld);
	}
}

void btSliderConstraint::testLinLimits(void)
{
	m_solveLinLim = false;
	m_linPos = m_depth[0];
	if (m_lowerLinLimit <= m_upperLinLimit)
	{
		if (m_depth[0] > m_upperLinLimit)
		{
			m_depth[0] -= m_upperLinLimit;
			m_solveLinLim = true;
		}
		else if (m_depth[0] < m_lowerLinLimit)
		{
			m_depth[0] -= m_lowerLinLimit;
			m_solveLinLim = true;
		}
		else
		{
			m_depth[0] = btScalar(0.);
		}
	}
	else
	{
		m_depth[0] = btScalar(0.);
	}
}

void btSliderConstraint::testAngLimits(void)
{
	m_angDepth = btScalar(0.);
	m_solveAngLim = false;
	if (m_lowerAngLimit <= m_upperAngLimit)
	{
		const btVector3 axisA0 = m_calculatedTransformA.getBasis().getColumn(1);
		const btVector3 axisA1 = m_calculatedTransformA.getBasis().getColumn(2);
		const btVector3 axisB0 = m_calculatedTransformB.getBasis().getColumn(1);
		//		btScalar rot = btAtan2Fast(axisB0.dot(axisA1), axisB0.dot(axisA0));
		btScalar rot = btAtan2(axisB0.dot(axisA1), axisB0.dot(axisA0));
		rot = btAdjustAngleToLimits(rot, m_lowerAngLimit, m_upperAngLimit);
		m_angPos = rot;
		if (rot < m_lowerAngLimit)
		{
			m_angDepth = rot - m_lowerAngLimit;
			m_solveAngLim = true;
		}
		else if (rot > m_upperAngLimit)
		{
			m_angDepth = rot - m_upperAngLimit;
			m_solveAngLim = true;
		}
	}
}

btVector3 btSliderConstraint::getAncorInA(void)
{
	btVector3 ancorInA;
	ancorInA = m_realPivotAInW + (m_lowerLinLimit + m_upperLinLimit) * btScalar(0.5) * m_sliderAxis;
	ancorInA = m_rbA.getCenterOfMassTransform().inverse() * ancorInA;
	return ancorInA;
}

btVector3 btSliderConstraint::getAncorInB(void)
{
	btVector3 ancorInB;
	ancorInB = m_frameInB.getOrigin();
	return ancorInB;
}

void btSliderConstraint::getInfo2NonVirtual(btConstraintInfo2* info, const btTransform& transA, const btTransform& transB, const btVector3& linVelA, const btVector3& linVelB, btScalar rbAinvMass, btScalar rbBinvMass)
{
	const btTransform& trA = getCalculatedTransformA();
	const btTransform& trB = getCalculatedTransformB();

	btAssert(!m_useSolveConstraintObsolete);
	int i, s = info->rowskip;

	btScalar signFact = m_useLinearReferenceFrameA ? btScalar(1.0f) : btScalar(-1.0f);

	// difference between frames in WCS
	btVector3 ofs = trB.getOrigin() - trA.getOrigin();
	// now get weight factors depending on masses
	btScalar miA = rbAinvMass;
	btScalar miB = rbBinvMass;
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
	btVector3 ax1, p, q;
	btVector3 ax1A = trA.getBasis().getColumn(0);
	btVector3 ax1B = trB.getBasis().getColumn(0);
	if (m_useOffsetForConstraintFrame)
	{
		// get the desired direction of slider axis
		// as weighted sum of X-orthos of frameA and frameB in WCS
		ax1 = ax1A * factA + ax1B * factB;
		ax1.normalize();
		// construct two orthos to slider axis
		btPlaneSpace1(ax1, p, q);
	}
	else
	{  // old way - use frameA
		ax1 = trA.getBasis().getColumn(0);
		// get 2 orthos to slider axis (Y, Z)
		p = trA.getBasis().getColumn(1);
		q = trA.getBasis().getColumn(2);
	}
	// make rotations around these orthos equal
	// the slider axis should be the only unconstrained
	// rotational axis, the angular velocity of the two bodies perpendicular to
	// the slider axis should be equal. thus the constraint equations are
	//    p*w1 - p*w2 = 0
	//    q*w1 - q*w2 = 0
	// where p and q are unit vectors normal to the slider axis, and w1 and w2
	// are the angular velocity vectors of the two bodies.
	info->m_J1angularAxis[0] = p[0];
	info->m_J1angularAxis[1] = p[1];
	info->m_J1angularAxis[2] = p[2];
	info->m_J1angularAxis[s + 0] = q[0];
	info->m_J1angularAxis[s + 1] = q[1];
	info->m_J1angularAxis[s + 2] = q[2];

	info->m_J2angularAxis[0] = -p[0];
	info->m_J2angularAxis[1] = -p[1];
	info->m_J2angularAxis[2] = -p[2];
	info->m_J2angularAxis[s + 0] = -q[0];
	info->m_J2angularAxis[s + 1] = -q[1];
	info->m_J2angularAxis[s + 2] = -q[2];
	// compute the right hand side of the constraint equation. set relative
	// body velocities along p and q to bring the slider back into alignment.
	// if ax1A,ax1B are the unit length slider axes as computed from bodyA and
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
	//	btScalar k = info->fps * info->erp * getSoftnessOrthoAng();
	btScalar currERP = (m_flags & BT_SLIDER_FLAGS_ERP_ORTANG) ? m_softnessOrthoAng : m_softnessOrthoAng * info->erp;
	btScalar k = info->fps * currERP;

	btVector3 u = ax1A.cross(ax1B);
	info->m_constraintError[0] = k * u.dot(p);
	info->m_constraintError[s] = k * u.dot(q);
	if (m_flags & BT_SLIDER_FLAGS_CFM_ORTANG)
	{
		info->cfm[0] = m_cfmOrthoAng;
		info->cfm[s] = m_cfmOrthoAng;
	}

	int nrow = 1;  // last filled row
	int srow;
	btScalar limit_err;
	int limit;

	// next two rows.
	// we want: velA + wA x relA == velB + wB x relB ... but this would
	// result in three equations, so we project along two orthos to the slider axis

	btTransform bodyA_trans = transA;
	btTransform bodyB_trans = transB;
	nrow++;
	int s2 = nrow * s;
	nrow++;
	int s3 = nrow * s;
	btVector3 tmpA(0, 0, 0), tmpB(0, 0, 0), relA(0, 0, 0), relB(0, 0, 0), c(0, 0, 0);
	if (m_useOffsetForConstraintFrame)
	{
		// get vector from bodyB to frameB in WCS
		relB = trB.getOrigin() - bodyB_trans.getOrigin();
		// get its projection to slider axis
		btVector3 projB = ax1 * relB.dot(ax1);
		// get vector directed from bodyB to slider axis (and orthogonal to it)
		btVector3 orthoB = relB - projB;
		// same for bodyA
		relA = trA.getOrigin() - bodyA_trans.getOrigin();
		btVector3 projA = ax1 * relA.dot(ax1);
		btVector3 orthoA = relA - projA;
		// get desired offset between frames A and B along slider axis
		btScalar sliderOffs = m_linPos - m_depth[0];
		// desired vector from projection of center of bodyA to projection of center of bodyB to slider axis
		btVector3 totalDist = projA + ax1 * sliderOffs - projB;
		// get offset vectors relA and relB
		relA = orthoA + totalDist * factA;
		relB = orthoB - totalDist * factB;
		// now choose average ortho to slider axis
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
		// fill two rows
		tmpA = relA.cross(p);
		tmpB = relB.cross(p);
		for (i = 0; i < 3; i++) info->m_J1angularAxis[s2 + i] = tmpA[i];
		for (i = 0; i < 3; i++) info->m_J2angularAxis[s2 + i] = -tmpB[i];
		tmpA = relA.cross(q);
		tmpB = relB.cross(q);
		if (hasStaticBody && getSolveAngLimit())
		{  // to make constraint between static and dynamic objects more rigid
			// remove wA (or wB) from equation if angular limit is hit
			tmpB *= factB;
			tmpA *= factA;
		}
		for (i = 0; i < 3; i++) info->m_J1angularAxis[s3 + i] = tmpA[i];
		for (i = 0; i < 3; i++) info->m_J2angularAxis[s3 + i] = -tmpB[i];
		for (i = 0; i < 3; i++) info->m_J1linearAxis[s2 + i] = p[i];
		for (i = 0; i < 3; i++) info->m_J1linearAxis[s3 + i] = q[i];
		for (i = 0; i < 3; i++) info->m_J2linearAxis[s2 + i] = -p[i];
		for (i = 0; i < 3; i++) info->m_J2linearAxis[s3 + i] = -q[i];
	}
	else
	{  // old way - maybe incorrect if bodies are not on the slider axis
		// see discussion "Bug in slider constraint" http://bulletphysics.org/Bullet/phpBB3/viewtopic.php?f=9&t=4024&start=0
		c = bodyB_trans.getOrigin() - bodyA_trans.getOrigin();
		btVector3 tmp = c.cross(p);
		for (i = 0; i < 3; i++) info->m_J1angularAxis[s2 + i] = factA * tmp[i];
		for (i = 0; i < 3; i++) info->m_J2angularAxis[s2 + i] = factB * tmp[i];
		tmp = c.cross(q);
		for (i = 0; i < 3; i++) info->m_J1angularAxis[s3 + i] = factA * tmp[i];
		for (i = 0; i < 3; i++) info->m_J2angularAxis[s3 + i] = factB * tmp[i];

		for (i = 0; i < 3; i++) info->m_J1linearAxis[s2 + i] = p[i];
		for (i = 0; i < 3; i++) info->m_J1linearAxis[s3 + i] = q[i];
		for (i = 0; i < 3; i++) info->m_J2linearAxis[s2 + i] = -p[i];
		for (i = 0; i < 3; i++) info->m_J2linearAxis[s3 + i] = -q[i];
	}
	// compute two elements of right hand side

	//	k = info->fps * info->erp * getSoftnessOrthoLin();
	currERP = (m_flags & BT_SLIDER_FLAGS_ERP_ORTLIN) ? m_softnessOrthoLin : m_softnessOrthoLin * info->erp;
	k = info->fps * currERP;

	btScalar rhs = k * p.dot(ofs);
	info->m_constraintError[s2] = rhs;
	rhs = k * q.dot(ofs);
	info->m_constraintError[s3] = rhs;
	if (m_flags & BT_SLIDER_FLAGS_CFM_ORTLIN)
	{
		info->cfm[s2] = m_cfmOrthoLin;
		info->cfm[s3] = m_cfmOrthoLin;
	}

	// check linear limits
	limit_err = btScalar(0.0);
	limit = 0;
	if (getSolveLinLimit())
	{
		limit_err = getLinDepth() * signFact;
		limit = (limit_err > btScalar(0.0)) ? 2 : 1;
	}
	bool powered = getPoweredLinMotor();
	// if the slider has joint limits or motor, add in the extra row
	if (limit || powered)
	{
		nrow++;
		srow = nrow * info->rowskip;
		info->m_J1linearAxis[srow + 0] = ax1[0];
		info->m_J1linearAxis[srow + 1] = ax1[1];
		info->m_J1linearAxis[srow + 2] = ax1[2];
		info->m_J2linearAxis[srow + 0] = -ax1[0];
		info->m_J2linearAxis[srow + 1] = -ax1[1];
		info->m_J2linearAxis[srow + 2] = -ax1[2];
		// linear torque decoupling step:
		//
		// we have to be careful that the linear constraint forces (+/- ax1) applied to the two bodies
		// do not create a torque couple. in other words, the points that the
		// constraint force is applied at must lie along the same ax1 axis.
		// a torque couple will result in limited slider-jointed free
		// bodies from gaining angular momentum.
		if (m_useOffsetForConstraintFrame)
		{
			// this is needed only when bodyA and bodyB are both dynamic.
			if (!hasStaticBody)
			{
				tmpA = relA.cross(ax1);
				tmpB = relB.cross(ax1);
				info->m_J1angularAxis[srow + 0] = tmpA[0];
				info->m_J1angularAxis[srow + 1] = tmpA[1];
				info->m_J1angularAxis[srow + 2] = tmpA[2];
				info->m_J2angularAxis[srow + 0] = -tmpB[0];
				info->m_J2angularAxis[srow + 1] = -tmpB[1];
				info->m_J2angularAxis[srow + 2] = -tmpB[2];
			}
		}
		else
		{                   // The old way. May be incorrect if bodies are not on the slider axis
			btVector3 ltd;  // Linear Torque Decoupling vector (a torque)
			ltd = c.cross(ax1);
			info->m_J1angularAxis[srow + 0] = factA * ltd[0];
			info->m_J1angularAxis[srow + 1] = factA * ltd[1];
			info->m_J1angularAxis[srow + 2] = factA * ltd[2];
			info->m_J2angularAxis[srow + 0] = factB * ltd[0];
			info->m_J2angularAxis[srow + 1] = factB * ltd[1];
			info->m_J2angularAxis[srow + 2] = factB * ltd[2];
		}
		// right-hand part
		btScalar lostop = getLowerLinLimit();
		btScalar histop = getUpperLinLimit();
		if (limit && (lostop == histop))
		{  // the joint motor is ineffective
			powered = false;
		}
		info->m_constraintError[srow] = 0.;
		info->m_lowerLimit[srow] = 0.;
		info->m_upperLimit[srow] = 0.;
		currERP = (m_flags & BT_SLIDER_FLAGS_ERP_LIMLIN) ? m_softnessLimLin : info->erp;
		if (powered)
		{
			if (m_flags & BT_SLIDER_FLAGS_CFM_DIRLIN)
			{
				info->cfm[srow] = m_cfmDirLin;
			}
			btScalar tag_vel = getTargetLinMotorVelocity();
			btScalar mot_fact = getMotorFactor(m_linPos, m_lowerLinLimit, m_upperLinLimit, tag_vel, info->fps * currERP);
			info->m_constraintError[srow] -= signFact * mot_fact * getTargetLinMotorVelocity();
			info->m_lowerLimit[srow] += -getMaxLinMotorForce() / info->fps;
			info->m_upperLimit[srow] += getMaxLinMotorForce() / info->fps;
		}
		if (limit)
		{
			k = info->fps * currERP;
			info->m_constraintError[srow] += k * limit_err;
			if (m_flags & BT_SLIDER_FLAGS_CFM_LIMLIN)
			{
				info->cfm[srow] = m_cfmLimLin;
			}
			if (lostop == histop)
			{  // limited low and high simultaneously
				info->m_lowerLimit[srow] = -SIMD_INFINITY;
				info->m_upperLimit[srow] = SIMD_INFINITY;
			}
			else if (limit == 1)
			{  // low limit
				info->m_lowerLimit[srow] = -SIMD_INFINITY;
				info->m_upperLimit[srow] = 0;
			}
			else
			{  // high limit
				info->m_lowerLimit[srow] = 0;
				info->m_upperLimit[srow] = SIMD_INFINITY;
			}
			// bounce (we'll use slider parameter abs(1.0 - m_dampingLimLin) for that)
			btScalar bounce = btFabs(btScalar(1.0) - getDampingLimLin());
			if (bounce > btScalar(0.0))
			{
				btScalar vel = linVelA.dot(ax1);
				vel -= linVelB.dot(ax1);
				vel *= signFact;
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
			info->m_constraintError[srow] *= getSoftnessLimLin();
		}  // if(limit)
	}      // if linear limit
	// check angular limits
	limit_err = btScalar(0.0);
	limit = 0;
	if (getSolveAngLimit())
	{
		limit_err = getAngDepth();
		limit = (limit_err > btScalar(0.0)) ? 1 : 2;
	}
	// if the slider has joint limits, add in the extra row
	powered = getPoweredAngMotor();
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

		btScalar lostop = getLowerAngLimit();
		btScalar histop = getUpperAngLimit();
		if (limit && (lostop == histop))
		{  // the joint motor is ineffective
			powered = false;
		}
		currERP = (m_flags & BT_SLIDER_FLAGS_ERP_LIMANG) ? m_softnessLimAng : info->erp;
		if (powered)
		{
			if (m_flags & BT_SLIDER_FLAGS_CFM_DIRANG)
			{
				info->cfm[srow] = m_cfmDirAng;
			}
			btScalar mot_fact = getMotorFactor(m_angPos, m_lowerAngLimit, m_upperAngLimit, getTargetAngMotorVelocity(), info->fps * currERP);
			info->m_constraintError[srow] = mot_fact * getTargetAngMotorVelocity();
			info->m_lowerLimit[srow] = -getMaxAngMotorForce() / info->fps;
			info->m_upperLimit[srow] = getMaxAngMotorForce() / info->fps;
		}
		if (limit)
		{
			k = info->fps * currERP;
			info->m_constraintError[srow] += k * limit_err;
			if (m_flags & BT_SLIDER_FLAGS_CFM_LIMANG)
			{
				info->cfm[srow] = m_cfmLimAng;
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
			btScalar bounce = btFabs(btScalar(1.0) - getDampingLimAng());
			if (bounce > btScalar(0.0))
			{
				btScalar vel = m_rbA.getAngularVelocity().dot(ax1);
				vel -= m_rbB.getAngularVelocity().dot(ax1);
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
			info->m_constraintError[srow] *= getSoftnessLimAng();
		}  // if(limit)
	}      // if angular limit or powered
}

///override the default global value of a parameter (such as ERP or CFM), optionally provide the axis (0..5).
///If no axis is provided, it uses the default axis for this constraint.
void btSliderConstraint::setParam(int num, btScalar value, int axis)
{
	switch (num)
	{
		case BT_CONSTRAINT_STOP_ERP:
			if (axis < 1)
			{
				m_softnessLimLin = value;
				m_flags |= BT_SLIDER_FLAGS_ERP_LIMLIN;
			}
			else if (axis < 3)
			{
				m_softnessOrthoLin = value;
				m_flags |= BT_SLIDER_FLAGS_ERP_ORTLIN;
			}
			else if (axis == 3)
			{
				m_softnessLimAng = value;
				m_flags |= BT_SLIDER_FLAGS_ERP_LIMANG;
			}
			else if (axis < 6)
			{
				m_softnessOrthoAng = value;
				m_flags |= BT_SLIDER_FLAGS_ERP_ORTANG;
			}
			else
			{
				btAssertConstrParams(0);
			}
			break;
		case BT_CONSTRAINT_CFM:
			if (axis < 1)
			{
				m_cfmDirLin = value;
				m_flags |= BT_SLIDER_FLAGS_CFM_DIRLIN;
			}
			else if (axis == 3)
			{
				m_cfmDirAng = value;
				m_flags |= BT_SLIDER_FLAGS_CFM_DIRANG;
			}
			else
			{
				btAssertConstrParams(0);
			}
			break;
		case BT_CONSTRAINT_STOP_CFM:
			if (axis < 1)
			{
				m_cfmLimLin = value;
				m_flags |= BT_SLIDER_FLAGS_CFM_LIMLIN;
			}
			else if (axis < 3)
			{
				m_cfmOrthoLin = value;
				m_flags |= BT_SLIDER_FLAGS_CFM_ORTLIN;
			}
			else if (axis == 3)
			{
				m_cfmLimAng = value;
				m_flags |= BT_SLIDER_FLAGS_CFM_LIMANG;
			}
			else if (axis < 6)
			{
				m_cfmOrthoAng = value;
				m_flags |= BT_SLIDER_FLAGS_CFM_ORTANG;
			}
			else
			{
				btAssertConstrParams(0);
			}
			break;
	}
}

///return the local value of parameter
btScalar btSliderConstraint::getParam(int num, int axis) const
{
	btScalar retVal(SIMD_INFINITY);
	switch (num)
	{
		case BT_CONSTRAINT_STOP_ERP:
			if (axis < 1)
			{
				btAssertConstrParams(m_flags & BT_SLIDER_FLAGS_ERP_LIMLIN);
				retVal = m_softnessLimLin;
			}
			else if (axis < 3)
			{
				btAssertConstrParams(m_flags & BT_SLIDER_FLAGS_ERP_ORTLIN);
				retVal = m_softnessOrthoLin;
			}
			else if (axis == 3)
			{
				btAssertConstrParams(m_flags & BT_SLIDER_FLAGS_ERP_LIMANG);
				retVal = m_softnessLimAng;
			}
			else if (axis < 6)
			{
				btAssertConstrParams(m_flags & BT_SLIDER_FLAGS_ERP_ORTANG);
				retVal = m_softnessOrthoAng;
			}
			else
			{
				btAssertConstrParams(0);
			}
			break;
		case BT_CONSTRAINT_CFM:
			if (axis < 1)
			{
				btAssertConstrParams(m_flags & BT_SLIDER_FLAGS_CFM_DIRLIN);
				retVal = m_cfmDirLin;
			}
			else if (axis == 3)
			{
				btAssertConstrParams(m_flags & BT_SLIDER_FLAGS_CFM_DIRANG);
				retVal = m_cfmDirAng;
			}
			else
			{
				btAssertConstrParams(0);
			}
			break;
		case BT_CONSTRAINT_STOP_CFM:
			if (axis < 1)
			{
				btAssertConstrParams(m_flags & BT_SLIDER_FLAGS_CFM_LIMLIN);
				retVal = m_cfmLimLin;
			}
			else if (axis < 3)
			{
				btAssertConstrParams(m_flags & BT_SLIDER_FLAGS_CFM_ORTLIN);
				retVal = m_cfmOrthoLin;
			}
			else if (axis == 3)
			{
				btAssertConstrParams(m_flags & BT_SLIDER_FLAGS_CFM_LIMANG);
				retVal = m_cfmLimAng;
			}
			else if (axis < 6)
			{
				btAssertConstrParams(m_flags & BT_SLIDER_FLAGS_CFM_ORTANG);
				retVal = m_cfmOrthoAng;
			}
			else
			{
				btAssertConstrParams(0);
			}
			break;
	}
	return retVal;
}
