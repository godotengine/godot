/*
Bullet Continuous Collision Detection and Physics Library
btConeTwistConstraint is Copyright (c) 2007 Starbreeze Studios

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

Written by: Marcus Hennix
*/

#include "btConeTwistConstraint.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"
#include "LinearMath/btTransformUtil.h"
#include "LinearMath/btMinMax.h"
#include <cmath>
#include <new>

//#define CONETWIST_USE_OBSOLETE_SOLVER true
#define CONETWIST_USE_OBSOLETE_SOLVER false
#define CONETWIST_DEF_FIX_THRESH btScalar(.05f)

SIMD_FORCE_INLINE btScalar computeAngularImpulseDenominator(const btVector3& axis, const btMatrix3x3& invInertiaWorld)
{
	btVector3 vec = axis * invInertiaWorld;
	return axis.dot(vec);
}

btConeTwistConstraint::btConeTwistConstraint(btRigidBody& rbA, btRigidBody& rbB,
											 const btTransform& rbAFrame, const btTransform& rbBFrame)
	: btTypedConstraint(CONETWIST_CONSTRAINT_TYPE, rbA, rbB), m_rbAFrame(rbAFrame), m_rbBFrame(rbBFrame), m_angularOnly(false), m_useSolveConstraintObsolete(CONETWIST_USE_OBSOLETE_SOLVER)
{
	init();
}

btConeTwistConstraint::btConeTwistConstraint(btRigidBody& rbA, const btTransform& rbAFrame)
	: btTypedConstraint(CONETWIST_CONSTRAINT_TYPE, rbA), m_rbAFrame(rbAFrame), m_angularOnly(false), m_useSolveConstraintObsolete(CONETWIST_USE_OBSOLETE_SOLVER)
{
	m_rbBFrame = m_rbAFrame;
	m_rbBFrame.setOrigin(btVector3(0., 0., 0.));
	init();
}

void btConeTwistConstraint::init()
{
	m_angularOnly = false;
	m_solveTwistLimit = false;
	m_solveSwingLimit = false;
	m_bMotorEnabled = false;
	m_maxMotorImpulse = btScalar(-1);

	setLimit(btScalar(BT_LARGE_FLOAT), btScalar(BT_LARGE_FLOAT), btScalar(BT_LARGE_FLOAT));
	m_damping = btScalar(0.01);
	m_fixThresh = CONETWIST_DEF_FIX_THRESH;
	m_flags = 0;
	m_linCFM = btScalar(0.f);
	m_linERP = btScalar(0.7f);
	m_angCFM = btScalar(0.f);
}

void btConeTwistConstraint::getInfo1(btConstraintInfo1* info)
{
	if (m_useSolveConstraintObsolete)
	{
		info->m_numConstraintRows = 0;
		info->nub = 0;
	}
	else
	{
		info->m_numConstraintRows = 3;
		info->nub = 3;
		calcAngleInfo2(m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform(), m_rbA.getInvInertiaTensorWorld(), m_rbB.getInvInertiaTensorWorld());
		if (m_solveSwingLimit)
		{
			info->m_numConstraintRows++;
			info->nub--;
			if ((m_swingSpan1 < m_fixThresh) && (m_swingSpan2 < m_fixThresh))
			{
				info->m_numConstraintRows++;
				info->nub--;
			}
		}
		if (m_solveTwistLimit)
		{
			info->m_numConstraintRows++;
			info->nub--;
		}
	}
}

void btConeTwistConstraint::getInfo1NonVirtual(btConstraintInfo1* info)
{
	//always reserve 6 rows: object transform is not available on SPU
	info->m_numConstraintRows = 6;
	info->nub = 0;
}

void btConeTwistConstraint::getInfo2(btConstraintInfo2* info)
{
	getInfo2NonVirtual(info, m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform(), m_rbA.getInvInertiaTensorWorld(), m_rbB.getInvInertiaTensorWorld());
}

void btConeTwistConstraint::getInfo2NonVirtual(btConstraintInfo2* info, const btTransform& transA, const btTransform& transB, const btMatrix3x3& invInertiaWorldA, const btMatrix3x3& invInertiaWorldB)
{
	calcAngleInfo2(transA, transB, invInertiaWorldA, invInertiaWorldB);

	btAssert(!m_useSolveConstraintObsolete);
	// set jacobian
	info->m_J1linearAxis[0] = 1;
	info->m_J1linearAxis[info->rowskip + 1] = 1;
	info->m_J1linearAxis[2 * info->rowskip + 2] = 1;
	btVector3 a1 = transA.getBasis() * m_rbAFrame.getOrigin();
	{
		btVector3* angular0 = (btVector3*)(info->m_J1angularAxis);
		btVector3* angular1 = (btVector3*)(info->m_J1angularAxis + info->rowskip);
		btVector3* angular2 = (btVector3*)(info->m_J1angularAxis + 2 * info->rowskip);
		btVector3 a1neg = -a1;
		a1neg.getSkewSymmetricMatrix(angular0, angular1, angular2);
	}
	info->m_J2linearAxis[0] = -1;
	info->m_J2linearAxis[info->rowskip + 1] = -1;
	info->m_J2linearAxis[2 * info->rowskip + 2] = -1;
	btVector3 a2 = transB.getBasis() * m_rbBFrame.getOrigin();
	{
		btVector3* angular0 = (btVector3*)(info->m_J2angularAxis);
		btVector3* angular1 = (btVector3*)(info->m_J2angularAxis + info->rowskip);
		btVector3* angular2 = (btVector3*)(info->m_J2angularAxis + 2 * info->rowskip);
		a2.getSkewSymmetricMatrix(angular0, angular1, angular2);
	}
	// set right hand side
	btScalar linERP = (m_flags & BT_CONETWIST_FLAGS_LIN_ERP) ? m_linERP : info->erp;
	btScalar k = info->fps * linERP;
	int j;
	for (j = 0; j < 3; j++)
	{
		info->m_constraintError[j * info->rowskip] = k * (a2[j] + transB.getOrigin()[j] - a1[j] - transA.getOrigin()[j]);
		info->m_lowerLimit[j * info->rowskip] = -SIMD_INFINITY;
		info->m_upperLimit[j * info->rowskip] = SIMD_INFINITY;
		if (m_flags & BT_CONETWIST_FLAGS_LIN_CFM)
		{
			info->cfm[j * info->rowskip] = m_linCFM;
		}
	}
	int row = 3;
	int srow = row * info->rowskip;
	btVector3 ax1;
	// angular limits
	if (m_solveSwingLimit)
	{
		btScalar* J1 = info->m_J1angularAxis;
		btScalar* J2 = info->m_J2angularAxis;
		if ((m_swingSpan1 < m_fixThresh) && (m_swingSpan2 < m_fixThresh))
		{
			btTransform trA = transA * m_rbAFrame;
			btVector3 p = trA.getBasis().getColumn(1);
			btVector3 q = trA.getBasis().getColumn(2);
			int srow1 = srow + info->rowskip;
			J1[srow + 0] = p[0];
			J1[srow + 1] = p[1];
			J1[srow + 2] = p[2];
			J1[srow1 + 0] = q[0];
			J1[srow1 + 1] = q[1];
			J1[srow1 + 2] = q[2];
			J2[srow + 0] = -p[0];
			J2[srow + 1] = -p[1];
			J2[srow + 2] = -p[2];
			J2[srow1 + 0] = -q[0];
			J2[srow1 + 1] = -q[1];
			J2[srow1 + 2] = -q[2];
			btScalar fact = info->fps * m_relaxationFactor;
			info->m_constraintError[srow] = fact * m_swingAxis.dot(p);
			info->m_constraintError[srow1] = fact * m_swingAxis.dot(q);
			info->m_lowerLimit[srow] = -SIMD_INFINITY;
			info->m_upperLimit[srow] = SIMD_INFINITY;
			info->m_lowerLimit[srow1] = -SIMD_INFINITY;
			info->m_upperLimit[srow1] = SIMD_INFINITY;
			srow = srow1 + info->rowskip;
		}
		else
		{
			ax1 = m_swingAxis * m_relaxationFactor * m_relaxationFactor;
			J1[srow + 0] = ax1[0];
			J1[srow + 1] = ax1[1];
			J1[srow + 2] = ax1[2];
			J2[srow + 0] = -ax1[0];
			J2[srow + 1] = -ax1[1];
			J2[srow + 2] = -ax1[2];
			btScalar k = info->fps * m_biasFactor;

			info->m_constraintError[srow] = k * m_swingCorrection;
			if (m_flags & BT_CONETWIST_FLAGS_ANG_CFM)
			{
				info->cfm[srow] = m_angCFM;
			}
			// m_swingCorrection is always positive or 0
			info->m_lowerLimit[srow] = 0;
			info->m_upperLimit[srow] = (m_bMotorEnabled && m_maxMotorImpulse >= 0.0f) ? m_maxMotorImpulse : SIMD_INFINITY;
			srow += info->rowskip;
		}
	}
	if (m_solveTwistLimit)
	{
		ax1 = m_twistAxis * m_relaxationFactor * m_relaxationFactor;
		btScalar* J1 = info->m_J1angularAxis;
		btScalar* J2 = info->m_J2angularAxis;
		J1[srow + 0] = ax1[0];
		J1[srow + 1] = ax1[1];
		J1[srow + 2] = ax1[2];
		J2[srow + 0] = -ax1[0];
		J2[srow + 1] = -ax1[1];
		J2[srow + 2] = -ax1[2];
		btScalar k = info->fps * m_biasFactor;
		info->m_constraintError[srow] = k * m_twistCorrection;
		if (m_flags & BT_CONETWIST_FLAGS_ANG_CFM)
		{
			info->cfm[srow] = m_angCFM;
		}
		if (m_twistSpan > 0.0f)
		{
			if (m_twistCorrection > 0.0f)
			{
				info->m_lowerLimit[srow] = 0;
				info->m_upperLimit[srow] = SIMD_INFINITY;
			}
			else
			{
				info->m_lowerLimit[srow] = -SIMD_INFINITY;
				info->m_upperLimit[srow] = 0;
			}
		}
		else
		{
			info->m_lowerLimit[srow] = -SIMD_INFINITY;
			info->m_upperLimit[srow] = SIMD_INFINITY;
		}
		srow += info->rowskip;
	}
}

void btConeTwistConstraint::buildJacobian()
{
	if (m_useSolveConstraintObsolete)
	{
		m_appliedImpulse = btScalar(0.);
		m_accTwistLimitImpulse = btScalar(0.);
		m_accSwingLimitImpulse = btScalar(0.);
		m_accMotorImpulse = btVector3(0., 0., 0.);

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

		calcAngleInfo2(m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform(), m_rbA.getInvInertiaTensorWorld(), m_rbB.getInvInertiaTensorWorld());
	}
}

void btConeTwistConstraint::solveConstraintObsolete(btSolverBody& bodyA, btSolverBody& bodyB, btScalar timeStep)
{
#ifndef __SPU__
	if (m_useSolveConstraintObsolete)
	{
		btVector3 pivotAInW = m_rbA.getCenterOfMassTransform() * m_rbAFrame.getOrigin();
		btVector3 pivotBInW = m_rbB.getCenterOfMassTransform() * m_rbBFrame.getOrigin();

		btScalar tau = btScalar(0.3);

		//linear part
		if (!m_angularOnly)
		{
			btVector3 rel_pos1 = pivotAInW - m_rbA.getCenterOfMassPosition();
			btVector3 rel_pos2 = pivotBInW - m_rbB.getCenterOfMassPosition();

			btVector3 vel1;
			bodyA.internalGetVelocityInLocalPointObsolete(rel_pos1, vel1);
			btVector3 vel2;
			bodyB.internalGetVelocityInLocalPointObsolete(rel_pos2, vel2);
			btVector3 vel = vel1 - vel2;

			for (int i = 0; i < 3; i++)
			{
				const btVector3& normal = m_jac[i].m_linearJointAxis;
				btScalar jacDiagABInv = btScalar(1.) / m_jac[i].getDiagonal();

				btScalar rel_vel;
				rel_vel = normal.dot(vel);
				//positional error (zeroth order error)
				btScalar depth = -(pivotAInW - pivotBInW).dot(normal);  //this is the error projected on the normal
				btScalar impulse = depth * tau / timeStep * jacDiagABInv - rel_vel * jacDiagABInv;
				m_appliedImpulse += impulse;

				btVector3 ftorqueAxis1 = rel_pos1.cross(normal);
				btVector3 ftorqueAxis2 = rel_pos2.cross(normal);
				bodyA.internalApplyImpulse(normal * m_rbA.getInvMass(), m_rbA.getInvInertiaTensorWorld() * ftorqueAxis1, impulse);
				bodyB.internalApplyImpulse(normal * m_rbB.getInvMass(), m_rbB.getInvInertiaTensorWorld() * ftorqueAxis2, -impulse);
			}
		}

		// apply motor
		if (m_bMotorEnabled)
		{
			// compute current and predicted transforms
			btTransform trACur = m_rbA.getCenterOfMassTransform();
			btTransform trBCur = m_rbB.getCenterOfMassTransform();
			btVector3 omegaA;
			bodyA.internalGetAngularVelocity(omegaA);
			btVector3 omegaB;
			bodyB.internalGetAngularVelocity(omegaB);
			btTransform trAPred;
			trAPred.setIdentity();
			btVector3 zerovec(0, 0, 0);
			btTransformUtil::integrateTransform(
				trACur, zerovec, omegaA, timeStep, trAPred);
			btTransform trBPred;
			trBPred.setIdentity();
			btTransformUtil::integrateTransform(
				trBCur, zerovec, omegaB, timeStep, trBPred);

			// compute desired transforms in world
			btTransform trPose(m_qTarget);
			btTransform trABDes = m_rbBFrame * trPose * m_rbAFrame.inverse();
			btTransform trADes = trBPred * trABDes;
			btTransform trBDes = trAPred * trABDes.inverse();

			// compute desired omegas in world
			btVector3 omegaADes, omegaBDes;

			btTransformUtil::calculateVelocity(trACur, trADes, timeStep, zerovec, omegaADes);
			btTransformUtil::calculateVelocity(trBCur, trBDes, timeStep, zerovec, omegaBDes);

			// compute delta omegas
			btVector3 dOmegaA = omegaADes - omegaA;
			btVector3 dOmegaB = omegaBDes - omegaB;

			// compute weighted avg axis of dOmega (weighting based on inertias)
			btVector3 axisA, axisB;
			btScalar kAxisAInv = 0, kAxisBInv = 0;

			if (dOmegaA.length2() > SIMD_EPSILON)
			{
				axisA = dOmegaA.normalized();
				kAxisAInv = getRigidBodyA().computeAngularImpulseDenominator(axisA);
			}

			if (dOmegaB.length2() > SIMD_EPSILON)
			{
				axisB = dOmegaB.normalized();
				kAxisBInv = getRigidBodyB().computeAngularImpulseDenominator(axisB);
			}

			btVector3 avgAxis = kAxisAInv * axisA + kAxisBInv * axisB;

			static bool bDoTorque = true;
			if (bDoTorque && avgAxis.length2() > SIMD_EPSILON)
			{
				avgAxis.normalize();
				kAxisAInv = getRigidBodyA().computeAngularImpulseDenominator(avgAxis);
				kAxisBInv = getRigidBodyB().computeAngularImpulseDenominator(avgAxis);
				btScalar kInvCombined = kAxisAInv + kAxisBInv;

				btVector3 impulse = (kAxisAInv * dOmegaA - kAxisBInv * dOmegaB) /
									(kInvCombined * kInvCombined);

				if (m_maxMotorImpulse >= 0)
				{
					btScalar fMaxImpulse = m_maxMotorImpulse;
					if (m_bNormalizedMotorStrength)
						fMaxImpulse = fMaxImpulse / kAxisAInv;

					btVector3 newUnclampedAccImpulse = m_accMotorImpulse + impulse;
					btScalar newUnclampedMag = newUnclampedAccImpulse.length();
					if (newUnclampedMag > fMaxImpulse)
					{
						newUnclampedAccImpulse.normalize();
						newUnclampedAccImpulse *= fMaxImpulse;
						impulse = newUnclampedAccImpulse - m_accMotorImpulse;
					}
					m_accMotorImpulse += impulse;
				}

				btScalar impulseMag = impulse.length();
				btVector3 impulseAxis = impulse / impulseMag;

				bodyA.internalApplyImpulse(btVector3(0, 0, 0), m_rbA.getInvInertiaTensorWorld() * impulseAxis, impulseMag);
				bodyB.internalApplyImpulse(btVector3(0, 0, 0), m_rbB.getInvInertiaTensorWorld() * impulseAxis, -impulseMag);
			}
		}
		else if (m_damping > SIMD_EPSILON)  // no motor: do a little damping
		{
			btVector3 angVelA;
			bodyA.internalGetAngularVelocity(angVelA);
			btVector3 angVelB;
			bodyB.internalGetAngularVelocity(angVelB);
			btVector3 relVel = angVelB - angVelA;
			if (relVel.length2() > SIMD_EPSILON)
			{
				btVector3 relVelAxis = relVel.normalized();
				btScalar m_kDamping = btScalar(1.) /
									  (getRigidBodyA().computeAngularImpulseDenominator(relVelAxis) +
									   getRigidBodyB().computeAngularImpulseDenominator(relVelAxis));
				btVector3 impulse = m_damping * m_kDamping * relVel;

				btScalar impulseMag = impulse.length();
				btVector3 impulseAxis = impulse / impulseMag;
				bodyA.internalApplyImpulse(btVector3(0, 0, 0), m_rbA.getInvInertiaTensorWorld() * impulseAxis, impulseMag);
				bodyB.internalApplyImpulse(btVector3(0, 0, 0), m_rbB.getInvInertiaTensorWorld() * impulseAxis, -impulseMag);
			}
		}

		// joint limits
		{
			///solve angular part
			btVector3 angVelA;
			bodyA.internalGetAngularVelocity(angVelA);
			btVector3 angVelB;
			bodyB.internalGetAngularVelocity(angVelB);

			// solve swing limit
			if (m_solveSwingLimit)
			{
				btScalar amplitude = m_swingLimitRatio * m_swingCorrection * m_biasFactor / timeStep;
				btScalar relSwingVel = (angVelB - angVelA).dot(m_swingAxis);
				if (relSwingVel > 0)
					amplitude += m_swingLimitRatio * relSwingVel * m_relaxationFactor;
				btScalar impulseMag = amplitude * m_kSwing;

				// Clamp the accumulated impulse
				btScalar temp = m_accSwingLimitImpulse;
				m_accSwingLimitImpulse = btMax(m_accSwingLimitImpulse + impulseMag, btScalar(0.0));
				impulseMag = m_accSwingLimitImpulse - temp;

				btVector3 impulse = m_swingAxis * impulseMag;

				// don't let cone response affect twist
				// (this can happen since body A's twist doesn't match body B's AND we use an elliptical cone limit)
				{
					btVector3 impulseTwistCouple = impulse.dot(m_twistAxisA) * m_twistAxisA;
					btVector3 impulseNoTwistCouple = impulse - impulseTwistCouple;
					impulse = impulseNoTwistCouple;
				}

				impulseMag = impulse.length();
				btVector3 noTwistSwingAxis = impulse / impulseMag;

				bodyA.internalApplyImpulse(btVector3(0, 0, 0), m_rbA.getInvInertiaTensorWorld() * noTwistSwingAxis, impulseMag);
				bodyB.internalApplyImpulse(btVector3(0, 0, 0), m_rbB.getInvInertiaTensorWorld() * noTwistSwingAxis, -impulseMag);
			}

			// solve twist limit
			if (m_solveTwistLimit)
			{
				btScalar amplitude = m_twistLimitRatio * m_twistCorrection * m_biasFactor / timeStep;
				btScalar relTwistVel = (angVelB - angVelA).dot(m_twistAxis);
				if (relTwistVel > 0)  // only damp when moving towards limit (m_twistAxis flipping is important)
					amplitude += m_twistLimitRatio * relTwistVel * m_relaxationFactor;
				btScalar impulseMag = amplitude * m_kTwist;

				// Clamp the accumulated impulse
				btScalar temp = m_accTwistLimitImpulse;
				m_accTwistLimitImpulse = btMax(m_accTwistLimitImpulse + impulseMag, btScalar(0.0));
				impulseMag = m_accTwistLimitImpulse - temp;

				//		btVector3 impulse = m_twistAxis * impulseMag;

				bodyA.internalApplyImpulse(btVector3(0, 0, 0), m_rbA.getInvInertiaTensorWorld() * m_twistAxis, impulseMag);
				bodyB.internalApplyImpulse(btVector3(0, 0, 0), m_rbB.getInvInertiaTensorWorld() * m_twistAxis, -impulseMag);
			}
		}
	}
#else
	btAssert(0);
#endif  //__SPU__
}

void btConeTwistConstraint::updateRHS(btScalar timeStep)
{
	(void)timeStep;
}

#ifndef __SPU__
void btConeTwistConstraint::calcAngleInfo()
{
	m_swingCorrection = btScalar(0.);
	m_twistLimitSign = btScalar(0.);
	m_solveTwistLimit = false;
	m_solveSwingLimit = false;

	btVector3 b1Axis1(0, 0, 0), b1Axis2(0, 0, 0), b1Axis3(0, 0, 0);
	btVector3 b2Axis1(0, 0, 0), b2Axis2(0, 0, 0);

	b1Axis1 = getRigidBodyA().getCenterOfMassTransform().getBasis() * this->m_rbAFrame.getBasis().getColumn(0);
	b2Axis1 = getRigidBodyB().getCenterOfMassTransform().getBasis() * this->m_rbBFrame.getBasis().getColumn(0);

	btScalar swing1 = btScalar(0.), swing2 = btScalar(0.);

	btScalar swx = btScalar(0.), swy = btScalar(0.);
	btScalar thresh = btScalar(10.);
	btScalar fact;

	// Get Frame into world space
	if (m_swingSpan1 >= btScalar(0.05f))
	{
		b1Axis2 = getRigidBodyA().getCenterOfMassTransform().getBasis() * this->m_rbAFrame.getBasis().getColumn(1);
		swx = b2Axis1.dot(b1Axis1);
		swy = b2Axis1.dot(b1Axis2);
		swing1 = btAtan2Fast(swy, swx);
		fact = (swy * swy + swx * swx) * thresh * thresh;
		fact = fact / (fact + btScalar(1.0));
		swing1 *= fact;
	}

	if (m_swingSpan2 >= btScalar(0.05f))
	{
		b1Axis3 = getRigidBodyA().getCenterOfMassTransform().getBasis() * this->m_rbAFrame.getBasis().getColumn(2);
		swx = b2Axis1.dot(b1Axis1);
		swy = b2Axis1.dot(b1Axis3);
		swing2 = btAtan2Fast(swy, swx);
		fact = (swy * swy + swx * swx) * thresh * thresh;
		fact = fact / (fact + btScalar(1.0));
		swing2 *= fact;
	}

	btScalar RMaxAngle1Sq = 1.0f / (m_swingSpan1 * m_swingSpan1);
	btScalar RMaxAngle2Sq = 1.0f / (m_swingSpan2 * m_swingSpan2);
	btScalar EllipseAngle = btFabs(swing1 * swing1) * RMaxAngle1Sq + btFabs(swing2 * swing2) * RMaxAngle2Sq;

	if (EllipseAngle > 1.0f)
	{
		m_swingCorrection = EllipseAngle - 1.0f;
		m_solveSwingLimit = true;
		// Calculate necessary axis & factors
		m_swingAxis = b2Axis1.cross(b1Axis2 * b2Axis1.dot(b1Axis2) + b1Axis3 * b2Axis1.dot(b1Axis3));
		m_swingAxis.normalize();
		btScalar swingAxisSign = (b2Axis1.dot(b1Axis1) >= 0.0f) ? 1.0f : -1.0f;
		m_swingAxis *= swingAxisSign;
	}

	// Twist limits
	if (m_twistSpan >= btScalar(0.))
	{
		btVector3 b2Axis2 = getRigidBodyB().getCenterOfMassTransform().getBasis() * this->m_rbBFrame.getBasis().getColumn(1);
		btQuaternion rotationArc = shortestArcQuat(b2Axis1, b1Axis1);
		btVector3 TwistRef = quatRotate(rotationArc, b2Axis2);
		btScalar twist = btAtan2Fast(TwistRef.dot(b1Axis3), TwistRef.dot(b1Axis2));
		m_twistAngle = twist;

		//		btScalar lockedFreeFactor = (m_twistSpan > btScalar(0.05f)) ? m_limitSoftness : btScalar(0.);
		btScalar lockedFreeFactor = (m_twistSpan > btScalar(0.05f)) ? btScalar(1.0f) : btScalar(0.);
		if (twist <= -m_twistSpan * lockedFreeFactor)
		{
			m_twistCorrection = -(twist + m_twistSpan);
			m_solveTwistLimit = true;
			m_twistAxis = (b2Axis1 + b1Axis1) * 0.5f;
			m_twistAxis.normalize();
			m_twistAxis *= -1.0f;
		}
		else if (twist > m_twistSpan * lockedFreeFactor)
		{
			m_twistCorrection = (twist - m_twistSpan);
			m_solveTwistLimit = true;
			m_twistAxis = (b2Axis1 + b1Axis1) * 0.5f;
			m_twistAxis.normalize();
		}
	}
}
#endif  //__SPU__

static btVector3 vTwist(1, 0, 0);  // twist axis in constraint's space

void btConeTwistConstraint::calcAngleInfo2(const btTransform& transA, const btTransform& transB, const btMatrix3x3& invInertiaWorldA, const btMatrix3x3& invInertiaWorldB)
{
	m_swingCorrection = btScalar(0.);
	m_twistLimitSign = btScalar(0.);
	m_solveTwistLimit = false;
	m_solveSwingLimit = false;
	// compute rotation of A wrt B (in constraint space)
	if (m_bMotorEnabled && (!m_useSolveConstraintObsolete))
	{  // it is assumed that setMotorTarget() was alredy called
		// and motor target m_qTarget is within constraint limits
		// TODO : split rotation to pure swing and pure twist
		// compute desired transforms in world
		btTransform trPose(m_qTarget);
		btTransform trA = transA * m_rbAFrame;
		btTransform trB = transB * m_rbBFrame;
		btTransform trDeltaAB = trB * trPose * trA.inverse();
		btQuaternion qDeltaAB = trDeltaAB.getRotation();
		btVector3 swingAxis = btVector3(qDeltaAB.x(), qDeltaAB.y(), qDeltaAB.z());
		btScalar swingAxisLen2 = swingAxis.length2();
		if (btFuzzyZero(swingAxisLen2))
		{
			return;
		}
		m_swingAxis = swingAxis;
		m_swingAxis.normalize();
		m_swingCorrection = qDeltaAB.getAngle();
		if (!btFuzzyZero(m_swingCorrection))
		{
			m_solveSwingLimit = true;
		}
		return;
	}

	{
		// compute rotation of A wrt B (in constraint space)
		btQuaternion qA = transA.getRotation() * m_rbAFrame.getRotation();
		btQuaternion qB = transB.getRotation() * m_rbBFrame.getRotation();
		btQuaternion qAB = qB.inverse() * qA;
		// split rotation into cone and twist
		// (all this is done from B's perspective. Maybe I should be averaging axes...)
		btVector3 vConeNoTwist = quatRotate(qAB, vTwist);
		vConeNoTwist.normalize();
		btQuaternion qABCone = shortestArcQuat(vTwist, vConeNoTwist);
		qABCone.normalize();
		btQuaternion qABTwist = qABCone.inverse() * qAB;
		qABTwist.normalize();

		if (m_swingSpan1 >= m_fixThresh && m_swingSpan2 >= m_fixThresh)
		{
			btScalar swingAngle, swingLimit = 0;
			btVector3 swingAxis;
			computeConeLimitInfo(qABCone, swingAngle, swingAxis, swingLimit);

			if (swingAngle > swingLimit * m_limitSoftness)
			{
				m_solveSwingLimit = true;

				// compute limit ratio: 0->1, where
				// 0 == beginning of soft limit
				// 1 == hard/real limit
				m_swingLimitRatio = 1.f;
				if (swingAngle < swingLimit && m_limitSoftness < 1.f - SIMD_EPSILON)
				{
					m_swingLimitRatio = (swingAngle - swingLimit * m_limitSoftness) /
										(swingLimit - swingLimit * m_limitSoftness);
				}

				// swing correction tries to get back to soft limit
				m_swingCorrection = swingAngle - (swingLimit * m_limitSoftness);

				// adjustment of swing axis (based on ellipse normal)
				adjustSwingAxisToUseEllipseNormal(swingAxis);

				// Calculate necessary axis & factors
				m_swingAxis = quatRotate(qB, -swingAxis);

				m_twistAxisA.setValue(0, 0, 0);

				m_kSwing = btScalar(1.) /
						   (computeAngularImpulseDenominator(m_swingAxis, invInertiaWorldA) +
							computeAngularImpulseDenominator(m_swingAxis, invInertiaWorldB));
			}
		}
		else
		{
			// you haven't set any limits;
			// or you're trying to set at least one of the swing limits too small. (if so, do you really want a conetwist constraint?)
			// anyway, we have either hinge or fixed joint
			btVector3 ivA = transA.getBasis() * m_rbAFrame.getBasis().getColumn(0);
			btVector3 jvA = transA.getBasis() * m_rbAFrame.getBasis().getColumn(1);
			btVector3 kvA = transA.getBasis() * m_rbAFrame.getBasis().getColumn(2);
			btVector3 ivB = transB.getBasis() * m_rbBFrame.getBasis().getColumn(0);
			btVector3 target;
			btScalar x = ivB.dot(ivA);
			btScalar y = ivB.dot(jvA);
			btScalar z = ivB.dot(kvA);
			if ((m_swingSpan1 < m_fixThresh) && (m_swingSpan2 < m_fixThresh))
			{  // fixed. We'll need to add one more row to constraint
				if ((!btFuzzyZero(y)) || (!(btFuzzyZero(z))))
				{
					m_solveSwingLimit = true;
					m_swingAxis = -ivB.cross(ivA);
				}
			}
			else
			{
				if (m_swingSpan1 < m_fixThresh)
				{  // hinge around Y axis
					//					if(!(btFuzzyZero(y)))
					if ((!(btFuzzyZero(x))) || (!(btFuzzyZero(z))))
					{
						m_solveSwingLimit = true;
						if (m_swingSpan2 >= m_fixThresh)
						{
							y = btScalar(0.f);
							btScalar span2 = btAtan2(z, x);
							if (span2 > m_swingSpan2)
							{
								x = btCos(m_swingSpan2);
								z = btSin(m_swingSpan2);
							}
							else if (span2 < -m_swingSpan2)
							{
								x = btCos(m_swingSpan2);
								z = -btSin(m_swingSpan2);
							}
						}
					}
				}
				else
				{  // hinge around Z axis
					//					if(!btFuzzyZero(z))
					if ((!(btFuzzyZero(x))) || (!(btFuzzyZero(y))))
					{
						m_solveSwingLimit = true;
						if (m_swingSpan1 >= m_fixThresh)
						{
							z = btScalar(0.f);
							btScalar span1 = btAtan2(y, x);
							if (span1 > m_swingSpan1)
							{
								x = btCos(m_swingSpan1);
								y = btSin(m_swingSpan1);
							}
							else if (span1 < -m_swingSpan1)
							{
								x = btCos(m_swingSpan1);
								y = -btSin(m_swingSpan1);
							}
						}
					}
				}
				target[0] = x * ivA[0] + y * jvA[0] + z * kvA[0];
				target[1] = x * ivA[1] + y * jvA[1] + z * kvA[1];
				target[2] = x * ivA[2] + y * jvA[2] + z * kvA[2];
				target.normalize();
				m_swingAxis = -ivB.cross(target);
				m_swingCorrection = m_swingAxis.length();

				if (!btFuzzyZero(m_swingCorrection))
					m_swingAxis.normalize();
			}
		}

		if (m_twistSpan >= btScalar(0.f))
		{
			btVector3 twistAxis;
			computeTwistLimitInfo(qABTwist, m_twistAngle, twistAxis);

			if (m_twistAngle > m_twistSpan * m_limitSoftness)
			{
				m_solveTwistLimit = true;

				m_twistLimitRatio = 1.f;
				if (m_twistAngle < m_twistSpan && m_limitSoftness < 1.f - SIMD_EPSILON)
				{
					m_twistLimitRatio = (m_twistAngle - m_twistSpan * m_limitSoftness) /
										(m_twistSpan - m_twistSpan * m_limitSoftness);
				}

				// twist correction tries to get back to soft limit
				m_twistCorrection = m_twistAngle - (m_twistSpan * m_limitSoftness);

				m_twistAxis = quatRotate(qB, -twistAxis);

				m_kTwist = btScalar(1.) /
						   (computeAngularImpulseDenominator(m_twistAxis, invInertiaWorldA) +
							computeAngularImpulseDenominator(m_twistAxis, invInertiaWorldB));
			}

			if (m_solveSwingLimit)
				m_twistAxisA = quatRotate(qA, -twistAxis);
		}
		else
		{
			m_twistAngle = btScalar(0.f);
		}
	}
}

// given a cone rotation in constraint space, (pre: twist must already be removed)
// this method computes its corresponding swing angle and axis.
// more interestingly, it computes the cone/swing limit (angle) for this cone "pose".
void btConeTwistConstraint::computeConeLimitInfo(const btQuaternion& qCone,
												 btScalar& swingAngle,   // out
												 btVector3& vSwingAxis,  // out
												 btScalar& swingLimit)   // out
{
	swingAngle = qCone.getAngle();
	if (swingAngle > SIMD_EPSILON)
	{
		vSwingAxis = btVector3(qCone.x(), qCone.y(), qCone.z());
		vSwingAxis.normalize();
#if 0
        // non-zero twist?! this should never happen.
       btAssert(fabs(vSwingAxis.x()) <= SIMD_EPSILON));
#endif

		// Compute limit for given swing. tricky:
		// Given a swing axis, we're looking for the intersection with the bounding cone ellipse.
		// (Since we're dealing with angles, this ellipse is embedded on the surface of a sphere.)

		// For starters, compute the direction from center to surface of ellipse.
		// This is just the perpendicular (ie. rotate 2D vector by PI/2) of the swing axis.
		// (vSwingAxis is the cone rotation (in z,y); change vars and rotate to (x,y) coords.)
		btScalar xEllipse = vSwingAxis.y();
		btScalar yEllipse = -vSwingAxis.z();

		// Now, we use the slope of the vector (using x/yEllipse) and find the length
		// of the line that intersects the ellipse:
		//  x^2   y^2
		//  --- + --- = 1, where a and b are semi-major axes 2 and 1 respectively (ie. the limits)
		//  a^2   b^2
		// Do the math and it should be clear.

		swingLimit = m_swingSpan1;  // if xEllipse == 0, we have a pure vSwingAxis.z rotation: just use swingspan1
		if (fabs(xEllipse) > SIMD_EPSILON)
		{
			btScalar surfaceSlope2 = (yEllipse * yEllipse) / (xEllipse * xEllipse);
			btScalar norm = 1 / (m_swingSpan2 * m_swingSpan2);
			norm += surfaceSlope2 / (m_swingSpan1 * m_swingSpan1);
			btScalar swingLimit2 = (1 + surfaceSlope2) / norm;
			swingLimit = std::sqrt(swingLimit2);
		}

		// test!
		/*swingLimit = m_swingSpan2;
		if (fabs(vSwingAxis.z()) > SIMD_EPSILON)
		{
		btScalar mag_2 = m_swingSpan1*m_swingSpan1 + m_swingSpan2*m_swingSpan2;
		btScalar sinphi = m_swingSpan2 / sqrt(mag_2);
		btScalar phi = asin(sinphi);
		btScalar theta = atan2(fabs(vSwingAxis.y()),fabs(vSwingAxis.z()));
		btScalar alpha = 3.14159f - theta - phi;
		btScalar sinalpha = sin(alpha);
		swingLimit = m_swingSpan1 * sinphi/sinalpha;
		}*/
	}
	else if (swingAngle < 0)
	{
		// this should never happen!
#if 0
        btAssert(0);
#endif
	}
}

btVector3 btConeTwistConstraint::GetPointForAngle(btScalar fAngleInRadians, btScalar fLength) const
{
	// compute x/y in ellipse using cone angle (0 -> 2*PI along surface of cone)
	btScalar xEllipse = btCos(fAngleInRadians);
	btScalar yEllipse = btSin(fAngleInRadians);

	// Use the slope of the vector (using x/yEllipse) and find the length
	// of the line that intersects the ellipse:
	//  x^2   y^2
	//  --- + --- = 1, where a and b are semi-major axes 2 and 1 respectively (ie. the limits)
	//  a^2   b^2
	// Do the math and it should be clear.

	btScalar swingLimit = m_swingSpan1;  // if xEllipse == 0, just use axis b (1)
	if (fabs(xEllipse) > SIMD_EPSILON)
	{
		btScalar surfaceSlope2 = (yEllipse * yEllipse) / (xEllipse * xEllipse);
		btScalar norm = 1 / (m_swingSpan2 * m_swingSpan2);
		norm += surfaceSlope2 / (m_swingSpan1 * m_swingSpan1);
		btScalar swingLimit2 = (1 + surfaceSlope2) / norm;
		swingLimit = std::sqrt(swingLimit2);
	}

	// convert into point in constraint space:
	// note: twist is x-axis, swing 1 and 2 are along the z and y axes respectively
	btVector3 vSwingAxis(0, xEllipse, -yEllipse);
	btQuaternion qSwing(vSwingAxis, swingLimit);
	btVector3 vPointInConstraintSpace(fLength, 0, 0);
	return quatRotate(qSwing, vPointInConstraintSpace);
}

// given a twist rotation in constraint space, (pre: cone must already be removed)
// this method computes its corresponding angle and axis.
void btConeTwistConstraint::computeTwistLimitInfo(const btQuaternion& qTwist,
												  btScalar& twistAngle,   // out
												  btVector3& vTwistAxis)  // out
{
	btQuaternion qMinTwist = qTwist;
	twistAngle = qTwist.getAngle();

	if (twistAngle > SIMD_PI)  // long way around. flip quat and recalculate.
	{
		qMinTwist = -(qTwist);
		twistAngle = qMinTwist.getAngle();
	}
	if (twistAngle < 0)
	{
		// this should never happen
#if 0
        btAssert(0);
#endif
	}

	vTwistAxis = btVector3(qMinTwist.x(), qMinTwist.y(), qMinTwist.z());
	if (twistAngle > SIMD_EPSILON)
		vTwistAxis.normalize();
}

void btConeTwistConstraint::adjustSwingAxisToUseEllipseNormal(btVector3& vSwingAxis) const
{
	// the swing axis is computed as the "twist-free" cone rotation,
	// but the cone limit is not circular, but elliptical (if swingspan1 != swingspan2).
	// so, if we're outside the limits, the closest way back inside the cone isn't
	// along the vector back to the center. better (and more stable) to use the ellipse normal.

	// convert swing axis to direction from center to surface of ellipse
	// (ie. rotate 2D vector by PI/2)
	btScalar y = -vSwingAxis.z();
	btScalar z = vSwingAxis.y();

	// do the math...
	if (fabs(z) > SIMD_EPSILON)  // avoid division by 0. and we don't need an update if z == 0.
	{
		// compute gradient/normal of ellipse surface at current "point"
		btScalar grad = y / z;
		grad *= m_swingSpan2 / m_swingSpan1;

		// adjust y/z to represent normal at point (instead of vector to point)
		if (y > 0)
			y = fabs(grad * z);
		else
			y = -fabs(grad * z);

		// convert ellipse direction back to swing axis
		vSwingAxis.setZ(-y);
		vSwingAxis.setY(z);
		vSwingAxis.normalize();
	}
}

void btConeTwistConstraint::setMotorTarget(const btQuaternion& q)
{
	//btTransform trACur = m_rbA.getCenterOfMassTransform();
	//btTransform trBCur = m_rbB.getCenterOfMassTransform();
	//	btTransform trABCur = trBCur.inverse() * trACur;
	//	btQuaternion qABCur = trABCur.getRotation();
	//	btTransform trConstraintCur = (trBCur * m_rbBFrame).inverse() * (trACur * m_rbAFrame);
	//btQuaternion qConstraintCur = trConstraintCur.getRotation();

	btQuaternion qConstraint = m_rbBFrame.getRotation().inverse() * q * m_rbAFrame.getRotation();
	setMotorTargetInConstraintSpace(qConstraint);
}

void btConeTwistConstraint::setMotorTargetInConstraintSpace(const btQuaternion& q)
{
	m_qTarget = q;

	// clamp motor target to within limits
	{
		btScalar softness = 1.f;  //m_limitSoftness;

		// split into twist and cone
		btVector3 vTwisted = quatRotate(m_qTarget, vTwist);
		btQuaternion qTargetCone = shortestArcQuat(vTwist, vTwisted);
		qTargetCone.normalize();
		btQuaternion qTargetTwist = qTargetCone.inverse() * m_qTarget;
		qTargetTwist.normalize();

		// clamp cone
		if (m_swingSpan1 >= btScalar(0.05f) && m_swingSpan2 >= btScalar(0.05f))
		{
			btScalar swingAngle, swingLimit;
			btVector3 swingAxis;
			computeConeLimitInfo(qTargetCone, swingAngle, swingAxis, swingLimit);

			if (fabs(swingAngle) > SIMD_EPSILON)
			{
				if (swingAngle > swingLimit * softness)
					swingAngle = swingLimit * softness;
				else if (swingAngle < -swingLimit * softness)
					swingAngle = -swingLimit * softness;
				qTargetCone = btQuaternion(swingAxis, swingAngle);
			}
		}

		// clamp twist
		if (m_twistSpan >= btScalar(0.05f))
		{
			btScalar twistAngle;
			btVector3 twistAxis;
			computeTwistLimitInfo(qTargetTwist, twistAngle, twistAxis);

			if (fabs(twistAngle) > SIMD_EPSILON)
			{
				// eddy todo: limitSoftness used here???
				if (twistAngle > m_twistSpan * softness)
					twistAngle = m_twistSpan * softness;
				else if (twistAngle < -m_twistSpan * softness)
					twistAngle = -m_twistSpan * softness;
				qTargetTwist = btQuaternion(twistAxis, twistAngle);
			}
		}

		m_qTarget = qTargetCone * qTargetTwist;
	}
}

///override the default global value of a parameter (such as ERP or CFM), optionally provide the axis (0..5).
///If no axis is provided, it uses the default axis for this constraint.
void btConeTwistConstraint::setParam(int num, btScalar value, int axis)
{
	switch (num)
	{
		case BT_CONSTRAINT_ERP:
		case BT_CONSTRAINT_STOP_ERP:
			if ((axis >= 0) && (axis < 3))
			{
				m_linERP = value;
				m_flags |= BT_CONETWIST_FLAGS_LIN_ERP;
			}
			else
			{
				m_biasFactor = value;
			}
			break;
		case BT_CONSTRAINT_CFM:
		case BT_CONSTRAINT_STOP_CFM:
			if ((axis >= 0) && (axis < 3))
			{
				m_linCFM = value;
				m_flags |= BT_CONETWIST_FLAGS_LIN_CFM;
			}
			else
			{
				m_angCFM = value;
				m_flags |= BT_CONETWIST_FLAGS_ANG_CFM;
			}
			break;
		default:
			btAssertConstrParams(0);
			break;
	}
}

///return the local value of parameter
btScalar btConeTwistConstraint::getParam(int num, int axis) const
{
	btScalar retVal = 0;
	switch (num)
	{
		case BT_CONSTRAINT_ERP:
		case BT_CONSTRAINT_STOP_ERP:
			if ((axis >= 0) && (axis < 3))
			{
				btAssertConstrParams(m_flags & BT_CONETWIST_FLAGS_LIN_ERP);
				retVal = m_linERP;
			}
			else if ((axis >= 3) && (axis < 6))
			{
				retVal = m_biasFactor;
			}
			else
			{
				btAssertConstrParams(0);
			}
			break;
		case BT_CONSTRAINT_CFM:
		case BT_CONSTRAINT_STOP_CFM:
			if ((axis >= 0) && (axis < 3))
			{
				btAssertConstrParams(m_flags & BT_CONETWIST_FLAGS_LIN_CFM);
				retVal = m_linCFM;
			}
			else if ((axis >= 3) && (axis < 6))
			{
				btAssertConstrParams(m_flags & BT_CONETWIST_FLAGS_ANG_CFM);
				retVal = m_angCFM;
			}
			else
			{
				btAssertConstrParams(0);
			}
			break;
		default:
			btAssertConstrParams(0);
	}
	return retVal;
}

void btConeTwistConstraint::setFrames(const btTransform& frameA, const btTransform& frameB)
{
	m_rbAFrame = frameA;
	m_rbBFrame = frameB;
	buildJacobian();
	//calculateTransforms();
}
