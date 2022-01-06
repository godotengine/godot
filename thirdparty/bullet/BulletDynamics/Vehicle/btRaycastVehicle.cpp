/*
 * Copyright (c) 2005 Erwin Coumans https://bulletphysics.org
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Erwin Coumans makes no representations about the suitability 
 * of this software for any purpose.  
 * It is provided "as is" without express or implied warranty.
*/

#include "LinearMath/btVector3.h"
#include "btRaycastVehicle.h"

#include "BulletDynamics/ConstraintSolver/btSolve2LinearConstraint.h"
#include "BulletDynamics/ConstraintSolver/btJacobianEntry.h"
#include "LinearMath/btQuaternion.h"
#include "BulletDynamics/Dynamics/btDynamicsWorld.h"
#include "btVehicleRaycaster.h"
#include "btWheelInfo.h"
#include "LinearMath/btMinMax.h"
#include "LinearMath/btIDebugDraw.h"
#include "BulletDynamics/ConstraintSolver/btContactConstraint.h"

#define ROLLING_INFLUENCE_FIX

btRigidBody& btActionInterface::getFixedBody()
{
	static btRigidBody s_fixed(0, 0, 0);
	s_fixed.setMassProps(btScalar(0.), btVector3(btScalar(0.), btScalar(0.), btScalar(0.)));
	return s_fixed;
}

btRaycastVehicle::btRaycastVehicle(const btVehicleTuning& tuning, btRigidBody* chassis, btVehicleRaycaster* raycaster)
	: m_vehicleRaycaster(raycaster),
	  m_pitchControl(btScalar(0.))
{
	m_chassisBody = chassis;
	m_indexRightAxis = 0;
	m_indexUpAxis = 2;
	m_indexForwardAxis = 1;
	defaultInit(tuning);
}

void btRaycastVehicle::defaultInit(const btVehicleTuning& tuning)
{
	(void)tuning;
	m_currentVehicleSpeedKmHour = btScalar(0.);
	m_steeringValue = btScalar(0.);
}

btRaycastVehicle::~btRaycastVehicle()
{
}

//
// basically most of the code is general for 2 or 4 wheel vehicles, but some of it needs to be reviewed
//
btWheelInfo& btRaycastVehicle::addWheel(const btVector3& connectionPointCS, const btVector3& wheelDirectionCS0, const btVector3& wheelAxleCS, btScalar suspensionRestLength, btScalar wheelRadius, const btVehicleTuning& tuning, bool isFrontWheel)
{
	btWheelInfoConstructionInfo ci;

	ci.m_chassisConnectionCS = connectionPointCS;
	ci.m_wheelDirectionCS = wheelDirectionCS0;
	ci.m_wheelAxleCS = wheelAxleCS;
	ci.m_suspensionRestLength = suspensionRestLength;
	ci.m_wheelRadius = wheelRadius;
	ci.m_suspensionStiffness = tuning.m_suspensionStiffness;
	ci.m_wheelsDampingCompression = tuning.m_suspensionCompression;
	ci.m_wheelsDampingRelaxation = tuning.m_suspensionDamping;
	ci.m_frictionSlip = tuning.m_frictionSlip;
	ci.m_bIsFrontWheel = isFrontWheel;
	ci.m_maxSuspensionTravelCm = tuning.m_maxSuspensionTravelCm;
	ci.m_maxSuspensionForce = tuning.m_maxSuspensionForce;

	m_wheelInfo.push_back(btWheelInfo(ci));

	btWheelInfo& wheel = m_wheelInfo[getNumWheels() - 1];

	updateWheelTransformsWS(wheel, false);
	updateWheelTransform(getNumWheels() - 1, false);
	return wheel;
}

const btTransform& btRaycastVehicle::getWheelTransformWS(int wheelIndex) const
{
	btAssert(wheelIndex < getNumWheels());
	const btWheelInfo& wheel = m_wheelInfo[wheelIndex];
	return wheel.m_worldTransform;
}

void btRaycastVehicle::updateWheelTransform(int wheelIndex, bool interpolatedTransform)
{
	btWheelInfo& wheel = m_wheelInfo[wheelIndex];
	updateWheelTransformsWS(wheel, interpolatedTransform);
	btVector3 up = -wheel.m_raycastInfo.m_wheelDirectionWS;
	const btVector3& right = wheel.m_raycastInfo.m_wheelAxleWS;
	btVector3 fwd = up.cross(right);
	fwd = fwd.normalize();
	//	up = right.cross(fwd);
	//	up.normalize();

	//rotate around steering over de wheelAxleWS
	btScalar steering = wheel.m_steering;

	btQuaternion steeringOrn(up, steering);  //wheel.m_steering);
	btMatrix3x3 steeringMat(steeringOrn);

	btQuaternion rotatingOrn(right, -wheel.m_rotation);
	btMatrix3x3 rotatingMat(rotatingOrn);

	btMatrix3x3 basis2;
	basis2[0][m_indexRightAxis] = -right[0];
	basis2[1][m_indexRightAxis] = -right[1];
	basis2[2][m_indexRightAxis] = -right[2];

	basis2[0][m_indexUpAxis] = up[0];
	basis2[1][m_indexUpAxis] = up[1];
	basis2[2][m_indexUpAxis] = up[2];

	basis2[0][m_indexForwardAxis] = fwd[0];
	basis2[1][m_indexForwardAxis] = fwd[1];
	basis2[2][m_indexForwardAxis] = fwd[2];

	wheel.m_worldTransform.setBasis(steeringMat * rotatingMat * basis2);
	wheel.m_worldTransform.setOrigin(
		wheel.m_raycastInfo.m_hardPointWS + wheel.m_raycastInfo.m_wheelDirectionWS * wheel.m_raycastInfo.m_suspensionLength);
}

void btRaycastVehicle::resetSuspension()
{
	int i;
	for (i = 0; i < m_wheelInfo.size(); i++)
	{
		btWheelInfo& wheel = m_wheelInfo[i];
		wheel.m_raycastInfo.m_suspensionLength = wheel.getSuspensionRestLength();
		wheel.m_suspensionRelativeVelocity = btScalar(0.0);

		wheel.m_raycastInfo.m_contactNormalWS = -wheel.m_raycastInfo.m_wheelDirectionWS;
		//wheel_info.setContactFriction(btScalar(0.0));
		wheel.m_clippedInvContactDotSuspension = btScalar(1.0);
	}
}

void btRaycastVehicle::updateWheelTransformsWS(btWheelInfo& wheel, bool interpolatedTransform)
{
	wheel.m_raycastInfo.m_isInContact = false;

	btTransform chassisTrans = getChassisWorldTransform();
	if (interpolatedTransform && (getRigidBody()->getMotionState()))
	{
		getRigidBody()->getMotionState()->getWorldTransform(chassisTrans);
	}

	wheel.m_raycastInfo.m_hardPointWS = chassisTrans(wheel.m_chassisConnectionPointCS);
	wheel.m_raycastInfo.m_wheelDirectionWS = chassisTrans.getBasis() * wheel.m_wheelDirectionCS;
	wheel.m_raycastInfo.m_wheelAxleWS = chassisTrans.getBasis() * wheel.m_wheelAxleCS;
}

btScalar btRaycastVehicle::rayCast(btWheelInfo& wheel)
{
	updateWheelTransformsWS(wheel, false);

	btScalar depth = -1;

	btScalar raylen = wheel.getSuspensionRestLength() + wheel.m_wheelsRadius;

	btVector3 rayvector = wheel.m_raycastInfo.m_wheelDirectionWS * (raylen);
	const btVector3& source = wheel.m_raycastInfo.m_hardPointWS;
	wheel.m_raycastInfo.m_contactPointWS = source + rayvector;
	const btVector3& target = wheel.m_raycastInfo.m_contactPointWS;

	btScalar param = btScalar(0.);

	btVehicleRaycaster::btVehicleRaycasterResult rayResults;

	btAssert(m_vehicleRaycaster);

	void* object = m_vehicleRaycaster->castRay(source, target, rayResults);

	wheel.m_raycastInfo.m_groundObject = 0;

	if (object)
	{
		param = rayResults.m_distFraction;
		depth = raylen * rayResults.m_distFraction;
		wheel.m_raycastInfo.m_contactNormalWS = rayResults.m_hitNormalInWorld;
		wheel.m_raycastInfo.m_isInContact = true;

		wheel.m_raycastInfo.m_groundObject = &getFixedBody();  ///@todo for driving on dynamic/movable objects!;
		//wheel.m_raycastInfo.m_groundObject = object;

		btScalar hitDistance = param * raylen;
		wheel.m_raycastInfo.m_suspensionLength = hitDistance - wheel.m_wheelsRadius;
		//clamp on max suspension travel

		btScalar minSuspensionLength = wheel.getSuspensionRestLength() - wheel.m_maxSuspensionTravelCm * btScalar(0.01);
		btScalar maxSuspensionLength = wheel.getSuspensionRestLength() + wheel.m_maxSuspensionTravelCm * btScalar(0.01);
		if (wheel.m_raycastInfo.m_suspensionLength < minSuspensionLength)
		{
			wheel.m_raycastInfo.m_suspensionLength = minSuspensionLength;
		}
		if (wheel.m_raycastInfo.m_suspensionLength > maxSuspensionLength)
		{
			wheel.m_raycastInfo.m_suspensionLength = maxSuspensionLength;
		}

		wheel.m_raycastInfo.m_contactPointWS = rayResults.m_hitPointInWorld;

		btScalar denominator = wheel.m_raycastInfo.m_contactNormalWS.dot(wheel.m_raycastInfo.m_wheelDirectionWS);

		btVector3 chassis_velocity_at_contactPoint;
		btVector3 relpos = wheel.m_raycastInfo.m_contactPointWS - getRigidBody()->getCenterOfMassPosition();

		chassis_velocity_at_contactPoint = getRigidBody()->getVelocityInLocalPoint(relpos);

		btScalar projVel = wheel.m_raycastInfo.m_contactNormalWS.dot(chassis_velocity_at_contactPoint);

		if (denominator >= btScalar(-0.1))
		{
			wheel.m_suspensionRelativeVelocity = btScalar(0.0);
			wheel.m_clippedInvContactDotSuspension = btScalar(1.0) / btScalar(0.1);
		}
		else
		{
			btScalar inv = btScalar(-1.) / denominator;
			wheel.m_suspensionRelativeVelocity = projVel * inv;
			wheel.m_clippedInvContactDotSuspension = inv;
		}
	}
	else
	{
		//put wheel info as in rest position
		wheel.m_raycastInfo.m_suspensionLength = wheel.getSuspensionRestLength();
		wheel.m_suspensionRelativeVelocity = btScalar(0.0);
		wheel.m_raycastInfo.m_contactNormalWS = -wheel.m_raycastInfo.m_wheelDirectionWS;
		wheel.m_clippedInvContactDotSuspension = btScalar(1.0);
	}

	return depth;
}

const btTransform& btRaycastVehicle::getChassisWorldTransform() const
{
	/*if (getRigidBody()->getMotionState())
	{
		btTransform chassisWorldTrans;
		getRigidBody()->getMotionState()->getWorldTransform(chassisWorldTrans);
		return chassisWorldTrans;
	}
	*/

	return getRigidBody()->getCenterOfMassTransform();
}

void btRaycastVehicle::updateVehicle(btScalar step)
{
	{
		for (int i = 0; i < getNumWheels(); i++)
		{
			updateWheelTransform(i, false);
		}
	}

	m_currentVehicleSpeedKmHour = btScalar(3.6) * getRigidBody()->getLinearVelocity().length();

	const btTransform& chassisTrans = getChassisWorldTransform();

	btVector3 forwardW(
		chassisTrans.getBasis()[0][m_indexForwardAxis],
		chassisTrans.getBasis()[1][m_indexForwardAxis],
		chassisTrans.getBasis()[2][m_indexForwardAxis]);

	if (forwardW.dot(getRigidBody()->getLinearVelocity()) < btScalar(0.))
	{
		m_currentVehicleSpeedKmHour *= btScalar(-1.);
	}

	//
	// simulate suspension
	//

	int i = 0;
	for (i = 0; i < m_wheelInfo.size(); i++)
	{
		//btScalar depth;
		//depth =
		rayCast(m_wheelInfo[i]);
	}

	updateSuspension(step);

	for (i = 0; i < m_wheelInfo.size(); i++)
	{
		//apply suspension force
		btWheelInfo& wheel = m_wheelInfo[i];

		btScalar suspensionForce = wheel.m_wheelsSuspensionForce;

		if (suspensionForce > wheel.m_maxSuspensionForce)
		{
			suspensionForce = wheel.m_maxSuspensionForce;
		}
		btVector3 impulse = wheel.m_raycastInfo.m_contactNormalWS * suspensionForce * step;
		btVector3 relpos = wheel.m_raycastInfo.m_contactPointWS - getRigidBody()->getCenterOfMassPosition();

		getRigidBody()->applyImpulse(impulse, relpos);
	}

	updateFriction(step);

	for (i = 0; i < m_wheelInfo.size(); i++)
	{
		btWheelInfo& wheel = m_wheelInfo[i];
		btVector3 relpos = wheel.m_raycastInfo.m_hardPointWS - getRigidBody()->getCenterOfMassPosition();
		btVector3 vel = getRigidBody()->getVelocityInLocalPoint(relpos);

		if (wheel.m_raycastInfo.m_isInContact)
		{
			const btTransform& chassisWorldTransform = getChassisWorldTransform();

			btVector3 fwd(
				chassisWorldTransform.getBasis()[0][m_indexForwardAxis],
				chassisWorldTransform.getBasis()[1][m_indexForwardAxis],
				chassisWorldTransform.getBasis()[2][m_indexForwardAxis]);

			btScalar proj = fwd.dot(wheel.m_raycastInfo.m_contactNormalWS);
			fwd -= wheel.m_raycastInfo.m_contactNormalWS * proj;

			btScalar proj2 = fwd.dot(vel);

			wheel.m_deltaRotation = (proj2 * step) / (wheel.m_wheelsRadius);
			wheel.m_rotation += wheel.m_deltaRotation;
		}
		else
		{
			wheel.m_rotation += wheel.m_deltaRotation;
		}

		wheel.m_deltaRotation *= btScalar(0.99);  //damping of rotation when not in contact
	}
}

void btRaycastVehicle::setSteeringValue(btScalar steering, int wheel)
{
	btAssert(wheel >= 0 && wheel < getNumWheels());

	btWheelInfo& wheelInfo = getWheelInfo(wheel);
	wheelInfo.m_steering = steering;
}

btScalar btRaycastVehicle::getSteeringValue(int wheel) const
{
	return getWheelInfo(wheel).m_steering;
}

void btRaycastVehicle::applyEngineForce(btScalar force, int wheel)
{
	btAssert(wheel >= 0 && wheel < getNumWheels());
	btWheelInfo& wheelInfo = getWheelInfo(wheel);
	wheelInfo.m_engineForce = force;
}

const btWheelInfo& btRaycastVehicle::getWheelInfo(int index) const
{
	btAssert((index >= 0) && (index < getNumWheels()));

	return m_wheelInfo[index];
}

btWheelInfo& btRaycastVehicle::getWheelInfo(int index)
{
	btAssert((index >= 0) && (index < getNumWheels()));

	return m_wheelInfo[index];
}

void btRaycastVehicle::setBrake(btScalar brake, int wheelIndex)
{
	btAssert((wheelIndex >= 0) && (wheelIndex < getNumWheels()));
	getWheelInfo(wheelIndex).m_brake = brake;
}

void btRaycastVehicle::updateSuspension(btScalar deltaTime)
{
	(void)deltaTime;

	btScalar chassisMass = btScalar(1.) / m_chassisBody->getInvMass();

	for (int w_it = 0; w_it < getNumWheels(); w_it++)
	{
		btWheelInfo& wheel_info = m_wheelInfo[w_it];

		if (wheel_info.m_raycastInfo.m_isInContact)
		{
			btScalar force;
			//	Spring
			{
				btScalar susp_length = wheel_info.getSuspensionRestLength();
				btScalar current_length = wheel_info.m_raycastInfo.m_suspensionLength;

				btScalar length_diff = (susp_length - current_length);

				force = wheel_info.m_suspensionStiffness * length_diff * wheel_info.m_clippedInvContactDotSuspension;
			}

			// Damper
			{
				btScalar projected_rel_vel = wheel_info.m_suspensionRelativeVelocity;
				{
					btScalar susp_damping;
					if (projected_rel_vel < btScalar(0.0))
					{
						susp_damping = wheel_info.m_wheelsDampingCompression;
					}
					else
					{
						susp_damping = wheel_info.m_wheelsDampingRelaxation;
					}
					force -= susp_damping * projected_rel_vel;
				}
			}

			// RESULT
			wheel_info.m_wheelsSuspensionForce = force * chassisMass;
			if (wheel_info.m_wheelsSuspensionForce < btScalar(0.))
			{
				wheel_info.m_wheelsSuspensionForce = btScalar(0.);
			}
		}
		else
		{
			wheel_info.m_wheelsSuspensionForce = btScalar(0.0);
		}
	}
}

struct btWheelContactPoint
{
	btRigidBody* m_body0;
	btRigidBody* m_body1;
	btVector3 m_frictionPositionWorld;
	btVector3 m_frictionDirectionWorld;
	btScalar m_jacDiagABInv;
	btScalar m_maxImpulse;

	btWheelContactPoint(btRigidBody* body0, btRigidBody* body1, const btVector3& frictionPosWorld, const btVector3& frictionDirectionWorld, btScalar maxImpulse)
		: m_body0(body0),
		  m_body1(body1),
		  m_frictionPositionWorld(frictionPosWorld),
		  m_frictionDirectionWorld(frictionDirectionWorld),
		  m_maxImpulse(maxImpulse)
	{
		btScalar denom0 = body0->computeImpulseDenominator(frictionPosWorld, frictionDirectionWorld);
		btScalar denom1 = body1->computeImpulseDenominator(frictionPosWorld, frictionDirectionWorld);
		btScalar relaxation = 1.f;
		m_jacDiagABInv = relaxation / (denom0 + denom1);
	}
};

btScalar calcRollingFriction(btWheelContactPoint& contactPoint, int numWheelsOnGround);
btScalar calcRollingFriction(btWheelContactPoint& contactPoint, int numWheelsOnGround)
{
	btScalar j1 = 0.f;

	const btVector3& contactPosWorld = contactPoint.m_frictionPositionWorld;

	btVector3 rel_pos1 = contactPosWorld - contactPoint.m_body0->getCenterOfMassPosition();
	btVector3 rel_pos2 = contactPosWorld - contactPoint.m_body1->getCenterOfMassPosition();

	btScalar maxImpulse = contactPoint.m_maxImpulse;

	btVector3 vel1 = contactPoint.m_body0->getVelocityInLocalPoint(rel_pos1);
	btVector3 vel2 = contactPoint.m_body1->getVelocityInLocalPoint(rel_pos2);
	btVector3 vel = vel1 - vel2;

	btScalar vrel = contactPoint.m_frictionDirectionWorld.dot(vel);

	// calculate j that moves us to zero relative velocity
	j1 = -vrel * contactPoint.m_jacDiagABInv / btScalar(numWheelsOnGround);
	btSetMin(j1, maxImpulse);
	btSetMax(j1, -maxImpulse);

	return j1;
}

btScalar sideFrictionStiffness2 = btScalar(1.0);
void btRaycastVehicle::updateFriction(btScalar timeStep)
{
	//calculate the impulse, so that the wheels don't move sidewards
	int numWheel = getNumWheels();
	if (!numWheel)
		return;

	m_forwardWS.resize(numWheel);
	m_axle.resize(numWheel);
	m_forwardImpulse.resize(numWheel);
	m_sideImpulse.resize(numWheel);

	int numWheelsOnGround = 0;

	//collapse all those loops into one!
	for (int i = 0; i < getNumWheels(); i++)
	{
		btWheelInfo& wheelInfo = m_wheelInfo[i];
		class btRigidBody* groundObject = (class btRigidBody*)wheelInfo.m_raycastInfo.m_groundObject;
		if (groundObject)
			numWheelsOnGround++;
		m_sideImpulse[i] = btScalar(0.);
		m_forwardImpulse[i] = btScalar(0.);
	}

	{
		for (int i = 0; i < getNumWheels(); i++)
		{
			btWheelInfo& wheelInfo = m_wheelInfo[i];

			class btRigidBody* groundObject = (class btRigidBody*)wheelInfo.m_raycastInfo.m_groundObject;

			if (groundObject)
			{
				const btTransform& wheelTrans = getWheelTransformWS(i);

				btMatrix3x3 wheelBasis0 = wheelTrans.getBasis();
				m_axle[i] = -btVector3(
					wheelBasis0[0][m_indexRightAxis],
					wheelBasis0[1][m_indexRightAxis],
					wheelBasis0[2][m_indexRightAxis]);

				const btVector3& surfNormalWS = wheelInfo.m_raycastInfo.m_contactNormalWS;
				btScalar proj = m_axle[i].dot(surfNormalWS);
				m_axle[i] -= surfNormalWS * proj;
				m_axle[i] = m_axle[i].normalize();

				m_forwardWS[i] = surfNormalWS.cross(m_axle[i]);
				m_forwardWS[i].normalize();

				resolveSingleBilateral(*m_chassisBody, wheelInfo.m_raycastInfo.m_contactPointWS,
									   *groundObject, wheelInfo.m_raycastInfo.m_contactPointWS,
									   btScalar(0.), m_axle[i], m_sideImpulse[i], timeStep);

				m_sideImpulse[i] *= sideFrictionStiffness2;
			}
		}
	}

	btScalar sideFactor = btScalar(1.);
	btScalar fwdFactor = 0.5;

	bool sliding = false;
	{
		for (int wheel = 0; wheel < getNumWheels(); wheel++)
		{
			btWheelInfo& wheelInfo = m_wheelInfo[wheel];
			class btRigidBody* groundObject = (class btRigidBody*)wheelInfo.m_raycastInfo.m_groundObject;

			btScalar rollingFriction = 0.f;

			if (groundObject)
			{
				if (wheelInfo.m_engineForce != 0.f)
				{
					rollingFriction = wheelInfo.m_engineForce * timeStep;
				}
				else
				{
					btScalar defaultRollingFrictionImpulse = 0.f;
					btScalar maxImpulse = wheelInfo.m_brake ? wheelInfo.m_brake : defaultRollingFrictionImpulse;
					btWheelContactPoint contactPt(m_chassisBody, groundObject, wheelInfo.m_raycastInfo.m_contactPointWS, m_forwardWS[wheel], maxImpulse);
					btAssert(numWheelsOnGround > 0);
					rollingFriction = calcRollingFriction(contactPt, numWheelsOnGround);
				}
			}

			//switch between active rolling (throttle), braking and non-active rolling friction (no throttle/break)

			m_forwardImpulse[wheel] = btScalar(0.);
			m_wheelInfo[wheel].m_skidInfo = btScalar(1.);

			if (groundObject)
			{
				m_wheelInfo[wheel].m_skidInfo = btScalar(1.);

				btScalar maximp = wheelInfo.m_wheelsSuspensionForce * timeStep * wheelInfo.m_frictionSlip;
				btScalar maximpSide = maximp;

				btScalar maximpSquared = maximp * maximpSide;

				m_forwardImpulse[wheel] = rollingFriction;  //wheelInfo.m_engineForce* timeStep;

				btScalar x = (m_forwardImpulse[wheel]) * fwdFactor;
				btScalar y = (m_sideImpulse[wheel]) * sideFactor;

				btScalar impulseSquared = (x * x + y * y);

				if (impulseSquared > maximpSquared)
				{
					sliding = true;

					btScalar factor = maximp / btSqrt(impulseSquared);

					m_wheelInfo[wheel].m_skidInfo *= factor;
				}
			}
		}
	}

	if (sliding)
	{
		for (int wheel = 0; wheel < getNumWheels(); wheel++)
		{
			if (m_sideImpulse[wheel] != btScalar(0.))
			{
				if (m_wheelInfo[wheel].m_skidInfo < btScalar(1.))
				{
					m_forwardImpulse[wheel] *= m_wheelInfo[wheel].m_skidInfo;
					m_sideImpulse[wheel] *= m_wheelInfo[wheel].m_skidInfo;
				}
			}
		}
	}

	// apply the impulses
	{
		for (int wheel = 0; wheel < getNumWheels(); wheel++)
		{
			btWheelInfo& wheelInfo = m_wheelInfo[wheel];

			btVector3 rel_pos = wheelInfo.m_raycastInfo.m_contactPointWS -
								m_chassisBody->getCenterOfMassPosition();

			if (m_forwardImpulse[wheel] != btScalar(0.))
			{
				m_chassisBody->applyImpulse(m_forwardWS[wheel] * (m_forwardImpulse[wheel]), rel_pos);
			}
			if (m_sideImpulse[wheel] != btScalar(0.))
			{
				class btRigidBody* groundObject = (class btRigidBody*)m_wheelInfo[wheel].m_raycastInfo.m_groundObject;

				btVector3 rel_pos2 = wheelInfo.m_raycastInfo.m_contactPointWS -
									 groundObject->getCenterOfMassPosition();

				btVector3 sideImp = m_axle[wheel] * m_sideImpulse[wheel];

#if defined ROLLING_INFLUENCE_FIX  // fix. It only worked if car's up was along Y - VT.
				btVector3 vChassisWorldUp = getRigidBody()->getCenterOfMassTransform().getBasis().getColumn(m_indexUpAxis);
				rel_pos -= vChassisWorldUp * (vChassisWorldUp.dot(rel_pos) * (1.f - wheelInfo.m_rollInfluence));
#else
				rel_pos[m_indexUpAxis] *= wheelInfo.m_rollInfluence;
#endif
				m_chassisBody->applyImpulse(sideImp, rel_pos);

				//apply friction impulse on the ground
				groundObject->applyImpulse(-sideImp, rel_pos2);
			}
		}
	}
}

void btRaycastVehicle::debugDraw(btIDebugDraw* debugDrawer)
{
	for (int v = 0; v < this->getNumWheels(); v++)
	{
		btVector3 wheelColor(0, 1, 1);
		if (getWheelInfo(v).m_raycastInfo.m_isInContact)
		{
			wheelColor.setValue(0, 0, 1);
		}
		else
		{
			wheelColor.setValue(1, 0, 1);
		}

		btVector3 wheelPosWS = getWheelInfo(v).m_worldTransform.getOrigin();

		btVector3 axle = btVector3(
			getWheelInfo(v).m_worldTransform.getBasis()[0][getRightAxis()],
			getWheelInfo(v).m_worldTransform.getBasis()[1][getRightAxis()],
			getWheelInfo(v).m_worldTransform.getBasis()[2][getRightAxis()]);

		//debug wheels (cylinders)
		debugDrawer->drawLine(wheelPosWS, wheelPosWS + axle, wheelColor);
		debugDrawer->drawLine(wheelPosWS, getWheelInfo(v).m_raycastInfo.m_contactPointWS, wheelColor);
	}
}

void* btDefaultVehicleRaycaster::castRay(const btVector3& from, const btVector3& to, btVehicleRaycasterResult& result)
{
	//	RayResultCallback& resultCallback;

	btCollisionWorld::ClosestRayResultCallback rayCallback(from, to);

	m_dynamicsWorld->rayTest(from, to, rayCallback);

	if (rayCallback.hasHit())
	{
		const btRigidBody* body = btRigidBody::upcast(rayCallback.m_collisionObject);
		if (body && body->hasContactResponse())
		{
			result.m_hitPointInWorld = rayCallback.m_hitPointWorld;
			result.m_hitNormalInWorld = rayCallback.m_hitNormalWorld;
			result.m_hitNormalInWorld.normalize();
			result.m_distFraction = rayCallback.m_closestHitFraction;
			return (void*)body;
		}
	}
	return 0;
}
