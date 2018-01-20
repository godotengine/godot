/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2008 Erwin Coumans  http://bulletphysics.com

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#include <stdio.h>
#include "LinearMath/btIDebugDraw.h"
#include "BulletCollision/CollisionDispatch/btGhostObject.h"
#include "BulletCollision/CollisionShapes/btMultiSphereShape.h"
#include "BulletCollision/BroadphaseCollision/btOverlappingPairCache.h"
#include "BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h"
#include "BulletCollision/CollisionDispatch/btCollisionWorld.h"
#include "LinearMath/btDefaultMotionState.h"
#include "btKinematicCharacterController.h"


// static helper method
static btVector3
getNormalizedVector(const btVector3& v)
{
	btVector3 n(0, 0, 0);

	if (v.length() > SIMD_EPSILON) {
		n = v.normalized();
	}
	return n;
}


///@todo Interact with dynamic objects,
///Ride kinematicly animated platforms properly
///More realistic (or maybe just a config option) falling
/// -> Should integrate falling velocity manually and use that in stepDown()
///Support jumping
///Support ducking
class btKinematicClosestNotMeRayResultCallback : public btCollisionWorld::ClosestRayResultCallback
{
public:
	btKinematicClosestNotMeRayResultCallback (btCollisionObject* me) : btCollisionWorld::ClosestRayResultCallback(btVector3(0.0, 0.0, 0.0), btVector3(0.0, 0.0, 0.0))
	{
		m_me = me;
	}

	virtual btScalar addSingleResult(btCollisionWorld::LocalRayResult& rayResult,bool normalInWorldSpace)
	{
		if (rayResult.m_collisionObject == m_me)
			return 1.0;

		return ClosestRayResultCallback::addSingleResult (rayResult, normalInWorldSpace);
	}
protected:
	btCollisionObject* m_me;
};

class btKinematicClosestNotMeConvexResultCallback : public btCollisionWorld::ClosestConvexResultCallback
{
public:
	btKinematicClosestNotMeConvexResultCallback (btCollisionObject* me, const btVector3& up, btScalar minSlopeDot)
	: btCollisionWorld::ClosestConvexResultCallback(btVector3(0.0, 0.0, 0.0), btVector3(0.0, 0.0, 0.0))
	, m_me(me)
	, m_up(up)
	, m_minSlopeDot(minSlopeDot)
	{
	}

	virtual btScalar addSingleResult(btCollisionWorld::LocalConvexResult& convexResult,bool normalInWorldSpace)
	{
		if (convexResult.m_hitCollisionObject == m_me)
			return btScalar(1.0);

		if (!convexResult.m_hitCollisionObject->hasContactResponse())
			return btScalar(1.0);

		btVector3 hitNormalWorld;
		if (normalInWorldSpace)
		{
			hitNormalWorld = convexResult.m_hitNormalLocal;
		} else
		{
			///need to transform normal into worldspace
			hitNormalWorld = convexResult.m_hitCollisionObject->getWorldTransform().getBasis()*convexResult.m_hitNormalLocal;
		}

		btScalar dotUp = m_up.dot(hitNormalWorld);
		if (dotUp < m_minSlopeDot) {
			return btScalar(1.0);
		}

		return ClosestConvexResultCallback::addSingleResult (convexResult, normalInWorldSpace);
	}
protected:
	btCollisionObject* m_me;
	const btVector3 m_up;
	btScalar m_minSlopeDot;
};

/*
 * Returns the reflection direction of a ray going 'direction' hitting a surface with normal 'normal'
 *
 * from: http://www-cs-students.stanford.edu/~adityagp/final/node3.html
 */
btVector3 btKinematicCharacterController::computeReflectionDirection (const btVector3& direction, const btVector3& normal)
{
	return direction - (btScalar(2.0) * direction.dot(normal)) * normal;
}

/*
 * Returns the portion of 'direction' that is parallel to 'normal'
 */
btVector3 btKinematicCharacterController::parallelComponent (const btVector3& direction, const btVector3& normal)
{
	btScalar magnitude = direction.dot(normal);
	return normal * magnitude;
}

/*
 * Returns the portion of 'direction' that is perpindicular to 'normal'
 */
btVector3 btKinematicCharacterController::perpindicularComponent (const btVector3& direction, const btVector3& normal)
{
	return direction - parallelComponent(direction, normal);
}

btKinematicCharacterController::btKinematicCharacterController (btPairCachingGhostObject* ghostObject,btConvexShape* convexShape,btScalar stepHeight, const btVector3& up)
{
	m_ghostObject = ghostObject;
	m_up.setValue(0.0f, 0.0f, 1.0f);
	m_jumpAxis.setValue(0.0f, 0.0f, 1.0f);
	m_addedMargin = 0.02;
	m_walkDirection.setValue(0.0,0.0,0.0);
	m_AngVel.setValue(0.0, 0.0, 0.0);
	m_useGhostObjectSweepTest = true;	
	m_turnAngle = btScalar(0.0);
	m_convexShape=convexShape;	
	m_useWalkDirection = true;	// use walk direction by default, legacy behavior
	m_velocityTimeInterval = 0.0;
	m_verticalVelocity = 0.0;
	m_verticalOffset = 0.0;
	m_gravity = 9.8 * 3.0 ; // 3G acceleration.
	m_fallSpeed = 55.0; // Terminal velocity of a sky diver in m/s.
	m_jumpSpeed = 10.0; // ?
	m_SetjumpSpeed = m_jumpSpeed;
	m_wasOnGround = false;
	m_wasJumping = false;
	m_interpolateUp = true;
	m_currentStepOffset = 0.0;
	m_maxPenetrationDepth = 0.2;
	full_drop = false;
	bounce_fix = false;
	m_linearDamping = btScalar(0.0);
	m_angularDamping = btScalar(0.0);

	setUp(up);
	setStepHeight(stepHeight);
	setMaxSlope(btRadians(45.0));
}

btKinematicCharacterController::~btKinematicCharacterController ()
{
}

btPairCachingGhostObject* btKinematicCharacterController::getGhostObject()
{
	return m_ghostObject;
}

bool btKinematicCharacterController::recoverFromPenetration ( btCollisionWorld* collisionWorld)
{
	// Here we must refresh the overlapping paircache as the penetrating movement itself or the
	// previous recovery iteration might have used setWorldTransform and pushed us into an object
	// that is not in the previous cache contents from the last timestep, as will happen if we
	// are pushed into a new AABB overlap. Unhandled this means the next convex sweep gets stuck.
	//
	// Do this by calling the broadphase's setAabb with the moved AABB, this will update the broadphase
	// paircache and the ghostobject's internal paircache at the same time.    /BW

	btVector3 minAabb, maxAabb;
	m_convexShape->getAabb(m_ghostObject->getWorldTransform(), minAabb,maxAabb);
	collisionWorld->getBroadphase()->setAabb(m_ghostObject->getBroadphaseHandle(), 
						 minAabb, 
						 maxAabb, 
						 collisionWorld->getDispatcher());
						 
	bool penetration = false;

	collisionWorld->getDispatcher()->dispatchAllCollisionPairs(m_ghostObject->getOverlappingPairCache(), collisionWorld->getDispatchInfo(), collisionWorld->getDispatcher());

	m_currentPosition = m_ghostObject->getWorldTransform().getOrigin();
	
//	btScalar maxPen = btScalar(0.0);
	for (int i = 0; i < m_ghostObject->getOverlappingPairCache()->getNumOverlappingPairs(); i++)
	{
		m_manifoldArray.resize(0);

		btBroadphasePair* collisionPair = &m_ghostObject->getOverlappingPairCache()->getOverlappingPairArray()[i];

		btCollisionObject* obj0 = static_cast<btCollisionObject*>(collisionPair->m_pProxy0->m_clientObject);
        btCollisionObject* obj1 = static_cast<btCollisionObject*>(collisionPair->m_pProxy1->m_clientObject);

		if ((obj0 && !obj0->hasContactResponse()) || (obj1 && !obj1->hasContactResponse()))
			continue;

		if (!needsCollision(obj0, obj1))
			continue;
		
		if (collisionPair->m_algorithm)
			collisionPair->m_algorithm->getAllContactManifolds(m_manifoldArray);

		
		for (int j=0;j<m_manifoldArray.size();j++)
		{
			btPersistentManifold* manifold = m_manifoldArray[j];
			btScalar directionSign = manifold->getBody0() == m_ghostObject ? btScalar(-1.0) : btScalar(1.0);
			for (int p=0;p<manifold->getNumContacts();p++)
			{
				const btManifoldPoint&pt = manifold->getContactPoint(p);

				btScalar dist = pt.getDistance();

				if (dist < -m_maxPenetrationDepth)
				{
					// TODO: cause problems on slopes, not sure if it is needed
					//if (dist < maxPen)
					//{
					//	maxPen = dist;
					//	m_touchingNormal = pt.m_normalWorldOnB * directionSign;//??

					//}
					m_currentPosition += pt.m_normalWorldOnB * directionSign * dist * btScalar(0.2);
					penetration = true;
				} else {
					//printf("touching %f\n", dist);
				}
			}
			
			//manifold->clearManifold();
		}
	}
	btTransform newTrans = m_ghostObject->getWorldTransform();
	newTrans.setOrigin(m_currentPosition);
	m_ghostObject->setWorldTransform(newTrans);
//	printf("m_touchingNormal = %f,%f,%f\n",m_touchingNormal[0],m_touchingNormal[1],m_touchingNormal[2]);
	return penetration;
}

void btKinematicCharacterController::stepUp ( btCollisionWorld* world)
{
	btScalar stepHeight = 0.0f;
	if (m_verticalVelocity < 0.0)
		stepHeight = m_stepHeight;

	// phase 1: up
	btTransform start, end;

	start.setIdentity ();
	end.setIdentity ();

	/* FIXME: Handle penetration properly */
	start.setOrigin(m_currentPosition);

	m_targetPosition = m_currentPosition + m_up * (stepHeight) + m_jumpAxis * ((m_verticalOffset > 0.f ? m_verticalOffset : 0.f));
	m_currentPosition = m_targetPosition;

	end.setOrigin (m_targetPosition);

	start.setRotation(m_currentOrientation);
	end.setRotation(m_targetOrientation);

	btKinematicClosestNotMeConvexResultCallback callback(m_ghostObject, -m_up, m_maxSlopeCosine);
	callback.m_collisionFilterGroup = getGhostObject()->getBroadphaseHandle()->m_collisionFilterGroup;
	callback.m_collisionFilterMask = getGhostObject()->getBroadphaseHandle()->m_collisionFilterMask;
	
	if (m_useGhostObjectSweepTest)
	{
		m_ghostObject->convexSweepTest (m_convexShape, start, end, callback, world->getDispatchInfo().m_allowedCcdPenetration);
	}
	else
	{
		world->convexSweepTest(m_convexShape, start, end, callback, world->getDispatchInfo().m_allowedCcdPenetration);
	}

	if (callback.hasHit() && m_ghostObject->hasContactResponse() && needsCollision(m_ghostObject, callback.m_hitCollisionObject))
	{
		// Only modify the position if the hit was a slope and not a wall or ceiling.
		if (callback.m_hitNormalWorld.dot(m_up) > 0.0)
		{
			// we moved up only a fraction of the step height
			m_currentStepOffset = stepHeight * callback.m_closestHitFraction;
			if (m_interpolateUp == true)
				m_currentPosition.setInterpolate3 (m_currentPosition, m_targetPosition, callback.m_closestHitFraction);
			else
				m_currentPosition = m_targetPosition;
		}

		btTransform& xform = m_ghostObject->getWorldTransform();
		xform.setOrigin(m_currentPosition);
		m_ghostObject->setWorldTransform(xform);

		// fix penetration if we hit a ceiling for example
		int numPenetrationLoops = 0;
		m_touchingContact = false;
		while (recoverFromPenetration(world))
		{
			numPenetrationLoops++;
			m_touchingContact = true;
			if (numPenetrationLoops > 4)
			{
				//printf("character could not recover from penetration = %d\n", numPenetrationLoops);
				break;
			}
		}
		m_targetPosition = m_ghostObject->getWorldTransform().getOrigin();
		m_currentPosition = m_targetPosition;

		if (m_verticalOffset > 0)
		{
			m_verticalOffset = 0.0;
			m_verticalVelocity = 0.0;
			m_currentStepOffset = m_stepHeight;
		}
	} else {
		m_currentStepOffset = stepHeight;
		m_currentPosition = m_targetPosition;
	}
}

bool btKinematicCharacterController::needsCollision(const btCollisionObject* body0, const btCollisionObject* body1)
{
	bool collides = (body0->getBroadphaseHandle()->m_collisionFilterGroup & body1->getBroadphaseHandle()->m_collisionFilterMask) != 0;
	collides = collides && (body1->getBroadphaseHandle()->m_collisionFilterGroup & body0->getBroadphaseHandle()->m_collisionFilterMask);
	return collides;
}

void btKinematicCharacterController::updateTargetPositionBasedOnCollision (const btVector3& hitNormal, btScalar tangentMag, btScalar normalMag)
{
	btVector3 movementDirection = m_targetPosition - m_currentPosition;
	btScalar movementLength = movementDirection.length();
	if (movementLength>SIMD_EPSILON)
	{
		movementDirection.normalize();

		btVector3 reflectDir = computeReflectionDirection (movementDirection, hitNormal);
		reflectDir.normalize();

		btVector3 parallelDir, perpindicularDir;

		parallelDir = parallelComponent (reflectDir, hitNormal);
		perpindicularDir = perpindicularComponent (reflectDir, hitNormal);

		m_targetPosition = m_currentPosition;
		if (0)//tangentMag != 0.0)
		{
			btVector3 parComponent = parallelDir * btScalar (tangentMag*movementLength);
//			printf("parComponent=%f,%f,%f\n",parComponent[0],parComponent[1],parComponent[2]);
			m_targetPosition +=  parComponent;
		}

		if (normalMag != 0.0)
		{
			btVector3 perpComponent = perpindicularDir * btScalar (normalMag*movementLength);
//			printf("perpComponent=%f,%f,%f\n",perpComponent[0],perpComponent[1],perpComponent[2]);
			m_targetPosition += perpComponent;
		}
	} else
	{
//		printf("movementLength don't normalize a zero vector\n");
	}
}

void btKinematicCharacterController::stepForwardAndStrafe ( btCollisionWorld* collisionWorld, const btVector3& walkMove)
{
	// printf("m_normalizedDirection=%f,%f,%f\n",
	// 	m_normalizedDirection[0],m_normalizedDirection[1],m_normalizedDirection[2]);
	// phase 2: forward and strafe
	btTransform start, end;

	m_targetPosition = m_currentPosition + walkMove;

	start.setIdentity ();
	end.setIdentity ();
	
	btScalar fraction = 1.0;
	btScalar distance2 = (m_currentPosition-m_targetPosition).length2();
//	printf("distance2=%f\n",distance2);

	int maxIter = 10;

	while (fraction > btScalar(0.01) && maxIter-- > 0)
	{
		start.setOrigin (m_currentPosition);
		end.setOrigin (m_targetPosition);
		btVector3 sweepDirNegative(m_currentPosition - m_targetPosition);

		start.setRotation(m_currentOrientation);
		end.setRotation(m_targetOrientation);

		btKinematicClosestNotMeConvexResultCallback callback (m_ghostObject, sweepDirNegative, btScalar(0.0));
		callback.m_collisionFilterGroup = getGhostObject()->getBroadphaseHandle()->m_collisionFilterGroup;
		callback.m_collisionFilterMask = getGhostObject()->getBroadphaseHandle()->m_collisionFilterMask;


		btScalar margin = m_convexShape->getMargin();
		m_convexShape->setMargin(margin + m_addedMargin);

		if (!(start == end))
		{
			if (m_useGhostObjectSweepTest)
			{
				m_ghostObject->convexSweepTest(m_convexShape, start, end, callback, collisionWorld->getDispatchInfo().m_allowedCcdPenetration);
			}
			else
			{
				collisionWorld->convexSweepTest(m_convexShape, start, end, callback, collisionWorld->getDispatchInfo().m_allowedCcdPenetration);
			}
		}
		m_convexShape->setMargin(margin);

		
		fraction -= callback.m_closestHitFraction;

		if (callback.hasHit() && m_ghostObject->hasContactResponse() && needsCollision(m_ghostObject, callback.m_hitCollisionObject))
		{	
			// we moved only a fraction
			//btScalar hitDistance;
			//hitDistance = (callback.m_hitPointWorld - m_currentPosition).length();

//			m_currentPosition.setInterpolate3 (m_currentPosition, m_targetPosition, callback.m_closestHitFraction);

			updateTargetPositionBasedOnCollision (callback.m_hitNormalWorld);
			btVector3 currentDir = m_targetPosition - m_currentPosition;
			distance2 = currentDir.length2();
			if (distance2 > SIMD_EPSILON)
			{
				currentDir.normalize();
				/* See Quake2: "If velocity is against original velocity, stop ead to avoid tiny oscilations in sloping corners." */
				if (currentDir.dot(m_normalizedDirection) <= btScalar(0.0))
				{
					break;
				}
			} else
			{
//				printf("currentDir: don't normalize a zero vector\n");
				break;
			}

		}
        else
        {
            m_currentPosition = m_targetPosition;
		}
	}
}

void btKinematicCharacterController::stepDown ( btCollisionWorld* collisionWorld, btScalar dt)
{
	btTransform start, end, end_double;
	bool runonce = false;

	// phase 3: down
	/*btScalar additionalDownStep = (m_wasOnGround && !onGround()) ? m_stepHeight : 0.0;
	btVector3 step_drop = m_up * (m_currentStepOffset + additionalDownStep);
	btScalar downVelocity = (additionalDownStep == 0.0 && m_verticalVelocity<0.0?-m_verticalVelocity:0.0) * dt;
	btVector3 gravity_drop = m_up * downVelocity; 
	m_targetPosition -= (step_drop + gravity_drop);*/

	btVector3 orig_position = m_targetPosition;
	
	btScalar downVelocity = (m_verticalVelocity<0.f?-m_verticalVelocity:0.f) * dt;

	if (m_verticalVelocity > 0.0)
		return;

	if(downVelocity > 0.0 && downVelocity > m_fallSpeed
		&& (m_wasOnGround || !m_wasJumping))
		downVelocity = m_fallSpeed;

	btVector3 step_drop = m_up * (m_currentStepOffset + downVelocity);
	m_targetPosition -= step_drop;

	btKinematicClosestNotMeConvexResultCallback callback(m_ghostObject, m_up, m_maxSlopeCosine);
        callback.m_collisionFilterGroup = getGhostObject()->getBroadphaseHandle()->m_collisionFilterGroup;
        callback.m_collisionFilterMask = getGhostObject()->getBroadphaseHandle()->m_collisionFilterMask;

	btKinematicClosestNotMeConvexResultCallback callback2(m_ghostObject, m_up, m_maxSlopeCosine);
        callback2.m_collisionFilterGroup = getGhostObject()->getBroadphaseHandle()->m_collisionFilterGroup;
        callback2.m_collisionFilterMask = getGhostObject()->getBroadphaseHandle()->m_collisionFilterMask;

	while (1)
	{
		start.setIdentity ();
		end.setIdentity ();

		end_double.setIdentity ();

		start.setOrigin (m_currentPosition);
		end.setOrigin (m_targetPosition);

		start.setRotation(m_currentOrientation);
		end.setRotation(m_targetOrientation);

		//set double test for 2x the step drop, to check for a large drop vs small drop
		end_double.setOrigin (m_targetPosition - step_drop);

		if (m_useGhostObjectSweepTest)
		{
			m_ghostObject->convexSweepTest (m_convexShape, start, end, callback, collisionWorld->getDispatchInfo().m_allowedCcdPenetration);

			if (!callback.hasHit() && m_ghostObject->hasContactResponse())
			{
				//test a double fall height, to see if the character should interpolate it's fall (full) or not (partial)
				m_ghostObject->convexSweepTest (m_convexShape, start, end_double, callback2, collisionWorld->getDispatchInfo().m_allowedCcdPenetration);
			}
		} else
		{
			collisionWorld->convexSweepTest (m_convexShape, start, end, callback, collisionWorld->getDispatchInfo().m_allowedCcdPenetration);

			if (!callback.hasHit() && m_ghostObject->hasContactResponse())
			{
				//test a double fall height, to see if the character should interpolate it's fall (large) or not (small)
				collisionWorld->convexSweepTest (m_convexShape, start, end_double, callback2, collisionWorld->getDispatchInfo().m_allowedCcdPenetration);
			}
		}
	
		btScalar downVelocity2 = (m_verticalVelocity<0.f?-m_verticalVelocity:0.f) * dt;
		bool has_hit;
		if (bounce_fix == true)
			has_hit = (callback.hasHit() || callback2.hasHit()) && m_ghostObject->hasContactResponse() && needsCollision(m_ghostObject, callback.m_hitCollisionObject);
		else
			has_hit = callback2.hasHit() && m_ghostObject->hasContactResponse() && needsCollision(m_ghostObject, callback2.m_hitCollisionObject);

		btScalar stepHeight = 0.0f;
		if (m_verticalVelocity < 0.0)
			stepHeight = m_stepHeight;

		if (downVelocity2 > 0.0 && downVelocity2 < stepHeight && has_hit == true && runonce == false
					&& (m_wasOnGround || !m_wasJumping))
		{
			//redo the velocity calculation when falling a small amount, for fast stairs motion
			//for larger falls, use the smoother/slower interpolated movement by not touching the target position

			m_targetPosition = orig_position;
			downVelocity = stepHeight;

			step_drop = m_up * (m_currentStepOffset + downVelocity);
			m_targetPosition -= step_drop;
			runonce = true;
			continue; //re-run previous tests
		}
		break;
	}

	if ((m_ghostObject->hasContactResponse() && (callback.hasHit() && needsCollision(m_ghostObject, callback.m_hitCollisionObject))) || runonce == true)
	{
		// we dropped a fraction of the height -> hit floor
		btScalar fraction = (m_currentPosition.getY() - callback.m_hitPointWorld.getY()) / 2;

		//printf("hitpoint: %g - pos %g\n", callback.m_hitPointWorld.getY(), m_currentPosition.getY());

		if (bounce_fix == true)
		{
			if (full_drop == true)
				m_currentPosition.setInterpolate3 (m_currentPosition, m_targetPosition, callback.m_closestHitFraction);
            else
				//due to errors in the closestHitFraction variable when used with large polygons, calculate the hit fraction manually
				m_currentPosition.setInterpolate3 (m_currentPosition, m_targetPosition, fraction);
		}
		else
			m_currentPosition.setInterpolate3 (m_currentPosition, m_targetPosition, callback.m_closestHitFraction);

		full_drop = false;

		m_verticalVelocity = 0.0;
		m_verticalOffset = 0.0;
		m_wasJumping = false;
	} else {
		// we dropped the full height

		full_drop = true;

		if (bounce_fix == true)
		{
			downVelocity = (m_verticalVelocity<0.f?-m_verticalVelocity:0.f) * dt;
			if (downVelocity > m_fallSpeed && (m_wasOnGround || !m_wasJumping))
			{
				m_targetPosition += step_drop; //undo previous target change
				downVelocity = m_fallSpeed;
				step_drop = m_up * (m_currentStepOffset + downVelocity);
				m_targetPosition -= step_drop;
			}
		}
		//printf("full drop - %g, %g\n", m_currentPosition.getY(), m_targetPosition.getY());

		m_currentPosition = m_targetPosition;
	}
}



void btKinematicCharacterController::setWalkDirection
(
const btVector3& walkDirection
)
{
	m_useWalkDirection = true;
	m_walkDirection = walkDirection;
	m_normalizedDirection = getNormalizedVector(m_walkDirection);
}



void btKinematicCharacterController::setVelocityForTimeInterval
(
const btVector3& velocity,
btScalar timeInterval
)
{
//	printf("setVelocity!\n");
//	printf("  interval: %f\n", timeInterval);
//	printf("  velocity: (%f, %f, %f)\n",
//		 velocity.x(), velocity.y(), velocity.z());

	m_useWalkDirection = false;
	m_walkDirection = velocity;
	m_normalizedDirection = getNormalizedVector(m_walkDirection);
	m_velocityTimeInterval += timeInterval;
}

void btKinematicCharacterController::setAngularVelocity(const btVector3& velocity)
{
	m_AngVel = velocity;
}

const btVector3& btKinematicCharacterController::getAngularVelocity() const
{
	return m_AngVel;
}

void btKinematicCharacterController::setLinearVelocity(const btVector3& velocity)
{
	m_walkDirection = velocity;

	// HACK: if we are moving in the direction of the up, treat it as a jump :(
	if (m_walkDirection.length2() > 0)
	{
		btVector3 w = velocity.normalized();
		btScalar c = w.dot(m_up);
		if (c != 0)
		{
			//there is a component in walkdirection for vertical velocity
			btVector3 upComponent = m_up * (btSin(SIMD_HALF_PI - btAcos(c)) * m_walkDirection.length());
			m_walkDirection -= upComponent;
			m_verticalVelocity = (c < 0.0f ? -1 : 1) * upComponent.length();
			
			if (c > 0.0f)
			{
				m_wasJumping = true;
				m_jumpPosition = m_ghostObject->getWorldTransform().getOrigin();
			}
		}
	}
	else
		m_verticalVelocity = 0.0f;
}

btVector3 btKinematicCharacterController::getLinearVelocity() const
{
	return m_walkDirection + (m_verticalVelocity * m_up);
}

void btKinematicCharacterController::reset ( btCollisionWorld* collisionWorld )
{
    m_verticalVelocity = 0.0;
    m_verticalOffset = 0.0;
    m_wasOnGround = false;
    m_wasJumping = false;
    m_walkDirection.setValue(0,0,0);
    m_velocityTimeInterval = 0.0;

    //clear pair cache
    btHashedOverlappingPairCache *cache = m_ghostObject->getOverlappingPairCache();
    while (cache->getOverlappingPairArray().size() > 0)
    {
            cache->removeOverlappingPair(cache->getOverlappingPairArray()[0].m_pProxy0, cache->getOverlappingPairArray()[0].m_pProxy1, collisionWorld->getDispatcher());
    }
}

void btKinematicCharacterController::warp (const btVector3& origin)
{
	btTransform xform;
	xform.setIdentity();
	xform.setOrigin (origin);
	m_ghostObject->setWorldTransform (xform);
}


void btKinematicCharacterController::preStep (  btCollisionWorld* collisionWorld)
{
	m_currentPosition = m_ghostObject->getWorldTransform().getOrigin();
	m_targetPosition = m_currentPosition;

	m_currentOrientation = m_ghostObject->getWorldTransform().getRotation();
	m_targetOrientation = m_currentOrientation;
//	printf("m_targetPosition=%f,%f,%f\n",m_targetPosition[0],m_targetPosition[1],m_targetPosition[2]);
}

void btKinematicCharacterController::playerStep (  btCollisionWorld* collisionWorld, btScalar dt)
{
//	printf("playerStep(): ");
//	printf("  dt = %f", dt);

	if (m_AngVel.length2() > 0.0f)
	{
		m_AngVel *= btPow(btScalar(1) - m_angularDamping, dt);
	}

	// integrate for angular velocity
	if (m_AngVel.length2() > 0.0f)
	{
		btTransform xform;
		xform = m_ghostObject->getWorldTransform();

		btQuaternion rot(m_AngVel.normalized(), m_AngVel.length() * dt);

		btQuaternion orn = rot * xform.getRotation();

		xform.setRotation(orn);
		m_ghostObject->setWorldTransform(xform);

		m_currentPosition = m_ghostObject->getWorldTransform().getOrigin();
		m_targetPosition = m_currentPosition;
		m_currentOrientation = m_ghostObject->getWorldTransform().getRotation();
		m_targetOrientation = m_currentOrientation;
	}

	// quick check...
	if (!m_useWalkDirection && (m_velocityTimeInterval <= 0.0)) {
//		printf("\n");
		return;		// no motion
	}

	m_wasOnGround = onGround();

	//btVector3 lvel = m_walkDirection;
	//btScalar c = 0.0f;
	
	if (m_walkDirection.length2() > 0)
	{
		// apply damping
		m_walkDirection *= btPow(btScalar(1) - m_linearDamping, dt);
	}

	m_verticalVelocity *= btPow(btScalar(1) - m_linearDamping, dt);
	
	// Update fall velocity.
	m_verticalVelocity -= m_gravity * dt;
	if (m_verticalVelocity > 0.0 && m_verticalVelocity > m_jumpSpeed)
	{
		m_verticalVelocity = m_jumpSpeed;
	}
	if (m_verticalVelocity < 0.0 && btFabs(m_verticalVelocity) > btFabs(m_fallSpeed))
	{
		m_verticalVelocity = -btFabs(m_fallSpeed);
	}
	m_verticalOffset = m_verticalVelocity * dt;

	btTransform xform;
	xform = m_ghostObject->getWorldTransform();

//	printf("walkDirection(%f,%f,%f)\n",walkDirection[0],walkDirection[1],walkDirection[2]);
//	printf("walkSpeed=%f\n",walkSpeed);

	stepUp(collisionWorld);
	//todo: Experimenting with behavior of controller when it hits a ceiling..
	//bool hitUp = stepUp (collisionWorld);	
	//if (hitUp)
	//{
	//	m_verticalVelocity -= m_gravity * dt;
	//	if (m_verticalVelocity > 0.0 && m_verticalVelocity > m_jumpSpeed)
	//	{
	//		m_verticalVelocity = m_jumpSpeed;
	//	}
	//	if (m_verticalVelocity < 0.0 && btFabs(m_verticalVelocity) > btFabs(m_fallSpeed))
	//	{
	//		m_verticalVelocity = -btFabs(m_fallSpeed);
	//	}
	//	m_verticalOffset = m_verticalVelocity * dt;

	//	xform = m_ghostObject->getWorldTransform();
	//}

	if (m_useWalkDirection) {
		stepForwardAndStrafe (collisionWorld, m_walkDirection);
	} else {
		//printf("  time: %f", m_velocityTimeInterval);
		// still have some time left for moving!
		btScalar dtMoving =
			(dt < m_velocityTimeInterval) ? dt : m_velocityTimeInterval;
		m_velocityTimeInterval -= dt;

		// how far will we move while we are moving?
		btVector3 move = m_walkDirection * dtMoving;

		//printf("  dtMoving: %f", dtMoving);

		// okay, step
		stepForwardAndStrafe(collisionWorld, move);
	}
	stepDown (collisionWorld, dt);

	//todo: Experimenting with max jump height
	//if (m_wasJumping)
	//{
	//	btScalar ds = m_currentPosition[m_upAxis] - m_jumpPosition[m_upAxis];
	//	if (ds > m_maxJumpHeight)
	//	{
	//		// substract the overshoot
	//		m_currentPosition[m_upAxis] -= ds - m_maxJumpHeight;

	//		// max height was reached, so potential energy is at max 
	//		// and kinematic energy is 0, thus velocity is 0.
	//		if (m_verticalVelocity > 0.0)
	//			m_verticalVelocity = 0.0;
	//	}
	//}
	// printf("\n");

	xform.setOrigin (m_currentPosition);
	m_ghostObject->setWorldTransform (xform);

	int numPenetrationLoops = 0;
	m_touchingContact = false;
	while (recoverFromPenetration(collisionWorld))
	{
		numPenetrationLoops++;
		m_touchingContact = true;
		if (numPenetrationLoops > 4)
		{
			//printf("character could not recover from penetration = %d\n", numPenetrationLoops);
			break;
		}
	}
}

void btKinematicCharacterController::setFallSpeed (btScalar fallSpeed)
{
	m_fallSpeed = fallSpeed;
}

void btKinematicCharacterController::setJumpSpeed (btScalar jumpSpeed)
{
	m_jumpSpeed = jumpSpeed;
	m_SetjumpSpeed = m_jumpSpeed;
}

void btKinematicCharacterController::setMaxJumpHeight (btScalar maxJumpHeight)
{
	m_maxJumpHeight = maxJumpHeight;
}

bool btKinematicCharacterController::canJump () const
{
	return onGround();
}

void btKinematicCharacterController::jump(const btVector3& v)
{
	m_jumpSpeed = v.length2() == 0 ? m_SetjumpSpeed : v.length();
	m_verticalVelocity = m_jumpSpeed;
	m_wasJumping = true;

	m_jumpAxis = v.length2() == 0 ? m_up : v.normalized();

	m_jumpPosition = m_ghostObject->getWorldTransform().getOrigin();

#if 0
	currently no jumping.
	btTransform xform;
	m_rigidBody->getMotionState()->getWorldTransform (xform);
	btVector3 up = xform.getBasis()[1];
	up.normalize ();
	btScalar magnitude = (btScalar(1.0)/m_rigidBody->getInvMass()) * btScalar(8.0);
	m_rigidBody->applyCentralImpulse (up * magnitude);
#endif
}

void btKinematicCharacterController::setGravity(const btVector3& gravity)
{
	if (gravity.length2() > 0) setUpVector(-gravity);

	m_gravity = gravity.length();
}

btVector3 btKinematicCharacterController::getGravity() const
{
	return -m_gravity * m_up;
}

void btKinematicCharacterController::setMaxSlope(btScalar slopeRadians)
{
	m_maxSlopeRadians = slopeRadians;
	m_maxSlopeCosine = btCos(slopeRadians);
}

btScalar btKinematicCharacterController::getMaxSlope() const
{
	return m_maxSlopeRadians;
}

void btKinematicCharacterController::setMaxPenetrationDepth(btScalar d)
{
	m_maxPenetrationDepth = d;
}

btScalar btKinematicCharacterController::getMaxPenetrationDepth() const
{
	return m_maxPenetrationDepth;
}

bool btKinematicCharacterController::onGround () const
{
	return (fabs(m_verticalVelocity) < SIMD_EPSILON) && (fabs(m_verticalOffset) < SIMD_EPSILON);
}

void btKinematicCharacterController::setStepHeight(btScalar h) 
{
	m_stepHeight = h;
}

btVector3* btKinematicCharacterController::getUpAxisDirections()
{
	static btVector3 sUpAxisDirection[3] = { btVector3(1.0f, 0.0f, 0.0f), btVector3(0.0f, 1.0f, 0.0f), btVector3(0.0f, 0.0f, 1.0f) };
	
	return sUpAxisDirection;
}

void btKinematicCharacterController::debugDraw(btIDebugDraw* debugDrawer)
{
}

void btKinematicCharacterController::setUpInterpolate(bool value)
{
	m_interpolateUp = value;
}

void btKinematicCharacterController::setUp(const btVector3& up)
{
	if (up.length2() > 0 && m_gravity > 0.0f)
	{
		setGravity(-m_gravity * up.normalized());
		return;
	}

	setUpVector(up);
}

void btKinematicCharacterController::setUpVector(const btVector3& up)
{
	if (m_up == up)
		return;

	btVector3 u = m_up;

	if (up.length2() > 0)
		m_up = up.normalized();
	else
		m_up = btVector3(0.0, 0.0, 0.0);

	if (!m_ghostObject) return;
	btQuaternion rot = getRotation(m_up, u);

	//set orientation with new up
	btTransform xform;
	xform = m_ghostObject->getWorldTransform();
	btQuaternion orn = rot.inverse() * xform.getRotation();
	xform.setRotation(orn);
	m_ghostObject->setWorldTransform(xform);
}

btQuaternion btKinematicCharacterController::getRotation(btVector3& v0, btVector3& v1) const
{
	if (v0.length2() == 0.0f || v1.length2() == 0.0f)
	{
		btQuaternion q;
		return q;
	}

	return shortestArcQuatNormalize2(v0, v1);
}

