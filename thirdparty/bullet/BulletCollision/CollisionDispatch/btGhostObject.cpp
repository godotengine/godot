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

#include "btGhostObject.h"
#include "btCollisionWorld.h"
#include "BulletCollision/CollisionShapes/btConvexShape.h"
#include "LinearMath/btAabbUtil2.h"

btGhostObject::btGhostObject()
{
	m_internalType = CO_GHOST_OBJECT;
}

btGhostObject::~btGhostObject()
{
	///btGhostObject should have been removed from the world, so no overlapping objects
	btAssert(!m_overlappingObjects.size());
}

void btGhostObject::addOverlappingObjectInternal(btBroadphaseProxy* otherProxy, btBroadphaseProxy* thisProxy)
{
	btCollisionObject* otherObject = (btCollisionObject*)otherProxy->m_clientObject;
	btAssert(otherObject);
	///if this linearSearch becomes too slow (too many overlapping objects) we should add a more appropriate data structure
	int index = m_overlappingObjects.findLinearSearch(otherObject);
	if (index == m_overlappingObjects.size())
	{
		//not found
		m_overlappingObjects.push_back(otherObject);
	}
}

void btGhostObject::removeOverlappingObjectInternal(btBroadphaseProxy* otherProxy, btDispatcher* dispatcher, btBroadphaseProxy* thisProxy)
{
	btCollisionObject* otherObject = (btCollisionObject*)otherProxy->m_clientObject;
	btAssert(otherObject);
	int index = m_overlappingObjects.findLinearSearch(otherObject);
	if (index < m_overlappingObjects.size())
	{
		m_overlappingObjects[index] = m_overlappingObjects[m_overlappingObjects.size() - 1];
		m_overlappingObjects.pop_back();
	}
}

btPairCachingGhostObject::btPairCachingGhostObject()
{
	m_hashPairCache = new (btAlignedAlloc(sizeof(btHashedOverlappingPairCache), 16)) btHashedOverlappingPairCache();
}

btPairCachingGhostObject::~btPairCachingGhostObject()
{
	m_hashPairCache->~btHashedOverlappingPairCache();
	btAlignedFree(m_hashPairCache);
}

void btPairCachingGhostObject::addOverlappingObjectInternal(btBroadphaseProxy* otherProxy, btBroadphaseProxy* thisProxy)
{
	btBroadphaseProxy* actualThisProxy = thisProxy ? thisProxy : getBroadphaseHandle();
	btAssert(actualThisProxy);

	btCollisionObject* otherObject = (btCollisionObject*)otherProxy->m_clientObject;
	btAssert(otherObject);
	int index = m_overlappingObjects.findLinearSearch(otherObject);
	if (index == m_overlappingObjects.size())
	{
		m_overlappingObjects.push_back(otherObject);
		m_hashPairCache->addOverlappingPair(actualThisProxy, otherProxy);
	}
}

void btPairCachingGhostObject::removeOverlappingObjectInternal(btBroadphaseProxy* otherProxy, btDispatcher* dispatcher, btBroadphaseProxy* thisProxy1)
{
	btCollisionObject* otherObject = (btCollisionObject*)otherProxy->m_clientObject;
	btBroadphaseProxy* actualThisProxy = thisProxy1 ? thisProxy1 : getBroadphaseHandle();
	btAssert(actualThisProxy);

	btAssert(otherObject);
	int index = m_overlappingObjects.findLinearSearch(otherObject);
	if (index < m_overlappingObjects.size())
	{
		m_overlappingObjects[index] = m_overlappingObjects[m_overlappingObjects.size() - 1];
		m_overlappingObjects.pop_back();
		m_hashPairCache->removeOverlappingPair(actualThisProxy, otherProxy, dispatcher);
	}
}

void btGhostObject::convexSweepTest(const btConvexShape* castShape, const btTransform& convexFromWorld, const btTransform& convexToWorld, btCollisionWorld::ConvexResultCallback& resultCallback, btScalar allowedCcdPenetration) const
{
	btTransform convexFromTrans, convexToTrans;
	convexFromTrans = convexFromWorld;
	convexToTrans = convexToWorld;
	btVector3 castShapeAabbMin, castShapeAabbMax;
	/* Compute AABB that encompasses angular movement */
	{
		btVector3 linVel, angVel;
		btTransformUtil::calculateVelocity(convexFromTrans, convexToTrans, 1.0, linVel, angVel);
		btTransform R;
		R.setIdentity();
		R.setRotation(convexFromTrans.getRotation());
		castShape->calculateTemporalAabb(R, linVel, angVel, 1.0, castShapeAabbMin, castShapeAabbMax);
	}

	/// go over all objects, and if the ray intersects their aabb + cast shape aabb,
	// do a ray-shape query using convexCaster (CCD)
	int i;
	for (i = 0; i < m_overlappingObjects.size(); i++)
	{
		btCollisionObject* collisionObject = m_overlappingObjects[i];
		//only perform raycast if filterMask matches
		if (resultCallback.needsCollision(collisionObject->getBroadphaseHandle()))
		{
			//RigidcollisionObject* collisionObject = ctrl->GetRigidcollisionObject();
			btVector3 collisionObjectAabbMin, collisionObjectAabbMax;
			collisionObject->getCollisionShape()->getAabb(collisionObject->getWorldTransform(), collisionObjectAabbMin, collisionObjectAabbMax);
			AabbExpand(collisionObjectAabbMin, collisionObjectAabbMax, castShapeAabbMin, castShapeAabbMax);
			btScalar hitLambda = btScalar(1.);  //could use resultCallback.m_closestHitFraction, but needs testing
			btVector3 hitNormal;
			if (btRayAabb(convexFromWorld.getOrigin(), convexToWorld.getOrigin(), collisionObjectAabbMin, collisionObjectAabbMax, hitLambda, hitNormal))
			{
				btCollisionWorld::objectQuerySingle(castShape, convexFromTrans, convexToTrans,
													collisionObject,
													collisionObject->getCollisionShape(),
													collisionObject->getWorldTransform(),
													resultCallback,
													allowedCcdPenetration);
			}
		}
	}
}

void btGhostObject::rayTest(const btVector3& rayFromWorld, const btVector3& rayToWorld, btCollisionWorld::RayResultCallback& resultCallback) const
{
	btTransform rayFromTrans;
	rayFromTrans.setIdentity();
	rayFromTrans.setOrigin(rayFromWorld);
	btTransform rayToTrans;
	rayToTrans.setIdentity();
	rayToTrans.setOrigin(rayToWorld);

	int i;
	for (i = 0; i < m_overlappingObjects.size(); i++)
	{
		btCollisionObject* collisionObject = m_overlappingObjects[i];
		//only perform raycast if filterMask matches
		if (resultCallback.needsCollision(collisionObject->getBroadphaseHandle()))
		{
			btCollisionWorld::rayTestSingle(rayFromTrans, rayToTrans,
											collisionObject,
											collisionObject->getCollisionShape(),
											collisionObject->getWorldTransform(),
											resultCallback);
		}
	}
}
