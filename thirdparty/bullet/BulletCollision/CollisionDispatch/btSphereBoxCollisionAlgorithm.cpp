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

#include "btSphereBoxCollisionAlgorithm.h"
#include "BulletCollision/CollisionDispatch/btCollisionDispatcher.h"
#include "BulletCollision/CollisionShapes/btSphereShape.h"
#include "BulletCollision/CollisionShapes/btBoxShape.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h"
//#include <stdio.h>

btSphereBoxCollisionAlgorithm::btSphereBoxCollisionAlgorithm(btPersistentManifold* mf, const btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* col0Wrap, const btCollisionObjectWrapper* col1Wrap, bool isSwapped)
	: btActivatingCollisionAlgorithm(ci, col0Wrap, col1Wrap),
	  m_ownManifold(false),
	  m_manifoldPtr(mf),
	  m_isSwapped(isSwapped)
{
	const btCollisionObjectWrapper* sphereObjWrap = m_isSwapped ? col1Wrap : col0Wrap;
	const btCollisionObjectWrapper* boxObjWrap = m_isSwapped ? col0Wrap : col1Wrap;

	if (!m_manifoldPtr && m_dispatcher->needsCollision(sphereObjWrap->getCollisionObject(), boxObjWrap->getCollisionObject()))
	{
		m_manifoldPtr = m_dispatcher->getNewManifold(sphereObjWrap->getCollisionObject(), boxObjWrap->getCollisionObject());
		m_ownManifold = true;
	}
}

btSphereBoxCollisionAlgorithm::~btSphereBoxCollisionAlgorithm()
{
	if (m_ownManifold)
	{
		if (m_manifoldPtr)
			m_dispatcher->releaseManifold(m_manifoldPtr);
	}
}

void btSphereBoxCollisionAlgorithm::processCollision(const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut)
{
	(void)dispatchInfo;
	(void)resultOut;
	if (!m_manifoldPtr)
		return;

	const btCollisionObjectWrapper* sphereObjWrap = m_isSwapped ? body1Wrap : body0Wrap;
	const btCollisionObjectWrapper* boxObjWrap = m_isSwapped ? body0Wrap : body1Wrap;

	btVector3 pOnBox;

	btVector3 normalOnSurfaceB;
	btScalar penetrationDepth;
	btVector3 sphereCenter = sphereObjWrap->getWorldTransform().getOrigin();
	const btSphereShape* sphere0 = (const btSphereShape*)sphereObjWrap->getCollisionShape();
	btScalar radius = sphere0->getRadius();
	btScalar maxContactDistance = m_manifoldPtr->getContactBreakingThreshold();

	resultOut->setPersistentManifold(m_manifoldPtr);

	if (getSphereDistance(boxObjWrap, pOnBox, normalOnSurfaceB, penetrationDepth, sphereCenter, radius, maxContactDistance))
	{
		/// report a contact. internally this will be kept persistent, and contact reduction is done
		resultOut->addContactPoint(normalOnSurfaceB, pOnBox, penetrationDepth);
	}

	if (m_ownManifold)
	{
		if (m_manifoldPtr->getNumContacts())
		{
			resultOut->refreshContactPoints();
		}
	}
}

btScalar btSphereBoxCollisionAlgorithm::calculateTimeOfImpact(btCollisionObject* col0, btCollisionObject* col1, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut)
{
	(void)resultOut;
	(void)dispatchInfo;
	(void)col0;
	(void)col1;

	//not yet
	return btScalar(1.);
}

bool btSphereBoxCollisionAlgorithm::getSphereDistance(const btCollisionObjectWrapper* boxObjWrap, btVector3& pointOnBox, btVector3& normal, btScalar& penetrationDepth, const btVector3& sphereCenter, btScalar fRadius, btScalar maxContactDistance)
{
	const btBoxShape* boxShape = (const btBoxShape*)boxObjWrap->getCollisionShape();
	btVector3 const& boxHalfExtent = boxShape->getHalfExtentsWithoutMargin();
	btScalar boxMargin = boxShape->getMargin();
	penetrationDepth = 1.0f;

	// convert the sphere position to the box's local space
	btTransform const& m44T = boxObjWrap->getWorldTransform();
	btVector3 sphereRelPos = m44T.invXform(sphereCenter);

	// Determine the closest point to the sphere center in the box
	btVector3 closestPoint = sphereRelPos;
	closestPoint.setX(btMin(boxHalfExtent.getX(), closestPoint.getX()));
	closestPoint.setX(btMax(-boxHalfExtent.getX(), closestPoint.getX()));
	closestPoint.setY(btMin(boxHalfExtent.getY(), closestPoint.getY()));
	closestPoint.setY(btMax(-boxHalfExtent.getY(), closestPoint.getY()));
	closestPoint.setZ(btMin(boxHalfExtent.getZ(), closestPoint.getZ()));
	closestPoint.setZ(btMax(-boxHalfExtent.getZ(), closestPoint.getZ()));

	btScalar intersectionDist = fRadius + boxMargin;
	btScalar contactDist = intersectionDist + maxContactDistance;
	normal = sphereRelPos - closestPoint;

	//if there is no penetration, we are done
	btScalar dist2 = normal.length2();
	if (dist2 > contactDist * contactDist)
	{
		return false;
	}

	btScalar distance;

	//special case if the sphere center is inside the box
	if (dist2 <= SIMD_EPSILON)
	{
		distance = -getSpherePenetration(boxHalfExtent, sphereRelPos, closestPoint, normal);
	}
	else  //compute the penetration details
	{
		distance = normal.length();
		normal /= distance;
	}

	pointOnBox = closestPoint + normal * boxMargin;
	//	v3PointOnSphere = sphereRelPos - (normal * fRadius);
	penetrationDepth = distance - intersectionDist;

	// transform back in world space
	btVector3 tmp = m44T(pointOnBox);
	pointOnBox = tmp;
	//	tmp = m44T(v3PointOnSphere);
	//	v3PointOnSphere = tmp;
	tmp = m44T.getBasis() * normal;
	normal = tmp;

	return true;
}

btScalar btSphereBoxCollisionAlgorithm::getSpherePenetration(btVector3 const& boxHalfExtent, btVector3 const& sphereRelPos, btVector3& closestPoint, btVector3& normal)
{
	//project the center of the sphere on the closest face of the box
	btScalar faceDist = boxHalfExtent.getX() - sphereRelPos.getX();
	btScalar minDist = faceDist;
	closestPoint.setX(boxHalfExtent.getX());
	normal.setValue(btScalar(1.0f), btScalar(0.0f), btScalar(0.0f));

	faceDist = boxHalfExtent.getX() + sphereRelPos.getX();
	if (faceDist < minDist)
	{
		minDist = faceDist;
		closestPoint = sphereRelPos;
		closestPoint.setX(-boxHalfExtent.getX());
		normal.setValue(btScalar(-1.0f), btScalar(0.0f), btScalar(0.0f));
	}

	faceDist = boxHalfExtent.getY() - sphereRelPos.getY();
	if (faceDist < minDist)
	{
		minDist = faceDist;
		closestPoint = sphereRelPos;
		closestPoint.setY(boxHalfExtent.getY());
		normal.setValue(btScalar(0.0f), btScalar(1.0f), btScalar(0.0f));
	}

	faceDist = boxHalfExtent.getY() + sphereRelPos.getY();
	if (faceDist < minDist)
	{
		minDist = faceDist;
		closestPoint = sphereRelPos;
		closestPoint.setY(-boxHalfExtent.getY());
		normal.setValue(btScalar(0.0f), btScalar(-1.0f), btScalar(0.0f));
	}

	faceDist = boxHalfExtent.getZ() - sphereRelPos.getZ();
	if (faceDist < minDist)
	{
		minDist = faceDist;
		closestPoint = sphereRelPos;
		closestPoint.setZ(boxHalfExtent.getZ());
		normal.setValue(btScalar(0.0f), btScalar(0.0f), btScalar(1.0f));
	}

	faceDist = boxHalfExtent.getZ() + sphereRelPos.getZ();
	if (faceDist < minDist)
	{
		minDist = faceDist;
		closestPoint = sphereRelPos;
		closestPoint.setZ(-boxHalfExtent.getZ());
		normal.setValue(btScalar(0.0f), btScalar(0.0f), btScalar(-1.0f));
	}

	return minDist;
}
