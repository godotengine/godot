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

#ifndef BT_RAYCAST_TRI_CALLBACK_H
#define BT_RAYCAST_TRI_CALLBACK_H

#include "BulletCollision/CollisionShapes/btTriangleCallback.h"
#include "LinearMath/btTransform.h"
struct btBroadphaseProxy;
class btConvexShape;

class  btTriangleRaycastCallback: public btTriangleCallback
{
public:

	//input
	btVector3 m_from;
	btVector3 m_to;

   //@BP Mod - allow backface filtering and unflipped normals
   enum EFlags
   {
	  kF_None                 = 0,
      kF_FilterBackfaces      = 1 << 0,
      kF_KeepUnflippedNormal  = 1 << 1,   // Prevents returned face normal getting flipped when a ray hits a back-facing triangle
		///SubSimplexConvexCastRaytest is the default, even if kF_None is set.
	  kF_UseSubSimplexConvexCastRaytest = 1 << 2,   // Uses an approximate but faster ray versus convex intersection algorithm
	  kF_UseGjkConvexCastRaytest = 1 << 3,
      kF_Terminator        = 0xFFFFFFFF
   };
   unsigned int m_flags;

	btScalar	m_hitFraction;

	btTriangleRaycastCallback(const btVector3& from,const btVector3& to, unsigned int flags=0);
	
	virtual void processTriangle(btVector3* triangle, int partId, int triangleIndex);

	virtual btScalar reportHit(const btVector3& hitNormalLocal, btScalar hitFraction, int partId, int triangleIndex ) = 0;
	
};

class btTriangleConvexcastCallback : public btTriangleCallback
{
public:
	const btConvexShape* m_convexShape;
	btTransform m_convexShapeFrom;
	btTransform m_convexShapeTo;
	btTransform m_triangleToWorld;
	btScalar m_hitFraction;
	btScalar m_triangleCollisionMargin;
	btScalar m_allowedPenetration;

	btTriangleConvexcastCallback (const btConvexShape* convexShape, const btTransform& convexShapeFrom, const btTransform& convexShapeTo, const btTransform& triangleToWorld, const btScalar triangleCollisionMargin);

	virtual void processTriangle (btVector3* triangle, int partId, int triangleIndex);

	virtual btScalar reportHit (const btVector3& hitNormalLocal, const btVector3& hitPointLocal, btScalar hitFraction, int partId, int triangleIndex) = 0;
};

#endif //BT_RAYCAST_TRI_CALLBACK_H

