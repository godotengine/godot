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

//#include <stdio.h>

#include "BulletCollision/CollisionShapes/btConvexShape.h"
#include "BulletCollision/CollisionShapes/btTriangleShape.h"
#include "BulletCollision/NarrowPhaseCollision/btSubSimplexConvexCast.h"
#include "BulletCollision/NarrowPhaseCollision/btGjkConvexCast.h"
#include "BulletCollision/NarrowPhaseCollision/btContinuousConvexCollision.h"
#include "BulletCollision/NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h"
#include "btRaycastCallback.h"

btTriangleRaycastCallback::btTriangleRaycastCallback(const btVector3& from,const btVector3& to, unsigned int flags)
	:
	m_from(from),
	m_to(to),
   //@BP Mod
   m_flags(flags),
	m_hitFraction(btScalar(1.))
{

}



void btTriangleRaycastCallback::processTriangle(btVector3* triangle,int partId, int triangleIndex)
{
	const btVector3 &vert0=triangle[0];
	const btVector3 &vert1=triangle[1];
	const btVector3 &vert2=triangle[2];

	btVector3 v10; v10 = vert1 - vert0 ;
	btVector3 v20; v20 = vert2 - vert0 ;

	btVector3 triangleNormal; triangleNormal = v10.cross( v20 );
	
	const btScalar dist = vert0.dot(triangleNormal);
	btScalar dist_a = triangleNormal.dot(m_from) ;
	dist_a-= dist;
	btScalar dist_b = triangleNormal.dot(m_to);
	dist_b -= dist;

	if ( dist_a * dist_b >= btScalar(0.0) )
	{
		return ; // same sign
	}

	if (((m_flags & kF_FilterBackfaces) != 0) && (dist_a <= btScalar(0.0)))
	{
		// Backface, skip check
		return;
	}

	
	const btScalar proj_length=dist_a-dist_b;
	const btScalar distance = (dist_a)/(proj_length);
	// Now we have the intersection point on the plane, we'll see if it's inside the triangle
	// Add an epsilon as a tolerance for the raycast,
	// in case the ray hits exacly on the edge of the triangle.
	// It must be scaled for the triangle size.
	
	if(distance < m_hitFraction)
	{
		

		btScalar edge_tolerance =triangleNormal.length2();		
		edge_tolerance *= btScalar(-0.0001);
		btVector3 point; point.setInterpolate3( m_from, m_to, distance);
		{
			btVector3 v0p; v0p = vert0 - point;
			btVector3 v1p; v1p = vert1 - point;
			btVector3 cp0; cp0 = v0p.cross( v1p );

			if ( (btScalar)(cp0.dot(triangleNormal)) >=edge_tolerance) 
			{
						

				btVector3 v2p; v2p = vert2 -  point;
				btVector3 cp1;
				cp1 = v1p.cross( v2p);
				if ( (btScalar)(cp1.dot(triangleNormal)) >=edge_tolerance) 
				{
					btVector3 cp2;
					cp2 = v2p.cross(v0p);
					
					if ( (btScalar)(cp2.dot(triangleNormal)) >=edge_tolerance) 
					{
					  //@BP Mod
					  // Triangle normal isn't normalized
				      triangleNormal.normalize();

					 //@BP Mod - Allow for unflipped normal when raycasting against backfaces
						if (((m_flags & kF_KeepUnflippedNormal) == 0) && (dist_a <= btScalar(0.0)))
						{
							m_hitFraction = reportHit(-triangleNormal,distance,partId,triangleIndex);
						}
						else
						{
							m_hitFraction = reportHit(triangleNormal,distance,partId,triangleIndex);
						}
					}
				}
			}
		}
	}
}


btTriangleConvexcastCallback::btTriangleConvexcastCallback (const btConvexShape* convexShape, const btTransform& convexShapeFrom, const btTransform& convexShapeTo, const btTransform& triangleToWorld, const btScalar triangleCollisionMargin)
{
	m_convexShape = convexShape;
	m_convexShapeFrom = convexShapeFrom;
	m_convexShapeTo = convexShapeTo;
	m_triangleToWorld = triangleToWorld;
	m_hitFraction = 1.0f;
	m_triangleCollisionMargin = triangleCollisionMargin;
	m_allowedPenetration = 0.f;
}

void
btTriangleConvexcastCallback::processTriangle (btVector3* triangle, int partId, int triangleIndex)
{
	btTriangleShape triangleShape (triangle[0], triangle[1], triangle[2]);
    triangleShape.setMargin(m_triangleCollisionMargin);

	btVoronoiSimplexSolver	simplexSolver;
	btGjkEpaPenetrationDepthSolver	gjkEpaPenetrationSolver;

//#define  USE_SUBSIMPLEX_CONVEX_CAST 1
//if you reenable USE_SUBSIMPLEX_CONVEX_CAST see commented out code below
#ifdef USE_SUBSIMPLEX_CONVEX_CAST
	btSubsimplexConvexCast convexCaster(m_convexShape, &triangleShape, &simplexSolver);
#else
	//btGjkConvexCast	convexCaster(m_convexShape,&triangleShape,&simplexSolver);
	btContinuousConvexCollision convexCaster(m_convexShape,&triangleShape,&simplexSolver,&gjkEpaPenetrationSolver);
#endif //#USE_SUBSIMPLEX_CONVEX_CAST
	
	btConvexCast::CastResult castResult;
	castResult.m_fraction = btScalar(1.);
	castResult.m_allowedPenetration = m_allowedPenetration;
	if (convexCaster.calcTimeOfImpact(m_convexShapeFrom,m_convexShapeTo,m_triangleToWorld, m_triangleToWorld, castResult))
	{
		//add hit
		if (castResult.m_normal.length2() > btScalar(0.0001))
		{					
			if (castResult.m_fraction < m_hitFraction)
			{
/* btContinuousConvexCast's normal is already in world space */
/*
#ifdef USE_SUBSIMPLEX_CONVEX_CAST
				//rotate normal into worldspace
				castResult.m_normal = m_convexShapeFrom.getBasis() * castResult.m_normal;
#endif //USE_SUBSIMPLEX_CONVEX_CAST
*/
				castResult.m_normal.normalize();

				reportHit (castResult.m_normal,
							castResult.m_hitPoint,
							castResult.m_fraction,
							partId,
							triangleIndex);
			}
		}
	}
}
