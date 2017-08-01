/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2009 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#include "btConvexInternalShape.h"



btConvexInternalShape::btConvexInternalShape()
: m_localScaling(btScalar(1.),btScalar(1.),btScalar(1.)),
m_collisionMargin(CONVEX_DISTANCE_MARGIN)
{
}


void	btConvexInternalShape::setLocalScaling(const btVector3& scaling)
{
	m_localScaling = scaling.absolute();
}



void	btConvexInternalShape::getAabbSlow(const btTransform& trans,btVector3&minAabb,btVector3&maxAabb) const
{
#ifndef __SPU__
	//use localGetSupportingVertexWithoutMargin?
	btScalar margin = getMargin();
	for (int i=0;i<3;i++)
	{
		btVector3 vec(btScalar(0.),btScalar(0.),btScalar(0.));
		vec[i] = btScalar(1.);

		btVector3 sv = localGetSupportingVertex(vec*trans.getBasis());

		btVector3 tmp = trans(sv);
		maxAabb[i] = tmp[i]+margin;
		vec[i] = btScalar(-1.);
		tmp = trans(localGetSupportingVertex(vec*trans.getBasis()));
		minAabb[i] = tmp[i]-margin;
	}
#endif
}



btVector3	btConvexInternalShape::localGetSupportingVertex(const btVector3& vec)const
{
#ifndef __SPU__

	 btVector3	supVertex = localGetSupportingVertexWithoutMargin(vec);

	if ( getMargin()!=btScalar(0.) )
	{
		btVector3 vecnorm = vec;
		if (vecnorm .length2() < (SIMD_EPSILON*SIMD_EPSILON))
		{
			vecnorm.setValue(btScalar(-1.),btScalar(-1.),btScalar(-1.));
		} 
		vecnorm.normalize();
		supVertex+= getMargin() * vecnorm;
	}
	return supVertex;

#else
	btAssert(0);
	return btVector3(0,0,0);
#endif //__SPU__

 }


btConvexInternalAabbCachingShape::btConvexInternalAabbCachingShape()
	:	btConvexInternalShape(),
m_localAabbMin(1,1,1),
m_localAabbMax(-1,-1,-1),
m_isLocalAabbValid(false)
{
}


void btConvexInternalAabbCachingShape::getAabb(const btTransform& trans,btVector3& aabbMin,btVector3& aabbMax) const
{
	getNonvirtualAabb(trans,aabbMin,aabbMax,getMargin());
}

void	btConvexInternalAabbCachingShape::setLocalScaling(const btVector3& scaling)
{
	btConvexInternalShape::setLocalScaling(scaling);
	recalcLocalAabb();
}


void	btConvexInternalAabbCachingShape::recalcLocalAabb()
{
	m_isLocalAabbValid = true;
	
	#if 1
	static const btVector3 _directions[] =
	{
		btVector3( 1.,  0.,  0.),
		btVector3( 0.,  1.,  0.),
		btVector3( 0.,  0.,  1.),
		btVector3( -1., 0.,  0.),
		btVector3( 0., -1.,  0.),
		btVector3( 0.,  0., -1.)
	};
	
	btVector3 _supporting[] =
	{
		btVector3( 0., 0., 0.),
		btVector3( 0., 0., 0.),
		btVector3( 0., 0., 0.),
		btVector3( 0., 0., 0.),
		btVector3( 0., 0., 0.),
		btVector3( 0., 0., 0.)
	};
	
	batchedUnitVectorGetSupportingVertexWithoutMargin(_directions, _supporting, 6);
	
	for ( int i = 0; i < 3; ++i )
	{
		m_localAabbMax[i] = _supporting[i][i] + m_collisionMargin;
		m_localAabbMin[i] = _supporting[i + 3][i] - m_collisionMargin;
	}
	
	#else

	for (int i=0;i<3;i++)
	{
		btVector3 vec(btScalar(0.),btScalar(0.),btScalar(0.));
		vec[i] = btScalar(1.);
		btVector3 tmp = localGetSupportingVertex(vec);
		m_localAabbMax[i] = tmp[i]+m_collisionMargin;
		vec[i] = btScalar(-1.);
		tmp = localGetSupportingVertex(vec);
		m_localAabbMin[i] = tmp[i]-m_collisionMargin;
	}
	#endif
}
