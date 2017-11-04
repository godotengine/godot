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

#include "btConvexPointCloudShape.h"
#include "BulletCollision/CollisionShapes/btCollisionMargin.h"

#include "LinearMath/btQuaternion.h"

void btConvexPointCloudShape::setLocalScaling(const btVector3& scaling)
{
	m_localScaling = scaling;
	recalcLocalAabb();
}

#ifndef __SPU__
btVector3	btConvexPointCloudShape::localGetSupportingVertexWithoutMargin(const btVector3& vec0)const
{
	btVector3 supVec(btScalar(0.),btScalar(0.),btScalar(0.));
	btScalar maxDot = btScalar(-BT_LARGE_FLOAT);

	btVector3 vec = vec0;
	btScalar lenSqr = vec.length2();
	if (lenSqr < btScalar(0.0001))
	{
		vec.setValue(1,0,0);
	} else
	{
		btScalar rlen = btScalar(1.) / btSqrt(lenSqr );
		vec *= rlen;
	}
    
    if( m_numPoints > 0 )
    {
        // Here we take advantage of dot(a*b, c) = dot( a, b*c) to do less work. Note this transformation is true mathematically, not numerically.
    //    btVector3 scaled = vec * m_localScaling;
        int index = (int) vec.maxDot( &m_unscaledPoints[0], m_numPoints, maxDot);   //FIXME: may violate encapsulation of m_unscaledPoints
        return getScaledPoint(index);
    }

	return supVec;
}

void	btConvexPointCloudShape::batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors,btVector3* supportVerticesOut,int numVectors) const
{
    for( int j = 0; j < numVectors; j++ )
    {
        const btVector3& vec = vectors[j] * m_localScaling;  // dot( a*c, b) = dot(a, b*c)
        btScalar maxDot;
        int index = (int) vec.maxDot( &m_unscaledPoints[0], m_numPoints, maxDot);
        supportVerticesOut[j][3] = btScalar(-BT_LARGE_FLOAT);
        if( 0 <= index )
        {
            //WARNING: don't swap next lines, the w component would get overwritten!
            supportVerticesOut[j] = getScaledPoint(index);
            supportVerticesOut[j][3] = maxDot;
        }
    }

}
	


btVector3	btConvexPointCloudShape::localGetSupportingVertex(const btVector3& vec)const
{
	btVector3 supVertex = localGetSupportingVertexWithoutMargin(vec);

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
}


#endif






//currently just for debugging (drawing), perhaps future support for algebraic continuous collision detection
//Please note that you can debug-draw btConvexHullShape with the Raytracer Demo
int	btConvexPointCloudShape::getNumVertices() const
{
	return m_numPoints;
}

int btConvexPointCloudShape::getNumEdges() const
{
	return 0;
}

void btConvexPointCloudShape::getEdge(int i,btVector3& pa,btVector3& pb) const
{
	btAssert (0);
}

void btConvexPointCloudShape::getVertex(int i,btVector3& vtx) const
{
	vtx = m_unscaledPoints[i]*m_localScaling;
}

int	btConvexPointCloudShape::getNumPlanes() const
{
	return 0;
}

void btConvexPointCloudShape::getPlane(btVector3& ,btVector3& ,int ) const
{

	btAssert(0);
}

//not yet
bool btConvexPointCloudShape::isInside(const btVector3& ,btScalar ) const
{
	btAssert(0);
	return false;
}

