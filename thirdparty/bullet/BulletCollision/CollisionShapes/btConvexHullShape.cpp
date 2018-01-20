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

#if defined (_WIN32) || defined (__i386__)
#define BT_USE_SSE_IN_API
#endif

#include "btConvexHullShape.h"
#include "BulletCollision/CollisionShapes/btCollisionMargin.h"

#include "LinearMath/btQuaternion.h"
#include "LinearMath/btSerializer.h"
#include "btConvexPolyhedron.h"
#include "LinearMath/btConvexHullComputer.h"

btConvexHullShape ::btConvexHullShape (const btScalar* points,int numPoints,int stride) : btPolyhedralConvexAabbCachingShape ()
{
	m_shapeType = CONVEX_HULL_SHAPE_PROXYTYPE;
	m_unscaledPoints.resize(numPoints);

	unsigned char* pointsAddress = (unsigned char*)points;

	for (int i=0;i<numPoints;i++)
	{
		btScalar* point = (btScalar*)pointsAddress;
		m_unscaledPoints[i] = btVector3(point[0], point[1], point[2]);
		pointsAddress += stride;
	}

	recalcLocalAabb();

}



void btConvexHullShape::setLocalScaling(const btVector3& scaling)
{
	m_localScaling = scaling;
	recalcLocalAabb();
}

void btConvexHullShape::addPoint(const btVector3& point, bool recalculateLocalAabb)
{
	m_unscaledPoints.push_back(point);
	if (recalculateLocalAabb)
		recalcLocalAabb();

}

btVector3	btConvexHullShape::localGetSupportingVertexWithoutMargin(const btVector3& vec)const
{
	btVector3 supVec(btScalar(0.),btScalar(0.),btScalar(0.));
	btScalar maxDot = btScalar(-BT_LARGE_FLOAT);

    // Here we take advantage of dot(a, b*c) = dot(a*b, c).  Note: This is true mathematically, but not numerically. 
    if( 0 < m_unscaledPoints.size() )
    {
        btVector3 scaled = vec * m_localScaling;
        int index = (int) scaled.maxDot( &m_unscaledPoints[0], m_unscaledPoints.size(), maxDot); // FIXME: may violate encapsulation of m_unscaledPoints
        return m_unscaledPoints[index] * m_localScaling;
    }

    return supVec;
}

void	btConvexHullShape::batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors,btVector3* supportVerticesOut,int numVectors) const
{
	btScalar newDot;
	//use 'w' component of supportVerticesOut?
	{
		for (int i=0;i<numVectors;i++)
		{
			supportVerticesOut[i][3] = btScalar(-BT_LARGE_FLOAT);
		}
	}

    for (int j=0;j<numVectors;j++)
    {
        btVector3 vec = vectors[j] * m_localScaling;        // dot(a*b,c) = dot(a,b*c)
        if( 0 <  m_unscaledPoints.size() )
        {
            int i = (int) vec.maxDot( &m_unscaledPoints[0], m_unscaledPoints.size(), newDot);
            supportVerticesOut[j] = getScaledPoint(i);
            supportVerticesOut[j][3] = newDot;        
        }
        else
            supportVerticesOut[j][3] = -BT_LARGE_FLOAT;
    }



}
	


btVector3	btConvexHullShape::localGetSupportingVertex(const btVector3& vec)const
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


void btConvexHullShape::optimizeConvexHull()
{
	btConvexHullComputer conv;
	conv.compute(&m_unscaledPoints[0].getX(), sizeof(btVector3),m_unscaledPoints.size(),0.f,0.f);
	int numVerts = conv.vertices.size();
	m_unscaledPoints.resize(0);
	for (int i=0;i<numVerts;i++)
    {
        m_unscaledPoints.push_back(conv.vertices[i]);
    }
}



//currently just for debugging (drawing), perhaps future support for algebraic continuous collision detection
//Please note that you can debug-draw btConvexHullShape with the Raytracer Demo
int	btConvexHullShape::getNumVertices() const
{
	return m_unscaledPoints.size();
}

int btConvexHullShape::getNumEdges() const
{
	return m_unscaledPoints.size();
}

void btConvexHullShape::getEdge(int i,btVector3& pa,btVector3& pb) const
{

	int index0 = i%m_unscaledPoints.size();
	int index1 = (i+1)%m_unscaledPoints.size();
	pa = getScaledPoint(index0);
	pb = getScaledPoint(index1);
}

void btConvexHullShape::getVertex(int i,btVector3& vtx) const
{
	vtx = getScaledPoint(i);
}

int	btConvexHullShape::getNumPlanes() const
{
	return 0;
}

void btConvexHullShape::getPlane(btVector3& ,btVector3& ,int ) const
{

	btAssert(0);
}

//not yet
bool btConvexHullShape::isInside(const btVector3& ,btScalar ) const
{
	btAssert(0);
	return false;
}

///fills the dataBuffer and returns the struct name (and 0 on failure)
const char*	btConvexHullShape::serialize(void* dataBuffer, btSerializer* serializer) const
{
	//int szc = sizeof(btConvexHullShapeData);
	btConvexHullShapeData* shapeData = (btConvexHullShapeData*) dataBuffer;
	btConvexInternalShape::serialize(&shapeData->m_convexInternalShapeData, serializer);

	int numElem = m_unscaledPoints.size();
	shapeData->m_numUnscaledPoints = numElem;
#ifdef BT_USE_DOUBLE_PRECISION
	shapeData->m_unscaledPointsFloatPtr = 0;
	shapeData->m_unscaledPointsDoublePtr = numElem ? (btVector3Data*)serializer->getUniquePointer((void*)&m_unscaledPoints[0]):  0;
#else
	shapeData->m_unscaledPointsFloatPtr = numElem ? (btVector3Data*)serializer->getUniquePointer((void*)&m_unscaledPoints[0]):  0;
	shapeData->m_unscaledPointsDoublePtr = 0;
#endif
	
	if (numElem)
	{
		int sz = sizeof(btVector3Data);
	//	int sz2 = sizeof(btVector3DoubleData);
	//	int sz3 = sizeof(btVector3FloatData);
		btChunk* chunk = serializer->allocate(sz,numElem);
		btVector3Data* memPtr = (btVector3Data*)chunk->m_oldPtr;
		for (int i=0;i<numElem;i++,memPtr++)
		{
			m_unscaledPoints[i].serialize(*memPtr);
		}
		serializer->finalizeChunk(chunk,btVector3DataName,BT_ARRAY_CODE,(void*)&m_unscaledPoints[0]);
	}

	// Fill padding with zeros to appease msan.
	memset(shapeData->m_padding3, 0, sizeof(shapeData->m_padding3));

	return "btConvexHullShapeData";
}

void btConvexHullShape::project(const btTransform& trans, const btVector3& dir, btScalar& minProj, btScalar& maxProj, btVector3& witnesPtMin,btVector3& witnesPtMax) const
{
#if 1
	minProj = FLT_MAX;
	maxProj = -FLT_MAX;

	int numVerts = m_unscaledPoints.size();
	for(int i=0;i<numVerts;i++)
	{
		btVector3 vtx = m_unscaledPoints[i] * m_localScaling;
		btVector3 pt = trans * vtx;
		btScalar dp = pt.dot(dir);
		if(dp < minProj)	
		{
			minProj = dp;
			witnesPtMin = pt;
		}
		if(dp > maxProj)	
		{
			maxProj = dp;
			witnesPtMax=pt;
		}
	}
#else
	btVector3 localAxis = dir*trans.getBasis();
	witnesPtMin  = trans(localGetSupportingVertex(localAxis));
	witnesPtMax = trans(localGetSupportingVertex(-localAxis));

	minProj = witnesPtMin.dot(dir);
	maxProj = witnesPtMax.dot(dir);
#endif

	if(minProj>maxProj)
	{
		btSwap(minProj,maxProj);
		btSwap(witnesPtMin,witnesPtMax);
	}


}


