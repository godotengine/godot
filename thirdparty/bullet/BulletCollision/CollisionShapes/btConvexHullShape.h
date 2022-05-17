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

#ifndef BT_CONVEX_HULL_SHAPE_H
#define BT_CONVEX_HULL_SHAPE_H

#include "btPolyhedralConvexShape.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"  // for the types
#include "LinearMath/btAlignedObjectArray.h"

///The btConvexHullShape implements an implicit convex hull of an array of vertices.
///Bullet provides a general and fast collision detector for convex shapes based on GJK and EPA using localGetSupportingVertex.
ATTRIBUTE_ALIGNED16(class)
btConvexHullShape : public btPolyhedralConvexAabbCachingShape
{
protected:
	btAlignedObjectArray<btVector3> m_unscaledPoints;

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	///this constructor optionally takes in a pointer to points. Each point is assumed to be 3 consecutive btScalar (x,y,z), the striding defines the number of bytes between each point, in memory.
	///It is easier to not pass any points in the constructor, and just add one point at a time, using addPoint.
	///btConvexHullShape make an internal copy of the points.
	btConvexHullShape(const btScalar* points = 0, int numPoints = 0, int stride = sizeof(btVector3));

	void addPoint(const btVector3& point, bool recalculateLocalAabb = true);

	btVector3* getUnscaledPoints()
	{
		return &m_unscaledPoints[0];
	}

	const btVector3* getUnscaledPoints() const
	{
		return &m_unscaledPoints[0];
	}

	///getPoints is obsolete, please use getUnscaledPoints
	const btVector3* getPoints() const
	{
		return getUnscaledPoints();
	}

	void optimizeConvexHull();

	SIMD_FORCE_INLINE btVector3 getScaledPoint(int i) const
	{
		return m_unscaledPoints[i] * m_localScaling;
	}

	SIMD_FORCE_INLINE int getNumPoints() const
	{
		return m_unscaledPoints.size();
	}

	virtual btVector3 localGetSupportingVertex(const btVector3& vec) const;
	virtual btVector3 localGetSupportingVertexWithoutMargin(const btVector3& vec) const;
	virtual void batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors, btVector3* supportVerticesOut, int numVectors) const;

	virtual void project(const btTransform& trans, const btVector3& dir, btScalar& minProj, btScalar& maxProj, btVector3& witnesPtMin, btVector3& witnesPtMax) const;

	//debugging
	virtual const char* getName() const { return "Convex"; }

	virtual int getNumVertices() const;
	virtual int getNumEdges() const;
	virtual void getEdge(int i, btVector3& pa, btVector3& pb) const;
	virtual void getVertex(int i, btVector3& vtx) const;
	virtual int getNumPlanes() const;
	virtual void getPlane(btVector3 & planeNormal, btVector3 & planeSupport, int i) const;
	virtual bool isInside(const btVector3& pt, btScalar tolerance) const;

	///in case we receive negative scaling
	virtual void setLocalScaling(const btVector3& scaling);

	virtual int calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual const char* serialize(void* dataBuffer, btSerializer* serializer) const;
};

// clang-format off

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct	btConvexHullShapeData
{
	btConvexInternalShapeData	m_convexInternalShapeData;

	btVector3FloatData	*m_unscaledPointsFloatPtr;
	btVector3DoubleData	*m_unscaledPointsDoublePtr;

	int		m_numUnscaledPoints;
	char m_padding3[4];

};

// clang-format on

SIMD_FORCE_INLINE int btConvexHullShape::calculateSerializeBufferSize() const
{
	return sizeof(btConvexHullShapeData);
}

#endif  //BT_CONVEX_HULL_SHAPE_H
