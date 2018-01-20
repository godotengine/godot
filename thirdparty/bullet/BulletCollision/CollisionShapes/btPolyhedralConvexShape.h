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

#ifndef BT_POLYHEDRAL_CONVEX_SHAPE_H
#define BT_POLYHEDRAL_CONVEX_SHAPE_H

#include "LinearMath/btMatrix3x3.h"
#include "btConvexInternalShape.h"
class btConvexPolyhedron;


///The btPolyhedralConvexShape is an internal interface class for polyhedral convex shapes.
ATTRIBUTE_ALIGNED16(class) btPolyhedralConvexShape : public btConvexInternalShape
{
	

protected:
	
	btConvexPolyhedron* m_polyhedron;

public:
	
	BT_DECLARE_ALIGNED_ALLOCATOR();
	

	btPolyhedralConvexShape();

	virtual ~btPolyhedralConvexShape();

	///optional method mainly used to generate multiple contact points by clipping polyhedral features (faces/edges)
	///experimental/work-in-progress
	virtual bool	initializePolyhedralFeatures(int shiftVerticesByMargin=0);

	const btConvexPolyhedron*	getConvexPolyhedron() const
	{
		return m_polyhedron;
	}

	//brute force implementations

	virtual btVector3	localGetSupportingVertexWithoutMargin(const btVector3& vec)const;
	virtual void	batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors,btVector3* supportVerticesOut,int numVectors) const;
	
	virtual void	calculateLocalInertia(btScalar mass,btVector3& inertia) const;
	
	
	virtual int	getNumVertices() const = 0 ;
	virtual int getNumEdges() const = 0;
	virtual void getEdge(int i,btVector3& pa,btVector3& pb) const = 0;
	virtual void getVertex(int i,btVector3& vtx) const = 0;
	virtual int	getNumPlanes() const = 0;
	virtual void getPlane(btVector3& planeNormal,btVector3& planeSupport,int i ) const = 0;
//	virtual int getIndex(int i) const = 0 ; 

	virtual	bool isInside(const btVector3& pt,btScalar tolerance) const = 0;
	
};


///The btPolyhedralConvexAabbCachingShape adds aabb caching to the btPolyhedralConvexShape
class btPolyhedralConvexAabbCachingShape : public btPolyhedralConvexShape
{

	btVector3	m_localAabbMin;
	btVector3	m_localAabbMax;
	bool		m_isLocalAabbValid;
		
protected:

	void setCachedLocalAabb (const btVector3& aabbMin, const btVector3& aabbMax)
	{
		m_isLocalAabbValid = true;
		m_localAabbMin = aabbMin;
		m_localAabbMax = aabbMax;
	}

	inline void getCachedLocalAabb (btVector3& aabbMin, btVector3& aabbMax) const
	{
		btAssert(m_isLocalAabbValid);
		aabbMin = m_localAabbMin;
		aabbMax = m_localAabbMax;
	}

protected:

	btPolyhedralConvexAabbCachingShape();
	
public:
	
	inline void getNonvirtualAabb(const btTransform& trans,btVector3& aabbMin,btVector3& aabbMax, btScalar margin) const
	{

		//lazy evaluation of local aabb
		btAssert(m_isLocalAabbValid);
		btTransformAabb(m_localAabbMin,m_localAabbMax,margin,trans,aabbMin,aabbMax);
	}

	virtual void	setLocalScaling(const btVector3& scaling);

	virtual void getAabb(const btTransform& t,btVector3& aabbMin,btVector3& aabbMax) const;

	void	recalcLocalAabb();

};

#endif //BT_POLYHEDRAL_CONVEX_SHAPE_H
