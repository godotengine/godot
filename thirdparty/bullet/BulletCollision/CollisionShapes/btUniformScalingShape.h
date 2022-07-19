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

#ifndef BT_UNIFORM_SCALING_SHAPE_H
#define BT_UNIFORM_SCALING_SHAPE_H

#include "btConvexShape.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"  // for the types

///The btUniformScalingShape allows to re-use uniform scaled instances of btConvexShape in a memory efficient way.
///Istead of using btUniformScalingShape, it is better to use the non-uniform setLocalScaling method on convex shapes that implement it.
ATTRIBUTE_ALIGNED16(class)
btUniformScalingShape : public btConvexShape
{
	btConvexShape* m_childConvexShape;

	btScalar m_uniformScalingFactor;

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	btUniformScalingShape(btConvexShape * convexChildShape, btScalar uniformScalingFactor);

	virtual ~btUniformScalingShape();

	virtual btVector3 localGetSupportingVertexWithoutMargin(const btVector3& vec) const;

	virtual btVector3 localGetSupportingVertex(const btVector3& vec) const;

	virtual void batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors, btVector3* supportVerticesOut, int numVectors) const;

	virtual void calculateLocalInertia(btScalar mass, btVector3 & inertia) const;

	btScalar getUniformScalingFactor() const
	{
		return m_uniformScalingFactor;
	}

	btConvexShape* getChildShape()
	{
		return m_childConvexShape;
	}

	const btConvexShape* getChildShape() const
	{
		return m_childConvexShape;
	}

	virtual const char* getName() const
	{
		return "UniformScalingShape";
	}

	///////////////////////////

	///getAabb's default implementation is brute force, expected derived classes to implement a fast dedicated version
	void getAabb(const btTransform& t, btVector3& aabbMin, btVector3& aabbMax) const;

	virtual void getAabbSlow(const btTransform& t, btVector3& aabbMin, btVector3& aabbMax) const;

	virtual void setLocalScaling(const btVector3& scaling);
	virtual const btVector3& getLocalScaling() const;

	virtual void setMargin(btScalar margin);
	virtual btScalar getMargin() const;

	virtual int getNumPreferredPenetrationDirections() const;

	virtual void getPreferredPenetrationDirection(int index, btVector3& penetrationVector) const;
};

#endif  //BT_UNIFORM_SCALING_SHAPE_H
