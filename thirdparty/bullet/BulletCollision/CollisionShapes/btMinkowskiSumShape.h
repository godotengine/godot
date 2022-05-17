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

#ifndef BT_MINKOWSKI_SUM_SHAPE_H
#define BT_MINKOWSKI_SUM_SHAPE_H

#include "btConvexInternalShape.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"  // for the types

/// The btMinkowskiSumShape is only for advanced users. This shape represents implicit based minkowski sum of two convex implicit shapes.
ATTRIBUTE_ALIGNED16(class)
btMinkowskiSumShape : public btConvexInternalShape
{
	btTransform m_transA;
	btTransform m_transB;
	const btConvexShape* m_shapeA;
	const btConvexShape* m_shapeB;

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	btMinkowskiSumShape(const btConvexShape* shapeA, const btConvexShape* shapeB);

	virtual btVector3 localGetSupportingVertexWithoutMargin(const btVector3& vec) const;

	virtual void batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors, btVector3* supportVerticesOut, int numVectors) const;

	virtual void calculateLocalInertia(btScalar mass, btVector3 & inertia) const;

	void setTransformA(const btTransform& transA) { m_transA = transA; }
	void setTransformB(const btTransform& transB) { m_transB = transB; }

	const btTransform& getTransformA() const { return m_transA; }
	const btTransform& getTransformB() const { return m_transB; }

	// keep this for backward compatibility
	const btTransform& GetTransformB() const { return m_transB; }

	virtual btScalar getMargin() const;

	const btConvexShape* getShapeA() const { return m_shapeA; }
	const btConvexShape* getShapeB() const { return m_shapeB; }

	virtual const char* getName() const
	{
		return "MinkowskiSum";
	}
};

#endif  //BT_MINKOWSKI_SUM_SHAPE_H
