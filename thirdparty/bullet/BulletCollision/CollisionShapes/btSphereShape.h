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
#ifndef BT_SPHERE_MINKOWSKI_H
#define BT_SPHERE_MINKOWSKI_H

#include "btConvexInternalShape.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"  // for the types

///The btSphereShape implements an implicit sphere, centered around a local origin with radius.
ATTRIBUTE_ALIGNED16(class)
btSphereShape : public btConvexInternalShape

{
public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	btSphereShape(btScalar radius) : btConvexInternalShape()
	{
		m_shapeType = SPHERE_SHAPE_PROXYTYPE;
		m_localScaling.setValue(1.0, 1.0, 1.0);
		m_implicitShapeDimensions.setZero();
		m_implicitShapeDimensions.setX(radius);
		m_collisionMargin = radius;
		m_padding = 0;
	}

	virtual btVector3 localGetSupportingVertex(const btVector3& vec) const;
	virtual btVector3 localGetSupportingVertexWithoutMargin(const btVector3& vec) const;
	//notice that the vectors should be unit length
	virtual void batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors, btVector3* supportVerticesOut, int numVectors) const;

	virtual void calculateLocalInertia(btScalar mass, btVector3 & inertia) const;

	virtual void getAabb(const btTransform& t, btVector3& aabbMin, btVector3& aabbMax) const;

	btScalar getRadius() const { return m_implicitShapeDimensions.getX() * m_localScaling.getX(); }

	void setUnscaledRadius(btScalar radius)
	{
		m_implicitShapeDimensions.setX(radius);
		btConvexInternalShape::setMargin(radius);
	}

	//debugging
	virtual const char* getName() const { return "SPHERE"; }

	virtual void setMargin(btScalar margin)
	{
		btConvexInternalShape::setMargin(margin);
	}
	virtual btScalar getMargin() const
	{
		//to improve gjk behaviour, use radius+margin as the full margin, so never get into the penetration case
		//this means, non-uniform scaling is not supported anymore
		return getRadius();
	}
};

#endif  //BT_SPHERE_MINKOWSKI_H
