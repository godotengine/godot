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

#ifndef BT_TRIANGLE_MESH_SHAPE_H
#define BT_TRIANGLE_MESH_SHAPE_H

#include "btConcaveShape.h"
#include "btStridingMeshInterface.h"

///The btTriangleMeshShape is an internal concave triangle mesh interface. Don't use this class directly, use btBvhTriangleMeshShape instead.
ATTRIBUTE_ALIGNED16(class)
btTriangleMeshShape : public btConcaveShape
{
protected:
	btVector3 m_localAabbMin;
	btVector3 m_localAabbMax;
	btStridingMeshInterface* m_meshInterface;

	///btTriangleMeshShape constructor has been disabled/protected, so that users will not mistakenly use this class.
	///Don't use btTriangleMeshShape but use btBvhTriangleMeshShape instead!
	btTriangleMeshShape(btStridingMeshInterface * meshInterface);

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	virtual ~btTriangleMeshShape();

	virtual btVector3 localGetSupportingVertex(const btVector3& vec) const;

	virtual btVector3 localGetSupportingVertexWithoutMargin(const btVector3& vec) const
	{
		btAssert(0);
		return localGetSupportingVertex(vec);
	}

	void recalcLocalAabb();

	virtual void getAabb(const btTransform& t, btVector3& aabbMin, btVector3& aabbMax) const;

	virtual void processAllTriangles(btTriangleCallback * callback, const btVector3& aabbMin, const btVector3& aabbMax) const;

	virtual void calculateLocalInertia(btScalar mass, btVector3 & inertia) const;

	virtual void setLocalScaling(const btVector3& scaling);
	virtual const btVector3& getLocalScaling() const;

	btStridingMeshInterface* getMeshInterface()
	{
		return m_meshInterface;
	}

	const btStridingMeshInterface* getMeshInterface() const
	{
		return m_meshInterface;
	}

	const btVector3& getLocalAabbMin() const
	{
		return m_localAabbMin;
	}
	const btVector3& getLocalAabbMax() const
	{
		return m_localAabbMax;
	}

	//debugging
	virtual const char* getName() const { return "TRIANGLEMESH"; }
};

#endif  //BT_TRIANGLE_MESH_SHAPE_H
