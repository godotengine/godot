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

#ifndef BT_SCALED_BVH_TRIANGLE_MESH_SHAPE_H
#define BT_SCALED_BVH_TRIANGLE_MESH_SHAPE_H

#include "BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h"

///The btScaledBvhTriangleMeshShape allows to instance a scaled version of an existing btBvhTriangleMeshShape.
///Note that each btBvhTriangleMeshShape still can have its own local scaling, independent from this btScaledBvhTriangleMeshShape 'localScaling'
ATTRIBUTE_ALIGNED16(class)
btScaledBvhTriangleMeshShape : public btConcaveShape
{
	btVector3 m_localScaling;

	btBvhTriangleMeshShape* m_bvhTriMeshShape;

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	btScaledBvhTriangleMeshShape(btBvhTriangleMeshShape * childShape, const btVector3& localScaling);

	virtual ~btScaledBvhTriangleMeshShape();

	virtual void getAabb(const btTransform& t, btVector3& aabbMin, btVector3& aabbMax) const;
	virtual void setLocalScaling(const btVector3& scaling);
	virtual const btVector3& getLocalScaling() const;
	virtual void calculateLocalInertia(btScalar mass, btVector3 & inertia) const;

	virtual void processAllTriangles(btTriangleCallback * callback, const btVector3& aabbMin, const btVector3& aabbMax) const;

	btBvhTriangleMeshShape* getChildShape()
	{
		return m_bvhTriMeshShape;
	}

	const btBvhTriangleMeshShape* getChildShape() const
	{
		return m_bvhTriMeshShape;
	}

	//debugging
	virtual const char* getName() const { return "SCALEDBVHTRIANGLEMESH"; }

	virtual int calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual const char* serialize(void* dataBuffer, btSerializer* serializer) const;
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btScaledTriangleMeshShapeData
{
	btTriangleMeshShapeData m_trimeshShapeData;

	btVector3FloatData m_localScaling;
};

SIMD_FORCE_INLINE int btScaledBvhTriangleMeshShape::calculateSerializeBufferSize() const
{
	return sizeof(btScaledTriangleMeshShapeData);
}

///fills the dataBuffer and returns the struct name (and 0 on failure)
SIMD_FORCE_INLINE const char* btScaledBvhTriangleMeshShape::serialize(void* dataBuffer, btSerializer* serializer) const
{
	btScaledTriangleMeshShapeData* scaledMeshData = (btScaledTriangleMeshShapeData*)dataBuffer;
	m_bvhTriMeshShape->serialize(&scaledMeshData->m_trimeshShapeData, serializer);
	scaledMeshData->m_trimeshShapeData.m_collisionShapeData.m_shapeType = SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE;
	m_localScaling.serializeFloat(scaledMeshData->m_localScaling);
	return "btScaledTriangleMeshShapeData";
}

#endif  //BT_SCALED_BVH_TRIANGLE_MESH_SHAPE_H
