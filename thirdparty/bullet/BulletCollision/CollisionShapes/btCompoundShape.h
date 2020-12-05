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

#ifndef BT_COMPOUND_SHAPE_H
#define BT_COMPOUND_SHAPE_H

#include "btCollisionShape.h"

#include "LinearMath/btVector3.h"
#include "LinearMath/btTransform.h"
#include "LinearMath/btMatrix3x3.h"
#include "btCollisionMargin.h"
#include "LinearMath/btAlignedObjectArray.h"

//class btOptimizedBvh;
struct btDbvt;

ATTRIBUTE_ALIGNED16(struct)
btCompoundShapeChild
{
	BT_DECLARE_ALIGNED_ALLOCATOR();

	btTransform m_transform;
	btCollisionShape* m_childShape;
	int m_childShapeType;
	btScalar m_childMargin;
	struct btDbvtNode* m_node;
};

SIMD_FORCE_INLINE bool operator==(const btCompoundShapeChild& c1, const btCompoundShapeChild& c2)
{
	return (c1.m_transform == c2.m_transform &&
			c1.m_childShape == c2.m_childShape &&
			c1.m_childShapeType == c2.m_childShapeType &&
			c1.m_childMargin == c2.m_childMargin);
}

/// The btCompoundShape allows to store multiple other btCollisionShapes
/// This allows for moving concave collision objects. This is more general then the static concave btBvhTriangleMeshShape.
/// It has an (optional) dynamic aabb tree to accelerate early rejection tests.
/// @todo: This aabb tree can also be use to speed up ray tests on btCompoundShape, see http://code.google.com/p/bullet/issues/detail?id=25
/// Currently, removal of child shapes is only supported when disabling the aabb tree (pass 'false' in the constructor of btCompoundShape)
ATTRIBUTE_ALIGNED16(class)
btCompoundShape : public btCollisionShape
{
protected:
	btAlignedObjectArray<btCompoundShapeChild> m_children;
	btVector3 m_localAabbMin;
	btVector3 m_localAabbMax;

	btDbvt* m_dynamicAabbTree;

	///increment m_updateRevision when adding/removing/replacing child shapes, so that some caches can be updated
	int m_updateRevision;

	btScalar m_collisionMargin;

	btVector3 m_localScaling;

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	explicit btCompoundShape(bool enableDynamicAabbTree = true, const int initialChildCapacity = 0);

	virtual ~btCompoundShape();

	void addChildShape(const btTransform& localTransform, btCollisionShape* shape);

	/// Remove all children shapes that contain the specified shape
	virtual void removeChildShape(btCollisionShape * shape);

	void removeChildShapeByIndex(int childShapeindex);

	int getNumChildShapes() const
	{
		return int(m_children.size());
	}

	btCollisionShape* getChildShape(int index)
	{
		return m_children[index].m_childShape;
	}
	const btCollisionShape* getChildShape(int index) const
	{
		return m_children[index].m_childShape;
	}

	btTransform& getChildTransform(int index)
	{
		return m_children[index].m_transform;
	}
	const btTransform& getChildTransform(int index) const
	{
		return m_children[index].m_transform;
	}

	///set a new transform for a child, and update internal data structures (local aabb and dynamic tree)
	void updateChildTransform(int childIndex, const btTransform& newChildTransform, bool shouldRecalculateLocalAabb = true);

	btCompoundShapeChild* getChildList()
	{
		return &m_children[0];
	}

	///getAabb's default implementation is brute force, expected derived classes to implement a fast dedicated version
	virtual void getAabb(const btTransform& t, btVector3& aabbMin, btVector3& aabbMax) const;

	/** Re-calculate the local Aabb. Is called at the end of removeChildShapes. 
	Use this yourself if you modify the children or their transforms. */
	virtual void recalculateLocalAabb();

	virtual void setLocalScaling(const btVector3& scaling);

	virtual const btVector3& getLocalScaling() const
	{
		return m_localScaling;
	}

	virtual void calculateLocalInertia(btScalar mass, btVector3 & inertia) const;

	virtual void setMargin(btScalar margin)
	{
		m_collisionMargin = margin;
	}
	virtual btScalar getMargin() const
	{
		return m_collisionMargin;
	}
	virtual const char* getName() const
	{
		return "Compound";
	}

	const btDbvt* getDynamicAabbTree() const
	{
		return m_dynamicAabbTree;
	}

	btDbvt* getDynamicAabbTree()
	{
		return m_dynamicAabbTree;
	}

	void createAabbTreeFromChildren();

	///computes the exact moment of inertia and the transform from the coordinate system defined by the principal axes of the moment of inertia
	///and the center of mass to the current coordinate system. "masses" points to an array of masses of the children. The resulting transform
	///"principal" has to be applied inversely to all children transforms in order for the local coordinate system of the compound
	///shape to be centered at the center of mass and to coincide with the principal axes. This also necessitates a correction of the world transform
	///of the collision object by the principal transform.
	void calculatePrincipalAxisTransform(const btScalar* masses, btTransform& principal, btVector3& inertia) const;

	int getUpdateRevision() const
	{
		return m_updateRevision;
	}

	virtual int calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual const char* serialize(void* dataBuffer, btSerializer* serializer) const;
};

// clang-format off

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btCompoundShapeChildData
{
	btTransformFloatData	m_transform;
	btCollisionShapeData	*m_childShape;
	int						m_childShapeType;
	float					m_childMargin;
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct	btCompoundShapeData
{
	btCollisionShapeData		m_collisionShapeData;

	btCompoundShapeChildData	*m_childShapePtr;

	int							m_numChildShapes;

	float	m_collisionMargin;

};

// clang-format on

SIMD_FORCE_INLINE int btCompoundShape::calculateSerializeBufferSize() const
{
	return sizeof(btCompoundShapeData);
}

#endif  //BT_COMPOUND_SHAPE_H
