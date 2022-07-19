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

#ifndef BT_COLLISION_SHAPE_H
#define BT_COLLISION_SHAPE_H

#include "LinearMath/btTransform.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btMatrix3x3.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"  //for the shape types
class btSerializer;

///The btCollisionShape class provides an interface for collision shapes that can be shared among btCollisionObjects.
ATTRIBUTE_ALIGNED16(class)
btCollisionShape
{
protected:
	int m_shapeType;
	void* m_userPointer;
	int m_userIndex;

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	btCollisionShape() : m_shapeType(INVALID_SHAPE_PROXYTYPE), m_userPointer(0), m_userIndex(-1)
	{
	}

	virtual ~btCollisionShape()
	{
	}

	///getAabb returns the axis aligned bounding box in the coordinate frame of the given transform t.
	virtual void getAabb(const btTransform& t, btVector3& aabbMin, btVector3& aabbMax) const = 0;

	virtual void getBoundingSphere(btVector3 & center, btScalar & radius) const;

	///getAngularMotionDisc returns the maximum radius needed for Conservative Advancement to handle time-of-impact with rotations.
	virtual btScalar getAngularMotionDisc() const;

	virtual btScalar getContactBreakingThreshold(btScalar defaultContactThresholdFactor) const;

	///calculateTemporalAabb calculates the enclosing aabb for the moving object over interval [0..timeStep)
	///result is conservative
	void calculateTemporalAabb(const btTransform& curTrans, const btVector3& linvel, const btVector3& angvel, btScalar timeStep, btVector3& temporalAabbMin, btVector3& temporalAabbMax) const;

	SIMD_FORCE_INLINE bool isPolyhedral() const
	{
		return btBroadphaseProxy::isPolyhedral(getShapeType());
	}

	SIMD_FORCE_INLINE bool isConvex2d() const
	{
		return btBroadphaseProxy::isConvex2d(getShapeType());
	}

	SIMD_FORCE_INLINE bool isConvex() const
	{
		return btBroadphaseProxy::isConvex(getShapeType());
	}
	SIMD_FORCE_INLINE bool isNonMoving() const
	{
		return btBroadphaseProxy::isNonMoving(getShapeType());
	}
	SIMD_FORCE_INLINE bool isConcave() const
	{
		return btBroadphaseProxy::isConcave(getShapeType());
	}
	SIMD_FORCE_INLINE bool isCompound() const
	{
		return btBroadphaseProxy::isCompound(getShapeType());
	}

	SIMD_FORCE_INLINE bool isSoftBody() const
	{
		return btBroadphaseProxy::isSoftBody(getShapeType());
	}

	///isInfinite is used to catch simulation error (aabb check)
	SIMD_FORCE_INLINE bool isInfinite() const
	{
		return btBroadphaseProxy::isInfinite(getShapeType());
	}

#ifndef __SPU__
	virtual void setLocalScaling(const btVector3& scaling) = 0;
	virtual const btVector3& getLocalScaling() const = 0;
	virtual void calculateLocalInertia(btScalar mass, btVector3 & inertia) const = 0;

	//debugging support
	virtual const char* getName() const = 0;
#endif  //__SPU__

	int getShapeType() const
	{
		return m_shapeType;
	}

	///the getAnisotropicRollingFrictionDirection can be used in combination with setAnisotropicFriction
	///See Bullet/Demos/RollingFrictionDemo for an example
	virtual btVector3 getAnisotropicRollingFrictionDirection() const
	{
		return btVector3(1, 1, 1);
	}
	virtual void setMargin(btScalar margin) = 0;
	virtual btScalar getMargin() const = 0;

	///optional user data pointer
	void setUserPointer(void* userPtr)
	{
		m_userPointer = userPtr;
	}

	void* getUserPointer() const
	{
		return m_userPointer;
	}
	void setUserIndex(int index)
	{
		m_userIndex = index;
	}

	int getUserIndex() const
	{
		return m_userIndex;
	}

	virtual int calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual const char* serialize(void* dataBuffer, btSerializer* serializer) const;

	virtual void serializeSingleShape(btSerializer * serializer) const;
};

// clang-format off
// parser needs * with the name
///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct	btCollisionShapeData
{
	char	*m_name;
	int		m_shapeType;
	char	m_padding[4];
};
// clang-format on
SIMD_FORCE_INLINE int btCollisionShape::calculateSerializeBufferSize() const
{
	return sizeof(btCollisionShapeData);
}

#endif  //BT_COLLISION_SHAPE_H
