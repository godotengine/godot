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

#ifndef BT_MULTI_SPHERE_MINKOWSKI_H
#define BT_MULTI_SPHERE_MINKOWSKI_H

#include "btConvexInternalShape.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h" // for the types
#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btAabbUtil2.h"



///The btMultiSphereShape represents the convex hull of a collection of spheres. You can create special capsules or other smooth volumes.
///It is possible to animate the spheres for deformation, but call 'recalcLocalAabb' after changing any sphere position/radius
ATTRIBUTE_ALIGNED16(class) btMultiSphereShape : public btConvexInternalAabbCachingShape
{
	
	btAlignedObjectArray<btVector3> m_localPositionArray;
	btAlignedObjectArray<btScalar>  m_radiArray;
	
public:
	BT_DECLARE_ALIGNED_ALLOCATOR();
	
	btMultiSphereShape (const btVector3* positions,const btScalar* radi,int numSpheres);

	///CollisionShape Interface
	virtual void	calculateLocalInertia(btScalar mass,btVector3& inertia) const;

	/// btConvexShape Interface
	virtual btVector3	localGetSupportingVertexWithoutMargin(const btVector3& vec)const;

	virtual void	batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors,btVector3* supportVerticesOut,int numVectors) const;
	
	int	getSphereCount() const
	{
		return m_localPositionArray.size();
	}

	const btVector3&	getSpherePosition(int index) const
	{
		return m_localPositionArray[index];
	}

	btScalar	getSphereRadius(int index) const
	{
		return m_radiArray[index];
	}


	virtual const char*	getName()const 
	{
		return "MultiSphere";
	}

	virtual	int	calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual	const char*	serialize(void* dataBuffer, btSerializer* serializer) const;


};


struct	btPositionAndRadius
{
	btVector3FloatData	m_pos;
	float		m_radius;
};

struct	btMultiSphereShapeData
{
	btConvexInternalShapeData	m_convexInternalShapeData;

	btPositionAndRadius	*m_localPositionArrayPtr;
	int				m_localPositionArraySize;
	char	m_padding[4];
};



SIMD_FORCE_INLINE	int	btMultiSphereShape::calculateSerializeBufferSize() const
{
	return sizeof(btMultiSphereShapeData);
}



#endif //BT_MULTI_SPHERE_MINKOWSKI_H
