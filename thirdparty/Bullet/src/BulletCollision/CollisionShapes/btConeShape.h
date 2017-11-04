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

#ifndef BT_CONE_MINKOWSKI_H
#define BT_CONE_MINKOWSKI_H

#include "btConvexInternalShape.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h" // for the types

///The btConeShape implements a cone shape primitive, centered around the origin and aligned with the Y axis. The btConeShapeX is aligned around the X axis and btConeShapeZ around the Z axis.
ATTRIBUTE_ALIGNED16(class) btConeShape : public btConvexInternalShape

{

	btScalar m_sinAngle;
	btScalar m_radius;
	btScalar m_height;
	int		m_coneIndices[3];
	btVector3 coneLocalSupport(const btVector3& v) const;


public:
	BT_DECLARE_ALIGNED_ALLOCATOR();
	
	btConeShape (btScalar radius,btScalar height);
	
	virtual btVector3	localGetSupportingVertex(const btVector3& vec) const;
	virtual btVector3	localGetSupportingVertexWithoutMargin(const btVector3& vec) const;
	virtual void	batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors,btVector3* supportVerticesOut,int numVectors) const;

	btScalar getRadius() const { return m_radius;}
	btScalar getHeight() const { return m_height;}

	void setRadius(const btScalar radius)
	{
		m_radius = radius;
	}
	void setHeight(const btScalar height)
	{
		m_height = height;
	}


	virtual void	calculateLocalInertia(btScalar mass,btVector3& inertia) const
	{
		btTransform identity;
		identity.setIdentity();
		btVector3 aabbMin,aabbMax;
		getAabb(identity,aabbMin,aabbMax);

		btVector3 halfExtents = (aabbMax-aabbMin)*btScalar(0.5);

		btScalar margin = getMargin();

		btScalar lx=btScalar(2.)*(halfExtents.x()+margin);
		btScalar ly=btScalar(2.)*(halfExtents.y()+margin);
		btScalar lz=btScalar(2.)*(halfExtents.z()+margin);
		const btScalar x2 = lx*lx;
		const btScalar y2 = ly*ly;
		const btScalar z2 = lz*lz;
		const btScalar scaledmass = mass * btScalar(0.08333333);

		inertia = scaledmass * (btVector3(y2+z2,x2+z2,x2+y2));

//		inertia.x() = scaledmass * (y2+z2);
//		inertia.y() = scaledmass * (x2+z2);
//		inertia.z() = scaledmass * (x2+y2);
	}


		virtual const char*	getName()const 
		{
			return "Cone";
		}
		
		///choose upAxis index
		void	setConeUpIndex(int upIndex);
		
		int	getConeUpIndex() const
		{
			return m_coneIndices[1];
		}

	virtual btVector3	getAnisotropicRollingFrictionDirection() const
	{
		return btVector3 (0,1,0);
	}

	virtual void	setLocalScaling(const btVector3& scaling);
	
	
	virtual	int	calculateSerializeBufferSize() const;
	
	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual	const char*	serialize(void* dataBuffer, btSerializer* serializer) const;
	

};

///btConeShape implements a Cone shape, around the X axis
class btConeShapeX : public btConeShape
{
	public:
		btConeShapeX(btScalar radius,btScalar height);

	virtual btVector3	getAnisotropicRollingFrictionDirection() const
	{
		return btVector3 (1,0,0);
	}

	//debugging
	virtual const char*	getName()const
	{
		return "ConeX";
	}
	
	
};

///btConeShapeZ implements a Cone shape, around the Z axis
class btConeShapeZ : public btConeShape
{
public:
	btConeShapeZ(btScalar radius,btScalar height);

	virtual btVector3	getAnisotropicRollingFrictionDirection() const
	{
		return btVector3 (0,0,1);
	}

	//debugging
	virtual const char*	getName()const
	{
		return "ConeZ";
	}
	
	
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct	btConeShapeData
{
	btConvexInternalShapeData	m_convexInternalShapeData;
	
	int	m_upIndex;
	
	char	m_padding[4];
};

SIMD_FORCE_INLINE	int	btConeShape::calculateSerializeBufferSize() const
{
	return sizeof(btConeShapeData);
}

///fills the dataBuffer and returns the struct name (and 0 on failure)
SIMD_FORCE_INLINE	const char*	btConeShape::serialize(void* dataBuffer, btSerializer* serializer) const
{
	btConeShapeData* shapeData = (btConeShapeData*) dataBuffer;

	btConvexInternalShape::serialize(&shapeData->m_convexInternalShapeData,serializer);

	shapeData->m_upIndex = m_coneIndices[1];

	// Fill padding with zeros to appease msan.
	shapeData->m_padding[0] = 0;
	shapeData->m_padding[1] = 0;
	shapeData->m_padding[2] = 0;
	shapeData->m_padding[3] = 0;

	return "btConeShapeData";
}

#endif //BT_CONE_MINKOWSKI_H

