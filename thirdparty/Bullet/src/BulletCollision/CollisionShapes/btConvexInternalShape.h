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

#ifndef BT_CONVEX_INTERNAL_SHAPE_H
#define BT_CONVEX_INTERNAL_SHAPE_H

#include "btConvexShape.h"
#include "LinearMath/btAabbUtil2.h"


///The btConvexInternalShape is an internal base class, shared by most convex shape implementations.
///The btConvexInternalShape uses a default collision margin set to CONVEX_DISTANCE_MARGIN.
///This collision margin used by Gjk and some other algorithms, see also btCollisionMargin.h
///Note that when creating small shapes (derived from btConvexInternalShape), 
///you need to make sure to set a smaller collision margin, using the 'setMargin' API
///There is a automatic mechanism 'setSafeMargin' used by btBoxShape and btCylinderShape
ATTRIBUTE_ALIGNED16(class) btConvexInternalShape : public btConvexShape
{

	protected:

	//local scaling. collisionMargin is not scaled !
	btVector3	m_localScaling;

	btVector3	m_implicitShapeDimensions;
	
	btScalar	m_collisionMargin;

	btScalar	m_padding;

	btConvexInternalShape();

public:

	BT_DECLARE_ALIGNED_ALLOCATOR();

	virtual ~btConvexInternalShape()
	{

	}

	virtual btVector3	localGetSupportingVertex(const btVector3& vec)const;

	const btVector3& getImplicitShapeDimensions() const
	{
		return m_implicitShapeDimensions;
	}

	///warning: use setImplicitShapeDimensions with care
	///changing a collision shape while the body is in the world is not recommended,
	///it is best to remove the body from the world, then make the change, and re-add it
	///alternatively flush the contact points, see documentation for 'cleanProxyFromPairs'
	void	setImplicitShapeDimensions(const btVector3& dimensions)
	{
		m_implicitShapeDimensions = dimensions;
	}

	void	setSafeMargin(btScalar minDimension, btScalar defaultMarginMultiplier = 0.1f)
	{
		btScalar safeMargin = defaultMarginMultiplier*minDimension;
		if (safeMargin < getMargin())
		{
			setMargin(safeMargin);
		}
	}
	void	setSafeMargin(const btVector3& halfExtents, btScalar defaultMarginMultiplier = 0.1f)
	{
		//see http://code.google.com/p/bullet/issues/detail?id=349
		//this margin check could could be added to other collision shapes too,
		//or add some assert/warning somewhere
		btScalar minDimension=halfExtents[halfExtents.minAxis()]; 		
		setSafeMargin(minDimension, defaultMarginMultiplier);
	}

	///getAabb's default implementation is brute force, expected derived classes to implement a fast dedicated version
	void getAabb(const btTransform& t,btVector3& aabbMin,btVector3& aabbMax) const
	{
		getAabbSlow(t,aabbMin,aabbMax);
	}


	
	virtual void getAabbSlow(const btTransform& t,btVector3& aabbMin,btVector3& aabbMax) const;


	virtual void	setLocalScaling(const btVector3& scaling);
	virtual const btVector3& getLocalScaling() const 
	{
		return m_localScaling;
	}

	const btVector3& getLocalScalingNV() const 
	{
		return m_localScaling;
	}

	virtual void	setMargin(btScalar margin)
	{
		m_collisionMargin = margin;
	}
	virtual btScalar	getMargin() const
	{
		return m_collisionMargin;
	}

	btScalar	getMarginNV() const
	{
		return m_collisionMargin;
	}

	virtual int		getNumPreferredPenetrationDirections() const
	{
		return 0;
	}
	
	virtual void	getPreferredPenetrationDirection(int index, btVector3& penetrationVector) const
	{
		(void)penetrationVector;
		(void)index;
		btAssert(0);
	}

	virtual	int	calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual	const char*	serialize(void* dataBuffer, btSerializer* serializer) const;

	
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct	btConvexInternalShapeData
{
	btCollisionShapeData	m_collisionShapeData;

	btVector3FloatData	m_localScaling;

	btVector3FloatData	m_implicitShapeDimensions;
	
	float			m_collisionMargin;

	int	m_padding;

};



SIMD_FORCE_INLINE	int	btConvexInternalShape::calculateSerializeBufferSize() const
{
	return sizeof(btConvexInternalShapeData);
}

///fills the dataBuffer and returns the struct name (and 0 on failure)
SIMD_FORCE_INLINE	const char*	btConvexInternalShape::serialize(void* dataBuffer, btSerializer* serializer) const
{
	btConvexInternalShapeData* shapeData = (btConvexInternalShapeData*) dataBuffer;
	btCollisionShape::serialize(&shapeData->m_collisionShapeData, serializer);

	m_implicitShapeDimensions.serializeFloat(shapeData->m_implicitShapeDimensions);
	m_localScaling.serializeFloat(shapeData->m_localScaling);
	shapeData->m_collisionMargin = float(m_collisionMargin);

	// Fill padding with zeros to appease msan.
	shapeData->m_padding = 0;

	return "btConvexInternalShapeData";
}




///btConvexInternalAabbCachingShape adds local aabb caching for convex shapes, to avoid expensive bounding box calculations
class btConvexInternalAabbCachingShape : public btConvexInternalShape
{
	btVector3	m_localAabbMin;
	btVector3	m_localAabbMax;
	bool		m_isLocalAabbValid;
	
protected:
					
	btConvexInternalAabbCachingShape();
	
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

	inline void getNonvirtualAabb(const btTransform& trans,btVector3& aabbMin,btVector3& aabbMax, btScalar margin) const
	{

		//lazy evaluation of local aabb
		btAssert(m_isLocalAabbValid);
		btTransformAabb(m_localAabbMin,m_localAabbMax,margin,trans,aabbMin,aabbMax);
	}
		
public:
		
	virtual void	setLocalScaling(const btVector3& scaling);

	virtual void getAabb(const btTransform& t,btVector3& aabbMin,btVector3& aabbMax) const;

	void	recalcLocalAabb();

};

#endif //BT_CONVEX_INTERNAL_SHAPE_H
