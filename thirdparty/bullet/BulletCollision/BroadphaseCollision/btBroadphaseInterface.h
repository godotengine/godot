/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef		BT_BROADPHASE_INTERFACE_H
#define 	BT_BROADPHASE_INTERFACE_H



struct btDispatcherInfo;
class btDispatcher;
#include "btBroadphaseProxy.h"

class btOverlappingPairCache;



struct	btBroadphaseAabbCallback
{
	virtual ~btBroadphaseAabbCallback() {}
	virtual bool	process(const btBroadphaseProxy* proxy) = 0;
};


struct	btBroadphaseRayCallback : public btBroadphaseAabbCallback
{
	///added some cached data to accelerate ray-AABB tests
	btVector3		m_rayDirectionInverse;
	unsigned int	m_signs[3];
	btScalar		m_lambda_max;

	virtual ~btBroadphaseRayCallback() {}
	
protected:
    
    btBroadphaseRayCallback() {}
};

#include "LinearMath/btVector3.h"

///The btBroadphaseInterface class provides an interface to detect aabb-overlapping object pairs.
///Some implementations for this broadphase interface include btAxisSweep3, bt32BitAxisSweep3 and btDbvtBroadphase.
///The actual overlapping pair management, storage, adding and removing of pairs is dealt by the btOverlappingPairCache class.
class btBroadphaseInterface
{
public:
	virtual ~btBroadphaseInterface() {}

	virtual btBroadphaseProxy*	createProxy(  const btVector3& aabbMin,  const btVector3& aabbMax,int shapeType,void* userPtr,  int collisionFilterGroup, int collisionFilterMask, btDispatcher* dispatcher) =0;
	virtual void	destroyProxy(btBroadphaseProxy* proxy,btDispatcher* dispatcher)=0;
	virtual void	setAabb(btBroadphaseProxy* proxy,const btVector3& aabbMin,const btVector3& aabbMax, btDispatcher* dispatcher)=0;
	virtual void	getAabb(btBroadphaseProxy* proxy,btVector3& aabbMin, btVector3& aabbMax ) const =0;

	virtual void	rayTest(const btVector3& rayFrom,const btVector3& rayTo, btBroadphaseRayCallback& rayCallback, const btVector3& aabbMin=btVector3(0,0,0), const btVector3& aabbMax = btVector3(0,0,0)) = 0;

	virtual void	aabbTest(const btVector3& aabbMin, const btVector3& aabbMax, btBroadphaseAabbCallback& callback) = 0;

	///calculateOverlappingPairs is optional: incremental algorithms (sweep and prune) might do it during the set aabb
	virtual void	calculateOverlappingPairs(btDispatcher* dispatcher)=0;

	virtual	btOverlappingPairCache*	getOverlappingPairCache()=0;
	virtual	const btOverlappingPairCache*	getOverlappingPairCache() const =0;

	///getAabb returns the axis aligned bounding box in the 'global' coordinate frame
	///will add some transform later
	virtual void getBroadphaseAabb(btVector3& aabbMin,btVector3& aabbMax) const =0;

	///reset broadphase internal structures, to ensure determinism/reproducability
	virtual void resetPool(btDispatcher* dispatcher) { (void) dispatcher; };

	virtual void	printStats() = 0;

};

#endif //BT_BROADPHASE_INTERFACE_H
