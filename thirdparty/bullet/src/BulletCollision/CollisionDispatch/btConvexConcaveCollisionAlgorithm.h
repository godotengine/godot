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

#ifndef BT_CONVEX_CONCAVE_COLLISION_ALGORITHM_H
#define BT_CONVEX_CONCAVE_COLLISION_ALGORITHM_H

#include "btActivatingCollisionAlgorithm.h"
#include "BulletCollision/BroadphaseCollision/btDispatcher.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseInterface.h"
#include "BulletCollision/CollisionShapes/btTriangleCallback.h"
#include "BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"
class btDispatcher;
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"
#include "btCollisionCreateFunc.h"

///For each triangle in the concave mesh that overlaps with the AABB of a convex (m_convexProxy), processTriangle is called.
ATTRIBUTE_ALIGNED16(class)  btConvexTriangleCallback : public btTriangleCallback
{

	btVector3	m_aabbMin;
	btVector3	m_aabbMax ;

	const btCollisionObjectWrapper* m_convexBodyWrap;
	const btCollisionObjectWrapper* m_triBodyWrap;



	btManifoldResult* m_resultOut;
	btDispatcher*	m_dispatcher;
	const btDispatcherInfo* m_dispatchInfoPtr;
	btScalar m_collisionMarginTriangle;
	
public:
	BT_DECLARE_ALIGNED_ALLOCATOR();
	
int	m_triangleCount;
	
	btPersistentManifold*	m_manifoldPtr;

	btConvexTriangleCallback(btDispatcher* dispatcher,const btCollisionObjectWrapper* body0Wrap,const btCollisionObjectWrapper* body1Wrap,bool isSwapped);

	void	setTimeStepAndCounters(btScalar collisionMarginTriangle,const btDispatcherInfo& dispatchInfo,const btCollisionObjectWrapper* convexBodyWrap, const btCollisionObjectWrapper* triBodyWrap, btManifoldResult* resultOut);

	void	clearWrapperData()
	{
		m_convexBodyWrap = 0;
		m_triBodyWrap = 0;
	}
	virtual ~btConvexTriangleCallback();

	virtual void processTriangle(btVector3* triangle, int partId, int triangleIndex);
	
	void clearCache();

	SIMD_FORCE_INLINE const btVector3& getAabbMin() const
	{
		return m_aabbMin;
	}
	SIMD_FORCE_INLINE const btVector3& getAabbMax() const
	{
		return m_aabbMax;
	}

};




/// btConvexConcaveCollisionAlgorithm  supports collision between convex shapes and (concave) trianges meshes.
ATTRIBUTE_ALIGNED16(class)  btConvexConcaveCollisionAlgorithm  : public btActivatingCollisionAlgorithm
{

	btConvexTriangleCallback m_btConvexTriangleCallback;

	bool	m_isSwapped;



public:

	BT_DECLARE_ALIGNED_ALLOCATOR();
	
	btConvexConcaveCollisionAlgorithm( const btCollisionAlgorithmConstructionInfo& ci,const btCollisionObjectWrapper* body0Wrap,const btCollisionObjectWrapper* body1Wrap,bool isSwapped);

	virtual ~btConvexConcaveCollisionAlgorithm();

	virtual void processCollision (const btCollisionObjectWrapper* body0Wrap,const btCollisionObjectWrapper* body1Wrap,const btDispatcherInfo& dispatchInfo,btManifoldResult* resultOut);

	btScalar	calculateTimeOfImpact(btCollisionObject* body0,btCollisionObject* body1,const btDispatcherInfo& dispatchInfo,btManifoldResult* resultOut);

	virtual	void	getAllContactManifolds(btManifoldArray&	manifoldArray);
	
	void	clearCache();

	struct CreateFunc :public 	btCollisionAlgorithmCreateFunc
	{
		virtual	btCollisionAlgorithm* CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* body0Wrap,const btCollisionObjectWrapper* body1Wrap)
		{
			void* mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(btConvexConcaveCollisionAlgorithm));
			return new(mem) btConvexConcaveCollisionAlgorithm(ci,body0Wrap,body1Wrap,false);
		}
	};

	struct SwappedCreateFunc :public 	btCollisionAlgorithmCreateFunc
	{
		virtual	btCollisionAlgorithm* CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* body0Wrap,const btCollisionObjectWrapper* body1Wrap)
		{
			void* mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(btConvexConcaveCollisionAlgorithm));
			return new(mem) btConvexConcaveCollisionAlgorithm(ci,body0Wrap,body1Wrap,true);
		}
	};

};

#endif //BT_CONVEX_CONCAVE_COLLISION_ALGORITHM_H
