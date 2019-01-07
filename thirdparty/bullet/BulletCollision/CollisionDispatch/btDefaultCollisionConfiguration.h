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

#ifndef BT_DEFAULT_COLLISION_CONFIGURATION
#define BT_DEFAULT_COLLISION_CONFIGURATION

#include "btCollisionConfiguration.h"
class btVoronoiSimplexSolver;
class btConvexPenetrationDepthSolver;

struct btDefaultCollisionConstructionInfo
{
	btPoolAllocator* m_persistentManifoldPool;
	btPoolAllocator* m_collisionAlgorithmPool;
	int m_defaultMaxPersistentManifoldPoolSize;
	int m_defaultMaxCollisionAlgorithmPoolSize;
	int m_customCollisionAlgorithmMaxElementSize;
	int m_useEpaPenetrationAlgorithm;

	btDefaultCollisionConstructionInfo()
		: m_persistentManifoldPool(0),
		  m_collisionAlgorithmPool(0),
		  m_defaultMaxPersistentManifoldPoolSize(4096),
		  m_defaultMaxCollisionAlgorithmPoolSize(4096),
		  m_customCollisionAlgorithmMaxElementSize(0),
		  m_useEpaPenetrationAlgorithm(true)
	{
	}
};

///btCollisionConfiguration allows to configure Bullet collision detection
///stack allocator, pool memory allocators
///@todo: describe the meaning
class btDefaultCollisionConfiguration : public btCollisionConfiguration
{
protected:
	int m_persistentManifoldPoolSize;

	btPoolAllocator* m_persistentManifoldPool;
	bool m_ownsPersistentManifoldPool;

	btPoolAllocator* m_collisionAlgorithmPool;
	bool m_ownsCollisionAlgorithmPool;

	//default penetration depth solver
	btConvexPenetrationDepthSolver* m_pdSolver;

	//default CreationFunctions, filling the m_doubleDispatch table
	btCollisionAlgorithmCreateFunc* m_convexConvexCreateFunc;
	btCollisionAlgorithmCreateFunc* m_convexConcaveCreateFunc;
	btCollisionAlgorithmCreateFunc* m_swappedConvexConcaveCreateFunc;
	btCollisionAlgorithmCreateFunc* m_compoundCreateFunc;
	btCollisionAlgorithmCreateFunc* m_compoundCompoundCreateFunc;

	btCollisionAlgorithmCreateFunc* m_swappedCompoundCreateFunc;
	btCollisionAlgorithmCreateFunc* m_emptyCreateFunc;
	btCollisionAlgorithmCreateFunc* m_sphereSphereCF;
	btCollisionAlgorithmCreateFunc* m_sphereBoxCF;
	btCollisionAlgorithmCreateFunc* m_boxSphereCF;

	btCollisionAlgorithmCreateFunc* m_boxBoxCF;
	btCollisionAlgorithmCreateFunc* m_sphereTriangleCF;
	btCollisionAlgorithmCreateFunc* m_triangleSphereCF;
	btCollisionAlgorithmCreateFunc* m_planeConvexCF;
	btCollisionAlgorithmCreateFunc* m_convexPlaneCF;

public:
	btDefaultCollisionConfiguration(const btDefaultCollisionConstructionInfo& constructionInfo = btDefaultCollisionConstructionInfo());

	virtual ~btDefaultCollisionConfiguration();

	///memory pools
	virtual btPoolAllocator* getPersistentManifoldPool()
	{
		return m_persistentManifoldPool;
	}

	virtual btPoolAllocator* getCollisionAlgorithmPool()
	{
		return m_collisionAlgorithmPool;
	}

	virtual btCollisionAlgorithmCreateFunc* getCollisionAlgorithmCreateFunc(int proxyType0, int proxyType1);

	virtual btCollisionAlgorithmCreateFunc* getClosestPointsAlgorithmCreateFunc(int proxyType0, int proxyType1);

	///Use this method to allow to generate multiple contact points between at once, between two objects using the generic convex-convex algorithm.
	///By default, this feature is disabled for best performance.
	///@param numPerturbationIterations controls the number of collision queries. Set it to zero to disable the feature.
	///@param minimumPointsPerturbationThreshold is the minimum number of points in the contact cache, above which the feature is disabled
	///3 is a good value for both params, if you want to enable the feature. This is because the default contact cache contains a maximum of 4 points, and one collision query at the unperturbed orientation is performed first.
	///See Bullet/Demos/CollisionDemo for an example how this feature gathers multiple points.
	///@todo we could add a per-object setting of those parameters, for level-of-detail collision detection.
	void setConvexConvexMultipointIterations(int numPerturbationIterations = 3, int minimumPointsPerturbationThreshold = 3);

	void setPlaneConvexMultipointIterations(int numPerturbationIterations = 3, int minimumPointsPerturbationThreshold = 3);
};

#endif  //BT_DEFAULT_COLLISION_CONFIGURATION
