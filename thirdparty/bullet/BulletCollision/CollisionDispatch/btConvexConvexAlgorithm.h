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

#ifndef BT_CONVEX_CONVEX_ALGORITHM_H
#define BT_CONVEX_CONVEX_ALGORITHM_H

#include "btActivatingCollisionAlgorithm.h"
#include "BulletCollision/NarrowPhaseCollision/btGjkPairDetector.h"
#include "BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"
#include "BulletCollision/NarrowPhaseCollision/btVoronoiSimplexSolver.h"
#include "btCollisionCreateFunc.h"
#include "btCollisionDispatcher.h"
#include "LinearMath/btTransformUtil.h"  //for btConvexSeparatingDistanceUtil
#include "BulletCollision/NarrowPhaseCollision/btPolyhedralContactClipping.h"

class btConvexPenetrationDepthSolver;

///Enabling USE_SEPDISTANCE_UTIL2 requires 100% reliable distance computation. However, when using large size ratios GJK can be imprecise
///so the distance is not conservative. In that case, enabling this USE_SEPDISTANCE_UTIL2 would result in failing/missing collisions.
///Either improve GJK for large size ratios (testing a 100 units versus a 0.1 unit object) or only enable the util
///for certain pairs that have a small size ratio

//#define USE_SEPDISTANCE_UTIL2 1

///The convexConvexAlgorithm collision algorithm implements time of impact, convex closest points and penetration depth calculations between two convex objects.
///Multiple contact points are calculated by perturbing the orientation of the smallest object orthogonal to the separating normal.
///This idea was described by Gino van den Bergen in this forum topic http://www.bulletphysics.com/Bullet/phpBB3/viewtopic.php?f=4&t=288&p=888#p888
class btConvexConvexAlgorithm : public btActivatingCollisionAlgorithm
{
#ifdef USE_SEPDISTANCE_UTIL2
	btConvexSeparatingDistanceUtil m_sepDistance;
#endif
	btConvexPenetrationDepthSolver* m_pdSolver;

	btVertexArray worldVertsB1;
	btVertexArray worldVertsB2;

	bool m_ownManifold;
	btPersistentManifold* m_manifoldPtr;
	bool m_lowLevelOfDetail;

	int m_numPerturbationIterations;
	int m_minimumPointsPerturbationThreshold;

	///cache separating vector to speedup collision detection

public:
	btConvexConvexAlgorithm(btPersistentManifold* mf, const btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap, btConvexPenetrationDepthSolver* pdSolver, int numPerturbationIterations, int minimumPointsPerturbationThreshold);

	virtual ~btConvexConvexAlgorithm();

	virtual void processCollision(const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut);

	virtual btScalar calculateTimeOfImpact(btCollisionObject* body0, btCollisionObject* body1, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut);

	virtual void getAllContactManifolds(btManifoldArray& manifoldArray)
	{
		///should we use m_ownManifold to avoid adding duplicates?
		if (m_manifoldPtr && m_ownManifold)
			manifoldArray.push_back(m_manifoldPtr);
	}

	void setLowLevelOfDetail(bool useLowLevel);

	const btPersistentManifold* getManifold()
	{
		return m_manifoldPtr;
	}

	struct CreateFunc : public btCollisionAlgorithmCreateFunc
	{
		btConvexPenetrationDepthSolver* m_pdSolver;
		int m_numPerturbationIterations;
		int m_minimumPointsPerturbationThreshold;

		CreateFunc(btConvexPenetrationDepthSolver* pdSolver);

		virtual ~CreateFunc();

		virtual btCollisionAlgorithm* CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap)
		{
			void* mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(btConvexConvexAlgorithm));
			return new (mem) btConvexConvexAlgorithm(ci.m_manifold, ci, body0Wrap, body1Wrap, m_pdSolver, m_numPerturbationIterations, m_minimumPointsPerturbationThreshold);
		}
	};
};

#endif  //BT_CONVEX_CONVEX_ALGORITHM_H
