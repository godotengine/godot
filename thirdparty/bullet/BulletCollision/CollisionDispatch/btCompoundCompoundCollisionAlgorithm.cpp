/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2013 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

*/

#include "btCompoundCompoundCollisionAlgorithm.h"
#include "LinearMath/btQuickprof.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/CollisionShapes/btCompoundShape.h"
#include "BulletCollision/BroadphaseCollision/btDbvt.h"
#include "LinearMath/btIDebugDraw.h"
#include "LinearMath/btAabbUtil2.h"
#include "BulletCollision/CollisionDispatch/btManifoldResult.h"
#include "BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h"

//USE_LOCAL_STACK will avoid most (often all) dynamic memory allocations due to resizing in processCollision and MycollideTT
#define USE_LOCAL_STACK 1

btShapePairCallback gCompoundCompoundChildShapePairCallback = 0;

btCompoundCompoundCollisionAlgorithm::btCompoundCompoundCollisionAlgorithm(const btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap, bool isSwapped)
	: btCompoundCollisionAlgorithm(ci, body0Wrap, body1Wrap, isSwapped)
{
	void* ptr = btAlignedAlloc(sizeof(btHashedSimplePairCache), 16);
	m_childCollisionAlgorithmCache = new (ptr) btHashedSimplePairCache();

	const btCollisionObjectWrapper* col0ObjWrap = body0Wrap;
	btAssert(col0ObjWrap->getCollisionShape()->isCompound());

	const btCollisionObjectWrapper* col1ObjWrap = body1Wrap;
	btAssert(col1ObjWrap->getCollisionShape()->isCompound());

	const btCompoundShape* compoundShape0 = static_cast<const btCompoundShape*>(col0ObjWrap->getCollisionShape());
	m_compoundShapeRevision0 = compoundShape0->getUpdateRevision();

	const btCompoundShape* compoundShape1 = static_cast<const btCompoundShape*>(col1ObjWrap->getCollisionShape());
	m_compoundShapeRevision1 = compoundShape1->getUpdateRevision();
}

btCompoundCompoundCollisionAlgorithm::~btCompoundCompoundCollisionAlgorithm()
{
	removeChildAlgorithms();
	m_childCollisionAlgorithmCache->~btHashedSimplePairCache();
	btAlignedFree(m_childCollisionAlgorithmCache);
}

void btCompoundCompoundCollisionAlgorithm::getAllContactManifolds(btManifoldArray& manifoldArray)
{
	int i;
	btSimplePairArray& pairs = m_childCollisionAlgorithmCache->getOverlappingPairArray();
	for (i = 0; i < pairs.size(); i++)
	{
		if (pairs[i].m_userPointer)
		{
			((btCollisionAlgorithm*)pairs[i].m_userPointer)->getAllContactManifolds(manifoldArray);
		}
	}
}

void btCompoundCompoundCollisionAlgorithm::removeChildAlgorithms()
{
	btSimplePairArray& pairs = m_childCollisionAlgorithmCache->getOverlappingPairArray();

	int numChildren = pairs.size();
	int i;
	for (i = 0; i < numChildren; i++)
	{
		if (pairs[i].m_userPointer)
		{
			btCollisionAlgorithm* algo = (btCollisionAlgorithm*)pairs[i].m_userPointer;
			algo->~btCollisionAlgorithm();
			m_dispatcher->freeCollisionAlgorithm(algo);
		}
	}
	m_childCollisionAlgorithmCache->removeAllPairs();
}

struct btCompoundCompoundLeafCallback : btDbvt::ICollide
{
	int m_numOverlapPairs;

	const btCollisionObjectWrapper* m_compound0ColObjWrap;
	const btCollisionObjectWrapper* m_compound1ColObjWrap;
	btDispatcher* m_dispatcher;
	const btDispatcherInfo& m_dispatchInfo;
	btManifoldResult* m_resultOut;

	class btHashedSimplePairCache* m_childCollisionAlgorithmCache;

	btPersistentManifold* m_sharedManifold;

	btCompoundCompoundLeafCallback(const btCollisionObjectWrapper* compound1ObjWrap,
								   const btCollisionObjectWrapper* compound0ObjWrap,
								   btDispatcher* dispatcher,
								   const btDispatcherInfo& dispatchInfo,
								   btManifoldResult* resultOut,
								   btHashedSimplePairCache* childAlgorithmsCache,
								   btPersistentManifold* sharedManifold)
		: m_numOverlapPairs(0), m_compound0ColObjWrap(compound1ObjWrap), m_compound1ColObjWrap(compound0ObjWrap), m_dispatcher(dispatcher), m_dispatchInfo(dispatchInfo), m_resultOut(resultOut), m_childCollisionAlgorithmCache(childAlgorithmsCache), m_sharedManifold(sharedManifold)
	{
	}

	void Process(const btDbvtNode* leaf0, const btDbvtNode* leaf1)
	{
		BT_PROFILE("btCompoundCompoundLeafCallback::Process");
		m_numOverlapPairs++;

		int childIndex0 = leaf0->dataAsInt;
		int childIndex1 = leaf1->dataAsInt;

		btAssert(childIndex0 >= 0);
		btAssert(childIndex1 >= 0);

		const btCompoundShape* compoundShape0 = static_cast<const btCompoundShape*>(m_compound0ColObjWrap->getCollisionShape());
		btAssert(childIndex0 < compoundShape0->getNumChildShapes());

		const btCompoundShape* compoundShape1 = static_cast<const btCompoundShape*>(m_compound1ColObjWrap->getCollisionShape());
		btAssert(childIndex1 < compoundShape1->getNumChildShapes());

		const btCollisionShape* childShape0 = compoundShape0->getChildShape(childIndex0);
		const btCollisionShape* childShape1 = compoundShape1->getChildShape(childIndex1);

		//backup
		btTransform orgTrans0 = m_compound0ColObjWrap->getWorldTransform();
		const btTransform& childTrans0 = compoundShape0->getChildTransform(childIndex0);
		btTransform newChildWorldTrans0 = orgTrans0 * childTrans0;

		btTransform orgTrans1 = m_compound1ColObjWrap->getWorldTransform();
		const btTransform& childTrans1 = compoundShape1->getChildTransform(childIndex1);
		btTransform newChildWorldTrans1 = orgTrans1 * childTrans1;

		//perform an AABB check first
		btVector3 aabbMin0, aabbMax0, aabbMin1, aabbMax1;
		childShape0->getAabb(newChildWorldTrans0, aabbMin0, aabbMax0);
		childShape1->getAabb(newChildWorldTrans1, aabbMin1, aabbMax1);

		btVector3 thresholdVec(m_resultOut->m_closestPointDistanceThreshold, m_resultOut->m_closestPointDistanceThreshold, m_resultOut->m_closestPointDistanceThreshold);

		aabbMin0 -= thresholdVec;
		aabbMax0 += thresholdVec;

		if (gCompoundCompoundChildShapePairCallback)
		{
			if (!gCompoundCompoundChildShapePairCallback(childShape0, childShape1))
				return;
		}

		if (TestAabbAgainstAabb2(aabbMin0, aabbMax0, aabbMin1, aabbMax1))
		{
			btCollisionObjectWrapper compoundWrap0(this->m_compound0ColObjWrap, childShape0, m_compound0ColObjWrap->getCollisionObject(), newChildWorldTrans0, -1, childIndex0);
			btCollisionObjectWrapper compoundWrap1(this->m_compound1ColObjWrap, childShape1, m_compound1ColObjWrap->getCollisionObject(), newChildWorldTrans1, -1, childIndex1);

			btSimplePair* pair = m_childCollisionAlgorithmCache->findPair(childIndex0, childIndex1);
			bool removePair = false;
			btCollisionAlgorithm* colAlgo = 0;
			if (m_resultOut->m_closestPointDistanceThreshold > 0)
			{
				colAlgo = m_dispatcher->findAlgorithm(&compoundWrap0, &compoundWrap1, 0, BT_CLOSEST_POINT_ALGORITHMS);
				removePair = true;
			}
			else
			{
				if (pair)
				{
					colAlgo = (btCollisionAlgorithm*)pair->m_userPointer;
				}
				else
				{
					colAlgo = m_dispatcher->findAlgorithm(&compoundWrap0, &compoundWrap1, m_sharedManifold, BT_CONTACT_POINT_ALGORITHMS);
					pair = m_childCollisionAlgorithmCache->addOverlappingPair(childIndex0, childIndex1);
					btAssert(pair);
					pair->m_userPointer = colAlgo;
				}
			}

			btAssert(colAlgo);

			const btCollisionObjectWrapper* tmpWrap0 = 0;
			const btCollisionObjectWrapper* tmpWrap1 = 0;

			tmpWrap0 = m_resultOut->getBody0Wrap();
			tmpWrap1 = m_resultOut->getBody1Wrap();

			m_resultOut->setBody0Wrap(&compoundWrap0);
			m_resultOut->setBody1Wrap(&compoundWrap1);

			m_resultOut->setShapeIdentifiersA(-1, childIndex0);
			m_resultOut->setShapeIdentifiersB(-1, childIndex1);

			colAlgo->processCollision(&compoundWrap0, &compoundWrap1, m_dispatchInfo, m_resultOut);

			m_resultOut->setBody0Wrap(tmpWrap0);
			m_resultOut->setBody1Wrap(tmpWrap1);

			if (removePair)
			{
				colAlgo->~btCollisionAlgorithm();
				m_dispatcher->freeCollisionAlgorithm(colAlgo);
			}
		}
	}
};

static DBVT_INLINE bool MyIntersect(const btDbvtAabbMm& a,
									const btDbvtAabbMm& b, const btTransform& xform, btScalar distanceThreshold)
{
	btVector3 newmin, newmax;
	btTransformAabb(b.Mins(), b.Maxs(), 0.f, xform, newmin, newmax);
	newmin -= btVector3(distanceThreshold, distanceThreshold, distanceThreshold);
	newmax += btVector3(distanceThreshold, distanceThreshold, distanceThreshold);
	btDbvtAabbMm newb = btDbvtAabbMm::FromMM(newmin, newmax);
	return Intersect(a, newb);
}

static inline void MycollideTT(const btDbvtNode* root0,
							   const btDbvtNode* root1,
							   const btTransform& xform,
							   btCompoundCompoundLeafCallback* callback, btScalar distanceThreshold)
{
	if (root0 && root1)
	{
		int depth = 1;
		int treshold = btDbvt::DOUBLE_STACKSIZE - 4;
		btAlignedObjectArray<btDbvt::sStkNN> stkStack;
#ifdef USE_LOCAL_STACK
		ATTRIBUTE_ALIGNED16(btDbvt::sStkNN localStack[btDbvt::DOUBLE_STACKSIZE]);
		stkStack.initializeFromBuffer(&localStack, btDbvt::DOUBLE_STACKSIZE, btDbvt::DOUBLE_STACKSIZE);
#else
		stkStack.resize(btDbvt::DOUBLE_STACKSIZE);
#endif
		stkStack[0] = btDbvt::sStkNN(root0, root1);
		do
		{
			btDbvt::sStkNN p = stkStack[--depth];
			if (MyIntersect(p.a->volume, p.b->volume, xform, distanceThreshold))
			{
				if (depth > treshold)
				{
					stkStack.resize(stkStack.size() * 2);
					treshold = stkStack.size() - 4;
				}
				if (p.a->isinternal())
				{
					if (p.b->isinternal())
					{
						stkStack[depth++] = btDbvt::sStkNN(p.a->childs[0], p.b->childs[0]);
						stkStack[depth++] = btDbvt::sStkNN(p.a->childs[1], p.b->childs[0]);
						stkStack[depth++] = btDbvt::sStkNN(p.a->childs[0], p.b->childs[1]);
						stkStack[depth++] = btDbvt::sStkNN(p.a->childs[1], p.b->childs[1]);
					}
					else
					{
						stkStack[depth++] = btDbvt::sStkNN(p.a->childs[0], p.b);
						stkStack[depth++] = btDbvt::sStkNN(p.a->childs[1], p.b);
					}
				}
				else
				{
					if (p.b->isinternal())
					{
						stkStack[depth++] = btDbvt::sStkNN(p.a, p.b->childs[0]);
						stkStack[depth++] = btDbvt::sStkNN(p.a, p.b->childs[1]);
					}
					else
					{
						callback->Process(p.a, p.b);
					}
				}
			}
		} while (depth);
	}
}

void btCompoundCompoundCollisionAlgorithm::processCollision(const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut)
{
	const btCollisionObjectWrapper* col0ObjWrap = body0Wrap;
	const btCollisionObjectWrapper* col1ObjWrap = body1Wrap;

	btAssert(col0ObjWrap->getCollisionShape()->isCompound());
	btAssert(col1ObjWrap->getCollisionShape()->isCompound());
	const btCompoundShape* compoundShape0 = static_cast<const btCompoundShape*>(col0ObjWrap->getCollisionShape());
	const btCompoundShape* compoundShape1 = static_cast<const btCompoundShape*>(col1ObjWrap->getCollisionShape());

	const btDbvt* tree0 = compoundShape0->getDynamicAabbTree();
	const btDbvt* tree1 = compoundShape1->getDynamicAabbTree();
	if (!tree0 || !tree1)
	{
		return btCompoundCollisionAlgorithm::processCollision(body0Wrap, body1Wrap, dispatchInfo, resultOut);
	}
	///btCompoundShape might have changed:
	////make sure the internal child collision algorithm caches are still valid
	if ((compoundShape0->getUpdateRevision() != m_compoundShapeRevision0) || (compoundShape1->getUpdateRevision() != m_compoundShapeRevision1))
	{
		///clear all
		removeChildAlgorithms();
		m_compoundShapeRevision0 = compoundShape0->getUpdateRevision();
		m_compoundShapeRevision1 = compoundShape1->getUpdateRevision();
	}

	///we need to refresh all contact manifolds
	///note that we should actually recursively traverse all children, btCompoundShape can nested more then 1 level deep
	///so we should add a 'refreshManifolds' in the btCollisionAlgorithm
	{
		int i;
		btManifoldArray manifoldArray;
#ifdef USE_LOCAL_STACK
		btPersistentManifold localManifolds[4];
		manifoldArray.initializeFromBuffer(&localManifolds, 0, 4);
#endif
		btSimplePairArray& pairs = m_childCollisionAlgorithmCache->getOverlappingPairArray();
		for (i = 0; i < pairs.size(); i++)
		{
			if (pairs[i].m_userPointer)
			{
				btCollisionAlgorithm* algo = (btCollisionAlgorithm*)pairs[i].m_userPointer;
				algo->getAllContactManifolds(manifoldArray);
				for (int m = 0; m < manifoldArray.size(); m++)
				{
					if (manifoldArray[m]->getNumContacts())
					{
						resultOut->setPersistentManifold(manifoldArray[m]);
						resultOut->refreshContactPoints();
						resultOut->setPersistentManifold(0);
					}
				}
				manifoldArray.resize(0);
			}
		}
	}

	btCompoundCompoundLeafCallback callback(col0ObjWrap, col1ObjWrap, this->m_dispatcher, dispatchInfo, resultOut, this->m_childCollisionAlgorithmCache, m_sharedManifold);

	const btTransform xform = col0ObjWrap->getWorldTransform().inverse() * col1ObjWrap->getWorldTransform();
	MycollideTT(tree0->m_root, tree1->m_root, xform, &callback, resultOut->m_closestPointDistanceThreshold);

	//printf("#compound-compound child/leaf overlap =%d                      \r",callback.m_numOverlapPairs);

	//remove non-overlapping child pairs

	{
		btAssert(m_removePairs.size() == 0);

		//iterate over all children, perform an AABB check inside ProcessChildShape
		btSimplePairArray& pairs = m_childCollisionAlgorithmCache->getOverlappingPairArray();

		int i;
		btManifoldArray manifoldArray;

		btVector3 aabbMin0, aabbMax0, aabbMin1, aabbMax1;

		for (i = 0; i < pairs.size(); i++)
		{
			if (pairs[i].m_userPointer)
			{
				btCollisionAlgorithm* algo = (btCollisionAlgorithm*)pairs[i].m_userPointer;

				{
					const btCollisionShape* childShape0 = 0;

					btTransform newChildWorldTrans0;
					childShape0 = compoundShape0->getChildShape(pairs[i].m_indexA);
					const btTransform& childTrans0 = compoundShape0->getChildTransform(pairs[i].m_indexA);
					newChildWorldTrans0 = col0ObjWrap->getWorldTransform() * childTrans0;
					childShape0->getAabb(newChildWorldTrans0, aabbMin0, aabbMax0);
				}
				btVector3 thresholdVec(resultOut->m_closestPointDistanceThreshold, resultOut->m_closestPointDistanceThreshold, resultOut->m_closestPointDistanceThreshold);
				aabbMin0 -= thresholdVec;
				aabbMax0 += thresholdVec;
				{
					const btCollisionShape* childShape1 = 0;
					btTransform newChildWorldTrans1;

					childShape1 = compoundShape1->getChildShape(pairs[i].m_indexB);
					const btTransform& childTrans1 = compoundShape1->getChildTransform(pairs[i].m_indexB);
					newChildWorldTrans1 = col1ObjWrap->getWorldTransform() * childTrans1;
					childShape1->getAabb(newChildWorldTrans1, aabbMin1, aabbMax1);
				}

				aabbMin1 -= thresholdVec;
				aabbMax1 += thresholdVec;

				if (!TestAabbAgainstAabb2(aabbMin0, aabbMax0, aabbMin1, aabbMax1))
				{
					algo->~btCollisionAlgorithm();
					m_dispatcher->freeCollisionAlgorithm(algo);
					m_removePairs.push_back(btSimplePair(pairs[i].m_indexA, pairs[i].m_indexB));
				}
			}
		}
		for (int i = 0; i < m_removePairs.size(); i++)
		{
			m_childCollisionAlgorithmCache->removeOverlappingPair(m_removePairs[i].m_indexA, m_removePairs[i].m_indexB);
		}
		m_removePairs.clear();
	}
}

btScalar btCompoundCompoundCollisionAlgorithm::calculateTimeOfImpact(btCollisionObject* body0, btCollisionObject* body1, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut)
{
	btAssert(0);
	return 0.f;
}
