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

#include "BulletCollision/CollisionDispatch/btCompoundCollisionAlgorithm.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/CollisionShapes/btCompoundShape.h"
#include "BulletCollision/BroadphaseCollision/btDbvt.h"
#include "LinearMath/btIDebugDraw.h"
#include "LinearMath/btAabbUtil2.h"
#include "btManifoldResult.h"
#include "BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h"

btShapePairCallback gCompoundChildShapePairCallback = 0;

btCompoundCollisionAlgorithm::btCompoundCollisionAlgorithm(const btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap, bool isSwapped)
	: btActivatingCollisionAlgorithm(ci, body0Wrap, body1Wrap),
	  m_isSwapped(isSwapped),
	  m_sharedManifold(ci.m_manifold)
{
	m_ownsManifold = false;

	const btCollisionObjectWrapper* colObjWrap = m_isSwapped ? body1Wrap : body0Wrap;
	btAssert(colObjWrap->getCollisionShape()->isCompound());

	const btCompoundShape* compoundShape = static_cast<const btCompoundShape*>(colObjWrap->getCollisionShape());
	m_compoundShapeRevision = compoundShape->getUpdateRevision();

	preallocateChildAlgorithms(body0Wrap, body1Wrap);
}

void btCompoundCollisionAlgorithm::preallocateChildAlgorithms(const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap)
{
	const btCollisionObjectWrapper* colObjWrap = m_isSwapped ? body1Wrap : body0Wrap;
	const btCollisionObjectWrapper* otherObjWrap = m_isSwapped ? body0Wrap : body1Wrap;
	btAssert(colObjWrap->getCollisionShape()->isCompound());

	const btCompoundShape* compoundShape = static_cast<const btCompoundShape*>(colObjWrap->getCollisionShape());

	int numChildren = compoundShape->getNumChildShapes();
	int i;

	m_childCollisionAlgorithms.resize(numChildren);
	for (i = 0; i < numChildren; i++)
	{
		if (compoundShape->getDynamicAabbTree())
		{
			m_childCollisionAlgorithms[i] = 0;
		}
		else
		{
			const btCollisionShape* childShape = compoundShape->getChildShape(i);

			btCollisionObjectWrapper childWrap(colObjWrap, childShape, colObjWrap->getCollisionObject(), colObjWrap->getWorldTransform(), -1, i);  //wrong child trans, but unused (hopefully)
			m_childCollisionAlgorithms[i] = m_dispatcher->findAlgorithm(&childWrap, otherObjWrap, m_sharedManifold, BT_CONTACT_POINT_ALGORITHMS);

			btAlignedObjectArray<btCollisionAlgorithm*> m_childCollisionAlgorithmsContact;
			btAlignedObjectArray<btCollisionAlgorithm*> m_childCollisionAlgorithmsClosestPoints;
		}
	}
}

void btCompoundCollisionAlgorithm::removeChildAlgorithms()
{
	int numChildren = m_childCollisionAlgorithms.size();
	int i;
	for (i = 0; i < numChildren; i++)
	{
		if (m_childCollisionAlgorithms[i])
		{
			m_childCollisionAlgorithms[i]->~btCollisionAlgorithm();
			m_dispatcher->freeCollisionAlgorithm(m_childCollisionAlgorithms[i]);
		}
	}
}

btCompoundCollisionAlgorithm::~btCompoundCollisionAlgorithm()
{
	removeChildAlgorithms();
}

struct btCompoundLeafCallback : btDbvt::ICollide
{
public:
	const btCollisionObjectWrapper* m_compoundColObjWrap;
	const btCollisionObjectWrapper* m_otherObjWrap;
	btDispatcher* m_dispatcher;
	const btDispatcherInfo& m_dispatchInfo;
	btManifoldResult* m_resultOut;
	btCollisionAlgorithm** m_childCollisionAlgorithms;
	btPersistentManifold* m_sharedManifold;

	btCompoundLeafCallback(const btCollisionObjectWrapper* compoundObjWrap, const btCollisionObjectWrapper* otherObjWrap, btDispatcher* dispatcher, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut, btCollisionAlgorithm** childCollisionAlgorithms, btPersistentManifold* sharedManifold)
		: m_compoundColObjWrap(compoundObjWrap), m_otherObjWrap(otherObjWrap), m_dispatcher(dispatcher), m_dispatchInfo(dispatchInfo), m_resultOut(resultOut), m_childCollisionAlgorithms(childCollisionAlgorithms), m_sharedManifold(sharedManifold)
	{
	}

	void ProcessChildShape(const btCollisionShape* childShape, int index)
	{
		btAssert(index >= 0);
		const btCompoundShape* compoundShape = static_cast<const btCompoundShape*>(m_compoundColObjWrap->getCollisionShape());
		btAssert(index < compoundShape->getNumChildShapes());

		if (gCompoundChildShapePairCallback)
		{
			if (!gCompoundChildShapePairCallback(m_otherObjWrap->getCollisionShape(), childShape))
				return;
		}

		//backup
		btTransform orgTrans = m_compoundColObjWrap->getWorldTransform();

		const btTransform& childTrans = compoundShape->getChildTransform(index);
		btTransform newChildWorldTrans = orgTrans * childTrans;

		//perform an AABB check first
		btVector3 aabbMin0, aabbMax0;
		childShape->getAabb(newChildWorldTrans, aabbMin0, aabbMax0);

		btVector3 extendAabb(m_resultOut->m_closestPointDistanceThreshold, m_resultOut->m_closestPointDistanceThreshold, m_resultOut->m_closestPointDistanceThreshold);
		aabbMin0 -= extendAabb;
		aabbMax0 += extendAabb;

		btVector3 aabbMin1, aabbMax1;
		m_otherObjWrap->getCollisionShape()->getAabb(m_otherObjWrap->getWorldTransform(), aabbMin1, aabbMax1);


		if (TestAabbAgainstAabb2(aabbMin0, aabbMax0, aabbMin1, aabbMax1))
		{
			btCollisionObjectWrapper compoundWrap(this->m_compoundColObjWrap, childShape, m_compoundColObjWrap->getCollisionObject(), newChildWorldTrans, childTrans, -1, index);

			btCollisionAlgorithm* algo = 0;
			bool allocatedAlgorithm = false;

			if (m_resultOut->m_closestPointDistanceThreshold > 0)
			{
				algo = m_dispatcher->findAlgorithm(&compoundWrap, m_otherObjWrap, 0, BT_CLOSEST_POINT_ALGORITHMS);
				allocatedAlgorithm = true;
			}
			else
			{
				//the contactpoint is still projected back using the original inverted worldtrans
				if (!m_childCollisionAlgorithms[index])
				{
					m_childCollisionAlgorithms[index] = m_dispatcher->findAlgorithm(&compoundWrap, m_otherObjWrap, m_sharedManifold, BT_CONTACT_POINT_ALGORITHMS);
				}
				algo = m_childCollisionAlgorithms[index];
			}

			const btCollisionObjectWrapper* tmpWrap = 0;

			///detect swapping case
			if (m_resultOut->getBody0Internal() == m_compoundColObjWrap->getCollisionObject())
			{
				tmpWrap = m_resultOut->getBody0Wrap();
				m_resultOut->setBody0Wrap(&compoundWrap);
				m_resultOut->setShapeIdentifiersA(-1, index);
			}
			else
			{
				tmpWrap = m_resultOut->getBody1Wrap();
				m_resultOut->setBody1Wrap(&compoundWrap);
				m_resultOut->setShapeIdentifiersB(-1, index);
			}

			algo->processCollision(&compoundWrap, m_otherObjWrap, m_dispatchInfo, m_resultOut);

#if 0
			if (m_dispatchInfo.m_debugDraw && (m_dispatchInfo.m_debugDraw->getDebugMode() & btIDebugDraw::DBG_DrawAabb))
			{
				btVector3 worldAabbMin,worldAabbMax;
				m_dispatchInfo.m_debugDraw->drawAabb(aabbMin0,aabbMax0,btVector3(1,1,1));
				m_dispatchInfo.m_debugDraw->drawAabb(aabbMin1,aabbMax1,btVector3(1,1,1));
			}
#endif

			if (m_resultOut->getBody0Internal() == m_compoundColObjWrap->getCollisionObject())
			{
				m_resultOut->setBody0Wrap(tmpWrap);
			}
			else
			{
				m_resultOut->setBody1Wrap(tmpWrap);
			}
			if (allocatedAlgorithm)
			{
				algo->~btCollisionAlgorithm();
				m_dispatcher->freeCollisionAlgorithm(algo);
			}
		}
	}
	void Process(const btDbvtNode* leaf)
	{
		int index = leaf->dataAsInt;

		const btCompoundShape* compoundShape = static_cast<const btCompoundShape*>(m_compoundColObjWrap->getCollisionShape());
		const btCollisionShape* childShape = compoundShape->getChildShape(index);

#if 0
		if (m_dispatchInfo.m_debugDraw && (m_dispatchInfo.m_debugDraw->getDebugMode() & btIDebugDraw::DBG_DrawAabb))
		{
			btVector3 worldAabbMin,worldAabbMax;
			btTransform	orgTrans = m_compoundColObjWrap->getWorldTransform();
			btTransformAabb(leaf->volume.Mins(),leaf->volume.Maxs(),0.,orgTrans,worldAabbMin,worldAabbMax);
			m_dispatchInfo.m_debugDraw->drawAabb(worldAabbMin,worldAabbMax,btVector3(1,0,0));
		}
#endif

		ProcessChildShape(childShape, index);
	}
};

void btCompoundCollisionAlgorithm::processCollision(const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut)
{
	const btCollisionObjectWrapper* colObjWrap = m_isSwapped ? body1Wrap : body0Wrap;
	const btCollisionObjectWrapper* otherObjWrap = m_isSwapped ? body0Wrap : body1Wrap;

	btAssert(colObjWrap->getCollisionShape()->isCompound());
	const btCompoundShape* compoundShape = static_cast<const btCompoundShape*>(colObjWrap->getCollisionShape());

	///btCompoundShape might have changed:
	////make sure the internal child collision algorithm caches are still valid
	if (compoundShape->getUpdateRevision() != m_compoundShapeRevision)
	{
		///clear and update all
		removeChildAlgorithms();

		preallocateChildAlgorithms(body0Wrap, body1Wrap);
		m_compoundShapeRevision = compoundShape->getUpdateRevision();
	}

	if (m_childCollisionAlgorithms.size() == 0)
		return;

	const btDbvt* tree = compoundShape->getDynamicAabbTree();
	//use a dynamic aabb tree to cull potential child-overlaps
	btCompoundLeafCallback callback(colObjWrap, otherObjWrap, m_dispatcher, dispatchInfo, resultOut, &m_childCollisionAlgorithms[0], m_sharedManifold);

	///we need to refresh all contact manifolds
	///note that we should actually recursively traverse all children, btCompoundShape can nested more then 1 level deep
	///so we should add a 'refreshManifolds' in the btCollisionAlgorithm
	{
		int i;
		manifoldArray.resize(0);
		for (i = 0; i < m_childCollisionAlgorithms.size(); i++)
		{
			if (m_childCollisionAlgorithms[i])
			{
				m_childCollisionAlgorithms[i]->getAllContactManifolds(manifoldArray);
				for (int m = 0; m < manifoldArray.size(); m++)
				{
					if (manifoldArray[m]->getNumContacts())
					{
						resultOut->setPersistentManifold(manifoldArray[m]);
						resultOut->refreshContactPoints();
						resultOut->setPersistentManifold(0);  //??necessary?
					}
				}
				manifoldArray.resize(0);
			}
		}
	}

	if (tree)
	{
		btVector3 localAabbMin, localAabbMax;
		btTransform otherInCompoundSpace;
		otherInCompoundSpace = colObjWrap->getWorldTransform().inverse() * otherObjWrap->getWorldTransform();
		otherObjWrap->getCollisionShape()->getAabb(otherInCompoundSpace, localAabbMin, localAabbMax);
		btVector3 extraExtends(resultOut->m_closestPointDistanceThreshold, resultOut->m_closestPointDistanceThreshold, resultOut->m_closestPointDistanceThreshold);
		localAabbMin -= extraExtends;
		localAabbMax += extraExtends;

		const ATTRIBUTE_ALIGNED16(btDbvtVolume) bounds = btDbvtVolume::FromMM(localAabbMin, localAabbMax);
		//process all children, that overlap with  the given AABB bounds
		tree->collideTVNoStackAlloc(tree->m_root, bounds, stack2, callback);
	}
	else
	{
		//iterate over all children, perform an AABB check inside ProcessChildShape
		int numChildren = m_childCollisionAlgorithms.size();
		int i;
		for (i = 0; i < numChildren; i++)
		{
			callback.ProcessChildShape(compoundShape->getChildShape(i), i);
		}
	}

	{
		//iterate over all children, perform an AABB check inside ProcessChildShape
		int numChildren = m_childCollisionAlgorithms.size();
		int i;
		manifoldArray.resize(0);
		const btCollisionShape* childShape = 0;
		btTransform orgTrans;

		btTransform newChildWorldTrans;
		btVector3 aabbMin0, aabbMax0, aabbMin1, aabbMax1;

		for (i = 0; i < numChildren; i++)
		{
			if (m_childCollisionAlgorithms[i])
			{
				childShape = compoundShape->getChildShape(i);
				//if not longer overlapping, remove the algorithm
				orgTrans = colObjWrap->getWorldTransform();

				const btTransform& childTrans = compoundShape->getChildTransform(i);
				newChildWorldTrans = orgTrans * childTrans;

				//perform an AABB check first
				childShape->getAabb(newChildWorldTrans, aabbMin0, aabbMax0);
				otherObjWrap->getCollisionShape()->getAabb(otherObjWrap->getWorldTransform(), aabbMin1, aabbMax1);

				if (!TestAabbAgainstAabb2(aabbMin0, aabbMax0, aabbMin1, aabbMax1))
				{
					m_childCollisionAlgorithms[i]->~btCollisionAlgorithm();
					m_dispatcher->freeCollisionAlgorithm(m_childCollisionAlgorithms[i]);
					m_childCollisionAlgorithms[i] = 0;
				}
			}
		}
	}
}

btScalar btCompoundCollisionAlgorithm::calculateTimeOfImpact(btCollisionObject* body0, btCollisionObject* body1, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut)
{
	btAssert(0);
	//needs to be fixed, using btCollisionObjectWrapper and NOT modifying internal data structures
	btCollisionObject* colObj = m_isSwapped ? body1 : body0;
	btCollisionObject* otherObj = m_isSwapped ? body0 : body1;

	btAssert(colObj->getCollisionShape()->isCompound());

	btCompoundShape* compoundShape = static_cast<btCompoundShape*>(colObj->getCollisionShape());

	//We will use the OptimizedBVH, AABB tree to cull potential child-overlaps
	//If both proxies are Compound, we will deal with that directly, by performing sequential/parallel tree traversals
	//given Proxy0 and Proxy1, if both have a tree, Tree0 and Tree1, this means:
	//determine overlapping nodes of Proxy1 using Proxy0 AABB against Tree1
	//then use each overlapping node AABB against Tree0
	//and vise versa.

	btScalar hitFraction = btScalar(1.);

	int numChildren = m_childCollisionAlgorithms.size();
	int i;
	btTransform orgTrans;
	btScalar frac;
	for (i = 0; i < numChildren; i++)
	{
		//btCollisionShape* childShape = compoundShape->getChildShape(i);

		//backup
		orgTrans = colObj->getWorldTransform();

		const btTransform& childTrans = compoundShape->getChildTransform(i);
		//btTransform	newChildWorldTrans = orgTrans*childTrans ;
		colObj->setWorldTransform(orgTrans * childTrans);

		//btCollisionShape* tmpShape = colObj->getCollisionShape();
		//colObj->internalSetTemporaryCollisionShape( childShape );
		frac = m_childCollisionAlgorithms[i]->calculateTimeOfImpact(colObj, otherObj, dispatchInfo, resultOut);
		if (frac < hitFraction)
		{
			hitFraction = frac;
		}
		//revert back
		//colObj->internalSetTemporaryCollisionShape( tmpShape);
		colObj->setWorldTransform(orgTrans);
	}
	return hitFraction;
}
