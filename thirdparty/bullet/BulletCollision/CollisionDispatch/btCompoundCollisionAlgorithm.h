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

#ifndef BT_COMPOUND_COLLISION_ALGORITHM_H
#define BT_COMPOUND_COLLISION_ALGORITHM_H

#include "btActivatingCollisionAlgorithm.h"
#include "BulletCollision/BroadphaseCollision/btDispatcher.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseInterface.h"

#include "BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"
class btDispatcher;
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"
#include "btCollisionCreateFunc.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "BulletCollision/BroadphaseCollision/btDbvt.h"
class btDispatcher;
class btCollisionObject;

class btCollisionShape;
typedef bool (*btShapePairCallback)(const btCollisionShape* pShape0, const btCollisionShape* pShape1);
extern btShapePairCallback gCompoundChildShapePairCallback;

/// btCompoundCollisionAlgorithm  supports collision between CompoundCollisionShapes and other collision shapes
class btCompoundCollisionAlgorithm : public btActivatingCollisionAlgorithm
{
	btNodeStack stack2;
	btManifoldArray manifoldArray;

protected:
	btAlignedObjectArray<btCollisionAlgorithm*> m_childCollisionAlgorithms;
	bool m_isSwapped;

	class btPersistentManifold* m_sharedManifold;
	bool m_ownsManifold;

	int m_compoundShapeRevision;  //to keep track of changes, so that childAlgorithm array can be updated

	void removeChildAlgorithms();

	void preallocateChildAlgorithms(const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap);

public:
	btCompoundCollisionAlgorithm(const btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap, bool isSwapped);

	virtual ~btCompoundCollisionAlgorithm();

	btCollisionAlgorithm* getChildAlgorithm(int n) const
	{
		return m_childCollisionAlgorithms[n];
	}

	virtual void processCollision(const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut);

	btScalar calculateTimeOfImpact(btCollisionObject* body0, btCollisionObject* body1, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut);

	virtual void getAllContactManifolds(btManifoldArray& manifoldArray)
	{
		int i;
		for (i = 0; i < m_childCollisionAlgorithms.size(); i++)
		{
			if (m_childCollisionAlgorithms[i])
				m_childCollisionAlgorithms[i]->getAllContactManifolds(manifoldArray);
		}
	}

	struct CreateFunc : public btCollisionAlgorithmCreateFunc
	{
		virtual btCollisionAlgorithm* CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap)
		{
			void* mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(btCompoundCollisionAlgorithm));
			return new (mem) btCompoundCollisionAlgorithm(ci, body0Wrap, body1Wrap, false);
		}
	};

	struct SwappedCreateFunc : public btCollisionAlgorithmCreateFunc
	{
		virtual btCollisionAlgorithm* CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap)
		{
			void* mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(btCompoundCollisionAlgorithm));
			return new (mem) btCompoundCollisionAlgorithm(ci, body0Wrap, body1Wrap, true);
		}
	};
};

#endif  //BT_COMPOUND_COLLISION_ALGORITHM_H
