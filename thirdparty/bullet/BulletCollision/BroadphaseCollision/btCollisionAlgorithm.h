/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_COLLISION_ALGORITHM_H
#define BT_COLLISION_ALGORITHM_H

#include "LinearMath/btScalar.h"
#include "LinearMath/btAlignedObjectArray.h"

struct btBroadphaseProxy;
class btDispatcher;
class btManifoldResult;
class btCollisionObject;
struct btCollisionObjectWrapper;
struct btDispatcherInfo;
class btPersistentManifold;

typedef btAlignedObjectArray<btPersistentManifold*> btManifoldArray;

struct btCollisionAlgorithmConstructionInfo
{
	btCollisionAlgorithmConstructionInfo()
		: m_dispatcher1(0),
		  m_manifold(0)
	{
	}
	btCollisionAlgorithmConstructionInfo(btDispatcher* dispatcher, int temp)
		: m_dispatcher1(dispatcher)
	{
		(void)temp;
	}

	btDispatcher* m_dispatcher1;
	btPersistentManifold* m_manifold;

	//	int	getDispatcherId();
};

///btCollisionAlgorithm is an collision interface that is compatible with the Broadphase and btDispatcher.
///It is persistent over frames
class btCollisionAlgorithm
{
protected:
	btDispatcher* m_dispatcher;

protected:
	//	int	getDispatcherId();

public:
	btCollisionAlgorithm(){};

	btCollisionAlgorithm(const btCollisionAlgorithmConstructionInfo& ci);

	virtual ~btCollisionAlgorithm(){};

	virtual void processCollision(const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut) = 0;

	virtual btScalar calculateTimeOfImpact(btCollisionObject* body0, btCollisionObject* body1, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut) = 0;

	virtual void getAllContactManifolds(btManifoldArray& manifoldArray) = 0;
};

#endif  //BT_COLLISION_ALGORITHM_H
