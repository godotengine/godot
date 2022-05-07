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

#ifndef BT_SOFT_RIGID_COLLISION_ALGORITHM_H
#define BT_SOFT_RIGID_COLLISION_ALGORITHM_H

#include "BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"
#include "BulletCollision/CollisionDispatch/btCollisionCreateFunc.h"
class btPersistentManifold;
#include "BulletCollision/CollisionDispatch/btCollisionDispatcher.h"

#include "LinearMath/btVector3.h"
class btSoftBody;

/// btSoftRigidCollisionAlgorithm  provides collision detection between btSoftBody and btRigidBody
class btSoftRigidCollisionAlgorithm : public btCollisionAlgorithm
{
	//	bool	m_ownManifold;
	//	btPersistentManifold*	m_manifoldPtr;

	//btSoftBody*				m_softBody;
	//btCollisionObject*		m_rigidCollisionObject;

	///for rigid versus soft (instead of soft versus rigid), we use this swapped boolean
	bool m_isSwapped;

public:
	btSoftRigidCollisionAlgorithm(btPersistentManifold* mf, const btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* col0, const btCollisionObjectWrapper* col1Wrap, bool isSwapped);

	virtual ~btSoftRigidCollisionAlgorithm();

	virtual void processCollision(const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut);

	virtual btScalar calculateTimeOfImpact(btCollisionObject* body0, btCollisionObject* body1, const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut);

	virtual void getAllContactManifolds(btManifoldArray& manifoldArray)
	{
		//we don't add any manifolds
	}

	struct CreateFunc : public btCollisionAlgorithmCreateFunc
	{
		virtual btCollisionAlgorithm* CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap)
		{
			void* mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(btSoftRigidCollisionAlgorithm));
			if (!m_swapped)
			{
				return new (mem) btSoftRigidCollisionAlgorithm(0, ci, body0Wrap, body1Wrap, false);
			}
			else
			{
				return new (mem) btSoftRigidCollisionAlgorithm(0, ci, body0Wrap, body1Wrap, true);
			}
		}
	};
};

#endif  //BT_SOFT_RIGID_COLLISION_ALGORITHM_H
