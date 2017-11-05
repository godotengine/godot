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

#ifndef BT_BOX_2D_BOX_2D__COLLISION_ALGORITHM_H
#define BT_BOX_2D_BOX_2D__COLLISION_ALGORITHM_H

#include "BulletCollision/CollisionDispatch/btActivatingCollisionAlgorithm.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"
#include "BulletCollision/BroadphaseCollision/btDispatcher.h"
#include "BulletCollision/CollisionDispatch/btCollisionCreateFunc.h"

class btPersistentManifold;

///box-box collision detection
class btBox2dBox2dCollisionAlgorithm : public btActivatingCollisionAlgorithm
{
	bool	m_ownManifold;
	btPersistentManifold*	m_manifoldPtr;
	
public:
	btBox2dBox2dCollisionAlgorithm(const btCollisionAlgorithmConstructionInfo& ci)
		: btActivatingCollisionAlgorithm(ci) {}

	virtual void processCollision (const btCollisionObjectWrapper* body0Wrap,const btCollisionObjectWrapper* body1Wrap,const btDispatcherInfo& dispatchInfo,btManifoldResult* resultOut);

	virtual btScalar calculateTimeOfImpact(btCollisionObject* body0,btCollisionObject* body1,const btDispatcherInfo& dispatchInfo,btManifoldResult* resultOut);

	btBox2dBox2dCollisionAlgorithm(btPersistentManifold* mf,const btCollisionAlgorithmConstructionInfo& ci,const btCollisionObjectWrapper* body0Wrap,const btCollisionObjectWrapper* body1Wrap);

	virtual ~btBox2dBox2dCollisionAlgorithm();

	virtual	void	getAllContactManifolds(btManifoldArray&	manifoldArray)
	{
		if (m_manifoldPtr && m_ownManifold)
		{
			manifoldArray.push_back(m_manifoldPtr);
		}
	}


	struct CreateFunc :public 	btCollisionAlgorithmCreateFunc
	{
		virtual	btCollisionAlgorithm* CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* body0Wrap,const btCollisionObjectWrapper* body1Wrap)
		{
			int bbsize = sizeof(btBox2dBox2dCollisionAlgorithm);
			void* ptr = ci.m_dispatcher1->allocateCollisionAlgorithm(bbsize);
			return new(ptr) btBox2dBox2dCollisionAlgorithm(0,ci,body0Wrap,body1Wrap);
		}
	};

};

#endif //BT_BOX_2D_BOX_2D__COLLISION_ALGORITHM_H

