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

#ifndef BT_SOFTBODY_RIGIDBODY_COLLISION_CONFIGURATION
#define BT_SOFTBODY_RIGIDBODY_COLLISION_CONFIGURATION

#include "BulletCollision/CollisionDispatch/btDefaultCollisionConfiguration.h"

class btVoronoiSimplexSolver;
class btGjkEpaPenetrationDepthSolver;


///btSoftBodyRigidBodyCollisionConfiguration add softbody interaction on top of btDefaultCollisionConfiguration
class	btSoftBodyRigidBodyCollisionConfiguration : public btDefaultCollisionConfiguration
{

	//default CreationFunctions, filling the m_doubleDispatch table
	btCollisionAlgorithmCreateFunc*	m_softSoftCreateFunc;
	btCollisionAlgorithmCreateFunc*	m_softRigidConvexCreateFunc;
	btCollisionAlgorithmCreateFunc*	m_swappedSoftRigidConvexCreateFunc;
	btCollisionAlgorithmCreateFunc*	m_softRigidConcaveCreateFunc;
	btCollisionAlgorithmCreateFunc*	m_swappedSoftRigidConcaveCreateFunc;

public:

	btSoftBodyRigidBodyCollisionConfiguration(const btDefaultCollisionConstructionInfo& constructionInfo = btDefaultCollisionConstructionInfo());

	virtual ~btSoftBodyRigidBodyCollisionConfiguration();

	///creation of soft-soft and soft-rigid, and otherwise fallback to base class implementation
	virtual btCollisionAlgorithmCreateFunc* getCollisionAlgorithmCreateFunc(int proxyType0,int proxyType1);

};

#endif //BT_SOFTBODY_RIGIDBODY_COLLISION_CONFIGURATION

