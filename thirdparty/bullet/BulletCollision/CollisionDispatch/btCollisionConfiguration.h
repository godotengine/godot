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

#ifndef BT_COLLISION_CONFIGURATION
#define BT_COLLISION_CONFIGURATION

struct btCollisionAlgorithmCreateFunc;

class btPoolAllocator;

///btCollisionConfiguration allows to configure Bullet collision detection
///stack allocator size, default collision algorithms and persistent manifold pool size
///@todo: describe the meaning
class btCollisionConfiguration
{
public:
	virtual ~btCollisionConfiguration()
	{
	}

	///memory pools
	virtual btPoolAllocator* getPersistentManifoldPool() = 0;

	virtual btPoolAllocator* getCollisionAlgorithmPool() = 0;

	virtual btCollisionAlgorithmCreateFunc* getCollisionAlgorithmCreateFunc(int proxyType0, int proxyType1) = 0;

	virtual btCollisionAlgorithmCreateFunc* getClosestPointsAlgorithmCreateFunc(int proxyType0, int proxyType1) = 0;
};

#endif  //BT_COLLISION_CONFIGURATION
