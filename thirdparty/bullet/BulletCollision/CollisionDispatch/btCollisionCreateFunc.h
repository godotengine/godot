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

#ifndef BT_COLLISION_CREATE_FUNC
#define BT_COLLISION_CREATE_FUNC

#include "LinearMath/btAlignedObjectArray.h"
class btCollisionAlgorithm;
class btCollisionObject;
struct btCollisionObjectWrapper;
struct btCollisionAlgorithmConstructionInfo;

///Used by the btCollisionDispatcher to register and create instances for btCollisionAlgorithm
struct btCollisionAlgorithmCreateFunc
{
	bool m_swapped;

	btCollisionAlgorithmCreateFunc()
		: m_swapped(false)
	{
	}
	virtual ~btCollisionAlgorithmCreateFunc(){};

	virtual btCollisionAlgorithm* CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo&, const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap)
	{
		(void)body0Wrap;
		(void)body1Wrap;
		return 0;
	}
};
#endif  //BT_COLLISION_CREATE_FUNC
