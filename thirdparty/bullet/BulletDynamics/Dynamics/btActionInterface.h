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

#ifndef _BT_ACTION_INTERFACE_H
#define _BT_ACTION_INTERFACE_H

class btIDebugDraw;
class btCollisionWorld;

#include "LinearMath/btScalar.h"
#include "btRigidBody.h"

///Basic interface to allow actions such as vehicles and characters to be updated inside a btDynamicsWorld
class btActionInterface
{
protected:

	static btRigidBody& getFixedBody();
	
	
public:

	virtual ~btActionInterface()
	{
	}

	virtual void updateAction( btCollisionWorld* collisionWorld, btScalar deltaTimeStep)=0;

	virtual void debugDraw(btIDebugDraw* debugDrawer) = 0;

};

#endif //_BT_ACTION_INTERFACE_H

