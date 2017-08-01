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

#ifndef BT_POINT_COLLECTOR_H
#define BT_POINT_COLLECTOR_H

#include "btDiscreteCollisionDetectorInterface.h"



struct btPointCollector : public btDiscreteCollisionDetectorInterface::Result
{
	
	
	btVector3 m_normalOnBInWorld;
	btVector3 m_pointInWorld;
	btScalar	m_distance;//negative means penetration

	bool	m_hasResult;

	btPointCollector () 
		: m_distance(btScalar(BT_LARGE_FLOAT)),m_hasResult(false)
	{
	}

	virtual void setShapeIdentifiersA(int partId0,int index0)
	{
		(void)partId0;
		(void)index0;
			
	}
	virtual void setShapeIdentifiersB(int partId1,int index1)
	{
		(void)partId1;
		(void)index1;
	}

	virtual void addContactPoint(const btVector3& normalOnBInWorld,const btVector3& pointInWorld,btScalar depth)
	{
		if (depth< m_distance)
		{
			m_hasResult = true;
			m_normalOnBInWorld = normalOnBInWorld;
			m_pointInWorld = pointInWorld;
			//negative means penetration
			m_distance = depth;
		}
	}
};

#endif //BT_POINT_COLLECTOR_H

