/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2014 Erwin Coumans http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef GJK_COLLISION_DESCRIPTION_H
#define GJK_COLLISION_DESCRIPTION_H

#include "LinearMath/btVector3.h"

struct btGjkCollisionDescription
{
	btVector3 m_firstDir;
	int m_maxGjkIterations;
	btScalar m_maximumDistanceSquared;
	btScalar m_gjkRelError2;
	btGjkCollisionDescription()
		: m_firstDir(0, 1, 0),
		  m_maxGjkIterations(1000),
		  m_maximumDistanceSquared(1e30f),
		  m_gjkRelError2(1.0e-6)
	{
	}
	virtual ~btGjkCollisionDescription()
	{
	}
};

#endif  //GJK_COLLISION_DESCRIPTION_H
