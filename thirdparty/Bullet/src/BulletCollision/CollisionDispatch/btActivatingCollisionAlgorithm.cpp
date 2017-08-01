/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2008 Erwin Coumans  http://bulletphysics.com

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "btActivatingCollisionAlgorithm.h"
#include "btCollisionDispatcher.h"
#include "btCollisionObject.h"

btActivatingCollisionAlgorithm::btActivatingCollisionAlgorithm (const btCollisionAlgorithmConstructionInfo& ci)
:btCollisionAlgorithm(ci)
//,
//m_colObj0(0),
//m_colObj1(0)
{
}
btActivatingCollisionAlgorithm::btActivatingCollisionAlgorithm (const btCollisionAlgorithmConstructionInfo& ci, const btCollisionObjectWrapper* ,const btCollisionObjectWrapper* )
:btCollisionAlgorithm(ci)
//,
//m_colObj0(0),
//m_colObj1(0)
{
//	if (ci.m_dispatcher1->needsCollision(colObj0,colObj1))
//	{
//		m_colObj0 = colObj0;
//		m_colObj1 = colObj1;
//		
//		m_colObj0->activate();
//		m_colObj1->activate();
//	}
}

btActivatingCollisionAlgorithm::~btActivatingCollisionAlgorithm()
{
//		m_colObj0->activate();
//		m_colObj1->activate();
}
