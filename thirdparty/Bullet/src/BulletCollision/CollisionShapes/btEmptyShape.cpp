/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2009 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "btEmptyShape.h"


#include "btCollisionShape.h"


btEmptyShape::btEmptyShape() : btConcaveShape ()
{
	m_shapeType = EMPTY_SHAPE_PROXYTYPE;
}


btEmptyShape::~btEmptyShape()
{
}


	///getAabb's default implementation is brute force, expected derived classes to implement a fast dedicated version
void btEmptyShape::getAabb(const btTransform& t,btVector3& aabbMin,btVector3& aabbMax) const
{
	btVector3 margin(getMargin(),getMargin(),getMargin());

	aabbMin = t.getOrigin() - margin;

	aabbMax = t.getOrigin() + margin;

}

void	btEmptyShape::calculateLocalInertia(btScalar ,btVector3& ) const
{
	btAssert(0);
}

	
	
