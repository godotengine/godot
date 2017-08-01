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

#ifndef BT_EMPTY_SHAPE_H
#define BT_EMPTY_SHAPE_H

#include "btConcaveShape.h"

#include "LinearMath/btVector3.h"
#include "LinearMath/btTransform.h"
#include "LinearMath/btMatrix3x3.h"
#include "btCollisionMargin.h"




/// The btEmptyShape is a collision shape without actual collision detection shape, so most users should ignore this class.
/// It can be replaced by another shape during runtime, but the inertia tensor should be recomputed.
ATTRIBUTE_ALIGNED16(class) btEmptyShape	: public btConcaveShape
{
public:
	BT_DECLARE_ALIGNED_ALLOCATOR();
	
	btEmptyShape();

	virtual ~btEmptyShape();


	///getAabb's default implementation is brute force, expected derived classes to implement a fast dedicated version
	void getAabb(const btTransform& t,btVector3& aabbMin,btVector3& aabbMax) const;


	virtual void	setLocalScaling(const btVector3& scaling)
	{
		m_localScaling = scaling;
	}
	virtual const btVector3& getLocalScaling() const 
	{
		return m_localScaling;
	}

	virtual void	calculateLocalInertia(btScalar mass,btVector3& inertia) const;
	
	virtual const char*	getName()const
	{
		return "Empty";
	}

	virtual void processAllTriangles(btTriangleCallback* ,const btVector3& ,const btVector3& ) const
	{
	}

protected:
	btVector3	m_localScaling;

};



#endif //BT_EMPTY_SHAPE_H
