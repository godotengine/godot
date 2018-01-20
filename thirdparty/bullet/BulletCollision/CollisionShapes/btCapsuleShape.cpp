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


#include "btCapsuleShape.h"

#include "LinearMath/btQuaternion.h"

btCapsuleShape::btCapsuleShape(btScalar radius, btScalar height) : btConvexInternalShape ()
{
	m_collisionMargin = radius;
	m_shapeType = CAPSULE_SHAPE_PROXYTYPE;
	m_upAxis = 1;
	m_implicitShapeDimensions.setValue(radius,0.5f*height,radius);
}

 
 btVector3	btCapsuleShape::localGetSupportingVertexWithoutMargin(const btVector3& vec0)const
{

	btVector3 supVec(0,0,0);

	btScalar maxDot(btScalar(-BT_LARGE_FLOAT));

	btVector3 vec = vec0;
	btScalar lenSqr = vec.length2();
	if (lenSqr < btScalar(0.0001))
	{
		vec.setValue(1,0,0);
	} else
	{
		btScalar rlen = btScalar(1.) / btSqrt(lenSqr );
		vec *= rlen;
	}

	btVector3 vtx;
	btScalar newDot;
	
	

	{
		btVector3 pos(0,0,0);
		pos[getUpAxis()] = getHalfHeight();

		vtx = pos;
		newDot = vec.dot(vtx);
		if (newDot > maxDot)
		{
			maxDot = newDot;
			supVec = vtx;
		}
	}
	{
		btVector3 pos(0,0,0);
		pos[getUpAxis()] = -getHalfHeight();

		vtx = pos;
		newDot = vec.dot(vtx);
		if (newDot > maxDot)
		{
			maxDot = newDot;
			supVec = vtx;
		}
	}

	return supVec;

}

 void	btCapsuleShape::batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors,btVector3* supportVerticesOut,int numVectors) const
{

	
	
	for (int j=0;j<numVectors;j++)
	{
		btScalar maxDot(btScalar(-BT_LARGE_FLOAT));
		const btVector3& vec = vectors[j];

		btVector3 vtx;
		btScalar newDot;
		{
			btVector3 pos(0,0,0);
			pos[getUpAxis()] = getHalfHeight();
			vtx = pos;
			newDot = vec.dot(vtx);
			if (newDot > maxDot)
			{
				maxDot = newDot;
				supportVerticesOut[j] = vtx;
			}
		}
		{
			btVector3 pos(0,0,0);
			pos[getUpAxis()] = -getHalfHeight();
			vtx = pos;
			newDot = vec.dot(vtx);
			if (newDot > maxDot)
			{
				maxDot = newDot;
				supportVerticesOut[j] = vtx;
			}
		}
		
	}
}


void	btCapsuleShape::calculateLocalInertia(btScalar mass,btVector3& inertia) const
{
	//as an approximation, take the inertia of the box that bounds the spheres

	btTransform ident;
	ident.setIdentity();

	
	btScalar radius = getRadius();

	btVector3 halfExtents(radius,radius,radius);
	halfExtents[getUpAxis()]+=getHalfHeight();

	btScalar lx=btScalar(2.)*(halfExtents[0]);
	btScalar ly=btScalar(2.)*(halfExtents[1]);
	btScalar lz=btScalar(2.)*(halfExtents[2]);
	const btScalar x2 = lx*lx;
	const btScalar y2 = ly*ly;
	const btScalar z2 = lz*lz;
	const btScalar scaledmass = mass * btScalar(.08333333);

	inertia[0] = scaledmass * (y2+z2);
	inertia[1] = scaledmass * (x2+z2);
	inertia[2] = scaledmass * (x2+y2);

}

btCapsuleShapeX::btCapsuleShapeX(btScalar radius,btScalar height)
{
	m_collisionMargin = radius;
	m_upAxis = 0;
	m_implicitShapeDimensions.setValue(0.5f*height, radius,radius);
}






btCapsuleShapeZ::btCapsuleShapeZ(btScalar radius,btScalar height)
{
	m_collisionMargin = radius;
	m_upAxis = 2;
	m_implicitShapeDimensions.setValue(radius,radius,0.5f*height);
}




