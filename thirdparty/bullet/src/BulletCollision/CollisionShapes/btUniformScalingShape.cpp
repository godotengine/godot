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

#include "btUniformScalingShape.h"

btUniformScalingShape::btUniformScalingShape(	btConvexShape* convexChildShape,btScalar uniformScalingFactor):
btConvexShape (), m_childConvexShape(convexChildShape),
m_uniformScalingFactor(uniformScalingFactor)
{
	m_shapeType = UNIFORM_SCALING_SHAPE_PROXYTYPE;
}
	
btUniformScalingShape::~btUniformScalingShape()
{
}
	

btVector3	btUniformScalingShape::localGetSupportingVertexWithoutMargin(const btVector3& vec)const
{
	btVector3 tmpVertex;
	tmpVertex = m_childConvexShape->localGetSupportingVertexWithoutMargin(vec);
	return tmpVertex*m_uniformScalingFactor;
}

void	btUniformScalingShape::batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors,btVector3* supportVerticesOut,int numVectors) const
{
	m_childConvexShape->batchedUnitVectorGetSupportingVertexWithoutMargin(vectors,supportVerticesOut,numVectors);
	int i;
	for (i=0;i<numVectors;i++)
	{
		supportVerticesOut[i] = supportVerticesOut[i] * m_uniformScalingFactor;
	}
}


btVector3	btUniformScalingShape::localGetSupportingVertex(const btVector3& vec)const
{
	btVector3 tmpVertex;
	tmpVertex = m_childConvexShape->localGetSupportingVertex(vec);
	return tmpVertex*m_uniformScalingFactor;
}


void	btUniformScalingShape::calculateLocalInertia(btScalar mass,btVector3& inertia) const
{

	///this linear upscaling is not realistic, but we don't deal with large mass ratios...
	btVector3 tmpInertia;
	m_childConvexShape->calculateLocalInertia(mass,tmpInertia);
	inertia = tmpInertia * m_uniformScalingFactor;
}


	///getAabb's default implementation is brute force, expected derived classes to implement a fast dedicated version
void btUniformScalingShape::getAabb(const btTransform& trans,btVector3& aabbMin,btVector3& aabbMax) const
{
	getAabbSlow(trans,aabbMin,aabbMax);

}

void btUniformScalingShape::getAabbSlow(const btTransform& t,btVector3& aabbMin,btVector3& aabbMax) const
{
#if 1
	btVector3 _directions[] =
	{
		btVector3( 1.,  0.,  0.),
		btVector3( 0.,  1.,  0.),
		btVector3( 0.,  0.,  1.),
		btVector3( -1., 0.,  0.),
		btVector3( 0., -1.,  0.),
		btVector3( 0.,  0., -1.)
	};
	
	btVector3 _supporting[] =
	{
		btVector3( 0., 0., 0.),
		btVector3( 0., 0., 0.),
		btVector3( 0., 0., 0.),
		btVector3( 0., 0., 0.),
		btVector3( 0., 0., 0.),
		btVector3( 0., 0., 0.)
	};

	for (int i=0;i<6;i++)
	{
		_directions[i] = _directions[i]*t.getBasis();
	}
	
	batchedUnitVectorGetSupportingVertexWithoutMargin(_directions, _supporting, 6);
	
	btVector3 aabbMin1(0,0,0),aabbMax1(0,0,0);

	for ( int i = 0; i < 3; ++i )
	{
		aabbMax1[i] = t(_supporting[i])[i];
		aabbMin1[i] = t(_supporting[i + 3])[i];
	}
	btVector3 marginVec(getMargin(),getMargin(),getMargin());
	aabbMin = aabbMin1-marginVec;
	aabbMax = aabbMax1+marginVec;
	
#else

	btScalar margin = getMargin();
	for (int i=0;i<3;i++)
	{
		btVector3 vec(btScalar(0.),btScalar(0.),btScalar(0.));
		vec[i] = btScalar(1.);
		btVector3 sv = localGetSupportingVertex(vec*t.getBasis());
		btVector3 tmp = t(sv);
		aabbMax[i] = tmp[i]+margin;
		vec[i] = btScalar(-1.);
		sv = localGetSupportingVertex(vec*t.getBasis());
		tmp = t(sv);
		aabbMin[i] = tmp[i]-margin;
	}

#endif
}

void	btUniformScalingShape::setLocalScaling(const btVector3& scaling) 
{
	m_childConvexShape->setLocalScaling(scaling);
}

const btVector3& btUniformScalingShape::getLocalScaling() const
{
	return m_childConvexShape->getLocalScaling();
}

void	btUniformScalingShape::setMargin(btScalar margin)
{
	m_childConvexShape->setMargin(margin);
}
btScalar	btUniformScalingShape::getMargin() const
{
	return m_childConvexShape->getMargin() * m_uniformScalingFactor;
}

int		btUniformScalingShape::getNumPreferredPenetrationDirections() const
{
	return m_childConvexShape->getNumPreferredPenetrationDirections();
}
	
void	btUniformScalingShape::getPreferredPenetrationDirection(int index, btVector3& penetrationVector) const
{
	m_childConvexShape->getPreferredPenetrationDirection(index,penetrationVector);
}
