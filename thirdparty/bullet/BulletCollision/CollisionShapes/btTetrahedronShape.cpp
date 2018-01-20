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

#include "btTetrahedronShape.h"
#include "LinearMath/btMatrix3x3.h"

btBU_Simplex1to4::btBU_Simplex1to4() : btPolyhedralConvexAabbCachingShape (),
m_numVertices(0)
{
	m_shapeType = TETRAHEDRAL_SHAPE_PROXYTYPE;
}

btBU_Simplex1to4::btBU_Simplex1to4(const btVector3& pt0) : btPolyhedralConvexAabbCachingShape (),
m_numVertices(0)
{
	m_shapeType = TETRAHEDRAL_SHAPE_PROXYTYPE;
	addVertex(pt0);
}

btBU_Simplex1to4::btBU_Simplex1to4(const btVector3& pt0,const btVector3& pt1) : btPolyhedralConvexAabbCachingShape (),
m_numVertices(0)
{
	m_shapeType = TETRAHEDRAL_SHAPE_PROXYTYPE;
	addVertex(pt0);
	addVertex(pt1);
}

btBU_Simplex1to4::btBU_Simplex1to4(const btVector3& pt0,const btVector3& pt1,const btVector3& pt2) : btPolyhedralConvexAabbCachingShape (),
m_numVertices(0)
{
	m_shapeType = TETRAHEDRAL_SHAPE_PROXYTYPE;
	addVertex(pt0);
	addVertex(pt1);
	addVertex(pt2);
}

btBU_Simplex1to4::btBU_Simplex1to4(const btVector3& pt0,const btVector3& pt1,const btVector3& pt2,const btVector3& pt3) : btPolyhedralConvexAabbCachingShape (),
m_numVertices(0)
{
	m_shapeType = TETRAHEDRAL_SHAPE_PROXYTYPE;
	addVertex(pt0);
	addVertex(pt1);
	addVertex(pt2);
	addVertex(pt3);
}


void btBU_Simplex1to4::getAabb(const btTransform& t,btVector3& aabbMin,btVector3& aabbMax) const
{
#if 1
	btPolyhedralConvexAabbCachingShape::getAabb(t,aabbMin,aabbMax);
#else
	aabbMin.setValue(BT_LARGE_FLOAT,BT_LARGE_FLOAT,BT_LARGE_FLOAT);
	aabbMax.setValue(-BT_LARGE_FLOAT,-BT_LARGE_FLOAT,-BT_LARGE_FLOAT);

	//just transform the vertices in worldspace, and take their AABB
	for (int i=0;i<m_numVertices;i++)
	{
		btVector3 worldVertex = t(m_vertices[i]);
		aabbMin.setMin(worldVertex);
		aabbMax.setMax(worldVertex);
	}
#endif
}





void btBU_Simplex1to4::addVertex(const btVector3& pt)
{
	m_vertices[m_numVertices++] = pt;
	recalcLocalAabb();
}


int	btBU_Simplex1to4::getNumVertices() const
{
	return m_numVertices;
}

int btBU_Simplex1to4::getNumEdges() const
{
	//euler formula, F-E+V = 2, so E = F+V-2

	switch (m_numVertices)
	{
	case 0:
		return 0;
	case 1: return 0;
	case 2: return 1;
	case 3: return 3;
	case 4: return 6;


	}

	return 0;
}

void btBU_Simplex1to4::getEdge(int i,btVector3& pa,btVector3& pb) const
{
	
    switch (m_numVertices)
	{

	case 2: 
		pa = m_vertices[0];
		pb = m_vertices[1];
		break;
	case 3:  
		switch (i)
		{
		case 0:
			pa = m_vertices[0];
			pb = m_vertices[1];
			break;
		case 1:
			pa = m_vertices[1];
			pb = m_vertices[2];
			break;
		case 2:
			pa = m_vertices[2];
			pb = m_vertices[0];
			break;

		}
		break;
	case 4: 
		switch (i)
		{
		case 0:
			pa = m_vertices[0];
			pb = m_vertices[1];
			break;
		case 1:
			pa = m_vertices[1];
			pb = m_vertices[2];
			break;
		case 2:
			pa = m_vertices[2];
			pb = m_vertices[0];
			break;
		case 3:
			pa = m_vertices[0];
			pb = m_vertices[3];
			break;
		case 4:
			pa = m_vertices[1];
			pb = m_vertices[3];
			break;
		case 5:
			pa = m_vertices[2];
			pb = m_vertices[3];
			break;
		}

	}




}

void btBU_Simplex1to4::getVertex(int i,btVector3& vtx) const
{
	vtx = m_vertices[i];
}

int	btBU_Simplex1to4::getNumPlanes() const
{
	switch (m_numVertices)
	{
	case 0:
			return 0;
	case 1:
			return 0;
	case 2:
			return 0;
	case 3:
			return 2;
	case 4:
			return 4;
	default:
		{
		}
	}
	return 0;
}


void btBU_Simplex1to4::getPlane(btVector3&, btVector3& ,int ) const
{
	
}

int btBU_Simplex1to4::getIndex(int ) const
{
	return 0;
}

bool btBU_Simplex1to4::isInside(const btVector3& ,btScalar ) const
{
	return false;
}

