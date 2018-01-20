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


#include "btTriangleMesh.h"



btTriangleMesh::btTriangleMesh (bool use32bitIndices,bool use4componentVertices)
:m_use32bitIndices(use32bitIndices),
m_use4componentVertices(use4componentVertices),
m_weldingThreshold(0.0)
{
	btIndexedMesh meshIndex;
	meshIndex.m_numTriangles = 0;
	meshIndex.m_numVertices = 0;
	meshIndex.m_indexType = PHY_INTEGER;
	meshIndex.m_triangleIndexBase = 0;
	meshIndex.m_triangleIndexStride = 3*sizeof(int);
	meshIndex.m_vertexBase = 0;
	meshIndex.m_vertexStride = sizeof(btVector3);
	m_indexedMeshes.push_back(meshIndex);

	if (m_use32bitIndices)
	{
		m_indexedMeshes[0].m_numTriangles = m_32bitIndices.size()/3;
		m_indexedMeshes[0].m_triangleIndexBase = 0;
		m_indexedMeshes[0].m_indexType = PHY_INTEGER;
		m_indexedMeshes[0].m_triangleIndexStride = 3*sizeof(int);
	} else
	{
		m_indexedMeshes[0].m_numTriangles = m_16bitIndices.size()/3;
		m_indexedMeshes[0].m_triangleIndexBase = 0;
		m_indexedMeshes[0].m_indexType = PHY_SHORT;
		m_indexedMeshes[0].m_triangleIndexStride = 3*sizeof(short int);
	}

	if (m_use4componentVertices)
	{
		m_indexedMeshes[0].m_numVertices = m_4componentVertices.size();
		m_indexedMeshes[0].m_vertexBase = 0;
		m_indexedMeshes[0].m_vertexStride = sizeof(btVector3);
	} else
	{
		m_indexedMeshes[0].m_numVertices = m_3componentVertices.size()/3;
		m_indexedMeshes[0].m_vertexBase = 0;
		m_indexedMeshes[0].m_vertexStride = 3*sizeof(btScalar);
	}


}

void	btTriangleMesh::addIndex(int index)
{
	if (m_use32bitIndices)
	{
		m_32bitIndices.push_back(index);
		m_indexedMeshes[0].m_triangleIndexBase = (unsigned char*) &m_32bitIndices[0];
	} else
	{
		m_16bitIndices.push_back(index);
		m_indexedMeshes[0].m_triangleIndexBase = (unsigned char*) &m_16bitIndices[0];
	}
}

void	btTriangleMesh::addTriangleIndices(int index1, int index2, int index3 )
{
	m_indexedMeshes[0].m_numTriangles++;
	addIndex( index1 );
	addIndex( index2 );
	addIndex( index3 );
}

int	btTriangleMesh::findOrAddVertex(const btVector3& vertex, bool removeDuplicateVertices)
{
	//return index of new/existing vertex
	///@todo: could use acceleration structure for this
	if (m_use4componentVertices)
	{
		if (removeDuplicateVertices)
			{
			for (int i=0;i< m_4componentVertices.size();i++)
			{
				if ((m_4componentVertices[i]-vertex).length2() <= m_weldingThreshold)
				{
					return i;
				}
			}
		}
		m_indexedMeshes[0].m_numVertices++;
		m_4componentVertices.push_back(vertex);
		m_indexedMeshes[0].m_vertexBase = (unsigned char*)&m_4componentVertices[0];

		return m_4componentVertices.size()-1;
		
	} else
	{
		
		if (removeDuplicateVertices)
		{
			for (int i=0;i< m_3componentVertices.size();i+=3)
			{
				btVector3 vtx(m_3componentVertices[i],m_3componentVertices[i+1],m_3componentVertices[i+2]);
				if ((vtx-vertex).length2() <= m_weldingThreshold)
				{
					return i/3;
				}
			}
		}
		m_3componentVertices.push_back(vertex.getX());
		m_3componentVertices.push_back(vertex.getY());
		m_3componentVertices.push_back(vertex.getZ());
		m_indexedMeshes[0].m_numVertices++;
		m_indexedMeshes[0].m_vertexBase = (unsigned char*)&m_3componentVertices[0];
		return (m_3componentVertices.size()/3)-1;
	}

}
		
void	btTriangleMesh::addTriangle(const btVector3& vertex0,const btVector3& vertex1,const btVector3& vertex2,bool removeDuplicateVertices)
{
	m_indexedMeshes[0].m_numTriangles++;
	addIndex(findOrAddVertex(vertex0,removeDuplicateVertices));
	addIndex(findOrAddVertex(vertex1,removeDuplicateVertices));
	addIndex(findOrAddVertex(vertex2,removeDuplicateVertices));
}

int btTriangleMesh::getNumTriangles() const
{
	if (m_use32bitIndices)
	{
		return m_32bitIndices.size() / 3;
	}
	return m_16bitIndices.size() / 3;
}

void btTriangleMesh::preallocateVertices(int numverts)
{
	if (m_use4componentVertices)
	{
		m_4componentVertices.reserve(numverts);
	} else
	{
		m_3componentVertices.reserve(numverts);
	}
}

void btTriangleMesh::preallocateIndices(int numindices)
{
	if (m_use32bitIndices)
	{
		m_32bitIndices.reserve(numindices);
	} else
	{
		m_16bitIndices.reserve(numindices);
	}
}
