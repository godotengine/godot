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

#ifndef BT_TRIANGLE_MESH_H
#define BT_TRIANGLE_MESH_H

#include "btTriangleIndexVertexArray.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btAlignedObjectArray.h"

///The btTriangleMesh class is a convenience class derived from btTriangleIndexVertexArray, that provides storage for a concave triangle mesh. It can be used as data for the btBvhTriangleMeshShape.
///It allows either 32bit or 16bit indices, and 4 (x-y-z-w) or 3 (x-y-z) component vertices.
///If you want to share triangle/index data between graphics mesh and collision mesh (btBvhTriangleMeshShape), you can directly use btTriangleIndexVertexArray or derive your own class from btStridingMeshInterface.
///Performance of btTriangleMesh and btTriangleIndexVertexArray used in a btBvhTriangleMeshShape is the same.
class btTriangleMesh : public btTriangleIndexVertexArray
{
	btAlignedObjectArray<btVector3>	m_4componentVertices;
	btAlignedObjectArray<btScalar>	m_3componentVertices;

	btAlignedObjectArray<unsigned int>		m_32bitIndices;
	btAlignedObjectArray<unsigned short int>		m_16bitIndices;
	bool	m_use32bitIndices;
	bool	m_use4componentVertices;
	

	public:
		btScalar	m_weldingThreshold;

		btTriangleMesh (bool use32bitIndices=true,bool use4componentVertices=true);

		bool	getUse32bitIndices() const
		{
			return m_use32bitIndices;
		}

		bool	getUse4componentVertices() const
		{
			return m_use4componentVertices;
		}
		///By default addTriangle won't search for duplicate vertices, because the search is very slow for large triangle meshes.
		///In general it is better to directly use btTriangleIndexVertexArray instead.
		void	addTriangle(const btVector3& vertex0,const btVector3& vertex1,const btVector3& vertex2, bool removeDuplicateVertices=false);

		///Add a triangle using its indices. Make sure the indices are pointing within the vertices array, so add the vertices first (and to be sure, avoid removal of duplicate vertices)	
		void	addTriangleIndices(int index1, int index2, int index3 );
	
		int getNumTriangles() const;

		virtual void	preallocateVertices(int numverts);
		virtual void	preallocateIndices(int numindices);

		///findOrAddVertex is an internal method, use addTriangle instead
		int		findOrAddVertex(const btVector3& vertex, bool removeDuplicateVertices);
		///addIndex is an internal method, use addTriangle instead
		void	addIndex(int index);
		
};

#endif //BT_TRIANGLE_MESH_H

