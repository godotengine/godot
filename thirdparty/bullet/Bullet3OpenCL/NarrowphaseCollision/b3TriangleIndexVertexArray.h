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

#ifndef B3_TRIANGLE_INDEX_VERTEX_ARRAY_H
#define B3_TRIANGLE_INDEX_VERTEX_ARRAY_H

#include "b3StridingMeshInterface.h"
#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3Common/b3Scalar.h"

///The b3IndexedMesh indexes a single vertex and index array. Multiple b3IndexedMesh objects can be passed into a b3TriangleIndexVertexArray using addIndexedMesh.
///Instead of the number of indices, we pass the number of triangles.
B3_ATTRIBUTE_ALIGNED16(struct)
b3IndexedMesh
{
	B3_DECLARE_ALIGNED_ALLOCATOR();

	int m_numTriangles;
	const unsigned char* m_triangleIndexBase;
	// Size in byte of the indices for one triangle (3*sizeof(index_type) if the indices are tightly packed)
	int m_triangleIndexStride;
	int m_numVertices;
	const unsigned char* m_vertexBase;
	// Size of a vertex, in bytes
	int m_vertexStride;

	// The index type is set when adding an indexed mesh to the
	// b3TriangleIndexVertexArray, do not set it manually
	PHY_ScalarType m_indexType;

	// The vertex type has a default type similar to Bullet's precision mode (float or double)
	// but can be set manually if you for example run Bullet with double precision but have
	// mesh data in single precision..
	PHY_ScalarType m_vertexType;

	b3IndexedMesh()
		: m_indexType(PHY_INTEGER),
#ifdef B3_USE_DOUBLE_PRECISION
		  m_vertexType(PHY_DOUBLE)
#else   // B3_USE_DOUBLE_PRECISION
		  m_vertexType(PHY_FLOAT)
#endif  // B3_USE_DOUBLE_PRECISION
	{
	}
};

typedef b3AlignedObjectArray<b3IndexedMesh> IndexedMeshArray;

///The b3TriangleIndexVertexArray allows to access multiple triangle meshes, by indexing into existing triangle/index arrays.
///Additional meshes can be added using addIndexedMesh
///No duplcate is made of the vertex/index data, it only indexes into external vertex/index arrays.
///So keep those arrays around during the lifetime of this b3TriangleIndexVertexArray.
B3_ATTRIBUTE_ALIGNED16(class)
b3TriangleIndexVertexArray : public b3StridingMeshInterface
{
protected:
	IndexedMeshArray m_indexedMeshes;
	int m_pad[2];
	mutable int m_hasAabb;  // using int instead of bool to maintain alignment
	mutable b3Vector3 m_aabbMin;
	mutable b3Vector3 m_aabbMax;

public:
	B3_DECLARE_ALIGNED_ALLOCATOR();

	b3TriangleIndexVertexArray() : m_hasAabb(0)
	{
	}

	virtual ~b3TriangleIndexVertexArray();

	//just to be backwards compatible
	b3TriangleIndexVertexArray(int numTriangles, int* triangleIndexBase, int triangleIndexStride, int numVertices, b3Scalar* vertexBase, int vertexStride);

	void addIndexedMesh(const b3IndexedMesh& mesh, PHY_ScalarType indexType = PHY_INTEGER)
	{
		m_indexedMeshes.push_back(mesh);
		m_indexedMeshes[m_indexedMeshes.size() - 1].m_indexType = indexType;
	}

	virtual void getLockedVertexIndexBase(unsigned char** vertexbase, int& numverts, PHY_ScalarType& type, int& vertexStride, unsigned char** indexbase, int& indexstride, int& numfaces, PHY_ScalarType& indicestype, int subpart = 0);

	virtual void getLockedReadOnlyVertexIndexBase(const unsigned char** vertexbase, int& numverts, PHY_ScalarType& type, int& vertexStride, const unsigned char** indexbase, int& indexstride, int& numfaces, PHY_ScalarType& indicestype, int subpart = 0) const;

	/// unLockVertexBase finishes the access to a subpart of the triangle mesh
	/// make a call to unLockVertexBase when the read and write access (using getLockedVertexIndexBase) is finished
	virtual void unLockVertexBase(int subpart) { (void)subpart; }

	virtual void unLockReadOnlyVertexBase(int subpart) const { (void)subpart; }

	/// getNumSubParts returns the number of separate subparts
	/// each subpart has a continuous array of vertices and indices
	virtual int getNumSubParts() const
	{
		return (int)m_indexedMeshes.size();
	}

	IndexedMeshArray& getIndexedMeshArray()
	{
		return m_indexedMeshes;
	}

	const IndexedMeshArray& getIndexedMeshArray() const
	{
		return m_indexedMeshes;
	}

	virtual void preallocateVertices(int numverts) { (void)numverts; }
	virtual void preallocateIndices(int numindices) { (void)numindices; }

	virtual bool hasPremadeAabb() const;
	virtual void setPremadeAabb(const b3Vector3& aabbMin, const b3Vector3& aabbMax) const;
	virtual void getPremadeAabb(b3Vector3 * aabbMin, b3Vector3 * aabbMax) const;
};

#endif  //B3_TRIANGLE_INDEX_VERTEX_ARRAY_H
