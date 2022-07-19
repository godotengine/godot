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

/// This file was created by Alex Silverman

#ifndef BT_BVH_TRIANGLE_MATERIAL_MESH_SHAPE_H
#define BT_BVH_TRIANGLE_MATERIAL_MESH_SHAPE_H

#include "btBvhTriangleMeshShape.h"
#include "btMaterial.h"

///The BvhTriangleMaterialMeshShape extends the btBvhTriangleMeshShape. Its main contribution is the interface into a material array, which allows per-triangle friction and restitution.
ATTRIBUTE_ALIGNED16(class)
btMultimaterialTriangleMeshShape : public btBvhTriangleMeshShape
{
	btAlignedObjectArray<btMaterial *> m_materialList;

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	btMultimaterialTriangleMeshShape(btStridingMeshInterface * meshInterface, bool useQuantizedAabbCompression, bool buildBvh = true) : btBvhTriangleMeshShape(meshInterface, useQuantizedAabbCompression, buildBvh)
	{
		m_shapeType = MULTIMATERIAL_TRIANGLE_MESH_PROXYTYPE;

		const unsigned char *vertexbase;
		int numverts;
		PHY_ScalarType type;
		int stride;
		const unsigned char *indexbase;
		int indexstride;
		int numfaces;
		PHY_ScalarType indicestype;

		//m_materialLookup = (int**)(btAlignedAlloc(sizeof(int*) * meshInterface->getNumSubParts(), 16));

		for (int i = 0; i < meshInterface->getNumSubParts(); i++)
		{
			m_meshInterface->getLockedReadOnlyVertexIndexBase(
				&vertexbase,
				numverts,
				type,
				stride,
				&indexbase,
				indexstride,
				numfaces,
				indicestype,
				i);
			//m_materialLookup[i] = (int*)(btAlignedAlloc(sizeof(int) * numfaces, 16));
		}
	}

	///optionally pass in a larger bvh aabb, used for quantization. This allows for deformations within this aabb
	btMultimaterialTriangleMeshShape(btStridingMeshInterface * meshInterface, bool useQuantizedAabbCompression, const btVector3 &bvhAabbMin, const btVector3 &bvhAabbMax, bool buildBvh = true) : btBvhTriangleMeshShape(meshInterface, useQuantizedAabbCompression, bvhAabbMin, bvhAabbMax, buildBvh)
	{
		m_shapeType = MULTIMATERIAL_TRIANGLE_MESH_PROXYTYPE;

		const unsigned char *vertexbase;
		int numverts;
		PHY_ScalarType type;
		int stride;
		const unsigned char *indexbase;
		int indexstride;
		int numfaces;
		PHY_ScalarType indicestype;

		//m_materialLookup = (int**)(btAlignedAlloc(sizeof(int*) * meshInterface->getNumSubParts(), 16));

		for (int i = 0; i < meshInterface->getNumSubParts(); i++)
		{
			m_meshInterface->getLockedReadOnlyVertexIndexBase(
				&vertexbase,
				numverts,
				type,
				stride,
				&indexbase,
				indexstride,
				numfaces,
				indicestype,
				i);
			//m_materialLookup[i] = (int*)(btAlignedAlloc(sizeof(int) * numfaces * 2, 16));
		}
	}

	virtual ~btMultimaterialTriangleMeshShape()
	{
		/*
        for(int i = 0; i < m_meshInterface->getNumSubParts(); i++)
        {
            btAlignedFree(m_materialValues[i]);
            m_materialLookup[i] = NULL;
        }
        btAlignedFree(m_materialValues);
        m_materialLookup = NULL;
*/
	}
	//debugging
	virtual const char *getName() const { return "MULTIMATERIALTRIANGLEMESH"; }

	///Obtains the material for a specific triangle
	const btMaterial *getMaterialProperties(int partID, int triIndex);
};

#endif  //BT_BVH_TRIANGLE_MATERIAL_MESH_SHAPE_H
