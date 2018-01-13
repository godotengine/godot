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

///This file was created by Alex Silverman

#ifndef BT_MULTIMATERIAL_TRIANGLE_INDEX_VERTEX_ARRAY_H
#define BT_MULTIMATERIAL_TRIANGLE_INDEX_VERTEX_ARRAY_H

#include "btTriangleIndexVertexArray.h"


ATTRIBUTE_ALIGNED16( struct)	btMaterialProperties
{
    ///m_materialBase ==========> 2 btScalar values make up one material, friction then restitution
    int m_numMaterials;
    const unsigned char * m_materialBase;
    int m_materialStride;
    PHY_ScalarType m_materialType;
    ///m_numTriangles <=========== This exists in the btIndexedMesh object for the same subpart, but since we're
    ///                           padding the structure, it can be reproduced at no real cost
    ///m_triangleMaterials =====> 1 integer value makes up one entry
    ///                           eg: m_triangleMaterials[1] = 5; // This will set triangle 2 to use material 5
    int m_numTriangles; 
    const unsigned char * m_triangleMaterialsBase;
    int m_triangleMaterialStride;
    ///m_triangleType <========== Automatically set in addMaterialProperties
    PHY_ScalarType m_triangleType;
};

typedef btAlignedObjectArray<btMaterialProperties>	MaterialArray;

///Teh btTriangleIndexVertexMaterialArray is built on TriangleIndexVertexArray
///The addition of a material array allows for the utilization of the partID and
///triangleIndex that are returned in the ContactAddedCallback.  As with
///TriangleIndexVertexArray, no duplicate is made of the material data, so it
///is the users responsibility to maintain the array during the lifetime of the
///TriangleIndexVertexMaterialArray.
ATTRIBUTE_ALIGNED16(class) btTriangleIndexVertexMaterialArray : public btTriangleIndexVertexArray
{
protected:
    MaterialArray       m_materials;
		
public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

    btTriangleIndexVertexMaterialArray()
	{
	}

    btTriangleIndexVertexMaterialArray(int numTriangles,int* triangleIndexBase,int triangleIndexStride,
        int numVertices,btScalar* vertexBase,int vertexStride,
        int numMaterials, unsigned char* materialBase, int materialStride,
        int* triangleMaterialsBase, int materialIndexStride);

    virtual ~btTriangleIndexVertexMaterialArray() {}

    void	addMaterialProperties(const btMaterialProperties& mat, PHY_ScalarType triangleType = PHY_INTEGER)
    {
        m_materials.push_back(mat);
        m_materials[m_materials.size()-1].m_triangleType = triangleType;
    }

    virtual void getLockedMaterialBase(unsigned char **materialBase, int& numMaterials, PHY_ScalarType& materialType, int& materialStride,
        unsigned char ** triangleMaterialBase, int& numTriangles, int& triangleMaterialStride, PHY_ScalarType& triangleType ,int subpart = 0);

    virtual void getLockedReadOnlyMaterialBase(const unsigned char **materialBase, int& numMaterials, PHY_ScalarType& materialType, int& materialStride,
        const unsigned char ** triangleMaterialBase, int& numTriangles, int& triangleMaterialStride, PHY_ScalarType& triangleType, int subpart = 0);

}
;

#endif //BT_MULTIMATERIAL_TRIANGLE_INDEX_VERTEX_ARRAY_H
