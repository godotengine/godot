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

#include "btTriangleIndexVertexMaterialArray.h"

btTriangleIndexVertexMaterialArray::btTriangleIndexVertexMaterialArray(int numTriangles, int* triangleIndexBase, int triangleIndexStride,
																	   int numVertices, btScalar* vertexBase, int vertexStride,
																	   int numMaterials, unsigned char* materialBase, int materialStride,
																	   int* triangleMaterialsBase, int materialIndexStride) : btTriangleIndexVertexArray(numTriangles, triangleIndexBase, triangleIndexStride, numVertices, vertexBase, vertexStride)
{
	btMaterialProperties mat;

	mat.m_numMaterials = numMaterials;
	mat.m_materialBase = materialBase;
	mat.m_materialStride = materialStride;
#ifdef BT_USE_DOUBLE_PRECISION
	mat.m_materialType = PHY_DOUBLE;
#else
	mat.m_materialType = PHY_FLOAT;
#endif

	mat.m_numTriangles = numTriangles;
	mat.m_triangleMaterialsBase = (unsigned char*)triangleMaterialsBase;
	mat.m_triangleMaterialStride = materialIndexStride;
	mat.m_triangleType = PHY_INTEGER;

	addMaterialProperties(mat);
}

void btTriangleIndexVertexMaterialArray::getLockedMaterialBase(unsigned char** materialBase, int& numMaterials, PHY_ScalarType& materialType, int& materialStride,
															   unsigned char** triangleMaterialBase, int& numTriangles, int& triangleMaterialStride, PHY_ScalarType& triangleType, int subpart)
{
	btAssert(subpart < getNumSubParts());

	btMaterialProperties& mats = m_materials[subpart];

	numMaterials = mats.m_numMaterials;
	(*materialBase) = (unsigned char*)mats.m_materialBase;
#ifdef BT_USE_DOUBLE_PRECISION
	materialType = PHY_DOUBLE;
#else
	materialType = PHY_FLOAT;
#endif
	materialStride = mats.m_materialStride;

	numTriangles = mats.m_numTriangles;
	(*triangleMaterialBase) = (unsigned char*)mats.m_triangleMaterialsBase;
	triangleMaterialStride = mats.m_triangleMaterialStride;
	triangleType = mats.m_triangleType;
}

void btTriangleIndexVertexMaterialArray::getLockedReadOnlyMaterialBase(const unsigned char** materialBase, int& numMaterials, PHY_ScalarType& materialType, int& materialStride,
																	   const unsigned char** triangleMaterialBase, int& numTriangles, int& triangleMaterialStride, PHY_ScalarType& triangleType, int subpart)
{
	btMaterialProperties& mats = m_materials[subpart];

	numMaterials = mats.m_numMaterials;
	(*materialBase) = (const unsigned char*)mats.m_materialBase;
#ifdef BT_USE_DOUBLE_PRECISION
	materialType = PHY_DOUBLE;
#else
	materialType = PHY_FLOAT;
#endif
	materialStride = mats.m_materialStride;

	numTriangles = mats.m_numTriangles;
	(*triangleMaterialBase) = (const unsigned char*)mats.m_triangleMaterialsBase;
	triangleMaterialStride = mats.m_triangleMaterialStride;
	triangleType = mats.m_triangleType;
}
