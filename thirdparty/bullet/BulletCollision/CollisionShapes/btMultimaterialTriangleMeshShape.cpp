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

#include "BulletCollision/CollisionShapes/btMultimaterialTriangleMeshShape.h"
#include "BulletCollision/CollisionShapes/btTriangleIndexVertexMaterialArray.h"
//#include "BulletCollision/CollisionShapes/btOptimizedBvh.h"

///Obtains the material for a specific triangle
const btMaterial *btMultimaterialTriangleMeshShape::getMaterialProperties(int partID, int triIndex)
{
	const unsigned char *materialBase = 0;
	int numMaterials;
	PHY_ScalarType materialType;
	int materialStride;
	const unsigned char *triangleMaterialBase = 0;
	int numTriangles;
	int triangleMaterialStride;
	PHY_ScalarType triangleType;

	((btTriangleIndexVertexMaterialArray *)m_meshInterface)->getLockedReadOnlyMaterialBase(&materialBase, numMaterials, materialType, materialStride, &triangleMaterialBase, numTriangles, triangleMaterialStride, triangleType, partID);

	// return the pointer to the place with the friction for the triangle
	// TODO: This depends on whether it's a moving mesh or not
	// BUG IN GIMPACT
	//return (btScalar*)(&materialBase[triangleMaterialBase[(triIndex-1) * triangleMaterialStride] * materialStride]);
	int *matInd = (int *)(&(triangleMaterialBase[(triIndex * triangleMaterialStride)]));
	btMaterial *matVal = (btMaterial *)(&(materialBase[*matInd * materialStride]));
	return (matVal);
}
