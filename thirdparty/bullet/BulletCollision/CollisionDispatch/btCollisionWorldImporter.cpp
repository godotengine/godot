/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2014 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "btCollisionWorldImporter.h"
#include "btBulletCollisionCommon.h"
#include "LinearMath/btSerializer.h"  //for btBulletSerializedArrays definition

#ifdef SUPPORT_GIMPACT_SHAPE_IMPORT
#include "BulletCollision/Gimpact/btGImpactShape.h"
#endif  //SUPPORT_GIMPACT_SHAPE_IMPORT

btCollisionWorldImporter::btCollisionWorldImporter(btCollisionWorld* world)
	: m_collisionWorld(world),
	  m_verboseMode(0)
{
}

btCollisionWorldImporter::~btCollisionWorldImporter()
{
}

bool btCollisionWorldImporter::convertAllObjects(btBulletSerializedArrays* arrays)
{
	m_shapeMap.clear();
	m_bodyMap.clear();

	int i;

	for (i = 0; i < arrays->m_bvhsDouble.size(); i++)
	{
		btOptimizedBvh* bvh = createOptimizedBvh();
		btQuantizedBvhDoubleData* bvhData = arrays->m_bvhsDouble[i];
		bvh->deSerializeDouble(*bvhData);
		m_bvhMap.insert(arrays->m_bvhsDouble[i], bvh);
	}
	for (i = 0; i < arrays->m_bvhsFloat.size(); i++)
	{
		btOptimizedBvh* bvh = createOptimizedBvh();
		btQuantizedBvhFloatData* bvhData = arrays->m_bvhsFloat[i];
		bvh->deSerializeFloat(*bvhData);
		m_bvhMap.insert(arrays->m_bvhsFloat[i], bvh);
	}

	for (i = 0; i < arrays->m_colShapeData.size(); i++)
	{
		btCollisionShapeData* shapeData = arrays->m_colShapeData[i];
		btCollisionShape* shape = convertCollisionShape(shapeData);
		if (shape)
		{
			//		printf("shapeMap.insert(%x,%x)\n",shapeData,shape);
			m_shapeMap.insert(shapeData, shape);
		}

		if (shape && shapeData->m_name)
		{
			char* newname = duplicateName(shapeData->m_name);
			m_objectNameMap.insert(shape, newname);
			m_nameShapeMap.insert(newname, shape);
		}
	}

	for (i = 0; i < arrays->m_collisionObjectDataDouble.size(); i++)
	{
		btCollisionObjectDoubleData* colObjData = arrays->m_collisionObjectDataDouble[i];
		btCollisionShape** shapePtr = m_shapeMap.find(colObjData->m_collisionShape);
		if (shapePtr && *shapePtr)
		{
			btTransform startTransform;
			colObjData->m_worldTransform.m_origin.m_floats[3] = 0.f;
			startTransform.deSerializeDouble(colObjData->m_worldTransform);

			btCollisionShape* shape = (btCollisionShape*)*shapePtr;
			btCollisionObject* body = createCollisionObject(startTransform, shape, colObjData->m_name);
			body->setFriction(btScalar(colObjData->m_friction));
			body->setRestitution(btScalar(colObjData->m_restitution));

#ifdef USE_INTERNAL_EDGE_UTILITY
			if (shape->getShapeType() == TRIANGLE_MESH_SHAPE_PROXYTYPE)
			{
				btBvhTriangleMeshShape* trimesh = (btBvhTriangleMeshShape*)shape;
				if (trimesh->getTriangleInfoMap())
				{
					body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);
				}
			}
#endif  //USE_INTERNAL_EDGE_UTILITY
			m_bodyMap.insert(colObjData, body);
		}
		else
		{
			printf("error: no shape found\n");
		}
	}
	for (i = 0; i < arrays->m_collisionObjectDataFloat.size(); i++)
	{
		btCollisionObjectFloatData* colObjData = arrays->m_collisionObjectDataFloat[i];
		btCollisionShape** shapePtr = m_shapeMap.find(colObjData->m_collisionShape);
		if (shapePtr && *shapePtr)
		{
			btTransform startTransform;
			colObjData->m_worldTransform.m_origin.m_floats[3] = 0.f;
			startTransform.deSerializeFloat(colObjData->m_worldTransform);

			btCollisionShape* shape = (btCollisionShape*)*shapePtr;
			btCollisionObject* body = createCollisionObject(startTransform, shape, colObjData->m_name);

#ifdef USE_INTERNAL_EDGE_UTILITY
			if (shape->getShapeType() == TRIANGLE_MESH_SHAPE_PROXYTYPE)
			{
				btBvhTriangleMeshShape* trimesh = (btBvhTriangleMeshShape*)shape;
				if (trimesh->getTriangleInfoMap())
				{
					body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);
				}
			}
#endif  //USE_INTERNAL_EDGE_UTILITY
			m_bodyMap.insert(colObjData, body);
		}
		else
		{
			printf("error: no shape found\n");
		}
	}

	return true;
}

void btCollisionWorldImporter::deleteAllData()
{
	int i;

	for (i = 0; i < m_allocatedCollisionObjects.size(); i++)
	{
		if (m_collisionWorld)
			m_collisionWorld->removeCollisionObject(m_allocatedCollisionObjects[i]);
		delete m_allocatedCollisionObjects[i];
	}

	m_allocatedCollisionObjects.clear();

	for (i = 0; i < m_allocatedCollisionShapes.size(); i++)
	{
		delete m_allocatedCollisionShapes[i];
	}
	m_allocatedCollisionShapes.clear();

	for (i = 0; i < m_allocatedBvhs.size(); i++)
	{
		delete m_allocatedBvhs[i];
	}
	m_allocatedBvhs.clear();

	for (i = 0; i < m_allocatedTriangleInfoMaps.size(); i++)
	{
		delete m_allocatedTriangleInfoMaps[i];
	}
	m_allocatedTriangleInfoMaps.clear();
	for (i = 0; i < m_allocatedTriangleIndexArrays.size(); i++)
	{
		delete m_allocatedTriangleIndexArrays[i];
	}
	m_allocatedTriangleIndexArrays.clear();
	for (i = 0; i < m_allocatedNames.size(); i++)
	{
		delete[] m_allocatedNames[i];
	}
	m_allocatedNames.clear();

	for (i = 0; i < m_allocatedbtStridingMeshInterfaceDatas.size(); i++)
	{
		btStridingMeshInterfaceData* curData = m_allocatedbtStridingMeshInterfaceDatas[i];

		for (int a = 0; a < curData->m_numMeshParts; a++)
		{
			btMeshPartData* curPart = &curData->m_meshPartsPtr[a];
			if (curPart->m_vertices3f)
				delete[] curPart->m_vertices3f;

			if (curPart->m_vertices3d)
				delete[] curPart->m_vertices3d;

			if (curPart->m_indices32)
				delete[] curPart->m_indices32;

			if (curPart->m_3indices16)
				delete[] curPart->m_3indices16;

			if (curPart->m_indices16)
				delete[] curPart->m_indices16;

			if (curPart->m_3indices8)
				delete[] curPart->m_3indices8;
		}
		delete[] curData->m_meshPartsPtr;
		delete curData;
	}
	m_allocatedbtStridingMeshInterfaceDatas.clear();

	for (i = 0; i < m_indexArrays.size(); i++)
	{
		btAlignedFree(m_indexArrays[i]);
	}
	m_indexArrays.clear();

	for (i = 0; i < m_shortIndexArrays.size(); i++)
	{
		btAlignedFree(m_shortIndexArrays[i]);
	}
	m_shortIndexArrays.clear();

	for (i = 0; i < m_charIndexArrays.size(); i++)
	{
		btAlignedFree(m_charIndexArrays[i]);
	}
	m_charIndexArrays.clear();

	for (i = 0; i < m_floatVertexArrays.size(); i++)
	{
		btAlignedFree(m_floatVertexArrays[i]);
	}
	m_floatVertexArrays.clear();

	for (i = 0; i < m_doubleVertexArrays.size(); i++)
	{
		btAlignedFree(m_doubleVertexArrays[i]);
	}
	m_doubleVertexArrays.clear();
}

btCollisionShape* btCollisionWorldImporter::convertCollisionShape(btCollisionShapeData* shapeData)
{
	btCollisionShape* shape = 0;

	switch (shapeData->m_shapeType)
	{
		case STATIC_PLANE_PROXYTYPE:
		{
			btStaticPlaneShapeData* planeData = (btStaticPlaneShapeData*)shapeData;
			btVector3 planeNormal, localScaling;
			planeNormal.deSerializeFloat(planeData->m_planeNormal);
			localScaling.deSerializeFloat(planeData->m_localScaling);
			shape = createPlaneShape(planeNormal, planeData->m_planeConstant);
			shape->setLocalScaling(localScaling);

			break;
		}
		case SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE:
		{
			btScaledTriangleMeshShapeData* scaledMesh = (btScaledTriangleMeshShapeData*)shapeData;
			btCollisionShapeData* colShapeData = (btCollisionShapeData*)&scaledMesh->m_trimeshShapeData;
			colShapeData->m_shapeType = TRIANGLE_MESH_SHAPE_PROXYTYPE;
			btCollisionShape* childShape = convertCollisionShape(colShapeData);
			btBvhTriangleMeshShape* meshShape = (btBvhTriangleMeshShape*)childShape;
			btVector3 localScaling;
			localScaling.deSerializeFloat(scaledMesh->m_localScaling);

			shape = createScaledTrangleMeshShape(meshShape, localScaling);
			break;
		}
#ifdef SUPPORT_GIMPACT_SHAPE_IMPORT
		case GIMPACT_SHAPE_PROXYTYPE:
		{
			btGImpactMeshShapeData* gimpactData = (btGImpactMeshShapeData*)shapeData;
			if (gimpactData->m_gimpactSubType == CONST_GIMPACT_TRIMESH_SHAPE)
			{
				btStridingMeshInterfaceData* interfaceData = createStridingMeshInterfaceData(&gimpactData->m_meshInterface);
				btTriangleIndexVertexArray* meshInterface = createMeshInterface(*interfaceData);

				btGImpactMeshShape* gimpactShape = createGimpactShape(meshInterface);
				btVector3 localScaling;
				localScaling.deSerializeFloat(gimpactData->m_localScaling);
				gimpactShape->setLocalScaling(localScaling);
				gimpactShape->setMargin(btScalar(gimpactData->m_collisionMargin));
				gimpactShape->updateBound();
				shape = gimpactShape;
			}
			else
			{
				printf("unsupported gimpact sub type\n");
			}
			break;
		}
#endif  //SUPPORT_GIMPACT_SHAPE_IMPORT                                                                        \
		//The btCapsuleShape* API has issue passing the margin/scaling/halfextents unmodified through the API \
		//so deal with this
		case CAPSULE_SHAPE_PROXYTYPE:
		{
			btCapsuleShapeData* capData = (btCapsuleShapeData*)shapeData;

			switch (capData->m_upAxis)
			{
				case 0:
				{
					shape = createCapsuleShapeX(1, 1);
					break;
				}
				case 1:
				{
					shape = createCapsuleShapeY(1, 1);
					break;
				}
				case 2:
				{
					shape = createCapsuleShapeZ(1, 1);
					break;
				}
				default:
				{
					printf("error: wrong up axis for btCapsuleShape\n");
				}
			};
			if (shape)
			{
				btCapsuleShape* cap = (btCapsuleShape*)shape;
				cap->deSerializeFloat(capData);
			}
			break;
		}
		case CYLINDER_SHAPE_PROXYTYPE:
		case CONE_SHAPE_PROXYTYPE:
		case BOX_SHAPE_PROXYTYPE:
		case SPHERE_SHAPE_PROXYTYPE:
		case MULTI_SPHERE_SHAPE_PROXYTYPE:
		case CONVEX_HULL_SHAPE_PROXYTYPE:
		{
			btConvexInternalShapeData* bsd = (btConvexInternalShapeData*)shapeData;
			btVector3 implicitShapeDimensions;
			implicitShapeDimensions.deSerializeFloat(bsd->m_implicitShapeDimensions);
			btVector3 localScaling;
			localScaling.deSerializeFloat(bsd->m_localScaling);
			btVector3 margin(bsd->m_collisionMargin, bsd->m_collisionMargin, bsd->m_collisionMargin);
			switch (shapeData->m_shapeType)
			{
				case BOX_SHAPE_PROXYTYPE:
				{
					btBoxShape* box = (btBoxShape*)createBoxShape(implicitShapeDimensions / localScaling + margin);
					//box->initializePolyhedralFeatures();
					shape = box;

					break;
				}
				case SPHERE_SHAPE_PROXYTYPE:
				{
					shape = createSphereShape(implicitShapeDimensions.getX());
					break;
				}

				case CYLINDER_SHAPE_PROXYTYPE:
				{
					btCylinderShapeData* cylData = (btCylinderShapeData*)shapeData;
					btVector3 halfExtents = implicitShapeDimensions + margin;
					switch (cylData->m_upAxis)
					{
						case 0:
						{
							shape = createCylinderShapeX(halfExtents.getY(), halfExtents.getX());
							break;
						}
						case 1:
						{
							shape = createCylinderShapeY(halfExtents.getX(), halfExtents.getY());
							break;
						}
						case 2:
						{
							shape = createCylinderShapeZ(halfExtents.getX(), halfExtents.getZ());
							break;
						}
						default:
						{
							printf("unknown Cylinder up axis\n");
						}
					};

					break;
				}
				case CONE_SHAPE_PROXYTYPE:
				{
					btConeShapeData* conData = (btConeShapeData*)shapeData;
					btVector3 halfExtents = implicitShapeDimensions;  //+margin;
					switch (conData->m_upIndex)
					{
						case 0:
						{
							shape = createConeShapeX(halfExtents.getY(), halfExtents.getX());
							break;
						}
						case 1:
						{
							shape = createConeShapeY(halfExtents.getX(), halfExtents.getY());
							break;
						}
						case 2:
						{
							shape = createConeShapeZ(halfExtents.getX(), halfExtents.getZ());
							break;
						}
						default:
						{
							printf("unknown Cone up axis\n");
						}
					};

					break;
				}
				case MULTI_SPHERE_SHAPE_PROXYTYPE:
				{
					btMultiSphereShapeData* mss = (btMultiSphereShapeData*)bsd;
					int numSpheres = mss->m_localPositionArraySize;

					btAlignedObjectArray<btVector3> tmpPos;
					btAlignedObjectArray<btScalar> radii;
					radii.resize(numSpheres);
					tmpPos.resize(numSpheres);
					int i;
					for (i = 0; i < numSpheres; i++)
					{
						tmpPos[i].deSerializeFloat(mss->m_localPositionArrayPtr[i].m_pos);
						radii[i] = mss->m_localPositionArrayPtr[i].m_radius;
					}
					shape = createMultiSphereShape(&tmpPos[0], &radii[0], numSpheres);
					break;
				}
				case CONVEX_HULL_SHAPE_PROXYTYPE:
				{
					//	int sz = sizeof(btConvexHullShapeData);
					//	int sz2 = sizeof(btConvexInternalShapeData);
					//	int sz3 = sizeof(btCollisionShapeData);
					btConvexHullShapeData* convexData = (btConvexHullShapeData*)bsd;
					int numPoints = convexData->m_numUnscaledPoints;

					btAlignedObjectArray<btVector3> tmpPoints;
					tmpPoints.resize(numPoints);
					int i;
					for (i = 0; i < numPoints; i++)
					{
#ifdef BT_USE_DOUBLE_PRECISION
						if (convexData->m_unscaledPointsDoublePtr)
							tmpPoints[i].deSerialize(convexData->m_unscaledPointsDoublePtr[i]);
						if (convexData->m_unscaledPointsFloatPtr)
							tmpPoints[i].deSerializeFloat(convexData->m_unscaledPointsFloatPtr[i]);
#else
						if (convexData->m_unscaledPointsFloatPtr)
							tmpPoints[i].deSerialize(convexData->m_unscaledPointsFloatPtr[i]);
						if (convexData->m_unscaledPointsDoublePtr)
							tmpPoints[i].deSerializeDouble(convexData->m_unscaledPointsDoublePtr[i]);
#endif  //BT_USE_DOUBLE_PRECISION
					}
					btConvexHullShape* hullShape = createConvexHullShape();
					for (i = 0; i < numPoints; i++)
					{
						hullShape->addPoint(tmpPoints[i]);
					}
					hullShape->setMargin(bsd->m_collisionMargin);
					//hullShape->initializePolyhedralFeatures();
					shape = hullShape;
					break;
				}
				default:
				{
					printf("error: cannot create shape type (%d)\n", shapeData->m_shapeType);
				}
			}

			if (shape)
			{
				shape->setMargin(bsd->m_collisionMargin);

				btVector3 localScaling;
				localScaling.deSerializeFloat(bsd->m_localScaling);
				shape->setLocalScaling(localScaling);
			}
			break;
		}
		case TRIANGLE_MESH_SHAPE_PROXYTYPE:
		{
			btTriangleMeshShapeData* trimesh = (btTriangleMeshShapeData*)shapeData;
			btStridingMeshInterfaceData* interfaceData = createStridingMeshInterfaceData(&trimesh->m_meshInterface);
			btTriangleIndexVertexArray* meshInterface = createMeshInterface(*interfaceData);
			if (!meshInterface->getNumSubParts())
			{
				return 0;
			}

			btVector3 scaling;
			scaling.deSerializeFloat(trimesh->m_meshInterface.m_scaling);
			meshInterface->setScaling(scaling);

			btOptimizedBvh* bvh = 0;
#if 1
			if (trimesh->m_quantizedFloatBvh)
			{
				btOptimizedBvh** bvhPtr = m_bvhMap.find(trimesh->m_quantizedFloatBvh);
				if (bvhPtr && *bvhPtr)
				{
					bvh = *bvhPtr;
				}
				else
				{
					bvh = createOptimizedBvh();
					bvh->deSerializeFloat(*trimesh->m_quantizedFloatBvh);
				}
			}
			if (trimesh->m_quantizedDoubleBvh)
			{
				btOptimizedBvh** bvhPtr = m_bvhMap.find(trimesh->m_quantizedDoubleBvh);
				if (bvhPtr && *bvhPtr)
				{
					bvh = *bvhPtr;
				}
				else
				{
					bvh = createOptimizedBvh();
					bvh->deSerializeDouble(*trimesh->m_quantizedDoubleBvh);
				}
			}
#endif

			btBvhTriangleMeshShape* trimeshShape = createBvhTriangleMeshShape(meshInterface, bvh);
			trimeshShape->setMargin(trimesh->m_collisionMargin);
			shape = trimeshShape;

			if (trimesh->m_triangleInfoMap)
			{
				btTriangleInfoMap* map = createTriangleInfoMap();
				map->deSerialize(*trimesh->m_triangleInfoMap);
				trimeshShape->setTriangleInfoMap(map);

#ifdef USE_INTERNAL_EDGE_UTILITY
				gContactAddedCallback = btAdjustInternalEdgeContactsCallback;
#endif  //USE_INTERNAL_EDGE_UTILITY
			}

			//printf("trimesh->m_collisionMargin=%f\n",trimesh->m_collisionMargin);
			break;
		}
		case COMPOUND_SHAPE_PROXYTYPE:
		{
			btCompoundShapeData* compoundData = (btCompoundShapeData*)shapeData;
			btCompoundShape* compoundShape = createCompoundShape();

			//btCompoundShapeChildData* childShapeDataArray = &compoundData->m_childShapePtr[0];

			btAlignedObjectArray<btCollisionShape*> childShapes;
			for (int i = 0; i < compoundData->m_numChildShapes; i++)
			{
				//btCompoundShapeChildData* ptr = &compoundData->m_childShapePtr[i];

				btCollisionShapeData* cd = compoundData->m_childShapePtr[i].m_childShape;

				btCollisionShape* childShape = convertCollisionShape(cd);
				if (childShape)
				{
					btTransform localTransform;
					localTransform.deSerializeFloat(compoundData->m_childShapePtr[i].m_transform);
					compoundShape->addChildShape(localTransform, childShape);
				}
				else
				{
#ifdef _DEBUG
					printf("error: couldn't create childShape for compoundShape\n");
#endif
				}
			}
			shape = compoundShape;

			break;
		}
		case SOFTBODY_SHAPE_PROXYTYPE:
		{
			return 0;
		}
		default:
		{
#ifdef _DEBUG
			printf("unsupported shape type (%d)\n", shapeData->m_shapeType);
#endif
		}
	}

	return shape;
}

char* btCollisionWorldImporter::duplicateName(const char* name)
{
	if (name)
	{
		int l = (int)strlen(name);
		char* newName = new char[l + 1];
		memcpy(newName, name, l);
		newName[l] = 0;
		m_allocatedNames.push_back(newName);
		return newName;
	}
	return 0;
}

btTriangleIndexVertexArray* btCollisionWorldImporter::createMeshInterface(btStridingMeshInterfaceData& meshData)
{
	btTriangleIndexVertexArray* meshInterface = createTriangleMeshContainer();

	for (int i = 0; i < meshData.m_numMeshParts; i++)
	{
		btIndexedMesh meshPart;
		meshPart.m_numTriangles = meshData.m_meshPartsPtr[i].m_numTriangles;
		meshPart.m_numVertices = meshData.m_meshPartsPtr[i].m_numVertices;

		if (meshData.m_meshPartsPtr[i].m_indices32)
		{
			meshPart.m_indexType = PHY_INTEGER;
			meshPart.m_triangleIndexStride = 3 * sizeof(int);
			int* indexArray = (int*)btAlignedAlloc(sizeof(int) * 3 * meshPart.m_numTriangles, 16);
			m_indexArrays.push_back(indexArray);
			for (int j = 0; j < 3 * meshPart.m_numTriangles; j++)
			{
				indexArray[j] = meshData.m_meshPartsPtr[i].m_indices32[j].m_value;
			}
			meshPart.m_triangleIndexBase = (const unsigned char*)indexArray;
		}
		else
		{
			if (meshData.m_meshPartsPtr[i].m_3indices16)
			{
				meshPart.m_indexType = PHY_SHORT;
				meshPart.m_triangleIndexStride = sizeof(short int) * 3;  //sizeof(btShortIntIndexTripletData);

				short int* indexArray = (short int*)btAlignedAlloc(sizeof(short int) * 3 * meshPart.m_numTriangles, 16);
				m_shortIndexArrays.push_back(indexArray);

				for (int j = 0; j < meshPart.m_numTriangles; j++)
				{
					indexArray[3 * j] = meshData.m_meshPartsPtr[i].m_3indices16[j].m_values[0];
					indexArray[3 * j + 1] = meshData.m_meshPartsPtr[i].m_3indices16[j].m_values[1];
					indexArray[3 * j + 2] = meshData.m_meshPartsPtr[i].m_3indices16[j].m_values[2];
				}

				meshPart.m_triangleIndexBase = (const unsigned char*)indexArray;
			}
			if (meshData.m_meshPartsPtr[i].m_indices16)
			{
				meshPart.m_indexType = PHY_SHORT;
				meshPart.m_triangleIndexStride = 3 * sizeof(short int);
				short int* indexArray = (short int*)btAlignedAlloc(sizeof(short int) * 3 * meshPart.m_numTriangles, 16);
				m_shortIndexArrays.push_back(indexArray);
				for (int j = 0; j < 3 * meshPart.m_numTriangles; j++)
				{
					indexArray[j] = meshData.m_meshPartsPtr[i].m_indices16[j].m_value;
				}

				meshPart.m_triangleIndexBase = (const unsigned char*)indexArray;
			}

			if (meshData.m_meshPartsPtr[i].m_3indices8)
			{
				meshPart.m_indexType = PHY_UCHAR;
				meshPart.m_triangleIndexStride = sizeof(unsigned char) * 3;

				unsigned char* indexArray = (unsigned char*)btAlignedAlloc(sizeof(unsigned char) * 3 * meshPart.m_numTriangles, 16);
				m_charIndexArrays.push_back(indexArray);

				for (int j = 0; j < meshPart.m_numTriangles; j++)
				{
					indexArray[3 * j] = meshData.m_meshPartsPtr[i].m_3indices8[j].m_values[0];
					indexArray[3 * j + 1] = meshData.m_meshPartsPtr[i].m_3indices8[j].m_values[1];
					indexArray[3 * j + 2] = meshData.m_meshPartsPtr[i].m_3indices8[j].m_values[2];
				}

				meshPart.m_triangleIndexBase = (const unsigned char*)indexArray;
			}
		}

		if (meshData.m_meshPartsPtr[i].m_vertices3f)
		{
			meshPart.m_vertexType = PHY_FLOAT;
			meshPart.m_vertexStride = sizeof(btVector3FloatData);
			btVector3FloatData* vertices = (btVector3FloatData*)btAlignedAlloc(sizeof(btVector3FloatData) * meshPart.m_numVertices, 16);
			m_floatVertexArrays.push_back(vertices);

			for (int j = 0; j < meshPart.m_numVertices; j++)
			{
				vertices[j].m_floats[0] = meshData.m_meshPartsPtr[i].m_vertices3f[j].m_floats[0];
				vertices[j].m_floats[1] = meshData.m_meshPartsPtr[i].m_vertices3f[j].m_floats[1];
				vertices[j].m_floats[2] = meshData.m_meshPartsPtr[i].m_vertices3f[j].m_floats[2];
				vertices[j].m_floats[3] = meshData.m_meshPartsPtr[i].m_vertices3f[j].m_floats[3];
			}
			meshPart.m_vertexBase = (const unsigned char*)vertices;
		}
		else
		{
			meshPart.m_vertexType = PHY_DOUBLE;
			meshPart.m_vertexStride = sizeof(btVector3DoubleData);

			btVector3DoubleData* vertices = (btVector3DoubleData*)btAlignedAlloc(sizeof(btVector3DoubleData) * meshPart.m_numVertices, 16);
			m_doubleVertexArrays.push_back(vertices);

			for (int j = 0; j < meshPart.m_numVertices; j++)
			{
				vertices[j].m_floats[0] = meshData.m_meshPartsPtr[i].m_vertices3d[j].m_floats[0];
				vertices[j].m_floats[1] = meshData.m_meshPartsPtr[i].m_vertices3d[j].m_floats[1];
				vertices[j].m_floats[2] = meshData.m_meshPartsPtr[i].m_vertices3d[j].m_floats[2];
				vertices[j].m_floats[3] = meshData.m_meshPartsPtr[i].m_vertices3d[j].m_floats[3];
			}
			meshPart.m_vertexBase = (const unsigned char*)vertices;
		}

		if (meshPart.m_triangleIndexBase && meshPart.m_vertexBase)
		{
			meshInterface->addIndexedMesh(meshPart, meshPart.m_indexType);
		}
	}

	return meshInterface;
}

btStridingMeshInterfaceData* btCollisionWorldImporter::createStridingMeshInterfaceData(btStridingMeshInterfaceData* interfaceData)
{
	//create a new btStridingMeshInterfaceData that is an exact copy of shapedata and store it in the WorldImporter
	btStridingMeshInterfaceData* newData = new btStridingMeshInterfaceData;

	newData->m_scaling = interfaceData->m_scaling;
	newData->m_numMeshParts = interfaceData->m_numMeshParts;
	newData->m_meshPartsPtr = new btMeshPartData[newData->m_numMeshParts];

	for (int i = 0; i < newData->m_numMeshParts; i++)
	{
		btMeshPartData* curPart = &interfaceData->m_meshPartsPtr[i];
		btMeshPartData* curNewPart = &newData->m_meshPartsPtr[i];

		curNewPart->m_numTriangles = curPart->m_numTriangles;
		curNewPart->m_numVertices = curPart->m_numVertices;

		if (curPart->m_vertices3f)
		{
			curNewPart->m_vertices3f = new btVector3FloatData[curNewPart->m_numVertices];
			memcpy(curNewPart->m_vertices3f, curPart->m_vertices3f, sizeof(btVector3FloatData) * curNewPart->m_numVertices);
		}
		else
			curNewPart->m_vertices3f = NULL;

		if (curPart->m_vertices3d)
		{
			curNewPart->m_vertices3d = new btVector3DoubleData[curNewPart->m_numVertices];
			memcpy(curNewPart->m_vertices3d, curPart->m_vertices3d, sizeof(btVector3DoubleData) * curNewPart->m_numVertices);
		}
		else
			curNewPart->m_vertices3d = NULL;

		int numIndices = curNewPart->m_numTriangles * 3;
		///the m_3indices8 was not initialized in some Bullet versions, this can cause crashes at loading time
		///we catch it by only dealing with m_3indices8 if none of the other indices are initialized
		bool uninitialized3indices8Workaround = false;

		if (curPart->m_indices32)
		{
			uninitialized3indices8Workaround = true;
			curNewPart->m_indices32 = new btIntIndexData[numIndices];
			memcpy(curNewPart->m_indices32, curPart->m_indices32, sizeof(btIntIndexData) * numIndices);
		}
		else
			curNewPart->m_indices32 = NULL;

		if (curPart->m_3indices16)
		{
			uninitialized3indices8Workaround = true;
			curNewPart->m_3indices16 = new btShortIntIndexTripletData[curNewPart->m_numTriangles];
			memcpy(curNewPart->m_3indices16, curPart->m_3indices16, sizeof(btShortIntIndexTripletData) * curNewPart->m_numTriangles);
		}
		else
			curNewPart->m_3indices16 = NULL;

		if (curPart->m_indices16)
		{
			uninitialized3indices8Workaround = true;
			curNewPart->m_indices16 = new btShortIntIndexData[numIndices];
			memcpy(curNewPart->m_indices16, curPart->m_indices16, sizeof(btShortIntIndexData) * numIndices);
		}
		else
			curNewPart->m_indices16 = NULL;

		if (!uninitialized3indices8Workaround && curPart->m_3indices8)
		{
			curNewPart->m_3indices8 = new btCharIndexTripletData[curNewPart->m_numTriangles];
			memcpy(curNewPart->m_3indices8, curPart->m_3indices8, sizeof(btCharIndexTripletData) * curNewPart->m_numTriangles);
		}
		else
			curNewPart->m_3indices8 = NULL;
	}

	m_allocatedbtStridingMeshInterfaceDatas.push_back(newData);

	return (newData);
}

#ifdef USE_INTERNAL_EDGE_UTILITY
extern ContactAddedCallback gContactAddedCallback;

static bool btAdjustInternalEdgeContactsCallback(btManifoldPoint& cp, const btCollisionObject* colObj0, int partId0, int index0, const btCollisionObject* colObj1, int partId1, int index1)
{
	btAdjustInternalEdgeContacts(cp, colObj1, colObj0, partId1, index1);
	//btAdjustInternalEdgeContacts(cp,colObj1,colObj0, partId1,index1, BT_TRIANGLE_CONVEX_BACKFACE_MODE);
	//btAdjustInternalEdgeContacts(cp,colObj1,colObj0, partId1,index1, BT_TRIANGLE_CONVEX_DOUBLE_SIDED+BT_TRIANGLE_CONCAVE_DOUBLE_SIDED);
	return true;
}
#endif  //USE_INTERNAL_EDGE_UTILITY

/*
btRigidBody*  btWorldImporter::createRigidBody(bool isDynamic, btScalar mass, const btTransform& startTransform,btCollisionShape* shape,const char* bodyName)
{
	btVector3 localInertia;
	localInertia.setZero();

	if (mass)
		shape->calculateLocalInertia(mass,localInertia);

	btRigidBody* body = new btRigidBody(mass,0,shape,localInertia);
	body->setWorldTransform(startTransform);

	if (m_dynamicsWorld)
		m_dynamicsWorld->addRigidBody(body);

	if (bodyName)
	{
		char* newname = duplicateName(bodyName);
		m_objectNameMap.insert(body,newname);
		m_nameBodyMap.insert(newname,body);
	}
	m_allocatedRigidBodies.push_back(body);
	return body;

}
*/

btCollisionObject* btCollisionWorldImporter::getCollisionObjectByName(const char* name)
{
	btCollisionObject** bodyPtr = m_nameColObjMap.find(name);
	if (bodyPtr && *bodyPtr)
	{
		return *bodyPtr;
	}
	return 0;
}

btCollisionObject* btCollisionWorldImporter::createCollisionObject(const btTransform& startTransform, btCollisionShape* shape, const char* bodyName)
{
	btCollisionObject* colObj = new btCollisionObject();
	colObj->setWorldTransform(startTransform);
	colObj->setCollisionShape(shape);
	m_collisionWorld->addCollisionObject(colObj);  //todo: flags etc

	if (bodyName)
	{
		char* newname = duplicateName(bodyName);
		m_objectNameMap.insert(colObj, newname);
		m_nameColObjMap.insert(newname, colObj);
	}
	m_allocatedCollisionObjects.push_back(colObj);

	return colObj;
}

btCollisionShape* btCollisionWorldImporter::createPlaneShape(const btVector3& planeNormal, btScalar planeConstant)
{
	btStaticPlaneShape* shape = new btStaticPlaneShape(planeNormal, planeConstant);
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}
btCollisionShape* btCollisionWorldImporter::createBoxShape(const btVector3& halfExtents)
{
	btBoxShape* shape = new btBoxShape(halfExtents);
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}
btCollisionShape* btCollisionWorldImporter::createSphereShape(btScalar radius)
{
	btSphereShape* shape = new btSphereShape(radius);
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btCollisionShape* btCollisionWorldImporter::createCapsuleShapeX(btScalar radius, btScalar height)
{
	btCapsuleShapeX* shape = new btCapsuleShapeX(radius, height);
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btCollisionShape* btCollisionWorldImporter::createCapsuleShapeY(btScalar radius, btScalar height)
{
	btCapsuleShape* shape = new btCapsuleShape(radius, height);
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btCollisionShape* btCollisionWorldImporter::createCapsuleShapeZ(btScalar radius, btScalar height)
{
	btCapsuleShapeZ* shape = new btCapsuleShapeZ(radius, height);
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btCollisionShape* btCollisionWorldImporter::createCylinderShapeX(btScalar radius, btScalar height)
{
	btCylinderShapeX* shape = new btCylinderShapeX(btVector3(height, radius, radius));
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btCollisionShape* btCollisionWorldImporter::createCylinderShapeY(btScalar radius, btScalar height)
{
	btCylinderShape* shape = new btCylinderShape(btVector3(radius, height, radius));
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btCollisionShape* btCollisionWorldImporter::createCylinderShapeZ(btScalar radius, btScalar height)
{
	btCylinderShapeZ* shape = new btCylinderShapeZ(btVector3(radius, radius, height));
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btCollisionShape* btCollisionWorldImporter::createConeShapeX(btScalar radius, btScalar height)
{
	btConeShapeX* shape = new btConeShapeX(radius, height);
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btCollisionShape* btCollisionWorldImporter::createConeShapeY(btScalar radius, btScalar height)
{
	btConeShape* shape = new btConeShape(radius, height);
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btCollisionShape* btCollisionWorldImporter::createConeShapeZ(btScalar radius, btScalar height)
{
	btConeShapeZ* shape = new btConeShapeZ(radius, height);
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btTriangleIndexVertexArray* btCollisionWorldImporter::createTriangleMeshContainer()
{
	btTriangleIndexVertexArray* in = new btTriangleIndexVertexArray();
	m_allocatedTriangleIndexArrays.push_back(in);
	return in;
}

btOptimizedBvh* btCollisionWorldImporter::createOptimizedBvh()
{
	btOptimizedBvh* bvh = new btOptimizedBvh();
	m_allocatedBvhs.push_back(bvh);
	return bvh;
}

btTriangleInfoMap* btCollisionWorldImporter::createTriangleInfoMap()
{
	btTriangleInfoMap* tim = new btTriangleInfoMap();
	m_allocatedTriangleInfoMaps.push_back(tim);
	return tim;
}

btBvhTriangleMeshShape* btCollisionWorldImporter::createBvhTriangleMeshShape(btStridingMeshInterface* trimesh, btOptimizedBvh* bvh)
{
	if (bvh)
	{
		btBvhTriangleMeshShape* bvhTriMesh = new btBvhTriangleMeshShape(trimesh, bvh->isQuantized(), false);
		bvhTriMesh->setOptimizedBvh(bvh);
		m_allocatedCollisionShapes.push_back(bvhTriMesh);
		return bvhTriMesh;
	}

	btBvhTriangleMeshShape* ts = new btBvhTriangleMeshShape(trimesh, true);
	m_allocatedCollisionShapes.push_back(ts);
	return ts;
}
btCollisionShape* btCollisionWorldImporter::createConvexTriangleMeshShape(btStridingMeshInterface* trimesh)
{
	return 0;
}
#ifdef SUPPORT_GIMPACT_SHAPE_IMPORT
btGImpactMeshShape* btCollisionWorldImporter::createGimpactShape(btStridingMeshInterface* trimesh)
{
	btGImpactMeshShape* shape = new btGImpactMeshShape(trimesh);
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}
#endif  //SUPPORT_GIMPACT_SHAPE_IMPORT

btConvexHullShape* btCollisionWorldImporter::createConvexHullShape()
{
	btConvexHullShape* shape = new btConvexHullShape();
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btCompoundShape* btCollisionWorldImporter::createCompoundShape()
{
	btCompoundShape* shape = new btCompoundShape();
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btScaledBvhTriangleMeshShape* btCollisionWorldImporter::createScaledTrangleMeshShape(btBvhTriangleMeshShape* meshShape, const btVector3& localScaling)
{
	btScaledBvhTriangleMeshShape* shape = new btScaledBvhTriangleMeshShape(meshShape, localScaling);
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

btMultiSphereShape* btCollisionWorldImporter::createMultiSphereShape(const btVector3* positions, const btScalar* radi, int numSpheres)
{
	btMultiSphereShape* shape = new btMultiSphereShape(positions, radi, numSpheres);
	m_allocatedCollisionShapes.push_back(shape);
	return shape;
}

// query for data
int btCollisionWorldImporter::getNumCollisionShapes() const
{
	return m_allocatedCollisionShapes.size();
}

btCollisionShape* btCollisionWorldImporter::getCollisionShapeByIndex(int index)
{
	return m_allocatedCollisionShapes[index];
}

btCollisionShape* btCollisionWorldImporter::getCollisionShapeByName(const char* name)
{
	btCollisionShape** shapePtr = m_nameShapeMap.find(name);
	if (shapePtr && *shapePtr)
	{
		return *shapePtr;
	}
	return 0;
}

const char* btCollisionWorldImporter::getNameForPointer(const void* ptr) const
{
	const char* const* namePtr = m_objectNameMap.find(ptr);
	if (namePtr && *namePtr)
		return *namePtr;
	return 0;
}

int btCollisionWorldImporter::getNumRigidBodies() const
{
	return m_allocatedRigidBodies.size();
}

btCollisionObject* btCollisionWorldImporter::getRigidBodyByIndex(int index) const
{
	return m_allocatedRigidBodies[index];
}

int btCollisionWorldImporter::getNumBvhs() const
{
	return m_allocatedBvhs.size();
}
btOptimizedBvh* btCollisionWorldImporter::getBvhByIndex(int index) const
{
	return m_allocatedBvhs[index];
}

int btCollisionWorldImporter::getNumTriangleInfoMaps() const
{
	return m_allocatedTriangleInfoMaps.size();
}

btTriangleInfoMap* btCollisionWorldImporter::getTriangleInfoMapByIndex(int index) const
{
	return m_allocatedTriangleInfoMaps[index];
}
