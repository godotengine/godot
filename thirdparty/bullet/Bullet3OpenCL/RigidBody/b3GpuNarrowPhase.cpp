#include "b3GpuNarrowPhase.h"

#include "Bullet3OpenCL/ParallelPrimitives/b3OpenCLArray.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3ConvexPolyhedronData.h"
#include "Bullet3OpenCL/NarrowphaseCollision/b3ConvexHullContact.h"
#include "Bullet3OpenCL/BroadphaseCollision/b3SapAabb.h"
#include <string.h>
#include "Bullet3Collision/NarrowPhaseCollision/b3Config.h"
#include "Bullet3OpenCL/NarrowphaseCollision/b3OptimizedBvh.h"
#include "Bullet3OpenCL/NarrowphaseCollision/b3TriangleIndexVertexArray.h"
#include "Bullet3Geometry/b3AabbUtil.h"
#include "Bullet3OpenCL/NarrowphaseCollision/b3BvhInfo.h"

#include "b3GpuNarrowPhaseInternalData.h"
#include "Bullet3OpenCL/NarrowphaseCollision/b3QuantizedBvh.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3ConvexUtility.h"

b3GpuNarrowPhase::b3GpuNarrowPhase(cl_context ctx, cl_device_id device, cl_command_queue queue, const b3Config& config)
	: m_data(0), m_planeBodyIndex(-1), m_static0Index(-1), m_context(ctx), m_device(device), m_queue(queue)
{
	m_data = new b3GpuNarrowPhaseInternalData();
	m_data->m_currentContactBuffer = 0;

	memset(m_data, 0, sizeof(b3GpuNarrowPhaseInternalData));

	m_data->m_config = config;

	m_data->m_gpuSatCollision = new GpuSatCollision(ctx, device, queue);

	m_data->m_triangleConvexPairs = new b3OpenCLArray<b3Int4>(m_context, m_queue, config.m_maxTriConvexPairCapacity);

	//m_data->m_convexPairsOutGPU = new b3OpenCLArray<b3Int2>(ctx,queue,config.m_maxBroadphasePairs,false);
	//m_data->m_planePairs = new b3OpenCLArray<b3Int2>(ctx,queue,config.m_maxBroadphasePairs,false);

	m_data->m_pBufContactOutCPU = new b3AlignedObjectArray<b3Contact4>();
	m_data->m_pBufContactOutCPU->resize(config.m_maxBroadphasePairs);
	m_data->m_bodyBufferCPU = new b3AlignedObjectArray<b3RigidBodyData>();
	m_data->m_bodyBufferCPU->resize(config.m_maxConvexBodies);

	m_data->m_inertiaBufferCPU = new b3AlignedObjectArray<b3InertiaData>();
	m_data->m_inertiaBufferCPU->resize(config.m_maxConvexBodies);

	m_data->m_pBufContactBuffersGPU[0] = new b3OpenCLArray<b3Contact4>(ctx, queue, config.m_maxContactCapacity, true);
	m_data->m_pBufContactBuffersGPU[1] = new b3OpenCLArray<b3Contact4>(ctx, queue, config.m_maxContactCapacity, true);

	m_data->m_inertiaBufferGPU = new b3OpenCLArray<b3InertiaData>(ctx, queue, config.m_maxConvexBodies, false);
	m_data->m_collidablesGPU = new b3OpenCLArray<b3Collidable>(ctx, queue, config.m_maxConvexShapes);
	m_data->m_collidablesCPU.reserve(config.m_maxConvexShapes);

	m_data->m_localShapeAABBCPU = new b3AlignedObjectArray<b3SapAabb>;
	m_data->m_localShapeAABBGPU = new b3OpenCLArray<b3SapAabb>(ctx, queue, config.m_maxConvexShapes);

	//m_data->m_solverDataGPU = adl::Solver<adl::TYPE_CL>::allocate(ctx,queue, config.m_maxBroadphasePairs,false);
	m_data->m_bodyBufferGPU = new b3OpenCLArray<b3RigidBodyData>(ctx, queue, config.m_maxConvexBodies, false);

	m_data->m_convexFacesGPU = new b3OpenCLArray<b3GpuFace>(ctx, queue, config.m_maxConvexShapes * config.m_maxFacesPerShape, false);
	m_data->m_convexFaces.reserve(config.m_maxConvexShapes * config.m_maxFacesPerShape);

	m_data->m_gpuChildShapes = new b3OpenCLArray<b3GpuChildShape>(ctx, queue, config.m_maxCompoundChildShapes, false);

	m_data->m_convexPolyhedraGPU = new b3OpenCLArray<b3ConvexPolyhedronData>(ctx, queue, config.m_maxConvexShapes, false);
	m_data->m_convexPolyhedra.reserve(config.m_maxConvexShapes);

	m_data->m_uniqueEdgesGPU = new b3OpenCLArray<b3Vector3>(ctx, queue, config.m_maxConvexUniqueEdges, true);
	m_data->m_uniqueEdges.reserve(config.m_maxConvexUniqueEdges);

	m_data->m_convexVerticesGPU = new b3OpenCLArray<b3Vector3>(ctx, queue, config.m_maxConvexVertices, true);
	m_data->m_convexVertices.reserve(config.m_maxConvexVertices);

	m_data->m_convexIndicesGPU = new b3OpenCLArray<int>(ctx, queue, config.m_maxConvexIndices, true);
	m_data->m_convexIndices.reserve(config.m_maxConvexIndices);

	m_data->m_worldVertsB1GPU = new b3OpenCLArray<b3Vector3>(ctx, queue, config.m_maxConvexBodies * config.m_maxVerticesPerFace);
	m_data->m_clippingFacesOutGPU = new b3OpenCLArray<b3Int4>(ctx, queue, config.m_maxConvexBodies);
	m_data->m_worldNormalsAGPU = new b3OpenCLArray<b3Vector3>(ctx, queue, config.m_maxConvexBodies);
	m_data->m_worldVertsA1GPU = new b3OpenCLArray<b3Vector3>(ctx, queue, config.m_maxConvexBodies * config.m_maxVerticesPerFace);
	m_data->m_worldVertsB2GPU = new b3OpenCLArray<b3Vector3>(ctx, queue, config.m_maxConvexBodies * config.m_maxVerticesPerFace);

	m_data->m_convexData = new b3AlignedObjectArray<b3ConvexUtility*>();

	m_data->m_convexData->resize(config.m_maxConvexShapes);
	m_data->m_convexPolyhedra.resize(config.m_maxConvexShapes);

	m_data->m_numAcceleratedShapes = 0;
	m_data->m_numAcceleratedRigidBodies = 0;

	m_data->m_subTreesGPU = new b3OpenCLArray<b3BvhSubtreeInfo>(this->m_context, this->m_queue);
	m_data->m_treeNodesGPU = new b3OpenCLArray<b3QuantizedBvhNode>(this->m_context, this->m_queue);
	m_data->m_bvhInfoGPU = new b3OpenCLArray<b3BvhInfo>(this->m_context, this->m_queue);

	//m_data->m_contactCGPU = new b3OpenCLArray<Constraint4>(ctx,queue,config.m_maxBroadphasePairs,false);
	//m_data->m_frictionCGPU = new b3OpenCLArray<adl::Solver<adl::TYPE_CL>::allocateFrictionConstraint( m_data->m_deviceCL, config.m_maxBroadphasePairs);
}

b3GpuNarrowPhase::~b3GpuNarrowPhase()
{
	delete m_data->m_gpuSatCollision;

	delete m_data->m_triangleConvexPairs;
	//delete m_data->m_convexPairsOutGPU;
	//delete m_data->m_planePairs;
	delete m_data->m_pBufContactOutCPU;
	delete m_data->m_bodyBufferCPU;
	delete m_data->m_inertiaBufferCPU;
	delete m_data->m_pBufContactBuffersGPU[0];
	delete m_data->m_pBufContactBuffersGPU[1];

	delete m_data->m_inertiaBufferGPU;
	delete m_data->m_collidablesGPU;
	delete m_data->m_localShapeAABBCPU;
	delete m_data->m_localShapeAABBGPU;
	delete m_data->m_bodyBufferGPU;
	delete m_data->m_convexFacesGPU;
	delete m_data->m_gpuChildShapes;
	delete m_data->m_convexPolyhedraGPU;
	delete m_data->m_uniqueEdgesGPU;
	delete m_data->m_convexVerticesGPU;
	delete m_data->m_convexIndicesGPU;
	delete m_data->m_worldVertsB1GPU;
	delete m_data->m_clippingFacesOutGPU;
	delete m_data->m_worldNormalsAGPU;
	delete m_data->m_worldVertsA1GPU;
	delete m_data->m_worldVertsB2GPU;

	delete m_data->m_bvhInfoGPU;

	for (int i = 0; i < m_data->m_bvhData.size(); i++)
	{
		delete m_data->m_bvhData[i];
	}
	for (int i = 0; i < m_data->m_meshInterfaces.size(); i++)
	{
		delete m_data->m_meshInterfaces[i];
	}
	m_data->m_meshInterfaces.clear();
	m_data->m_bvhData.clear();
	delete m_data->m_treeNodesGPU;
	delete m_data->m_subTreesGPU;

	delete m_data->m_convexData;
	delete m_data;
}

int b3GpuNarrowPhase::allocateCollidable()
{
	int curSize = m_data->m_collidablesCPU.size();
	if (curSize < m_data->m_config.m_maxConvexShapes)
	{
		m_data->m_collidablesCPU.expand();
		return curSize;
	}
	else
	{
		b3Error("allocateCollidable out-of-range %d\n", m_data->m_config.m_maxConvexShapes);
	}
	return -1;
}

int b3GpuNarrowPhase::registerSphereShape(float radius)
{
	int collidableIndex = allocateCollidable();
	if (collidableIndex < 0)
		return collidableIndex;

	b3Collidable& col = getCollidableCpu(collidableIndex);
	col.m_shapeType = SHAPE_SPHERE;
	col.m_shapeIndex = 0;
	col.m_radius = radius;

	if (col.m_shapeIndex >= 0)
	{
		b3SapAabb aabb;
		b3Vector3 myAabbMin = b3MakeVector3(-radius, -radius, -radius);
		b3Vector3 myAabbMax = b3MakeVector3(radius, radius, radius);

		aabb.m_min[0] = myAabbMin[0];  //s_convexHeightField->m_aabb.m_min.x;
		aabb.m_min[1] = myAabbMin[1];  //s_convexHeightField->m_aabb.m_min.y;
		aabb.m_min[2] = myAabbMin[2];  //s_convexHeightField->m_aabb.m_min.z;
		aabb.m_minIndices[3] = 0;

		aabb.m_max[0] = myAabbMax[0];  //s_convexHeightField->m_aabb.m_max.x;
		aabb.m_max[1] = myAabbMax[1];  //s_convexHeightField->m_aabb.m_max.y;
		aabb.m_max[2] = myAabbMax[2];  //s_convexHeightField->m_aabb.m_max.z;
		aabb.m_signedMaxIndices[3] = 0;

		m_data->m_localShapeAABBCPU->push_back(aabb);
		//		m_data->m_localShapeAABBGPU->push_back(aabb);
		clFinish(m_queue);
	}

	return collidableIndex;
}

int b3GpuNarrowPhase::registerFace(const b3Vector3& faceNormal, float faceConstant)
{
	int faceOffset = m_data->m_convexFaces.size();
	b3GpuFace& face = m_data->m_convexFaces.expand();
	face.m_plane = b3MakeVector3(faceNormal.x, faceNormal.y, faceNormal.z, faceConstant);
	return faceOffset;
}

int b3GpuNarrowPhase::registerPlaneShape(const b3Vector3& planeNormal, float planeConstant)
{
	int collidableIndex = allocateCollidable();
	if (collidableIndex < 0)
		return collidableIndex;

	b3Collidable& col = getCollidableCpu(collidableIndex);
	col.m_shapeType = SHAPE_PLANE;
	col.m_shapeIndex = registerFace(planeNormal, planeConstant);
	col.m_radius = planeConstant;

	if (col.m_shapeIndex >= 0)
	{
		b3SapAabb aabb;
		aabb.m_min[0] = -1e30f;
		aabb.m_min[1] = -1e30f;
		aabb.m_min[2] = -1e30f;
		aabb.m_minIndices[3] = 0;

		aabb.m_max[0] = 1e30f;
		aabb.m_max[1] = 1e30f;
		aabb.m_max[2] = 1e30f;
		aabb.m_signedMaxIndices[3] = 0;

		m_data->m_localShapeAABBCPU->push_back(aabb);
		//		m_data->m_localShapeAABBGPU->push_back(aabb);
		clFinish(m_queue);
	}

	return collidableIndex;
}

int b3GpuNarrowPhase::registerConvexHullShapeInternal(b3ConvexUtility* convexPtr, b3Collidable& col)
{
	m_data->m_convexData->resize(m_data->m_numAcceleratedShapes + 1);
	m_data->m_convexPolyhedra.resize(m_data->m_numAcceleratedShapes + 1);

	b3ConvexPolyhedronData& convex = m_data->m_convexPolyhedra.at(m_data->m_convexPolyhedra.size() - 1);
	convex.mC = convexPtr->mC;
	convex.mE = convexPtr->mE;
	convex.m_extents = convexPtr->m_extents;
	convex.m_localCenter = convexPtr->m_localCenter;
	convex.m_radius = convexPtr->m_radius;

	convex.m_numUniqueEdges = convexPtr->m_uniqueEdges.size();
	int edgeOffset = m_data->m_uniqueEdges.size();
	convex.m_uniqueEdgesOffset = edgeOffset;

	m_data->m_uniqueEdges.resize(edgeOffset + convex.m_numUniqueEdges);

	//convex data here
	int i;
	for (i = 0; i < convexPtr->m_uniqueEdges.size(); i++)
	{
		m_data->m_uniqueEdges[edgeOffset + i] = convexPtr->m_uniqueEdges[i];
	}

	int faceOffset = m_data->m_convexFaces.size();
	convex.m_faceOffset = faceOffset;
	convex.m_numFaces = convexPtr->m_faces.size();

	m_data->m_convexFaces.resize(faceOffset + convex.m_numFaces);

	for (i = 0; i < convexPtr->m_faces.size(); i++)
	{
		m_data->m_convexFaces[convex.m_faceOffset + i].m_plane = b3MakeVector3(convexPtr->m_faces[i].m_plane[0],
																			   convexPtr->m_faces[i].m_plane[1],
																			   convexPtr->m_faces[i].m_plane[2],
																			   convexPtr->m_faces[i].m_plane[3]);

		int indexOffset = m_data->m_convexIndices.size();
		int numIndices = convexPtr->m_faces[i].m_indices.size();
		m_data->m_convexFaces[convex.m_faceOffset + i].m_numIndices = numIndices;
		m_data->m_convexFaces[convex.m_faceOffset + i].m_indexOffset = indexOffset;
		m_data->m_convexIndices.resize(indexOffset + numIndices);
		for (int p = 0; p < numIndices; p++)
		{
			m_data->m_convexIndices[indexOffset + p] = convexPtr->m_faces[i].m_indices[p];
		}
	}

	convex.m_numVertices = convexPtr->m_vertices.size();
	int vertexOffset = m_data->m_convexVertices.size();
	convex.m_vertexOffset = vertexOffset;

	m_data->m_convexVertices.resize(vertexOffset + convex.m_numVertices);
	for (int i = 0; i < convexPtr->m_vertices.size(); i++)
	{
		m_data->m_convexVertices[vertexOffset + i] = convexPtr->m_vertices[i];
	}

	(*m_data->m_convexData)[m_data->m_numAcceleratedShapes] = convexPtr;

	return m_data->m_numAcceleratedShapes++;
}

int b3GpuNarrowPhase::registerConvexHullShape(const float* vertices, int strideInBytes, int numVertices, const float* scaling)
{
	b3AlignedObjectArray<b3Vector3> verts;

	unsigned char* vts = (unsigned char*)vertices;
	for (int i = 0; i < numVertices; i++)
	{
		float* vertex = (float*)&vts[i * strideInBytes];
		verts.push_back(b3MakeVector3(vertex[0] * scaling[0], vertex[1] * scaling[1], vertex[2] * scaling[2]));
	}

	b3ConvexUtility* utilPtr = new b3ConvexUtility();
	bool merge = true;
	if (numVertices)
	{
		utilPtr->initializePolyhedralFeatures(&verts[0], verts.size(), merge);
	}

	int collidableIndex = registerConvexHullShape(utilPtr);
	delete utilPtr;
	return collidableIndex;
}

int b3GpuNarrowPhase::registerConvexHullShape(b3ConvexUtility* utilPtr)
{
	int collidableIndex = allocateCollidable();
	if (collidableIndex < 0)
		return collidableIndex;

	b3Collidable& col = getCollidableCpu(collidableIndex);
	col.m_shapeType = SHAPE_CONVEX_HULL;
	col.m_shapeIndex = -1;

	{
		b3Vector3 localCenter = b3MakeVector3(0, 0, 0);
		for (int i = 0; i < utilPtr->m_vertices.size(); i++)
			localCenter += utilPtr->m_vertices[i];
		localCenter *= (1.f / utilPtr->m_vertices.size());
		utilPtr->m_localCenter = localCenter;

		col.m_shapeIndex = registerConvexHullShapeInternal(utilPtr, col);
	}

	if (col.m_shapeIndex >= 0)
	{
		b3SapAabb aabb;

		b3Vector3 myAabbMin = b3MakeVector3(1e30f, 1e30f, 1e30f);
		b3Vector3 myAabbMax = b3MakeVector3(-1e30f, -1e30f, -1e30f);

		for (int i = 0; i < utilPtr->m_vertices.size(); i++)
		{
			myAabbMin.setMin(utilPtr->m_vertices[i]);
			myAabbMax.setMax(utilPtr->m_vertices[i]);
		}
		aabb.m_min[0] = myAabbMin[0];
		aabb.m_min[1] = myAabbMin[1];
		aabb.m_min[2] = myAabbMin[2];
		aabb.m_minIndices[3] = 0;

		aabb.m_max[0] = myAabbMax[0];
		aabb.m_max[1] = myAabbMax[1];
		aabb.m_max[2] = myAabbMax[2];
		aabb.m_signedMaxIndices[3] = 0;

		m_data->m_localShapeAABBCPU->push_back(aabb);
		//		m_data->m_localShapeAABBGPU->push_back(aabb);
	}

	return collidableIndex;
}

int b3GpuNarrowPhase::registerCompoundShape(b3AlignedObjectArray<b3GpuChildShape>* childShapes)
{
	int collidableIndex = allocateCollidable();
	if (collidableIndex < 0)
		return collidableIndex;

	b3Collidable& col = getCollidableCpu(collidableIndex);
	col.m_shapeType = SHAPE_COMPOUND_OF_CONVEX_HULLS;
	col.m_shapeIndex = m_data->m_cpuChildShapes.size();
	col.m_compoundBvhIndex = m_data->m_bvhInfoCPU.size();

	{
		b3Assert(col.m_shapeIndex + childShapes->size() < m_data->m_config.m_maxCompoundChildShapes);
		for (int i = 0; i < childShapes->size(); i++)
		{
			m_data->m_cpuChildShapes.push_back(childShapes->at(i));
		}
	}

	col.m_numChildShapes = childShapes->size();

	b3SapAabb aabbLocalSpace;
	b3Vector3 myAabbMin = b3MakeVector3(1e30f, 1e30f, 1e30f);
	b3Vector3 myAabbMax = b3MakeVector3(-1e30f, -1e30f, -1e30f);

	b3AlignedObjectArray<b3Aabb> childLocalAabbs;
	childLocalAabbs.resize(childShapes->size());

	//compute local AABB of the compound of all children
	for (int i = 0; i < childShapes->size(); i++)
	{
		int childColIndex = childShapes->at(i).m_shapeIndex;
		//b3Collidable& childCol = getCollidableCpu(childColIndex);
		b3SapAabb aabbLoc = m_data->m_localShapeAABBCPU->at(childColIndex);

		b3Vector3 childLocalAabbMin = b3MakeVector3(aabbLoc.m_min[0], aabbLoc.m_min[1], aabbLoc.m_min[2]);
		b3Vector3 childLocalAabbMax = b3MakeVector3(aabbLoc.m_max[0], aabbLoc.m_max[1], aabbLoc.m_max[2]);
		b3Vector3 aMin, aMax;
		b3Scalar margin(0.f);
		b3Transform childTr;
		childTr.setIdentity();

		childTr.setOrigin(childShapes->at(i).m_childPosition);
		childTr.setRotation(b3Quaternion(childShapes->at(i).m_childOrientation));
		b3TransformAabb(childLocalAabbMin, childLocalAabbMax, margin, childTr, aMin, aMax);
		myAabbMin.setMin(aMin);
		myAabbMax.setMax(aMax);
		childLocalAabbs[i].m_min[0] = aMin[0];
		childLocalAabbs[i].m_min[1] = aMin[1];
		childLocalAabbs[i].m_min[2] = aMin[2];
		childLocalAabbs[i].m_min[3] = 0;
		childLocalAabbs[i].m_max[0] = aMax[0];
		childLocalAabbs[i].m_max[1] = aMax[1];
		childLocalAabbs[i].m_max[2] = aMax[2];
		childLocalAabbs[i].m_max[3] = 0;
	}

	aabbLocalSpace.m_min[0] = myAabbMin[0];  //s_convexHeightField->m_aabb.m_min.x;
	aabbLocalSpace.m_min[1] = myAabbMin[1];  //s_convexHeightField->m_aabb.m_min.y;
	aabbLocalSpace.m_min[2] = myAabbMin[2];  //s_convexHeightField->m_aabb.m_min.z;
	aabbLocalSpace.m_minIndices[3] = 0;

	aabbLocalSpace.m_max[0] = myAabbMax[0];  //s_convexHeightField->m_aabb.m_max.x;
	aabbLocalSpace.m_max[1] = myAabbMax[1];  //s_convexHeightField->m_aabb.m_max.y;
	aabbLocalSpace.m_max[2] = myAabbMax[2];  //s_convexHeightField->m_aabb.m_max.z;
	aabbLocalSpace.m_signedMaxIndices[3] = 0;

	m_data->m_localShapeAABBCPU->push_back(aabbLocalSpace);

	b3QuantizedBvh* bvh = new b3QuantizedBvh;
	bvh->setQuantizationValues(myAabbMin, myAabbMax);
	QuantizedNodeArray& nodes = bvh->getLeafNodeArray();
	int numNodes = childShapes->size();

	for (int i = 0; i < numNodes; i++)
	{
		b3QuantizedBvhNode node;
		b3Vector3 aabbMin, aabbMax;
		aabbMin = (b3Vector3&)childLocalAabbs[i].m_min;
		aabbMax = (b3Vector3&)childLocalAabbs[i].m_max;

		bvh->quantize(&node.m_quantizedAabbMin[0], aabbMin, 0);
		bvh->quantize(&node.m_quantizedAabbMax[0], aabbMax, 1);
		int partId = 0;
		node.m_escapeIndexOrTriangleIndex = (partId << (31 - MAX_NUM_PARTS_IN_BITS)) | i;
		nodes.push_back(node);
	}
	bvh->buildInternal();

	int numSubTrees = bvh->getSubtreeInfoArray().size();

	//void	setQuantizationValues(const b3Vector3& bvhAabbMin,const b3Vector3& bvhAabbMax,b3Scalar quantizationMargin=b3Scalar(1.0));
	//QuantizedNodeArray&	getLeafNodeArray() {			return	m_quantizedLeafNodes;	}
	///buildInternal is expert use only: assumes that setQuantizationValues and LeafNodeArray are initialized
	//void	buildInternal();

	b3BvhInfo bvhInfo;

	bvhInfo.m_aabbMin = bvh->m_bvhAabbMin;
	bvhInfo.m_aabbMax = bvh->m_bvhAabbMax;
	bvhInfo.m_quantization = bvh->m_bvhQuantization;
	bvhInfo.m_numNodes = numNodes;
	bvhInfo.m_numSubTrees = numSubTrees;
	bvhInfo.m_nodeOffset = m_data->m_treeNodesCPU.size();
	bvhInfo.m_subTreeOffset = m_data->m_subTreesCPU.size();

	int numNewNodes = bvh->getQuantizedNodeArray().size();

	for (int i = 0; i < numNewNodes - 1; i++)
	{
		if (bvh->getQuantizedNodeArray()[i].isLeafNode())
		{
			int orgIndex = bvh->getQuantizedNodeArray()[i].getTriangleIndex();

			b3Vector3 nodeMinVec = bvh->unQuantize(bvh->getQuantizedNodeArray()[i].m_quantizedAabbMin);
			b3Vector3 nodeMaxVec = bvh->unQuantize(bvh->getQuantizedNodeArray()[i].m_quantizedAabbMax);

			for (int c = 0; c < 3; c++)
			{
				if (childLocalAabbs[orgIndex].m_min[c] < nodeMinVec[c])
				{
					printf("min org (%f) and new (%f) ? at i:%d,c:%d\n", childLocalAabbs[i].m_min[c], nodeMinVec[c], i, c);
				}
				if (childLocalAabbs[orgIndex].m_max[c] > nodeMaxVec[c])
				{
					printf("max org (%f) and new (%f) ? at i:%d,c:%d\n", childLocalAabbs[i].m_max[c], nodeMaxVec[c], i, c);
				}
			}
		}
	}

	m_data->m_bvhInfoCPU.push_back(bvhInfo);

	int numNewSubtrees = bvh->getSubtreeInfoArray().size();
	m_data->m_subTreesCPU.reserve(m_data->m_subTreesCPU.size() + numNewSubtrees);
	for (int i = 0; i < numNewSubtrees; i++)
	{
		m_data->m_subTreesCPU.push_back(bvh->getSubtreeInfoArray()[i]);
	}
	int numNewTreeNodes = bvh->getQuantizedNodeArray().size();

	for (int i = 0; i < numNewTreeNodes; i++)
	{
		m_data->m_treeNodesCPU.push_back(bvh->getQuantizedNodeArray()[i]);
	}

	//	m_data->m_localShapeAABBGPU->push_back(aabbWS);
	clFinish(m_queue);
	return collidableIndex;
}

int b3GpuNarrowPhase::registerConcaveMesh(b3AlignedObjectArray<b3Vector3>* vertices, b3AlignedObjectArray<int>* indices, const float* scaling1)
{
	b3Vector3 scaling = b3MakeVector3(scaling1[0], scaling1[1], scaling1[2]);

	int collidableIndex = allocateCollidable();
	if (collidableIndex < 0)
		return collidableIndex;

	b3Collidable& col = getCollidableCpu(collidableIndex);

	col.m_shapeType = SHAPE_CONCAVE_TRIMESH;
	col.m_shapeIndex = registerConcaveMeshShape(vertices, indices, col, scaling);
	col.m_bvhIndex = m_data->m_bvhInfoCPU.size();

	b3SapAabb aabb;
	b3Vector3 myAabbMin = b3MakeVector3(1e30f, 1e30f, 1e30f);
	b3Vector3 myAabbMax = b3MakeVector3(-1e30f, -1e30f, -1e30f);

	for (int i = 0; i < vertices->size(); i++)
	{
		b3Vector3 vtx(vertices->at(i) * scaling);
		myAabbMin.setMin(vtx);
		myAabbMax.setMax(vtx);
	}
	aabb.m_min[0] = myAabbMin[0];
	aabb.m_min[1] = myAabbMin[1];
	aabb.m_min[2] = myAabbMin[2];
	aabb.m_minIndices[3] = 0;

	aabb.m_max[0] = myAabbMax[0];
	aabb.m_max[1] = myAabbMax[1];
	aabb.m_max[2] = myAabbMax[2];
	aabb.m_signedMaxIndices[3] = 0;

	m_data->m_localShapeAABBCPU->push_back(aabb);
	//	m_data->m_localShapeAABBGPU->push_back(aabb);

	b3OptimizedBvh* bvh = new b3OptimizedBvh();
	//void b3OptimizedBvh::build(b3StridingMeshInterface* triangles, bool useQuantizedAabbCompression, const b3Vector3& bvhAabbMin, const b3Vector3& bvhAabbMax)

	bool useQuantizedAabbCompression = true;
	b3TriangleIndexVertexArray* meshInterface = new b3TriangleIndexVertexArray();
	m_data->m_meshInterfaces.push_back(meshInterface);
	b3IndexedMesh mesh;
	mesh.m_numTriangles = indices->size() / 3;
	mesh.m_numVertices = vertices->size();
	mesh.m_vertexBase = (const unsigned char*)&vertices->at(0).x;
	mesh.m_vertexStride = sizeof(b3Vector3);
	mesh.m_triangleIndexStride = 3 * sizeof(int);  // or sizeof(int)
	mesh.m_triangleIndexBase = (const unsigned char*)&indices->at(0);

	meshInterface->addIndexedMesh(mesh);
	bvh->build(meshInterface, useQuantizedAabbCompression, (b3Vector3&)aabb.m_min, (b3Vector3&)aabb.m_max);
	m_data->m_bvhData.push_back(bvh);
	int numNodes = bvh->getQuantizedNodeArray().size();
	//b3OpenCLArray<b3QuantizedBvhNode>*	treeNodesGPU = new b3OpenCLArray<b3QuantizedBvhNode>(this->m_context,this->m_queue,numNodes);
	int numSubTrees = bvh->getSubtreeInfoArray().size();

	b3BvhInfo bvhInfo;

	bvhInfo.m_aabbMin = bvh->m_bvhAabbMin;
	bvhInfo.m_aabbMax = bvh->m_bvhAabbMax;
	bvhInfo.m_quantization = bvh->m_bvhQuantization;
	bvhInfo.m_numNodes = numNodes;
	bvhInfo.m_numSubTrees = numSubTrees;
	bvhInfo.m_nodeOffset = m_data->m_treeNodesCPU.size();
	bvhInfo.m_subTreeOffset = m_data->m_subTreesCPU.size();

	m_data->m_bvhInfoCPU.push_back(bvhInfo);

	int numNewSubtrees = bvh->getSubtreeInfoArray().size();
	m_data->m_subTreesCPU.reserve(m_data->m_subTreesCPU.size() + numNewSubtrees);
	for (int i = 0; i < numNewSubtrees; i++)
	{
		m_data->m_subTreesCPU.push_back(bvh->getSubtreeInfoArray()[i]);
	}
	int numNewTreeNodes = bvh->getQuantizedNodeArray().size();

	for (int i = 0; i < numNewTreeNodes; i++)
	{
		m_data->m_treeNodesCPU.push_back(bvh->getQuantizedNodeArray()[i]);
	}

	return collidableIndex;
}

int b3GpuNarrowPhase::registerConcaveMeshShape(b3AlignedObjectArray<b3Vector3>* vertices, b3AlignedObjectArray<int>* indices, b3Collidable& col, const float* scaling1)
{
	b3Vector3 scaling = b3MakeVector3(scaling1[0], scaling1[1], scaling1[2]);

	m_data->m_convexData->resize(m_data->m_numAcceleratedShapes + 1);
	m_data->m_convexPolyhedra.resize(m_data->m_numAcceleratedShapes + 1);

	b3ConvexPolyhedronData& convex = m_data->m_convexPolyhedra.at(m_data->m_convexPolyhedra.size() - 1);
	convex.mC = b3MakeVector3(0, 0, 0);
	convex.mE = b3MakeVector3(0, 0, 0);
	convex.m_extents = b3MakeVector3(0, 0, 0);
	convex.m_localCenter = b3MakeVector3(0, 0, 0);
	convex.m_radius = 0.f;

	convex.m_numUniqueEdges = 0;
	int edgeOffset = m_data->m_uniqueEdges.size();
	convex.m_uniqueEdgesOffset = edgeOffset;

	int faceOffset = m_data->m_convexFaces.size();
	convex.m_faceOffset = faceOffset;

	convex.m_numFaces = indices->size() / 3;
	m_data->m_convexFaces.resize(faceOffset + convex.m_numFaces);
	m_data->m_convexIndices.reserve(convex.m_numFaces * 3);
	for (int i = 0; i < convex.m_numFaces; i++)
	{
		if (i % 256 == 0)
		{
			//printf("i=%d out of %d", i,convex.m_numFaces);
		}
		b3Vector3 vert0(vertices->at(indices->at(i * 3)) * scaling);
		b3Vector3 vert1(vertices->at(indices->at(i * 3 + 1)) * scaling);
		b3Vector3 vert2(vertices->at(indices->at(i * 3 + 2)) * scaling);

		b3Vector3 normal = ((vert1 - vert0).cross(vert2 - vert0)).normalize();
		b3Scalar c = -(normal.dot(vert0));

		m_data->m_convexFaces[convex.m_faceOffset + i].m_plane = b3MakeVector4(normal.x, normal.y, normal.z, c);
		int indexOffset = m_data->m_convexIndices.size();
		int numIndices = 3;
		m_data->m_convexFaces[convex.m_faceOffset + i].m_numIndices = numIndices;
		m_data->m_convexFaces[convex.m_faceOffset + i].m_indexOffset = indexOffset;
		m_data->m_convexIndices.resize(indexOffset + numIndices);
		for (int p = 0; p < numIndices; p++)
		{
			int vi = indices->at(i * 3 + p);
			m_data->m_convexIndices[indexOffset + p] = vi;  //convexPtr->m_faces[i].m_indices[p];
		}
	}

	convex.m_numVertices = vertices->size();
	int vertexOffset = m_data->m_convexVertices.size();
	convex.m_vertexOffset = vertexOffset;
	m_data->m_convexVertices.resize(vertexOffset + convex.m_numVertices);
	for (int i = 0; i < vertices->size(); i++)
	{
		m_data->m_convexVertices[vertexOffset + i] = vertices->at(i) * scaling;
	}

	(*m_data->m_convexData)[m_data->m_numAcceleratedShapes] = 0;

	return m_data->m_numAcceleratedShapes++;
}

cl_mem b3GpuNarrowPhase::getBodiesGpu()
{
	return (cl_mem)m_data->m_bodyBufferGPU->getBufferCL();
}

const struct b3RigidBodyData* b3GpuNarrowPhase::getBodiesCpu() const
{
	return &m_data->m_bodyBufferCPU->at(0);
};

int b3GpuNarrowPhase::getNumBodiesGpu() const
{
	return m_data->m_bodyBufferGPU->size();
}

cl_mem b3GpuNarrowPhase::getBodyInertiasGpu()
{
	return (cl_mem)m_data->m_inertiaBufferGPU->getBufferCL();
}

int b3GpuNarrowPhase::getNumBodyInertiasGpu() const
{
	return m_data->m_inertiaBufferGPU->size();
}

b3Collidable& b3GpuNarrowPhase::getCollidableCpu(int collidableIndex)
{
	return m_data->m_collidablesCPU[collidableIndex];
}

const b3Collidable& b3GpuNarrowPhase::getCollidableCpu(int collidableIndex) const
{
	return m_data->m_collidablesCPU[collidableIndex];
}

cl_mem b3GpuNarrowPhase::getCollidablesGpu()
{
	return m_data->m_collidablesGPU->getBufferCL();
}

const struct b3Collidable* b3GpuNarrowPhase::getCollidablesCpu() const
{
	if (m_data->m_collidablesCPU.size())
		return &m_data->m_collidablesCPU[0];
	return 0;
}

const struct b3SapAabb* b3GpuNarrowPhase::getLocalSpaceAabbsCpu() const
{
	if (m_data->m_localShapeAABBCPU->size())
	{
		return &m_data->m_localShapeAABBCPU->at(0);
	}
	return 0;
}

cl_mem b3GpuNarrowPhase::getAabbLocalSpaceBufferGpu()
{
	return m_data->m_localShapeAABBGPU->getBufferCL();
}
int b3GpuNarrowPhase::getNumCollidablesGpu() const
{
	return m_data->m_collidablesGPU->size();
}

int b3GpuNarrowPhase::getNumContactsGpu() const
{
	return m_data->m_pBufContactBuffersGPU[m_data->m_currentContactBuffer]->size();
}
cl_mem b3GpuNarrowPhase::getContactsGpu()
{
	return m_data->m_pBufContactBuffersGPU[m_data->m_currentContactBuffer]->getBufferCL();
}

const b3Contact4* b3GpuNarrowPhase::getContactsCPU() const
{
	m_data->m_pBufContactBuffersGPU[m_data->m_currentContactBuffer]->copyToHost(*m_data->m_pBufContactOutCPU);
	return &m_data->m_pBufContactOutCPU->at(0);
}

void b3GpuNarrowPhase::computeContacts(cl_mem broadphasePairs, int numBroadphasePairs, cl_mem aabbsWorldSpace, int numObjects)
{
	cl_mem aabbsLocalSpace = m_data->m_localShapeAABBGPU->getBufferCL();

	int nContactOut = 0;

	//swap buffer
	m_data->m_currentContactBuffer = 1 - m_data->m_currentContactBuffer;

	//int curSize = m_data->m_pBufContactBuffersGPU[m_data->m_currentContactBuffer]->size();

	int maxTriConvexPairCapacity = m_data->m_config.m_maxTriConvexPairCapacity;
	int numTriConvexPairsOut = 0;

	b3OpenCLArray<b3Int4> broadphasePairsGPU(m_context, m_queue);
	broadphasePairsGPU.setFromOpenCLBuffer(broadphasePairs, numBroadphasePairs);

	b3OpenCLArray<b3Aabb> clAabbArrayWorldSpace(this->m_context, this->m_queue);
	clAabbArrayWorldSpace.setFromOpenCLBuffer(aabbsWorldSpace, numObjects);

	b3OpenCLArray<b3Aabb> clAabbArrayLocalSpace(this->m_context, this->m_queue);
	clAabbArrayLocalSpace.setFromOpenCLBuffer(aabbsLocalSpace, numObjects);

	m_data->m_gpuSatCollision->computeConvexConvexContactsGPUSAT(
		&broadphasePairsGPU, numBroadphasePairs,
		m_data->m_bodyBufferGPU,
		m_data->m_pBufContactBuffersGPU[m_data->m_currentContactBuffer],
		nContactOut,
		m_data->m_pBufContactBuffersGPU[1 - m_data->m_currentContactBuffer],
		m_data->m_config.m_maxContactCapacity,
		m_data->m_config.m_compoundPairCapacity,
		*m_data->m_convexPolyhedraGPU,
		*m_data->m_convexVerticesGPU,
		*m_data->m_uniqueEdgesGPU,
		*m_data->m_convexFacesGPU,
		*m_data->m_convexIndicesGPU,
		*m_data->m_collidablesGPU,
		*m_data->m_gpuChildShapes,
		clAabbArrayWorldSpace,
		clAabbArrayLocalSpace,
		*m_data->m_worldVertsB1GPU,
		*m_data->m_clippingFacesOutGPU,
		*m_data->m_worldNormalsAGPU,
		*m_data->m_worldVertsA1GPU,
		*m_data->m_worldVertsB2GPU,
		m_data->m_bvhData,
		m_data->m_treeNodesGPU,
		m_data->m_subTreesGPU,
		m_data->m_bvhInfoGPU,
		numObjects,
		maxTriConvexPairCapacity,
		*m_data->m_triangleConvexPairs,
		numTriConvexPairsOut);

	/*b3AlignedObjectArray<b3Int4> broadphasePairsCPU;
	broadphasePairsGPU.copyToHost(broadphasePairsCPU);
	printf("checking pairs\n");
	*/
}

const b3SapAabb& b3GpuNarrowPhase::getLocalSpaceAabb(int collidableIndex) const
{
	return m_data->m_localShapeAABBCPU->at(collidableIndex);
}

int b3GpuNarrowPhase::registerRigidBody(int collidableIndex, float mass, const float* position, const float* orientation, const float* aabbMinPtr, const float* aabbMaxPtr, bool writeToGpu)
{
	b3Vector3 aabbMin = b3MakeVector3(aabbMinPtr[0], aabbMinPtr[1], aabbMinPtr[2]);
	b3Vector3 aabbMax = b3MakeVector3(aabbMaxPtr[0], aabbMaxPtr[1], aabbMaxPtr[2]);

	if (m_data->m_numAcceleratedRigidBodies >= (m_data->m_config.m_maxConvexBodies))
	{
		b3Error("registerRigidBody: exceeding the number of rigid bodies, %d > %d \n", m_data->m_numAcceleratedRigidBodies, m_data->m_config.m_maxConvexBodies);
		return -1;
	}

	m_data->m_bodyBufferCPU->resize(m_data->m_numAcceleratedRigidBodies + 1);

	b3RigidBodyData& body = m_data->m_bodyBufferCPU->at(m_data->m_numAcceleratedRigidBodies);

	float friction = 1.f;
	float restitution = 0.f;

	body.m_frictionCoeff = friction;
	body.m_restituitionCoeff = restitution;
	body.m_angVel = b3MakeVector3(0, 0, 0);
	body.m_linVel = b3MakeVector3(0, 0, 0);  //.setZero();
	body.m_pos = b3MakeVector3(position[0], position[1], position[2]);
	body.m_quat.setValue(orientation[0], orientation[1], orientation[2], orientation[3]);
	body.m_collidableIdx = collidableIndex;
	if (collidableIndex >= 0)
	{
		//		body.m_shapeType = m_data->m_collidablesCPU.at(collidableIndex).m_shapeType;
	}
	else
	{
		//	body.m_shapeType = CollisionShape::SHAPE_PLANE;
		m_planeBodyIndex = m_data->m_numAcceleratedRigidBodies;
	}
	//body.m_shapeType = shapeType;

	body.m_invMass = mass ? 1.f / mass : 0.f;

	if (writeToGpu)
	{
		m_data->m_bodyBufferGPU->copyFromHostPointer(&body, 1, m_data->m_numAcceleratedRigidBodies);
	}

	b3InertiaData& shapeInfo = m_data->m_inertiaBufferCPU->at(m_data->m_numAcceleratedRigidBodies);

	if (mass == 0.f)
	{
		if (m_data->m_numAcceleratedRigidBodies == 0)
			m_static0Index = 0;

		shapeInfo.m_initInvInertia.setValue(0, 0, 0, 0, 0, 0, 0, 0, 0);
		shapeInfo.m_invInertiaWorld.setValue(0, 0, 0, 0, 0, 0, 0, 0, 0);
	}
	else
	{
		b3Assert(body.m_collidableIdx >= 0);

		//approximate using the aabb of the shape

		//Aabb aabb = (*m_data->m_shapePointers)[shapeIndex]->m_aabb;
		b3Vector3 halfExtents = (aabbMax - aabbMin);  //*0.5f;//fake larger inertia makes demos more stable ;-)

		b3Vector3 localInertia;

		float lx = 2.f * halfExtents[0];
		float ly = 2.f * halfExtents[1];
		float lz = 2.f * halfExtents[2];

		localInertia.setValue((mass / 12.0f) * (ly * ly + lz * lz),
							  (mass / 12.0f) * (lx * lx + lz * lz),
							  (mass / 12.0f) * (lx * lx + ly * ly));

		b3Vector3 invLocalInertia;
		invLocalInertia[0] = 1.f / localInertia[0];
		invLocalInertia[1] = 1.f / localInertia[1];
		invLocalInertia[2] = 1.f / localInertia[2];
		invLocalInertia[3] = 0.f;

		shapeInfo.m_initInvInertia.setValue(
			invLocalInertia[0], 0, 0,
			0, invLocalInertia[1], 0,
			0, 0, invLocalInertia[2]);

		b3Matrix3x3 m(body.m_quat);

		shapeInfo.m_invInertiaWorld = m.scaled(invLocalInertia) * m.transpose();
	}

	if (writeToGpu)
		m_data->m_inertiaBufferGPU->copyFromHostPointer(&shapeInfo, 1, m_data->m_numAcceleratedRigidBodies);

	return m_data->m_numAcceleratedRigidBodies++;
}

int b3GpuNarrowPhase::getNumRigidBodies() const
{
	return m_data->m_numAcceleratedRigidBodies;
}

void b3GpuNarrowPhase::writeAllBodiesToGpu()
{
	if (m_data->m_localShapeAABBCPU->size())
	{
		m_data->m_localShapeAABBGPU->copyFromHost(*m_data->m_localShapeAABBCPU);
	}

	m_data->m_gpuChildShapes->copyFromHost(m_data->m_cpuChildShapes);
	m_data->m_convexFacesGPU->copyFromHost(m_data->m_convexFaces);
	m_data->m_convexPolyhedraGPU->copyFromHost(m_data->m_convexPolyhedra);
	m_data->m_uniqueEdgesGPU->copyFromHost(m_data->m_uniqueEdges);
	m_data->m_convexVerticesGPU->copyFromHost(m_data->m_convexVertices);
	m_data->m_convexIndicesGPU->copyFromHost(m_data->m_convexIndices);
	m_data->m_bvhInfoGPU->copyFromHost(m_data->m_bvhInfoCPU);
	m_data->m_treeNodesGPU->copyFromHost(m_data->m_treeNodesCPU);
	m_data->m_subTreesGPU->copyFromHost(m_data->m_subTreesCPU);

	m_data->m_bodyBufferGPU->resize(m_data->m_numAcceleratedRigidBodies);
	m_data->m_inertiaBufferGPU->resize(m_data->m_numAcceleratedRigidBodies);

	if (m_data->m_numAcceleratedRigidBodies)
	{
		m_data->m_bodyBufferGPU->copyFromHostPointer(&m_data->m_bodyBufferCPU->at(0), m_data->m_numAcceleratedRigidBodies);
		m_data->m_inertiaBufferGPU->copyFromHostPointer(&m_data->m_inertiaBufferCPU->at(0), m_data->m_numAcceleratedRigidBodies);
	}
	if (m_data->m_collidablesCPU.size())
	{
		m_data->m_collidablesGPU->copyFromHost(m_data->m_collidablesCPU);
	}
}

void b3GpuNarrowPhase::reset()
{
	m_data->m_numAcceleratedShapes = 0;
	m_data->m_numAcceleratedRigidBodies = 0;
	this->m_static0Index = -1;
	m_data->m_uniqueEdges.resize(0);
	m_data->m_convexVertices.resize(0);
	m_data->m_convexPolyhedra.resize(0);
	m_data->m_convexIndices.resize(0);
	m_data->m_cpuChildShapes.resize(0);
	m_data->m_convexFaces.resize(0);
	m_data->m_collidablesCPU.resize(0);
	m_data->m_localShapeAABBCPU->resize(0);
	m_data->m_bvhData.resize(0);
	m_data->m_treeNodesCPU.resize(0);
	m_data->m_subTreesCPU.resize(0);
	m_data->m_bvhInfoCPU.resize(0);
}

void b3GpuNarrowPhase::readbackAllBodiesToCpu()
{
	m_data->m_bodyBufferGPU->copyToHostPointer(&m_data->m_bodyBufferCPU->at(0), m_data->m_numAcceleratedRigidBodies);
}

void b3GpuNarrowPhase::setObjectTransformCpu(float* position, float* orientation, int bodyIndex)
{
	if (bodyIndex >= 0 && bodyIndex < m_data->m_bodyBufferCPU->size())
	{
		m_data->m_bodyBufferCPU->at(bodyIndex).m_pos = b3MakeVector3(position[0], position[1], position[2]);
		m_data->m_bodyBufferCPU->at(bodyIndex).m_quat.setValue(orientation[0], orientation[1], orientation[2], orientation[3]);
	}
	else
	{
		b3Warning("setObjectVelocityCpu out of range.\n");
	}
}
void b3GpuNarrowPhase::setObjectVelocityCpu(float* linVel, float* angVel, int bodyIndex)
{
	if (bodyIndex >= 0 && bodyIndex < m_data->m_bodyBufferCPU->size())
	{
		m_data->m_bodyBufferCPU->at(bodyIndex).m_linVel = b3MakeVector3(linVel[0], linVel[1], linVel[2]);
		m_data->m_bodyBufferCPU->at(bodyIndex).m_angVel = b3MakeVector3(angVel[0], angVel[1], angVel[2]);
	}
	else
	{
		b3Warning("setObjectVelocityCpu out of range.\n");
	}
}

bool b3GpuNarrowPhase::getObjectTransformFromCpu(float* position, float* orientation, int bodyIndex) const
{
	if (bodyIndex >= 0 && bodyIndex < m_data->m_bodyBufferCPU->size())
	{
		position[0] = m_data->m_bodyBufferCPU->at(bodyIndex).m_pos.x;
		position[1] = m_data->m_bodyBufferCPU->at(bodyIndex).m_pos.y;
		position[2] = m_data->m_bodyBufferCPU->at(bodyIndex).m_pos.z;
		position[3] = 1.f;  //or 1

		orientation[0] = m_data->m_bodyBufferCPU->at(bodyIndex).m_quat.x;
		orientation[1] = m_data->m_bodyBufferCPU->at(bodyIndex).m_quat.y;
		orientation[2] = m_data->m_bodyBufferCPU->at(bodyIndex).m_quat.z;
		orientation[3] = m_data->m_bodyBufferCPU->at(bodyIndex).m_quat.w;
		return true;
	}

	b3Warning("getObjectTransformFromCpu out of range.\n");
	return false;
}
