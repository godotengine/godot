
#ifndef B3_GPU_NARROWPHASE_INTERNAL_DATA_H
#define B3_GPU_NARROWPHASE_INTERNAL_DATA_H

#include "Bullet3OpenCL/ParallelPrimitives/b3OpenCLArray.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3ConvexPolyhedronData.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3Config.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Collidable.h"

#include "Bullet3OpenCL/Initialize/b3OpenCLInclude.h"
#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3Common/b3Vector3.h"

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3Contact4.h"
#include "Bullet3OpenCL/BroadphaseCollision/b3SapAabb.h"

#include "Bullet3OpenCL/NarrowphaseCollision/b3QuantizedBvh.h"
#include "Bullet3OpenCL/NarrowphaseCollision/b3BvhInfo.h"
#include "Bullet3Common/shared/b3Int4.h"
#include "Bullet3Common/shared/b3Int2.h"

class b3ConvexUtility;

struct b3GpuNarrowPhaseInternalData
{
	b3AlignedObjectArray<b3ConvexUtility*>* m_convexData;

	b3AlignedObjectArray<b3ConvexPolyhedronData> m_convexPolyhedra;
	b3AlignedObjectArray<b3Vector3> m_uniqueEdges;
	b3AlignedObjectArray<b3Vector3> m_convexVertices;
	b3AlignedObjectArray<int> m_convexIndices;

	b3OpenCLArray<b3ConvexPolyhedronData>* m_convexPolyhedraGPU;
	b3OpenCLArray<b3Vector3>* m_uniqueEdgesGPU;
	b3OpenCLArray<b3Vector3>* m_convexVerticesGPU;
	b3OpenCLArray<int>* m_convexIndicesGPU;

	b3OpenCLArray<b3Vector3>* m_worldVertsB1GPU;
	b3OpenCLArray<b3Int4>* m_clippingFacesOutGPU;
	b3OpenCLArray<b3Vector3>* m_worldNormalsAGPU;
	b3OpenCLArray<b3Vector3>* m_worldVertsA1GPU;
	b3OpenCLArray<b3Vector3>* m_worldVertsB2GPU;

	b3AlignedObjectArray<b3GpuChildShape> m_cpuChildShapes;
	b3OpenCLArray<b3GpuChildShape>* m_gpuChildShapes;

	b3AlignedObjectArray<b3GpuFace> m_convexFaces;
	b3OpenCLArray<b3GpuFace>* m_convexFacesGPU;

	struct GpuSatCollision* m_gpuSatCollision;

	b3OpenCLArray<b3Int4>* m_triangleConvexPairs;

	b3OpenCLArray<b3Contact4>* m_pBufContactBuffersGPU[2];
	int m_currentContactBuffer;
	b3AlignedObjectArray<b3Contact4>* m_pBufContactOutCPU;

	b3AlignedObjectArray<b3RigidBodyData>* m_bodyBufferCPU;
	b3OpenCLArray<b3RigidBodyData>* m_bodyBufferGPU;

	b3AlignedObjectArray<b3InertiaData>* m_inertiaBufferCPU;
	b3OpenCLArray<b3InertiaData>* m_inertiaBufferGPU;

	int m_numAcceleratedShapes;
	int m_numAcceleratedRigidBodies;

	b3AlignedObjectArray<b3Collidable> m_collidablesCPU;
	b3OpenCLArray<b3Collidable>* m_collidablesGPU;

	b3OpenCLArray<b3SapAabb>* m_localShapeAABBGPU;
	b3AlignedObjectArray<b3SapAabb>* m_localShapeAABBCPU;

	b3AlignedObjectArray<class b3OptimizedBvh*> m_bvhData;
	b3AlignedObjectArray<class b3TriangleIndexVertexArray*> m_meshInterfaces;

	b3AlignedObjectArray<b3QuantizedBvhNode> m_treeNodesCPU;
	b3AlignedObjectArray<b3BvhSubtreeInfo> m_subTreesCPU;

	b3AlignedObjectArray<b3BvhInfo> m_bvhInfoCPU;
	b3OpenCLArray<b3BvhInfo>* m_bvhInfoGPU;

	b3OpenCLArray<b3QuantizedBvhNode>* m_treeNodesGPU;
	b3OpenCLArray<b3BvhSubtreeInfo>* m_subTreesGPU;

	b3Config m_config;
};

#endif  //B3_GPU_NARROWPHASE_INTERNAL_DATA_H
