
#ifndef _CONVEX_HULL_CONTACT_H
#define _CONVEX_HULL_CONTACT_H

#include "Bullet3OpenCL/ParallelPrimitives/b3OpenCLArray.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"
#include "Bullet3Common/b3AlignedObjectArray.h"

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3ConvexPolyhedronData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Collidable.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3Contact4.h"
#include "Bullet3Common/shared/b3Int2.h"
#include "Bullet3Common/shared/b3Int4.h"
#include "b3OptimizedBvh.h"
#include "b3BvhInfo.h"
#include "Bullet3Collision/BroadPhaseCollision/shared/b3Aabb.h"

//#include "../../dynamics/basic_demo/Stubs/ChNarrowPhase.h"

struct GpuSatCollision
{
	cl_context m_context;
	cl_device_id m_device;
	cl_command_queue m_queue;
	cl_kernel m_findSeparatingAxisKernel;
	cl_kernel m_mprPenetrationKernel;
	cl_kernel m_findSeparatingAxisUnitSphereKernel;

	cl_kernel m_findSeparatingAxisVertexFaceKernel;
	cl_kernel m_findSeparatingAxisEdgeEdgeKernel;

	cl_kernel m_findConcaveSeparatingAxisKernel;
	cl_kernel m_findConcaveSeparatingAxisVertexFaceKernel;
	cl_kernel m_findConcaveSeparatingAxisEdgeEdgeKernel;

	cl_kernel m_findCompoundPairsKernel;
	cl_kernel m_processCompoundPairsKernel;

	cl_kernel m_clipHullHullKernel;
	cl_kernel m_clipCompoundsHullHullKernel;

	cl_kernel m_clipFacesAndFindContacts;
	cl_kernel m_findClippingFacesKernel;

	cl_kernel m_clipHullHullConcaveConvexKernel;
	//	cl_kernel				m_extractManifoldAndAddContactKernel;
	cl_kernel m_newContactReductionKernel;

	cl_kernel m_bvhTraversalKernel;
	cl_kernel m_primitiveContactsKernel;
	cl_kernel m_findConcaveSphereContactsKernel;

	cl_kernel m_processCompoundPairsPrimitivesKernel;

	b3OpenCLArray<b3Vector3> m_unitSphereDirections;

	b3OpenCLArray<int> m_totalContactsOut;

	b3OpenCLArray<b3Vector3> m_sepNormals;
	b3OpenCLArray<float> m_dmins;

	b3OpenCLArray<int> m_hasSeparatingNormals;
	b3OpenCLArray<b3Vector3> m_concaveSepNormals;
	b3OpenCLArray<int> m_concaveHasSeparatingNormals;
	b3OpenCLArray<int> m_numConcavePairsOut;
	b3OpenCLArray<b3CompoundOverlappingPair> m_gpuCompoundPairs;
	b3OpenCLArray<b3Vector3> m_gpuCompoundSepNormals;
	b3OpenCLArray<int> m_gpuHasCompoundSepNormals;
	b3OpenCLArray<int> m_numCompoundPairsOut;

	GpuSatCollision(cl_context ctx, cl_device_id device, cl_command_queue q);
	virtual ~GpuSatCollision();

	void computeConvexConvexContactsGPUSAT(b3OpenCLArray<b3Int4>* pairs, int nPairs,
										   const b3OpenCLArray<b3RigidBodyData>* bodyBuf,
										   b3OpenCLArray<b3Contact4>* contactOut, int& nContacts,
										   const b3OpenCLArray<b3Contact4>* oldContacts,
										   int maxContactCapacity,
										   int compoundPairCapacity,
										   const b3OpenCLArray<b3ConvexPolyhedronData>& hostConvexData,
										   const b3OpenCLArray<b3Vector3>& vertices,
										   const b3OpenCLArray<b3Vector3>& uniqueEdges,
										   const b3OpenCLArray<b3GpuFace>& faces,
										   const b3OpenCLArray<int>& indices,
										   const b3OpenCLArray<b3Collidable>& gpuCollidables,
										   const b3OpenCLArray<b3GpuChildShape>& gpuChildShapes,

										   const b3OpenCLArray<b3Aabb>& clAabbsWorldSpace,
										   const b3OpenCLArray<b3Aabb>& clAabbsLocalSpace,

										   b3OpenCLArray<b3Vector3>& worldVertsB1GPU,
										   b3OpenCLArray<b3Int4>& clippingFacesOutGPU,
										   b3OpenCLArray<b3Vector3>& worldNormalsAGPU,
										   b3OpenCLArray<b3Vector3>& worldVertsA1GPU,
										   b3OpenCLArray<b3Vector3>& worldVertsB2GPU,
										   b3AlignedObjectArray<class b3OptimizedBvh*>& bvhData,
										   b3OpenCLArray<b3QuantizedBvhNode>* treeNodesGPU,
										   b3OpenCLArray<b3BvhSubtreeInfo>* subTreesGPU,
										   b3OpenCLArray<b3BvhInfo>* bvhInfo,
										   int numObjects,
										   int maxTriConvexPairCapacity,
										   b3OpenCLArray<b3Int4>& triangleConvexPairs,
										   int& numTriConvexPairsOut);
};

#endif  //_CONVEX_HULL_CONTACT_H
