
#ifndef B3_GPU_BROADPHASE_INTERFACE_H
#define B3_GPU_BROADPHASE_INTERFACE_H

#include "Bullet3OpenCL/Initialize/b3OpenCLInclude.h"
#include "Bullet3Common/b3Vector3.h"
#include "b3SapAabb.h"
#include "Bullet3Common/shared/b3Int2.h"
#include "Bullet3Common/shared/b3Int4.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3OpenCLArray.h"

class b3GpuBroadphaseInterface
{
public:
	typedef class b3GpuBroadphaseInterface*(CreateFunc)(cl_context ctx, cl_device_id device, cl_command_queue q);

	virtual ~b3GpuBroadphaseInterface()
	{
	}

	virtual void createProxy(const b3Vector3& aabbMin, const b3Vector3& aabbMax, int userPtr, int collisionFilterGroup, int collisionFilterMask) = 0;
	virtual void createLargeProxy(const b3Vector3& aabbMin, const b3Vector3& aabbMax, int userPtr, int collisionFilterGroup, int collisionFilterMask) = 0;

	virtual void calculateOverlappingPairs(int maxPairs) = 0;
	virtual void calculateOverlappingPairsHost(int maxPairs) = 0;

	//call writeAabbsToGpu after done making all changes (createProxy etc)
	virtual void writeAabbsToGpu() = 0;

	virtual cl_mem getAabbBufferWS() = 0;
	virtual int getNumOverlap() = 0;
	virtual cl_mem getOverlappingPairBuffer() = 0;

	virtual b3OpenCLArray<b3SapAabb>& getAllAabbsGPU() = 0;
	virtual b3AlignedObjectArray<b3SapAabb>& getAllAabbsCPU() = 0;

	virtual b3OpenCLArray<b3Int4>& getOverlappingPairsGPU() = 0;
	virtual b3OpenCLArray<int>& getSmallAabbIndicesGPU() = 0;
	virtual b3OpenCLArray<int>& getLargeAabbIndicesGPU() = 0;
};

#endif  //B3_GPU_BROADPHASE_INTERFACE_H
