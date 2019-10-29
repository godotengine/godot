#ifndef B3_GPU_GRID_BROADPHASE_H
#define B3_GPU_GRID_BROADPHASE_H

#include "b3GpuBroadphaseInterface.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3RadixSort32CL.h"

struct b3ParamsGridBroadphaseCL
{
	float m_invCellSize[4];
	int m_gridSize[4];

	int getMaxBodiesPerCell() const
	{
		return m_gridSize[3];
	}

	void setMaxBodiesPerCell(int maxOverlap)
	{
		m_gridSize[3] = maxOverlap;
	}
};

class b3GpuGridBroadphase : public b3GpuBroadphaseInterface
{
protected:
	cl_context m_context;
	cl_device_id m_device;
	cl_command_queue m_queue;

	b3OpenCLArray<b3SapAabb> m_allAabbsGPU1;
	b3AlignedObjectArray<b3SapAabb> m_allAabbsCPU1;

	b3OpenCLArray<int> m_smallAabbsMappingGPU;
	b3AlignedObjectArray<int> m_smallAabbsMappingCPU;

	b3OpenCLArray<int> m_largeAabbsMappingGPU;
	b3AlignedObjectArray<int> m_largeAabbsMappingCPU;

	b3AlignedObjectArray<b3Int4> m_hostPairs;
	b3OpenCLArray<b3Int4> m_gpuPairs;

	b3OpenCLArray<b3SortData> m_hashGpu;
	b3OpenCLArray<int> m_cellStartGpu;

	b3ParamsGridBroadphaseCL m_paramsCPU;
	b3OpenCLArray<b3ParamsGridBroadphaseCL> m_paramsGPU;

	class b3RadixSort32CL* m_sorter;

public:
	b3GpuGridBroadphase(cl_context ctx, cl_device_id device, cl_command_queue q);
	virtual ~b3GpuGridBroadphase();

	static b3GpuBroadphaseInterface* CreateFunc(cl_context ctx, cl_device_id device, cl_command_queue q)
	{
		return new b3GpuGridBroadphase(ctx, device, q);
	}

	virtual void createProxy(const b3Vector3& aabbMin, const b3Vector3& aabbMax, int userPtr, int collisionFilterGroup, int collisionFilterMask);
	virtual void createLargeProxy(const b3Vector3& aabbMin, const b3Vector3& aabbMax, int userPtr, int collisionFilterGroup, int collisionFilterMask);

	virtual void calculateOverlappingPairs(int maxPairs);
	virtual void calculateOverlappingPairsHost(int maxPairs);

	//call writeAabbsToGpu after done making all changes (createProxy etc)
	virtual void writeAabbsToGpu();

	virtual cl_mem getAabbBufferWS();
	virtual int getNumOverlap();
	virtual cl_mem getOverlappingPairBuffer();

	virtual b3OpenCLArray<b3SapAabb>& getAllAabbsGPU();
	virtual b3AlignedObjectArray<b3SapAabb>& getAllAabbsCPU();

	virtual b3OpenCLArray<b3Int4>& getOverlappingPairsGPU();
	virtual b3OpenCLArray<int>& getSmallAabbIndicesGPU();
	virtual b3OpenCLArray<int>& getLargeAabbIndicesGPU();
};

#endif  //B3_GPU_GRID_BROADPHASE_H