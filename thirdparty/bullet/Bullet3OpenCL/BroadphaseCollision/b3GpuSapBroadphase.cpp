
bool searchIncremental3dSapOnGpu = true;
#include <limits.h>
#include "b3GpuSapBroadphase.h"
#include "Bullet3Common/b3Vector3.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3LauncherCL.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3PrefixScanFloat4CL.h"

#include "Bullet3OpenCL/Initialize/b3OpenCLUtils.h"
#include "kernels/sapKernels.h"

#include "Bullet3Common/b3MinMax.h"

#define B3_BROADPHASE_SAP_PATH "src/Bullet3OpenCL/BroadphaseCollision/kernels/sap.cl"

/*
	
 
	
	
	
 
	b3OpenCLArray<int> m_pairCount;
 
 
	b3OpenCLArray<b3SapAabb>	m_allAabbsGPU;
	b3AlignedObjectArray<b3SapAabb>	m_allAabbsCPU;
 
	virtual b3OpenCLArray<b3SapAabb>&	getAllAabbsGPU()
	{
 return m_allAabbsGPU;
	}
	virtual b3AlignedObjectArray<b3SapAabb>&	getAllAabbsCPU()
	{
 return m_allAabbsCPU;
	}
 
	b3OpenCLArray<b3Vector3>	m_sum;
	b3OpenCLArray<b3Vector3>	m_sum2;
	b3OpenCLArray<b3Vector3>	m_dst;
 
	b3OpenCLArray<int>	m_smallAabbsMappingGPU;
	b3AlignedObjectArray<int> m_smallAabbsMappingCPU;
 
	b3OpenCLArray<int>	m_largeAabbsMappingGPU;
	b3AlignedObjectArray<int> m_largeAabbsMappingCPU;
 
	
	b3OpenCLArray<b3Int4>		m_overlappingPairs;
 
	//temporary gpu work memory
	b3OpenCLArray<b3SortData>	m_gpuSmallSortData;
	b3OpenCLArray<b3SapAabb>	m_gpuSmallSortedAabbs;
 
	class b3PrefixScanFloat4CL*		m_prefixScanFloat4;
 */

b3GpuSapBroadphase::b3GpuSapBroadphase(cl_context ctx, cl_device_id device, cl_command_queue q, b3GpuSapKernelType kernelType)
	: m_context(ctx),
	  m_device(device),
	  m_queue(q),

	  m_objectMinMaxIndexGPUaxis0(ctx, q),
	  m_objectMinMaxIndexGPUaxis1(ctx, q),
	  m_objectMinMaxIndexGPUaxis2(ctx, q),
	  m_objectMinMaxIndexGPUaxis0prev(ctx, q),
	  m_objectMinMaxIndexGPUaxis1prev(ctx, q),
	  m_objectMinMaxIndexGPUaxis2prev(ctx, q),
	  m_sortedAxisGPU0(ctx, q),
	  m_sortedAxisGPU1(ctx, q),
	  m_sortedAxisGPU2(ctx, q),
	  m_sortedAxisGPU0prev(ctx, q),
	  m_sortedAxisGPU1prev(ctx, q),
	  m_sortedAxisGPU2prev(ctx, q),
	  m_addedHostPairsGPU(ctx, q),
	  m_removedHostPairsGPU(ctx, q),
	  m_addedCountGPU(ctx, q),
	  m_removedCountGPU(ctx, q),
	  m_currentBuffer(-1),
	  m_pairCount(ctx, q),
	  m_allAabbsGPU(ctx, q),
	  m_sum(ctx, q),
	  m_sum2(ctx, q),
	  m_dst(ctx, q),
	  m_smallAabbsMappingGPU(ctx, q),
	  m_largeAabbsMappingGPU(ctx, q),
	  m_overlappingPairs(ctx, q),
	  m_gpuSmallSortData(ctx, q),
	  m_gpuSmallSortedAabbs(ctx, q)
{
	const char* sapSrc = sapCL;

	cl_int errNum = 0;

	b3Assert(m_context);
	b3Assert(m_device);
	cl_program sapProg = b3OpenCLUtils::compileCLProgramFromString(m_context, m_device, sapSrc, &errNum, "", B3_BROADPHASE_SAP_PATH);
	b3Assert(errNum == CL_SUCCESS);

	b3Assert(errNum == CL_SUCCESS);
#ifndef __APPLE__
	m_prefixScanFloat4 = new b3PrefixScanFloat4CL(m_context, m_device, m_queue);
#else
	m_prefixScanFloat4 = 0;
#endif
	m_sapKernel = 0;

	switch (kernelType)
	{
		case B3_GPU_SAP_KERNEL_BRUTE_FORCE_CPU:
		{
			m_sapKernel = 0;
			break;
		}
		case B3_GPU_SAP_KERNEL_BRUTE_FORCE_GPU:
		{
			m_sapKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device, sapSrc, "computePairsKernelBruteForce", &errNum, sapProg);
			break;
		}

		case B3_GPU_SAP_KERNEL_ORIGINAL:
		{
			m_sapKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device, sapSrc, "computePairsKernelOriginal", &errNum, sapProg);
			break;
		}
		case B3_GPU_SAP_KERNEL_BARRIER:
		{
			m_sapKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device, sapSrc, "computePairsKernelBarrier", &errNum, sapProg);
			break;
		}
		case B3_GPU_SAP_KERNEL_LOCAL_SHARED_MEMORY:
		{
			m_sapKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device, sapSrc, "computePairsKernelLocalSharedMemory", &errNum, sapProg);
			break;
		}

		default:
		{
			m_sapKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device, sapSrc, "computePairsKernelLocalSharedMemory", &errNum, sapProg);
			b3Error("Unknown 3D GPU SAP provided, fallback to computePairsKernelLocalSharedMemory");
		}
	};

	m_sap2Kernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device, sapSrc, "computePairsKernelTwoArrays", &errNum, sapProg);
	b3Assert(errNum == CL_SUCCESS);

	m_prepareSumVarianceKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device, sapSrc, "prepareSumVarianceKernel", &errNum, sapProg);
	b3Assert(errNum == CL_SUCCESS);

	m_flipFloatKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device, sapSrc, "flipFloatKernel", &errNum, sapProg);

	m_copyAabbsKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device, sapSrc, "copyAabbsKernel", &errNum, sapProg);

	m_scatterKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device, sapSrc, "scatterKernel", &errNum, sapProg);

	m_sorter = new b3RadixSort32CL(m_context, m_device, m_queue);
}

b3GpuSapBroadphase::~b3GpuSapBroadphase()
{
	delete m_sorter;
	delete m_prefixScanFloat4;

	clReleaseKernel(m_scatterKernel);
	clReleaseKernel(m_flipFloatKernel);
	clReleaseKernel(m_copyAabbsKernel);
	clReleaseKernel(m_sapKernel);
	clReleaseKernel(m_sap2Kernel);
	clReleaseKernel(m_prepareSumVarianceKernel);
}

/// conservative test for overlap between two aabbs
static bool TestAabbAgainstAabb2(const b3Vector3& aabbMin1, const b3Vector3& aabbMax1,
								 const b3Vector3& aabbMin2, const b3Vector3& aabbMax2)
{
	bool overlap = true;
	overlap = (aabbMin1.getX() > aabbMax2.getX() || aabbMax1.getX() < aabbMin2.getX()) ? false : overlap;
	overlap = (aabbMin1.getZ() > aabbMax2.getZ() || aabbMax1.getZ() < aabbMin2.getZ()) ? false : overlap;
	overlap = (aabbMin1.getY() > aabbMax2.getY() || aabbMax1.getY() < aabbMin2.getY()) ? false : overlap;
	return overlap;
}

//http://stereopsis.com/radix.html
static unsigned int FloatFlip(float fl)
{
	unsigned int f = *(unsigned int*)&fl;
	unsigned int mask = -(int)(f >> 31) | 0x80000000;
	return f ^ mask;
};

void b3GpuSapBroadphase::init3dSap()
{
	if (m_currentBuffer < 0)
	{
		m_allAabbsGPU.copyToHost(m_allAabbsCPU);

		m_currentBuffer = 0;
		for (int axis = 0; axis < 3; axis++)
		{
			for (int buf = 0; buf < 2; buf++)
			{
				int totalNumAabbs = m_allAabbsCPU.size();
				int numEndPoints = 2 * totalNumAabbs;
				m_sortedAxisCPU[axis][buf].resize(numEndPoints);

				if (buf == m_currentBuffer)
				{
					for (int i = 0; i < totalNumAabbs; i++)
					{
						m_sortedAxisCPU[axis][buf][i * 2].m_key = FloatFlip(m_allAabbsCPU[i].m_min[axis]) - 1;
						m_sortedAxisCPU[axis][buf][i * 2].m_value = i * 2;
						m_sortedAxisCPU[axis][buf][i * 2 + 1].m_key = FloatFlip(m_allAabbsCPU[i].m_max[axis]) + 1;
						m_sortedAxisCPU[axis][buf][i * 2 + 1].m_value = i * 2 + 1;
					}
				}
			}
		}

		for (int axis = 0; axis < 3; axis++)
		{
			m_sorter->executeHost(m_sortedAxisCPU[axis][m_currentBuffer]);
		}

		for (int axis = 0; axis < 3; axis++)
		{
			//int totalNumAabbs = m_allAabbsCPU.size();
			int numEndPoints = m_sortedAxisCPU[axis][m_currentBuffer].size();
			m_objectMinMaxIndexCPU[axis][m_currentBuffer].resize(numEndPoints);
			for (int i = 0; i < numEndPoints; i++)
			{
				int destIndex = m_sortedAxisCPU[axis][m_currentBuffer][i].m_value;
				int newDest = destIndex / 2;
				if (destIndex & 1)
				{
					m_objectMinMaxIndexCPU[axis][m_currentBuffer][newDest].y = i;
				}
				else
				{
					m_objectMinMaxIndexCPU[axis][m_currentBuffer][newDest].x = i;
				}
			}
		}
	}
}

static bool b3PairCmp(const b3Int4& p, const b3Int4& q)
{
	return ((p.x < q.x) || ((p.x == q.x) && (p.y < q.y)));
}

static bool operator==(const b3Int4& a, const b3Int4& b)
{
	return a.x == b.x && a.y == b.y;
};

static bool operator<(const b3Int4& a, const b3Int4& b)
{
	return a.x < b.x || (a.x == b.x && a.y < b.y);
};

static bool operator>(const b3Int4& a, const b3Int4& b)
{
	return a.x > b.x || (a.x == b.x && a.y > b.y);
};

b3AlignedObjectArray<b3Int4> addedHostPairs;
b3AlignedObjectArray<b3Int4> removedHostPairs;

b3AlignedObjectArray<b3SapAabb> preAabbs;

void b3GpuSapBroadphase::calculateOverlappingPairsHostIncremental3Sap()
{
	//static int framepje = 0;
	//printf("framepje=%d\n",framepje++);

	B3_PROFILE("calculateOverlappingPairsHostIncremental3Sap");

	addedHostPairs.resize(0);
	removedHostPairs.resize(0);

	b3Assert(m_currentBuffer >= 0);

	{
		preAabbs.resize(m_allAabbsCPU.size());
		for (int i = 0; i < preAabbs.size(); i++)
		{
			preAabbs[i] = m_allAabbsCPU[i];
		}
	}

	if (m_currentBuffer < 0)
		return;
	{
		B3_PROFILE("m_allAabbsGPU.copyToHost");
		m_allAabbsGPU.copyToHost(m_allAabbsCPU);
	}

	b3AlignedObjectArray<b3Int4> allPairs;
	{
		B3_PROFILE("m_overlappingPairs.copyToHost");
		m_overlappingPairs.copyToHost(allPairs);
	}
	if (0)
	{
		{
			printf("ab[40].min=%f,%f,%f,ab[40].max=%f,%f,%f\n",
				   m_allAabbsCPU[40].m_min[0], m_allAabbsCPU[40].m_min[1], m_allAabbsCPU[40].m_min[2],
				   m_allAabbsCPU[40].m_max[0], m_allAabbsCPU[40].m_max[1], m_allAabbsCPU[40].m_max[2]);
		}

		{
			printf("ab[53].min=%f,%f,%f,ab[53].max=%f,%f,%f\n",
				   m_allAabbsCPU[53].m_min[0], m_allAabbsCPU[53].m_min[1], m_allAabbsCPU[53].m_min[2],
				   m_allAabbsCPU[53].m_max[0], m_allAabbsCPU[53].m_max[1], m_allAabbsCPU[53].m_max[2]);
		}

		{
			b3Int4 newPair;
			newPair.x = 40;
			newPair.y = 53;
			int index = allPairs.findBinarySearch(newPair);
			printf("hasPair(40,53)=%d out of %d\n", index, allPairs.size());

			{
				int overlap = TestAabbAgainstAabb2((const b3Vector3&)m_allAabbsCPU[40].m_min, (const b3Vector3&)m_allAabbsCPU[40].m_max, (const b3Vector3&)m_allAabbsCPU[53].m_min, (const b3Vector3&)m_allAabbsCPU[53].m_max);
				printf("overlap=%d\n", overlap);
			}

			if (preAabbs.size())
			{
				int prevOverlap = TestAabbAgainstAabb2((const b3Vector3&)preAabbs[40].m_min, (const b3Vector3&)preAabbs[40].m_max, (const b3Vector3&)preAabbs[53].m_min, (const b3Vector3&)preAabbs[53].m_max);
				printf("prevoverlap=%d\n", prevOverlap);
			}
			else
			{
				printf("unknown prevoverlap\n");
			}
		}
	}

	if (0)
	{
		for (int i = 0; i < m_allAabbsCPU.size(); i++)
		{
			//printf("aabb[%d] min=%f,%f,%f max=%f,%f,%f\n",i,m_allAabbsCPU[i].m_min[0],m_allAabbsCPU[i].m_min[1],m_allAabbsCPU[i].m_min[2],			m_allAabbsCPU[i].m_max[0],m_allAabbsCPU[i].m_max[1],m_allAabbsCPU[i].m_max[2]);
		}

		for (int axis = 0; axis < 3; axis++)
		{
			for (int buf = 0; buf < 2; buf++)
			{
				b3Assert(m_sortedAxisCPU[axis][buf].size() == m_allAabbsCPU.size() * 2);
			}
		}
	}

	m_currentBuffer = 1 - m_currentBuffer;

	int totalNumAabbs = m_allAabbsCPU.size();

	{
		B3_PROFILE("assign m_sortedAxisCPU(FloatFlip)");
		for (int i = 0; i < totalNumAabbs; i++)
		{
			unsigned int keyMin[3];
			unsigned int keyMax[3];
			for (int axis = 0; axis < 3; axis++)
			{
				float vmin = m_allAabbsCPU[i].m_min[axis];
				float vmax = m_allAabbsCPU[i].m_max[axis];
				keyMin[axis] = FloatFlip(vmin);
				keyMax[axis] = FloatFlip(vmax);

				m_sortedAxisCPU[axis][m_currentBuffer][i * 2].m_key = keyMin[axis] - 1;
				m_sortedAxisCPU[axis][m_currentBuffer][i * 2].m_value = i * 2;
				m_sortedAxisCPU[axis][m_currentBuffer][i * 2 + 1].m_key = keyMax[axis] + 1;
				m_sortedAxisCPU[axis][m_currentBuffer][i * 2 + 1].m_value = i * 2 + 1;
			}
			//printf("aabb[%d] min=%u,%u,%u max %u,%u,%u\n", i,keyMin[0],keyMin[1],keyMin[2],keyMax[0],keyMax[1],keyMax[2]);
		}
	}

	{
		B3_PROFILE("sort m_sortedAxisCPU");
		for (int axis = 0; axis < 3; axis++)
			m_sorter->executeHost(m_sortedAxisCPU[axis][m_currentBuffer]);
	}

#if 0
	if (0)
	{
		for (int axis=0;axis<3;axis++)
		{
			//printf("axis %d\n",axis);
			for (int i=0;i<m_sortedAxisCPU[axis][m_currentBuffer].size();i++)
			{
				//int key = m_sortedAxisCPU[axis][m_currentBuffer][i].m_key;
				//int value = m_sortedAxisCPU[axis][m_currentBuffer][i].m_value;
				//printf("[%d]=%d\n",i,value);
			}

		}
	}
#endif

	{
		B3_PROFILE("assign m_objectMinMaxIndexCPU");
		for (int axis = 0; axis < 3; axis++)
		{
			int totalNumAabbs = m_allAabbsCPU.size();
			int numEndPoints = m_sortedAxisCPU[axis][m_currentBuffer].size();
			m_objectMinMaxIndexCPU[axis][m_currentBuffer].resize(totalNumAabbs);
			for (int i = 0; i < numEndPoints; i++)
			{
				int destIndex = m_sortedAxisCPU[axis][m_currentBuffer][i].m_value;
				int newDest = destIndex / 2;
				if (destIndex & 1)
				{
					m_objectMinMaxIndexCPU[axis][m_currentBuffer][newDest].y = i;
				}
				else
				{
					m_objectMinMaxIndexCPU[axis][m_currentBuffer][newDest].x = i;
				}
			}
		}
	}

#if 0
	if (0)
	{	
		printf("==========================\n");
		for (int axis=0;axis<3;axis++)
		{
			unsigned int curMinIndex40 = m_objectMinMaxIndexCPU[axis][m_currentBuffer][40].x;
			unsigned int curMaxIndex40 = m_objectMinMaxIndexCPU[axis][m_currentBuffer][40].y;
			unsigned int prevMaxIndex40 = m_objectMinMaxIndexCPU[axis][1-m_currentBuffer][40].y;
			unsigned int prevMinIndex40 = m_objectMinMaxIndexCPU[axis][1-m_currentBuffer][40].x;

			int dmin40 = curMinIndex40 - prevMinIndex40;
			int dmax40 = curMinIndex40 - prevMinIndex40;
			printf("axis %d curMinIndex40=%d prevMinIndex40=%d\n",axis,curMinIndex40, prevMinIndex40);
			printf("axis %d curMaxIndex40=%d prevMaxIndex40=%d\n",axis,curMaxIndex40, prevMaxIndex40);
		}
		printf(".........................\n");
		for (int axis=0;axis<3;axis++)
		{
			unsigned int curMinIndex53 = m_objectMinMaxIndexCPU[axis][m_currentBuffer][53].x;
			unsigned int curMaxIndex53 = m_objectMinMaxIndexCPU[axis][m_currentBuffer][53].y;
			unsigned int prevMaxIndex53 = m_objectMinMaxIndexCPU[axis][1-m_currentBuffer][53].y;
			unsigned int prevMinIndex53 = m_objectMinMaxIndexCPU[axis][1-m_currentBuffer][53].x;

			int dmin40 = curMinIndex53 - prevMinIndex53;
			int dmax40 = curMinIndex53 - prevMinIndex53;
			printf("axis %d curMinIndex53=%d prevMinIndex53=%d\n",axis,curMinIndex53, prevMinIndex53);
			printf("axis %d curMaxIndex53=%d prevMaxIndex53=%d\n",axis,curMaxIndex53, prevMaxIndex53);
		}

	}
#endif

	int a = m_objectMinMaxIndexCPU[0][m_currentBuffer].size();
	int b = m_objectMinMaxIndexCPU[1][m_currentBuffer].size();
	int c = m_objectMinMaxIndexCPU[2][m_currentBuffer].size();
	b3Assert(a == b);
	b3Assert(b == c);
	/*
	if (searchIncremental3dSapOnGpu)
	{
		B3_PROFILE("computePairsIncremental3dSapKernelGPU");
		int numObjects = m_objectMinMaxIndexCPU[0][m_currentBuffer].size();
		int maxCapacity = 1024*1024;
		{
			B3_PROFILE("copy from host");
			m_objectMinMaxIndexGPUaxis0.copyFromHost(m_objectMinMaxIndexCPU[0][m_currentBuffer]);
			m_objectMinMaxIndexGPUaxis1.copyFromHost(m_objectMinMaxIndexCPU[1][m_currentBuffer]);
			m_objectMinMaxIndexGPUaxis2.copyFromHost(m_objectMinMaxIndexCPU[2][m_currentBuffer]);
			m_objectMinMaxIndexGPUaxis0prev.copyFromHost(m_objectMinMaxIndexCPU[0][1-m_currentBuffer]);
			m_objectMinMaxIndexGPUaxis1prev.copyFromHost(m_objectMinMaxIndexCPU[1][1-m_currentBuffer]);
			m_objectMinMaxIndexGPUaxis2prev.copyFromHost(m_objectMinMaxIndexCPU[2][1-m_currentBuffer]);

			m_sortedAxisGPU0.copyFromHost(m_sortedAxisCPU[0][m_currentBuffer]);
			m_sortedAxisGPU1.copyFromHost(m_sortedAxisCPU[1][m_currentBuffer]);
			m_sortedAxisGPU2.copyFromHost(m_sortedAxisCPU[2][m_currentBuffer]);
			m_sortedAxisGPU0prev.copyFromHost(m_sortedAxisCPU[0][1-m_currentBuffer]);
			m_sortedAxisGPU1prev.copyFromHost(m_sortedAxisCPU[1][1-m_currentBuffer]);
			m_sortedAxisGPU2prev.copyFromHost(m_sortedAxisCPU[2][1-m_currentBuffer]);

		
			m_addedHostPairsGPU.resize(maxCapacity);
			m_removedHostPairsGPU.resize(maxCapacity);

			m_addedCountGPU.resize(0);
			m_addedCountGPU.push_back(0);
			m_removedCountGPU.resize(0);
			m_removedCountGPU.push_back(0);
		}

		{
			B3_PROFILE("launch1D");
			b3LauncherCL launcher(m_queue,  m_computePairsIncremental3dSapKernel,"m_computePairsIncremental3dSapKernel");
			launcher.setBuffer(m_objectMinMaxIndexGPUaxis0.getBufferCL());
			launcher.setBuffer(m_objectMinMaxIndexGPUaxis1.getBufferCL());
			launcher.setBuffer(m_objectMinMaxIndexGPUaxis2.getBufferCL());
			launcher.setBuffer(m_objectMinMaxIndexGPUaxis0prev.getBufferCL());
			launcher.setBuffer(m_objectMinMaxIndexGPUaxis1prev.getBufferCL());
			launcher.setBuffer(m_objectMinMaxIndexGPUaxis2prev.getBufferCL());

			launcher.setBuffer(m_sortedAxisGPU0.getBufferCL());
			launcher.setBuffer(m_sortedAxisGPU1.getBufferCL());
			launcher.setBuffer(m_sortedAxisGPU2.getBufferCL());
			launcher.setBuffer(m_sortedAxisGPU0prev.getBufferCL());
			launcher.setBuffer(m_sortedAxisGPU1prev.getBufferCL());
			launcher.setBuffer(m_sortedAxisGPU2prev.getBufferCL());

		
			launcher.setBuffer(m_addedHostPairsGPU.getBufferCL());
			launcher.setBuffer(m_removedHostPairsGPU.getBufferCL());
			launcher.setBuffer(m_addedCountGPU.getBufferCL());
			launcher.setBuffer(m_removedCountGPU.getBufferCL());
			launcher.setConst(maxCapacity);
			launcher.setConst( numObjects);
			launcher.launch1D( numObjects);
			clFinish(m_queue);
		}

		{
			B3_PROFILE("copy to host");
			int addedCountGPU = m_addedCountGPU.at(0);
			m_addedHostPairsGPU.resize(addedCountGPU);
			m_addedHostPairsGPU.copyToHost(addedHostPairs);

			//printf("addedCountGPU=%d\n",addedCountGPU);
			int removedCountGPU = m_removedCountGPU.at(0);
			m_removedHostPairsGPU.resize(removedCountGPU);
			m_removedHostPairsGPU.copyToHost(removedHostPairs);
			//printf("removedCountGPU=%d\n",removedCountGPU);

		}



	} 
	else
	*/
	{
		int numObjects = m_objectMinMaxIndexCPU[0][m_currentBuffer].size();

		B3_PROFILE("actual search");
		for (int i = 0; i < numObjects; i++)
		{
			//int numObjects = m_objectMinMaxIndexCPU[axis][m_currentBuffer].size();
			//int checkObjects[]={40,53};
			//int numCheckObjects = sizeof(checkObjects)/sizeof(int);

			//for (int a=0;a<numCheckObjects ;a++)

			for (int axis = 0; axis < 3; axis++)
			{
				//int i = checkObjects[a];

				unsigned int curMinIndex = m_objectMinMaxIndexCPU[axis][m_currentBuffer][i].x;
				unsigned int curMaxIndex = m_objectMinMaxIndexCPU[axis][m_currentBuffer][i].y;
				unsigned int prevMinIndex = m_objectMinMaxIndexCPU[axis][1 - m_currentBuffer][i].x;
				int dmin = curMinIndex - prevMinIndex;

				unsigned int prevMaxIndex = m_objectMinMaxIndexCPU[axis][1 - m_currentBuffer][i].y;

				int dmax = curMaxIndex - prevMaxIndex;
				if (dmin != 0)
				{
					//printf("for object %d, dmin=%d\n",i,dmin);
				}
				if (dmax != 0)
				{
					//printf("for object %d, dmax=%d\n",i,dmax);
				}
				for (int otherbuffer = 0; otherbuffer < 2; otherbuffer++)
				{
					if (dmin != 0)
					{
						int stepMin = dmin < 0 ? -1 : 1;
						for (int j = prevMinIndex; j != curMinIndex; j += stepMin)
						{
							int otherIndex2 = m_sortedAxisCPU[axis][otherbuffer][j].y;
							int otherIndex = otherIndex2 / 2;
							if (otherIndex != i)
							{
								bool otherIsMax = ((otherIndex2 & 1) != 0);

								if (otherIsMax)
								{
									//bool overlap = TestAabbAgainstAabb2((const b3Vector3&)m_allAabbsCPU[i].m_min, (const b3Vector3&)m_allAabbsCPU[i].m_max,(const b3Vector3&)m_allAabbsCPU[otherIndex].m_min,(const b3Vector3&)m_allAabbsCPU[otherIndex].m_max);
									//bool prevOverlap = TestAabbAgainstAabb2((const b3Vector3&)preAabbs[i].m_min, (const b3Vector3&)preAabbs[i].m_max,(const b3Vector3&)preAabbs[otherIndex].m_min,(const b3Vector3&)preAabbs[otherIndex].m_max);

									bool overlap = true;

									for (int ax = 0; ax < 3; ax++)
									{
										if ((m_objectMinMaxIndexCPU[ax][m_currentBuffer][i].x > m_objectMinMaxIndexCPU[ax][m_currentBuffer][otherIndex].y) ||
											(m_objectMinMaxIndexCPU[ax][m_currentBuffer][i].y < m_objectMinMaxIndexCPU[ax][m_currentBuffer][otherIndex].x))
											overlap = false;
									}

									//	b3Assert(overlap2==overlap);

									bool prevOverlap = true;

									for (int ax = 0; ax < 3; ax++)
									{
										if ((m_objectMinMaxIndexCPU[ax][1 - m_currentBuffer][i].x > m_objectMinMaxIndexCPU[ax][1 - m_currentBuffer][otherIndex].y) ||
											(m_objectMinMaxIndexCPU[ax][1 - m_currentBuffer][i].y < m_objectMinMaxIndexCPU[ax][1 - m_currentBuffer][otherIndex].x))
											prevOverlap = false;
									}

									//b3Assert(overlap==overlap2);

									if (dmin < 0)
									{
										if (overlap && !prevOverlap)
										{
											//add a pair
											b3Int4 newPair;
											if (i <= otherIndex)
											{
												newPair.x = i;
												newPair.y = otherIndex;
											}
											else
											{
												newPair.x = otherIndex;
												newPair.y = i;
											}
											addedHostPairs.push_back(newPair);
										}
									}
									else
									{
										if (!overlap && prevOverlap)
										{
											//remove a pair
											b3Int4 removedPair;
											if (i <= otherIndex)
											{
												removedPair.x = i;
												removedPair.y = otherIndex;
											}
											else
											{
												removedPair.x = otherIndex;
												removedPair.y = i;
											}
											removedHostPairs.push_back(removedPair);
										}
									}  //otherisMax
								}      //if (dmin<0)
							}          //if (otherIndex!=i)
						}              //for (int j=
					}

					if (dmax != 0)
					{
						int stepMax = dmax < 0 ? -1 : 1;
						for (int j = prevMaxIndex; j != curMaxIndex; j += stepMax)
						{
							int otherIndex2 = m_sortedAxisCPU[axis][otherbuffer][j].y;
							int otherIndex = otherIndex2 / 2;
							if (otherIndex != i)
							{
								//bool otherIsMin = ((otherIndex2&1)==0);
								//if (otherIsMin)
								{
									//bool overlap = TestAabbAgainstAabb2((const b3Vector3&)m_allAabbsCPU[i].m_min, (const b3Vector3&)m_allAabbsCPU[i].m_max,(const b3Vector3&)m_allAabbsCPU[otherIndex].m_min,(const b3Vector3&)m_allAabbsCPU[otherIndex].m_max);
									//bool prevOverlap = TestAabbAgainstAabb2((const b3Vector3&)preAabbs[i].m_min, (const b3Vector3&)preAabbs[i].m_max,(const b3Vector3&)preAabbs[otherIndex].m_min,(const b3Vector3&)preAabbs[otherIndex].m_max);

									bool overlap = true;

									for (int ax = 0; ax < 3; ax++)
									{
										if ((m_objectMinMaxIndexCPU[ax][m_currentBuffer][i].x > m_objectMinMaxIndexCPU[ax][m_currentBuffer][otherIndex].y) ||
											(m_objectMinMaxIndexCPU[ax][m_currentBuffer][i].y < m_objectMinMaxIndexCPU[ax][m_currentBuffer][otherIndex].x))
											overlap = false;
									}
									//b3Assert(overlap2==overlap);

									bool prevOverlap = true;

									for (int ax = 0; ax < 3; ax++)
									{
										if ((m_objectMinMaxIndexCPU[ax][1 - m_currentBuffer][i].x > m_objectMinMaxIndexCPU[ax][1 - m_currentBuffer][otherIndex].y) ||
											(m_objectMinMaxIndexCPU[ax][1 - m_currentBuffer][i].y < m_objectMinMaxIndexCPU[ax][1 - m_currentBuffer][otherIndex].x))
											prevOverlap = false;
									}

									if (dmax > 0)
									{
										if (overlap && !prevOverlap)
										{
											//add a pair
											b3Int4 newPair;
											if (i <= otherIndex)
											{
												newPair.x = i;
												newPair.y = otherIndex;
											}
											else
											{
												newPair.x = otherIndex;
												newPair.y = i;
											}
											addedHostPairs.push_back(newPair);
										}
									}
									else
									{
										if (!overlap && prevOverlap)
										{
											//if (otherIndex2&1==0) -> min?
											//remove a pair
											b3Int4 removedPair;
											if (i <= otherIndex)
											{
												removedPair.x = i;
												removedPair.y = otherIndex;
											}
											else
											{
												removedPair.x = otherIndex;
												removedPair.y = i;
											}
											removedHostPairs.push_back(removedPair);
										}
									}

								}  //if (dmin<0)
							}      //if (otherIndex!=i)
						}          //for (int j=
					}
				}  //for (int otherbuffer
			}      //for (int axis=0;
		}          //for (int i=0;i<numObjects
	}

	//remove duplicates and add/remove then to existing m_overlappingPairs

	{
		{
			B3_PROFILE("sort allPairs");
			allPairs.quickSort(b3PairCmp);
		}
		{
			B3_PROFILE("sort addedHostPairs");
			addedHostPairs.quickSort(b3PairCmp);
		}
		{
			B3_PROFILE("sort removedHostPairs");
			removedHostPairs.quickSort(b3PairCmp);
		}
	}

	b3Int4 prevPair;
	prevPair.x = -1;
	prevPair.y = -1;

	int uniqueRemovedPairs = 0;

	b3AlignedObjectArray<int> removedPositions;

	{
		B3_PROFILE("actual removing");
		for (int i = 0; i < removedHostPairs.size(); i++)
		{
			b3Int4 removedPair = removedHostPairs[i];
			if ((removedPair.x != prevPair.x) || (removedPair.y != prevPair.y))
			{
				int index1 = allPairs.findBinarySearch(removedPair);

				//#ifdef _DEBUG

				int index2 = allPairs.findLinearSearch(removedPair);
				b3Assert(index1 == index2);

				//b3Assert(index1!=allPairs.size());
				if (index1 < allPairs.size())
				//#endif//_DEBUG
				{
					uniqueRemovedPairs++;
					removedPositions.push_back(index1);
					{
						//printf("framepje(%d) remove pair(%d):%d,%d\n",framepje,i,removedPair.x,removedPair.y);
					}
				}
			}
			prevPair = removedPair;
		}

		if (uniqueRemovedPairs)
		{
			for (int i = 0; i < removedPositions.size(); i++)
			{
				allPairs[removedPositions[i]].x = INT_MAX;
				allPairs[removedPositions[i]].y = INT_MAX;
			}
			allPairs.quickSort(b3PairCmp);
			allPairs.resize(allPairs.size() - uniqueRemovedPairs);
		}
	}
	//if (uniqueRemovedPairs)
	//	printf("uniqueRemovedPairs=%d\n",uniqueRemovedPairs);
	//printf("removedHostPairs.size = %d\n",removedHostPairs.size());

	prevPair.x = -1;
	prevPair.y = -1;

	int uniqueAddedPairs = 0;
	b3AlignedObjectArray<b3Int4> actualAddedPairs;

	{
		B3_PROFILE("actual adding");
		for (int i = 0; i < addedHostPairs.size(); i++)
		{
			b3Int4 newPair = addedHostPairs[i];
			if ((newPair.x != prevPair.x) || (newPair.y != prevPair.y))
			{
				//#ifdef _DEBUG
				int index1 = allPairs.findBinarySearch(newPair);

				int index2 = allPairs.findLinearSearch(newPair);
				b3Assert(index1 == index2);

				b3Assert(index1 == allPairs.size());
				if (index1 != allPairs.size())
				{
					printf("??\n");
				}

				if (index1 == allPairs.size())
				//#endif //_DEBUG
				{
					uniqueAddedPairs++;
					actualAddedPairs.push_back(newPair);
				}
			}
			prevPair = newPair;
		}
		for (int i = 0; i < actualAddedPairs.size(); i++)
		{
			//printf("framepje (%d), new pair(%d):%d,%d\n",framepje,i,actualAddedPairs[i].x,actualAddedPairs[i].y);
			allPairs.push_back(actualAddedPairs[i]);
		}
	}

	//if (uniqueAddedPairs)
	//	printf("uniqueAddedPairs=%d\n", uniqueAddedPairs);

	{
		B3_PROFILE("m_overlappingPairs.copyFromHost");
		m_overlappingPairs.copyFromHost(allPairs);
	}
}

void b3GpuSapBroadphase::calculateOverlappingPairsHost(int maxPairs)
{
	//test
	//	if (m_currentBuffer>=0)
	//	return calculateOverlappingPairsHostIncremental3Sap();

	b3Assert(m_allAabbsCPU.size() == m_allAabbsGPU.size());
	m_allAabbsGPU.copyToHost(m_allAabbsCPU);

	int axis = 0;
	{
		B3_PROFILE("CPU compute best variance axis");
		b3Vector3 s = b3MakeVector3(0, 0, 0), s2 = b3MakeVector3(0, 0, 0);
		int numRigidBodies = m_smallAabbsMappingCPU.size();

		for (int i = 0; i < numRigidBodies; i++)
		{
			b3SapAabb aabb = this->m_allAabbsCPU[m_smallAabbsMappingCPU[i]];

			b3Vector3 maxAabb = b3MakeVector3(aabb.m_max[0], aabb.m_max[1], aabb.m_max[2]);
			b3Vector3 minAabb = b3MakeVector3(aabb.m_min[0], aabb.m_min[1], aabb.m_min[2]);
			b3Vector3 centerAabb = (maxAabb + minAabb) * 0.5f;

			s += centerAabb;
			s2 += centerAabb * centerAabb;
		}
		b3Vector3 v = s2 - (s * s) / (float)numRigidBodies;

		if (v[1] > v[0])
			axis = 1;
		if (v[2] > v[axis])
			axis = 2;
	}

	b3AlignedObjectArray<b3Int4> hostPairs;

	{
		int numSmallAabbs = m_smallAabbsMappingCPU.size();
		for (int i = 0; i < numSmallAabbs; i++)
		{
			b3SapAabb smallAabbi = m_allAabbsCPU[m_smallAabbsMappingCPU[i]];
			//float reference = smallAabbi.m_max[axis];

			for (int j = i + 1; j < numSmallAabbs; j++)
			{
				b3SapAabb smallAabbj = m_allAabbsCPU[m_smallAabbsMappingCPU[j]];

				if (TestAabbAgainstAabb2((b3Vector3&)smallAabbi.m_min, (b3Vector3&)smallAabbi.m_max,
										 (b3Vector3&)smallAabbj.m_min, (b3Vector3&)smallAabbj.m_max))
				{
					b3Int4 pair;
					int a = smallAabbi.m_minIndices[3];
					int b = smallAabbj.m_minIndices[3];
					if (a <= b)
					{
						pair.x = a;  //store the original index in the unsorted aabb array
						pair.y = b;
					}
					else
					{
						pair.x = b;  //store the original index in the unsorted aabb array
						pair.y = a;
					}
					hostPairs.push_back(pair);
				}
			}
		}
	}

	{
		int numSmallAabbs = m_smallAabbsMappingCPU.size();
		for (int i = 0; i < numSmallAabbs; i++)
		{
			b3SapAabb smallAabbi = m_allAabbsCPU[m_smallAabbsMappingCPU[i]];

			//float reference = smallAabbi.m_max[axis];
			int numLargeAabbs = m_largeAabbsMappingCPU.size();

			for (int j = 0; j < numLargeAabbs; j++)
			{
				b3SapAabb largeAabbj = m_allAabbsCPU[m_largeAabbsMappingCPU[j]];
				if (TestAabbAgainstAabb2((b3Vector3&)smallAabbi.m_min, (b3Vector3&)smallAabbi.m_max,
										 (b3Vector3&)largeAabbj.m_min, (b3Vector3&)largeAabbj.m_max))
				{
					b3Int4 pair;
					int a = largeAabbj.m_minIndices[3];
					int b = smallAabbi.m_minIndices[3];
					if (a <= b)
					{
						pair.x = a;
						pair.y = b;  //store the original index in the unsorted aabb array
					}
					else
					{
						pair.x = b;
						pair.y = a;  //store the original index in the unsorted aabb array
					}

					hostPairs.push_back(pair);
				}
			}
		}
	}

	if (hostPairs.size() > maxPairs)
	{
		hostPairs.resize(maxPairs);
	}

	if (hostPairs.size())
	{
		m_overlappingPairs.copyFromHost(hostPairs);
	}
	else
	{
		m_overlappingPairs.resize(0);
	}

	//init3dSap();
}

void b3GpuSapBroadphase::reset()
{
	m_allAabbsGPU.resize(0);
	m_allAabbsCPU.resize(0);

	m_smallAabbsMappingGPU.resize(0);
	m_smallAabbsMappingCPU.resize(0);

	m_pairCount.resize(0);

	m_largeAabbsMappingGPU.resize(0);
	m_largeAabbsMappingCPU.resize(0);
}

void b3GpuSapBroadphase::calculateOverlappingPairs(int maxPairs)
{
	if (m_sapKernel == 0)
	{
		calculateOverlappingPairsHost(maxPairs);
		return;
	}

	//if (m_currentBuffer>=0)
	//	return calculateOverlappingPairsHostIncremental3Sap();

	//calculateOverlappingPairsHost(maxPairs);

	B3_PROFILE("GPU 1-axis SAP calculateOverlappingPairs");

	int axis = 0;

	{
		//bool syncOnHost = false;

		int numSmallAabbs = m_smallAabbsMappingCPU.size();
		if (m_prefixScanFloat4 && numSmallAabbs)
		{
			B3_PROFILE("GPU compute best variance axis");

			if (m_dst.size() != (numSmallAabbs + 1))
			{
				m_dst.resize(numSmallAabbs + 128);
				m_sum.resize(numSmallAabbs + 128);
				m_sum2.resize(numSmallAabbs + 128);
				m_sum.at(numSmallAabbs) = b3MakeVector3(0, 0, 0);   //slow?
				m_sum2.at(numSmallAabbs) = b3MakeVector3(0, 0, 0);  //slow?
			}

			b3LauncherCL launcher(m_queue, m_prepareSumVarianceKernel, "m_prepareSumVarianceKernel");
			launcher.setBuffer(m_allAabbsGPU.getBufferCL());

			launcher.setBuffer(m_smallAabbsMappingGPU.getBufferCL());
			launcher.setBuffer(m_sum.getBufferCL());
			launcher.setBuffer(m_sum2.getBufferCL());
			launcher.setConst(numSmallAabbs);
			int num = numSmallAabbs;
			launcher.launch1D(num);

			b3Vector3 s;
			b3Vector3 s2;
			m_prefixScanFloat4->execute(m_sum, m_dst, numSmallAabbs + 1, &s);
			m_prefixScanFloat4->execute(m_sum2, m_dst, numSmallAabbs + 1, &s2);

			b3Vector3 v = s2 - (s * s) / (float)numSmallAabbs;

			if (v[1] > v[0])
				axis = 1;
			if (v[2] > v[axis])
				axis = 2;
		}

		m_gpuSmallSortData.resize(numSmallAabbs);

#if 1
		if (m_smallAabbsMappingGPU.size())
		{
			B3_PROFILE("flipFloatKernel");
			b3BufferInfoCL bInfo[] = {
				b3BufferInfoCL(m_allAabbsGPU.getBufferCL(), true),
				b3BufferInfoCL(m_smallAabbsMappingGPU.getBufferCL(), true),
				b3BufferInfoCL(m_gpuSmallSortData.getBufferCL())};
			b3LauncherCL launcher(m_queue, m_flipFloatKernel, "m_flipFloatKernel");
			launcher.setBuffers(bInfo, sizeof(bInfo) / sizeof(b3BufferInfoCL));
			launcher.setConst(numSmallAabbs);
			launcher.setConst(axis);

			int num = numSmallAabbs;
			launcher.launch1D(num);
			clFinish(m_queue);
		}

		if (m_gpuSmallSortData.size())
		{
			B3_PROFILE("gpu radix sort");
			m_sorter->execute(m_gpuSmallSortData);
			clFinish(m_queue);
		}

		m_gpuSmallSortedAabbs.resize(numSmallAabbs);
		if (numSmallAabbs)
		{
			B3_PROFILE("scatterKernel");

			b3BufferInfoCL bInfo[] = {
				b3BufferInfoCL(m_allAabbsGPU.getBufferCL(), true),
				b3BufferInfoCL(m_smallAabbsMappingGPU.getBufferCL(), true),
				b3BufferInfoCL(m_gpuSmallSortData.getBufferCL(), true),
				b3BufferInfoCL(m_gpuSmallSortedAabbs.getBufferCL())};
			b3LauncherCL launcher(m_queue, m_scatterKernel, "m_scatterKernel ");
			launcher.setBuffers(bInfo, sizeof(bInfo) / sizeof(b3BufferInfoCL));
			launcher.setConst(numSmallAabbs);
			int num = numSmallAabbs;
			launcher.launch1D(num);
			clFinish(m_queue);
		}

		m_overlappingPairs.resize(maxPairs);

		m_pairCount.resize(0);
		m_pairCount.push_back(0);
		int numPairs = 0;

		{
			int numLargeAabbs = m_largeAabbsMappingGPU.size();
			if (numLargeAabbs && numSmallAabbs)
			{
				//@todo
				B3_PROFILE("sap2Kernel");
				b3BufferInfoCL bInfo[] = {
					b3BufferInfoCL(m_allAabbsGPU.getBufferCL()),
					b3BufferInfoCL(m_largeAabbsMappingGPU.getBufferCL()),
					b3BufferInfoCL(m_smallAabbsMappingGPU.getBufferCL()),
					b3BufferInfoCL(m_overlappingPairs.getBufferCL()),
					b3BufferInfoCL(m_pairCount.getBufferCL())};
				b3LauncherCL launcher(m_queue, m_sap2Kernel, "m_sap2Kernel");
				launcher.setBuffers(bInfo, sizeof(bInfo) / sizeof(b3BufferInfoCL));
				launcher.setConst(numLargeAabbs);
				launcher.setConst(numSmallAabbs);
				launcher.setConst(axis);
				launcher.setConst(maxPairs);
				//@todo: use actual maximum work item sizes of the device instead of hardcoded values
				launcher.launch2D(numLargeAabbs, numSmallAabbs, 4, 64);

				numPairs = m_pairCount.at(0);
				if (numPairs > maxPairs)
				{
					b3Error("Error running out of pairs: numPairs = %d, maxPairs = %d.\n", numPairs, maxPairs);
					numPairs = maxPairs;
				}
			}
		}
		if (m_gpuSmallSortedAabbs.size())
		{
			B3_PROFILE("sapKernel");
			b3BufferInfoCL bInfo[] = {b3BufferInfoCL(m_gpuSmallSortedAabbs.getBufferCL()), b3BufferInfoCL(m_overlappingPairs.getBufferCL()), b3BufferInfoCL(m_pairCount.getBufferCL())};
			b3LauncherCL launcher(m_queue, m_sapKernel, "m_sapKernel");
			launcher.setBuffers(bInfo, sizeof(bInfo) / sizeof(b3BufferInfoCL));
			launcher.setConst(numSmallAabbs);
			launcher.setConst(axis);
			launcher.setConst(maxPairs);

			int num = numSmallAabbs;
#if 0                
                int buffSize = launcher.getSerializationBufferSize();
                unsigned char* buf = new unsigned char[buffSize+sizeof(int)];
                for (int i=0;i<buffSize+1;i++)
                {
                    unsigned char* ptr = (unsigned char*)&buf[i];
                    *ptr = 0xff;
                }
                int actualWrite = launcher.serializeArguments(buf,buffSize);
                
                unsigned char* cptr = (unsigned char*)&buf[buffSize];
    //            printf("buf[buffSize] = %d\n",*cptr);
                
                assert(buf[buffSize]==0xff);//check for buffer overrun
                int* ptr = (int*)&buf[buffSize];
                
                *ptr = num;
                
                FILE* f = fopen("m_sapKernelArgs.bin","wb");
                fwrite(buf,buffSize+sizeof(int),1,f);
                fclose(f);
#endif  //

			launcher.launch1D(num);
			clFinish(m_queue);

			numPairs = m_pairCount.at(0);
			if (numPairs > maxPairs)
			{
				b3Error("Error running out of pairs: numPairs = %d, maxPairs = %d.\n", numPairs, maxPairs);
				numPairs = maxPairs;
				m_pairCount.resize(0);
				m_pairCount.push_back(maxPairs);
			}
		}

#else
		int numPairs = 0;

		b3LauncherCL launcher(m_queue, m_sapKernel);

		const char* fileName = "m_sapKernelArgs.bin";
		FILE* f = fopen(fileName, "rb");
		if (f)
		{
			int sizeInBytes = 0;
			if (fseek(f, 0, SEEK_END) || (sizeInBytes = ftell(f)) == EOF || fseek(f, 0, SEEK_SET))
			{
				printf("error, cannot get file size\n");
				exit(0);
			}

			unsigned char* buf = (unsigned char*)malloc(sizeInBytes);
			fread(buf, sizeInBytes, 1, f);
			int serializedBytes = launcher.deserializeArgs(buf, sizeInBytes, m_context);
			int num = *(int*)&buf[serializedBytes];
			launcher.launch1D(num);

			b3OpenCLArray<int> pairCount(m_context, m_queue);
			int numElements = launcher.m_arrays[2]->size() / sizeof(int);
			pairCount.setFromOpenCLBuffer(launcher.m_arrays[2]->getBufferCL(), numElements);
			numPairs = pairCount.at(0);
			//printf("overlapping pairs = %d\n",numPairs);
			b3AlignedObjectArray<b3Int4> hostOoverlappingPairs;
			b3OpenCLArray<b3Int4> tmpGpuPairs(m_context, m_queue);
			tmpGpuPairs.setFromOpenCLBuffer(launcher.m_arrays[1]->getBufferCL(), numPairs);

			tmpGpuPairs.copyToHost(hostOoverlappingPairs);
			m_overlappingPairs.copyFromHost(hostOoverlappingPairs);
			//printf("hello %d\n", m_overlappingPairs.size());
			free(buf);
			fclose(f);
		}
		else
		{
			printf("error: cannot find file %s\n", fileName);
		}

		clFinish(m_queue);

#endif

		m_overlappingPairs.resize(numPairs);

	}  //B3_PROFILE("GPU_RADIX SORT");
	   //init3dSap();
}

void b3GpuSapBroadphase::writeAabbsToGpu()
{
	m_smallAabbsMappingGPU.copyFromHost(m_smallAabbsMappingCPU);
	m_largeAabbsMappingGPU.copyFromHost(m_largeAabbsMappingCPU);

	m_allAabbsGPU.copyFromHost(m_allAabbsCPU);  //might not be necessary, the 'setupGpuAabbsFull' already takes care of this
}

void b3GpuSapBroadphase::createLargeProxy(const b3Vector3& aabbMin, const b3Vector3& aabbMax, int userPtr, int collisionFilterGroup, int collisionFilterMask)
{
	int index = userPtr;
	b3SapAabb aabb;
	for (int i = 0; i < 4; i++)
	{
		aabb.m_min[i] = aabbMin[i];
		aabb.m_max[i] = aabbMax[i];
	}
	aabb.m_minIndices[3] = index;
	aabb.m_signedMaxIndices[3] = m_allAabbsCPU.size();
	m_largeAabbsMappingCPU.push_back(m_allAabbsCPU.size());

	m_allAabbsCPU.push_back(aabb);
}

void b3GpuSapBroadphase::createProxy(const b3Vector3& aabbMin, const b3Vector3& aabbMax, int userPtr, int collisionFilterGroup, int collisionFilterMask)
{
	int index = userPtr;
	b3SapAabb aabb;
	for (int i = 0; i < 4; i++)
	{
		aabb.m_min[i] = aabbMin[i];
		aabb.m_max[i] = aabbMax[i];
	}
	aabb.m_minIndices[3] = index;
	aabb.m_signedMaxIndices[3] = m_allAabbsCPU.size();
	m_smallAabbsMappingCPU.push_back(m_allAabbsCPU.size());

	m_allAabbsCPU.push_back(aabb);
}

cl_mem b3GpuSapBroadphase::getAabbBufferWS()
{
	return m_allAabbsGPU.getBufferCL();
}

int b3GpuSapBroadphase::getNumOverlap()
{
	return m_overlappingPairs.size();
}
cl_mem b3GpuSapBroadphase::getOverlappingPairBuffer()
{
	return m_overlappingPairs.getBufferCL();
}

b3OpenCLArray<b3Int4>& b3GpuSapBroadphase::getOverlappingPairsGPU()
{
	return m_overlappingPairs;
}
b3OpenCLArray<int>& b3GpuSapBroadphase::getSmallAabbIndicesGPU()
{
	return m_smallAabbsMappingGPU;
}
b3OpenCLArray<int>& b3GpuSapBroadphase::getLargeAabbIndicesGPU()
{
	return m_largeAabbsMappingGPU;
}
