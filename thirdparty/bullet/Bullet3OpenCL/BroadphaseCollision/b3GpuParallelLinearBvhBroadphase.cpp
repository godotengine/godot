/*
This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Initial Author Jackson Lee, 2014

#include "b3GpuParallelLinearBvhBroadphase.h"

b3GpuParallelLinearBvhBroadphase::b3GpuParallelLinearBvhBroadphase(cl_context context, cl_device_id device, cl_command_queue queue) : m_plbvh(context, device, queue),

																																	  m_overlappingPairsGpu(context, queue),

																																	  m_aabbsGpu(context, queue),
																																	  m_smallAabbsMappingGpu(context, queue),
																																	  m_largeAabbsMappingGpu(context, queue)
{
}

void b3GpuParallelLinearBvhBroadphase::createProxy(const b3Vector3& aabbMin, const b3Vector3& aabbMax, int userPtr, int collisionFilterGroup, int collisionFilterMask)
{
	int newAabbIndex = m_aabbsCpu.size();

	b3SapAabb aabb;
	aabb.m_minVec = aabbMin;
	aabb.m_maxVec = aabbMax;

	aabb.m_minIndices[3] = userPtr;
	aabb.m_signedMaxIndices[3] = newAabbIndex;

	m_smallAabbsMappingCpu.push_back(newAabbIndex);

	m_aabbsCpu.push_back(aabb);
}
void b3GpuParallelLinearBvhBroadphase::createLargeProxy(const b3Vector3& aabbMin, const b3Vector3& aabbMax, int userPtr, int collisionFilterGroup, int collisionFilterMask)
{
	int newAabbIndex = m_aabbsCpu.size();

	b3SapAabb aabb;
	aabb.m_minVec = aabbMin;
	aabb.m_maxVec = aabbMax;

	aabb.m_minIndices[3] = userPtr;
	aabb.m_signedMaxIndices[3] = newAabbIndex;

	m_largeAabbsMappingCpu.push_back(newAabbIndex);

	m_aabbsCpu.push_back(aabb);
}

void b3GpuParallelLinearBvhBroadphase::calculateOverlappingPairs(int maxPairs)
{
	//Reconstruct BVH
	m_plbvh.build(m_aabbsGpu, m_smallAabbsMappingGpu, m_largeAabbsMappingGpu);

	//
	m_overlappingPairsGpu.resize(maxPairs);
	m_plbvh.calculateOverlappingPairs(m_overlappingPairsGpu);
}
void b3GpuParallelLinearBvhBroadphase::calculateOverlappingPairsHost(int maxPairs)
{
	b3Assert(0);  //CPU version not implemented
}

void b3GpuParallelLinearBvhBroadphase::writeAabbsToGpu()
{
	m_aabbsGpu.copyFromHost(m_aabbsCpu);
	m_smallAabbsMappingGpu.copyFromHost(m_smallAabbsMappingCpu);
	m_largeAabbsMappingGpu.copyFromHost(m_largeAabbsMappingCpu);
}
