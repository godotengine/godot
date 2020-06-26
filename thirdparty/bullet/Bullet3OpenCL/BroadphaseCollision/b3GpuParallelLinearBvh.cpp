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

#include "Bullet3OpenCL/Initialize/b3OpenCLUtils.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3LauncherCL.h"

#include "b3GpuParallelLinearBvh.h"

b3GpuParallelLinearBvh::b3GpuParallelLinearBvh(cl_context context, cl_device_id device, cl_command_queue queue) : m_queue(queue),
																												  m_radixSorter(context, device, queue),

																												  m_rootNodeIndex(context, queue),
																												  m_maxDistanceFromRoot(context, queue),
																												  m_temp(context, queue),

																												  m_internalNodeAabbs(context, queue),
																												  m_internalNodeLeafIndexRanges(context, queue),
																												  m_internalNodeChildNodes(context, queue),
																												  m_internalNodeParentNodes(context, queue),

																												  m_commonPrefixes(context, queue),
																												  m_commonPrefixLengths(context, queue),
																												  m_distanceFromRoot(context, queue),

																												  m_leafNodeParentNodes(context, queue),
																												  m_mortonCodesAndAabbIndicies(context, queue),
																												  m_mergedAabb(context, queue),
																												  m_leafNodeAabbs(context, queue),

																												  m_largeAabbs(context, queue)
{
	m_rootNodeIndex.resize(1);
	m_maxDistanceFromRoot.resize(1);
	m_temp.resize(1);

	//
	const char CL_PROGRAM_PATH[] = "src/Bullet3OpenCL/BroadphaseCollision/kernels/parallelLinearBvh.cl";

	const char* kernelSource = parallelLinearBvhCL;  //parallelLinearBvhCL.h
	cl_int error;
	char* additionalMacros = 0;
	m_parallelLinearBvhProgram = b3OpenCLUtils::compileCLProgramFromString(context, device, kernelSource, &error, additionalMacros, CL_PROGRAM_PATH);
	b3Assert(m_parallelLinearBvhProgram);

	m_separateAabbsKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "separateAabbs", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_separateAabbsKernel);
	m_findAllNodesMergedAabbKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "findAllNodesMergedAabb", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_findAllNodesMergedAabbKernel);
	m_assignMortonCodesAndAabbIndiciesKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "assignMortonCodesAndAabbIndicies", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_assignMortonCodesAndAabbIndiciesKernel);

	m_computeAdjacentPairCommonPrefixKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "computeAdjacentPairCommonPrefix", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_computeAdjacentPairCommonPrefixKernel);
	m_buildBinaryRadixTreeLeafNodesKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "buildBinaryRadixTreeLeafNodes", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_buildBinaryRadixTreeLeafNodesKernel);
	m_buildBinaryRadixTreeInternalNodesKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "buildBinaryRadixTreeInternalNodes", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_buildBinaryRadixTreeInternalNodesKernel);
	m_findDistanceFromRootKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "findDistanceFromRoot", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_findDistanceFromRootKernel);
	m_buildBinaryRadixTreeAabbsRecursiveKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "buildBinaryRadixTreeAabbsRecursive", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_buildBinaryRadixTreeAabbsRecursiveKernel);

	m_findLeafIndexRangesKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "findLeafIndexRanges", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_findLeafIndexRangesKernel);

	m_plbvhCalculateOverlappingPairsKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "plbvhCalculateOverlappingPairs", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_plbvhCalculateOverlappingPairsKernel);
	m_plbvhRayTraverseKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "plbvhRayTraverse", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_plbvhRayTraverseKernel);
	m_plbvhLargeAabbAabbTestKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "plbvhLargeAabbAabbTest", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_plbvhLargeAabbAabbTestKernel);
	m_plbvhLargeAabbRayTestKernel = b3OpenCLUtils::compileCLKernelFromString(context, device, kernelSource, "plbvhLargeAabbRayTest", &error, m_parallelLinearBvhProgram, additionalMacros);
	b3Assert(m_plbvhLargeAabbRayTestKernel);
}

b3GpuParallelLinearBvh::~b3GpuParallelLinearBvh()
{
	clReleaseKernel(m_separateAabbsKernel);
	clReleaseKernel(m_findAllNodesMergedAabbKernel);
	clReleaseKernel(m_assignMortonCodesAndAabbIndiciesKernel);

	clReleaseKernel(m_computeAdjacentPairCommonPrefixKernel);
	clReleaseKernel(m_buildBinaryRadixTreeLeafNodesKernel);
	clReleaseKernel(m_buildBinaryRadixTreeInternalNodesKernel);
	clReleaseKernel(m_findDistanceFromRootKernel);
	clReleaseKernel(m_buildBinaryRadixTreeAabbsRecursiveKernel);

	clReleaseKernel(m_findLeafIndexRangesKernel);

	clReleaseKernel(m_plbvhCalculateOverlappingPairsKernel);
	clReleaseKernel(m_plbvhRayTraverseKernel);
	clReleaseKernel(m_plbvhLargeAabbAabbTestKernel);
	clReleaseKernel(m_plbvhLargeAabbRayTestKernel);

	clReleaseProgram(m_parallelLinearBvhProgram);
}

void b3GpuParallelLinearBvh::build(const b3OpenCLArray<b3SapAabb>& worldSpaceAabbs, const b3OpenCLArray<int>& smallAabbIndices,
								   const b3OpenCLArray<int>& largeAabbIndices)
{
	B3_PROFILE("b3ParallelLinearBvh::build()");

	int numLargeAabbs = largeAabbIndices.size();
	int numSmallAabbs = smallAabbIndices.size();

	//Since all AABBs(both large and small) are input as a contiguous array,
	//with 2 additional arrays used to indicate the indices of large and small AABBs,
	//it is necessary to separate the AABBs so that the large AABBs will not degrade the quality of the BVH.
	{
		B3_PROFILE("Separate large and small AABBs");

		m_largeAabbs.resize(numLargeAabbs);
		m_leafNodeAabbs.resize(numSmallAabbs);

		//Write large AABBs into m_largeAabbs
		{
			b3BufferInfoCL bufferInfo[] =
				{
					b3BufferInfoCL(worldSpaceAabbs.getBufferCL()),
					b3BufferInfoCL(largeAabbIndices.getBufferCL()),

					b3BufferInfoCL(m_largeAabbs.getBufferCL())};

			b3LauncherCL launcher(m_queue, m_separateAabbsKernel, "m_separateAabbsKernel");
			launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
			launcher.setConst(numLargeAabbs);

			launcher.launch1D(numLargeAabbs);
		}

		//Write small AABBs into m_leafNodeAabbs
		{
			b3BufferInfoCL bufferInfo[] =
				{
					b3BufferInfoCL(worldSpaceAabbs.getBufferCL()),
					b3BufferInfoCL(smallAabbIndices.getBufferCL()),

					b3BufferInfoCL(m_leafNodeAabbs.getBufferCL())};

			b3LauncherCL launcher(m_queue, m_separateAabbsKernel, "m_separateAabbsKernel");
			launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
			launcher.setConst(numSmallAabbs);

			launcher.launch1D(numSmallAabbs);
		}

		clFinish(m_queue);
	}

	//
	int numLeaves = numSmallAabbs;  //Number of leaves in the BVH == Number of rigid bodies with small AABBs
	int numInternalNodes = numLeaves - 1;

	if (numLeaves < 2)
	{
		//Number of leaf nodes is checked in calculateOverlappingPairs() and testRaysAgainstBvhAabbs(),
		//so it does not matter if numLeaves == 0 and rootNodeIndex == -1
		int rootNodeIndex = numLeaves - 1;
		m_rootNodeIndex.copyFromHostPointer(&rootNodeIndex, 1);

		//Since the AABBs need to be rearranged(sorted) for the BVH construction algorithm,
		//m_mortonCodesAndAabbIndicies.m_value is used to map a sorted AABB index to the unsorted AABB index
		//instead of directly moving the AABBs. It needs to be set for the ray cast traversal kernel to work.
		//( m_mortonCodesAndAabbIndicies[].m_value == unsorted index == index of m_leafNodeAabbs )
		if (numLeaves == 1)
		{
			b3SortData leaf;
			leaf.m_value = 0;  //1 leaf so index is always 0; leaf.m_key does not need to be set

			m_mortonCodesAndAabbIndicies.resize(1);
			m_mortonCodesAndAabbIndicies.copyFromHostPointer(&leaf, 1);
		}

		return;
	}

	//
	{
		m_internalNodeAabbs.resize(numInternalNodes);
		m_internalNodeLeafIndexRanges.resize(numInternalNodes);
		m_internalNodeChildNodes.resize(numInternalNodes);
		m_internalNodeParentNodes.resize(numInternalNodes);

		m_commonPrefixes.resize(numInternalNodes);
		m_commonPrefixLengths.resize(numInternalNodes);
		m_distanceFromRoot.resize(numInternalNodes);

		m_leafNodeParentNodes.resize(numLeaves);
		m_mortonCodesAndAabbIndicies.resize(numLeaves);
		m_mergedAabb.resize(numLeaves);
	}

	//Find the merged AABB of all small AABBs; this is used to define the size of
	//each cell in the virtual grid for the next kernel(2^10 cells in each dimension).
	{
		B3_PROFILE("Find AABB of merged nodes");

		m_mergedAabb.copyFromOpenCLArray(m_leafNodeAabbs);  //Need to make a copy since the kernel modifies the array

		for (int numAabbsNeedingMerge = numLeaves; numAabbsNeedingMerge >= 2;
			 numAabbsNeedingMerge = numAabbsNeedingMerge / 2 + numAabbsNeedingMerge % 2)
		{
			b3BufferInfoCL bufferInfo[] =
				{
					b3BufferInfoCL(m_mergedAabb.getBufferCL())  //Resulting AABB is stored in m_mergedAabb[0]
				};

			b3LauncherCL launcher(m_queue, m_findAllNodesMergedAabbKernel, "m_findAllNodesMergedAabbKernel");
			launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
			launcher.setConst(numAabbsNeedingMerge);

			launcher.launch1D(numAabbsNeedingMerge);
		}

		clFinish(m_queue);
	}

	//Insert the center of the AABBs into a virtual grid,
	//then convert the discrete grid coordinates into a morton code
	//For each element in m_mortonCodesAndAabbIndicies, set
	//	m_key == morton code (value to sort by)
	//	m_value == small AABB index
	{
		B3_PROFILE("Assign morton codes");

		b3BufferInfoCL bufferInfo[] =
			{
				b3BufferInfoCL(m_leafNodeAabbs.getBufferCL()),
				b3BufferInfoCL(m_mergedAabb.getBufferCL()),
				b3BufferInfoCL(m_mortonCodesAndAabbIndicies.getBufferCL())};

		b3LauncherCL launcher(m_queue, m_assignMortonCodesAndAabbIndiciesKernel, "m_assignMortonCodesAndAabbIndiciesKernel");
		launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
		launcher.setConst(numLeaves);

		launcher.launch1D(numLeaves);
		clFinish(m_queue);
	}

	//
	{
		B3_PROFILE("Sort leaves by morton codes");

		m_radixSorter.execute(m_mortonCodesAndAabbIndicies);
		clFinish(m_queue);
	}

	//
	constructBinaryRadixTree();

	//Since it is a sorted binary radix tree, each internal node contains a contiguous subset of leaf node indices.
	//The root node contains leaf node indices in the range [0, numLeafNodes - 1].
	//The child nodes of each node split their parent's index range into 2 contiguous halves.
	//
	//For example, if the root has indices [0, 31], its children might partition that range into [0, 11] and [12, 31].
	//The next level in the tree could then split those ranges into [0, 2], [3, 11], [12, 22], and [23, 31].
	//
	//This property can be used for optimizing calculateOverlappingPairs(), to avoid testing each AABB pair twice
	{
		B3_PROFILE("m_findLeafIndexRangesKernel");

		b3BufferInfoCL bufferInfo[] =
			{
				b3BufferInfoCL(m_internalNodeChildNodes.getBufferCL()),
				b3BufferInfoCL(m_internalNodeLeafIndexRanges.getBufferCL())};

		b3LauncherCL launcher(m_queue, m_findLeafIndexRangesKernel, "m_findLeafIndexRangesKernel");
		launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
		launcher.setConst(numInternalNodes);

		launcher.launch1D(numInternalNodes);
		clFinish(m_queue);
	}
}

void b3GpuParallelLinearBvh::calculateOverlappingPairs(b3OpenCLArray<b3Int4>& out_overlappingPairs)
{
	int maxPairs = out_overlappingPairs.size();
	b3OpenCLArray<int>& numPairsGpu = m_temp;

	int reset = 0;
	numPairsGpu.copyFromHostPointer(&reset, 1);

	//
	if (m_leafNodeAabbs.size() > 1)
	{
		B3_PROFILE("PLBVH small-small AABB test");

		int numQueryAabbs = m_leafNodeAabbs.size();

		b3BufferInfoCL bufferInfo[] =
			{
				b3BufferInfoCL(m_leafNodeAabbs.getBufferCL()),

				b3BufferInfoCL(m_rootNodeIndex.getBufferCL()),
				b3BufferInfoCL(m_internalNodeChildNodes.getBufferCL()),
				b3BufferInfoCL(m_internalNodeAabbs.getBufferCL()),
				b3BufferInfoCL(m_internalNodeLeafIndexRanges.getBufferCL()),
				b3BufferInfoCL(m_mortonCodesAndAabbIndicies.getBufferCL()),

				b3BufferInfoCL(numPairsGpu.getBufferCL()),
				b3BufferInfoCL(out_overlappingPairs.getBufferCL())};

		b3LauncherCL launcher(m_queue, m_plbvhCalculateOverlappingPairsKernel, "m_plbvhCalculateOverlappingPairsKernel");
		launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
		launcher.setConst(maxPairs);
		launcher.setConst(numQueryAabbs);

		launcher.launch1D(numQueryAabbs);
		clFinish(m_queue);
	}

	int numLargeAabbRigids = m_largeAabbs.size();
	if (numLargeAabbRigids > 0 && m_leafNodeAabbs.size() > 0)
	{
		B3_PROFILE("PLBVH large-small AABB test");

		int numQueryAabbs = m_leafNodeAabbs.size();

		b3BufferInfoCL bufferInfo[] =
			{
				b3BufferInfoCL(m_leafNodeAabbs.getBufferCL()),
				b3BufferInfoCL(m_largeAabbs.getBufferCL()),

				b3BufferInfoCL(numPairsGpu.getBufferCL()),
				b3BufferInfoCL(out_overlappingPairs.getBufferCL())};

		b3LauncherCL launcher(m_queue, m_plbvhLargeAabbAabbTestKernel, "m_plbvhLargeAabbAabbTestKernel");
		launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
		launcher.setConst(maxPairs);
		launcher.setConst(numLargeAabbRigids);
		launcher.setConst(numQueryAabbs);

		launcher.launch1D(numQueryAabbs);
		clFinish(m_queue);
	}

	//
	int numPairs = -1;
	numPairsGpu.copyToHostPointer(&numPairs, 1);
	if (numPairs > maxPairs)
	{
		b3Error("Error running out of pairs: numPairs = %d, maxPairs = %d.\n", numPairs, maxPairs);
		numPairs = maxPairs;
		numPairsGpu.copyFromHostPointer(&maxPairs, 1);
	}

	out_overlappingPairs.resize(numPairs);
}

void b3GpuParallelLinearBvh::testRaysAgainstBvhAabbs(const b3OpenCLArray<b3RayInfo>& rays,
													 b3OpenCLArray<int>& out_numRayRigidPairs, b3OpenCLArray<b3Int2>& out_rayRigidPairs)
{
	B3_PROFILE("PLBVH testRaysAgainstBvhAabbs()");

	int numRays = rays.size();
	int maxRayRigidPairs = out_rayRigidPairs.size();

	int reset = 0;
	out_numRayRigidPairs.copyFromHostPointer(&reset, 1);

	//
	if (m_leafNodeAabbs.size() > 0)
	{
		B3_PROFILE("PLBVH ray test small AABB");

		b3BufferInfoCL bufferInfo[] =
			{
				b3BufferInfoCL(m_leafNodeAabbs.getBufferCL()),

				b3BufferInfoCL(m_rootNodeIndex.getBufferCL()),
				b3BufferInfoCL(m_internalNodeChildNodes.getBufferCL()),
				b3BufferInfoCL(m_internalNodeAabbs.getBufferCL()),
				b3BufferInfoCL(m_internalNodeLeafIndexRanges.getBufferCL()),
				b3BufferInfoCL(m_mortonCodesAndAabbIndicies.getBufferCL()),

				b3BufferInfoCL(rays.getBufferCL()),

				b3BufferInfoCL(out_numRayRigidPairs.getBufferCL()),
				b3BufferInfoCL(out_rayRigidPairs.getBufferCL())};

		b3LauncherCL launcher(m_queue, m_plbvhRayTraverseKernel, "m_plbvhRayTraverseKernel");
		launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
		launcher.setConst(maxRayRigidPairs);
		launcher.setConst(numRays);

		launcher.launch1D(numRays);
		clFinish(m_queue);
	}

	int numLargeAabbRigids = m_largeAabbs.size();
	if (numLargeAabbRigids > 0)
	{
		B3_PROFILE("PLBVH ray test large AABB");

		b3BufferInfoCL bufferInfo[] =
			{
				b3BufferInfoCL(m_largeAabbs.getBufferCL()),
				b3BufferInfoCL(rays.getBufferCL()),

				b3BufferInfoCL(out_numRayRigidPairs.getBufferCL()),
				b3BufferInfoCL(out_rayRigidPairs.getBufferCL())};

		b3LauncherCL launcher(m_queue, m_plbvhLargeAabbRayTestKernel, "m_plbvhLargeAabbRayTestKernel");
		launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
		launcher.setConst(numLargeAabbRigids);
		launcher.setConst(maxRayRigidPairs);
		launcher.setConst(numRays);

		launcher.launch1D(numRays);
		clFinish(m_queue);
	}

	//
	int numRayRigidPairs = -1;
	out_numRayRigidPairs.copyToHostPointer(&numRayRigidPairs, 1);

	if (numRayRigidPairs > maxRayRigidPairs)
		b3Error("Error running out of rayRigid pairs: numRayRigidPairs = %d, maxRayRigidPairs = %d.\n", numRayRigidPairs, maxRayRigidPairs);
}

void b3GpuParallelLinearBvh::constructBinaryRadixTree()
{
	B3_PROFILE("b3GpuParallelLinearBvh::constructBinaryRadixTree()");

	int numLeaves = m_leafNodeAabbs.size();
	int numInternalNodes = numLeaves - 1;

	//Each internal node is placed in between 2 leaf nodes.
	//By using this arrangement and computing the common prefix between
	//these 2 adjacent leaf nodes, it is possible to quickly construct a binary radix tree.
	{
		B3_PROFILE("m_computeAdjacentPairCommonPrefixKernel");

		b3BufferInfoCL bufferInfo[] =
			{
				b3BufferInfoCL(m_mortonCodesAndAabbIndicies.getBufferCL()),
				b3BufferInfoCL(m_commonPrefixes.getBufferCL()),
				b3BufferInfoCL(m_commonPrefixLengths.getBufferCL())};

		b3LauncherCL launcher(m_queue, m_computeAdjacentPairCommonPrefixKernel, "m_computeAdjacentPairCommonPrefixKernel");
		launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
		launcher.setConst(numInternalNodes);

		launcher.launch1D(numInternalNodes);
		clFinish(m_queue);
	}

	//For each leaf node, select its parent node by
	//comparing the 2 nearest internal nodes and assign child node indices
	{
		B3_PROFILE("m_buildBinaryRadixTreeLeafNodesKernel");

		b3BufferInfoCL bufferInfo[] =
			{
				b3BufferInfoCL(m_commonPrefixLengths.getBufferCL()),
				b3BufferInfoCL(m_leafNodeParentNodes.getBufferCL()),
				b3BufferInfoCL(m_internalNodeChildNodes.getBufferCL())};

		b3LauncherCL launcher(m_queue, m_buildBinaryRadixTreeLeafNodesKernel, "m_buildBinaryRadixTreeLeafNodesKernel");
		launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
		launcher.setConst(numLeaves);

		launcher.launch1D(numLeaves);
		clFinish(m_queue);
	}

	//For each internal node, perform 2 binary searches among the other internal nodes
	//to its left and right to find its potential parent nodes and assign child node indices
	{
		B3_PROFILE("m_buildBinaryRadixTreeInternalNodesKernel");

		b3BufferInfoCL bufferInfo[] =
			{
				b3BufferInfoCL(m_commonPrefixes.getBufferCL()),
				b3BufferInfoCL(m_commonPrefixLengths.getBufferCL()),
				b3BufferInfoCL(m_internalNodeChildNodes.getBufferCL()),
				b3BufferInfoCL(m_internalNodeParentNodes.getBufferCL()),
				b3BufferInfoCL(m_rootNodeIndex.getBufferCL())};

		b3LauncherCL launcher(m_queue, m_buildBinaryRadixTreeInternalNodesKernel, "m_buildBinaryRadixTreeInternalNodesKernel");
		launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
		launcher.setConst(numInternalNodes);

		launcher.launch1D(numInternalNodes);
		clFinish(m_queue);
	}

	//Find the number of nodes separating each internal node and the root node
	//so that the AABBs can be set using the next kernel.
	//Also determine the maximum number of nodes separating an internal node and the root node.
	{
		B3_PROFILE("m_findDistanceFromRootKernel");

		b3BufferInfoCL bufferInfo[] =
			{
				b3BufferInfoCL(m_rootNodeIndex.getBufferCL()),
				b3BufferInfoCL(m_internalNodeParentNodes.getBufferCL()),
				b3BufferInfoCL(m_maxDistanceFromRoot.getBufferCL()),
				b3BufferInfoCL(m_distanceFromRoot.getBufferCL())};

		b3LauncherCL launcher(m_queue, m_findDistanceFromRootKernel, "m_findDistanceFromRootKernel");
		launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
		launcher.setConst(numInternalNodes);

		launcher.launch1D(numInternalNodes);
		clFinish(m_queue);
	}

	//Starting from the internal nodes nearest to the leaf nodes, recursively move up
	//the tree towards the root to set the AABBs of each internal node; each internal node
	//checks its children and merges their AABBs
	{
		B3_PROFILE("m_buildBinaryRadixTreeAabbsRecursiveKernel");

		int maxDistanceFromRoot = -1;
		{
			B3_PROFILE("copy maxDistanceFromRoot to CPU");
			m_maxDistanceFromRoot.copyToHostPointer(&maxDistanceFromRoot, 1);
			clFinish(m_queue);
		}

		for (int distanceFromRoot = maxDistanceFromRoot; distanceFromRoot >= 0; --distanceFromRoot)
		{
			b3BufferInfoCL bufferInfo[] =
				{
					b3BufferInfoCL(m_distanceFromRoot.getBufferCL()),
					b3BufferInfoCL(m_mortonCodesAndAabbIndicies.getBufferCL()),
					b3BufferInfoCL(m_internalNodeChildNodes.getBufferCL()),
					b3BufferInfoCL(m_leafNodeAabbs.getBufferCL()),
					b3BufferInfoCL(m_internalNodeAabbs.getBufferCL())};

			b3LauncherCL launcher(m_queue, m_buildBinaryRadixTreeAabbsRecursiveKernel, "m_buildBinaryRadixTreeAabbsRecursiveKernel");
			launcher.setBuffers(bufferInfo, sizeof(bufferInfo) / sizeof(b3BufferInfoCL));
			launcher.setConst(maxDistanceFromRoot);
			launcher.setConst(distanceFromRoot);
			launcher.setConst(numInternalNodes);

			//It may seem inefficent to launch a thread for each internal node when a
			//much smaller number of nodes is actually processed, but this is actually
			//faster than determining the exact nodes that are ready to merge their child AABBs.
			launcher.launch1D(numInternalNodes);
		}

		clFinish(m_queue);
	}
}
