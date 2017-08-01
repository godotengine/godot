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

#ifndef B3_GPU_PARALLEL_LINEAR_BVH_H
#define B3_GPU_PARALLEL_LINEAR_BVH_H

//#include "Bullet3Collision/BroadPhaseCollision/shared/b3Aabb.h"
#include "Bullet3OpenCL/BroadphaseCollision/b3SapAabb.h"
#include "Bullet3Common/shared/b3Int2.h"
#include "Bullet3Common/shared/b3Int4.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3RaycastInfo.h"

#include "Bullet3OpenCL/ParallelPrimitives/b3FillCL.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3RadixSort32CL.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3PrefixScanCL.h"

#include "Bullet3OpenCL/BroadphaseCollision/kernels/parallelLinearBvhKernels.h"

#define b3Int64 cl_long

///@brief GPU Parallel Linearized Bounding Volume Heirarchy(LBVH) that is reconstructed every frame
///@remarks
///See presentation in docs/b3GpuParallelLinearBvh.pdf for algorithm details.
///@par
///Related papers: \n
///"Fast BVH Construction on GPUs" [Lauterbach et al. 2009] \n
///"Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d trees" [Karras 2012] \n
///@par
///The basic algorithm for building the BVH as presented in [Lauterbach et al. 2009] consists of 4 stages:
/// - [fully parallel] Assign morton codes for each AABB using its center (after quantizing the AABB centers into a virtual grid) 
/// - [fully parallel] Sort morton codes
/// - [somewhat parallel] Build binary radix tree (assign parent/child pointers for internal nodes of the BVH) 
/// - [somewhat parallel] Set internal node AABBs 
///@par
///[Karras 2012] improves on the algorithm by introducing fully parallel methods for the last 2 stages.
///The BVH implementation here shares many concepts with [Karras 2012], but a different method is used for constructing the tree.
///Instead of searching for the child nodes of each internal node, we search for the parent node of each node.
///Additionally, a non-atomic traversal that starts from the leaf nodes and moves towards the root node is used to set the AABBs.
class b3GpuParallelLinearBvh
{
	cl_command_queue m_queue;
	
	cl_program m_parallelLinearBvhProgram;
	
	cl_kernel m_separateAabbsKernel;
	cl_kernel m_findAllNodesMergedAabbKernel;
	cl_kernel m_assignMortonCodesAndAabbIndiciesKernel;
	
	//Binary radix tree construction kernels
	cl_kernel m_computeAdjacentPairCommonPrefixKernel;
	cl_kernel m_buildBinaryRadixTreeLeafNodesKernel;
	cl_kernel m_buildBinaryRadixTreeInternalNodesKernel;
	cl_kernel m_findDistanceFromRootKernel;
	cl_kernel m_buildBinaryRadixTreeAabbsRecursiveKernel;
	
	cl_kernel m_findLeafIndexRangesKernel;
	
	//Traversal kernels
	cl_kernel m_plbvhCalculateOverlappingPairsKernel;
	cl_kernel m_plbvhRayTraverseKernel;
	cl_kernel m_plbvhLargeAabbAabbTestKernel;
	cl_kernel m_plbvhLargeAabbRayTestKernel;
	
	b3RadixSort32CL m_radixSorter;
	
	//1 element
	b3OpenCLArray<int> m_rootNodeIndex;							//Most significant bit(0x80000000) is set to indicate internal node
	b3OpenCLArray<int> m_maxDistanceFromRoot;					//Max number of internal nodes between an internal node and the root node
	b3OpenCLArray<int> m_temp;									//Used to hold the number of pairs in calculateOverlappingPairs()
	
	//1 element per internal node (number_of_internal_nodes == number_of_leaves - 1)
	b3OpenCLArray<b3SapAabb> m_internalNodeAabbs;
	b3OpenCLArray<b3Int2> m_internalNodeLeafIndexRanges;		//x == min leaf index, y == max leaf index
	b3OpenCLArray<b3Int2> m_internalNodeChildNodes;				//x == left child, y == right child; msb(0x80000000) is set to indicate internal node
	b3OpenCLArray<int> m_internalNodeParentNodes;				//For parent node index, msb(0x80000000) is not set since it is always internal
	
	//1 element per internal node; for binary radix tree construction
	b3OpenCLArray<b3Int64> m_commonPrefixes;
	b3OpenCLArray<int> m_commonPrefixLengths;
	b3OpenCLArray<int> m_distanceFromRoot;						//Number of internal nodes between this node and the root
	
	//1 element per leaf node (leaf nodes only include small AABBs)
	b3OpenCLArray<int> m_leafNodeParentNodes;					//For parent node index, msb(0x80000000) is not set since it is always internal
	b3OpenCLArray<b3SortData> m_mortonCodesAndAabbIndicies;		//m_key == morton code, m_value == aabb index in m_leafNodeAabbs
	b3OpenCLArray<b3SapAabb> m_mergedAabb;						//m_mergedAabb[0] contains the merged AABB of all leaf nodes
	b3OpenCLArray<b3SapAabb> m_leafNodeAabbs;					//Contains only small AABBs
	
	//1 element per large AABB, which is not stored in the BVH
	b3OpenCLArray<b3SapAabb> m_largeAabbs;
	
public:
	b3GpuParallelLinearBvh(cl_context context, cl_device_id device, cl_command_queue queue);
	virtual ~b3GpuParallelLinearBvh();
	
	///Must be called before any other function
	void build(const b3OpenCLArray<b3SapAabb>& worldSpaceAabbs, const b3OpenCLArray<int>& smallAabbIndices, 
				const b3OpenCLArray<int>& largeAabbIndices);
	
	///calculateOverlappingPairs() uses the worldSpaceAabbs parameter of b3GpuParallelLinearBvh::build() as the query AABBs.
	///@param out_overlappingPairs The size() of this array is used to determine the max number of pairs.
	///If the number of overlapping pairs is < out_overlappingPairs.size(), out_overlappingPairs is resized.
	void calculateOverlappingPairs(b3OpenCLArray<b3Int4>& out_overlappingPairs);
	
	///@param out_numRigidRayPairs Array of length 1; contains the number of detected ray-rigid AABB intersections;
	///this value may be greater than out_rayRigidPairs.size() if out_rayRigidPairs is not large enough.
	///@param out_rayRigidPairs Contains an array of rays intersecting rigid AABBs; x == ray index, y == rigid body index.
	///If the size of this array is insufficient to hold all ray-rigid AABB intersections, additional intersections are discarded.
	void testRaysAgainstBvhAabbs(const b3OpenCLArray<b3RayInfo>& rays, 
								b3OpenCLArray<int>& out_numRayRigidPairs, b3OpenCLArray<b3Int2>& out_rayRigidPairs);
								
private:
	void constructBinaryRadixTree();
};

#endif
