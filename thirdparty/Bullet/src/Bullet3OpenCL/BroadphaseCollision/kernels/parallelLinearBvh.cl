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

typedef float b3Scalar;
typedef float4 b3Vector3;
#define b3Max max
#define b3Min min
#define b3Sqrt sqrt

typedef struct
{
	unsigned int m_key;
	unsigned int m_value;
} SortDataCL;

typedef struct 
{
	union
	{
		float4	m_min;
		float   m_minElems[4];
		int			m_minIndices[4];
	};
	union
	{
		float4	m_max;
		float   m_maxElems[4];
		int			m_maxIndices[4];
	};
} b3AabbCL;


unsigned int interleaveBits(unsigned int x)
{
	//........ ........ ......12 3456789A	//x
	//....1..2 ..3..4.. 5..6..7. .8..9..A	//x after interleaving bits
	
	//......12 3456789A ......12 3456789A	//x ^ (x << 16)
	//11111111 ........ ........ 11111111	//0x FF 00 00 FF
	//......12 ........ ........ 3456789A	//x = (x ^ (x << 16)) & 0xFF0000FF;
	
	//......12 ........ 3456789A 3456789A	//x ^ (x <<  8)
	//......11 ........ 1111.... ....1111	//0x 03 00 F0 0F
	//......12 ........ 3456.... ....789A	//x = (x ^ (x <<  8)) & 0x0300F00F;
	
	//..12..12 ....3456 3456.... 789A789A	//x ^ (x <<  4)
	//......11 ....11.. ..11.... 11....11	//0x 03 0C 30 C3
	//......12 ....34.. ..56.... 78....9A	//x = (x ^ (x <<  4)) & 0x030C30C3;
	
	//....1212 ..3434.. 5656..78 78..9A9A	//x ^ (x <<  2)
	//....1..1 ..1..1.. 1..1..1. .1..1..1	//0x 09 24 92 49
	//....1..2 ..3..4.. 5..6..7. .8..9..A	//x = (x ^ (x <<  2)) & 0x09249249;
	
	//........ ........ ......11 11111111	//0x000003FF
	x &= 0x000003FF;		//Clear all bits above bit 10
	
	x = (x ^ (x << 16)) & 0xFF0000FF;
	x = (x ^ (x <<  8)) & 0x0300F00F;
	x = (x ^ (x <<  4)) & 0x030C30C3;
	x = (x ^ (x <<  2)) & 0x09249249;
	
	return x;
}
unsigned int getMortonCode(unsigned int x, unsigned int y, unsigned int z)
{
	return interleaveBits(x) << 0 | interleaveBits(y) << 1 | interleaveBits(z) << 2;
}

__kernel void separateAabbs(__global b3AabbCL* unseparatedAabbs, __global int* aabbIndices, __global b3AabbCL* out_aabbs, int numAabbsToSeparate)
{
	int separatedAabbIndex = get_global_id(0);
	if(separatedAabbIndex >= numAabbsToSeparate) return;

	int unseparatedAabbIndex = aabbIndices[separatedAabbIndex];
	out_aabbs[separatedAabbIndex] = unseparatedAabbs[unseparatedAabbIndex];
}

//Should replace with an optimized parallel reduction
__kernel void findAllNodesMergedAabb(__global b3AabbCL* out_mergedAabb, int numAabbsNeedingMerge)
{
	//Each time this kernel is added to the command queue, 
	//the number of AABBs needing to be merged is halved
	//
	//Example with 159 AABBs:
	//	numRemainingAabbs == 159 / 2 + 159 % 2 == 80
	//	numMergedAabbs == 159 - 80 == 79
	//So, indices [0, 78] are merged with [0 + 80, 78 + 80]
	
	int numRemainingAabbs = numAabbsNeedingMerge / 2 + numAabbsNeedingMerge % 2;
	int numMergedAabbs = numAabbsNeedingMerge - numRemainingAabbs;
	
	int aabbIndex = get_global_id(0);
	if(aabbIndex >= numMergedAabbs) return;
	
	int otherAabbIndex = aabbIndex + numRemainingAabbs;
	
	b3AabbCL aabb = out_mergedAabb[aabbIndex];
	b3AabbCL otherAabb = out_mergedAabb[otherAabbIndex];
		
	b3AabbCL mergedAabb;
	mergedAabb.m_min = b3Min(aabb.m_min, otherAabb.m_min);
	mergedAabb.m_max = b3Max(aabb.m_max, otherAabb.m_max);
	out_mergedAabb[aabbIndex] = mergedAabb;
}

__kernel void assignMortonCodesAndAabbIndicies(__global b3AabbCL* worldSpaceAabbs, __global b3AabbCL* mergedAabbOfAllNodes, 
												__global SortDataCL* out_mortonCodesAndAabbIndices, int numAabbs)
{
	int leafNodeIndex = get_global_id(0);	//Leaf node index == AABB index
	if(leafNodeIndex >= numAabbs) return;
	
	b3AabbCL mergedAabb = mergedAabbOfAllNodes[0];
	b3Vector3 gridCenter = (mergedAabb.m_min + mergedAabb.m_max) * 0.5f;
	b3Vector3 gridCellSize = (mergedAabb.m_max - mergedAabb.m_min) / (float)1024;
	
	b3AabbCL aabb = worldSpaceAabbs[leafNodeIndex];
	b3Vector3 aabbCenter = (aabb.m_min + aabb.m_max) * 0.5f;
	b3Vector3 aabbCenterRelativeToGrid = aabbCenter - gridCenter;
	
	//Quantize into integer coordinates
	//floor() is needed to prevent the center cell, at (0,0,0) from being twice the size
	b3Vector3 gridPosition = aabbCenterRelativeToGrid / gridCellSize;
	
	int4 discretePosition;
	discretePosition.x = (int)( (gridPosition.x >= 0.0f) ? gridPosition.x : floor(gridPosition.x) );
	discretePosition.y = (int)( (gridPosition.y >= 0.0f) ? gridPosition.y : floor(gridPosition.y) );
	discretePosition.z = (int)( (gridPosition.z >= 0.0f) ? gridPosition.z : floor(gridPosition.z) );
	
	//Clamp coordinates into [-512, 511], then convert range from [-512, 511] to [0, 1023]
	discretePosition = b3Max( -512, b3Min(discretePosition, 511) );
	discretePosition += 512;
	
	//Interleave bits(assign a morton code, also known as a z-curve)
	unsigned int mortonCode = getMortonCode(discretePosition.x, discretePosition.y, discretePosition.z);
	
	//
	SortDataCL mortonCodeIndexPair;
	mortonCodeIndexPair.m_key = mortonCode;
	mortonCodeIndexPair.m_value = leafNodeIndex;
	
	out_mortonCodesAndAabbIndices[leafNodeIndex] = mortonCodeIndexPair;
}

#define B3_PLVBH_TRAVERSE_MAX_STACK_SIZE 128

//The most significant bit(0x80000000) of a int32 is used to distinguish between leaf and internal nodes.
//If it is set, then the index is for an internal node; otherwise, it is a leaf node. 
//In both cases, the bit should be cleared to access the actual node index.
int isLeafNode(int index) { return (index >> 31 == 0); }
int getIndexWithInternalNodeMarkerRemoved(int index) { return index & (~0x80000000); }
int getIndexWithInternalNodeMarkerSet(int isLeaf, int index) { return (isLeaf) ? index : (index | 0x80000000); }

//From sap.cl
#define NEW_PAIR_MARKER -1

bool TestAabbAgainstAabb2(const b3AabbCL* aabb1, const b3AabbCL* aabb2)
{
	bool overlap = true;
	overlap = (aabb1->m_min.x > aabb2->m_max.x || aabb1->m_max.x < aabb2->m_min.x) ? false : overlap;
	overlap = (aabb1->m_min.z > aabb2->m_max.z || aabb1->m_max.z < aabb2->m_min.z) ? false : overlap;
	overlap = (aabb1->m_min.y > aabb2->m_max.y || aabb1->m_max.y < aabb2->m_min.y) ? false : overlap;
	return overlap;
}
//From sap.cl

__kernel void plbvhCalculateOverlappingPairs(__global b3AabbCL* rigidAabbs, 

											__global int* rootNodeIndex, 
											__global int2* internalNodeChildIndices, 
											__global b3AabbCL* internalNodeAabbs,
											__global int2* internalNodeLeafIndexRanges,
											
											__global SortDataCL* mortonCodesAndAabbIndices,
											__global int* out_numPairs, __global int4* out_overlappingPairs, 
											int maxPairs, int numQueryAabbs)
{
	//Using get_group_id()/get_local_id() is Faster than get_global_id(0) since
	//mortonCodesAndAabbIndices[] contains rigid body indices sorted along the z-curve (more spatially coherent)
	int queryBvhNodeIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);
	if(queryBvhNodeIndex >= numQueryAabbs) return;
	
	int queryRigidIndex = mortonCodesAndAabbIndices[queryBvhNodeIndex].m_value;
	b3AabbCL queryAabb = rigidAabbs[queryRigidIndex];
	
	int stack[B3_PLVBH_TRAVERSE_MAX_STACK_SIZE];
	
	int stackSize = 1;
	stack[0] = *rootNodeIndex;
	
	while(stackSize)
	{
		int internalOrLeafNodeIndex = stack[ stackSize - 1 ];
		--stackSize;
		
		int isLeaf = isLeafNode(internalOrLeafNodeIndex);	//Internal node if false
		int bvhNodeIndex = getIndexWithInternalNodeMarkerRemoved(internalOrLeafNodeIndex);
		
		//Optimization - if the BVH is structured as a binary radix tree, then
		//each internal node corresponds to a contiguous range of leaf nodes(internalNodeLeafIndexRanges[]).
		//This can be used to avoid testing each AABB-AABB pair twice, including preventing each node from colliding with itself.
		{
			int highestLeafIndex = (isLeaf) ? bvhNodeIndex : internalNodeLeafIndexRanges[bvhNodeIndex].y;
			if(highestLeafIndex <= queryBvhNodeIndex) continue;
		}
		
		//bvhRigidIndex is not used if internal node
		int bvhRigidIndex = (isLeaf) ? mortonCodesAndAabbIndices[bvhNodeIndex].m_value : -1;
	
		b3AabbCL bvhNodeAabb = (isLeaf) ? rigidAabbs[bvhRigidIndex] : internalNodeAabbs[bvhNodeIndex];
		if( TestAabbAgainstAabb2(&queryAabb, &bvhNodeAabb) )
		{
			if(isLeaf)
			{
				int4 pair;
				pair.x = rigidAabbs[queryRigidIndex].m_minIndices[3];
				pair.y = rigidAabbs[bvhRigidIndex].m_minIndices[3];
				pair.z = NEW_PAIR_MARKER;
				pair.w = NEW_PAIR_MARKER;
				
				int pairIndex = atomic_inc(out_numPairs);
				if(pairIndex < maxPairs) out_overlappingPairs[pairIndex] = pair;
			}
			
			if(!isLeaf)	//Internal node
			{
				if(stackSize + 2 > B3_PLVBH_TRAVERSE_MAX_STACK_SIZE)
				{
					//Error
				}
				else
				{
					stack[ stackSize++ ] = internalNodeChildIndices[bvhNodeIndex].x;
					stack[ stackSize++ ] = internalNodeChildIndices[bvhNodeIndex].y;
				}
			}
		}
		
	}
}


//From rayCastKernels.cl
typedef struct
{
	float4 m_from;
	float4 m_to;
} b3RayInfo;
//From rayCastKernels.cl

b3Vector3 b3Vector3_normalize(b3Vector3 v)
{
	b3Vector3 normal = (b3Vector3){v.x, v.y, v.z, 0.f};
	return normalize(normal);	//OpenCL normalize == vector4 normalize
}
b3Scalar b3Vector3_length2(b3Vector3 v) { return v.x*v.x + v.y*v.y + v.z*v.z; }
b3Scalar b3Vector3_dot(b3Vector3 a, b3Vector3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

int rayIntersectsAabb(b3Vector3 rayOrigin, b3Scalar rayLength, b3Vector3 rayNormalizedDirection, b3AabbCL aabb)
{
	//AABB is considered as 3 pairs of 2 planes( {x_min, x_max}, {y_min, y_max}, {z_min, z_max} ).
	//t_min is the point of intersection with the closer plane, t_max is the point of intersection with the farther plane.
	//
	//if (rayNormalizedDirection.x < 0.0f), then max.x will be the near plane 
	//and min.x will be the far plane; otherwise, it is reversed.
	//
	//In order for there to be a collision, the t_min and t_max of each pair must overlap.
	//This can be tested for by selecting the highest t_min and lowest t_max and comparing them.
	
	int4 isNegative = isless( rayNormalizedDirection, ((b3Vector3){0.0f, 0.0f, 0.0f, 0.0f}) );	//isless(x,y) returns (x < y)
	
	//When using vector types, the select() function checks the most signficant bit, 
	//but isless() sets the least significant bit.
	isNegative <<= 31;

	//select(b, a, condition) == condition ? a : b
	//When using select() with vector types, (condition[i]) is true if its most significant bit is 1
	b3Vector3 t_min = ( select(aabb.m_min, aabb.m_max, isNegative) - rayOrigin ) / rayNormalizedDirection;
	b3Vector3 t_max = ( select(aabb.m_max, aabb.m_min, isNegative) - rayOrigin ) / rayNormalizedDirection;
	
	b3Scalar t_min_final = 0.0f;
	b3Scalar t_max_final = rayLength;
	
	//Must use fmin()/fmax(); if one of the parameters is NaN, then the parameter that is not NaN is returned. 
	//Behavior of min()/max() with NaNs is undefined. (See OpenCL Specification 1.2 [6.12.2] and [6.12.4])
	//Since the innermost fmin()/fmax() is always not NaN, this should never return NaN.
	t_min_final = fmax( t_min.z, fmax(t_min.y, fmax(t_min.x, t_min_final)) );
	t_max_final = fmin( t_max.z, fmin(t_max.y, fmin(t_max.x, t_max_final)) );
	
	return (t_min_final <= t_max_final);
}

__kernel void plbvhRayTraverse(__global b3AabbCL* rigidAabbs,

								__global int* rootNodeIndex, 
								__global int2* internalNodeChildIndices, 
								__global b3AabbCL* internalNodeAabbs,
								__global int2* internalNodeLeafIndexRanges,
								__global SortDataCL* mortonCodesAndAabbIndices,
								
								__global b3RayInfo* rays,
								
								__global int* out_numRayRigidPairs, 
								__global int2* out_rayRigidPairs,
								int maxRayRigidPairs, int numRays)
{
	int rayIndex = get_global_id(0);
	if(rayIndex >= numRays) return;
	
	//
	b3Vector3 rayFrom = rays[rayIndex].m_from;
	b3Vector3 rayTo = rays[rayIndex].m_to;
	b3Vector3 rayNormalizedDirection = b3Vector3_normalize(rayTo - rayFrom);
	b3Scalar rayLength = b3Sqrt( b3Vector3_length2(rayTo - rayFrom) );
	
	//
	int stack[B3_PLVBH_TRAVERSE_MAX_STACK_SIZE];
	
	int stackSize = 1;
	stack[0] = *rootNodeIndex;
	
	while(stackSize)
	{
		int internalOrLeafNodeIndex = stack[ stackSize - 1 ];
		--stackSize;
		
		int isLeaf = isLeafNode(internalOrLeafNodeIndex);	//Internal node if false
		int bvhNodeIndex = getIndexWithInternalNodeMarkerRemoved(internalOrLeafNodeIndex);
		
		//bvhRigidIndex is not used if internal node
		int bvhRigidIndex = (isLeaf) ? mortonCodesAndAabbIndices[bvhNodeIndex].m_value : -1;
	
		b3AabbCL bvhNodeAabb = (isLeaf) ? rigidAabbs[bvhRigidIndex] : internalNodeAabbs[bvhNodeIndex];
		if( rayIntersectsAabb(rayFrom, rayLength, rayNormalizedDirection, bvhNodeAabb)  )
		{
			if(isLeaf)
			{
				int2 rayRigidPair;
				rayRigidPair.x = rayIndex;
				rayRigidPair.y = rigidAabbs[bvhRigidIndex].m_minIndices[3];
				
				int pairIndex = atomic_inc(out_numRayRigidPairs);
				if(pairIndex < maxRayRigidPairs) out_rayRigidPairs[pairIndex] = rayRigidPair;
			}
			
			if(!isLeaf)	//Internal node
			{
				if(stackSize + 2 > B3_PLVBH_TRAVERSE_MAX_STACK_SIZE)
				{
					//Error
				}
				else
				{
					stack[ stackSize++ ] = internalNodeChildIndices[bvhNodeIndex].x;
					stack[ stackSize++ ] = internalNodeChildIndices[bvhNodeIndex].y;
				}
			}
		}
	}
}

__kernel void plbvhLargeAabbAabbTest(__global b3AabbCL* smallAabbs, __global b3AabbCL* largeAabbs, 
									__global int* out_numPairs, __global int4* out_overlappingPairs, 
									int maxPairs, int numLargeAabbRigids, int numSmallAabbRigids)
{
	int smallAabbIndex = get_global_id(0);
	if(smallAabbIndex >= numSmallAabbRigids) return;
	
	b3AabbCL smallAabb = smallAabbs[smallAabbIndex];
	for(int i = 0; i < numLargeAabbRigids; ++i)
	{
		b3AabbCL largeAabb = largeAabbs[i];
		if( TestAabbAgainstAabb2(&smallAabb, &largeAabb) )
		{
			int4 pair;
			pair.x = largeAabb.m_minIndices[3];
			pair.y = smallAabb.m_minIndices[3];
			pair.z = NEW_PAIR_MARKER;
			pair.w = NEW_PAIR_MARKER;
			
			int pairIndex = atomic_inc(out_numPairs);
			if(pairIndex < maxPairs) out_overlappingPairs[pairIndex] = pair;
		}
	}
}
__kernel void plbvhLargeAabbRayTest(__global b3AabbCL* largeRigidAabbs, __global b3RayInfo* rays,
									__global int* out_numRayRigidPairs,  __global int2* out_rayRigidPairs,
									int numLargeAabbRigids, int maxRayRigidPairs, int numRays)
{
	int rayIndex = get_global_id(0);
	if(rayIndex >= numRays) return;
	
	b3Vector3 rayFrom = rays[rayIndex].m_from;
	b3Vector3 rayTo = rays[rayIndex].m_to;
	b3Vector3 rayNormalizedDirection = b3Vector3_normalize(rayTo - rayFrom);
	b3Scalar rayLength = b3Sqrt( b3Vector3_length2(rayTo - rayFrom) );
	
	for(int i = 0; i < numLargeAabbRigids; ++i)
	{
		b3AabbCL rigidAabb = largeRigidAabbs[i];
		if( rayIntersectsAabb(rayFrom, rayLength, rayNormalizedDirection, rigidAabb) )
		{
			int2 rayRigidPair;
			rayRigidPair.x = rayIndex;
			rayRigidPair.y = rigidAabb.m_minIndices[3];
			
			int pairIndex = atomic_inc(out_numRayRigidPairs);
			if(pairIndex < maxRayRigidPairs) out_rayRigidPairs[pairIndex] = rayRigidPair;
		}
	}
}


//Set so that it is always greater than the actual common prefixes, and never selected as a parent node.
//If there are no duplicates, then the highest common prefix is 32 or 64, depending on the number of bits used for the z-curve.
//Duplicate common prefixes increase the highest common prefix at most by the number of bits used to index the leaf node.
//Since 32 bit ints are used to index leaf nodes, the max prefix is 64(32 + 32 bit z-curve) or 96(32 + 64 bit z-curve).
#define B3_PLBVH_INVALID_COMMON_PREFIX 128

#define B3_PLBVH_ROOT_NODE_MARKER -1

#define b3Int64 long

int computeCommonPrefixLength(b3Int64 i, b3Int64 j) { return (int)clz(i ^ j); }
b3Int64 computeCommonPrefix(b3Int64 i, b3Int64 j) 
{
	//This function only needs to return (i & j) in order for the algorithm to work,
	//but it may help with debugging to mask out the lower bits.

	b3Int64 commonPrefixLength = (b3Int64)computeCommonPrefixLength(i, j);

	b3Int64 sharedBits = i & j;
	b3Int64 bitmask = ((b3Int64)(~0)) << (64 - commonPrefixLength);	//Set all bits after the common prefix to 0
	
	return sharedBits & bitmask;
}

//Same as computeCommonPrefixLength(), but allows for prefixes with different lengths
int getSharedPrefixLength(b3Int64 prefixA, int prefixLengthA, b3Int64 prefixB, int prefixLengthB)
{
	return b3Min( computeCommonPrefixLength(prefixA, prefixB), b3Min(prefixLengthA, prefixLengthB) );
}

__kernel void computeAdjacentPairCommonPrefix(__global SortDataCL* mortonCodesAndAabbIndices,
											__global b3Int64* out_commonPrefixes,
											__global int* out_commonPrefixLengths,
											int numInternalNodes)
{
	int internalNodeIndex = get_global_id(0);
	if (internalNodeIndex >= numInternalNodes) return;
	
	//Here, (internalNodeIndex + 1) is never out of bounds since it is a leaf node index,
	//and the number of internal nodes is always numLeafNodes - 1
	int leftLeafIndex = internalNodeIndex;
	int rightLeafIndex = internalNodeIndex + 1;
	
	int leftLeafMortonCode = mortonCodesAndAabbIndices[leftLeafIndex].m_key;
	int rightLeafMortonCode = mortonCodesAndAabbIndices[rightLeafIndex].m_key;
	
	//Binary radix tree construction algorithm does not work if there are duplicate morton codes.
	//Append the index of each leaf node to each morton code so that there are no duplicates.
	//The algorithm also requires that the morton codes are sorted in ascending order; this requirement
	//is also satisfied with this method, as (leftLeafIndex < rightLeafIndex) is always true.
	//
	//upsample(a, b) == ( ((b3Int64)a) << 32) | b
	b3Int64 nonduplicateLeftMortonCode = upsample(leftLeafMortonCode, leftLeafIndex);
	b3Int64 nonduplicateRightMortonCode = upsample(rightLeafMortonCode, rightLeafIndex);
	
	out_commonPrefixes[internalNodeIndex] = computeCommonPrefix(nonduplicateLeftMortonCode, nonduplicateRightMortonCode);
	out_commonPrefixLengths[internalNodeIndex] = computeCommonPrefixLength(nonduplicateLeftMortonCode, nonduplicateRightMortonCode);
}


__kernel void buildBinaryRadixTreeLeafNodes(__global int* commonPrefixLengths, __global int* out_leafNodeParentNodes,
											__global int2* out_childNodes, int numLeafNodes)
{
	int leafNodeIndex = get_global_id(0);
	if (leafNodeIndex >= numLeafNodes) return;
	
	int numInternalNodes = numLeafNodes - 1;
	
	int leftSplitIndex = leafNodeIndex - 1;
	int rightSplitIndex = leafNodeIndex;
	
	int leftCommonPrefix = (leftSplitIndex >= 0) ? commonPrefixLengths[leftSplitIndex] : B3_PLBVH_INVALID_COMMON_PREFIX;
	int rightCommonPrefix = (rightSplitIndex < numInternalNodes) ? commonPrefixLengths[rightSplitIndex] : B3_PLBVH_INVALID_COMMON_PREFIX;
	
	//Parent node is the highest adjacent common prefix that is lower than the node's common prefix
	//Leaf nodes are considered as having the highest common prefix
	int isLeftHigherCommonPrefix = (leftCommonPrefix > rightCommonPrefix);
	
	//Handle cases for the edge nodes; the first and last node
	//For leaf nodes, leftCommonPrefix and rightCommonPrefix should never both be B3_PLBVH_INVALID_COMMON_PREFIX
	if(leftCommonPrefix == B3_PLBVH_INVALID_COMMON_PREFIX) isLeftHigherCommonPrefix = false;
	if(rightCommonPrefix == B3_PLBVH_INVALID_COMMON_PREFIX) isLeftHigherCommonPrefix = true;
	
	int parentNodeIndex = (isLeftHigherCommonPrefix) ? leftSplitIndex : rightSplitIndex;
	out_leafNodeParentNodes[leafNodeIndex] = parentNodeIndex;
	
	int isRightChild = (isLeftHigherCommonPrefix);	//If the left node is the parent, then this node is its right child and vice versa
	
	//out_childNodesAsInt[0] == int2.x == left child
	//out_childNodesAsInt[1] == int2.y == right child
	int isLeaf = 1;
	__global int* out_childNodesAsInt = (__global int*)(&out_childNodes[parentNodeIndex]);
	out_childNodesAsInt[isRightChild] = getIndexWithInternalNodeMarkerSet(isLeaf, leafNodeIndex);
}

__kernel void buildBinaryRadixTreeInternalNodes(__global b3Int64* commonPrefixes, __global int* commonPrefixLengths,
												__global int2* out_childNodes,
												__global int* out_internalNodeParentNodes, __global int* out_rootNodeIndex,
												int numInternalNodes)
{
	int internalNodeIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);
	if(internalNodeIndex >= numInternalNodes) return;
	
	b3Int64 nodePrefix = commonPrefixes[internalNodeIndex];
	int nodePrefixLength = commonPrefixLengths[internalNodeIndex];
	
//#define USE_LINEAR_SEARCH
#ifdef USE_LINEAR_SEARCH
	int leftIndex = -1;
	int rightIndex = -1;
	
	//Find nearest element to left with a lower common prefix
	for(int i = internalNodeIndex - 1; i >= 0; --i)
	{
		int nodeLeftSharedPrefixLength = getSharedPrefixLength(nodePrefix, nodePrefixLength, commonPrefixes[i], commonPrefixLengths[i]);
		if(nodeLeftSharedPrefixLength < nodePrefixLength)
		{
			leftIndex = i;
			break;
		}
	}
	
	//Find nearest element to right with a lower common prefix
	for(int i = internalNodeIndex + 1; i < numInternalNodes; ++i)
	{
		int nodeRightSharedPrefixLength = getSharedPrefixLength(nodePrefix, nodePrefixLength, commonPrefixes[i], commonPrefixLengths[i]);
		if(nodeRightSharedPrefixLength < nodePrefixLength)
		{
			rightIndex = i;
			break;
		}
	}
	
#else //Use binary search

	//Find nearest element to left with a lower common prefix
	int leftIndex = -1;
	{
		int lower = 0;
		int upper = internalNodeIndex - 1;
		
		while(lower <= upper)
		{
			int mid = (lower + upper) / 2;
			b3Int64 midPrefix = commonPrefixes[mid];
			int midPrefixLength = commonPrefixLengths[mid];
			
			int nodeMidSharedPrefixLength = getSharedPrefixLength(nodePrefix, nodePrefixLength, midPrefix, midPrefixLength);
			if(nodeMidSharedPrefixLength < nodePrefixLength) 
			{
				int right = mid + 1;
				if(right < internalNodeIndex)
				{
					b3Int64 rightPrefix = commonPrefixes[right];
					int rightPrefixLength = commonPrefixLengths[right];
					
					int nodeRightSharedPrefixLength = getSharedPrefixLength(nodePrefix, nodePrefixLength, rightPrefix, rightPrefixLength);
					if(nodeRightSharedPrefixLength < nodePrefixLength) 
					{
						lower = right;
						leftIndex = right;
					}
					else 
					{
						leftIndex = mid;
						break;
					}
				}
				else 
				{
					leftIndex = mid;
					break;
				}
			}
			else upper = mid - 1;
		}
	}
	
	//Find nearest element to right with a lower common prefix
	int rightIndex = -1;
	{
		int lower = internalNodeIndex + 1;
		int upper = numInternalNodes - 1;
		
		while(lower <= upper)
		{
			int mid = (lower + upper) / 2;
			b3Int64 midPrefix = commonPrefixes[mid];
			int midPrefixLength = commonPrefixLengths[mid];
			
			int nodeMidSharedPrefixLength = getSharedPrefixLength(nodePrefix, nodePrefixLength, midPrefix, midPrefixLength);
			if(nodeMidSharedPrefixLength < nodePrefixLength) 
			{
				int left = mid - 1;
				if(left > internalNodeIndex)
				{
					b3Int64 leftPrefix = commonPrefixes[left];
					int leftPrefixLength = commonPrefixLengths[left];
				
					int nodeLeftSharedPrefixLength = getSharedPrefixLength(nodePrefix, nodePrefixLength, leftPrefix, leftPrefixLength);
					if(nodeLeftSharedPrefixLength < nodePrefixLength) 
					{
						upper = left;
						rightIndex = left;
					}
					else 
					{
						rightIndex = mid;
						break;
					}
				}
				else 
				{
					rightIndex = mid;
					break;
				}
			}
			else lower = mid + 1;
		}
	}
#endif
	
	//Select parent
	{
		int leftPrefixLength = (leftIndex != -1) ? commonPrefixLengths[leftIndex] : B3_PLBVH_INVALID_COMMON_PREFIX;
		int rightPrefixLength =  (rightIndex != -1) ? commonPrefixLengths[rightIndex] : B3_PLBVH_INVALID_COMMON_PREFIX;
		
		int isLeftHigherPrefixLength = (leftPrefixLength > rightPrefixLength);
		
		if(leftPrefixLength == B3_PLBVH_INVALID_COMMON_PREFIX) isLeftHigherPrefixLength = false;
		else if(rightPrefixLength == B3_PLBVH_INVALID_COMMON_PREFIX) isLeftHigherPrefixLength = true;
		
		int parentNodeIndex = (isLeftHigherPrefixLength) ? leftIndex : rightIndex;
		
		int isRootNode = (leftIndex == -1 && rightIndex == -1);
		out_internalNodeParentNodes[internalNodeIndex] = (!isRootNode) ? parentNodeIndex : B3_PLBVH_ROOT_NODE_MARKER;
		
		int isLeaf = 0;
		if(!isRootNode)
		{
			int isRightChild = (isLeftHigherPrefixLength);	//If the left node is the parent, then this node is its right child and vice versa
			
			//out_childNodesAsInt[0] == int2.x == left child
			//out_childNodesAsInt[1] == int2.y == right child
			__global int* out_childNodesAsInt = (__global int*)(&out_childNodes[parentNodeIndex]);
			out_childNodesAsInt[isRightChild] = getIndexWithInternalNodeMarkerSet(isLeaf, internalNodeIndex);
		}
		else *out_rootNodeIndex = getIndexWithInternalNodeMarkerSet(isLeaf, internalNodeIndex);
	}
}

__kernel void findDistanceFromRoot(__global int* rootNodeIndex, __global int* internalNodeParentNodes,
									__global int* out_maxDistanceFromRoot, __global int* out_distanceFromRoot, int numInternalNodes)
{
	if( get_global_id(0) == 0 ) atomic_xchg(out_maxDistanceFromRoot, 0);

	int internalNodeIndex = get_global_id(0);
	if(internalNodeIndex >= numInternalNodes) return;
	
	//
	int distanceFromRoot = 0;
	{
		int parentIndex = internalNodeParentNodes[internalNodeIndex];
		while(parentIndex != B3_PLBVH_ROOT_NODE_MARKER)
		{
			parentIndex = internalNodeParentNodes[parentIndex];
			++distanceFromRoot;
		}
	}
	out_distanceFromRoot[internalNodeIndex] = distanceFromRoot;
	
	//
	__local int localMaxDistanceFromRoot;
	if( get_local_id(0) == 0 ) localMaxDistanceFromRoot = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	atomic_max(&localMaxDistanceFromRoot, distanceFromRoot);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if( get_local_id(0) == 0 ) atomic_max(out_maxDistanceFromRoot, localMaxDistanceFromRoot);
}

__kernel void buildBinaryRadixTreeAabbsRecursive(__global int* distanceFromRoot, __global SortDataCL* mortonCodesAndAabbIndices,
												__global int2* childNodes,
												__global b3AabbCL* leafNodeAabbs, __global b3AabbCL* internalNodeAabbs,
												int maxDistanceFromRoot, int processedDistance, int numInternalNodes)
{
	int internalNodeIndex = get_global_id(0);
	if(internalNodeIndex >= numInternalNodes) return;
	
	int distance = distanceFromRoot[internalNodeIndex];
	
	if(distance == processedDistance)
	{
		int leftChildIndex = childNodes[internalNodeIndex].x;
		int rightChildIndex = childNodes[internalNodeIndex].y;
		
		int isLeftChildLeaf = isLeafNode(leftChildIndex);
		int isRightChildLeaf = isLeafNode(rightChildIndex);
		
		leftChildIndex = getIndexWithInternalNodeMarkerRemoved(leftChildIndex);
		rightChildIndex = getIndexWithInternalNodeMarkerRemoved(rightChildIndex);
		
		//leftRigidIndex/rightRigidIndex is not used if internal node
		int leftRigidIndex = (isLeftChildLeaf) ? mortonCodesAndAabbIndices[leftChildIndex].m_value : -1;
		int rightRigidIndex = (isRightChildLeaf) ? mortonCodesAndAabbIndices[rightChildIndex].m_value : -1;
		
		b3AabbCL leftChildAabb = (isLeftChildLeaf) ? leafNodeAabbs[leftRigidIndex] : internalNodeAabbs[leftChildIndex];
		b3AabbCL rightChildAabb = (isRightChildLeaf) ? leafNodeAabbs[rightRigidIndex] : internalNodeAabbs[rightChildIndex];
		
		b3AabbCL mergedAabb;
		mergedAabb.m_min = b3Min(leftChildAabb.m_min, rightChildAabb.m_min);
		mergedAabb.m_max = b3Max(leftChildAabb.m_max, rightChildAabb.m_max);
		internalNodeAabbs[internalNodeIndex] = mergedAabb;
	}
}

__kernel void findLeafIndexRanges(__global int2* internalNodeChildNodes, __global int2* out_leafIndexRanges, int numInternalNodes)
{
	int internalNodeIndex = get_global_id(0);
	if(internalNodeIndex >= numInternalNodes) return;
	
	int numLeafNodes = numInternalNodes + 1;
	
	int2 childNodes = internalNodeChildNodes[internalNodeIndex];
	
	int2 leafIndexRange;	//x == min leaf index, y == max leaf index
	
	//Find lowest leaf index covered by this internal node
	{
		int lowestIndex = childNodes.x;		//childNodes.x == Left child
		while( !isLeafNode(lowestIndex) ) lowestIndex = internalNodeChildNodes[ getIndexWithInternalNodeMarkerRemoved(lowestIndex) ].x;
		leafIndexRange.x = lowestIndex;
	}
	
	//Find highest leaf index covered by this internal node
	{
		int highestIndex = childNodes.y;	//childNodes.y == Right child
		while( !isLeafNode(highestIndex) ) highestIndex = internalNodeChildNodes[ getIndexWithInternalNodeMarkerRemoved(highestIndex) ].y;
		leafIndexRange.y = highestIndex;
	}
	
	//
	out_leafIndexRanges[internalNodeIndex] = leafIndexRange;
}
