

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3UpdateAabbs.h"


__kernel void initializeGpuAabbsFull(  const int numNodes, __global b3RigidBodyData_t* gBodies,__global b3Collidable_t* collidables, __global b3Aabb_t* plocalShapeAABB, __global b3Aabb_t* pAABB)
{
	int nodeID = get_global_id(0);
	if( nodeID < numNodes )
	{
		b3ComputeWorldAabb(nodeID, gBodies, collidables, plocalShapeAABB,pAABB);
	}
}

__kernel void clearOverlappingPairsKernel(  __global int4* pairs, int numPairs)
{
	int pairId = get_global_id(0);
	if( pairId< numPairs )
	{
		pairs[pairId].z = 0xffffffff;
	}
}