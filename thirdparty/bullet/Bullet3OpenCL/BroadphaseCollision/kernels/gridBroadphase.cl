

int getPosHash(int4 gridPos, __global float4* pParams)
{
	int4 gridDim = *((__global int4*)(pParams + 1));
	gridPos.x &= gridDim.x - 1;
	gridPos.y &= gridDim.y - 1;
	gridPos.z &= gridDim.z - 1;
	int hash = gridPos.z * gridDim.y * gridDim.x + gridPos.y * gridDim.x + gridPos.x;
	return hash;
} 

int4 getGridPos(float4 worldPos, __global float4* pParams)
{
    int4 gridPos;
	int4 gridDim = *((__global int4*)(pParams + 1));
    gridPos.x = (int)floor(worldPos.x * pParams[0].x) & (gridDim.x - 1);
    gridPos.y = (int)floor(worldPos.y * pParams[0].y) & (gridDim.y - 1);
    gridPos.z = (int)floor(worldPos.z * pParams[0].z) & (gridDim.z - 1);
    return gridPos;
}


// calculate grid hash value for each body using its AABB
__kernel void kCalcHashAABB(int numObjects, __global float4* allpAABB, __global const int* smallAabbMapping, __global int2* pHash, __global float4* pParams )
{
    int index = get_global_id(0);
    if(index >= numObjects)
	{
		return;
	}
	float4 bbMin = allpAABB[smallAabbMapping[index]*2];
	float4 bbMax = allpAABB[smallAabbMapping[index]*2 + 1];
	float4 pos;
	pos.x = (bbMin.x + bbMax.x) * 0.5f;
	pos.y = (bbMin.y + bbMax.y) * 0.5f;
	pos.z = (bbMin.z + bbMax.z) * 0.5f;
	pos.w = 0.f;
    // get address in grid
    int4 gridPos = getGridPos(pos, pParams);
    int gridHash = getPosHash(gridPos, pParams);
    // store grid hash and body index
    int2 hashVal;
    hashVal.x = gridHash;
    hashVal.y = index;
    pHash[index] = hashVal;
}

__kernel void kClearCellStart(	int numCells, 
								__global int* pCellStart )
{
    int index = get_global_id(0);
    if(index >= numCells)
	{
		return;
	}
	pCellStart[index] = -1;
}

__kernel void kFindCellStart(int numObjects, __global int2* pHash, __global int* cellStart )
{
	__local int sharedHash[513];
    int index = get_global_id(0);
	int2 sortedData;

    if(index < numObjects)
	{
		sortedData = pHash[index];
		// Load hash data into shared memory so that we can look 
		// at neighboring body's hash value without loading
		// two hash values per thread
		sharedHash[get_local_id(0) + 1] = sortedData.x;
		if((index > 0) && (get_local_id(0) == 0))
		{
			// first thread in block must load neighbor body hash
			sharedHash[0] = pHash[index-1].x;
		}
	}
    barrier(CLK_LOCAL_MEM_FENCE);
    if(index < numObjects)
	{
		if((index == 0) || (sortedData.x != sharedHash[get_local_id(0)]))
		{
			cellStart[sortedData.x] = index;
		}
	}
}

int testAABBOverlap(float4 min0, float4 max0, float4 min1, float4 max1)
{
	return	(min0.x <= max1.x)&& (min1.x <= max0.x) && 
			(min0.y <= max1.y)&& (min1.y <= max0.y) && 
			(min0.z <= max1.z)&& (min1.z <= max0.z); 
}




//search for AABB 'index' against other AABBs' in this cell
void findPairsInCell(	int numObjects,
						int4	gridPos,
						int    index,
						__global int2*  pHash,
						__global int*   pCellStart,
						__global float4* allpAABB, 
						__global const int* smallAabbMapping,
						__global float4* pParams,
							volatile  __global int* pairCount,
						__global int4*   pPairBuff2,
						int maxPairs
						)
{
	int4 pGridDim = *((__global int4*)(pParams + 1));
	int maxBodiesPerCell = pGridDim.w;
    int gridHash = getPosHash(gridPos, pParams);
    // get start of bucket for this cell
    int bucketStart = pCellStart[gridHash];
    if (bucketStart == -1)
	{
        return;   // cell empty
	}
	// iterate over bodies in this cell
    int2 sortedData = pHash[index];
	int unsorted_indx = sortedData.y;
    float4 min0 = allpAABB[smallAabbMapping[unsorted_indx]*2 + 0]; 
	float4 max0 = allpAABB[smallAabbMapping[unsorted_indx]*2 + 1];
	int handleIndex =  as_int(min0.w);
	
	int bucketEnd = bucketStart + maxBodiesPerCell;
	bucketEnd = (bucketEnd > numObjects) ? numObjects : bucketEnd;
	for(int index2 = bucketStart; index2 < bucketEnd; index2++) 
	{
        int2 cellData = pHash[index2];
        if (cellData.x != gridHash)
        {
			break;   // no longer in same bucket
		}
		int unsorted_indx2 = cellData.y;
        //if (unsorted_indx2 < unsorted_indx) // check not colliding with self
		if (unsorted_indx2 != unsorted_indx) // check not colliding with self
        {   
			float4 min1 = allpAABB[smallAabbMapping[unsorted_indx2]*2 + 0];
			float4 max1 = allpAABB[smallAabbMapping[unsorted_indx2]*2 + 1];
			if(testAABBOverlap(min0, max0, min1, max1))
			{
				if (pairCount)
				{
					int handleIndex2 = as_int(min1.w);
					if (handleIndex<handleIndex2)
					{
						int curPair = atomic_add(pairCount,1);
						if (curPair<maxPairs)
						{
							int4 newpair;
							newpair.x = handleIndex;
							newpair.y = handleIndex2;
							newpair.z = -1;
							newpair.w = -1;
							pPairBuff2[curPair] = newpair;
						}
					}
				
				}
			}
		}
	}
}

__kernel void kFindOverlappingPairs(	int numObjects,
										__global float4* allpAABB, 
										__global const int* smallAabbMapping,
										__global int2* pHash, 
										__global int* pCellStart, 
										__global float4* pParams ,
										volatile  __global int* pairCount,
										__global int4*   pPairBuff2,
										int maxPairs
										)

{
    int index = get_global_id(0);
    if(index >= numObjects)
	{
		return;
	}
    int2 sortedData = pHash[index];
	int unsorted_indx = sortedData.y;
	float4 bbMin = allpAABB[smallAabbMapping[unsorted_indx]*2 + 0];
	float4 bbMax = allpAABB[smallAabbMapping[unsorted_indx]*2 + 1];
	float4 pos;
	pos.x = (bbMin.x + bbMax.x) * 0.5f;
	pos.y = (bbMin.y + bbMax.y) * 0.5f;
	pos.z = (bbMin.z + bbMax.z) * 0.5f;
    // get address in grid
    int4 gridPosA = getGridPos(pos, pParams);
    int4 gridPosB; 
    // examine only neighbouring cells
    for(int z=-1; z<=1; z++) 
    {
		gridPosB.z = gridPosA.z + z;
        for(int y=-1; y<=1; y++) 
        {
			gridPosB.y = gridPosA.y + y;
            for(int x=-1; x<=1; x++) 
            {
				gridPosB.x = gridPosA.x + x;
                findPairsInCell(numObjects, gridPosB, index, pHash, pCellStart, allpAABB,smallAabbMapping, pParams, pairCount,pPairBuff2, maxPairs);
            }
        }
    }
}





