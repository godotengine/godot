//keep this enum in sync with the CPU version (in btCollidable.h)
//written by Erwin Coumans

#define SHAPE_CONVEX_HULL 3
#define SHAPE_CONCAVE_TRIMESH 5
#define TRIANGLE_NUM_CONVEX_FACES 5
#define SHAPE_COMPOUND_OF_CONVEX_HULLS 6
#define SHAPE_SPHERE 7

typedef unsigned int u32;

#define MAX_NUM_PARTS_IN_BITS 10

///btQuantizedBvhNode is a compressed aabb node, 16 bytes.
///Node can be used for leafnode or internal node. Leafnodes can point to 32-bit triangle index (non-negative range).
typedef struct
{
	//12 bytes
	unsigned short int	m_quantizedAabbMin[3];
	unsigned short int	m_quantizedAabbMax[3];
	//4 bytes
	int	m_escapeIndexOrTriangleIndex;
} btQuantizedBvhNode;

typedef struct
{
	float4		m_aabbMin;
	float4		m_aabbMax;
	float4		m_quantization;
	int			m_numNodes;
	int			m_numSubTrees;
	int			m_nodeOffset;
	int			m_subTreeOffset;

} b3BvhInfo;

int	getTriangleIndex(const btQuantizedBvhNode* rootNode)
{
	unsigned int x=0;
	unsigned int y = (~(x&0))<<(31-MAX_NUM_PARTS_IN_BITS);
	// Get only the lower bits where the triangle index is stored
	return (rootNode->m_escapeIndexOrTriangleIndex&~(y));
}

int isLeaf(const btQuantizedBvhNode* rootNode)
{
	//skipindex is negative (internal node), triangleindex >=0 (leafnode)
	return (rootNode->m_escapeIndexOrTriangleIndex >= 0)? 1 : 0;
}
	
int getEscapeIndex(const btQuantizedBvhNode* rootNode)
{
	return -rootNode->m_escapeIndexOrTriangleIndex;
}

typedef struct
{
	//12 bytes
	unsigned short int	m_quantizedAabbMin[3];
	unsigned short int	m_quantizedAabbMax[3];
	//4 bytes, points to the root of the subtree
	int			m_rootNodeIndex;
	//4 bytes
	int			m_subtreeSize;
	int			m_padding[3];
} btBvhSubtreeInfo;

///keep this in sync with btCollidable.h
typedef struct
{
	int m_numChildShapes;
	int blaat2;
	int m_shapeType;
	int m_shapeIndex;
	
} btCollidableGpu;

typedef struct
{
	float4	m_childPosition;
	float4	m_childOrientation;
	int m_shapeIndex;
	int m_unused0;
	int m_unused1;
	int m_unused2;
} btGpuChildShape;


typedef struct
{
	float4 m_pos;
	float4 m_quat;
	float4 m_linVel;
	float4 m_angVel;

	u32 m_collidableIdx;
	float m_invMass;
	float m_restituitionCoeff;
	float m_frictionCoeff;
} BodyData;

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
} btAabbCL;


int testQuantizedAabbAgainstQuantizedAabb(
								const unsigned short int* aabbMin1,
								const unsigned short int* aabbMax1,
								const unsigned short int* aabbMin2,
								const unsigned short int* aabbMax2)
{
	//int overlap = 1;
	if (aabbMin1[0] > aabbMax2[0])
		return 0;
	if (aabbMax1[0] < aabbMin2[0])
		return 0;
	if (aabbMin1[1] > aabbMax2[1])
		return 0;
	if (aabbMax1[1] < aabbMin2[1])
		return 0;
	if (aabbMin1[2] > aabbMax2[2])
		return 0;
	if (aabbMax1[2] < aabbMin2[2])
		return 0;
	return 1;
	//overlap = ((aabbMin1[0] > aabbMax2[0]) || (aabbMax1[0] < aabbMin2[0])) ? 0 : overlap;
	//overlap = ((aabbMin1[2] > aabbMax2[2]) || (aabbMax1[2] < aabbMin2[2])) ? 0 : overlap;
	//overlap = ((aabbMin1[1] > aabbMax2[1]) || (aabbMax1[1] < aabbMin2[1])) ? 0 : overlap;
	//return overlap;
}


void quantizeWithClamp(unsigned short* out, float4 point2,int isMax, float4 bvhAabbMin, float4 bvhAabbMax, float4 bvhQuantization)
{
	float4 clampedPoint = max(point2,bvhAabbMin);
	clampedPoint = min (clampedPoint, bvhAabbMax);

	float4 v = (clampedPoint - bvhAabbMin) * bvhQuantization;
	if (isMax)
	{
		out[0] = (unsigned short) (((unsigned short)(v.x+1.f) | 1));
		out[1] = (unsigned short) (((unsigned short)(v.y+1.f) | 1));
		out[2] = (unsigned short) (((unsigned short)(v.z+1.f) | 1));
	} else
	{
		out[0] = (unsigned short) (((unsigned short)(v.x) & 0xfffe));
		out[1] = (unsigned short) (((unsigned short)(v.y) & 0xfffe));
		out[2] = (unsigned short) (((unsigned short)(v.z) & 0xfffe));
	}

}


// work-in-progress
__kernel void   bvhTraversalKernel( __global const int4* pairs, 
									__global const BodyData* rigidBodies, 
									__global const btCollidableGpu* collidables,
									__global btAabbCL* aabbs,
									__global int4* concavePairsOut,
									__global volatile int* numConcavePairsOut,
									__global const btBvhSubtreeInfo* subtreeHeadersRoot,
									__global const btQuantizedBvhNode* quantizedNodesRoot,
									__global const b3BvhInfo* bvhInfos,
									int numPairs,
									int maxNumConcavePairsCapacity)
{
	int id = get_global_id(0);
	if (id>=numPairs)
		return;
	
	int bodyIndexA = pairs[id].x;
	int bodyIndexB = pairs[id].y;
	int collidableIndexA = rigidBodies[bodyIndexA].m_collidableIdx;
	int collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;
	
	//once the broadphase avoids static-static pairs, we can remove this test
	if ((rigidBodies[bodyIndexA].m_invMass==0) &&(rigidBodies[bodyIndexB].m_invMass==0))
	{
		return;
	}
		
	if (collidables[collidableIndexA].m_shapeType!=SHAPE_CONCAVE_TRIMESH)
		return;

	int shapeTypeB = collidables[collidableIndexB].m_shapeType;
		
	if (shapeTypeB!=SHAPE_CONVEX_HULL &&
		shapeTypeB!=SHAPE_SPHERE	&&
		shapeTypeB!=SHAPE_COMPOUND_OF_CONVEX_HULLS
		)
		return;

	b3BvhInfo bvhInfo = bvhInfos[collidables[collidableIndexA].m_numChildShapes];

	float4 bvhAabbMin = bvhInfo.m_aabbMin;
	float4 bvhAabbMax = bvhInfo.m_aabbMax;
	float4 bvhQuantization = bvhInfo.m_quantization;
	int numSubtreeHeaders = bvhInfo.m_numSubTrees;
	__global const btBvhSubtreeInfo* subtreeHeaders = &subtreeHeadersRoot[bvhInfo.m_subTreeOffset];
	__global const btQuantizedBvhNode* quantizedNodes = &quantizedNodesRoot[bvhInfo.m_nodeOffset];
	

	unsigned short int quantizedQueryAabbMin[3];
	unsigned short int quantizedQueryAabbMax[3];
	quantizeWithClamp(quantizedQueryAabbMin,aabbs[bodyIndexB].m_min,false,bvhAabbMin, bvhAabbMax,bvhQuantization);
	quantizeWithClamp(quantizedQueryAabbMax,aabbs[bodyIndexB].m_max,true ,bvhAabbMin, bvhAabbMax,bvhQuantization);
	
	for (int i=0;i<numSubtreeHeaders;i++)
	{
		btBvhSubtreeInfo subtree = subtreeHeaders[i];
				
		int overlap = testQuantizedAabbAgainstQuantizedAabb(quantizedQueryAabbMin,quantizedQueryAabbMax,subtree.m_quantizedAabbMin,subtree.m_quantizedAabbMax);
		if (overlap != 0)
		{
			int startNodeIndex = subtree.m_rootNodeIndex;
			int endNodeIndex = subtree.m_rootNodeIndex+subtree.m_subtreeSize;
			int curIndex = startNodeIndex;
			int escapeIndex;
			int isLeafNode;
			int aabbOverlap;
			while (curIndex < endNodeIndex)
			{
				btQuantizedBvhNode rootNode = quantizedNodes[curIndex];
				aabbOverlap = testQuantizedAabbAgainstQuantizedAabb(quantizedQueryAabbMin,quantizedQueryAabbMax,rootNode.m_quantizedAabbMin,rootNode.m_quantizedAabbMax);
				isLeafNode = isLeaf(&rootNode);
				if (aabbOverlap)
				{
					if (isLeafNode)
					{
						int triangleIndex = getTriangleIndex(&rootNode);
						if (shapeTypeB==SHAPE_COMPOUND_OF_CONVEX_HULLS)
						{
								int numChildrenB = collidables[collidableIndexB].m_numChildShapes;
								int pairIdx = atomic_add(numConcavePairsOut,numChildrenB);
								for (int b=0;b<numChildrenB;b++)
								{
									if ((pairIdx+b)<maxNumConcavePairsCapacity)
									{
										int childShapeIndexB = collidables[collidableIndexB].m_shapeIndex+b;
										int4 newPair = (int4)(bodyIndexA,bodyIndexB,triangleIndex,childShapeIndexB);
										concavePairsOut[pairIdx+b] = newPair;
									}
								}
						} else
						{
							int pairIdx = atomic_inc(numConcavePairsOut);
							if (pairIdx<maxNumConcavePairsCapacity)
							{
								int4 newPair = (int4)(bodyIndexA,bodyIndexB,triangleIndex,0);
								concavePairsOut[pairIdx] = newPair;
							}
						}
					} 
					curIndex++;
				} else
				{
					if (isLeafNode)
					{
						curIndex++;
					} else
					{
						escapeIndex = getEscapeIndex(&rootNode);
						curIndex += escapeIndex;
					}
				}
			}
		}
	}

}