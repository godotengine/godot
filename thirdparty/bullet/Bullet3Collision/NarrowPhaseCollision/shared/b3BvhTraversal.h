

#include "Bullet3Common/shared/b3Int4.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Collidable.h"
#include "Bullet3Collision/BroadPhaseCollision/shared/b3Aabb.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3BvhSubtreeInfoData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3QuantizedBvhNodeData.h"



// work-in-progress
void   b3BvhTraversal( __global const b3Int4* pairs, 
									__global const b3RigidBodyData* rigidBodies, 
									__global const b3Collidable* collidables,
									__global b3Aabb* aabbs,
									__global b3Int4* concavePairsOut,
									__global volatile int* numConcavePairsOut,
									__global const b3BvhSubtreeInfo* subtreeHeadersRoot,
									__global const b3QuantizedBvhNode* quantizedNodesRoot,
									__global const b3BvhInfo* bvhInfos,
									int numPairs,
									int maxNumConcavePairsCapacity,
									int id)
{
	
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

	b3Float4	bvhAabbMin = bvhInfo.m_aabbMin;
	b3Float4	bvhAabbMax = bvhInfo.m_aabbMax;
	b3Float4	bvhQuantization = bvhInfo.m_quantization;
	int numSubtreeHeaders = bvhInfo.m_numSubTrees;
	__global const b3BvhSubtreeInfoData* subtreeHeaders = &subtreeHeadersRoot[bvhInfo.m_subTreeOffset];
	__global const b3QuantizedBvhNodeData* quantizedNodes = &quantizedNodesRoot[bvhInfo.m_nodeOffset];
	

	unsigned short int quantizedQueryAabbMin[3];
	unsigned short int quantizedQueryAabbMax[3];
	b3QuantizeWithClamp(quantizedQueryAabbMin,aabbs[bodyIndexB].m_minVec,false,bvhAabbMin, bvhAabbMax,bvhQuantization);
	b3QuantizeWithClamp(quantizedQueryAabbMax,aabbs[bodyIndexB].m_maxVec,true ,bvhAabbMin, bvhAabbMax,bvhQuantization);
	
	for (int i=0;i<numSubtreeHeaders;i++)
	{
		b3BvhSubtreeInfoData subtree = subtreeHeaders[i];
				
		int overlap = b3TestQuantizedAabbAgainstQuantizedAabbSlow(quantizedQueryAabbMin,quantizedQueryAabbMax,subtree.m_quantizedAabbMin,subtree.m_quantizedAabbMax);
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
				b3QuantizedBvhNodeData rootNode = quantizedNodes[curIndex];
				aabbOverlap = b3TestQuantizedAabbAgainstQuantizedAabbSlow(quantizedQueryAabbMin,quantizedQueryAabbMax,rootNode.m_quantizedAabbMin,rootNode.m_quantizedAabbMax);
				isLeafNode = b3IsLeaf(&rootNode);
				if (aabbOverlap)
				{
					if (isLeafNode)
					{
						int triangleIndex = b3GetTriangleIndex(&rootNode);
						if (shapeTypeB==SHAPE_COMPOUND_OF_CONVEX_HULLS)
						{
								int numChildrenB = collidables[collidableIndexB].m_numChildShapes;
								int pairIdx = b3AtomicAdd (numConcavePairsOut,numChildrenB);
								for (int b=0;b<numChildrenB;b++)
								{
									if ((pairIdx+b)<maxNumConcavePairsCapacity)
									{
										int childShapeIndexB = collidables[collidableIndexB].m_shapeIndex+b;
										b3Int4 newPair = b3MakeInt4(bodyIndexA,bodyIndexB,triangleIndex,childShapeIndexB);
										concavePairsOut[pairIdx+b] = newPair;
									}
								}
						} else
						{
							int pairIdx = b3AtomicInc(numConcavePairsOut);
							if (pairIdx<maxNumConcavePairsCapacity)
							{
								b3Int4 newPair = b3MakeInt4(bodyIndexA,bodyIndexB,triangleIndex,0);
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
						escapeIndex = b3GetEscapeIndex(&rootNode);
						curIndex += escapeIndex;
					}
				}
			}
		}
	}

}