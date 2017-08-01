/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2009 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#include "btOptimizedBvh.h"
#include "btStridingMeshInterface.h"
#include "LinearMath/btAabbUtil2.h"
#include "LinearMath/btIDebugDraw.h"


btOptimizedBvh::btOptimizedBvh()
{ 
}

btOptimizedBvh::~btOptimizedBvh()
{
}


void btOptimizedBvh::build(btStridingMeshInterface* triangles, bool useQuantizedAabbCompression, const btVector3& bvhAabbMin, const btVector3& bvhAabbMax)
{
	m_useQuantization = useQuantizedAabbCompression;


	// NodeArray	triangleNodes;

	struct	NodeTriangleCallback : public btInternalTriangleIndexCallback
	{

		NodeArray&	m_triangleNodes;

		NodeTriangleCallback& operator=(NodeTriangleCallback& other)
		{
			m_triangleNodes.copyFromArray(other.m_triangleNodes);
			return *this;
		}
		
		NodeTriangleCallback(NodeArray&	triangleNodes)
			:m_triangleNodes(triangleNodes)
		{
		}

		virtual void internalProcessTriangleIndex(btVector3* triangle,int partId,int  triangleIndex)
		{
			btOptimizedBvhNode node;
			btVector3	aabbMin,aabbMax;
			aabbMin.setValue(btScalar(BT_LARGE_FLOAT),btScalar(BT_LARGE_FLOAT),btScalar(BT_LARGE_FLOAT));
			aabbMax.setValue(btScalar(-BT_LARGE_FLOAT),btScalar(-BT_LARGE_FLOAT),btScalar(-BT_LARGE_FLOAT)); 
			aabbMin.setMin(triangle[0]);
			aabbMax.setMax(triangle[0]);
			aabbMin.setMin(triangle[1]);
			aabbMax.setMax(triangle[1]);
			aabbMin.setMin(triangle[2]);
			aabbMax.setMax(triangle[2]);

			//with quantization?
			node.m_aabbMinOrg = aabbMin;
			node.m_aabbMaxOrg = aabbMax;

			node.m_escapeIndex = -1;
	
			//for child nodes
			node.m_subPart = partId;
			node.m_triangleIndex = triangleIndex;
			m_triangleNodes.push_back(node);
		}
	};
	struct	QuantizedNodeTriangleCallback : public btInternalTriangleIndexCallback
	{
		QuantizedNodeArray&	m_triangleNodes;
		const btQuantizedBvh* m_optimizedTree; // for quantization

		QuantizedNodeTriangleCallback& operator=(QuantizedNodeTriangleCallback& other)
		{
			m_triangleNodes.copyFromArray(other.m_triangleNodes);
			m_optimizedTree = other.m_optimizedTree;
			return *this;
		}

		QuantizedNodeTriangleCallback(QuantizedNodeArray&	triangleNodes,const btQuantizedBvh* tree)
			:m_triangleNodes(triangleNodes),m_optimizedTree(tree)
		{
		}

		virtual void internalProcessTriangleIndex(btVector3* triangle,int partId,int  triangleIndex)
		{
			// The partId and triangle index must fit in the same (positive) integer
			btAssert(partId < (1<<MAX_NUM_PARTS_IN_BITS));
			btAssert(triangleIndex < (1<<(31-MAX_NUM_PARTS_IN_BITS)));
			//negative indices are reserved for escapeIndex
			btAssert(triangleIndex>=0);

			btQuantizedBvhNode node;
			btVector3	aabbMin,aabbMax;
			aabbMin.setValue(btScalar(BT_LARGE_FLOAT),btScalar(BT_LARGE_FLOAT),btScalar(BT_LARGE_FLOAT));
			aabbMax.setValue(btScalar(-BT_LARGE_FLOAT),btScalar(-BT_LARGE_FLOAT),btScalar(-BT_LARGE_FLOAT)); 
			aabbMin.setMin(triangle[0]);
			aabbMax.setMax(triangle[0]);
			aabbMin.setMin(triangle[1]);
			aabbMax.setMax(triangle[1]);
			aabbMin.setMin(triangle[2]);
			aabbMax.setMax(triangle[2]);

			//PCK: add these checks for zero dimensions of aabb
			const btScalar MIN_AABB_DIMENSION = btScalar(0.002);
			const btScalar MIN_AABB_HALF_DIMENSION = btScalar(0.001);
			if (aabbMax.x() - aabbMin.x() < MIN_AABB_DIMENSION)
			{
				aabbMax.setX(aabbMax.x() + MIN_AABB_HALF_DIMENSION);
				aabbMin.setX(aabbMin.x() - MIN_AABB_HALF_DIMENSION);
			}
			if (aabbMax.y() - aabbMin.y() < MIN_AABB_DIMENSION)
			{
				aabbMax.setY(aabbMax.y() + MIN_AABB_HALF_DIMENSION);
				aabbMin.setY(aabbMin.y() - MIN_AABB_HALF_DIMENSION);
			}
			if (aabbMax.z() - aabbMin.z() < MIN_AABB_DIMENSION)
			{
				aabbMax.setZ(aabbMax.z() + MIN_AABB_HALF_DIMENSION);
				aabbMin.setZ(aabbMin.z() - MIN_AABB_HALF_DIMENSION);
			}

			m_optimizedTree->quantize(&node.m_quantizedAabbMin[0],aabbMin,0);
			m_optimizedTree->quantize(&node.m_quantizedAabbMax[0],aabbMax,1);

			node.m_escapeIndexOrTriangleIndex = (partId<<(31-MAX_NUM_PARTS_IN_BITS)) | triangleIndex;

			m_triangleNodes.push_back(node);
		}
	};
	


	int numLeafNodes = 0;

	
	if (m_useQuantization)
	{

		//initialize quantization values
		setQuantizationValues(bvhAabbMin,bvhAabbMax);

		QuantizedNodeTriangleCallback	callback(m_quantizedLeafNodes,this);

	
		triangles->InternalProcessAllTriangles(&callback,m_bvhAabbMin,m_bvhAabbMax);

		//now we have an array of leafnodes in m_leafNodes
		numLeafNodes = m_quantizedLeafNodes.size();


		m_quantizedContiguousNodes.resize(2*numLeafNodes);


	} else
	{
		NodeTriangleCallback	callback(m_leafNodes);

		btVector3 aabbMin(btScalar(-BT_LARGE_FLOAT),btScalar(-BT_LARGE_FLOAT),btScalar(-BT_LARGE_FLOAT));
		btVector3 aabbMax(btScalar(BT_LARGE_FLOAT),btScalar(BT_LARGE_FLOAT),btScalar(BT_LARGE_FLOAT));

		triangles->InternalProcessAllTriangles(&callback,aabbMin,aabbMax);

		//now we have an array of leafnodes in m_leafNodes
		numLeafNodes = m_leafNodes.size();

		m_contiguousNodes.resize(2*numLeafNodes);
	}

	m_curNodeIndex = 0;

	buildTree(0,numLeafNodes);

	///if the entire tree is small then subtree size, we need to create a header info for the tree
	if(m_useQuantization && !m_SubtreeHeaders.size())
	{
		btBvhSubtreeInfo& subtree = m_SubtreeHeaders.expand();
		subtree.setAabbFromQuantizeNode(m_quantizedContiguousNodes[0]);
		subtree.m_rootNodeIndex = 0;
		subtree.m_subtreeSize = m_quantizedContiguousNodes[0].isLeafNode() ? 1 : m_quantizedContiguousNodes[0].getEscapeIndex();
	}

	//PCK: update the copy of the size
	m_subtreeHeaderCount = m_SubtreeHeaders.size();

	//PCK: clear m_quantizedLeafNodes and m_leafNodes, they are temporary
	m_quantizedLeafNodes.clear();
	m_leafNodes.clear();
}




void	btOptimizedBvh::refit(btStridingMeshInterface* meshInterface,const btVector3& aabbMin,const btVector3& aabbMax)
{
	if (m_useQuantization)
	{

		setQuantizationValues(aabbMin,aabbMax);

		updateBvhNodes(meshInterface,0,m_curNodeIndex,0);

		///now update all subtree headers

		int i;
		for (i=0;i<m_SubtreeHeaders.size();i++)
		{
			btBvhSubtreeInfo& subtree = m_SubtreeHeaders[i];
			subtree.setAabbFromQuantizeNode(m_quantizedContiguousNodes[subtree.m_rootNodeIndex]);
		}

	} else
	{

	}
}




void	btOptimizedBvh::refitPartial(btStridingMeshInterface* meshInterface,const btVector3& aabbMin,const btVector3& aabbMax)
{
	//incrementally initialize quantization values
	btAssert(m_useQuantization);

	btAssert(aabbMin.getX() > m_bvhAabbMin.getX());
	btAssert(aabbMin.getY() > m_bvhAabbMin.getY());
	btAssert(aabbMin.getZ() > m_bvhAabbMin.getZ());

	btAssert(aabbMax.getX() < m_bvhAabbMax.getX());
	btAssert(aabbMax.getY() < m_bvhAabbMax.getY());
	btAssert(aabbMax.getZ() < m_bvhAabbMax.getZ());

	///we should update all quantization values, using updateBvhNodes(meshInterface);
	///but we only update chunks that overlap the given aabb
	
	unsigned short	quantizedQueryAabbMin[3];
	unsigned short	quantizedQueryAabbMax[3];

	quantize(&quantizedQueryAabbMin[0],aabbMin,0);
	quantize(&quantizedQueryAabbMax[0],aabbMax,1);

	int i;
	for (i=0;i<this->m_SubtreeHeaders.size();i++)
	{
		btBvhSubtreeInfo& subtree = m_SubtreeHeaders[i];

		//PCK: unsigned instead of bool
		unsigned overlap = testQuantizedAabbAgainstQuantizedAabb(quantizedQueryAabbMin,quantizedQueryAabbMax,subtree.m_quantizedAabbMin,subtree.m_quantizedAabbMax);
		if (overlap != 0)
		{
			updateBvhNodes(meshInterface,subtree.m_rootNodeIndex,subtree.m_rootNodeIndex+subtree.m_subtreeSize,i);

			subtree.setAabbFromQuantizeNode(m_quantizedContiguousNodes[subtree.m_rootNodeIndex]);
		}
	}
	
}

void	btOptimizedBvh::updateBvhNodes(btStridingMeshInterface* meshInterface,int firstNode,int endNode,int index)
{
	(void)index;

	btAssert(m_useQuantization);

	int curNodeSubPart=-1;

	//get access info to trianglemesh data
		const unsigned char *vertexbase = 0;
		int numverts = 0;
		PHY_ScalarType type = PHY_INTEGER;
		int stride = 0;
		const unsigned char *indexbase = 0;
		int indexstride = 0;
		int numfaces = 0;
		PHY_ScalarType indicestype = PHY_INTEGER;

		btVector3	triangleVerts[3];
		btVector3	aabbMin,aabbMax;
		const btVector3& meshScaling = meshInterface->getScaling();
		
		int i;
		for (i=endNode-1;i>=firstNode;i--)
		{


			btQuantizedBvhNode& curNode = m_quantizedContiguousNodes[i];
			if (curNode.isLeafNode())
			{
				//recalc aabb from triangle data
				int nodeSubPart = curNode.getPartId();
				int nodeTriangleIndex = curNode.getTriangleIndex();
				if (nodeSubPart != curNodeSubPart)
				{
					if (curNodeSubPart >= 0)
						meshInterface->unLockReadOnlyVertexBase(curNodeSubPart);
					meshInterface->getLockedReadOnlyVertexIndexBase(&vertexbase,numverts,	type,stride,&indexbase,indexstride,numfaces,indicestype,nodeSubPart);

					curNodeSubPart = nodeSubPart;
					btAssert(indicestype==PHY_INTEGER||indicestype==PHY_SHORT);
				}
				//triangles->getLockedReadOnlyVertexIndexBase(vertexBase,numVerts,

				unsigned int* gfxbase = (unsigned int*)(indexbase+nodeTriangleIndex*indexstride);
				
				
				for (int j=2;j>=0;j--)
				{
					
					int graphicsindex = indicestype==PHY_SHORT?((unsigned short*)gfxbase)[j]:gfxbase[j];
					if (type == PHY_FLOAT)
					{
						float* graphicsbase = (float*)(vertexbase+graphicsindex*stride);
						triangleVerts[j] = btVector3(
							graphicsbase[0]*meshScaling.getX(),
							graphicsbase[1]*meshScaling.getY(),
							graphicsbase[2]*meshScaling.getZ());
					}
					else
					{
						double* graphicsbase = (double*)(vertexbase+graphicsindex*stride);
						triangleVerts[j] = btVector3( btScalar(graphicsbase[0]*meshScaling.getX()), btScalar(graphicsbase[1]*meshScaling.getY()), btScalar(graphicsbase[2]*meshScaling.getZ()));
					}
				}


				
				aabbMin.setValue(btScalar(BT_LARGE_FLOAT),btScalar(BT_LARGE_FLOAT),btScalar(BT_LARGE_FLOAT));
				aabbMax.setValue(btScalar(-BT_LARGE_FLOAT),btScalar(-BT_LARGE_FLOAT),btScalar(-BT_LARGE_FLOAT)); 
				aabbMin.setMin(triangleVerts[0]);
				aabbMax.setMax(triangleVerts[0]);
				aabbMin.setMin(triangleVerts[1]);
				aabbMax.setMax(triangleVerts[1]);
				aabbMin.setMin(triangleVerts[2]);
				aabbMax.setMax(triangleVerts[2]);

				quantize(&curNode.m_quantizedAabbMin[0],aabbMin,0);
				quantize(&curNode.m_quantizedAabbMax[0],aabbMax,1);
				
			} else
			{
				//combine aabb from both children

				btQuantizedBvhNode* leftChildNode = &m_quantizedContiguousNodes[i+1];
				
				btQuantizedBvhNode* rightChildNode = leftChildNode->isLeafNode() ? &m_quantizedContiguousNodes[i+2] :
					&m_quantizedContiguousNodes[i+1+leftChildNode->getEscapeIndex()];
				

				{
					for (int i=0;i<3;i++)
					{
						curNode.m_quantizedAabbMin[i] = leftChildNode->m_quantizedAabbMin[i];
						if (curNode.m_quantizedAabbMin[i]>rightChildNode->m_quantizedAabbMin[i])
							curNode.m_quantizedAabbMin[i]=rightChildNode->m_quantizedAabbMin[i];

						curNode.m_quantizedAabbMax[i] = leftChildNode->m_quantizedAabbMax[i];
						if (curNode.m_quantizedAabbMax[i] < rightChildNode->m_quantizedAabbMax[i])
							curNode.m_quantizedAabbMax[i] = rightChildNode->m_quantizedAabbMax[i];
					}
				}
			}

		}

		if (curNodeSubPart >= 0)
			meshInterface->unLockReadOnlyVertexBase(curNodeSubPart);

		
}

///deSerializeInPlace loads and initializes a BVH from a buffer in memory 'in place'
btOptimizedBvh* btOptimizedBvh::deSerializeInPlace(void *i_alignedDataBuffer, unsigned int i_dataBufferSize, bool i_swapEndian)
{
	btQuantizedBvh* bvh = btQuantizedBvh::deSerializeInPlace(i_alignedDataBuffer,i_dataBufferSize,i_swapEndian);
	
	//we don't add additional data so just do a static upcast
	return static_cast<btOptimizedBvh*>(bvh);
}
