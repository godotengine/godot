/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "b3QuantizedBvh.h"

#include "Bullet3Geometry/b3AabbUtil.h"

#define RAYAABB2

b3QuantizedBvh::b3QuantizedBvh() : m_bulletVersion(B3_BULLET_VERSION),
								   m_useQuantization(false),
								   m_traversalMode(TRAVERSAL_STACKLESS_CACHE_FRIENDLY)
								   //m_traversalMode(TRAVERSAL_STACKLESS)
								   //m_traversalMode(TRAVERSAL_RECURSIVE)
								   ,
								   m_subtreeHeaderCount(0)  //PCK: add this line
{
	m_bvhAabbMin.setValue(-B3_INFINITY, -B3_INFINITY, -B3_INFINITY);
	m_bvhAabbMax.setValue(B3_INFINITY, B3_INFINITY, B3_INFINITY);
}

void b3QuantizedBvh::buildInternal()
{
	///assumes that caller filled in the m_quantizedLeafNodes
	m_useQuantization = true;
	int numLeafNodes = 0;

	if (m_useQuantization)
	{
		//now we have an array of leafnodes in m_leafNodes
		numLeafNodes = m_quantizedLeafNodes.size();

		m_quantizedContiguousNodes.resize(2 * numLeafNodes);
	}

	m_curNodeIndex = 0;

	buildTree(0, numLeafNodes);

	///if the entire tree is small then subtree size, we need to create a header info for the tree
	if (m_useQuantization && !m_SubtreeHeaders.size())
	{
		b3BvhSubtreeInfo& subtree = m_SubtreeHeaders.expand();
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

///just for debugging, to visualize the individual patches/subtrees
#ifdef DEBUG_PATCH_COLORS
b3Vector3 color[4] =
	{
		b3Vector3(1, 0, 0),
		b3Vector3(0, 1, 0),
		b3Vector3(0, 0, 1),
		b3Vector3(0, 1, 1)};
#endif  //DEBUG_PATCH_COLORS

void b3QuantizedBvh::setQuantizationValues(const b3Vector3& bvhAabbMin, const b3Vector3& bvhAabbMax, b3Scalar quantizationMargin)
{
	//enlarge the AABB to avoid division by zero when initializing the quantization values
	b3Vector3 clampValue = b3MakeVector3(quantizationMargin, quantizationMargin, quantizationMargin);
	m_bvhAabbMin = bvhAabbMin - clampValue;
	m_bvhAabbMax = bvhAabbMax + clampValue;
	b3Vector3 aabbSize = m_bvhAabbMax - m_bvhAabbMin;
	m_bvhQuantization = b3MakeVector3(b3Scalar(65533.0), b3Scalar(65533.0), b3Scalar(65533.0)) / aabbSize;
	m_useQuantization = true;
}

b3QuantizedBvh::~b3QuantizedBvh()
{
}

#ifdef DEBUG_TREE_BUILDING
int gStackDepth = 0;
int gMaxStackDepth = 0;
#endif  //DEBUG_TREE_BUILDING

void b3QuantizedBvh::buildTree(int startIndex, int endIndex)
{
#ifdef DEBUG_TREE_BUILDING
	gStackDepth++;
	if (gStackDepth > gMaxStackDepth)
		gMaxStackDepth = gStackDepth;
#endif  //DEBUG_TREE_BUILDING

	int splitAxis, splitIndex, i;
	int numIndices = endIndex - startIndex;
	int curIndex = m_curNodeIndex;

	b3Assert(numIndices > 0);

	if (numIndices == 1)
	{
#ifdef DEBUG_TREE_BUILDING
		gStackDepth--;
#endif  //DEBUG_TREE_BUILDING

		assignInternalNodeFromLeafNode(m_curNodeIndex, startIndex);

		m_curNodeIndex++;
		return;
	}
	//calculate Best Splitting Axis and where to split it. Sort the incoming 'leafNodes' array within range 'startIndex/endIndex'.

	splitAxis = calcSplittingAxis(startIndex, endIndex);

	splitIndex = sortAndCalcSplittingIndex(startIndex, endIndex, splitAxis);

	int internalNodeIndex = m_curNodeIndex;

	//set the min aabb to 'inf' or a max value, and set the max aabb to a -inf/minimum value.
	//the aabb will be expanded during buildTree/mergeInternalNodeAabb with actual node values
	setInternalNodeAabbMin(m_curNodeIndex, m_bvhAabbMax);  //can't use b3Vector3(B3_INFINITY,B3_INFINITY,B3_INFINITY)) because of quantization
	setInternalNodeAabbMax(m_curNodeIndex, m_bvhAabbMin);  //can't use b3Vector3(-B3_INFINITY,-B3_INFINITY,-B3_INFINITY)) because of quantization

	for (i = startIndex; i < endIndex; i++)
	{
		mergeInternalNodeAabb(m_curNodeIndex, getAabbMin(i), getAabbMax(i));
	}

	m_curNodeIndex++;

	//internalNode->m_escapeIndex;

	int leftChildNodexIndex = m_curNodeIndex;

	//build left child tree
	buildTree(startIndex, splitIndex);

	int rightChildNodexIndex = m_curNodeIndex;
	//build right child tree
	buildTree(splitIndex, endIndex);

#ifdef DEBUG_TREE_BUILDING
	gStackDepth--;
#endif  //DEBUG_TREE_BUILDING

	int escapeIndex = m_curNodeIndex - curIndex;

	if (m_useQuantization)
	{
		//escapeIndex is the number of nodes of this subtree
		const int sizeQuantizedNode = sizeof(b3QuantizedBvhNode);
		const int treeSizeInBytes = escapeIndex * sizeQuantizedNode;
		if (treeSizeInBytes > MAX_SUBTREE_SIZE_IN_BYTES)
		{
			updateSubtreeHeaders(leftChildNodexIndex, rightChildNodexIndex);
		}
	}
	else
	{
	}

	setInternalNodeEscapeIndex(internalNodeIndex, escapeIndex);
}

void b3QuantizedBvh::updateSubtreeHeaders(int leftChildNodexIndex, int rightChildNodexIndex)
{
	b3Assert(m_useQuantization);

	b3QuantizedBvhNode& leftChildNode = m_quantizedContiguousNodes[leftChildNodexIndex];
	int leftSubTreeSize = leftChildNode.isLeafNode() ? 1 : leftChildNode.getEscapeIndex();
	int leftSubTreeSizeInBytes = leftSubTreeSize * static_cast<int>(sizeof(b3QuantizedBvhNode));

	b3QuantizedBvhNode& rightChildNode = m_quantizedContiguousNodes[rightChildNodexIndex];
	int rightSubTreeSize = rightChildNode.isLeafNode() ? 1 : rightChildNode.getEscapeIndex();
	int rightSubTreeSizeInBytes = rightSubTreeSize * static_cast<int>(sizeof(b3QuantizedBvhNode));

	if (leftSubTreeSizeInBytes <= MAX_SUBTREE_SIZE_IN_BYTES)
	{
		b3BvhSubtreeInfo& subtree = m_SubtreeHeaders.expand();
		subtree.setAabbFromQuantizeNode(leftChildNode);
		subtree.m_rootNodeIndex = leftChildNodexIndex;
		subtree.m_subtreeSize = leftSubTreeSize;
	}

	if (rightSubTreeSizeInBytes <= MAX_SUBTREE_SIZE_IN_BYTES)
	{
		b3BvhSubtreeInfo& subtree = m_SubtreeHeaders.expand();
		subtree.setAabbFromQuantizeNode(rightChildNode);
		subtree.m_rootNodeIndex = rightChildNodexIndex;
		subtree.m_subtreeSize = rightSubTreeSize;
	}

	//PCK: update the copy of the size
	m_subtreeHeaderCount = m_SubtreeHeaders.size();
}

int b3QuantizedBvh::sortAndCalcSplittingIndex(int startIndex, int endIndex, int splitAxis)
{
	int i;
	int splitIndex = startIndex;
	int numIndices = endIndex - startIndex;
	b3Scalar splitValue;

	b3Vector3 means = b3MakeVector3(b3Scalar(0.), b3Scalar(0.), b3Scalar(0.));
	for (i = startIndex; i < endIndex; i++)
	{
		b3Vector3 center = b3Scalar(0.5) * (getAabbMax(i) + getAabbMin(i));
		means += center;
	}
	means *= (b3Scalar(1.) / (b3Scalar)numIndices);

	splitValue = means[splitAxis];

	//sort leafNodes so all values larger then splitValue comes first, and smaller values start from 'splitIndex'.
	for (i = startIndex; i < endIndex; i++)
	{
		b3Vector3 center = b3Scalar(0.5) * (getAabbMax(i) + getAabbMin(i));
		if (center[splitAxis] > splitValue)
		{
			//swap
			swapLeafNodes(i, splitIndex);
			splitIndex++;
		}
	}

	//if the splitIndex causes unbalanced trees, fix this by using the center in between startIndex and endIndex
	//otherwise the tree-building might fail due to stack-overflows in certain cases.
	//unbalanced1 is unsafe: it can cause stack overflows
	//bool unbalanced1 = ((splitIndex==startIndex) || (splitIndex == (endIndex-1)));

	//unbalanced2 should work too: always use center (perfect balanced trees)
	//bool unbalanced2 = true;

	//this should be safe too:
	int rangeBalancedIndices = numIndices / 3;
	bool unbalanced = ((splitIndex <= (startIndex + rangeBalancedIndices)) || (splitIndex >= (endIndex - 1 - rangeBalancedIndices)));

	if (unbalanced)
	{
		splitIndex = startIndex + (numIndices >> 1);
	}

	bool unbal = (splitIndex == startIndex) || (splitIndex == (endIndex));
	(void)unbal;
	b3Assert(!unbal);

	return splitIndex;
}

int b3QuantizedBvh::calcSplittingAxis(int startIndex, int endIndex)
{
	int i;

	b3Vector3 means = b3MakeVector3(b3Scalar(0.), b3Scalar(0.), b3Scalar(0.));
	b3Vector3 variance = b3MakeVector3(b3Scalar(0.), b3Scalar(0.), b3Scalar(0.));
	int numIndices = endIndex - startIndex;

	for (i = startIndex; i < endIndex; i++)
	{
		b3Vector3 center = b3Scalar(0.5) * (getAabbMax(i) + getAabbMin(i));
		means += center;
	}
	means *= (b3Scalar(1.) / (b3Scalar)numIndices);

	for (i = startIndex; i < endIndex; i++)
	{
		b3Vector3 center = b3Scalar(0.5) * (getAabbMax(i) + getAabbMin(i));
		b3Vector3 diff2 = center - means;
		diff2 = diff2 * diff2;
		variance += diff2;
	}
	variance *= (b3Scalar(1.) / ((b3Scalar)numIndices - 1));

	return variance.maxAxis();
}

void b3QuantizedBvh::reportAabbOverlappingNodex(b3NodeOverlapCallback* nodeCallback, const b3Vector3& aabbMin, const b3Vector3& aabbMax) const
{
	//either choose recursive traversal (walkTree) or stackless (walkStacklessTree)

	if (m_useQuantization)
	{
		///quantize query AABB
		unsigned short int quantizedQueryAabbMin[3];
		unsigned short int quantizedQueryAabbMax[3];
		quantizeWithClamp(quantizedQueryAabbMin, aabbMin, 0);
		quantizeWithClamp(quantizedQueryAabbMax, aabbMax, 1);

		switch (m_traversalMode)
		{
			case TRAVERSAL_STACKLESS:
				walkStacklessQuantizedTree(nodeCallback, quantizedQueryAabbMin, quantizedQueryAabbMax, 0, m_curNodeIndex);
				break;
			case TRAVERSAL_STACKLESS_CACHE_FRIENDLY:
				walkStacklessQuantizedTreeCacheFriendly(nodeCallback, quantizedQueryAabbMin, quantizedQueryAabbMax);
				break;
			case TRAVERSAL_RECURSIVE:
			{
				const b3QuantizedBvhNode* rootNode = &m_quantizedContiguousNodes[0];
				walkRecursiveQuantizedTreeAgainstQueryAabb(rootNode, nodeCallback, quantizedQueryAabbMin, quantizedQueryAabbMax);
			}
			break;
			default:
				//unsupported
				b3Assert(0);
		}
	}
	else
	{
		walkStacklessTree(nodeCallback, aabbMin, aabbMax);
	}
}

static int b3s_maxIterations = 0;

void b3QuantizedBvh::walkStacklessTree(b3NodeOverlapCallback* nodeCallback, const b3Vector3& aabbMin, const b3Vector3& aabbMax) const
{
	b3Assert(!m_useQuantization);

	const b3OptimizedBvhNode* rootNode = &m_contiguousNodes[0];
	int escapeIndex, curIndex = 0;
	int walkIterations = 0;
	bool isLeafNode;
	//PCK: unsigned instead of bool
	unsigned aabbOverlap;

	while (curIndex < m_curNodeIndex)
	{
		//catch bugs in tree data
		b3Assert(walkIterations < m_curNodeIndex);

		walkIterations++;
		aabbOverlap = b3TestAabbAgainstAabb2(aabbMin, aabbMax, rootNode->m_aabbMinOrg, rootNode->m_aabbMaxOrg);
		isLeafNode = rootNode->m_escapeIndex == -1;

		//PCK: unsigned instead of bool
		if (isLeafNode && (aabbOverlap != 0))
		{
			nodeCallback->processNode(rootNode->m_subPart, rootNode->m_triangleIndex);
		}

		//PCK: unsigned instead of bool
		if ((aabbOverlap != 0) || isLeafNode)
		{
			rootNode++;
			curIndex++;
		}
		else
		{
			escapeIndex = rootNode->m_escapeIndex;
			rootNode += escapeIndex;
			curIndex += escapeIndex;
		}
	}
	if (b3s_maxIterations < walkIterations)
		b3s_maxIterations = walkIterations;
}

/*
///this was the original recursive traversal, before we optimized towards stackless traversal
void	b3QuantizedBvh::walkTree(b3OptimizedBvhNode* rootNode,b3NodeOverlapCallback* nodeCallback,const b3Vector3& aabbMin,const b3Vector3& aabbMax) const
{
	bool isLeafNode, aabbOverlap = TestAabbAgainstAabb2(aabbMin,aabbMax,rootNode->m_aabbMin,rootNode->m_aabbMax);
	if (aabbOverlap)
	{
		isLeafNode = (!rootNode->m_leftChild && !rootNode->m_rightChild);
		if (isLeafNode)
		{
			nodeCallback->processNode(rootNode);
		} else
		{
			walkTree(rootNode->m_leftChild,nodeCallback,aabbMin,aabbMax);
			walkTree(rootNode->m_rightChild,nodeCallback,aabbMin,aabbMax);
		}
	}

}
*/

void b3QuantizedBvh::walkRecursiveQuantizedTreeAgainstQueryAabb(const b3QuantizedBvhNode* currentNode, b3NodeOverlapCallback* nodeCallback, unsigned short int* quantizedQueryAabbMin, unsigned short int* quantizedQueryAabbMax) const
{
	b3Assert(m_useQuantization);

	bool isLeafNode;
	//PCK: unsigned instead of bool
	unsigned aabbOverlap;

	//PCK: unsigned instead of bool
	aabbOverlap = b3TestQuantizedAabbAgainstQuantizedAabb(quantizedQueryAabbMin, quantizedQueryAabbMax, currentNode->m_quantizedAabbMin, currentNode->m_quantizedAabbMax);
	isLeafNode = currentNode->isLeafNode();

	//PCK: unsigned instead of bool
	if (aabbOverlap != 0)
	{
		if (isLeafNode)
		{
			nodeCallback->processNode(currentNode->getPartId(), currentNode->getTriangleIndex());
		}
		else
		{
			//process left and right children
			const b3QuantizedBvhNode* leftChildNode = currentNode + 1;
			walkRecursiveQuantizedTreeAgainstQueryAabb(leftChildNode, nodeCallback, quantizedQueryAabbMin, quantizedQueryAabbMax);

			const b3QuantizedBvhNode* rightChildNode = leftChildNode->isLeafNode() ? leftChildNode + 1 : leftChildNode + leftChildNode->getEscapeIndex();
			walkRecursiveQuantizedTreeAgainstQueryAabb(rightChildNode, nodeCallback, quantizedQueryAabbMin, quantizedQueryAabbMax);
		}
	}
}

void b3QuantizedBvh::walkStacklessTreeAgainstRay(b3NodeOverlapCallback* nodeCallback, const b3Vector3& raySource, const b3Vector3& rayTarget, const b3Vector3& aabbMin, const b3Vector3& aabbMax, int startNodeIndex, int endNodeIndex) const
{
	b3Assert(!m_useQuantization);

	const b3OptimizedBvhNode* rootNode = &m_contiguousNodes[0];
	int escapeIndex, curIndex = 0;
	int walkIterations = 0;
	bool isLeafNode;
	//PCK: unsigned instead of bool
	unsigned aabbOverlap = 0;
	unsigned rayBoxOverlap = 0;
	b3Scalar lambda_max = 1.0;

	/* Quick pruning by quantized box */
	b3Vector3 rayAabbMin = raySource;
	b3Vector3 rayAabbMax = raySource;
	rayAabbMin.setMin(rayTarget);
	rayAabbMax.setMax(rayTarget);

	/* Add box cast extents to bounding box */
	rayAabbMin += aabbMin;
	rayAabbMax += aabbMax;

#ifdef RAYAABB2
	b3Vector3 rayDir = (rayTarget - raySource);
	rayDir.normalize();
	lambda_max = rayDir.dot(rayTarget - raySource);
	///what about division by zero? --> just set rayDirection[i] to 1.0
	b3Vector3 rayDirectionInverse;
	rayDirectionInverse[0] = rayDir[0] == b3Scalar(0.0) ? b3Scalar(B3_LARGE_FLOAT) : b3Scalar(1.0) / rayDir[0];
	rayDirectionInverse[1] = rayDir[1] == b3Scalar(0.0) ? b3Scalar(B3_LARGE_FLOAT) : b3Scalar(1.0) / rayDir[1];
	rayDirectionInverse[2] = rayDir[2] == b3Scalar(0.0) ? b3Scalar(B3_LARGE_FLOAT) : b3Scalar(1.0) / rayDir[2];
	unsigned int sign[3] = {rayDirectionInverse[0] < 0.0, rayDirectionInverse[1] < 0.0, rayDirectionInverse[2] < 0.0};
#endif

	b3Vector3 bounds[2];

	while (curIndex < m_curNodeIndex)
	{
		b3Scalar param = 1.0;
		//catch bugs in tree data
		b3Assert(walkIterations < m_curNodeIndex);

		walkIterations++;

		bounds[0] = rootNode->m_aabbMinOrg;
		bounds[1] = rootNode->m_aabbMaxOrg;
		/* Add box cast extents */
		bounds[0] -= aabbMax;
		bounds[1] -= aabbMin;

		aabbOverlap = b3TestAabbAgainstAabb2(rayAabbMin, rayAabbMax, rootNode->m_aabbMinOrg, rootNode->m_aabbMaxOrg);
		//perhaps profile if it is worth doing the aabbOverlap test first

#ifdef RAYAABB2
		///careful with this check: need to check division by zero (above) and fix the unQuantize method
		///thanks Joerg/hiker for the reproduction case!
		///http://www.bulletphysics.com/Bullet/phpBB3/viewtopic.php?f=9&t=1858
		rayBoxOverlap = aabbOverlap ? b3RayAabb2(raySource, rayDirectionInverse, sign, bounds, param, 0.0f, lambda_max) : false;

#else
		b3Vector3 normal;
		rayBoxOverlap = b3RayAabb(raySource, rayTarget, bounds[0], bounds[1], param, normal);
#endif

		isLeafNode = rootNode->m_escapeIndex == -1;

		//PCK: unsigned instead of bool
		if (isLeafNode && (rayBoxOverlap != 0))
		{
			nodeCallback->processNode(rootNode->m_subPart, rootNode->m_triangleIndex);
		}

		//PCK: unsigned instead of bool
		if ((rayBoxOverlap != 0) || isLeafNode)
		{
			rootNode++;
			curIndex++;
		}
		else
		{
			escapeIndex = rootNode->m_escapeIndex;
			rootNode += escapeIndex;
			curIndex += escapeIndex;
		}
	}
	if (b3s_maxIterations < walkIterations)
		b3s_maxIterations = walkIterations;
}

void b3QuantizedBvh::walkStacklessQuantizedTreeAgainstRay(b3NodeOverlapCallback* nodeCallback, const b3Vector3& raySource, const b3Vector3& rayTarget, const b3Vector3& aabbMin, const b3Vector3& aabbMax, int startNodeIndex, int endNodeIndex) const
{
	b3Assert(m_useQuantization);

	int curIndex = startNodeIndex;
	int walkIterations = 0;
	int subTreeSize = endNodeIndex - startNodeIndex;
	(void)subTreeSize;

	const b3QuantizedBvhNode* rootNode = &m_quantizedContiguousNodes[startNodeIndex];
	int escapeIndex;

	bool isLeafNode;
	//PCK: unsigned instead of bool
	unsigned boxBoxOverlap = 0;
	unsigned rayBoxOverlap = 0;

	b3Scalar lambda_max = 1.0;

#ifdef RAYAABB2
	b3Vector3 rayDirection = (rayTarget - raySource);
	rayDirection.normalize();
	lambda_max = rayDirection.dot(rayTarget - raySource);
	///what about division by zero? --> just set rayDirection[i] to 1.0
	rayDirection[0] = rayDirection[0] == b3Scalar(0.0) ? b3Scalar(B3_LARGE_FLOAT) : b3Scalar(1.0) / rayDirection[0];
	rayDirection[1] = rayDirection[1] == b3Scalar(0.0) ? b3Scalar(B3_LARGE_FLOAT) : b3Scalar(1.0) / rayDirection[1];
	rayDirection[2] = rayDirection[2] == b3Scalar(0.0) ? b3Scalar(B3_LARGE_FLOAT) : b3Scalar(1.0) / rayDirection[2];
	unsigned int sign[3] = {rayDirection[0] < 0.0, rayDirection[1] < 0.0, rayDirection[2] < 0.0};
#endif

	/* Quick pruning by quantized box */
	b3Vector3 rayAabbMin = raySource;
	b3Vector3 rayAabbMax = raySource;
	rayAabbMin.setMin(rayTarget);
	rayAabbMax.setMax(rayTarget);

	/* Add box cast extents to bounding box */
	rayAabbMin += aabbMin;
	rayAabbMax += aabbMax;

	unsigned short int quantizedQueryAabbMin[3];
	unsigned short int quantizedQueryAabbMax[3];
	quantizeWithClamp(quantizedQueryAabbMin, rayAabbMin, 0);
	quantizeWithClamp(quantizedQueryAabbMax, rayAabbMax, 1);

	while (curIndex < endNodeIndex)
	{
//#define VISUALLY_ANALYZE_BVH 1
#ifdef VISUALLY_ANALYZE_BVH
		//some code snippet to debugDraw aabb, to visually analyze bvh structure
		static int drawPatch = 0;
		//need some global access to a debugDrawer
		extern b3IDebugDraw* debugDrawerPtr;
		if (curIndex == drawPatch)
		{
			b3Vector3 aabbMin, aabbMax;
			aabbMin = unQuantize(rootNode->m_quantizedAabbMin);
			aabbMax = unQuantize(rootNode->m_quantizedAabbMax);
			b3Vector3 color(1, 0, 0);
			debugDrawerPtr->drawAabb(aabbMin, aabbMax, color);
		}
#endif  //VISUALLY_ANALYZE_BVH

		//catch bugs in tree data
		b3Assert(walkIterations < subTreeSize);

		walkIterations++;
		//PCK: unsigned instead of bool
		// only interested if this is closer than any previous hit
		b3Scalar param = 1.0;
		rayBoxOverlap = 0;
		boxBoxOverlap = b3TestQuantizedAabbAgainstQuantizedAabb(quantizedQueryAabbMin, quantizedQueryAabbMax, rootNode->m_quantizedAabbMin, rootNode->m_quantizedAabbMax);
		isLeafNode = rootNode->isLeafNode();
		if (boxBoxOverlap)
		{
			b3Vector3 bounds[2];
			bounds[0] = unQuantize(rootNode->m_quantizedAabbMin);
			bounds[1] = unQuantize(rootNode->m_quantizedAabbMax);
			/* Add box cast extents */
			bounds[0] -= aabbMax;
			bounds[1] -= aabbMin;
#if 0
			b3Vector3 normal;
			bool ra2 = b3RayAabb2 (raySource, rayDirection, sign, bounds, param, 0.0, lambda_max);
			bool ra = b3RayAabb (raySource, rayTarget, bounds[0], bounds[1], param, normal);
			if (ra2 != ra)
			{
				printf("functions don't match\n");
			}
#endif
#ifdef RAYAABB2
			///careful with this check: need to check division by zero (above) and fix the unQuantize method
			///thanks Joerg/hiker for the reproduction case!
			///http://www.bulletphysics.com/Bullet/phpBB3/viewtopic.php?f=9&t=1858

			//B3_PROFILE("b3RayAabb2");
			rayBoxOverlap = b3RayAabb2(raySource, rayDirection, sign, bounds, param, 0.0f, lambda_max);

#else
			rayBoxOverlap = true;  //b3RayAabb(raySource, rayTarget, bounds[0], bounds[1], param, normal);
#endif
		}

		if (isLeafNode && rayBoxOverlap)
		{
			nodeCallback->processNode(rootNode->getPartId(), rootNode->getTriangleIndex());
		}

		//PCK: unsigned instead of bool
		if ((rayBoxOverlap != 0) || isLeafNode)
		{
			rootNode++;
			curIndex++;
		}
		else
		{
			escapeIndex = rootNode->getEscapeIndex();
			rootNode += escapeIndex;
			curIndex += escapeIndex;
		}
	}
	if (b3s_maxIterations < walkIterations)
		b3s_maxIterations = walkIterations;
}

void b3QuantizedBvh::walkStacklessQuantizedTree(b3NodeOverlapCallback* nodeCallback, unsigned short int* quantizedQueryAabbMin, unsigned short int* quantizedQueryAabbMax, int startNodeIndex, int endNodeIndex) const
{
	b3Assert(m_useQuantization);

	int curIndex = startNodeIndex;
	int walkIterations = 0;
	int subTreeSize = endNodeIndex - startNodeIndex;
	(void)subTreeSize;

	const b3QuantizedBvhNode* rootNode = &m_quantizedContiguousNodes[startNodeIndex];
	int escapeIndex;

	bool isLeafNode;
	//PCK: unsigned instead of bool
	unsigned aabbOverlap;

	while (curIndex < endNodeIndex)
	{
//#define VISUALLY_ANALYZE_BVH 1
#ifdef VISUALLY_ANALYZE_BVH
		//some code snippet to debugDraw aabb, to visually analyze bvh structure
		static int drawPatch = 0;
		//need some global access to a debugDrawer
		extern b3IDebugDraw* debugDrawerPtr;
		if (curIndex == drawPatch)
		{
			b3Vector3 aabbMin, aabbMax;
			aabbMin = unQuantize(rootNode->m_quantizedAabbMin);
			aabbMax = unQuantize(rootNode->m_quantizedAabbMax);
			b3Vector3 color(1, 0, 0);
			debugDrawerPtr->drawAabb(aabbMin, aabbMax, color);
		}
#endif  //VISUALLY_ANALYZE_BVH

		//catch bugs in tree data
		b3Assert(walkIterations < subTreeSize);

		walkIterations++;
		//PCK: unsigned instead of bool
		aabbOverlap = b3TestQuantizedAabbAgainstQuantizedAabb(quantizedQueryAabbMin, quantizedQueryAabbMax, rootNode->m_quantizedAabbMin, rootNode->m_quantizedAabbMax);
		isLeafNode = rootNode->isLeafNode();

		if (isLeafNode && aabbOverlap)
		{
			nodeCallback->processNode(rootNode->getPartId(), rootNode->getTriangleIndex());
		}

		//PCK: unsigned instead of bool
		if ((aabbOverlap != 0) || isLeafNode)
		{
			rootNode++;
			curIndex++;
		}
		else
		{
			escapeIndex = rootNode->getEscapeIndex();
			rootNode += escapeIndex;
			curIndex += escapeIndex;
		}
	}
	if (b3s_maxIterations < walkIterations)
		b3s_maxIterations = walkIterations;
}

//This traversal can be called from Playstation 3 SPU
void b3QuantizedBvh::walkStacklessQuantizedTreeCacheFriendly(b3NodeOverlapCallback* nodeCallback, unsigned short int* quantizedQueryAabbMin, unsigned short int* quantizedQueryAabbMax) const
{
	b3Assert(m_useQuantization);

	int i;

	for (i = 0; i < this->m_SubtreeHeaders.size(); i++)
	{
		const b3BvhSubtreeInfo& subtree = m_SubtreeHeaders[i];

		//PCK: unsigned instead of bool
		unsigned overlap = b3TestQuantizedAabbAgainstQuantizedAabb(quantizedQueryAabbMin, quantizedQueryAabbMax, subtree.m_quantizedAabbMin, subtree.m_quantizedAabbMax);
		if (overlap != 0)
		{
			walkStacklessQuantizedTree(nodeCallback, quantizedQueryAabbMin, quantizedQueryAabbMax,
									   subtree.m_rootNodeIndex,
									   subtree.m_rootNodeIndex + subtree.m_subtreeSize);
		}
	}
}

void b3QuantizedBvh::reportRayOverlappingNodex(b3NodeOverlapCallback* nodeCallback, const b3Vector3& raySource, const b3Vector3& rayTarget) const
{
	reportBoxCastOverlappingNodex(nodeCallback, raySource, rayTarget, b3MakeVector3(0, 0, 0), b3MakeVector3(0, 0, 0));
}

void b3QuantizedBvh::reportBoxCastOverlappingNodex(b3NodeOverlapCallback* nodeCallback, const b3Vector3& raySource, const b3Vector3& rayTarget, const b3Vector3& aabbMin, const b3Vector3& aabbMax) const
{
	//always use stackless

	if (m_useQuantization)
	{
		walkStacklessQuantizedTreeAgainstRay(nodeCallback, raySource, rayTarget, aabbMin, aabbMax, 0, m_curNodeIndex);
	}
	else
	{
		walkStacklessTreeAgainstRay(nodeCallback, raySource, rayTarget, aabbMin, aabbMax, 0, m_curNodeIndex);
	}
	/*
	{
		//recursive traversal
		b3Vector3 qaabbMin = raySource;
		b3Vector3 qaabbMax = raySource;
		qaabbMin.setMin(rayTarget);
		qaabbMax.setMax(rayTarget);
		qaabbMin += aabbMin;
		qaabbMax += aabbMax;
		reportAabbOverlappingNodex(nodeCallback,qaabbMin,qaabbMax);
	}
	*/
}

void b3QuantizedBvh::swapLeafNodes(int i, int splitIndex)
{
	if (m_useQuantization)
	{
		b3QuantizedBvhNode tmp = m_quantizedLeafNodes[i];
		m_quantizedLeafNodes[i] = m_quantizedLeafNodes[splitIndex];
		m_quantizedLeafNodes[splitIndex] = tmp;
	}
	else
	{
		b3OptimizedBvhNode tmp = m_leafNodes[i];
		m_leafNodes[i] = m_leafNodes[splitIndex];
		m_leafNodes[splitIndex] = tmp;
	}
}

void b3QuantizedBvh::assignInternalNodeFromLeafNode(int internalNode, int leafNodeIndex)
{
	if (m_useQuantization)
	{
		m_quantizedContiguousNodes[internalNode] = m_quantizedLeafNodes[leafNodeIndex];
	}
	else
	{
		m_contiguousNodes[internalNode] = m_leafNodes[leafNodeIndex];
	}
}

//PCK: include
#include <new>

#if 0
//PCK: consts
static const unsigned BVH_ALIGNMENT = 16;
static const unsigned BVH_ALIGNMENT_MASK = BVH_ALIGNMENT-1;

static const unsigned BVH_ALIGNMENT_BLOCKS = 2;
#endif

unsigned int b3QuantizedBvh::getAlignmentSerializationPadding()
{
	// I changed this to 0 since the extra padding is not needed or used.
	return 0;  //BVH_ALIGNMENT_BLOCKS * BVH_ALIGNMENT;
}

unsigned b3QuantizedBvh::calculateSerializeBufferSize() const
{
	unsigned baseSize = sizeof(b3QuantizedBvh) + getAlignmentSerializationPadding();
	baseSize += sizeof(b3BvhSubtreeInfo) * m_subtreeHeaderCount;
	if (m_useQuantization)
	{
		return baseSize + m_curNodeIndex * sizeof(b3QuantizedBvhNode);
	}
	return baseSize + m_curNodeIndex * sizeof(b3OptimizedBvhNode);
}

bool b3QuantizedBvh::serialize(void* o_alignedDataBuffer, unsigned /*i_dataBufferSize */, bool i_swapEndian) const
{
	b3Assert(m_subtreeHeaderCount == m_SubtreeHeaders.size());
	m_subtreeHeaderCount = m_SubtreeHeaders.size();

	/*	if (i_dataBufferSize < calculateSerializeBufferSize() || o_alignedDataBuffer == NULL || (((unsigned)o_alignedDataBuffer & BVH_ALIGNMENT_MASK) != 0))
	{
		///check alignedment for buffer?
		b3Assert(0);
		return false;
	}
*/

	b3QuantizedBvh* targetBvh = (b3QuantizedBvh*)o_alignedDataBuffer;

	// construct the class so the virtual function table, etc will be set up
	// Also, m_leafNodes and m_quantizedLeafNodes will be initialized to default values by the constructor
	new (targetBvh) b3QuantizedBvh;

	if (i_swapEndian)
	{
		targetBvh->m_curNodeIndex = static_cast<int>(b3SwapEndian(m_curNodeIndex));

		b3SwapVector3Endian(m_bvhAabbMin, targetBvh->m_bvhAabbMin);
		b3SwapVector3Endian(m_bvhAabbMax, targetBvh->m_bvhAabbMax);
		b3SwapVector3Endian(m_bvhQuantization, targetBvh->m_bvhQuantization);

		targetBvh->m_traversalMode = (b3TraversalMode)b3SwapEndian(m_traversalMode);
		targetBvh->m_subtreeHeaderCount = static_cast<int>(b3SwapEndian(m_subtreeHeaderCount));
	}
	else
	{
		targetBvh->m_curNodeIndex = m_curNodeIndex;
		targetBvh->m_bvhAabbMin = m_bvhAabbMin;
		targetBvh->m_bvhAabbMax = m_bvhAabbMax;
		targetBvh->m_bvhQuantization = m_bvhQuantization;
		targetBvh->m_traversalMode = m_traversalMode;
		targetBvh->m_subtreeHeaderCount = m_subtreeHeaderCount;
	}

	targetBvh->m_useQuantization = m_useQuantization;

	unsigned char* nodeData = (unsigned char*)targetBvh;
	nodeData += sizeof(b3QuantizedBvh);

	unsigned sizeToAdd = 0;  //(BVH_ALIGNMENT-((unsigned)nodeData & BVH_ALIGNMENT_MASK))&BVH_ALIGNMENT_MASK;
	nodeData += sizeToAdd;

	int nodeCount = m_curNodeIndex;

	if (m_useQuantization)
	{
		targetBvh->m_quantizedContiguousNodes.initializeFromBuffer(nodeData, nodeCount, nodeCount);

		if (i_swapEndian)
		{
			for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
			{
				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[0] = b3SwapEndian(m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[0]);
				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[1] = b3SwapEndian(m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[1]);
				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[2] = b3SwapEndian(m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[2]);

				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[0] = b3SwapEndian(m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[0]);
				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[1] = b3SwapEndian(m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[1]);
				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[2] = b3SwapEndian(m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[2]);

				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_escapeIndexOrTriangleIndex = static_cast<int>(b3SwapEndian(m_quantizedContiguousNodes[nodeIndex].m_escapeIndexOrTriangleIndex));
			}
		}
		else
		{
			for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
			{
				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[0] = m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[0];
				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[1] = m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[1];
				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[2] = m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[2];

				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[0] = m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[0];
				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[1] = m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[1];
				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[2] = m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[2];

				targetBvh->m_quantizedContiguousNodes[nodeIndex].m_escapeIndexOrTriangleIndex = m_quantizedContiguousNodes[nodeIndex].m_escapeIndexOrTriangleIndex;
			}
		}
		nodeData += sizeof(b3QuantizedBvhNode) * nodeCount;

		// this clears the pointer in the member variable it doesn't really do anything to the data
		// it does call the destructor on the contained objects, but they are all classes with no destructor defined
		// so the memory (which is not freed) is left alone
		targetBvh->m_quantizedContiguousNodes.initializeFromBuffer(NULL, 0, 0);
	}
	else
	{
		targetBvh->m_contiguousNodes.initializeFromBuffer(nodeData, nodeCount, nodeCount);

		if (i_swapEndian)
		{
			for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
			{
				b3SwapVector3Endian(m_contiguousNodes[nodeIndex].m_aabbMinOrg, targetBvh->m_contiguousNodes[nodeIndex].m_aabbMinOrg);
				b3SwapVector3Endian(m_contiguousNodes[nodeIndex].m_aabbMaxOrg, targetBvh->m_contiguousNodes[nodeIndex].m_aabbMaxOrg);

				targetBvh->m_contiguousNodes[nodeIndex].m_escapeIndex = static_cast<int>(b3SwapEndian(m_contiguousNodes[nodeIndex].m_escapeIndex));
				targetBvh->m_contiguousNodes[nodeIndex].m_subPart = static_cast<int>(b3SwapEndian(m_contiguousNodes[nodeIndex].m_subPart));
				targetBvh->m_contiguousNodes[nodeIndex].m_triangleIndex = static_cast<int>(b3SwapEndian(m_contiguousNodes[nodeIndex].m_triangleIndex));
			}
		}
		else
		{
			for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
			{
				targetBvh->m_contiguousNodes[nodeIndex].m_aabbMinOrg = m_contiguousNodes[nodeIndex].m_aabbMinOrg;
				targetBvh->m_contiguousNodes[nodeIndex].m_aabbMaxOrg = m_contiguousNodes[nodeIndex].m_aabbMaxOrg;

				targetBvh->m_contiguousNodes[nodeIndex].m_escapeIndex = m_contiguousNodes[nodeIndex].m_escapeIndex;
				targetBvh->m_contiguousNodes[nodeIndex].m_subPart = m_contiguousNodes[nodeIndex].m_subPart;
				targetBvh->m_contiguousNodes[nodeIndex].m_triangleIndex = m_contiguousNodes[nodeIndex].m_triangleIndex;
			}
		}
		nodeData += sizeof(b3OptimizedBvhNode) * nodeCount;

		// this clears the pointer in the member variable it doesn't really do anything to the data
		// it does call the destructor on the contained objects, but they are all classes with no destructor defined
		// so the memory (which is not freed) is left alone
		targetBvh->m_contiguousNodes.initializeFromBuffer(NULL, 0, 0);
	}

	sizeToAdd = 0;  //(BVH_ALIGNMENT-((unsigned)nodeData & BVH_ALIGNMENT_MASK))&BVH_ALIGNMENT_MASK;
	nodeData += sizeToAdd;

	// Now serialize the subtree headers
	targetBvh->m_SubtreeHeaders.initializeFromBuffer(nodeData, m_subtreeHeaderCount, m_subtreeHeaderCount);
	if (i_swapEndian)
	{
		for (int i = 0; i < m_subtreeHeaderCount; i++)
		{
			targetBvh->m_SubtreeHeaders[i].m_quantizedAabbMin[0] = b3SwapEndian(m_SubtreeHeaders[i].m_quantizedAabbMin[0]);
			targetBvh->m_SubtreeHeaders[i].m_quantizedAabbMin[1] = b3SwapEndian(m_SubtreeHeaders[i].m_quantizedAabbMin[1]);
			targetBvh->m_SubtreeHeaders[i].m_quantizedAabbMin[2] = b3SwapEndian(m_SubtreeHeaders[i].m_quantizedAabbMin[2]);

			targetBvh->m_SubtreeHeaders[i].m_quantizedAabbMax[0] = b3SwapEndian(m_SubtreeHeaders[i].m_quantizedAabbMax[0]);
			targetBvh->m_SubtreeHeaders[i].m_quantizedAabbMax[1] = b3SwapEndian(m_SubtreeHeaders[i].m_quantizedAabbMax[1]);
			targetBvh->m_SubtreeHeaders[i].m_quantizedAabbMax[2] = b3SwapEndian(m_SubtreeHeaders[i].m_quantizedAabbMax[2]);

			targetBvh->m_SubtreeHeaders[i].m_rootNodeIndex = static_cast<int>(b3SwapEndian(m_SubtreeHeaders[i].m_rootNodeIndex));
			targetBvh->m_SubtreeHeaders[i].m_subtreeSize = static_cast<int>(b3SwapEndian(m_SubtreeHeaders[i].m_subtreeSize));
		}
	}
	else
	{
		for (int i = 0; i < m_subtreeHeaderCount; i++)
		{
			targetBvh->m_SubtreeHeaders[i].m_quantizedAabbMin[0] = (m_SubtreeHeaders[i].m_quantizedAabbMin[0]);
			targetBvh->m_SubtreeHeaders[i].m_quantizedAabbMin[1] = (m_SubtreeHeaders[i].m_quantizedAabbMin[1]);
			targetBvh->m_SubtreeHeaders[i].m_quantizedAabbMin[2] = (m_SubtreeHeaders[i].m_quantizedAabbMin[2]);

			targetBvh->m_SubtreeHeaders[i].m_quantizedAabbMax[0] = (m_SubtreeHeaders[i].m_quantizedAabbMax[0]);
			targetBvh->m_SubtreeHeaders[i].m_quantizedAabbMax[1] = (m_SubtreeHeaders[i].m_quantizedAabbMax[1]);
			targetBvh->m_SubtreeHeaders[i].m_quantizedAabbMax[2] = (m_SubtreeHeaders[i].m_quantizedAabbMax[2]);

			targetBvh->m_SubtreeHeaders[i].m_rootNodeIndex = (m_SubtreeHeaders[i].m_rootNodeIndex);
			targetBvh->m_SubtreeHeaders[i].m_subtreeSize = (m_SubtreeHeaders[i].m_subtreeSize);

			// need to clear padding in destination buffer
			targetBvh->m_SubtreeHeaders[i].m_padding[0] = 0;
			targetBvh->m_SubtreeHeaders[i].m_padding[1] = 0;
			targetBvh->m_SubtreeHeaders[i].m_padding[2] = 0;
		}
	}
	nodeData += sizeof(b3BvhSubtreeInfo) * m_subtreeHeaderCount;

	// this clears the pointer in the member variable it doesn't really do anything to the data
	// it does call the destructor on the contained objects, but they are all classes with no destructor defined
	// so the memory (which is not freed) is left alone
	targetBvh->m_SubtreeHeaders.initializeFromBuffer(NULL, 0, 0);

	// this wipes the virtual function table pointer at the start of the buffer for the class
	*((void**)o_alignedDataBuffer) = NULL;

	return true;
}

b3QuantizedBvh* b3QuantizedBvh::deSerializeInPlace(void* i_alignedDataBuffer, unsigned int i_dataBufferSize, bool i_swapEndian)
{
	if (i_alignedDataBuffer == NULL)  // || (((unsigned)i_alignedDataBuffer & BVH_ALIGNMENT_MASK) != 0))
	{
		return NULL;
	}
	b3QuantizedBvh* bvh = (b3QuantizedBvh*)i_alignedDataBuffer;

	if (i_swapEndian)
	{
		bvh->m_curNodeIndex = static_cast<int>(b3SwapEndian(bvh->m_curNodeIndex));

		b3UnSwapVector3Endian(bvh->m_bvhAabbMin);
		b3UnSwapVector3Endian(bvh->m_bvhAabbMax);
		b3UnSwapVector3Endian(bvh->m_bvhQuantization);

		bvh->m_traversalMode = (b3TraversalMode)b3SwapEndian(bvh->m_traversalMode);
		bvh->m_subtreeHeaderCount = static_cast<int>(b3SwapEndian(bvh->m_subtreeHeaderCount));
	}

	unsigned int calculatedBufSize = bvh->calculateSerializeBufferSize();
	b3Assert(calculatedBufSize <= i_dataBufferSize);

	if (calculatedBufSize > i_dataBufferSize)
	{
		return NULL;
	}

	unsigned char* nodeData = (unsigned char*)bvh;
	nodeData += sizeof(b3QuantizedBvh);

	unsigned sizeToAdd = 0;  //(BVH_ALIGNMENT-((unsigned)nodeData & BVH_ALIGNMENT_MASK))&BVH_ALIGNMENT_MASK;
	nodeData += sizeToAdd;

	int nodeCount = bvh->m_curNodeIndex;

	// Must call placement new to fill in virtual function table, etc, but we don't want to overwrite most data, so call a special version of the constructor
	// Also, m_leafNodes and m_quantizedLeafNodes will be initialized to default values by the constructor
	new (bvh) b3QuantizedBvh(*bvh, false);

	if (bvh->m_useQuantization)
	{
		bvh->m_quantizedContiguousNodes.initializeFromBuffer(nodeData, nodeCount, nodeCount);

		if (i_swapEndian)
		{
			for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
			{
				bvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[0] = b3SwapEndian(bvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[0]);
				bvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[1] = b3SwapEndian(bvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[1]);
				bvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[2] = b3SwapEndian(bvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[2]);

				bvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[0] = b3SwapEndian(bvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[0]);
				bvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[1] = b3SwapEndian(bvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[1]);
				bvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[2] = b3SwapEndian(bvh->m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[2]);

				bvh->m_quantizedContiguousNodes[nodeIndex].m_escapeIndexOrTriangleIndex = static_cast<int>(b3SwapEndian(bvh->m_quantizedContiguousNodes[nodeIndex].m_escapeIndexOrTriangleIndex));
			}
		}
		nodeData += sizeof(b3QuantizedBvhNode) * nodeCount;
	}
	else
	{
		bvh->m_contiguousNodes.initializeFromBuffer(nodeData, nodeCount, nodeCount);

		if (i_swapEndian)
		{
			for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
			{
				b3UnSwapVector3Endian(bvh->m_contiguousNodes[nodeIndex].m_aabbMinOrg);
				b3UnSwapVector3Endian(bvh->m_contiguousNodes[nodeIndex].m_aabbMaxOrg);

				bvh->m_contiguousNodes[nodeIndex].m_escapeIndex = static_cast<int>(b3SwapEndian(bvh->m_contiguousNodes[nodeIndex].m_escapeIndex));
				bvh->m_contiguousNodes[nodeIndex].m_subPart = static_cast<int>(b3SwapEndian(bvh->m_contiguousNodes[nodeIndex].m_subPart));
				bvh->m_contiguousNodes[nodeIndex].m_triangleIndex = static_cast<int>(b3SwapEndian(bvh->m_contiguousNodes[nodeIndex].m_triangleIndex));
			}
		}
		nodeData += sizeof(b3OptimizedBvhNode) * nodeCount;
	}

	sizeToAdd = 0;  //(BVH_ALIGNMENT-((unsigned)nodeData & BVH_ALIGNMENT_MASK))&BVH_ALIGNMENT_MASK;
	nodeData += sizeToAdd;

	// Now serialize the subtree headers
	bvh->m_SubtreeHeaders.initializeFromBuffer(nodeData, bvh->m_subtreeHeaderCount, bvh->m_subtreeHeaderCount);
	if (i_swapEndian)
	{
		for (int i = 0; i < bvh->m_subtreeHeaderCount; i++)
		{
			bvh->m_SubtreeHeaders[i].m_quantizedAabbMin[0] = b3SwapEndian(bvh->m_SubtreeHeaders[i].m_quantizedAabbMin[0]);
			bvh->m_SubtreeHeaders[i].m_quantizedAabbMin[1] = b3SwapEndian(bvh->m_SubtreeHeaders[i].m_quantizedAabbMin[1]);
			bvh->m_SubtreeHeaders[i].m_quantizedAabbMin[2] = b3SwapEndian(bvh->m_SubtreeHeaders[i].m_quantizedAabbMin[2]);

			bvh->m_SubtreeHeaders[i].m_quantizedAabbMax[0] = b3SwapEndian(bvh->m_SubtreeHeaders[i].m_quantizedAabbMax[0]);
			bvh->m_SubtreeHeaders[i].m_quantizedAabbMax[1] = b3SwapEndian(bvh->m_SubtreeHeaders[i].m_quantizedAabbMax[1]);
			bvh->m_SubtreeHeaders[i].m_quantizedAabbMax[2] = b3SwapEndian(bvh->m_SubtreeHeaders[i].m_quantizedAabbMax[2]);

			bvh->m_SubtreeHeaders[i].m_rootNodeIndex = static_cast<int>(b3SwapEndian(bvh->m_SubtreeHeaders[i].m_rootNodeIndex));
			bvh->m_SubtreeHeaders[i].m_subtreeSize = static_cast<int>(b3SwapEndian(bvh->m_SubtreeHeaders[i].m_subtreeSize));
		}
	}

	return bvh;
}

// Constructor that prevents b3Vector3's default constructor from being called
b3QuantizedBvh::b3QuantizedBvh(b3QuantizedBvh& self, bool /* ownsMemory */) : m_bvhAabbMin(self.m_bvhAabbMin),
																			  m_bvhAabbMax(self.m_bvhAabbMax),
																			  m_bvhQuantization(self.m_bvhQuantization),
																			  m_bulletVersion(B3_BULLET_VERSION)
{
}

void b3QuantizedBvh::deSerializeFloat(struct b3QuantizedBvhFloatData& quantizedBvhFloatData)
{
	m_bvhAabbMax.deSerializeFloat(quantizedBvhFloatData.m_bvhAabbMax);
	m_bvhAabbMin.deSerializeFloat(quantizedBvhFloatData.m_bvhAabbMin);
	m_bvhQuantization.deSerializeFloat(quantizedBvhFloatData.m_bvhQuantization);

	m_curNodeIndex = quantizedBvhFloatData.m_curNodeIndex;
	m_useQuantization = quantizedBvhFloatData.m_useQuantization != 0;

	{
		int numElem = quantizedBvhFloatData.m_numContiguousLeafNodes;
		m_contiguousNodes.resize(numElem);

		if (numElem)
		{
			b3OptimizedBvhNodeFloatData* memPtr = quantizedBvhFloatData.m_contiguousNodesPtr;

			for (int i = 0; i < numElem; i++, memPtr++)
			{
				m_contiguousNodes[i].m_aabbMaxOrg.deSerializeFloat(memPtr->m_aabbMaxOrg);
				m_contiguousNodes[i].m_aabbMinOrg.deSerializeFloat(memPtr->m_aabbMinOrg);
				m_contiguousNodes[i].m_escapeIndex = memPtr->m_escapeIndex;
				m_contiguousNodes[i].m_subPart = memPtr->m_subPart;
				m_contiguousNodes[i].m_triangleIndex = memPtr->m_triangleIndex;
			}
		}
	}

	{
		int numElem = quantizedBvhFloatData.m_numQuantizedContiguousNodes;
		m_quantizedContiguousNodes.resize(numElem);

		if (numElem)
		{
			b3QuantizedBvhNodeData* memPtr = quantizedBvhFloatData.m_quantizedContiguousNodesPtr;
			for (int i = 0; i < numElem; i++, memPtr++)
			{
				m_quantizedContiguousNodes[i].m_escapeIndexOrTriangleIndex = memPtr->m_escapeIndexOrTriangleIndex;
				m_quantizedContiguousNodes[i].m_quantizedAabbMax[0] = memPtr->m_quantizedAabbMax[0];
				m_quantizedContiguousNodes[i].m_quantizedAabbMax[1] = memPtr->m_quantizedAabbMax[1];
				m_quantizedContiguousNodes[i].m_quantizedAabbMax[2] = memPtr->m_quantizedAabbMax[2];
				m_quantizedContiguousNodes[i].m_quantizedAabbMin[0] = memPtr->m_quantizedAabbMin[0];
				m_quantizedContiguousNodes[i].m_quantizedAabbMin[1] = memPtr->m_quantizedAabbMin[1];
				m_quantizedContiguousNodes[i].m_quantizedAabbMin[2] = memPtr->m_quantizedAabbMin[2];
			}
		}
	}

	m_traversalMode = b3TraversalMode(quantizedBvhFloatData.m_traversalMode);

	{
		int numElem = quantizedBvhFloatData.m_numSubtreeHeaders;
		m_SubtreeHeaders.resize(numElem);
		if (numElem)
		{
			b3BvhSubtreeInfoData* memPtr = quantizedBvhFloatData.m_subTreeInfoPtr;
			for (int i = 0; i < numElem; i++, memPtr++)
			{
				m_SubtreeHeaders[i].m_quantizedAabbMax[0] = memPtr->m_quantizedAabbMax[0];
				m_SubtreeHeaders[i].m_quantizedAabbMax[1] = memPtr->m_quantizedAabbMax[1];
				m_SubtreeHeaders[i].m_quantizedAabbMax[2] = memPtr->m_quantizedAabbMax[2];
				m_SubtreeHeaders[i].m_quantizedAabbMin[0] = memPtr->m_quantizedAabbMin[0];
				m_SubtreeHeaders[i].m_quantizedAabbMin[1] = memPtr->m_quantizedAabbMin[1];
				m_SubtreeHeaders[i].m_quantizedAabbMin[2] = memPtr->m_quantizedAabbMin[2];
				m_SubtreeHeaders[i].m_rootNodeIndex = memPtr->m_rootNodeIndex;
				m_SubtreeHeaders[i].m_subtreeSize = memPtr->m_subtreeSize;
			}
		}
	}
}

void b3QuantizedBvh::deSerializeDouble(struct b3QuantizedBvhDoubleData& quantizedBvhDoubleData)
{
	m_bvhAabbMax.deSerializeDouble(quantizedBvhDoubleData.m_bvhAabbMax);
	m_bvhAabbMin.deSerializeDouble(quantizedBvhDoubleData.m_bvhAabbMin);
	m_bvhQuantization.deSerializeDouble(quantizedBvhDoubleData.m_bvhQuantization);

	m_curNodeIndex = quantizedBvhDoubleData.m_curNodeIndex;
	m_useQuantization = quantizedBvhDoubleData.m_useQuantization != 0;

	{
		int numElem = quantizedBvhDoubleData.m_numContiguousLeafNodes;
		m_contiguousNodes.resize(numElem);

		if (numElem)
		{
			b3OptimizedBvhNodeDoubleData* memPtr = quantizedBvhDoubleData.m_contiguousNodesPtr;

			for (int i = 0; i < numElem; i++, memPtr++)
			{
				m_contiguousNodes[i].m_aabbMaxOrg.deSerializeDouble(memPtr->m_aabbMaxOrg);
				m_contiguousNodes[i].m_aabbMinOrg.deSerializeDouble(memPtr->m_aabbMinOrg);
				m_contiguousNodes[i].m_escapeIndex = memPtr->m_escapeIndex;
				m_contiguousNodes[i].m_subPart = memPtr->m_subPart;
				m_contiguousNodes[i].m_triangleIndex = memPtr->m_triangleIndex;
			}
		}
	}

	{
		int numElem = quantizedBvhDoubleData.m_numQuantizedContiguousNodes;
		m_quantizedContiguousNodes.resize(numElem);

		if (numElem)
		{
			b3QuantizedBvhNodeData* memPtr = quantizedBvhDoubleData.m_quantizedContiguousNodesPtr;
			for (int i = 0; i < numElem; i++, memPtr++)
			{
				m_quantizedContiguousNodes[i].m_escapeIndexOrTriangleIndex = memPtr->m_escapeIndexOrTriangleIndex;
				m_quantizedContiguousNodes[i].m_quantizedAabbMax[0] = memPtr->m_quantizedAabbMax[0];
				m_quantizedContiguousNodes[i].m_quantizedAabbMax[1] = memPtr->m_quantizedAabbMax[1];
				m_quantizedContiguousNodes[i].m_quantizedAabbMax[2] = memPtr->m_quantizedAabbMax[2];
				m_quantizedContiguousNodes[i].m_quantizedAabbMin[0] = memPtr->m_quantizedAabbMin[0];
				m_quantizedContiguousNodes[i].m_quantizedAabbMin[1] = memPtr->m_quantizedAabbMin[1];
				m_quantizedContiguousNodes[i].m_quantizedAabbMin[2] = memPtr->m_quantizedAabbMin[2];
			}
		}
	}

	m_traversalMode = b3TraversalMode(quantizedBvhDoubleData.m_traversalMode);

	{
		int numElem = quantizedBvhDoubleData.m_numSubtreeHeaders;
		m_SubtreeHeaders.resize(numElem);
		if (numElem)
		{
			b3BvhSubtreeInfoData* memPtr = quantizedBvhDoubleData.m_subTreeInfoPtr;
			for (int i = 0; i < numElem; i++, memPtr++)
			{
				m_SubtreeHeaders[i].m_quantizedAabbMax[0] = memPtr->m_quantizedAabbMax[0];
				m_SubtreeHeaders[i].m_quantizedAabbMax[1] = memPtr->m_quantizedAabbMax[1];
				m_SubtreeHeaders[i].m_quantizedAabbMax[2] = memPtr->m_quantizedAabbMax[2];
				m_SubtreeHeaders[i].m_quantizedAabbMin[0] = memPtr->m_quantizedAabbMin[0];
				m_SubtreeHeaders[i].m_quantizedAabbMin[1] = memPtr->m_quantizedAabbMin[1];
				m_SubtreeHeaders[i].m_quantizedAabbMin[2] = memPtr->m_quantizedAabbMin[2];
				m_SubtreeHeaders[i].m_rootNodeIndex = memPtr->m_rootNodeIndex;
				m_SubtreeHeaders[i].m_subtreeSize = memPtr->m_subtreeSize;
			}
		}
	}
}

///fills the dataBuffer and returns the struct name (and 0 on failure)
const char* b3QuantizedBvh::serialize(void* dataBuffer, b3Serializer* serializer) const
{
	b3Assert(0);
	return 0;
}
