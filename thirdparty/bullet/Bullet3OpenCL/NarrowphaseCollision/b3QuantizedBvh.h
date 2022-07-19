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

#ifndef B3_QUANTIZED_BVH_H
#define B3_QUANTIZED_BVH_H

class b3Serializer;

//#define DEBUG_CHECK_DEQUANTIZATION 1
#ifdef DEBUG_CHECK_DEQUANTIZATION
#ifdef __SPU__
#define printf spu_printf
#endif  //__SPU__

#include <stdio.h>
#include <stdlib.h>
#endif  //DEBUG_CHECK_DEQUANTIZATION

#include "Bullet3Common/b3Vector3.h"
#include "Bullet3Common/b3AlignedAllocator.h"

#ifdef B3_USE_DOUBLE_PRECISION
#define b3QuantizedBvhData b3QuantizedBvhDoubleData
#define b3OptimizedBvhNodeData b3OptimizedBvhNodeDoubleData
#define b3QuantizedBvhDataName "b3QuantizedBvhDoubleData"
#else
#define b3QuantizedBvhData b3QuantizedBvhFloatData
#define b3OptimizedBvhNodeData b3OptimizedBvhNodeFloatData
#define b3QuantizedBvhDataName "b3QuantizedBvhFloatData"
#endif

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3QuantizedBvhNodeData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3BvhSubtreeInfoData.h"

//http://msdn.microsoft.com/library/default.asp?url=/library/en-us/vclang/html/vclrf__m128.asp

//Note: currently we have 16 bytes per quantized node
#define MAX_SUBTREE_SIZE_IN_BYTES 2048

// 10 gives the potential for 1024 parts, with at most 2^21 (2097152) (minus one
// actually) triangles each (since the sign bit is reserved
#define MAX_NUM_PARTS_IN_BITS 10

///b3QuantizedBvhNode is a compressed aabb node, 16 bytes.
///Node can be used for leafnode or internal node. Leafnodes can point to 32-bit triangle index (non-negative range).
B3_ATTRIBUTE_ALIGNED16(struct)
b3QuantizedBvhNode : public b3QuantizedBvhNodeData
{
	B3_DECLARE_ALIGNED_ALLOCATOR();

	bool isLeafNode() const
	{
		//skipindex is negative (internal node), triangleindex >=0 (leafnode)
		return (m_escapeIndexOrTriangleIndex >= 0);
	}
	int getEscapeIndex() const
	{
		b3Assert(!isLeafNode());
		return -m_escapeIndexOrTriangleIndex;
	}
	int getTriangleIndex() const
	{
		b3Assert(isLeafNode());
		unsigned int x = 0;
		unsigned int y = (~(x & 0)) << (31 - MAX_NUM_PARTS_IN_BITS);
		// Get only the lower bits where the triangle index is stored
		return (m_escapeIndexOrTriangleIndex & ~(y));
	}
	int getPartId() const
	{
		b3Assert(isLeafNode());
		// Get only the highest bits where the part index is stored
		return (m_escapeIndexOrTriangleIndex >> (31 - MAX_NUM_PARTS_IN_BITS));
	}
};

/// b3OptimizedBvhNode contains both internal and leaf node information.
/// Total node size is 44 bytes / node. You can use the compressed version of 16 bytes.
B3_ATTRIBUTE_ALIGNED16(struct)
b3OptimizedBvhNode
{
	B3_DECLARE_ALIGNED_ALLOCATOR();

	//32 bytes
	b3Vector3 m_aabbMinOrg;
	b3Vector3 m_aabbMaxOrg;

	//4
	int m_escapeIndex;

	//8
	//for child nodes
	int m_subPart;
	int m_triangleIndex;

	//pad the size to 64 bytes
	char m_padding[20];
};

///b3BvhSubtreeInfo provides info to gather a subtree of limited size
B3_ATTRIBUTE_ALIGNED16(class)
b3BvhSubtreeInfo : public b3BvhSubtreeInfoData
{
public:
	B3_DECLARE_ALIGNED_ALLOCATOR();

	b3BvhSubtreeInfo()
	{
		//memset(&m_padding[0], 0, sizeof(m_padding));
	}

	void setAabbFromQuantizeNode(const b3QuantizedBvhNode& quantizedNode)
	{
		m_quantizedAabbMin[0] = quantizedNode.m_quantizedAabbMin[0];
		m_quantizedAabbMin[1] = quantizedNode.m_quantizedAabbMin[1];
		m_quantizedAabbMin[2] = quantizedNode.m_quantizedAabbMin[2];
		m_quantizedAabbMax[0] = quantizedNode.m_quantizedAabbMax[0];
		m_quantizedAabbMax[1] = quantizedNode.m_quantizedAabbMax[1];
		m_quantizedAabbMax[2] = quantizedNode.m_quantizedAabbMax[2];
	}
};

class b3NodeOverlapCallback
{
public:
	virtual ~b3NodeOverlapCallback(){};

	virtual void processNode(int subPart, int triangleIndex) = 0;
};

#include "Bullet3Common/b3AlignedAllocator.h"
#include "Bullet3Common/b3AlignedObjectArray.h"

///for code readability:
typedef b3AlignedObjectArray<b3OptimizedBvhNode> NodeArray;
typedef b3AlignedObjectArray<b3QuantizedBvhNode> QuantizedNodeArray;
typedef b3AlignedObjectArray<b3BvhSubtreeInfo> BvhSubtreeInfoArray;

///The b3QuantizedBvh class stores an AABB tree that can be quickly traversed on CPU and Cell SPU.
///It is used by the b3BvhTriangleMeshShape as midphase
///It is recommended to use quantization for better performance and lower memory requirements.
B3_ATTRIBUTE_ALIGNED16(class)
b3QuantizedBvh
{
public:
	enum b3TraversalMode
	{
		TRAVERSAL_STACKLESS = 0,
		TRAVERSAL_STACKLESS_CACHE_FRIENDLY,
		TRAVERSAL_RECURSIVE
	};

	b3Vector3 m_bvhAabbMin;
	b3Vector3 m_bvhAabbMax;
	b3Vector3 m_bvhQuantization;

protected:
	int m_bulletVersion;  //for serialization versioning. It could also be used to detect endianess.

	int m_curNodeIndex;
	//quantization data
	bool m_useQuantization;

	NodeArray m_leafNodes;
	NodeArray m_contiguousNodes;
	QuantizedNodeArray m_quantizedLeafNodes;
	QuantizedNodeArray m_quantizedContiguousNodes;

	b3TraversalMode m_traversalMode;
	BvhSubtreeInfoArray m_SubtreeHeaders;

	//This is only used for serialization so we don't have to add serialization directly to b3AlignedObjectArray
	mutable int m_subtreeHeaderCount;

	///two versions, one for quantized and normal nodes. This allows code-reuse while maintaining readability (no template/macro!)
	///this might be refactored into a virtual, it is usually not calculated at run-time
	void setInternalNodeAabbMin(int nodeIndex, const b3Vector3& aabbMin)
	{
		if (m_useQuantization)
		{
			quantize(&m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[0], aabbMin, 0);
		}
		else
		{
			m_contiguousNodes[nodeIndex].m_aabbMinOrg = aabbMin;
		}
	}
	void setInternalNodeAabbMax(int nodeIndex, const b3Vector3& aabbMax)
	{
		if (m_useQuantization)
		{
			quantize(&m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[0], aabbMax, 1);
		}
		else
		{
			m_contiguousNodes[nodeIndex].m_aabbMaxOrg = aabbMax;
		}
	}

	b3Vector3 getAabbMin(int nodeIndex) const
	{
		if (m_useQuantization)
		{
			return unQuantize(&m_quantizedLeafNodes[nodeIndex].m_quantizedAabbMin[0]);
		}
		//non-quantized
		return m_leafNodes[nodeIndex].m_aabbMinOrg;
	}
	b3Vector3 getAabbMax(int nodeIndex) const
	{
		if (m_useQuantization)
		{
			return unQuantize(&m_quantizedLeafNodes[nodeIndex].m_quantizedAabbMax[0]);
		}
		//non-quantized
		return m_leafNodes[nodeIndex].m_aabbMaxOrg;
	}

	void setInternalNodeEscapeIndex(int nodeIndex, int escapeIndex)
	{
		if (m_useQuantization)
		{
			m_quantizedContiguousNodes[nodeIndex].m_escapeIndexOrTriangleIndex = -escapeIndex;
		}
		else
		{
			m_contiguousNodes[nodeIndex].m_escapeIndex = escapeIndex;
		}
	}

	void mergeInternalNodeAabb(int nodeIndex, const b3Vector3& newAabbMin, const b3Vector3& newAabbMax)
	{
		if (m_useQuantization)
		{
			unsigned short int quantizedAabbMin[3];
			unsigned short int quantizedAabbMax[3];
			quantize(quantizedAabbMin, newAabbMin, 0);
			quantize(quantizedAabbMax, newAabbMax, 1);
			for (int i = 0; i < 3; i++)
			{
				if (m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[i] > quantizedAabbMin[i])
					m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMin[i] = quantizedAabbMin[i];

				if (m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[i] < quantizedAabbMax[i])
					m_quantizedContiguousNodes[nodeIndex].m_quantizedAabbMax[i] = quantizedAabbMax[i];
			}
		}
		else
		{
			//non-quantized
			m_contiguousNodes[nodeIndex].m_aabbMinOrg.setMin(newAabbMin);
			m_contiguousNodes[nodeIndex].m_aabbMaxOrg.setMax(newAabbMax);
		}
	}

	void swapLeafNodes(int firstIndex, int secondIndex);

	void assignInternalNodeFromLeafNode(int internalNode, int leafNodeIndex);

protected:
	void buildTree(int startIndex, int endIndex);

	int calcSplittingAxis(int startIndex, int endIndex);

	int sortAndCalcSplittingIndex(int startIndex, int endIndex, int splitAxis);

	void walkStacklessTree(b3NodeOverlapCallback * nodeCallback, const b3Vector3& aabbMin, const b3Vector3& aabbMax) const;

	void walkStacklessQuantizedTreeAgainstRay(b3NodeOverlapCallback * nodeCallback, const b3Vector3& raySource, const b3Vector3& rayTarget, const b3Vector3& aabbMin, const b3Vector3& aabbMax, int startNodeIndex, int endNodeIndex) const;
	void walkStacklessQuantizedTree(b3NodeOverlapCallback * nodeCallback, unsigned short int* quantizedQueryAabbMin, unsigned short int* quantizedQueryAabbMax, int startNodeIndex, int endNodeIndex) const;
	void walkStacklessTreeAgainstRay(b3NodeOverlapCallback * nodeCallback, const b3Vector3& raySource, const b3Vector3& rayTarget, const b3Vector3& aabbMin, const b3Vector3& aabbMax, int startNodeIndex, int endNodeIndex) const;

	///tree traversal designed for small-memory processors like PS3 SPU
	void walkStacklessQuantizedTreeCacheFriendly(b3NodeOverlapCallback * nodeCallback, unsigned short int* quantizedQueryAabbMin, unsigned short int* quantizedQueryAabbMax) const;

	///use the 16-byte stackless 'skipindex' node tree to do a recursive traversal
	void walkRecursiveQuantizedTreeAgainstQueryAabb(const b3QuantizedBvhNode* currentNode, b3NodeOverlapCallback* nodeCallback, unsigned short int* quantizedQueryAabbMin, unsigned short int* quantizedQueryAabbMax) const;

	///use the 16-byte stackless 'skipindex' node tree to do a recursive traversal
	void walkRecursiveQuantizedTreeAgainstQuantizedTree(const b3QuantizedBvhNode* treeNodeA, const b3QuantizedBvhNode* treeNodeB, b3NodeOverlapCallback* nodeCallback) const;

	void updateSubtreeHeaders(int leftChildNodexIndex, int rightChildNodexIndex);

public:
	B3_DECLARE_ALIGNED_ALLOCATOR();

	b3QuantizedBvh();

	virtual ~b3QuantizedBvh();

	///***************************************** expert/internal use only *************************
	void setQuantizationValues(const b3Vector3& bvhAabbMin, const b3Vector3& bvhAabbMax, b3Scalar quantizationMargin = b3Scalar(1.0));
	QuantizedNodeArray& getLeafNodeArray() { return m_quantizedLeafNodes; }
	///buildInternal is expert use only: assumes that setQuantizationValues and LeafNodeArray are initialized
	void buildInternal();
	///***************************************** expert/internal use only *************************

	void reportAabbOverlappingNodex(b3NodeOverlapCallback * nodeCallback, const b3Vector3& aabbMin, const b3Vector3& aabbMax) const;
	void reportRayOverlappingNodex(b3NodeOverlapCallback * nodeCallback, const b3Vector3& raySource, const b3Vector3& rayTarget) const;
	void reportBoxCastOverlappingNodex(b3NodeOverlapCallback * nodeCallback, const b3Vector3& raySource, const b3Vector3& rayTarget, const b3Vector3& aabbMin, const b3Vector3& aabbMax) const;

	B3_FORCE_INLINE void quantize(unsigned short* out, const b3Vector3& point, int isMax) const
	{
		b3Assert(m_useQuantization);

		b3Assert(point.getX() <= m_bvhAabbMax.getX());
		b3Assert(point.getY() <= m_bvhAabbMax.getY());
		b3Assert(point.getZ() <= m_bvhAabbMax.getZ());

		b3Assert(point.getX() >= m_bvhAabbMin.getX());
		b3Assert(point.getY() >= m_bvhAabbMin.getY());
		b3Assert(point.getZ() >= m_bvhAabbMin.getZ());

		b3Vector3 v = (point - m_bvhAabbMin) * m_bvhQuantization;
		///Make sure rounding is done in a way that unQuantize(quantizeWithClamp(...)) is conservative
		///end-points always set the first bit, so that they are sorted properly (so that neighbouring AABBs overlap properly)
		///@todo: double-check this
		if (isMax)
		{
			out[0] = (unsigned short)(((unsigned short)(v.getX() + b3Scalar(1.)) | 1));
			out[1] = (unsigned short)(((unsigned short)(v.getY() + b3Scalar(1.)) | 1));
			out[2] = (unsigned short)(((unsigned short)(v.getZ() + b3Scalar(1.)) | 1));
		}
		else
		{
			out[0] = (unsigned short)(((unsigned short)(v.getX()) & 0xfffe));
			out[1] = (unsigned short)(((unsigned short)(v.getY()) & 0xfffe));
			out[2] = (unsigned short)(((unsigned short)(v.getZ()) & 0xfffe));
		}

#ifdef DEBUG_CHECK_DEQUANTIZATION
		b3Vector3 newPoint = unQuantize(out);
		if (isMax)
		{
			if (newPoint.getX() < point.getX())
			{
				printf("unconservative X, diffX = %f, oldX=%f,newX=%f\n", newPoint.getX() - point.getX(), newPoint.getX(), point.getX());
			}
			if (newPoint.getY() < point.getY())
			{
				printf("unconservative Y, diffY = %f, oldY=%f,newY=%f\n", newPoint.getY() - point.getY(), newPoint.getY(), point.getY());
			}
			if (newPoint.getZ() < point.getZ())
			{
				printf("unconservative Z, diffZ = %f, oldZ=%f,newZ=%f\n", newPoint.getZ() - point.getZ(), newPoint.getZ(), point.getZ());
			}
		}
		else
		{
			if (newPoint.getX() > point.getX())
			{
				printf("unconservative X, diffX = %f, oldX=%f,newX=%f\n", newPoint.getX() - point.getX(), newPoint.getX(), point.getX());
			}
			if (newPoint.getY() > point.getY())
			{
				printf("unconservative Y, diffY = %f, oldY=%f,newY=%f\n", newPoint.getY() - point.getY(), newPoint.getY(), point.getY());
			}
			if (newPoint.getZ() > point.getZ())
			{
				printf("unconservative Z, diffZ = %f, oldZ=%f,newZ=%f\n", newPoint.getZ() - point.getZ(), newPoint.getZ(), point.getZ());
			}
		}
#endif  //DEBUG_CHECK_DEQUANTIZATION
	}

	B3_FORCE_INLINE void quantizeWithClamp(unsigned short* out, const b3Vector3& point2, int isMax) const
	{
		b3Assert(m_useQuantization);

		b3Vector3 clampedPoint(point2);
		clampedPoint.setMax(m_bvhAabbMin);
		clampedPoint.setMin(m_bvhAabbMax);

		quantize(out, clampedPoint, isMax);
	}

	B3_FORCE_INLINE b3Vector3 unQuantize(const unsigned short* vecIn) const
	{
		b3Vector3 vecOut;
		vecOut.setValue(
			(b3Scalar)(vecIn[0]) / (m_bvhQuantization.getX()),
			(b3Scalar)(vecIn[1]) / (m_bvhQuantization.getY()),
			(b3Scalar)(vecIn[2]) / (m_bvhQuantization.getZ()));
		vecOut += m_bvhAabbMin;
		return vecOut;
	}

	///setTraversalMode let's you choose between stackless, recursive or stackless cache friendly tree traversal. Note this is only implemented for quantized trees.
	void setTraversalMode(b3TraversalMode traversalMode)
	{
		m_traversalMode = traversalMode;
	}

	B3_FORCE_INLINE QuantizedNodeArray& getQuantizedNodeArray()
	{
		return m_quantizedContiguousNodes;
	}

	B3_FORCE_INLINE BvhSubtreeInfoArray& getSubtreeInfoArray()
	{
		return m_SubtreeHeaders;
	}

	////////////////////////////////////////////////////////////////////

	/////Calculate space needed to store BVH for serialization
	unsigned calculateSerializeBufferSize() const;

	/// Data buffer MUST be 16 byte aligned
	virtual bool serialize(void* o_alignedDataBuffer, unsigned i_dataBufferSize, bool i_swapEndian) const;

	///deSerializeInPlace loads and initializes a BVH from a buffer in memory 'in place'
	static b3QuantizedBvh* deSerializeInPlace(void* i_alignedDataBuffer, unsigned int i_dataBufferSize, bool i_swapEndian);

	static unsigned int getAlignmentSerializationPadding();
	//////////////////////////////////////////////////////////////////////

	virtual int calculateSerializeBufferSizeNew() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual const char* serialize(void* dataBuffer, b3Serializer* serializer) const;

	virtual void deSerializeFloat(struct b3QuantizedBvhFloatData & quantizedBvhFloatData);

	virtual void deSerializeDouble(struct b3QuantizedBvhDoubleData & quantizedBvhDoubleData);

	////////////////////////////////////////////////////////////////////

	B3_FORCE_INLINE bool isQuantized()
	{
		return m_useQuantization;
	}

private:
	// Special "copy" constructor that allows for in-place deserialization
	// Prevents b3Vector3's default constructor from being called, but doesn't inialize much else
	// ownsMemory should most likely be false if deserializing, and if you are not, don't call this (it also changes the function signature, which we need)
	b3QuantizedBvh(b3QuantizedBvh & other, bool ownsMemory);
};

struct b3OptimizedBvhNodeFloatData
{
	b3Vector3FloatData m_aabbMinOrg;
	b3Vector3FloatData m_aabbMaxOrg;
	int m_escapeIndex;
	int m_subPart;
	int m_triangleIndex;
	char m_pad[4];
};

struct b3OptimizedBvhNodeDoubleData
{
	b3Vector3DoubleData m_aabbMinOrg;
	b3Vector3DoubleData m_aabbMaxOrg;
	int m_escapeIndex;
	int m_subPart;
	int m_triangleIndex;
	char m_pad[4];
};

struct b3QuantizedBvhFloatData
{
	b3Vector3FloatData m_bvhAabbMin;
	b3Vector3FloatData m_bvhAabbMax;
	b3Vector3FloatData m_bvhQuantization;
	int m_curNodeIndex;
	int m_useQuantization;
	int m_numContiguousLeafNodes;
	int m_numQuantizedContiguousNodes;
	b3OptimizedBvhNodeFloatData* m_contiguousNodesPtr;
	b3QuantizedBvhNodeData* m_quantizedContiguousNodesPtr;
	b3BvhSubtreeInfoData* m_subTreeInfoPtr;
	int m_traversalMode;
	int m_numSubtreeHeaders;
};

struct b3QuantizedBvhDoubleData
{
	b3Vector3DoubleData m_bvhAabbMin;
	b3Vector3DoubleData m_bvhAabbMax;
	b3Vector3DoubleData m_bvhQuantization;
	int m_curNodeIndex;
	int m_useQuantization;
	int m_numContiguousLeafNodes;
	int m_numQuantizedContiguousNodes;
	b3OptimizedBvhNodeDoubleData* m_contiguousNodesPtr;
	b3QuantizedBvhNodeData* m_quantizedContiguousNodesPtr;

	int m_traversalMode;
	int m_numSubtreeHeaders;
	b3BvhSubtreeInfoData* m_subTreeInfoPtr;
};

B3_FORCE_INLINE int b3QuantizedBvh::calculateSerializeBufferSizeNew() const
{
	return sizeof(b3QuantizedBvhData);
}

#endif  //B3_QUANTIZED_BVH_H
