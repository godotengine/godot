

#ifndef B3_QUANTIZED_BVH_NODE_H
#define B3_QUANTIZED_BVH_NODE_H

#include "Bullet3Common/shared/b3Float4.h"

#define B3_MAX_NUM_PARTS_IN_BITS 10

///b3QuantizedBvhNodeData is a compressed aabb node, 16 bytes.
///Node can be used for leafnode or internal node. Leafnodes can point to 32-bit triangle index (non-negative range).
typedef struct b3QuantizedBvhNodeData b3QuantizedBvhNodeData_t;

struct b3QuantizedBvhNodeData
{
	//12 bytes
	unsigned short int m_quantizedAabbMin[3];
	unsigned short int m_quantizedAabbMax[3];
	//4 bytes
	int m_escapeIndexOrTriangleIndex;
};

inline int b3GetTriangleIndex(const b3QuantizedBvhNodeData* rootNode)
{
	unsigned int x = 0;
	unsigned int y = (~(x & 0)) << (31 - B3_MAX_NUM_PARTS_IN_BITS);
	// Get only the lower bits where the triangle index is stored
	return (rootNode->m_escapeIndexOrTriangleIndex & ~(y));
}

inline int b3IsLeaf(const b3QuantizedBvhNodeData* rootNode)
{
	//skipindex is negative (internal node), triangleindex >=0 (leafnode)
	return (rootNode->m_escapeIndexOrTriangleIndex >= 0) ? 1 : 0;
}

inline int b3GetEscapeIndex(const b3QuantizedBvhNodeData* rootNode)
{
	return -rootNode->m_escapeIndexOrTriangleIndex;
}

inline void b3QuantizeWithClamp(unsigned short* out, b3Float4ConstArg point2, int isMax, b3Float4ConstArg bvhAabbMin, b3Float4ConstArg bvhAabbMax, b3Float4ConstArg bvhQuantization)
{
	b3Float4 clampedPoint = b3MaxFloat4(point2, bvhAabbMin);
	clampedPoint = b3MinFloat4(clampedPoint, bvhAabbMax);

	b3Float4 v = (clampedPoint - bvhAabbMin) * bvhQuantization;
	if (isMax)
	{
		out[0] = (unsigned short)(((unsigned short)(v.x + 1.f) | 1));
		out[1] = (unsigned short)(((unsigned short)(v.y + 1.f) | 1));
		out[2] = (unsigned short)(((unsigned short)(v.z + 1.f) | 1));
	}
	else
	{
		out[0] = (unsigned short)(((unsigned short)(v.x) & 0xfffe));
		out[1] = (unsigned short)(((unsigned short)(v.y) & 0xfffe));
		out[2] = (unsigned short)(((unsigned short)(v.z) & 0xfffe));
	}
}

inline int b3TestQuantizedAabbAgainstQuantizedAabbSlow(
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

#endif  //B3_QUANTIZED_BVH_NODE_H
