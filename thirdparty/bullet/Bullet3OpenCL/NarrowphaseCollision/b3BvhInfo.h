#ifndef B3_BVH_INFO_H
#define B3_BVH_INFO_H

#include "Bullet3Common/b3Vector3.h"

struct b3BvhInfo
{
	b3Vector3 m_aabbMin;
	b3Vector3 m_aabbMax;
	b3Vector3 m_quantization;
	int m_numNodes;
	int m_numSubTrees;
	int m_nodeOffset;
	int m_subTreeOffset;
};

#endif  //B3_BVH_INFO_H