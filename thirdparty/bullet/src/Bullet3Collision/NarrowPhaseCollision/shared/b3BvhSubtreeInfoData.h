
#ifndef B3_BVH_SUBTREE_INFO_DATA_H
#define B3_BVH_SUBTREE_INFO_DATA_H

typedef struct b3BvhSubtreeInfoData b3BvhSubtreeInfoData_t;

struct b3BvhSubtreeInfoData
{
	//12 bytes
	unsigned short int	m_quantizedAabbMin[3];
	unsigned short int	m_quantizedAabbMax[3];
	//4 bytes, points to the root of the subtree
	int			m_rootNodeIndex;
	//4 bytes
	int			m_subtreeSize;
	int			m_padding[3];
};

#endif //B3_BVH_SUBTREE_INFO_DATA_H

