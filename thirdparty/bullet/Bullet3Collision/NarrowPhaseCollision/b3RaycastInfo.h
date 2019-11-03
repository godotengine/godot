
#ifndef B3_RAYCAST_INFO_H
#define B3_RAYCAST_INFO_H

#include "Bullet3Common/b3Vector3.h"

B3_ATTRIBUTE_ALIGNED16(struct)
b3RayInfo
{
	b3Vector3 m_from;
	b3Vector3 m_to;
};

B3_ATTRIBUTE_ALIGNED16(struct)
b3RayHit
{
	b3Scalar m_hitFraction;
	int m_hitBody;
	int m_hitResult1;
	int m_hitResult2;
	b3Vector3 m_hitPoint;
	b3Vector3 m_hitNormal;
};

#endif  //B3_RAYCAST_INFO_H
