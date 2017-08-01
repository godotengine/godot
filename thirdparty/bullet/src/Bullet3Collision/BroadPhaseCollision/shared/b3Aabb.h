
#ifndef B3_AABB_H
#define B3_AABB_H


#include "Bullet3Common/shared/b3Float4.h"
#include "Bullet3Common/shared/b3Mat3x3.h"

typedef struct b3Aabb b3Aabb_t;

struct b3Aabb
{
	union
	{
		float m_min[4];
		b3Float4 m_minVec;
		int m_minIndices[4];
	};
	union
	{
		float	m_max[4];
		b3Float4 m_maxVec;
		int m_signedMaxIndices[4];
	};
};

inline void b3TransformAabb2(b3Float4ConstArg localAabbMin,b3Float4ConstArg localAabbMax, float margin,
						b3Float4ConstArg pos,
						b3QuatConstArg orn,
						b3Float4* aabbMinOut,b3Float4* aabbMaxOut)
{
		b3Float4 localHalfExtents = 0.5f*(localAabbMax-localAabbMin);
		localHalfExtents+=b3MakeFloat4(margin,margin,margin,0.f);
		b3Float4 localCenter = 0.5f*(localAabbMax+localAabbMin);
		b3Mat3x3 m;
		m = b3QuatGetRotationMatrix(orn);
		b3Mat3x3 abs_b = b3AbsoluteMat3x3(m);
		b3Float4 center = b3TransformPoint(localCenter,pos,orn);
		
		b3Float4 extent = b3MakeFloat4(b3Dot3F4(localHalfExtents,b3GetRow(abs_b,0)),
										 b3Dot3F4(localHalfExtents,b3GetRow(abs_b,1)),
										 b3Dot3F4(localHalfExtents,b3GetRow(abs_b,2)),
										 0.f);
		*aabbMinOut = center-extent;
		*aabbMaxOut = center+extent;
}

/// conservative test for overlap between two aabbs
inline bool b3TestAabbAgainstAabb(b3Float4ConstArg aabbMin1,b3Float4ConstArg aabbMax1,
								b3Float4ConstArg aabbMin2, b3Float4ConstArg aabbMax2)
{
	bool overlap = true;
	overlap = (aabbMin1.x > aabbMax2.x || aabbMax1.x < aabbMin2.x) ? false : overlap;
	overlap = (aabbMin1.z > aabbMax2.z || aabbMax1.z < aabbMin2.z) ? false : overlap;
	overlap = (aabbMin1.y > aabbMax2.y || aabbMax1.y < aabbMin2.y) ? false : overlap;
	return overlap;
}

#endif //B3_AABB_H
