#ifndef B3_QUAT_H
#define B3_QUAT_H

#include "Bullet3Common/shared/b3PlatformDefinitions.h"
#include "Bullet3Common/shared/b3Float4.h"

#ifdef __cplusplus
	#include "Bullet3Common/b3Quaternion.h"
	#include "Bullet3Common/b3Transform.h"

	#define b3Quat b3Quaternion
	#define b3QuatConstArg const b3Quaternion&
	inline b3Quat b3QuatInverse(b3QuatConstArg orn)
	{
		return orn.inverse();
	}

	inline b3Float4 b3TransformPoint(b3Float4ConstArg point, b3Float4ConstArg translation, b3QuatConstArg  orientation)
	{
		b3Transform tr;
		tr.setOrigin(translation);
		tr.setRotation(orientation);
		return tr(point);
	}

#else
	typedef float4	b3Quat;
	#define b3QuatConstArg const b3Quat
	
	
inline float4 b3FastNormalize4(float4 v)
{
	v = (float4)(v.xyz,0.f);
	return fast_normalize(v);
}
	
inline b3Quat b3QuatMul(b3Quat a, b3Quat b);
inline b3Quat b3QuatNormalized(b3QuatConstArg in);
inline b3Quat b3QuatRotate(b3QuatConstArg q, b3QuatConstArg vec);
inline b3Quat b3QuatInvert(b3QuatConstArg q);
inline b3Quat b3QuatInverse(b3QuatConstArg q);

inline b3Quat b3QuatMul(b3QuatConstArg a, b3QuatConstArg b)
{
	b3Quat ans;
	ans = b3Cross3( a, b );
	ans += a.w*b+b.w*a;
//	ans.w = a.w*b.w - (a.x*b.x+a.y*b.y+a.z*b.z);
	ans.w = a.w*b.w - b3Dot3F4(a, b);
	return ans;
}

inline b3Quat b3QuatNormalized(b3QuatConstArg in)
{
	b3Quat q;
	q=in;
	//return b3FastNormalize4(in);
	float len = native_sqrt(dot(q, q));
	if(len > 0.f)
	{
		q *= 1.f / len;
	}
	else
	{
		q.x = q.y = q.z = 0.f;
		q.w = 1.f;
	}
	return q;
}
inline float4 b3QuatRotate(b3QuatConstArg q, b3QuatConstArg vec)
{
	b3Quat qInv = b3QuatInvert( q );
	float4 vcpy = vec;
	vcpy.w = 0.f;
	float4 out = b3QuatMul(b3QuatMul(q,vcpy),qInv);
	return out;
}



inline b3Quat b3QuatInverse(b3QuatConstArg q)
{
	return (b3Quat)(-q.xyz, q.w);
}

inline b3Quat b3QuatInvert(b3QuatConstArg q)
{
	return (b3Quat)(-q.xyz, q.w);
}

inline float4 b3QuatInvRotate(b3QuatConstArg q, b3QuatConstArg vec)
{
	return b3QuatRotate( b3QuatInvert( q ), vec );
}

inline b3Float4 b3TransformPoint(b3Float4ConstArg point, b3Float4ConstArg translation, b3QuatConstArg  orientation)
{
	return b3QuatRotate( orientation, point ) + (translation);
}
	
#endif 

#endif //B3_QUAT_H
