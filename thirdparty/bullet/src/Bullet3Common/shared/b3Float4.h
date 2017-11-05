#ifndef B3_FLOAT4_H
#define B3_FLOAT4_H

#include "Bullet3Common/shared/b3PlatformDefinitions.h"

#ifdef __cplusplus
	#include "Bullet3Common/b3Vector3.h"
	#define b3Float4 b3Vector3
	#define b3Float4ConstArg const b3Vector3&
	#define b3Dot3F4 b3Dot
	#define b3Cross3 b3Cross
	#define	b3MakeFloat4  b3MakeVector3
	inline b3Vector3 b3Normalized(const b3Vector3& vec)
	{
		return vec.normalized();
	}

	inline b3Float4 b3FastNormalized3(b3Float4ConstArg v)
	{
		return v.normalized();
	}

	inline b3Float4 b3MaxFloat4 (const b3Float4& a, const b3Float4& b)
	{
		b3Float4 tmp = a;
		tmp.setMax(b);
		return tmp;
	}
	inline b3Float4 b3MinFloat4 (const b3Float4& a, const b3Float4& b)
	{
		b3Float4 tmp = a;
		tmp.setMin(b);
		return tmp;
	}



#else
	typedef float4	b3Float4;
	#define b3Float4ConstArg const b3Float4
	#define b3MakeFloat4 (float4)
	float b3Dot3F4(b3Float4ConstArg v0,b3Float4ConstArg v1)
	{
		float4 a1 = b3MakeFloat4(v0.xyz,0.f);
		float4 b1 = b3MakeFloat4(v1.xyz,0.f);
		return dot(a1, b1);
	}
	b3Float4 b3Cross3(b3Float4ConstArg v0,b3Float4ConstArg v1)
	{
		float4 a1 = b3MakeFloat4(v0.xyz,0.f);
		float4 b1 = b3MakeFloat4(v1.xyz,0.f);
		return cross(a1, b1);
	}
	#define b3MinFloat4 min
	#define b3MaxFloat4 max

	#define b3Normalized(a) normalize(a)

#endif 


		
inline bool b3IsAlmostZero(b3Float4ConstArg v)
{
	if(b3Fabs(v.x)>1e-6 || b3Fabs(v.y)>1e-6 || b3Fabs(v.z)>1e-6)	
		return false;
	return true;
}


inline int    b3MaxDot( b3Float4ConstArg vec, __global const b3Float4* vecArray, int vecLen, float* dotOut )
{
    float maxDot = -B3_INFINITY;
    int i = 0;
    int ptIndex = -1;
    for( i = 0; i < vecLen; i++ )
    {
        float dot = b3Dot3F4(vecArray[i],vec);
            
        if( dot > maxDot )
        {
            maxDot = dot;
            ptIndex = i;
        }
    }
	b3Assert(ptIndex>=0);
    if (ptIndex<0)
	{
		ptIndex = 0;
	}
    *dotOut = maxDot;
    return ptIndex;
}



#endif //B3_FLOAT4_H
