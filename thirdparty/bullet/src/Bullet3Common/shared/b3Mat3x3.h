
#ifndef B3_MAT3x3_H
#define B3_MAT3x3_H

#include "Bullet3Common/shared/b3Quat.h"


#ifdef __cplusplus

#include "Bullet3Common/b3Matrix3x3.h"

#define b3Mat3x3 b3Matrix3x3
#define b3Mat3x3ConstArg const b3Matrix3x3&

inline b3Mat3x3 b3QuatGetRotationMatrix(b3QuatConstArg quat)
{
	return b3Mat3x3(quat);
}

inline b3Mat3x3 b3AbsoluteMat3x3(b3Mat3x3ConstArg mat)
{
	return mat.absolute();
}

#define b3GetRow(m,row) m.getRow(row)

__inline
b3Float4 mtMul3(b3Float4ConstArg a, b3Mat3x3ConstArg b)
{
	return b*a;
}


#else

typedef struct
{
	b3Float4 m_row[3];
}b3Mat3x3;

#define b3Mat3x3ConstArg const b3Mat3x3
#define b3GetRow(m,row) (m.m_row[row])

inline b3Mat3x3 b3QuatGetRotationMatrix(b3Quat quat)
{
	b3Float4 quat2 = (b3Float4)(quat.x*quat.x, quat.y*quat.y, quat.z*quat.z, 0.f);
	b3Mat3x3 out;

	out.m_row[0].x=1-2*quat2.y-2*quat2.z;
	out.m_row[0].y=2*quat.x*quat.y-2*quat.w*quat.z;
	out.m_row[0].z=2*quat.x*quat.z+2*quat.w*quat.y;
	out.m_row[0].w = 0.f;

	out.m_row[1].x=2*quat.x*quat.y+2*quat.w*quat.z;
	out.m_row[1].y=1-2*quat2.x-2*quat2.z;
	out.m_row[1].z=2*quat.y*quat.z-2*quat.w*quat.x;
	out.m_row[1].w = 0.f;

	out.m_row[2].x=2*quat.x*quat.z-2*quat.w*quat.y;
	out.m_row[2].y=2*quat.y*quat.z+2*quat.w*quat.x;
	out.m_row[2].z=1-2*quat2.x-2*quat2.y;
	out.m_row[2].w = 0.f;

	return out;
}

inline b3Mat3x3 b3AbsoluteMat3x3(b3Mat3x3ConstArg matIn)
{
	b3Mat3x3 out;
	out.m_row[0] = fabs(matIn.m_row[0]);
	out.m_row[1] = fabs(matIn.m_row[1]);
	out.m_row[2] = fabs(matIn.m_row[2]);
	return out;
}


__inline
b3Mat3x3 mtZero();

__inline
b3Mat3x3 mtIdentity();

__inline
b3Mat3x3 mtTranspose(b3Mat3x3 m);

__inline
b3Mat3x3 mtMul(b3Mat3x3 a, b3Mat3x3 b);

__inline
b3Float4 mtMul1(b3Mat3x3 a, b3Float4 b);

__inline
b3Float4 mtMul3(b3Float4 a, b3Mat3x3 b);

__inline
b3Mat3x3 mtZero()
{
	b3Mat3x3 m;
	m.m_row[0] = (b3Float4)(0.f);
	m.m_row[1] = (b3Float4)(0.f);
	m.m_row[2] = (b3Float4)(0.f);
	return m;
}

__inline
b3Mat3x3 mtIdentity()
{
	b3Mat3x3 m;
	m.m_row[0] = (b3Float4)(1,0,0,0);
	m.m_row[1] = (b3Float4)(0,1,0,0);
	m.m_row[2] = (b3Float4)(0,0,1,0);
	return m;
}

__inline
b3Mat3x3 mtTranspose(b3Mat3x3 m)
{
	b3Mat3x3 out;
	out.m_row[0] = (b3Float4)(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x, 0.f);
	out.m_row[1] = (b3Float4)(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y, 0.f);
	out.m_row[2] = (b3Float4)(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z, 0.f);
	return out;
}

__inline
b3Mat3x3 mtMul(b3Mat3x3 a, b3Mat3x3 b)
{
	b3Mat3x3 transB;
	transB = mtTranspose( b );
	b3Mat3x3 ans;
	//	why this doesn't run when 0ing in the for{}
	a.m_row[0].w = 0.f;
	a.m_row[1].w = 0.f;
	a.m_row[2].w = 0.f;
	for(int i=0; i<3; i++)
	{
//	a.m_row[i].w = 0.f;
		ans.m_row[i].x = b3Dot3F4(a.m_row[i],transB.m_row[0]);
		ans.m_row[i].y = b3Dot3F4(a.m_row[i],transB.m_row[1]);
		ans.m_row[i].z = b3Dot3F4(a.m_row[i],transB.m_row[2]);
		ans.m_row[i].w = 0.f;
	}
	return ans;
}

__inline
b3Float4 mtMul1(b3Mat3x3 a, b3Float4 b)
{
	b3Float4 ans;
	ans.x = b3Dot3F4( a.m_row[0], b );
	ans.y = b3Dot3F4( a.m_row[1], b );
	ans.z = b3Dot3F4( a.m_row[2], b );
	ans.w = 0.f;
	return ans;
}

__inline
b3Float4 mtMul3(b3Float4 a, b3Mat3x3 b)
{
	b3Float4 colx = b3MakeFloat4(b.m_row[0].x, b.m_row[1].x, b.m_row[2].x, 0);
	b3Float4 coly = b3MakeFloat4(b.m_row[0].y, b.m_row[1].y, b.m_row[2].y, 0);
	b3Float4 colz = b3MakeFloat4(b.m_row[0].z, b.m_row[1].z, b.m_row[2].z, 0);

	b3Float4 ans;
	ans.x = b3Dot3F4( a, colx );
	ans.y = b3Dot3F4( a, coly );
	ans.z = b3Dot3F4( a, colz );
	return ans;
}


#endif






#endif //B3_MAT3x3_H
