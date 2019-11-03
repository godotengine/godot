/*
Copyright (c) 2012 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Takahiro Harada

#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

typedef unsigned int u32;
#define GET_GROUP_IDX get_group_id(0)
#define GET_LOCAL_IDX get_local_id(0)
#define GET_GLOBAL_IDX get_global_id(0)
#define GET_GROUP_SIZE get_local_size(0)
#define GROUP_LDS_BARRIER barrier(CLK_LOCAL_MEM_FENCE)
#define GROUP_MEM_FENCE mem_fence(CLK_LOCAL_MEM_FENCE)
#define AtomInc(x) atom_inc(&(x))
#define AtomInc1(x, out) out = atom_inc(&(x))

#define make_uint4 (uint4)
#define make_uint2 (uint2)
#define make_int2 (int2)

typedef struct
{
	int m_n;
	int m_padding[3];
} ConstBuffer;



__kernel
__attribute__((reqd_work_group_size(64,1,1)))
void Copy1F4Kernel(__global float4* dst, __global float4* src, 
					ConstBuffer cb)
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < cb.m_n )
	{
		float4 a0 = src[gIdx];

		dst[ gIdx ] = a0;
	}
}

__kernel
__attribute__((reqd_work_group_size(64,1,1)))
void Copy2F4Kernel(__global float4* dst, __global float4* src, 
					ConstBuffer cb)
{
	int gIdx = GET_GLOBAL_IDX;

	if( 2*gIdx <= cb.m_n )
	{
		float4 a0 = src[gIdx*2+0];
		float4 a1 = src[gIdx*2+1];

		dst[ gIdx*2+0 ] = a0;
		dst[ gIdx*2+1 ] = a1;
	}
}

__kernel
__attribute__((reqd_work_group_size(64,1,1)))
void Copy4F4Kernel(__global float4* dst, __global float4* src, 
					ConstBuffer cb)
{
	int gIdx = GET_GLOBAL_IDX;

	if( 4*gIdx <= cb.m_n )
	{
		int idx0 = gIdx*4+0;
		int idx1 = gIdx*4+1;
		int idx2 = gIdx*4+2;
		int idx3 = gIdx*4+3;

		float4 a0 = src[idx0];
		float4 a1 = src[idx1];
		float4 a2 = src[idx2];
		float4 a3 = src[idx3];

		dst[ idx0 ] = a0;
		dst[ idx1 ] = a1;
		dst[ idx2 ] = a2;
		dst[ idx3 ] = a3;
	}
}

__kernel
__attribute__((reqd_work_group_size(64,1,1)))
void CopyF1Kernel(__global float* dstF1, __global float* srcF1, 
					ConstBuffer cb)
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < cb.m_n )
	{
		float a0 = srcF1[gIdx];

		dstF1[ gIdx ] = a0;
	}
}

__kernel
__attribute__((reqd_work_group_size(64,1,1)))
void CopyF2Kernel(__global float2* dstF2, __global float2* srcF2, 
					ConstBuffer cb)
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < cb.m_n )
	{
		float2 a0 = srcF2[gIdx];

		dstF2[ gIdx ] = a0;
	}
}

