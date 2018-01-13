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
//Originally written by Erwin Coumans

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"

#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#ifdef cl_ext_atomic_counters_32
#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable
#else
#define counter32_t volatile __global int*
#endif

#define SIMD_WIDTH 64

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#define GET_GROUP_IDX get_group_id(0)
#define GET_LOCAL_IDX get_local_id(0)
#define GET_GLOBAL_IDX get_global_id(0)
#define GET_GROUP_SIZE get_local_size(0)
#define GET_NUM_GROUPS get_num_groups(0)
#define GROUP_LDS_BARRIER barrier(CLK_LOCAL_MEM_FENCE)
#define GROUP_MEM_FENCE mem_fence(CLK_LOCAL_MEM_FENCE)
#define AtomInc(x) atom_inc(&(x))
#define AtomInc1(x, out) out = atom_inc(&(x))
#define AppendInc(x, out) out = atomic_inc(x)
#define AtomAdd(x, value) atom_add(&(x), value)
#define AtomCmpxhg(x, cmp, value) atom_cmpxchg( &(x), cmp, value )
#define AtomXhg(x, value) atom_xchg ( &(x), value )


#define SELECT_UINT4( b, a, condition ) select( b,a,condition )

#define make_float4 (float4)
#define make_float2 (float2)
#define make_uint4 (uint4)
#define make_int4 (int4)
#define make_uint2 (uint2)
#define make_int2 (int2)


#define max2 max
#define min2 min


#define WG_SIZE 64





typedef struct 
{
	int m_n;
	int m_start;
	int m_staticIdx;
	int m_paddings[1];
} ConstBuffer;

typedef struct 
{
	int m_a;
	int m_b;
	u32 m_idx;
}Elem;





//	batching on the GPU
__kernel void CreateBatchesBruteForce( __global struct b3Contact4Data* gConstraints, 	__global const u32* gN, __global const u32* gStart, int m_staticIdx )
{
	int wgIdx = GET_GROUP_IDX;
	int lIdx = GET_LOCAL_IDX;
	
	const int m_n = gN[wgIdx];
	const int m_start = gStart[wgIdx];
		
	if( lIdx == 0 )
	{
		for (int i=0;i<m_n;i++)
		{
			int srcIdx = i+m_start;
			int batchIndex = i;
			gConstraints[ srcIdx ].m_batchIdx = batchIndex;	
		}
	}
}


#define CHECK_SIZE (WG_SIZE)




u32 readBuf(__local u32* buff, int idx)
{
	idx = idx % (32*CHECK_SIZE);
	int bitIdx = idx%32;
	int bufIdx = idx/32;
	return buff[bufIdx] & (1<<bitIdx);
}

void writeBuf(__local u32* buff, int idx)
{
	idx = idx % (32*CHECK_SIZE);
	int bitIdx = idx%32;
	int bufIdx = idx/32;
	buff[bufIdx] |= (1<<bitIdx);
	//atom_or( &buff[bufIdx], (1<<bitIdx) );
}

u32 tryWrite(__local u32* buff, int idx)
{
	idx = idx % (32*CHECK_SIZE);
	int bitIdx = idx%32;
	int bufIdx = idx/32;
	u32 ans = (u32)atom_or( &buff[bufIdx], (1<<bitIdx) );
	return ((ans >> bitIdx)&1) == 0;
}


//	batching on the GPU
__kernel void CreateBatchesNew( __global struct b3Contact4Data* gConstraints, __global const u32* gN, __global const u32* gStart, __global int* batchSizes, int staticIdx )
{
	int wgIdx = GET_GROUP_IDX;
	int lIdx = GET_LOCAL_IDX;
	const int numConstraints = gN[wgIdx];
	const int m_start = gStart[wgIdx];
	b3Contact4Data_t tmp;
	
	__local u32 ldsFixedBuffer[CHECK_SIZE];
		
	
	
	
	
	if( lIdx == 0 )
	{
	
		
		__global struct b3Contact4Data* cs = &gConstraints[m_start];	
	
		
		int numValidConstraints = 0;
		int batchIdx = 0;

		while( numValidConstraints < numConstraints)
		{
			int nCurrentBatch = 0;
			//	clear flag
	
			for(int i=0; i<CHECK_SIZE; i++) 
				ldsFixedBuffer[i] = 0;		

			for(int i=numValidConstraints; i<numConstraints; i++)
			{

				int bodyAS = cs[i].m_bodyAPtrAndSignBit;
				int bodyBS = cs[i].m_bodyBPtrAndSignBit;
				int bodyA = abs(bodyAS);
				int bodyB = abs(bodyBS);
				bool aIsStatic = (bodyAS<0) || bodyAS==staticIdx;
				bool bIsStatic = (bodyBS<0) || bodyBS==staticIdx;
				int aUnavailable = aIsStatic ? 0 : readBuf( ldsFixedBuffer, bodyA);
				int bUnavailable = bIsStatic ? 0 : readBuf( ldsFixedBuffer, bodyB);
				
				if( aUnavailable==0 && bUnavailable==0 ) // ok
				{
					if (!aIsStatic)
					{
						writeBuf( ldsFixedBuffer, bodyA );
					}
					if (!bIsStatic)
					{
						writeBuf( ldsFixedBuffer, bodyB );
					}

					cs[i].m_batchIdx = batchIdx;

					if (i!=numValidConstraints)
					{

						tmp = cs[i];
						cs[i] = cs[numValidConstraints];
						cs[numValidConstraints]  = tmp;


					}

					numValidConstraints++;
					
					nCurrentBatch++;
					if( nCurrentBatch == SIMD_WIDTH)
					{
						nCurrentBatch = 0;
						for(int i=0; i<CHECK_SIZE; i++) 
							ldsFixedBuffer[i] = 0;
						
					}
				}
			}//for
			batchIdx ++;
		}//while
		
		batchSizes[wgIdx] = batchIdx;

	}//if( lIdx == 0 )
	
	//return batchIdx;
}
