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

#define STACK_SIZE (WG_SIZE*10)
//#define STACK_SIZE (WG_SIZE)
#define RING_SIZE 1024
#define RING_SIZE_MASK (RING_SIZE-1)
#define CHECK_SIZE (WG_SIZE)


#define GET_RING_CAPACITY (RING_SIZE - ldsRingEnd)
#define RING_END ldsTmp

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
//	buff[bufIdx] |= (1<<bitIdx);
	atom_or( &buff[bufIdx], (1<<bitIdx) );
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
__kernel void CreateBatches( __global const struct b3Contact4Data* gConstraints, __global struct b3Contact4Data* gConstraintsOut,
		__global const u32* gN, __global const u32* gStart, __global int* batchSizes, 
		int m_staticIdx )
{
	__local u32 ldsStackIdx[STACK_SIZE];
	__local u32 ldsStackEnd;
	__local Elem ldsRingElem[RING_SIZE];
	__local u32 ldsRingEnd;
	__local u32 ldsTmp;
	__local u32 ldsCheckBuffer[CHECK_SIZE];
	__local u32 ldsFixedBuffer[CHECK_SIZE];
	__local u32 ldsGEnd;
	__local u32 ldsDstEnd;

	int wgIdx = GET_GROUP_IDX;
	int lIdx = GET_LOCAL_IDX;
	
	const int m_n = gN[wgIdx];
	const int m_start = gStart[wgIdx];
		
	if( lIdx == 0 )
	{
		ldsRingEnd = 0;
		ldsGEnd = 0;
		ldsStackEnd = 0;
		ldsDstEnd = m_start;
	}
	
	
	
//	while(1)
//was 250
	int ie=0;
	int maxBatch = 0;
	for(ie=0; ie<50; ie++)
	{
		ldsFixedBuffer[lIdx] = 0;

		for(int giter=0; giter<4; giter++)
		{
			int ringCap = GET_RING_CAPACITY;
		
			//	1. fill ring
			if( ldsGEnd < m_n )
			{
				while( ringCap > WG_SIZE )
				{
					if( ldsGEnd >= m_n ) break;
					if( lIdx < ringCap - WG_SIZE )
					{
						int srcIdx;
						AtomInc1( ldsGEnd, srcIdx );
						if( srcIdx < m_n )
						{
							int dstIdx;
							AtomInc1( ldsRingEnd, dstIdx );
							
							int a = gConstraints[m_start+srcIdx].m_bodyAPtrAndSignBit;
							int b = gConstraints[m_start+srcIdx].m_bodyBPtrAndSignBit;
							ldsRingElem[dstIdx].m_a = (a>b)? b:a;
							ldsRingElem[dstIdx].m_b = (a>b)? a:b;
							ldsRingElem[dstIdx].m_idx = srcIdx;
						}
					}
					ringCap = GET_RING_CAPACITY;
				}
			}

			GROUP_LDS_BARRIER;
	
			//	2. fill stack
			__local Elem* dst = ldsRingElem;
			if( lIdx == 0 ) RING_END = 0;

			int srcIdx=lIdx;
			int end = ldsRingEnd;

			{
				for(int ii=0; ii<end; ii+=WG_SIZE, srcIdx+=WG_SIZE)
				{
					Elem e;
					if(srcIdx<end) e = ldsRingElem[srcIdx];
					bool done = (srcIdx<end)?false:true;

					for(int i=lIdx; i<CHECK_SIZE; i+=WG_SIZE) ldsCheckBuffer[lIdx] = 0;
					
					if( !done )
					{
						int aUsed = readBuf( ldsFixedBuffer, abs(e.m_a));
						int bUsed = readBuf( ldsFixedBuffer, abs(e.m_b));

						if( aUsed==0 && bUsed==0 )
						{
							int aAvailable=1;
							int bAvailable=1;
							int ea = abs(e.m_a);
							int eb = abs(e.m_b);

							bool aStatic = (e.m_a<0) ||(ea==m_staticIdx);
							bool bStatic = (e.m_b<0) ||(eb==m_staticIdx);
							
							if (!aStatic)
								aAvailable = tryWrite( ldsCheckBuffer, ea );
							if (!bStatic)
								bAvailable = tryWrite( ldsCheckBuffer, eb );
							
							//aAvailable = aStatic? 1: aAvailable;
							//bAvailable = bStatic? 1: bAvailable;

							bool success = (aAvailable && bAvailable);
							if(success)
							{
							
								if (!aStatic)
									writeBuf( ldsFixedBuffer, ea );
								if (!bStatic)
									writeBuf( ldsFixedBuffer, eb );
							}
							done = success;
						}
					}

					//	put it aside
					if(srcIdx<end)
					{
						if( done )
						{
							int dstIdx; AtomInc1( ldsStackEnd, dstIdx );
							if( dstIdx < STACK_SIZE )
								ldsStackIdx[dstIdx] = e.m_idx;
							else{
								done = false;
								AtomAdd( ldsStackEnd, -1 );
							}
						}
						if( !done )
						{
							int dstIdx; AtomInc1( RING_END, dstIdx );
							dst[dstIdx] = e;
						}
					}

					//	if filled, flush
					if( ldsStackEnd == STACK_SIZE )
					{
						for(int i=lIdx; i<STACK_SIZE; i+=WG_SIZE)
						{
							int idx = m_start + ldsStackIdx[i];
							int dstIdx; AtomInc1( ldsDstEnd, dstIdx );
							gConstraintsOut[ dstIdx ] = gConstraints[ idx ];
							gConstraintsOut[ dstIdx ].m_batchIdx = ie;
						}
						if( lIdx == 0 ) ldsStackEnd = 0;

						//for(int i=lIdx; i<CHECK_SIZE; i+=WG_SIZE) 
						ldsFixedBuffer[lIdx] = 0;
					}
				}
			}

			if( lIdx == 0 ) ldsRingEnd = RING_END;
		}

		GROUP_LDS_BARRIER;

		for(int i=lIdx; i<ldsStackEnd; i+=WG_SIZE)
		{
			int idx = m_start + ldsStackIdx[i];
			int dstIdx; AtomInc1( ldsDstEnd, dstIdx );
			gConstraintsOut[ dstIdx ] = gConstraints[ idx ];
			gConstraintsOut[ dstIdx ].m_batchIdx = ie;
		}

		//	in case it couldn't consume any pair. Flush them
		//	todo. Serial batch worth while?
		if( ldsStackEnd == 0 )
		{
			for(int i=lIdx; i<ldsRingEnd; i+=WG_SIZE)
			{
				int idx = m_start + ldsRingElem[i].m_idx;
				int dstIdx; AtomInc1( ldsDstEnd, dstIdx );
				gConstraintsOut[ dstIdx ] = gConstraints[ idx ];
				int curBatch = 100+i;
				if (maxBatch < curBatch)
					maxBatch = curBatch;
				
				gConstraintsOut[ dstIdx ].m_batchIdx = curBatch;
				
			}
			GROUP_LDS_BARRIER;
			if( lIdx == 0 ) ldsRingEnd = 0;
		}

		if( lIdx == 0 ) ldsStackEnd = 0;

		GROUP_LDS_BARRIER;

		//	termination
		if( ldsGEnd == m_n && ldsRingEnd == 0 )
			break;
	}

	if( lIdx == 0 )
	{
		if (maxBatch < ie)
			maxBatch=ie;
		batchSizes[wgIdx]=maxBatch;
	}

}






















