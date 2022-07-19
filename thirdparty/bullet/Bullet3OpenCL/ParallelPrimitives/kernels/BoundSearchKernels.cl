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


typedef unsigned int u32;
#define GET_GROUP_IDX get_group_id(0)
#define GET_LOCAL_IDX get_local_id(0)
#define GET_GLOBAL_IDX get_global_id(0)
#define GET_GROUP_SIZE get_local_size(0)
#define GROUP_LDS_BARRIER barrier(CLK_LOCAL_MEM_FENCE)

typedef struct
{
	u32 m_key; 
	u32 m_value;
}SortData;



typedef struct
{
	u32 m_nSrc;
	u32 m_nDst;
	u32 m_padding[2];
} ConstBuffer;



__attribute__((reqd_work_group_size(64,1,1)))
__kernel
void SearchSortDataLowerKernel(__global SortData* src, __global u32 *dst, 
					unsigned int nSrc, unsigned int nDst)
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < nSrc )
	{
		SortData first; first.m_key = (u32)(-1); first.m_value = (u32)(-1);
		SortData end; end.m_key = nDst; end.m_value = nDst;

		SortData iData = (gIdx==0)? first: src[gIdx-1];
		SortData jData = (gIdx==nSrc)? end: src[gIdx];

		if( iData.m_key != jData.m_key )
		{
//			for(u32 k=iData.m_key+1; k<=min(jData.m_key, nDst-1); k++)
			u32 k = jData.m_key;
			{
				dst[k] = gIdx;
			}
		}
	}
}


__attribute__((reqd_work_group_size(64,1,1)))
__kernel
void SearchSortDataUpperKernel(__global SortData* src, __global u32 *dst, 
					unsigned int nSrc, unsigned int nDst)
{
	int gIdx = GET_GLOBAL_IDX+1;

	if( gIdx < nSrc+1 )
	{
		SortData first; first.m_key = 0; first.m_value = 0;
		SortData end; end.m_key = nDst; end.m_value = nDst;

		SortData iData = src[gIdx-1];
		SortData jData = (gIdx==nSrc)? end: src[gIdx];

		if( iData.m_key != jData.m_key )
		{
			u32 k = iData.m_key;
			{
				dst[k] = gIdx;
			}
		}
	}
}

__attribute__((reqd_work_group_size(64,1,1)))
__kernel
void SubtractKernel(__global u32* A, __global u32 *B, __global u32 *C, 
					unsigned int nSrc, unsigned int nDst)
{
	int gIdx = GET_GLOBAL_IDX;
	

	if( gIdx < nDst )
	{
		C[gIdx] = A[gIdx] - B[gIdx];
	}
}

