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

// takahiro end
#define WG_SIZE 128 
#define m_numElems x
#define m_numBlocks y
#define m_numScanBlocks z

/*typedef struct
{
	uint m_numElems;
	uint m_numBlocks;
	uint m_numScanBlocks;
	uint m_padding[1];
} ConstBuffer;
*/

float4 ScanExclusiveFloat4(__local float4* data, u32 n, int lIdx, int lSize)
{
	float4 blocksum;
    int offset = 1;
    for(int nActive=n>>1; nActive>0; nActive>>=1, offset<<=1)
    {
        GROUP_LDS_BARRIER;
        for(int iIdx=lIdx; iIdx<nActive; iIdx+=lSize)
        {
            int ai = offset*(2*iIdx+1)-1;
            int bi = offset*(2*iIdx+2)-1;
            data[bi] += data[ai];
        }
	}

    GROUP_LDS_BARRIER;

    if( lIdx == 0 )
	{
		blocksum = data[ n-1 ];
    data[ n-1 ] = 0;
	}

	GROUP_LDS_BARRIER;

	offset >>= 1;
    for(int nActive=1; nActive<n; nActive<<=1, offset>>=1 )
    {
        GROUP_LDS_BARRIER;
        for( int iIdx = lIdx; iIdx<nActive; iIdx += lSize )
        {
            int ai = offset*(2*iIdx+1)-1;
            int bi = offset*(2*iIdx+2)-1;
            float4 temp = data[ai];
            data[ai] = data[bi];
            data[bi] += temp;
        }
	}
	GROUP_LDS_BARRIER;

	return blocksum;
}

__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
__kernel
void LocalScanKernel(__global float4* dst, __global float4* src, __global float4* sumBuffer,	uint4 cb)
{
	__local float4 ldsData[WG_SIZE*2];

	int gIdx = GET_GLOBAL_IDX;
	int lIdx = GET_LOCAL_IDX;

	ldsData[2*lIdx]     = ( 2*gIdx < cb.m_numElems )? src[2*gIdx]: 0;
	ldsData[2*lIdx + 1] = ( 2*gIdx+1 < cb.m_numElems )? src[2*gIdx + 1]: 0;

	float4 sum = ScanExclusiveFloat4(ldsData, WG_SIZE*2, GET_LOCAL_IDX, GET_GROUP_SIZE);

	if( lIdx == 0 ) 
		sumBuffer[GET_GROUP_IDX] = sum;

	if( (2*gIdx) < cb.m_numElems )
    {
        dst[2*gIdx]     = ldsData[2*lIdx];
	}
	if( (2*gIdx + 1) < cb.m_numElems )
	{
        dst[2*gIdx + 1] = ldsData[2*lIdx + 1];
    }
}

__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
__kernel
void AddOffsetKernel(__global float4* dst, __global float4* blockSum, uint4 cb)
{
	const u32 blockSize = WG_SIZE*2;

	int myIdx = GET_GROUP_IDX+1;
	int lIdx = GET_LOCAL_IDX;

	float4 iBlockSum = blockSum[myIdx];

	int endValue = min((myIdx+1)*(blockSize), cb.m_numElems);
	for(int i=myIdx*blockSize+lIdx; i<endValue; i+=GET_GROUP_SIZE)
	{
		dst[i] += iBlockSum;
	}
}

__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
__kernel
void TopLevelScanKernel(__global float4* dst, uint4 cb)
{
	__local float4 ldsData[2048];
	int gIdx = GET_GLOBAL_IDX;
	int lIdx = GET_LOCAL_IDX;
	int lSize = GET_GROUP_SIZE;

	for(int i=lIdx; i<cb.m_numScanBlocks; i+=lSize )
	{
		ldsData[i] = (i<cb.m_numBlocks)? dst[i]:0;
	}

	GROUP_LDS_BARRIER;

	float4 sum = ScanExclusiveFloat4(ldsData, cb.m_numScanBlocks, GET_LOCAL_IDX, GET_GROUP_SIZE);

	for(int i=lIdx; i<cb.m_numBlocks; i+=lSize )
	{
		dst[i] = ldsData[i];
	}

	if( gIdx == 0 )
	{
		dst[cb.m_numBlocks] = sum;
	}
}
