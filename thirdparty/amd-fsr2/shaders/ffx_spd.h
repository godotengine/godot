// This file is part of the FidelityFX SDK.
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifdef FFX_CPU
FFX_STATIC void SpdSetup(FfxUInt32x2    dispatchThreadGroupCountXY,  // CPU side: dispatch thread group count xy
                         FfxUInt32x2    workGroupOffset,             // GPU side: pass in as constant
                         FfxUInt32x2    numWorkGroupsAndMips,        // GPU side: pass in as constant
                         FfxUInt32x4     rectInfo,                    // left, top, width, height
                         FfxInt32 mips)                        // optional: if -1, calculate based on rect width and height
{
    workGroupOffset[0] = rectInfo[0] / 64;  // rectInfo[0] = left
    workGroupOffset[1] = rectInfo[1] / 64;  // rectInfo[1] = top

    FfxUInt32 endIndexX = (rectInfo[0] + rectInfo[2] - 1) / 64;  // rectInfo[0] = left, rectInfo[2] = width
    FfxUInt32 endIndexY = (rectInfo[1] + rectInfo[3] - 1) / 64;  // rectInfo[1] = top, rectInfo[3] = height

    dispatchThreadGroupCountXY[0] = endIndexX + 1 - workGroupOffset[0];
    dispatchThreadGroupCountXY[1] = endIndexY + 1 - workGroupOffset[1];

    numWorkGroupsAndMips[0] = (dispatchThreadGroupCountXY[0]) * (dispatchThreadGroupCountXY[1]);

    if (mips >= 0)
    {
        numWorkGroupsAndMips[1] = FfxUInt32(mips);
    }
    else
    {  
        // calculate based on rect width and height
        FfxUInt32 resolution    = ffxMax(rectInfo[2], rectInfo[3]);
        numWorkGroupsAndMips[1] = FfxUInt32((ffxMin(floor(log2(FfxFloat32(resolution))), FfxFloat32(12))));
    }
}

FFX_STATIC void SpdSetup(FfxUInt32x2 dispatchThreadGroupCountXY,  // CPU side: dispatch thread group count xy
                         FfxUInt32x2 workGroupOffset,             // GPU side: pass in as constant
                         FfxUInt32x2 numWorkGroupsAndMips,        // GPU side: pass in as constant
                         FfxUInt32x4  rectInfo)                    // left, top, width, height
{
    SpdSetup(dispatchThreadGroupCountXY, workGroupOffset, numWorkGroupsAndMips, rectInfo, -1);
}
#endif // #ifdef FFX_CPU


//==============================================================================================================================
//                                                     NON-PACKED VERSION
//==============================================================================================================================
#ifdef FFX_GPU
#ifdef SPD_PACKED_ONLY
// Avoid compiler error
FfxFloat32x4 SpdLoadSourceImage(FfxInt32x2 p, FfxUInt32 slice)
{
    return FfxFloat32x4(0.0, 0.0, 0.0, 0.0);
}

FfxFloat32x4 SpdLoad(FfxInt32x2 p, FfxUInt32 slice)
{
    return FfxFloat32x4(0.0, 0.0, 0.0, 0.0);
}
void SpdStore(FfxInt32x2 p, FfxFloat32x4 value, FfxUInt32 mip, FfxUInt32 slice)
{
}
FfxFloat32x4 SpdLoadIntermediate(FfxUInt32 x, FfxUInt32 y)
{
    return FfxFloat32x4(0.0, 0.0, 0.0, 0.0);
}
void SpdStoreIntermediate(FfxUInt32 x, FfxUInt32 y, FfxFloat32x4 value)
{
}
FfxFloat32x4 SpdReduce4(FfxFloat32x4 v0, FfxFloat32x4 v1, FfxFloat32x4 v2, FfxFloat32x4 v3)
{
    return FfxFloat32x4(0.0, 0.0, 0.0, 0.0);
}
#endif // #ifdef SPD_PACKED_ONLY

//_____________________________________________________________/\_______________________________________________________________
#if defined(FFX_GLSL) && !defined(SPD_NO_WAVE_OPERATIONS)
#extension GL_KHR_shader_subgroup_quad:require
#endif

void SpdWorkgroupShuffleBarrier()
{
#ifdef FFX_GLSL
    barrier();
#endif
#ifdef FFX_HLSL
    GroupMemoryBarrierWithGroupSync();
#endif
}

// Only last active workgroup should proceed
bool SpdExitWorkgroup(FfxUInt32 numWorkGroups, FfxUInt32 localInvocationIndex, FfxUInt32 slice)
{
    // global atomic counter
    if (localInvocationIndex == 0)
    {
        SpdIncreaseAtomicCounter(slice);
    }

    SpdWorkgroupShuffleBarrier();
    return (SpdGetAtomicCounter() != (numWorkGroups - 1));
}

// User defined: FfxFloat32x4 SpdReduce4(FfxFloat32x4 v0, FfxFloat32x4 v1, FfxFloat32x4 v2, FfxFloat32x4 v3);
FfxFloat32x4 SpdReduceQuad(FfxFloat32x4 v)
{
#if defined(FFX_GLSL) && !defined(SPD_NO_WAVE_OPERATIONS)

    FfxFloat32x4 v0 = v;
    FfxFloat32x4 v1 = subgroupQuadSwapHorizontal(v);
    FfxFloat32x4 v2 = subgroupQuadSwapVertical(v);
    FfxFloat32x4 v3 = subgroupQuadSwapDiagonal(v);
    return SpdReduce4(v0, v1, v2, v3);

#elif defined(FFX_HLSL) && !defined(SPD_NO_WAVE_OPERATIONS)

    // requires SM6.0
    FfxUInt32 quad = WaveGetLaneIndex() & (~0x3);
    FfxFloat32x4     v0   = v;
    FfxFloat32x4     v1   = WaveReadLaneAt(v, quad | 1);
    FfxFloat32x4     v2   = WaveReadLaneAt(v, quad | 2);
    FfxFloat32x4     v3   = WaveReadLaneAt(v, quad | 3);
    return SpdReduce4(v0, v1, v2, v3);
/*
    // if SM6.0 is not available, you can use the AMD shader intrinsics
    // the AMD shader intrinsics are available in AMD GPU Services (AGS) library:
    // https://gpuopen.com/amd-gpu-services-ags-library/
    // works for DX11
    FfxFloat32x4 v0 = v;
    FfxFloat32x4 v1;
    v1.x = AmdExtD3DShaderIntrinsics_SwizzleF(v.x, AmdExtD3DShaderIntrinsicsSwizzle_SwapX1);
    v1.y = AmdExtD3DShaderIntrinsics_SwizzleF(v.y, AmdExtD3DShaderIntrinsicsSwizzle_SwapX1);
    v1.z = AmdExtD3DShaderIntrinsics_SwizzleF(v.z, AmdExtD3DShaderIntrinsicsSwizzle_SwapX1);
    v1.w = AmdExtD3DShaderIntrinsics_SwizzleF(v.w, AmdExtD3DShaderIntrinsicsSwizzle_SwapX1);
    FfxFloat32x4 v2;
    v2.x = AmdExtD3DShaderIntrinsics_SwizzleF(v.x, AmdExtD3DShaderIntrinsicsSwizzle_SwapX2);
    v2.y = AmdExtD3DShaderIntrinsics_SwizzleF(v.y, AmdExtD3DShaderIntrinsicsSwizzle_SwapX2);
    v2.z = AmdExtD3DShaderIntrinsics_SwizzleF(v.z, AmdExtD3DShaderIntrinsicsSwizzle_SwapX2);
    v2.w = AmdExtD3DShaderIntrinsics_SwizzleF(v.w, AmdExtD3DShaderIntrinsicsSwizzle_SwapX2);
    FfxFloat32x4 v3;
    v3.x = AmdExtD3DShaderIntrinsics_SwizzleF(v.x, AmdExtD3DShaderIntrinsicsSwizzle_ReverseX4);
    v3.y = AmdExtD3DShaderIntrinsics_SwizzleF(v.y, AmdExtD3DShaderIntrinsicsSwizzle_ReverseX4);
    v3.z = AmdExtD3DShaderIntrinsics_SwizzleF(v.z, AmdExtD3DShaderIntrinsicsSwizzle_ReverseX4);
    v3.w = AmdExtD3DShaderIntrinsics_SwizzleF(v.w, AmdExtD3DShaderIntrinsicsSwizzle_ReverseX4);
    return SpdReduce4(v0, v1, v2, v3);
    */
#endif
    return v;
}

FfxFloat32x4 SpdReduceIntermediate(FfxUInt32x2 i0, FfxUInt32x2 i1, FfxUInt32x2 i2, FfxUInt32x2 i3)
{
    FfxFloat32x4 v0 = SpdLoadIntermediate(i0.x, i0.y);
    FfxFloat32x4 v1 = SpdLoadIntermediate(i1.x, i1.y);
    FfxFloat32x4 v2 = SpdLoadIntermediate(i2.x, i2.y);
    FfxFloat32x4 v3 = SpdLoadIntermediate(i3.x, i3.y);
    return SpdReduce4(v0, v1, v2, v3);
}

FfxFloat32x4 SpdReduceLoad4(FfxUInt32x2 i0, FfxUInt32x2 i1, FfxUInt32x2 i2, FfxUInt32x2 i3, FfxUInt32 slice)
{
    FfxFloat32x4 v0 = SpdLoad(FfxInt32x2(i0), slice);
    FfxFloat32x4 v1 = SpdLoad(FfxInt32x2(i1), slice);
    FfxFloat32x4 v2 = SpdLoad(FfxInt32x2(i2), slice);
    FfxFloat32x4 v3 = SpdLoad(FfxInt32x2(i3), slice);
    return SpdReduce4(v0, v1, v2, v3);
}

FfxFloat32x4 SpdReduceLoad4(FfxUInt32x2 base, FfxUInt32 slice)
{
    return SpdReduceLoad4(FfxUInt32x2(base + FfxUInt32x2(0, 0)), FfxUInt32x2(base + FfxUInt32x2(0, 1)), FfxUInt32x2(base + FfxUInt32x2(1, 0)), FfxUInt32x2(base + FfxUInt32x2(1, 1)), slice);
}

FfxFloat32x4 SpdReduceLoadSourceImage4(FfxUInt32x2 i0, FfxUInt32x2 i1, FfxUInt32x2 i2, FfxUInt32x2 i3, FfxUInt32 slice)
{
    FfxFloat32x4 v0 = SpdLoadSourceImage(FfxInt32x2(i0), slice);
    FfxFloat32x4 v1 = SpdLoadSourceImage(FfxInt32x2(i1), slice);
    FfxFloat32x4 v2 = SpdLoadSourceImage(FfxInt32x2(i2), slice);
    FfxFloat32x4 v3 = SpdLoadSourceImage(FfxInt32x2(i3), slice);
    return SpdReduce4(v0, v1, v2, v3);
}

FfxFloat32x4 SpdReduceLoadSourceImage(FfxUInt32x2 base, FfxUInt32 slice)
{
#ifdef SPD_LINEAR_SAMPLER
    return SpdLoadSourceImage(FfxInt32x2(base), slice);
#else
    return SpdReduceLoadSourceImage4(FfxUInt32x2(base + FfxUInt32x2(0, 0)), FfxUInt32x2(base + FfxUInt32x2(0, 1)), FfxUInt32x2(base + FfxUInt32x2(1, 0)), FfxUInt32x2(base + FfxUInt32x2(1, 1)), slice);
#endif
}

void SpdDownsampleMips_0_1_Intrinsics(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mip, FfxUInt32 slice)
{
    FfxFloat32x4 v[4];

    FfxInt32x2 tex = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2, y * 2);
    FfxInt32x2 pix = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x, y);
    v[0]     = SpdReduceLoadSourceImage(tex, slice);
    SpdStore(pix, v[0], 0, slice);

    tex  = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2 + 32, y * 2);
    pix  = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x + 16, y);
    v[1] = SpdReduceLoadSourceImage(tex, slice);
    SpdStore(pix, v[1], 0, slice);

    tex  = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2, y * 2 + 32);
    pix  = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x, y + 16);
    v[2] = SpdReduceLoadSourceImage(tex, slice);
    SpdStore(pix, v[2], 0, slice);

    tex  = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2 + 32, y * 2 + 32);
    pix  = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x + 16, y + 16);
    v[3] = SpdReduceLoadSourceImage(tex, slice);
    SpdStore(pix, v[3], 0, slice);

    if (mip <= 1)
        return;

    v[0] = SpdReduceQuad(v[0]);
    v[1] = SpdReduceQuad(v[1]);
    v[2] = SpdReduceQuad(v[2]);
    v[3] = SpdReduceQuad(v[3]);

    if ((localInvocationIndex % 4) == 0)
    {
        SpdStore(FfxInt32x2(workGroupID.xy * 16) + FfxInt32x2(x / 2, y / 2), v[0], 1, slice);
        SpdStoreIntermediate(x / 2, y / 2, v[0]);

        SpdStore(FfxInt32x2(workGroupID.xy * 16) + FfxInt32x2(x / 2 + 8, y / 2), v[1], 1, slice);
        SpdStoreIntermediate(x / 2 + 8, y / 2, v[1]);

        SpdStore(FfxInt32x2(workGroupID.xy * 16) + FfxInt32x2(x / 2, y / 2 + 8), v[2], 1, slice);
        SpdStoreIntermediate(x / 2, y / 2 + 8, v[2]);

        SpdStore(FfxInt32x2(workGroupID.xy * 16) + FfxInt32x2(x / 2 + 8, y / 2 + 8), v[3], 1, slice);
        SpdStoreIntermediate(x / 2 + 8, y / 2 + 8, v[3]);
    }
}

void SpdDownsampleMips_0_1_LDS(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mip, FfxUInt32 slice)
{
    FfxFloat32x4 v[4];

    FfxInt32x2 tex = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2, y * 2);
    FfxInt32x2 pix = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x, y);
    v[0]     = SpdReduceLoadSourceImage(tex, slice);
    SpdStore(pix, v[0], 0, slice);

    tex  = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2 + 32, y * 2);
    pix  = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x + 16, y);
    v[1] = SpdReduceLoadSourceImage(tex, slice);
    SpdStore(pix, v[1], 0, slice);

    tex  = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2, y * 2 + 32);
    pix  = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x, y + 16);
    v[2] = SpdReduceLoadSourceImage(tex, slice);
    SpdStore(pix, v[2], 0, slice);

    tex  = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2 + 32, y * 2 + 32);
    pix  = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x + 16, y + 16);
    v[3] = SpdReduceLoadSourceImage(tex, slice);
    SpdStore(pix, v[3], 0, slice);

    if (mip <= 1)
        return;

    for (FfxUInt32 i = 0; i < 4; i++)
    {
        SpdStoreIntermediate(x, y, v[i]);
        SpdWorkgroupShuffleBarrier();
        if (localInvocationIndex < 64)
        {
            v[i] = SpdReduceIntermediate(FfxUInt32x2(x * 2 + 0, y * 2 + 0), FfxUInt32x2(x * 2 + 1, y * 2 + 0), FfxUInt32x2(x * 2 + 0, y * 2 + 1), FfxUInt32x2(x * 2 + 1, y * 2 + 1));
            SpdStore(FfxInt32x2(workGroupID.xy * 16) + FfxInt32x2(x + (i % 2) * 8, y + (i / 2) * 8), v[i], 1, slice);
        }
        SpdWorkgroupShuffleBarrier();
    }

    if (localInvocationIndex < 64)
    {
        SpdStoreIntermediate(x + 0, y + 0, v[0]);
        SpdStoreIntermediate(x + 8, y + 0, v[1]);
        SpdStoreIntermediate(x + 0, y + 8, v[2]);
        SpdStoreIntermediate(x + 8, y + 8, v[3]);
    }
}

void SpdDownsampleMips_0_1(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mip, FfxUInt32 slice)
{
#ifdef SPD_NO_WAVE_OPERATIONS
    SpdDownsampleMips_0_1_LDS(x, y, workGroupID, localInvocationIndex, mip, slice);
#else
    SpdDownsampleMips_0_1_Intrinsics(x, y, workGroupID, localInvocationIndex, mip, slice);
#endif
}


void SpdDownsampleMip_2(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mip, FfxUInt32 slice)
{
#ifdef SPD_NO_WAVE_OPERATIONS
    if (localInvocationIndex < 64)
    {
        FfxFloat32x4 v = SpdReduceIntermediate(FfxUInt32x2(x * 2 + 0, y * 2 + 0), FfxUInt32x2(x * 2 + 1, y * 2 + 0), FfxUInt32x2(x * 2 + 0, y * 2 + 1), FfxUInt32x2(x * 2 + 1, y * 2 + 1));
        SpdStore(FfxInt32x2(workGroupID.xy * 8) + FfxInt32x2(x, y), v, mip, slice);
        // store to LDS, try to reduce bank conflicts
        // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0 x
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
        // ...
        // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
        SpdStoreIntermediate(x * 2 + y % 2, y * 2, v);
    }
#else
    FfxFloat32x4 v = SpdLoadIntermediate(x, y);
    v        = SpdReduceQuad(v);
    // quad index 0 stores result
    if (localInvocationIndex % 4 == 0)
    {
        SpdStore(FfxInt32x2(workGroupID.xy * 8) + FfxInt32x2(x / 2, y / 2), v, mip, slice);
        SpdStoreIntermediate(x + (y / 2) % 2, y, v);
    }
#endif
}

void SpdDownsampleMip_3(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mip, FfxUInt32 slice)
{
#ifdef SPD_NO_WAVE_OPERATIONS
    if (localInvocationIndex < 16)
    {
        // x 0 x 0
        // 0 0 0 0
        // 0 x 0 x
        // 0 0 0 0
        FfxFloat32x4 v =
            SpdReduceIntermediate(FfxUInt32x2(x * 4 + 0 + 0, y * 4 + 0), FfxUInt32x2(x * 4 + 2 + 0, y * 4 + 0), FfxUInt32x2(x * 4 + 0 + 1, y * 4 + 2), FfxUInt32x2(x * 4 + 2 + 1, y * 4 + 2));
        SpdStore(FfxInt32x2(workGroupID.xy * 4) + FfxInt32x2(x, y), v, mip, slice);
        // store to LDS
        // x 0 0 0 x 0 0 0 x 0 0 0 x 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 x 0 0 0 x 0 0 0 x 0 0 0 x 0 0
        // ...
        // 0 0 x 0 0 0 x 0 0 0 x 0 0 0 x 0
        // ...
        // 0 0 0 x 0 0 0 x 0 0 0 x 0 0 0 x
        // ...
        SpdStoreIntermediate(x * 4 + y, y * 4, v);
    }
#else
    if (localInvocationIndex < 64)
    {
        FfxFloat32x4 v = SpdLoadIntermediate(x * 2 + y % 2, y * 2);
        v        = SpdReduceQuad(v);
        // quad index 0 stores result
        if (localInvocationIndex % 4 == 0)
        {
            SpdStore(FfxInt32x2(workGroupID.xy * 4) + FfxInt32x2(x / 2, y / 2), v, mip, slice);
            SpdStoreIntermediate(x * 2 + y / 2, y * 2, v);
        }
    }
#endif
}

void SpdDownsampleMip_4(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mip, FfxUInt32 slice)
{
#ifdef SPD_NO_WAVE_OPERATIONS
    if (localInvocationIndex < 4)
    {
        // x 0 0 0 x 0 0 0
        // ...
        // 0 x 0 0 0 x 0 0
        FfxFloat32x4 v = SpdReduceIntermediate(FfxUInt32x2(x * 8 + 0 + 0 + y * 2, y * 8 + 0),
                                         FfxUInt32x2(x * 8 + 4 + 0 + y * 2, y * 8 + 0),
                                         FfxUInt32x2(x * 8 + 0 + 1 + y * 2, y * 8 + 4),
                                         FfxUInt32x2(x * 8 + 4 + 1 + y * 2, y * 8 + 4));
        SpdStore(FfxInt32x2(workGroupID.xy * 2) + FfxInt32x2(x, y), v, mip, slice);
        // store to LDS
        // x x x x 0 ...
        // 0 ...
        SpdStoreIntermediate(x + y * 2, 0, v);
    }
#else
    if (localInvocationIndex < 16)
    {
        FfxFloat32x4 v = SpdLoadIntermediate(x * 4 + y, y * 4);
        v        = SpdReduceQuad(v);
        // quad index 0 stores result
        if (localInvocationIndex % 4 == 0)
        {
            SpdStore(FfxInt32x2(workGroupID.xy * 2) + FfxInt32x2(x / 2, y / 2), v, mip, slice);
            SpdStoreIntermediate(x / 2 + y, 0, v);
        }
    }
#endif
}

void SpdDownsampleMip_5(FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mip, FfxUInt32 slice)
{
#ifdef SPD_NO_WAVE_OPERATIONS
    if (localInvocationIndex < 1)
    {
        // x x x x 0 ...
        // 0 ...
        FfxFloat32x4 v = SpdReduceIntermediate(FfxUInt32x2(0, 0), FfxUInt32x2(1, 0), FfxUInt32x2(2, 0), FfxUInt32x2(3, 0));
        SpdStore(FfxInt32x2(workGroupID.xy), v, mip, slice);
    }
#else
    if (localInvocationIndex < 4)
    {
        FfxFloat32x4 v = SpdLoadIntermediate(localInvocationIndex, 0);
        v        = SpdReduceQuad(v);
        // quad index 0 stores result
        if (localInvocationIndex % 4 == 0)
        {
            SpdStore(FfxInt32x2(workGroupID.xy), v, mip, slice);
        }
    }
#endif
}

void SpdDownsampleMips_6_7(FfxUInt32 x, FfxUInt32 y, FfxUInt32 mips, FfxUInt32 slice)
{
    FfxInt32x2   tex = FfxInt32x2(x * 4 + 0, y * 4 + 0);
    FfxInt32x2   pix = FfxInt32x2(x * 2 + 0, y * 2 + 0);
    FfxFloat32x4 v0  = SpdReduceLoad4(tex, slice);
    SpdStore(pix, v0, 6, slice);

    tex       = FfxInt32x2(x * 4 + 2, y * 4 + 0);
    pix       = FfxInt32x2(x * 2 + 1, y * 2 + 0);
    FfxFloat32x4 v1 = SpdReduceLoad4(tex, slice);
    SpdStore(pix, v1, 6, slice);

    tex       = FfxInt32x2(x * 4 + 0, y * 4 + 2);
    pix       = FfxInt32x2(x * 2 + 0, y * 2 + 1);
    FfxFloat32x4 v2 = SpdReduceLoad4(tex, slice);
    SpdStore(pix, v2, 6, slice);

    tex       = FfxInt32x2(x * 4 + 2, y * 4 + 2);
    pix       = FfxInt32x2(x * 2 + 1, y * 2 + 1);
    FfxFloat32x4 v3 = SpdReduceLoad4(tex, slice);
    SpdStore(pix, v3, 6, slice);

    if (mips <= 7)
        return;
    // no barrier needed, working on values only from the same thread

    FfxFloat32x4 v = SpdReduce4(v0, v1, v2, v3);
    SpdStore(FfxInt32x2(x, y), v, 7, slice);
    SpdStoreIntermediate(x, y, v);
}

void SpdDownsampleNextFour(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 baseMip, FfxUInt32 mips, FfxUInt32 slice)
{
    if (mips <= baseMip)
        return;
    SpdWorkgroupShuffleBarrier();
    SpdDownsampleMip_2(x, y, workGroupID, localInvocationIndex, baseMip, slice);

    if (mips <= baseMip + 1)
        return;
    SpdWorkgroupShuffleBarrier();
    SpdDownsampleMip_3(x, y, workGroupID, localInvocationIndex, baseMip + 1, slice);

    if (mips <= baseMip + 2)
        return;
    SpdWorkgroupShuffleBarrier();
    SpdDownsampleMip_4(x, y, workGroupID, localInvocationIndex, baseMip + 2, slice);

    if (mips <= baseMip + 3)
        return;
    SpdWorkgroupShuffleBarrier();
    SpdDownsampleMip_5(workGroupID, localInvocationIndex, baseMip + 3, slice);
}

void SpdDownsample(FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mips, FfxUInt32 numWorkGroups, FfxUInt32 slice)
{
    FfxUInt32x2        sub_xy = ffxRemapForWaveReduction(localInvocationIndex % 64);
    FfxUInt32 x      = sub_xy.x + 8 * ((localInvocationIndex >> 6) % 2);
    FfxUInt32 y      = sub_xy.y + 8 * ((localInvocationIndex >> 7));
    SpdDownsampleMips_0_1(x, y, workGroupID, localInvocationIndex, mips, slice);

    SpdDownsampleNextFour(x, y, workGroupID, localInvocationIndex, 2, mips, slice);

    if (mips <= 6)
        return;

    if (SpdExitWorkgroup(numWorkGroups, localInvocationIndex, slice))
        return;

    SpdResetAtomicCounter(slice);

    // After mip 6 there is only a single workgroup left that downsamples the remaining up to 64x64 texels.
    SpdDownsampleMips_6_7(x, y, mips, slice);

    SpdDownsampleNextFour(x, y, FfxUInt32x2(0, 0), localInvocationIndex, 8, mips, slice);
}

void SpdDownsample(FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mips, FfxUInt32 numWorkGroups, FfxUInt32 slice, FfxUInt32x2 workGroupOffset)
{
    SpdDownsample(workGroupID + workGroupOffset, localInvocationIndex, mips, numWorkGroups, slice);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//==============================================================================================================================
//                                                       PACKED VERSION
//==============================================================================================================================

#if FFX_HALF

#ifdef FFX_GLSL
#extension GL_EXT_shader_subgroup_extended_types_float16:require
#endif

FfxFloat16x4 SpdReduceQuadH(FfxFloat16x4 v)
{
#if defined(FFX_GLSL) && !defined(SPD_NO_WAVE_OPERATIONS)
    FfxFloat16x4 v0 = v;
    FfxFloat16x4 v1 = subgroupQuadSwapHorizontal(v);
    FfxFloat16x4 v2 = subgroupQuadSwapVertical(v);
    FfxFloat16x4 v3 = subgroupQuadSwapDiagonal(v);
    return SpdReduce4H(v0, v1, v2, v3);
#elif defined(FFX_HLSL) && !defined(SPD_NO_WAVE_OPERATIONS)
    // requires SM6.0
    FfxUInt32 quad = WaveGetLaneIndex() & (~0x3);
    FfxFloat16x4        v0   = v;
    FfxFloat16x4        v1   = WaveReadLaneAt(v, quad | 1);
    FfxFloat16x4        v2   = WaveReadLaneAt(v, quad | 2);
    FfxFloat16x4        v3   = WaveReadLaneAt(v, quad | 3);
    return SpdReduce4H(v0, v1, v2, v3);
/*
    // if SM6.0 is not available, you can use the AMD shader intrinsics
    // the AMD shader intrinsics are available in AMD GPU Services (AGS) library:
    // https://gpuopen.com/amd-gpu-services-ags-library/
    // works for DX11
    FfxFloat16x4 v0 = v;
    FfxFloat16x4 v1;
    v1.x = AmdExtD3DShaderIntrinsics_SwizzleF(v.x, AmdExtD3DShaderIntrinsicsSwizzle_SwapX1);
    v1.y = AmdExtD3DShaderIntrinsics_SwizzleF(v.y, AmdExtD3DShaderIntrinsicsSwizzle_SwapX1);
    v1.z = AmdExtD3DShaderIntrinsics_SwizzleF(v.z, AmdExtD3DShaderIntrinsicsSwizzle_SwapX1);
    v1.w = AmdExtD3DShaderIntrinsics_SwizzleF(v.w, AmdExtD3DShaderIntrinsicsSwizzle_SwapX1);
    FfxFloat16x4 v2;
    v2.x = AmdExtD3DShaderIntrinsics_SwizzleF(v.x, AmdExtD3DShaderIntrinsicsSwizzle_SwapX2);
    v2.y = AmdExtD3DShaderIntrinsics_SwizzleF(v.y, AmdExtD3DShaderIntrinsicsSwizzle_SwapX2);
    v2.z = AmdExtD3DShaderIntrinsics_SwizzleF(v.z, AmdExtD3DShaderIntrinsicsSwizzle_SwapX2);
    v2.w = AmdExtD3DShaderIntrinsics_SwizzleF(v.w, AmdExtD3DShaderIntrinsicsSwizzle_SwapX2);
    FfxFloat16x4 v3;
    v3.x = AmdExtD3DShaderIntrinsics_SwizzleF(v.x, AmdExtD3DShaderIntrinsicsSwizzle_ReverseX4);
    v3.y = AmdExtD3DShaderIntrinsics_SwizzleF(v.y, AmdExtD3DShaderIntrinsicsSwizzle_ReverseX4);
    v3.z = AmdExtD3DShaderIntrinsics_SwizzleF(v.z, AmdExtD3DShaderIntrinsicsSwizzle_ReverseX4);
    v3.w = AmdExtD3DShaderIntrinsics_SwizzleF(v.w, AmdExtD3DShaderIntrinsicsSwizzle_ReverseX4);
    return SpdReduce4H(v0, v1, v2, v3);
    */
#endif
    return FfxFloat16x4(0.0, 0.0, 0.0, 0.0);
}

FfxFloat16x4 SpdReduceIntermediateH(FfxUInt32x2 i0, FfxUInt32x2 i1, FfxUInt32x2 i2, FfxUInt32x2 i3)
{
    FfxFloat16x4 v0 = SpdLoadIntermediateH(i0.x, i0.y);
    FfxFloat16x4 v1 = SpdLoadIntermediateH(i1.x, i1.y);
    FfxFloat16x4 v2 = SpdLoadIntermediateH(i2.x, i2.y);
    FfxFloat16x4 v3 = SpdLoadIntermediateH(i3.x, i3.y);
    return SpdReduce4H(v0, v1, v2, v3);
}

FfxFloat16x4 SpdReduceLoad4H(FfxUInt32x2 i0, FfxUInt32x2 i1, FfxUInt32x2 i2, FfxUInt32x2 i3, FfxUInt32 slice)
{
    FfxFloat16x4 v0 = SpdLoadH(FfxInt32x2(i0), slice);
    FfxFloat16x4 v1 = SpdLoadH(FfxInt32x2(i1), slice);
    FfxFloat16x4 v2 = SpdLoadH(FfxInt32x2(i2), slice);
    FfxFloat16x4 v3 = SpdLoadH(FfxInt32x2(i3), slice);
    return SpdReduce4H(v0, v1, v2, v3);
}

FfxFloat16x4 SpdReduceLoad4H(FfxUInt32x2 base, FfxUInt32 slice)
{
    return SpdReduceLoad4H(FfxUInt32x2(base + FfxUInt32x2(0, 0)), FfxUInt32x2(base + FfxUInt32x2(0, 1)), FfxUInt32x2(base + FfxUInt32x2(1, 0)), FfxUInt32x2(base + FfxUInt32x2(1, 1)), slice);
}

FfxFloat16x4 SpdReduceLoadSourceImage4H(FfxUInt32x2 i0, FfxUInt32x2 i1, FfxUInt32x2 i2, FfxUInt32x2 i3, FfxUInt32 slice)
{
    FfxFloat16x4 v0 = SpdLoadSourceImageH(FfxInt32x2(i0), slice);
    FfxFloat16x4 v1 = SpdLoadSourceImageH(FfxInt32x2(i1), slice);
    FfxFloat16x4 v2 = SpdLoadSourceImageH(FfxInt32x2(i2), slice);
    FfxFloat16x4 v3 = SpdLoadSourceImageH(FfxInt32x2(i3), slice);
    return SpdReduce4H(v0, v1, v2, v3);
}

FfxFloat16x4 SpdReduceLoadSourceImageH(FfxUInt32x2 base, FfxUInt32 slice)
{
#ifdef SPD_LINEAR_SAMPLER
    return SpdLoadSourceImageH(FfxInt32x2(base), slice);
#else
    return SpdReduceLoadSourceImage4H(FfxUInt32x2(base + FfxUInt32x2(0, 0)), FfxUInt32x2(base + FfxUInt32x2(0, 1)), FfxUInt32x2(base + FfxUInt32x2(1, 0)), FfxUInt32x2(base + FfxUInt32x2(1, 1)), slice);
#endif
}

void SpdDownsampleMips_0_1_IntrinsicsH(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mips, FfxUInt32 slice)
{
    FfxFloat16x4 v[4];

    FfxInt32x2 tex = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2, y * 2);
    FfxInt32x2 pix = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x, y);
    v[0]     = SpdReduceLoadSourceImageH(tex, slice);
    SpdStoreH(pix, v[0], 0, slice);

    tex  = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2 + 32, y * 2);
    pix  = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x + 16, y);
    v[1] = SpdReduceLoadSourceImageH(tex, slice);
    SpdStoreH(pix, v[1], 0, slice);

    tex  = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2, y * 2 + 32);
    pix  = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x, y + 16);
    v[2] = SpdReduceLoadSourceImageH(tex, slice);
    SpdStoreH(pix, v[2], 0, slice);

    tex  = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2 + 32, y * 2 + 32);
    pix  = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x + 16, y + 16);
    v[3] = SpdReduceLoadSourceImageH(tex, slice);
    SpdStoreH(pix, v[3], 0, slice);

    if (mips <= 1)
        return;

    v[0] = SpdReduceQuadH(v[0]);
    v[1] = SpdReduceQuadH(v[1]);
    v[2] = SpdReduceQuadH(v[2]);
    v[3] = SpdReduceQuadH(v[3]);

    if ((localInvocationIndex % 4) == 0)
    {
        SpdStoreH(FfxInt32x2(workGroupID.xy * 16) + FfxInt32x2(x / 2, y / 2), v[0], 1, slice);
        SpdStoreIntermediateH(x / 2, y / 2, v[0]);

        SpdStoreH(FfxInt32x2(workGroupID.xy * 16) + FfxInt32x2(x / 2 + 8, y / 2), v[1], 1, slice);
        SpdStoreIntermediateH(x / 2 + 8, y / 2, v[1]);

        SpdStoreH(FfxInt32x2(workGroupID.xy * 16) + FfxInt32x2(x / 2, y / 2 + 8), v[2], 1, slice);
        SpdStoreIntermediateH(x / 2, y / 2 + 8, v[2]);

        SpdStoreH(FfxInt32x2(workGroupID.xy * 16) + FfxInt32x2(x / 2 + 8, y / 2 + 8), v[3], 1, slice);
        SpdStoreIntermediateH(x / 2 + 8, y / 2 + 8, v[3]);
    }
}

void SpdDownsampleMips_0_1_LDSH(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mips, FfxUInt32 slice)
{
    FfxFloat16x4 v[4];

    FfxInt32x2 tex = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2, y * 2);
    FfxInt32x2 pix = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x, y);
    v[0]     = SpdReduceLoadSourceImageH(tex, slice);
    SpdStoreH(pix, v[0], 0, slice);

    tex  = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2 + 32, y * 2);
    pix  = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x + 16, y);
    v[1] = SpdReduceLoadSourceImageH(tex, slice);
    SpdStoreH(pix, v[1], 0, slice);

    tex  = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2, y * 2 + 32);
    pix  = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x, y + 16);
    v[2] = SpdReduceLoadSourceImageH(tex, slice);
    SpdStoreH(pix, v[2], 0, slice);

    tex  = FfxInt32x2(workGroupID.xy * 64) + FfxInt32x2(x * 2 + 32, y * 2 + 32);
    pix  = FfxInt32x2(workGroupID.xy * 32) + FfxInt32x2(x + 16, y + 16);
    v[3] = SpdReduceLoadSourceImageH(tex, slice);
    SpdStoreH(pix, v[3], 0, slice);

    if (mips <= 1)
        return;

    for (FfxInt32 i = 0; i < 4; i++)
    {
        SpdStoreIntermediateH(x, y, v[i]);
        SpdWorkgroupShuffleBarrier();
        if (localInvocationIndex < 64)
        {
            v[i] = SpdReduceIntermediateH(FfxUInt32x2(x * 2 + 0, y * 2 + 0), FfxUInt32x2(x * 2 + 1, y * 2 + 0), FfxUInt32x2(x * 2 + 0, y * 2 + 1), FfxUInt32x2(x * 2 + 1, y * 2 + 1));
            SpdStoreH(FfxInt32x2(workGroupID.xy * 16) + FfxInt32x2(x + (i % 2) * 8, y + (i / 2) * 8), v[i], 1, slice);
        }
        SpdWorkgroupShuffleBarrier();
    }

    if (localInvocationIndex < 64)
    {
        SpdStoreIntermediateH(x + 0, y + 0, v[0]);
        SpdStoreIntermediateH(x + 8, y + 0, v[1]);
        SpdStoreIntermediateH(x + 0, y + 8, v[2]);
        SpdStoreIntermediateH(x + 8, y + 8, v[3]);
    }
}

void SpdDownsampleMips_0_1H(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mips, FfxUInt32 slice)
{
#ifdef SPD_NO_WAVE_OPERATIONS
    SpdDownsampleMips_0_1_LDSH(x, y, workGroupID, localInvocationIndex, mips, slice);
#else
    SpdDownsampleMips_0_1_IntrinsicsH(x, y, workGroupID, localInvocationIndex, mips, slice);
#endif
}


void SpdDownsampleMip_2H(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mip, FfxUInt32 slice)
{
#ifdef SPD_NO_WAVE_OPERATIONS
    if (localInvocationIndex < 64)
    {
        FfxFloat16x4 v = SpdReduceIntermediateH(FfxUInt32x2(x * 2 + 0, y * 2 + 0), FfxUInt32x2(x * 2 + 1, y * 2 + 0), FfxUInt32x2(x * 2 + 0, y * 2 + 1), FfxUInt32x2(x * 2 + 1, y * 2 + 1));
        SpdStoreH(FfxInt32x2(workGroupID.xy * 8) + FfxInt32x2(x, y), v, mip, slice);
        // store to LDS, try to reduce bank conflicts
        // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0 x
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
        // ...
        // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
        SpdStoreIntermediateH(x * 2 + y % 2, y * 2, v);
    }
#else
    FfxFloat16x4 v = SpdLoadIntermediateH(x, y);
    v     = SpdReduceQuadH(v);
    // quad index 0 stores result
    if (localInvocationIndex % 4 == 0)
    {
        SpdStoreH(FfxInt32x2(workGroupID.xy * 8) + FfxInt32x2(x / 2, y / 2), v, mip, slice);
        SpdStoreIntermediateH(x + (y / 2) % 2, y, v);
    }
#endif
}

void SpdDownsampleMip_3H(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mip, FfxUInt32 slice)
{
#ifdef SPD_NO_WAVE_OPERATIONS
    if (localInvocationIndex < 16)
    {
        // x 0 x 0
        // 0 0 0 0
        // 0 x 0 x
        // 0 0 0 0
        FfxFloat16x4 v =
            SpdReduceIntermediateH(FfxUInt32x2(x * 4 + 0 + 0, y * 4 + 0), FfxUInt32x2(x * 4 + 2 + 0, y * 4 + 0), FfxUInt32x2(x * 4 + 0 + 1, y * 4 + 2), FfxUInt32x2(x * 4 + 2 + 1, y * 4 + 2));
        SpdStoreH(FfxInt32x2(workGroupID.xy * 4) + FfxInt32x2(x, y), v, mip, slice);
        // store to LDS
        // x 0 0 0 x 0 0 0 x 0 0 0 x 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 x 0 0 0 x 0 0 0 x 0 0 0 x 0 0
        // ...
        // 0 0 x 0 0 0 x 0 0 0 x 0 0 0 x 0
        // ...
        // 0 0 0 x 0 0 0 x 0 0 0 x 0 0 0 x
        // ...
        SpdStoreIntermediateH(x * 4 + y, y * 4, v);
    }
#else
    if (localInvocationIndex < 64)
    {
        FfxFloat16x4 v = SpdLoadIntermediateH(x * 2 + y % 2, y * 2);
        v     = SpdReduceQuadH(v);
        // quad index 0 stores result
        if (localInvocationIndex % 4 == 0)
        {
            SpdStoreH(FfxInt32x2(workGroupID.xy * 4) + FfxInt32x2(x / 2, y / 2), v, mip, slice);
            SpdStoreIntermediateH(x * 2 + y / 2, y * 2, v);
        }
    }
#endif
}

void SpdDownsampleMip_4H(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mip, FfxUInt32 slice)
{
#ifdef SPD_NO_WAVE_OPERATIONS
    if (localInvocationIndex < 4)
    {
        // x 0 0 0 x 0 0 0
        // ...
        // 0 x 0 0 0 x 0 0
        FfxFloat16x4 v = SpdReduceIntermediateH(FfxUInt32x2(x * 8 + 0 + 0 + y * 2, y * 8 + 0),
                                       FfxUInt32x2(x * 8 + 4 + 0 + y * 2, y * 8 + 0),
                                       FfxUInt32x2(x * 8 + 0 + 1 + y * 2, y * 8 + 4),
                                       FfxUInt32x2(x * 8 + 4 + 1 + y * 2, y * 8 + 4));
        SpdStoreH(FfxInt32x2(workGroupID.xy * 2) + FfxInt32x2(x, y), v, mip, slice);
        // store to LDS
        // x x x x 0 ...
        // 0 ...
        SpdStoreIntermediateH(x + y * 2, 0, v);
    }
#else
    if (localInvocationIndex < 16)
    {
        FfxFloat16x4 v = SpdLoadIntermediateH(x * 4 + y, y * 4);
        v     = SpdReduceQuadH(v);
        // quad index 0 stores result
        if (localInvocationIndex % 4 == 0)
        {
            SpdStoreH(FfxInt32x2(workGroupID.xy * 2) + FfxInt32x2(x / 2, y / 2), v, mip, slice);
            SpdStoreIntermediateH(x / 2 + y, 0, v);
        }
    }
#endif
}

void SpdDownsampleMip_5H(FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mip, FfxUInt32 slice)
{
#ifdef SPD_NO_WAVE_OPERATIONS
    if (localInvocationIndex < 1)
    {
        // x x x x 0 ...
        // 0 ...
        FfxFloat16x4 v = SpdReduceIntermediateH(FfxUInt32x2(0, 0), FfxUInt32x2(1, 0), FfxUInt32x2(2, 0), FfxUInt32x2(3, 0));
        SpdStoreH(FfxInt32x2(workGroupID.xy), v, mip, slice);
    }
#else
    if (localInvocationIndex < 4)
    {
        FfxFloat16x4 v = SpdLoadIntermediateH(localInvocationIndex, 0);
        v     = SpdReduceQuadH(v);
        // quad index 0 stores result
        if (localInvocationIndex % 4 == 0)
        {
            SpdStoreH(FfxInt32x2(workGroupID.xy), v, mip, slice);
        }
    }
#endif
}

void SpdDownsampleMips_6_7H(FfxUInt32 x, FfxUInt32 y, FfxUInt32 mips, FfxUInt32 slice)
{
    FfxInt32x2 tex = FfxInt32x2(x * 4 + 0, y * 4 + 0);
    FfxInt32x2 pix = FfxInt32x2(x * 2 + 0, y * 2 + 0);
    FfxFloat16x4  v0  = SpdReduceLoad4H(tex, slice);
    SpdStoreH(pix, v0, 6, slice);

    tex    = FfxInt32x2(x * 4 + 2, y * 4 + 0);
    pix    = FfxInt32x2(x * 2 + 1, y * 2 + 0);
    FfxFloat16x4 v1 = SpdReduceLoad4H(tex, slice);
    SpdStoreH(pix, v1, 6, slice);

    tex    = FfxInt32x2(x * 4 + 0, y * 4 + 2);
    pix    = FfxInt32x2(x * 2 + 0, y * 2 + 1);
    FfxFloat16x4 v2 = SpdReduceLoad4H(tex, slice);
    SpdStoreH(pix, v2, 6, slice);

    tex    = FfxInt32x2(x * 4 + 2, y * 4 + 2);
    pix    = FfxInt32x2(x * 2 + 1, y * 2 + 1);
    FfxFloat16x4 v3 = SpdReduceLoad4H(tex, slice);
    SpdStoreH(pix, v3, 6, slice);

    if (mips < 8)
        return;
    // no barrier needed, working on values only from the same thread

    FfxFloat16x4 v = SpdReduce4H(v0, v1, v2, v3);
    SpdStoreH(FfxInt32x2(x, y), v, 7, slice);
    SpdStoreIntermediateH(x, y, v);
}

void SpdDownsampleNextFourH(FfxUInt32 x, FfxUInt32 y, FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 baseMip, FfxUInt32 mips, FfxUInt32 slice)
{
    if (mips <= baseMip)
        return;
    SpdWorkgroupShuffleBarrier();
    SpdDownsampleMip_2H(x, y, workGroupID, localInvocationIndex, baseMip, slice);

    if (mips <= baseMip + 1)
        return;
    SpdWorkgroupShuffleBarrier();
    SpdDownsampleMip_3H(x, y, workGroupID, localInvocationIndex, baseMip + 1, slice);

    if (mips <= baseMip + 2)
        return;
    SpdWorkgroupShuffleBarrier();
    SpdDownsampleMip_4H(x, y, workGroupID, localInvocationIndex, baseMip + 2, slice);

    if (mips <= baseMip + 3)
        return;
    SpdWorkgroupShuffleBarrier();
    SpdDownsampleMip_5H(workGroupID, localInvocationIndex, baseMip + 3, slice);
}

void SpdDownsampleH(FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mips, FfxUInt32 numWorkGroups, FfxUInt32 slice)
{
    FfxUInt32x2        sub_xy = ffxRemapForWaveReduction(localInvocationIndex % 64);
    FfxUInt32 x      = sub_xy.x + 8 * ((localInvocationIndex >> 6) % 2);
    FfxUInt32 y      = sub_xy.y + 8 * ((localInvocationIndex >> 7));

    SpdDownsampleMips_0_1H(x, y, workGroupID, localInvocationIndex, mips, slice);

    SpdDownsampleNextFourH(x, y, workGroupID, localInvocationIndex, 2, mips, slice);

    if (mips < 7)
        return;

    if (SpdExitWorkgroup(numWorkGroups, localInvocationIndex, slice))
        return;

    SpdResetAtomicCounter(slice);

    // After mip 6 there is only a single workgroup left that downsamples the remaining up to 64x64 texels.
    SpdDownsampleMips_6_7H(x, y, mips, slice);

    SpdDownsampleNextFourH(x, y, FfxUInt32x2(0, 0), localInvocationIndex, 8, mips, slice);
}

void SpdDownsampleH(FfxUInt32x2 workGroupID, FfxUInt32 localInvocationIndex, FfxUInt32 mips, FfxUInt32 numWorkGroups, FfxUInt32 slice, FfxUInt32x2 workGroupOffset)
{
    SpdDownsampleH(workGroupID + workGroupOffset, localInvocationIndex, mips, numWorkGroups, slice);
}

#endif // #if FFX_HALF
#endif // #ifdef FFX_GPU
