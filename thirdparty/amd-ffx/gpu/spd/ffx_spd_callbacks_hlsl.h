// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
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

#include "ffx_spd_resources.h"

#if defined(FFX_GPU)
#ifdef __hlsl_dx_compiler
#pragma dxc diagnostic push
#pragma dxc diagnostic ignored "-Wambig-lit-shift"
#endif //__hlsl_dx_compiler
#include "ffx_core.h"
#ifdef __hlsl_dx_compiler
#pragma dxc diagnostic pop
#endif //__hlsl_dx_compiler

#ifndef FFX_PREFER_WAVE64
#define FFX_PREFER_WAVE64
#endif // #ifndef FFX_PREFER_WAVE64

#pragma warning(disable: 3205)  // conversion from larger type to smaller

#define FFX_DECLARE_SRV_REGISTER(regIndex)  t##regIndex
#define FFX_DECLARE_UAV_REGISTER(regIndex)  u##regIndex
#define FFX_DECLARE_CB_REGISTER(regIndex)   b##regIndex
#define FFX_SPD_DECLARE_SRV(regIndex)  register(FFX_DECLARE_SRV_REGISTER(regIndex))
#define FFX_SPD_DECLARE_UAV(regIndex)  register(FFX_DECLARE_UAV_REGISTER(regIndex))
#define FFX_SPD_DECLARE_CB(regIndex)   register(FFX_DECLARE_CB_REGISTER(regIndex))

#if defined(FFX_SPD_BIND_CB_SPD)
    cbuffer cbSPD : FFX_SPD_DECLARE_CB(FFX_SPD_BIND_CB_SPD)
    {
        FfxUInt32       mips;
        FfxUInt32       numWorkGroups;
        FfxUInt32x2     workGroupOffset;
        FfxFloat32x2    invInputSize;       // Only used for linear sampling mode
        FfxFloat32x2    padding;

       #define FFX_SPD_CONSTANT_BUFFER_1_SIZE 8  // Number of 32-bit values. This must be kept in sync with the cbSPD size.
    };
#else
    #define mips 0
    #define numWorkGroups 0
    #define workGroupOffset 0
    #define invInputSize 0
    #define padding 0
#endif

#define FFX_SPD_ROOTSIG_STRINGIFY(p) FFX_SPD_ROOTSIG_STR(p)
#define FFX_SPD_ROOTSIG_STR(p) #p
#define FFX_SPD_ROOTSIG [RootSignature( "DescriptorTable(UAV(u0, numDescriptors = " FFX_SPD_ROOTSIG_STRINGIFY(FFX_SPD_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "DescriptorTable(SRV(t0, numDescriptors = " FFX_SPD_ROOTSIG_STRINGIFY(FFX_SPD_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "CBV(b0), " \
                                    "StaticSampler(s0, filter = FILTER_MIN_MAG_LINEAR_MIP_POINT, " \
                                                      "addressU = TEXTURE_ADDRESS_CLAMP, " \
                                                      "addressV = TEXTURE_ADDRESS_CLAMP, " \
                                                      "addressW = TEXTURE_ADDRESS_CLAMP, " \
                                                      "comparisonFunc = COMPARISON_NEVER, " \
                                                      "borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK)" )]

#if defined(FFX_SPD_EMBED_ROOTSIG)
#define FFX_SPD_EMBED_ROOTSIG_CONTENT FFX_SPD_ROOTSIG
#else
#define FFX_SPD_EMBED_ROOTSIG_CONTENT
#endif // #if FFX_SPD_EMBED_ROOTSIG

FfxUInt32 Mips()
{
    return mips;
}

FfxUInt32 NumWorkGroups()
{
    return numWorkGroups;
}

FfxUInt32x2  WorkGroupOffset()
{
    return workGroupOffset;
}

FfxFloat32x2 InvInputSize()
{
    return invInputSize;
}

SamplerState s_LinearClamp : register(s0);

    // SRVs
    #if defined(FFX_SPD_BIND_SRV_INPUT_DOWNSAMPLE_SRC)
        Texture2DArray<FfxFloat32x4>                                r_input_downsample_src          : FFX_SPD_DECLARE_SRV(FFX_SPD_BIND_SRV_INPUT_DOWNSAMPLE_SRC);
    #endif

    // UAV declarations
    #if defined(FFX_SPD_BIND_UAV_INTERNAL_GLOBAL_ATOMIC)
        struct SpdGlobalAtomicBuffer { FfxUInt32 counter[6]; };
        globallycoherent RWStructuredBuffer<SpdGlobalAtomicBuffer>  rw_internal_global_atomic       : FFX_SPD_DECLARE_UAV(FFX_SPD_BIND_UAV_INTERNAL_GLOBAL_ATOMIC);
    #endif
    #if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)
        globallycoherent RWTexture2DArray<FfxFloat32x4>             rw_input_downsample_src_mid_mip : FFX_SPD_DECLARE_UAV(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP);
    #endif
    #if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)
        RWTexture2DArray<FfxFloat32x4>                              rw_input_downsample_src_mips[SPD_MAX_MIP_LEVELS+1] : FFX_SPD_DECLARE_UAV(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS);
    #endif

#if FFX_HALF

#if defined(FFX_SPD_BIND_SRV_INPUT_DOWNSAMPLE_SRC)
    FfxFloat16x4 SampleSrcImageH(FfxFloat32x2 uv, FfxUInt32 slice)
    {
        FfxFloat32x2 textureCoord = FfxFloat32x2(uv) * InvInputSize() + InvInputSize();
        FfxFloat32x4 result = r_input_downsample_src.SampleLevel(s_LinearClamp, FfxFloat32x3(textureCoord, slice), 0);
        return FfxFloat16x4(ffxSrgbFromLinear(result.x), ffxSrgbFromLinear(result.y), ffxSrgbFromLinear(result.z), result.w);
    }
    #endif // defined(FFX_SPD_BIND_SRV_INPUT_DOWNSAMPLE_SRC) 

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)
    FfxFloat16x4 LoadSrcImageH(FfxFloat32x2 uv, FfxUInt32 slice)
    {
        return FfxFloat16x4(rw_input_downsample_src_mips[0][FfxUInt32x3(uv, slice)]);
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS) 

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)
    void StoreSrcMipH(FfxFloat16x4 value, FfxInt32x2 uv, FfxUInt32 slice, FfxUInt32 mip)
    {
        rw_input_downsample_src_mips[mip][FfxUInt32x3(uv, slice)] = FfxFloat32x4(value);
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS) 

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)
    FfxFloat16x4 LoadMidMipH(FfxInt32x2 uv, FfxUInt32 slice)
    {
        return FfxFloat16x4(rw_input_downsample_src_mid_mip[FfxUInt32x3(uv, slice)]);
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP) 

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)
    void StoreMidMipH(FfxFloat16x4 value, FfxInt32x2 uv, FfxUInt32 slice)
    {
        rw_input_downsample_src_mid_mip[FfxUInt32x3(uv, slice)] = FfxFloat32x4(value);
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)

#else // FFX_HALF

#if defined(FFX_SPD_BIND_SRV_INPUT_DOWNSAMPLE_SRC)
    FfxFloat32x4 SampleSrcImage(FfxInt32x2 uv, FfxUInt32 slice)
    {
        FfxFloat32x2 textureCoord = FfxFloat32x2(uv) * InvInputSize() + InvInputSize();
        FfxFloat32x4 result = r_input_downsample_src.SampleLevel(s_LinearClamp, FfxFloat32x3(textureCoord, slice), 0);
        return FfxFloat32x4(ffxSrgbFromLinear(result.x), ffxSrgbFromLinear(result.y), ffxSrgbFromLinear(result.z), result.w);
    }
#endif // defined(FFX_SPD_BIND_SRV_INPUT_DOWNSAMPLE_SRC)

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)
    FfxFloat32x4 LoadSrcImage(FfxInt32x2 uv, FfxUInt32 slice)
    {
        return rw_input_downsample_src_mips[0][FfxUInt32x3(uv, slice)];
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS) 

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)
    void StoreSrcMip(FfxFloat32x4 value, FfxInt32x2 uv, FfxUInt32 slice, FfxUInt32 mip)
    {
        rw_input_downsample_src_mips[mip][FfxUInt32x3(uv, slice)] = value;
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)
    FfxFloat32x4 LoadMidMip(FfxInt32x2 uv, FfxUInt32 slice)
    { 
        return rw_input_downsample_src_mid_mip[FfxUInt32x3(uv, slice)];
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP) 

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)
    void StoreMidMip(FfxFloat32x4 value, FfxInt32x2 uv, FfxUInt32 slice)
    {
        rw_input_downsample_src_mid_mip[FfxUInt32x3(uv, slice)] = value;
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)

#endif // FFX_HALF

#if defined(FFX_SPD_BIND_UAV_INTERNAL_GLOBAL_ATOMIC)
void IncreaseAtomicCounter(FFX_PARAMETER_IN FfxUInt32 slice, FFX_PARAMETER_INOUT FfxUInt32 counter)
{
    InterlockedAdd(rw_internal_global_atomic[0].counter[slice], 1, counter);
}
#endif // defined(FFX_SPD_BIND_UAV_INTERNAL_GLOBAL_ATOMIC)

#if defined(FFX_SPD_BIND_UAV_INTERNAL_GLOBAL_ATOMIC)
void ResetAtomicCounter(FFX_PARAMETER_IN FfxUInt32 slice)
{
    rw_internal_global_atomic[0].counter[slice] = 0;
}
#endif // defined(FFX_SPD_BIND_UAV_INTERNAL_GLOBAL_ATOMIC)

#endif // #if defined(FFX_GPU)
