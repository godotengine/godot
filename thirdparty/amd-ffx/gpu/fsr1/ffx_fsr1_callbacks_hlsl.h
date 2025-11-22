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

#include "ffx_fsr1_resources.h"

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

#define DECLARE_SRV_REGISTER(regIndex)  t##regIndex
#define DECLARE_UAV_REGISTER(regIndex)  u##regIndex
#define DECLARE_CB_REGISTER(regIndex)   b##regIndex
#define FFX_FSR1_DECLARE_SRV(regIndex)  register(DECLARE_SRV_REGISTER(regIndex))
#define FFX_FSR1_DECLARE_UAV(regIndex)  register(DECLARE_UAV_REGISTER(regIndex))
#define FFX_FSR1_DECLARE_CB(regIndex)   register(DECLARE_CB_REGISTER(regIndex))

#if defined(FSR1_BIND_CB_FSR1)
    cbuffer cbFSR1 : FFX_FSR1_DECLARE_CB(FSR1_BIND_CB_FSR1)
    {
        FfxUInt32x4 const0;
        FfxUInt32x4 const1;
        FfxUInt32x4 const2;
        FfxUInt32x4 const3;
        FfxUInt32x4 sample;
       #define FFX_FSR1_CONSTANT_BUFFER_1_SIZE 20  // Number of 32-bit values. This must be kept in sync with the cbFSR1 size.
    };
#else
    #define const0 0
    #define const1 0
    #define const2 0
    #define const3 0
    #define sample 0
#endif

#if defined(FFX_GPU)
#define FFX_FSR1_ROOTSIG_STRINGIFY(p) FFX_FSR1_ROOTSIG_STR(p)
#define FFX_FSR1_ROOTSIG_STR(p) #p
#define FFX_FSR1_ROOTSIG [RootSignature( "DescriptorTable(UAV(u0, numDescriptors = " FFX_FSR1_ROOTSIG_STRINGIFY(FFX_FSR1_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "DescriptorTable(SRV(t0, numDescriptors = " FFX_FSR1_ROOTSIG_STRINGIFY(FFX_FSR1_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "CBV(b0), " \
                                    "StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, " \
                                                      "addressU = TEXTURE_ADDRESS_CLAMP, " \
                                                      "addressV = TEXTURE_ADDRESS_CLAMP, " \
                                                      "addressW = TEXTURE_ADDRESS_CLAMP, " \
                                                      "comparisonFunc = COMPARISON_NEVER, " \
                                                      "borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK)" )]

#if defined(FFX_FSR1_EMBED_ROOTSIG)
#define FFX_FSR1_EMBED_ROOTSIG_CONTENT FFX_FSR1_ROOTSIG
#else
#define FFX_FSR1_EMBED_ROOTSIG_CONTENT
#endif // #if FFX_FSR1_EMBED_ROOTSIG
#endif // #if defined(FFX_GPU)


FfxUInt32x4 Const0()
{
    return const0;
}

FfxUInt32x4 Const1()
{
    return const1;
}

FfxUInt32x4  Const2()
{
    return const2;
}

FfxUInt32x4 Const3()
{
    return const3;
}

FfxUInt32x4 EASUSample()
{
    return sample;
}

FfxUInt32x4 RCasSample()
{
    return sample;
}

FfxUInt32x4 RCasConfig()
{
    return const0;
}

SamplerState s_LinearClamp : register(s0);

    // SRVs
    #if defined FSR1_BIND_SRV_INPUT_COLOR
        Texture2D<FfxFloat32x4>                   r_input_color                 : FFX_FSR1_DECLARE_SRV(FSR1_BIND_SRV_INPUT_COLOR);
    #endif
    #if defined FSR1_BIND_SRV_INTERNAL_UPSCALED_COLOR
        Texture2D<FfxFloat32x4>                   r_internal_upscaled_color     : FFX_FSR1_DECLARE_SRV(FSR1_BIND_SRV_INTERNAL_UPSCALED_COLOR);
    #endif
    #if defined FSR1_BIND_SRV_UPSCALED_OUTPUT
        Texture2D<FfxFloat32x4>                   r_upscaled_output             : FFX_FSR1_DECLARE_SRV(FSR1_BIND_SRV_UPSCALED_OUTPUT);
    #endif

    // UAV declarations
    #if defined FSR1_BIND_UAV_INPUT_COLOR
        RWTexture2D<FfxFloat32x4>                    rw_input_color             : FFX_FSR1_DECLARE_UAV(FSR1_BIND_UAV_INPUT_COLOR);
    #endif
    #if defined FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR
        RWTexture2D<FfxFloat32x4>                   rw_internal_upscaled_color  : FFX_FSR1_DECLARE_UAV(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR);
    #endif
    #if defined FSR1_BIND_UAV_UPSCALED_OUTPUT
        RWTexture2D<FfxFloat32x4>                   rw_upscaled_output          : FFX_FSR1_DECLARE_UAV(FSR1_BIND_UAV_UPSCALED_OUTPUT);
    #endif

#if FFX_HALF

#if defined(FSR1_BIND_SRV_INPUT_COLOR)
        FfxFloat16x4 GatherEasuRed(FfxFloat32x2 fPxPos)
        {
            return (FfxFloat16x4)r_input_color.GatherRed(s_LinearClamp, fPxPos, FfxInt32x2(0,0));
        }
#endif // defined(FSR1_BIND_SRV_INPUT_COLOR)

#if defined(FSR1_BIND_SRV_INPUT_COLOR)
        FfxFloat16x4 GatherEasuGreen(FfxFloat32x2 fPxPos)
        {
            return (FfxFloat16x4)r_input_color.GatherGreen(s_LinearClamp, fPxPos, FfxInt32x2(0, 0));
        }
#endif // defined(FSR1_BIND_SRV_INPUT_COLOR)

#if defined(FSR1_BIND_SRV_INPUT_COLOR)
        FfxFloat16x4 GatherEasuBlue(FfxFloat32x2 fPxPos)
        {
            return (FfxFloat16x4)r_input_color.GatherBlue(s_LinearClamp, fPxPos, FfxInt32x2(0, 0));
        }
#endif // defined(FSR1_BIND_SRV_INPUT_COLOR)

#if FFX_FSR1_OPTION_APPLY_RCAS
    #if defined(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR)
        void StoreEASUOutput(FfxUInt32x2 iPxPos, FfxFloat16x3 fColor)
        {
            rw_internal_upscaled_color[iPxPos] = FfxFloat32x4(fColor, 1.f);
        }
    #endif // #if defined(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR)
#else
    #if defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)
        void StoreEASUOutput(FfxUInt32x2 iPxPos, FfxFloat16x3 fColor)
        {
            rw_upscaled_output[iPxPos] = FfxFloat32x4(fColor, 1.f);
        }
    #endif // #if defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)
#endif // #if FFX_FSR1_OPTION_APPLY_RCAS

#if defined(FSR1_BIND_SRV_INTERNAL_UPSCALED_COLOR)
        FfxFloat16x4 LoadRCas_Input(FfxInt16x2 iPxPos)
        {
            return (FfxFloat16x4)r_internal_upscaled_color[iPxPos];
        }
#endif // defined(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR)

#if defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)
        void StoreRCasOutputHx2(FfxInt16x2 iPxPos, FfxFloat16x2 fColorR, FfxFloat16x2 fColorG, FfxFloat16x2 fColorB, FfxFloat16x2 fColorA)
        {
            rw_upscaled_output[iPxPos] = FfxFloat32x4(fColorR.x, fColorG.x, fColorB.x, fColorA.x);
            iPxPos.x += 8;
            rw_upscaled_output[iPxPos] = FfxFloat32x4(fColorR.y, fColorG.y, fColorB.y, fColorA.y);
        }
#endif // defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)

#else // FFX_HALF

#if defined(FSR1_BIND_SRV_INPUT_COLOR)
        FfxFloat32x4 GatherEasuRed(FfxFloat32x2 fPxPos)
        {
            return r_input_color.GatherRed(s_LinearClamp, fPxPos, FfxInt32x2(0, 0));
        }
#endif // defined(FSR1_BIND_SRV_INPUT_COLOR)

#if defined(FSR1_BIND_SRV_INPUT_COLOR)
        FfxFloat32x4 GatherEasuGreen(FfxFloat32x2 fPxPos)
        {
            return r_input_color.GatherGreen(s_LinearClamp, fPxPos, FfxInt32x2(0, 0));
        }
#endif // defined(FSR1_BIND_SRV_INPUT_COLOR)

#if defined(FSR1_BIND_SRV_INPUT_COLOR)
        FfxFloat32x4 GatherEasuBlue(FfxFloat32x2 fPxPos)
        {
            return r_input_color.GatherBlue(s_LinearClamp, fPxPos, FfxInt32x2(0, 0));
        }
#endif // defined(FSR1_BIND_SRV_INPUT_COLOR)


#if FFX_FSR1_OPTION_APPLY_RCAS
    #if defined(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR)        
        void StoreEASUOutput(FfxUInt32x2 iPxPos, FfxFloat32x3 fColor)
        {
            rw_internal_upscaled_color[iPxPos] = FfxFloat32x4(fColor, 1.f);
        }
    #endif // #if defined(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR)
#else
    #if defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)
        void StoreEASUOutput(FfxUInt32x2 iPxPos, FfxFloat32x3 fColor)
        {
            rw_upscaled_output[iPxPos] = FfxFloat32x4(fColor, 1.f);
        }
    #endif // #if defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)
#endif // #if FFX_FSR1_OPTION_APPLY_RCAS

#if defined(FSR1_BIND_SRV_INTERNAL_UPSCALED_COLOR)
        FfxFloat32x4 LoadRCas_Input(FfxInt32x2 iPxPos)
        {
            return r_internal_upscaled_color[iPxPos];
        }
#endif // defined(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR)

#if defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)
        void StoreRCasOutput(FfxInt32x2 iPxPos, FfxFloat32x4 fColor)
        {
            rw_upscaled_output[iPxPos] = fColor;
        }
#endif // defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)

#endif // FFX_HALF

#endif // #if defined(FFX_GPU)
