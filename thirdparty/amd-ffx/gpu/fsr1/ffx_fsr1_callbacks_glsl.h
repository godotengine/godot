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
#include "ffx_core.h"

#ifndef FFX_PREFER_WAVE64
#define FFX_PREFER_WAVE64
#endif // #ifndef FFX_PREFER_WAVE64

#if defined(FSR1_BIND_CB_FSR1)
    layout (set = 0, binding = FSR1_BIND_CB_FSR1, std140) uniform cbFSR1_t
    {
        FfxUInt32x4 const0;
        FfxUInt32x4 const1;
        FfxUInt32x4 const2;
        FfxUInt32x4 const3;
        FfxUInt32x4 sample0;
    } cbFSR1;
#endif

FfxUInt32x4 Const0()
{
    return cbFSR1.const0;
}

FfxUInt32x4 Const1()
{
    return cbFSR1.const1;
}

FfxUInt32x4 Const2()
{
    return cbFSR1.const2;
}

FfxUInt32x4 Const3()
{
    return cbFSR1.const3;
}

FfxUInt32x4 EASUSample()
{
    return cbFSR1.sample0;
}

FfxUInt32x4 RCasSample()
{
    return cbFSR1.sample0;
}

FfxUInt32x4 RCasConfig()
{
    return cbFSR1.const0;
}

layout (set = 0, binding = 1000) uniform sampler s_LinearClamp;


// SRVs
#if defined FSR1_BIND_SRV_INPUT_COLOR
    layout (set = 0, binding = FSR1_BIND_SRV_INPUT_COLOR)              uniform texture2D  r_input_color;
#endif
#if defined FSR1_BIND_SRV_INTERNAL_UPSCALED_COLOR
    layout (set = 0, binding = FSR1_BIND_SRV_INTERNAL_UPSCALED_COLOR)  uniform texture2D  r_internal_upscaled_color;
#endif
#if defined FSR1_BIND_SRV_UPSCALED_OUTPUT
    layout (set = 0, binding = FSR1_BIND_SRV_UPSCALED_OUTPUT)          uniform texture2D  r_upscaled_output;
#endif 

// UAV declarations
#if defined FSR1_BIND_UAV_INPUT_COLOR
    layout (set = 0, binding = FSR1_BIND_UAV_INPUT_COLOR, rgba32f)              uniform image2D  rw_input_color;
#endif
#if defined FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR
    layout (set = 0, binding = FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR, rgba32f)  uniform image2D  rw_internal_upscaled_color;
#endif
#if defined FSR1_BIND_UAV_UPSCALED_OUTPUT
    layout (set = 0, binding = FSR1_BIND_UAV_UPSCALED_OUTPUT, rgba32f)          uniform image2D  rw_upscaled_output;
#endif

#if FFX_HALF

#if defined(FSR1_BIND_SRV_INPUT_COLOR)
        FfxFloat16x4 GatherEasuRed(FfxFloat32x2 fPxPos)
        {
            return FfxFloat16x4(textureGather(sampler2D(r_input_color, s_LinearClamp), fPxPos, 0));
        }
#endif // defined(FSR1_BIND_SRV_INPUT_COLOR)

#if defined(FSR1_BIND_SRV_INPUT_COLOR)
        FfxFloat16x4 GatherEasuGreen(FfxFloat32x2 fPxPos)
        {
            return FfxFloat16x4(textureGather(sampler2D(r_input_color, s_LinearClamp), fPxPos, 1));
        }
#endif // defined(FSR1_BIND_SRV_INPUT_COLOR)

#if defined(FSR1_BIND_SRV_INPUT_COLOR)
        FfxFloat16x4 GatherEasuBlue(FfxFloat32x2 fPxPos)
        {
            return FfxFloat16x4(textureGather(sampler2D(r_input_color, s_LinearClamp), fPxPos, 2));
        }
#endif // defined(FSR1_BIND_SRV_INPUT_COLOR)

#if FFX_FSR1_OPTION_APPLY_RCAS
    #if defined(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR)
        void StoreEASUOutput(FfxUInt32x2 iPxPos, FfxFloat16x3 fColor)
        {
            imageStore(rw_internal_upscaled_color, FfxInt32x2(iPxPos), FfxFloat32x4(fColor, 1.f));
        }
    #endif  // #if defined(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR)
#else
    #if defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)
    void StoreEASUOutput(FfxUInt32x2 iPxPos, FfxFloat16x3 fColor)
    {
        imageStore(rw_upscaled_output, FfxInt32x2(iPxPos), FfxFloat32x4(fColor, 1.f));
    }
    #endif  // #if defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)
#endif // #if FFX_FSR1_OPTION_APPLY_RCAS

#if defined(FSR1_BIND_SRV_INTERNAL_UPSCALED_COLOR)
        FfxFloat16x4 LoadRCas_Input(FfxInt16x2 iPxPos)
        {
            return FfxFloat16x4(texelFetch(r_internal_upscaled_color, FfxInt32x2(iPxPos), 0));
        }
#endif // defined(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR)

#if defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)
        void StoreRCasOutputHx2(FfxInt16x2 iPxPos, FfxFloat16x2 fColorR, FfxFloat16x2 fColorG, FfxFloat16x2 fColorB, FfxFloat16x2 fColorA)
        {
            imageStore(rw_upscaled_output, FfxInt32x2(iPxPos), FfxFloat32x4(fColorR.x, fColorG.x, fColorB.x, fColorA.x));
            iPxPos.x += FfxInt16(8);
            imageStore(rw_upscaled_output, FfxInt32x2(iPxPos), FfxFloat32x4(fColorR.y, fColorG.y, fColorB.y, fColorA.y));
        }

#endif // defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)

#else // FFX_HALF

#if defined(FSR1_BIND_SRV_INPUT_COLOR)
        FfxFloat32x4 GatherEasuRed(FfxFloat32x2 fPxPos)
        {
            return textureGather(sampler2D(r_input_color, s_LinearClamp), fPxPos, 0);
        }
#endif // defined(FSR1_BIND_SRV_INPUT_COLOR)

#if defined(FSR1_BIND_SRV_INPUT_COLOR)
        FfxFloat32x4 GatherEasuGreen(FfxFloat32x2 fPxPos)
        {
            return textureGather(sampler2D(r_input_color, s_LinearClamp), fPxPos, 1);
        }
#endif // defined(FSR1_BIND_SRV_INPUT_COLOR)

#if defined(FSR1_BIND_SRV_INPUT_COLOR)
        FfxFloat32x4 GatherEasuBlue(FfxFloat32x2 fPxPos)
        {
            return textureGather(sampler2D(r_input_color, s_LinearClamp), fPxPos, 2);
        }
#endif // defined(FSR1_BIND_SRV_INPUT_COLOR)

#if FFX_FSR1_OPTION_APPLY_RCAS
    #if defined(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR)
        void StoreEASUOutput(FfxUInt32x2 iPxPos, FfxFloat32x3 fColor)
        {
            imageStore(rw_internal_upscaled_color, FfxInt32x2(iPxPos), FfxFloat32x4(fColor, 1.f));
        }
    #endif  // #if defined(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR)
#else
    #if defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)
        void StoreEASUOutput(FfxUInt32x2 iPxPos, FfxFloat32x3 fColor)
        {
            imageStore(rw_upscaled_output, FfxInt32x2(iPxPos), FfxFloat32x4(fColor, 1.f));
        }
    #endif  // #if defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)
#endif // #if FFX_FSR1_OPTION_APPLY_RCAS

#if defined(FSR1_BIND_SRV_INTERNAL_UPSCALED_COLOR)
        FfxFloat32x4 LoadRCas_Input(FfxInt32x2 iPxPos)
        {
            return texelFetch(r_internal_upscaled_color, iPxPos, 0);
        }
#endif // defined(FSR1_BIND_UAV_INTERNAL_UPSCALED_COLOR)

#if defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)
        void StoreRCasOutput(FfxInt32x2 iPxPos, FfxFloat32x4 fColor)
        {
            imageStore(rw_upscaled_output, iPxPos, fColor);
        }

        void StoreRCasOutput(FfxInt32x2 iPxPos, FfxFloat32x3 fColor)
        {
            StoreRCasOutput(iPxPos, FfxFloat32x4(fColor, 1.f));
        }
#endif // defined(FSR1_BIND_UAV_UPSCALED_OUTPUT)

#endif // FFX_HALF

#endif // #if defined(FFX_GPU)
