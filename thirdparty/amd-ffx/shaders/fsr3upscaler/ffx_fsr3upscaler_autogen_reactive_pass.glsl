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

#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require

#define FSR3UPSCALER_BIND_SRV_INPUT_OPAQUE_ONLY                     0
#define FSR3UPSCALER_BIND_SRV_INPUT_COLOR                           1

#define FSR3UPSCALER_BIND_UAV_AUTOREACTIVE                          2
#define FSR3UPSCALER_BIND_UAV_AUTOCOMPOSITION                       3

#define FSR3UPSCALER_BIND_CB_FSR3UPSCALER                           4
#define FSR3UPSCALER_BIND_CB_REACTIVE                               5

#include "../../gpu/fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h"
#include "../../gpu/fsr3upscaler/ffx_fsr3upscaler_common.h"

#ifndef FFX_FSR3UPSCALER_THREAD_GROUP_WIDTH
#define FFX_FSR3UPSCALER_THREAD_GROUP_WIDTH 8
#endif // #ifndef FFX_FSR3UPSCALER_THREAD_GROUP_WIDTH
#ifndef FFX_FSR3UPSCALER_THREAD_GROUP_HEIGHT
#define FFX_FSR3UPSCALER_THREAD_GROUP_HEIGHT 8
#endif // FFX_FSR3UPSCALER_THREAD_GROUP_HEIGHT
#ifndef FFX_FSR3UPSCALER_THREAD_GROUP_DEPTH
#define FFX_FSR3UPSCALER_THREAD_GROUP_DEPTH 1
#endif // #ifndef FFX_FSR3UPSCALER_THREAD_GROUP_DEPTH
#ifndef FFX_FSR3UPSCALER_NUM_THREADS
#define FFX_FSR3UPSCALER_NUM_THREADS layout (local_size_x = FFX_FSR3UPSCALER_THREAD_GROUP_WIDTH, local_size_y = FFX_FSR3UPSCALER_THREAD_GROUP_HEIGHT, local_size_z = FFX_FSR3UPSCALER_THREAD_GROUP_DEPTH) in;
#endif // #ifndef FFX_FSR3UPSCALER_NUM_THREADS

FFX_FSR3UPSCALER_NUM_THREADS
void main()
{
    FfxUInt32x2 uGroupId = gl_WorkGroupID.xy;
    FfxUInt32x2 uGroupThreadId = gl_LocalInvocationID.xy;
    FfxUInt32x2 uDispatchThreadId = uGroupId * FfxUInt32x2(FFX_FSR3UPSCALER_THREAD_GROUP_WIDTH, FFX_FSR3UPSCALER_THREAD_GROUP_HEIGHT) + uGroupThreadId;

    FfxFloat32x3 ColorPreAlpha  = LoadOpaqueOnly(FFX_MIN16_I2(uDispatchThreadId)).rgb;
    FfxFloat32x3 ColorPostAlpha = LoadInputColor(FfxInt32x2(uDispatchThreadId)).rgb;

    if ((GenReactiveFlags() & FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_TONEMAP) != 0)
    {
        ColorPreAlpha = Tonemap(ColorPreAlpha);
        ColorPostAlpha = Tonemap(ColorPostAlpha);
    }

    if ((GenReactiveFlags() & FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_INVERSETONEMAP) != 0)
    {
        ColorPreAlpha = InverseTonemap(ColorPreAlpha);
        ColorPostAlpha = InverseTonemap(ColorPostAlpha);
    }

    FfxFloat32 out_reactive_value = 0.f;
    FfxFloat32x3 delta = abs(ColorPostAlpha - ColorPreAlpha);

    out_reactive_value = ((GenReactiveFlags() & FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_USE_COMPONENTS_MAX)!=0) ? max(delta.x, max(delta.y, delta.z)) : length(delta);
    out_reactive_value *= GenReactiveScale();

    out_reactive_value = ((GenReactiveFlags() & FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_THRESHOLD)!=0) ? ((out_reactive_value < GenReactiveThreshold()) ? 0 : GenReactiveBinaryValue()) : out_reactive_value;

    imageStore(rw_output_autoreactive, FfxInt32x2(uDispatchThreadId), FfxFloat32x4(out_reactive_value, 0.0, 0.0, 0.0));
}
