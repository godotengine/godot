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

//#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require

#define FSR2_BIND_SRV_INPUT_OPAQUE_ONLY                     0
#define FSR2_BIND_SRV_INPUT_COLOR                           1
#define FSR2_BIND_UAV_AUTOREACTIVE                          2
#define FSR2_BIND_CB_REACTIVE                               3
#define FSR2_BIND_CB_FSR2                                   4

#include "ffx_fsr2_callbacks_glsl.h"
#include "ffx_fsr2_common.h"

// layout (set = 1, binding = FSR2_BIND_SRV_PRE_ALPHA_COLOR)  uniform texture2D   r_input_color_pre_alpha;
// layout (set = 1, binding = FSR2_BIND_SRV_POST_ALPHA_COLOR) uniform texture2D   r_input_color_post_alpha;
// layout (set = 1, binding = FSR2_BIND_UAV_REACTIVE, r8)     uniform image2D     rw_output_reactive_mask;


#ifndef FFX_FSR2_THREAD_GROUP_WIDTH
#define FFX_FSR2_THREAD_GROUP_WIDTH 8
#endif // #ifndef FFX_FSR2_THREAD_GROUP_WIDTH
#ifndef FFX_FSR2_THREAD_GROUP_HEIGHT
#define FFX_FSR2_THREAD_GROUP_HEIGHT 8
#endif // FFX_FSR2_THREAD_GROUP_HEIGHT
#ifndef FFX_FSR2_THREAD_GROUP_DEPTH
#define FFX_FSR2_THREAD_GROUP_DEPTH 1
#endif // #ifndef FFX_FSR2_THREAD_GROUP_DEPTH
#ifndef FFX_FSR2_NUM_THREADS
#define FFX_FSR2_NUM_THREADS layout (local_size_x = FFX_FSR2_THREAD_GROUP_WIDTH, local_size_y = FFX_FSR2_THREAD_GROUP_HEIGHT, local_size_z = FFX_FSR2_THREAD_GROUP_DEPTH) in;
#endif // #ifndef FFX_FSR2_NUM_THREADS

#if defined(FSR2_BIND_CB_REACTIVE)
layout (set = 1, binding = FSR2_BIND_CB_REACTIVE, std140) uniform cbGenerateReactive_t
{
	float   scale;
	float   threshold;
	float   binaryValue;
	uint    flags;
} cbGenerateReactive;
#endif

FFX_FSR2_NUM_THREADS
void main()
{
    FfxUInt32x2 uDispatchThreadId = gl_GlobalInvocationID.xy;

    FfxFloat32x3 ColorPreAlpha  = LoadOpaqueOnly(FFX_MIN16_I2(uDispatchThreadId)).rgb;
    FfxFloat32x3 ColorPostAlpha = LoadInputColor(FFX_MIN16_I2(uDispatchThreadId)).rgb;
    
    if ((cbGenerateReactive.flags & FFX_FSR2_AUTOREACTIVEFLAGS_APPLY_TONEMAP) != 0)
    {
        ColorPreAlpha = Tonemap(ColorPreAlpha);
        ColorPostAlpha = Tonemap(ColorPostAlpha);
    }

    if ((cbGenerateReactive.flags & FFX_FSR2_AUTOREACTIVEFLAGS_APPLY_INVERSETONEMAP) != 0)
    {
        ColorPreAlpha = InverseTonemap(ColorPreAlpha);
        ColorPostAlpha = InverseTonemap(ColorPostAlpha);
    }

    FfxFloat32 out_reactive_value = 0.f;
    FfxFloat32x3 delta = abs(ColorPostAlpha - ColorPreAlpha);
    
    out_reactive_value = ((cbGenerateReactive.flags & FFX_FSR2_AUTOREACTIVEFLAGS_USE_COMPONENTS_MAX)!=0) ? max(delta.x, max(delta.y, delta.z)) : length(delta);
    out_reactive_value *= cbGenerateReactive.scale;

    out_reactive_value = ((cbGenerateReactive.flags & FFX_FSR2_AUTOREACTIVEFLAGS_APPLY_THRESHOLD)!=0) ? ((out_reactive_value < cbGenerateReactive.threshold) ? 0 : cbGenerateReactive.binaryValue) : out_reactive_value;

    imageStore(rw_output_autoreactive, FfxInt32x2(uDispatchThreadId), vec4(out_reactive_value));
}
