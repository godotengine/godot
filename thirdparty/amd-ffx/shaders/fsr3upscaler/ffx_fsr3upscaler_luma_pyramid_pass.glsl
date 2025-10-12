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

//#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require

#define FSR3UPSCALER_BIND_SRV_CURRENT_LUMA                    0
#define FSR3UPSCALER_BIND_SRV_FARTHEST_DEPTH                  1

#define FSR3UPSCALER_BIND_UAV_SPD_GLOBAL_ATOMIC               2
#define FSR3UPSCALER_BIND_UAV_FRAME_INFO                      3
#define FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_0                4
#define FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_1                5
#define FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_2                6
#define FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_3                7
#define FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_4                8
#define FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_5                9
#define FSR3UPSCALER_BIND_UAV_FARTHEST_DEPTH_MIP1             10

#define FSR3UPSCALER_BIND_CB_FSR3UPSCALER                     11
#define FSR3UPSCALER_BIND_CB_SPD                              12

#include "../../gpu/fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h"
#include "../../gpu/fsr3upscaler/ffx_fsr3upscaler_common.h"
#include "../../gpu/fsr3upscaler/ffx_fsr3upscaler_luma_pyramid.h"

#ifndef FFX_FSR3UPSCALER_THREAD_GROUP_WIDTH
#define FFX_FSR3UPSCALER_THREAD_GROUP_WIDTH 256
#endif // #ifndef FFX_FSR3UPSCALER_THREAD_GROUP_WIDTH
#ifndef FFX_FSR3UPSCALER_THREAD_GROUP_HEIGHT
#define FFX_FSR3UPSCALER_THREAD_GROUP_HEIGHT 1
#endif // #ifndef FFX_FSR3UPSCALER_THREAD_GROUP_HEIGHT
#ifndef FFX_FSR3UPSCALER_THREAD_GROUP_DEPTH
#define FFX_FSR3UPSCALER_THREAD_GROUP_DEPTH 1
#endif // #ifndef FFX_FSR3UPSCALER_THREAD_GROUP_DEPTH
#ifndef FFX_FSR3UPSCALER_NUM_THREADS
#define FFX_FSR3UPSCALER_NUM_THREADS layout (local_size_x = FFX_FSR3UPSCALER_THREAD_GROUP_WIDTH, local_size_y = FFX_FSR3UPSCALER_THREAD_GROUP_HEIGHT, local_size_z = FFX_FSR3UPSCALER_THREAD_GROUP_DEPTH) in;
#endif // #ifndef FFX_FSR3UPSCALER_NUM_THREADS

FFX_FSR3UPSCALER_NUM_THREADS
void main()
{
    ComputeAutoExposure(gl_WorkGroupID, gl_LocalInvocationIndex);
}
