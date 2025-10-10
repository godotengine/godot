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

#define FFX_FRAMEINTERPOLATION_BIND_SRV_OUTPUT                          0

#define FFX_FRAMEINTERPOLATION_BIND_UAV_COUNTERS                        1
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_0     2
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_1     3
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_2     4
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_3     5
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_4     6
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_5     7
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_6     8
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_7     9
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_8     10
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_9     11
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_10    12
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_11    13
#define FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_12    14

#define FFX_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION               15
#define FFX_FRAMEINTERPOLATION_BIND_CB_INPAINTING_PYRAMID               16

#ifdef FFX_HALF
    #undef FFX_HALF
    #define FFX_HALF 0
#endif

#include "../../gpu/frameinterpolation/ffx_frameinterpolation_callbacks_glsl.h"
#include "../../gpu/frameinterpolation/ffx_frameinterpolation_common.h"
#include "../../gpu/frameinterpolation/ffx_frameinterpolation_compute_inpainting_pyramid.h"

#ifndef FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH
#define FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH 256
#endif // #ifndef FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH
#ifndef FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT
#define FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT 1
#endif // #ifndef FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT
#ifndef FFX_FRAMEINTERPOLATION_THREAD_GROUP_DEPTH
#define FFX_FRAMEINTERPOLATION_THREAD_GROUP_DEPTH 1
#endif // #ifndef FFX_FRAMEINTERPOLATION_THREAD_GROUP_DEPTH
#ifndef FFX_FRAMEINTERPOLATION_NUM_THREADS
#define FFX_FRAMEINTERPOLATION_NUM_THREADS layout (local_size_x = FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH, local_size_y = FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT, local_size_z = FFX_FRAMEINTERPOLATION_THREAD_GROUP_DEPTH) in;
#endif // #ifndef FFX_FRAMEINTERPOLATION_NUM_THREADS

FFX_FRAMEINTERPOLATION_NUM_THREADS
void main()
{
    computeFrameinterpolationInpaintingPyramid(FfxInt32x3(gl_WorkGroupID), FfxInt32(gl_LocalInvocationIndex));
}
