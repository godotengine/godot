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
// Needed for rw_upscaled_output declaration
#extension GL_EXT_shader_image_load_formatted : require

#define FSR2_BIND_SRV_INPUT_EXPOSURE                         0
#define FSR2_BIND_SRV_DILATED_REACTIVE_MASKS                 1
#if FFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS
#define FSR2_BIND_SRV_DILATED_MOTION_VECTORS                 2
#else
#define FSR2_BIND_SRV_INPUT_MOTION_VECTORS                   2
#endif
#define FSR2_BIND_SRV_INTERNAL_UPSCALED                      3
#define FSR2_BIND_SRV_LOCK_STATUS                            4
//#define FSR2_BIND_SRV_INPUT_DEPTH_CLIP                       5
#define FSR2_BIND_SRV_PREPARED_INPUT_COLOR                   6
#define FSR2_BIND_SRV_LUMA_INSTABILITY                       7
#define FSR2_BIND_SRV_LANCZOS_LUT                            8
#define FSR2_BIND_SRV_UPSCALE_MAXIMUM_BIAS_LUT               9
#define FSR2_BIND_SRV_SCENE_LUMINANCE_MIPS                   10
#define FSR2_BIND_SRV_AUTO_EXPOSURE                          11
#define FSR2_BIND_SRV_LUMA_HISTORY                           12

#define FSR2_BIND_UAV_INTERNAL_UPSCALED                      13
#define FSR2_BIND_UAV_LOCK_STATUS                            14
#define FSR2_BIND_UAV_UPSCALED_OUTPUT                        15
#define FSR2_BIND_UAV_NEW_LOCKS                              16
#define FSR2_BIND_UAV_LUMA_HISTORY                           17

#define FSR2_BIND_CB_FSR2                                    18

#if FFX_FSR2_OPTION_GODOT_DERIVE_INVALID_MOTION_VECTORS
#define FSR2_BIND_SRV_INPUT_DEPTH                            5
#endif

#include "ffx_fsr2_callbacks_glsl.h"
#include "ffx_fsr2_common.h"
#include "ffx_fsr2_sample.h"
#include "ffx_fsr2_upsample.h"
#include "ffx_fsr2_postprocess_lock_status.h"
#include "ffx_fsr2_reproject.h"
#include "ffx_fsr2_accumulate.h"

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

FFX_FSR2_NUM_THREADS
void main()
{
	uvec2 uGroupId = gl_WorkGroupID.xy;
    const uint GroupRows = (uint(DisplaySize().y) + FFX_FSR2_THREAD_GROUP_HEIGHT - 1) / FFX_FSR2_THREAD_GROUP_HEIGHT;
    uGroupId.y = GroupRows - uGroupId.y - 1;

    uvec2 uDispatchThreadId = uGroupId * uvec2(FFX_FSR2_THREAD_GROUP_WIDTH, FFX_FSR2_THREAD_GROUP_HEIGHT) + gl_LocalInvocationID.xy;

    Accumulate(ivec2(uDispatchThreadId));
}