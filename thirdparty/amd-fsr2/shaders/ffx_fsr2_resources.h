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

#ifndef FFX_FSR2_RESOURCES_H
#define FFX_FSR2_RESOURCES_H

#if defined(FFX_CPU) || defined(FFX_GPU)
#define FFX_FSR2_RESOURCE_IDENTIFIER_NULL                                           0
#define FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_OPAQUE_ONLY                              1
#define FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_COLOR                                    2
#define FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS                           3
#define FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_DEPTH                                    4
#define FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_EXPOSURE                                 5
#define FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_REACTIVE_MASK                            6
#define FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK        7
#define FFX_FSR2_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH           8
#define FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS                         9
#define FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_DEPTH                                  10
#define FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR                        11
#define FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS                                    12
#define FFX_FSR2_RESOURCE_IDENTIFIER_NEW_LOCKS                                      13
#define FFX_FSR2_RESOURCE_IDENTIFIER_PREPARED_INPUT_COLOR                           14
#define FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY                                   15
#define FFX_FSR2_RESOURCE_IDENTIFIER_DEBUG_OUTPUT                                   16
#define FFX_FSR2_RESOURCE_IDENTIFIER_LANCZOS_LUT                                    17
#define FFX_FSR2_RESOURCE_IDENTIFIER_SPD_ATOMIC_COUNT                               18
#define FFX_FSR2_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT                                19
#define FFX_FSR2_RESOURCE_IDENTIFIER_RCAS_INPUT                                     20
#define FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS_1                                  21
#define FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS_2                                  22
#define FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_1                      23
#define FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_2                      24
#define FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DEFAULT_REACTIVITY                    25
#define FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DEFAULT_TRANSPARENCY_AND_COMPOSITION  26
#define FFX_FSR2_RESOURCE_IDENTITIER_UPSAMPLE_MAXIMUM_BIAS_LUT                      27
#define FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_REACTIVE_MASKS                         28
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE                                29 // same as FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_0
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_0                       29
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_1                       30
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_2                       31
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_3                       32
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_4                       33
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_5                       34
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_6                       35
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_7                       36
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_8                       37
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_9                       38
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_10                      39
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_11                      40
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_12                      41
#define FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DEFAULT_EXPOSURE                      42
#define FFX_FSR2_RESOURCE_IDENTIFIER_AUTO_EXPOSURE                                  43
#define FFX_FSR2_RESOURCE_IDENTIFIER_AUTOREACTIVE                                   44
#define FFX_FSR2_RESOURCE_IDENTIFIER_AUTOCOMPOSITION                                45

#define FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR                           46
#define FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR                          47
#define FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR_1                         48
#define FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR_1                        49
#define FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR_2                         50
#define FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR_2                        51
#define FFX_FSR2_RESOURCE_IDENTIFIER_PREVIOUS_DILATED_MOTION_VECTORS                52
#define FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DILATED_MOTION_VECTORS_1              53
#define FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DILATED_MOTION_VECTORS_2              54
#define FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY_1                                 55
#define FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY_2                                 56
#define FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_INPUT_LUMA                                57

// Shading change detection mip level setting, value must be in the range [FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_0, FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_12]
#define FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_SHADING_CHANGE          FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_4
#define FFX_FSR2_SHADING_CHANGE_MIP_LEVEL                                           (FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_SHADING_CHANGE - FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE)

#define FFX_FSR2_RESOURCE_IDENTIFIER_COUNT                                          58

#define FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_FSR2                                     0
#define FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_SPD                                      1
#define FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_RCAS                                     2
#define FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_GENREACTIVE                              3

#define FFX_FSR2_AUTOREACTIVEFLAGS_APPLY_TONEMAP                                    1
#define FFX_FSR2_AUTOREACTIVEFLAGS_APPLY_INVERSETONEMAP                             2
#define FFX_FSR2_AUTOREACTIVEFLAGS_APPLY_THRESHOLD                                  4
#define FFX_FSR2_AUTOREACTIVEFLAGS_USE_COMPONENTS_MAX                               8

#endif // #if defined(FFX_CPU) || defined(FFX_GPU)

#endif //!defined( FFX_FSR2_RESOURCES_H )
