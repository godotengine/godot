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

#ifndef FFX_FSR3_RESOURCES_H
#define FFX_FSR3_RESOURCES_H

#include "../fsr2/ffx_fsr2_resources.h"
#include "../frameinterpolation/ffx_frameinterpolation_resources.h"

#if defined(FFX_CPU) || defined(FFX_GPU)
#define FFX_FSR3_RESOURCE_IDENTIFIER_NULL                                           0

#define FFX_FSR3_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR                            1
#define FFX_FSR3_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_OUTPUT                        2
#define FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_DEPTH_0                                5
#define FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS_0                       6
#define FFX_FSR3_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH_0         7
#define FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_DEPTH_1                                8
#define FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS_1                       9
#define FFX_FSR3_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH_1         10
#define FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_DEPTH_2                                11
#define FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS_2                       12
#define FFX_FSR3_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH_2         13
#define FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_DEPTH_3                                14
#define FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS_3                       15
#define FFX_FSR3_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH_3         16

#define FFX_FSR3_RESOURCE_IDENTIFIER_COUNT                                          17
#define FFX_FSR3_RESOURCE_IDENTIFIER_UPSCALED_COUNT                                 3
#endif // #if defined(FFX_CPU) || defined(FFX_GPU)

#endif //!defined( FFX_FSR2_RESOURCES_H )
