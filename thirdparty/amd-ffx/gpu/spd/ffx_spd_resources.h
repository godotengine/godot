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

#ifndef FFX_SPD_RESOURCES_H
#define FFX_SPD_RESOURCES_H

#if defined(FFX_CPU) || defined(FFX_GPU)

// Assumes a maximum on 12 mips. If larger base resolution support is added,
// extra mip resources need to also be added
#define SPD_MAX_MIP_LEVELS 12

#define FFX_SPD_RESOURCE_IDENTIFIER_NULL                             0
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_GLOBAL_ATOMIC              1
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP  2
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC             3    // same as FFX_SPD_RESOURCE_DOWNSAMPLE_SRC_MIPMAP_0
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_0    3
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_1    4
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_2    5
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_3    6
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_4    7
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_5    8
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_6    9
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_7    10
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_8    11
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_9    12
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_10   13
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_11   14
#define FFX_SPD_RESOURCE_IDENTIFIER_INPUT_DOWNSAMPLE_SRC_MIPMAP_12   15
#define FFX_SPD_RESOURCE_IDENTIFIER_INTERNAL_GLOBAL_ATOMIC     16

#define FFX_SPD_RESOURCE_IDENTIFIER_COUNT                      17

// CBV resource definitions
#define FFX_SPD_CONSTANTBUFFER_IDENTIFIER_SPD                  0

#endif  // #if defined(FFX_CPU) || defined(FFX_GPU)

#endif  // FFX_SPD_RESOURCES_H
