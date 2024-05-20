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

/// @defgroup Core
/// @defgroup HLSL
/// @defgroup GLSL
/// @defgroup GPU
/// @defgroup CPU
/// @defgroup CAS
/// @defgroup FSR1

#if !defined(FFX_CORE_H)
#define FFX_CORE_H

#include "ffx_common_types.h"

#if defined(FFX_CPU)
    #include "ffx_core_cpu.h"
#endif // #if defined(FFX_CPU)

#if defined(FFX_GLSL) && defined(FFX_GPU)
    #include "ffx_core_glsl.h"
#endif // #if defined(FFX_GLSL) && defined(FFX_GPU)

#if defined(FFX_HLSL) && defined(FFX_GPU)
    #include "ffx_core_hlsl.h"
#endif // #if defined(FFX_HLSL) && defined(FFX_GPU)

#if defined(FFX_GPU)
    #include "ffx_core_gpu_common.h"
    #include "ffx_core_gpu_common_half.h"
    #include "ffx_core_portability.h"
#endif // #if defined(FFX_GPU)
#endif // #if !defined(FFX_CORE_H)