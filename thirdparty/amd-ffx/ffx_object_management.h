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

#pragma once

#include "ffx_types.h"
#include "ffx_interface.h"

#if defined(__cplusplus)
extern "C" {
#endif  // #if defined(__cplusplus)

FFX_API void ffxSafeReleasePipeline(FfxInterface* backendInterface, FfxPipelineState* pipeline, FfxUInt32 effectContextId);
FFX_API void ffxSafeReleaseCopyResource(FfxInterface* backendInterface, FfxResourceInternal resource, FfxUInt32 effectContextId);
FFX_API void ffxSafeReleaseResource(FfxInterface* backendInterface, FfxResourceInternal resource, FfxUInt32 effectContextId);

#if defined(__cplusplus)
}
#endif  // #if defined(__cplusplus)
