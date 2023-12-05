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

#pragma once

#include "ffx_types.h"

/// Typedef for error codes returned from functions in the FidelityFX SDK.
typedef int32_t FfxErrorCode;

static const FfxErrorCode FFX_OK                            = 0;           ///< The operation completed successfully.
static const FfxErrorCode FFX_ERROR_INVALID_POINTER         = 0x80000000;  ///< The operation failed due to an invalid pointer.
static const FfxErrorCode FFX_ERROR_INVALID_ALIGNMENT       = 0x80000001;  ///< The operation failed due to an invalid alignment.
static const FfxErrorCode FFX_ERROR_INVALID_SIZE            = 0x80000002;  ///< The operation failed due to an invalid size.
static const FfxErrorCode FFX_EOF                           = 0x80000003;  ///< The end of the file was encountered.
static const FfxErrorCode FFX_ERROR_INVALID_PATH            = 0x80000004;  ///< The operation failed because the specified path was invalid.
static const FfxErrorCode FFX_ERROR_EOF                     = 0x80000005;  ///< The operation failed because end of file was reached.
static const FfxErrorCode FFX_ERROR_MALFORMED_DATA          = 0x80000006;  ///< The operation failed because of some malformed data.
static const FfxErrorCode FFX_ERROR_OUT_OF_MEMORY           = 0x80000007;  ///< The operation failed because it ran out memory.
static const FfxErrorCode FFX_ERROR_INCOMPLETE_INTERFACE    = 0x80000008;  ///< The operation failed because the interface was not fully configured.
static const FfxErrorCode FFX_ERROR_INVALID_ENUM            = 0x80000009;  ///< The operation failed because of an invalid enumeration value.
static const FfxErrorCode FFX_ERROR_INVALID_ARGUMENT        = 0x8000000a;  ///< The operation failed because an argument was invalid.
static const FfxErrorCode FFX_ERROR_OUT_OF_RANGE            = 0x8000000b;  ///< The operation failed because a value was out of range.
static const FfxErrorCode FFX_ERROR_NULL_DEVICE             = 0x8000000c;  ///< The operation failed because a device was null.
static const FfxErrorCode FFX_ERROR_BACKEND_API_ERROR       = 0x8000000d;  ///< The operation failed because the backend API returned an error code.
static const FfxErrorCode FFX_ERROR_INSUFFICIENT_MEMORY     = 0x8000000e;  ///< The operation failed because there was not enough memory.

/// Helper macro to return error code y from a function when a specific condition, x, is not met.
#define FFX_RETURN_ON_ERROR(x, y)                   \
    if (!(x))                                       \
    {                                               \
        return (y);                                 \
    }

/// Helper macro to return error code x from a function when it is not FFX_OK.
#define FFX_VALIDATE(x)                             \
    {                                               \
        FfxErrorCode ret = x;                       \
        FFX_RETURN_ON_ERROR(ret == FFX_OK, ret);    \
    }

