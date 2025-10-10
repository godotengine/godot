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

#ifdef __cplusplus
extern "C" {
#endif  // #ifdef __cplusplus

/// @defgroup Messages Messages
/// Messages used by FidelityFX SDK functions
///
/// @ingroup ffxHost

/// Provides the ability to set a callback for print messages.
///
/// @param [in] callback                The callback function that will receive assert messages.
///
/// @ingroup Messages
FFX_API void ffxSetPrintMessageCallback(ffxMessageCallback callback, uint32_t debugLevel);

/// Function to print a message.
///
/// @param [in] type                    See FfxMsgType
/// @param [in] message                 The message to print.
///
/// @ingroup Messages
FFX_API void ffxPrintMessage(uint32_t type, const wchar_t* message);

/// Macro to print message
/// by calling application registered callback,
/// otherwise to debugger's TTY
///
/// @ingroup Messages
#define FFX_PRINT_MESSAGE( type, msg) \
    do                                \
    {                                 \
        ffxPrintMessage( type, msg);  \
    } while (0)
#ifdef __cplusplus
}
#endif  // #ifdef __cplusplus
