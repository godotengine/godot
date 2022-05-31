/*
 * Copyright 2010 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/core/SkTypes.h"

#if defined(SK_BUILD_FOR_WIN)

#include "src/core/SkLeanWindows.h"

#include <stdarg.h>
#include <stdio.h>

static const size_t kBufferSize = 2048;

void SkDebugf(const char format[], ...) {
    char    buffer[kBufferSize + 1];
    va_list args;

    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fflush(stderr);  // stderr seems to be buffered on Windows.

    va_start(args, format);
    vsnprintf(buffer, kBufferSize, format, args);
    va_end(args);

    OutputDebugStringA(buffer);
}
#endif//defined(SK_BUILD_FOR_WIN)
