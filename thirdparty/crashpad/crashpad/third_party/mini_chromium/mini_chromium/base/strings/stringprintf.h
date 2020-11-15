// Copyright 2010 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_STRINGS_STRINGPRINTF_H_
#define MINI_CHROMIUM_BASE_STRINGS_STRINGPRINTF_H_

#include <stdarg.h>

#include <string>

#include "base/compiler_specific.h"

namespace base {

std::string StringPrintf(const char* format, ...)
    PRINTF_FORMAT(1, 2);
void StringAppendV(std::string* dst, const char* format, va_list ap)
    PRINTF_FORMAT(2, 0);

}  // namespace base

#endif  // MINI_CHROMIUM_BASE_STRINGS_STRINGPRINTF_H_
