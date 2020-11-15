// Copyright 2006-2008 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_STRINGS_STRING_UTIL_H_
#define MINI_CHROMIUM_BASE_STRINGS_STRING_UTIL_H_

#include "base/compiler_specific.h"
#include "base/logging.h"

namespace base {

int vsnprintf(char* buffer, size_t size, const char* format, va_list arguments)
    PRINTF_FORMAT(3, 0);

size_t strlcpy(char* dst, const char* src, size_t dst_size);
size_t wcslcpy(wchar_t* dst, const wchar_t* src, size_t dst_size);

}  // namespace base

#if defined(OS_WIN)
#include "base/strings/string_util_win.h"
#elif defined(OS_POSIX)
#include "base/strings/string_util_posix.h"
#endif

template <class string_type>
inline typename string_type::value_type* WriteInto(string_type* str,
                                                   size_t length_with_null) {
  DCHECK_NE(0u, length_with_null);
  str->reserve(length_with_null);
  str->resize(length_with_null - 1);

  if (length_with_null <= 1)
    return NULL;

  return &((*str)[0]);
}

#endif  // MINI_CHROMIUM_BASE_STRINGS_STRING_UTIL_H_
