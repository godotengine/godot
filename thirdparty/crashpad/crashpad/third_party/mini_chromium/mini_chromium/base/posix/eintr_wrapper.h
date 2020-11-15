// Copyright 2009 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// This provides a wrapper around system calls which may be interrupted by a
// signal and return EINTR. See man 7 signal.
//
// On Windows, this wrapper macro does nothing.

#ifndef MINI_CHROMIUM_BASE_POSIX_EINTR_WRAPPER_H_
#define MINI_CHROMIUM_BASE_POSIX_EINTR_WRAPPER_H_

#include "build/build_config.h"

// On Fuchsia, these wrapper macros do nothing because there are no signals.
#if defined(OS_POSIX) && !defined(OS_FUCHSIA)

#include <errno.h>

#define HANDLE_EINTR(x) ({ \
  decltype(x) eintr_wrapper_result; \
  do { \
    eintr_wrapper_result = (x); \
  } while (eintr_wrapper_result == -1 && errno == EINTR); \
  eintr_wrapper_result; \
})

#define IGNORE_EINTR(x) ({ \
  decltype(x) eintr_wrapper_result; \
  do { \
    eintr_wrapper_result = (x); \
    if (eintr_wrapper_result == -1 && errno == EINTR) { \
      eintr_wrapper_result = 0; \
    } \
  } while (0); \
  eintr_wrapper_result; \
})

#else

#define HANDLE_EINTR(x) (x)
#define IGNORE_EINTR(x) (x)

#endif  // OS_POSIX && !OS_FUCHSIA

#endif  // MINI_CHROMIUM_BASE_POSIX_EINTR_WRAPPER_H_
