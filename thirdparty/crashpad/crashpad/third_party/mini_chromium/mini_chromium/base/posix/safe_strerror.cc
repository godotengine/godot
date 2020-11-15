// Copyright 2009 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/posix/safe_strerror.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "base/macros.h"

namespace base {

void safe_strerror_r(int err, char* buf, size_t len) {
#if defined(__GLIBC__)
  char* ret = strerror_r(err, buf, len);
  if (ret != buf) {
    snprintf(buf, len, "%s", ret);
  }
#else
  int result = strerror_r(err, buf, len);
  if (result != 0) {
    snprintf(buf,
             len,
             "Error %d while retrieving error %d",
             result > 0 ? result : errno,
             err);
  }
#endif
}

std::string safe_strerror(int err) {
  char buf[256];
  safe_strerror_r(err, buf, arraysize(buf));
  return std::string(buf);
}

}  // namespace base
