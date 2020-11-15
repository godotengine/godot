// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/fuchsia/fuchsia_logging.h"

#include <zircon/status.h>

#include <iomanip>

namespace logging {

ZxLogMessage::ZxLogMessage(const char* function,
                           const char* file_path,
                           int line,
                           LogSeverity severity,
                           zx_status_t zx_err)
    : LogMessage(function, file_path, line, severity), zx_err_(zx_err) {}

ZxLogMessage::~ZxLogMessage() {
  // zx_status_t error values are negative, so log the numeric version as
  // decimal rather than hex. This is also useful to match zircon/errors.h for
  // grepping.
  stream() << ": " << zx_status_get_string(zx_err_) << " (" << zx_err_ << ")";
}

}  // namespace logging
