// Copyright 2017 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "util/linux/scoped_pr_set_ptracer.h"

#include <errno.h>
#include <sys/prctl.h>

#include "base/logging.h"

namespace crashpad {

ScopedPrSetPtracer::ScopedPrSetPtracer(pid_t pid, bool may_log)
    : success_(false), may_log_(may_log) {
  success_ = prctl(PR_SET_PTRACER, pid, 0, 0, 0) == 0;
  PLOG_IF(ERROR, !success_ && may_log && errno != EINVAL) << "prctl";
}

ScopedPrSetPtracer::~ScopedPrSetPtracer() {
  if (success_) {
    int res = prctl(PR_SET_PTRACER, 0, 0, 0, 0);
    PLOG_IF(ERROR, res != 0 && may_log_) << "prctl";
  }
}

}  // namespace crashpad
