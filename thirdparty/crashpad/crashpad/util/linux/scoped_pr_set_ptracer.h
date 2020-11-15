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

#ifndef CRASHPAD_UTIL_LINUX_SCOPED_PR_SET_PTRACER_H_
#define CRASHPAD_UTIL_LINUX_SCOPED_PR_SET_PTRACER_H_

#include <sys/types.h>

#include "base/macros.h"

namespace crashpad {

class ScopedPrSetPtracer {
 public:
  //! \brief Uses `PR_SET_PTRACER` to set \a pid as the caller's ptracer.
  //!
  //! `PR_SET_PTRACER` is only supported if the Yama Linux security module (LSM)
  //! is enabled. Otherwise, `prctl(PR_SET_PTRACER, ...)` fails with `EINVAL`.
  //! See linux-4.9.20/security/yama/yama_lsm.c yama_task_prctl() and
  //! linux-4.9.20/kernel/sys.c [sys_]prctl().
  //!
  //! An error message will be logged on failure only if \a may_log is `true`
  //! and `prctl` does not fail with `EINVAL`;
  //!
  //! \param[in] pid The process ID of the process to make the caller's ptracer.
  //! \param[in] may_log if `true`, this class may log error messages.
  ScopedPrSetPtracer(pid_t pid, bool may_log);

  ~ScopedPrSetPtracer();

 private:
  bool success_;
  bool may_log_;

  DISALLOW_COPY_AND_ASSIGN(ScopedPrSetPtracer);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_LINUX_SCOPED_PR_SET_PTRACER_H_
