// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_CLIENT_SIMULATE_CRASH_LINUX_H_
#define CRASHPAD_CLIENT_SIMULATE_CRASH_LINUX_H_

#include "client/crashpad_client.h"
#include "util/misc/capture_context.h"

//! \file

//! \brief Captures the CPU context and simulates an exception without crashing.
#define CRASHPAD_SIMULATE_CRASH()                                            \
  do {                                                                       \
    crashpad::NativeCPUContext simulate_crash_cpu_context;                   \
    crashpad::CaptureContext(&simulate_crash_cpu_context);                   \
    crashpad::CrashpadClient::DumpWithoutCrash(&simulate_crash_cpu_context); \
  } while (false)

#endif  // CRASHPAD_CLIENT_SIMULATE_CRASH_LINUX_H_
