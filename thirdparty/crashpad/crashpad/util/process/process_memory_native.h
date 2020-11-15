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

#include "build/build_config.h"

#if defined(OS_FUCHSIA)
#include "util/process/process_memory_fuchsia.h"
#elif defined(OS_LINUX) || defined(OS_ANDROID)
#include "util/process/process_memory_linux.h"
#endif

namespace crashpad {

#if defined(OS_FUCHSIA) || DOXYGEN
//! \brief Alias for platform-specific native implementation of ProcessMemory.
using ProcessMemoryNative = ProcessMemoryFuchsia;
#elif defined(OS_LINUX) || defined(OS_ANDROID)
using ProcessMemoryNative = ProcessMemoryLinux;
#else
#error Port.
#endif

}  // namespace crashpad
