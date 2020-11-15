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

#include "util/misc/capture_context.h"

#include <stdint.h>

namespace crashpad {
namespace test {

//! \brief Sanity check conditions that should be true for any NativeCPUContext
//!     produced by CaptureContext().
//!
//! If the context structure has fields that tell whether it’s valid, such as
//! magic numbers or size fields, sanity-checks those fields for validity with
//! fatal gtest assertions. For other fields, where it’s possible to reason
//! about their validity based solely on their contents, sanity-checks via
//! nonfatal gtest assertions.
//!
//! \param[in] context The context to check.
void SanityCheckContext(const NativeCPUContext& context);

//! \brief Return the value of the program counter from a NativeCPUContext.
uintptr_t ProgramCounterFromContext(const NativeCPUContext& context);

//! \brief Return the value of the stack pointer from a NativeCPUContext.
uintptr_t StackPointerFromContext(const NativeCPUContext& context);

}  // namespace test
}  // namespace crashpad
