// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_SNAPSHOT_TEST_TEST_CPU_CONTEXT_H_
#define CRASHPAD_SNAPSHOT_TEST_TEST_CPU_CONTEXT_H_

#include <stdint.h>

#include "snapshot/cpu_context.h"

namespace crashpad {
namespace test {

//! \brief Initializes an `fxsave` context substructure for testing.
//!
//! \param[out] fxsave The structure to initialize.
//! \param[in,out] seed The seed value. Initializing two `fxsave` structures of
//!     the same type with identical seed values should produce identical
//!     structures. Initialization with a different seed value should produce
//!     a different `fxsave` structure. If \a seed is `0`, \a fxsave is zeroed
//!     out entirely. If \a seed is nonzero, \a fxsave will be populated
//!     entirely with nonzero values. \a seed will be updated by this function
//!     to allow the caller to perform subsequent initialization of the context
//!     structure containing \a fxsave.
//!
//! \{
void InitializeCPUContextX86Fxsave(CPUContextX86::Fxsave* fxsave,
                                   uint32_t* seed);
void InitializeCPUContextX86_64Fxsave(CPUContextX86_64::Fxsave* fxsave,
                                      uint32_t* seed);
//! \}

//! \brief Initializes a context structure for testing.
//!
//! Initialization is compatible with the initialization used by minidump
//! context test initialization functions such as InitializeMinidumpContextX86()
//! and InitializeMinidumpContextAMD64() for identical \a seed values.
//!
//! \param[out] context The structure to initialize.
//! \param[in] seed The seed value. Initializing two context structures of the
//!     same type with identical seed values should produce identical context
//!     structures. Initialization with a different seed value should produce
//!     a different context structure. If \a seed is `0`, \a context is zeroed
//!     out entirely except for the CPUContext::architecture field, which will
//!     identify the context type. If \a seed is nonzero, \a context will be
//!     populated entirely with nonzero values.
//!
//! \{
void InitializeCPUContextX86(CPUContext* context, uint32_t seed);
void InitializeCPUContextX86_64(CPUContext* context, uint32_t seed);
void InitializeCPUContextARM(CPUContext* context, uint32_t seed);
void InitializeCPUContextARM64(CPUContext* context, uint32_t seed);
void InitializeCPUContextMIPS(CPUContext* context, uint32_t seed);
void InitializeCPUContextMIPS64(CPUContext* context, uint32_t seed);
//! \}

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_TEST_TEST_CPU_CONTEXT_H_
