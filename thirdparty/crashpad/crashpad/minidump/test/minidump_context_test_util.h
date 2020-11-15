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

#ifndef CRASHPAD_MINIDUMP_TEST_MINIDUMP_CONTEXT_TEST_UTIL_H_
#define CRASHPAD_MINIDUMP_TEST_MINIDUMP_CONTEXT_TEST_UTIL_H_

#include <stdint.h>

#include "minidump/minidump_context.h"

namespace crashpad {
namespace test {

//! \brief Initializes a context structure for testing.
//!
//! Initialization is compatible with the initialization used by CPUContext test
//! initialization functions such as InitializeCPUContextX86() and
//! InitializeCPUContextX86_64() for identical \a seed values.
//!
//! \param[out] context The structure to initialize.
//! \param[in] seed The seed value. Initializing two context structures of the
//!     same type with identical seed values should produce identical context
//!     structures. Initialization with a different seed value should produce
//!     a different context structure. If \a seed is `0`, \a context is zeroed
//!     out entirely except for the flags field, which will identify the context
//!     type. If \a seed is nonzero, \a context will be populated entirely with
//!     nonzero values.
//!
//! \{
void InitializeMinidumpContextX86(MinidumpContextX86* context, uint32_t seed);
void InitializeMinidumpContextAMD64(MinidumpContextAMD64* context,
                                    uint32_t seed);
void InitializeMinidumpContextARM(MinidumpContextARM* context, uint32_t seed);
void InitializeMinidumpContextARM64(MinidumpContextARM64* context,
                                    uint32_t seed);
void InitializeMinidumpContextMIPS(MinidumpContextMIPS* context, uint32_t seed);
void InitializeMinidumpContextMIPS64(MinidumpContextMIPS* context,
                                     uint32_t seed);
//! \}

//! \brief Verifies, via gtest assertions, that a context structure contains
//!     expected values.
//!
//! \param[in] expect_seed The seed value used to initialize a context
//!     structure. This is the seed value used with
//!     InitializeMinidumpContext*().
//! \param[in] observed The context structure to check. All fields of this
//!     structure will be compared against the expected context structure, one
//!     initialized with \a expect_seed.
//! \param[in] snapshot If `true`, compare \a observed to a context structure
//!     expected to be produced from a CPUContext snapshot. If `false`, compare
//!     \a observed to a native minidump context structure. CPUContext snapshot
//!     structures may carry different sets of data than native minidump context
//!     structures in meaningless ways. When `true`, fields not found in
//!     CPUContext structures are expected to be `0`. When `false`, all fields
//!     are compared. This makes it possible to test both that these fields are
//!     passed through correctly by the native minidump writer and are zeroed
//!     out when creating a minidump context structure from a CPUContext
//!     structure.
//! \{
void ExpectMinidumpContextX86(
    uint32_t expect_seed, const MinidumpContextX86* observed, bool snapshot);
void ExpectMinidumpContextAMD64(
    uint32_t expect_seed, const MinidumpContextAMD64* observed, bool snapshot);
void ExpectMinidumpContextARM(uint32_t expect_seed,
                              const MinidumpContextARM* observed,
                              bool snapshot);
void ExpectMinidumpContextARM64(uint32_t expect_seed,
                                const MinidumpContextARM64* observed,
                                bool snapshot);
void ExpectMinidumpContextMIPS(uint32_t expect_seed,
                               const MinidumpContextMIPS* observed,
                               bool snapshot);
void ExpectMinidumpContextMIPS64(uint32_t expect_seed,
                                 const MinidumpContextMIPS64* observed,
                                 bool snapshot);
//! \}

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_TEST_MINIDUMP_CONTEXT_TEST_UTIL_H_
