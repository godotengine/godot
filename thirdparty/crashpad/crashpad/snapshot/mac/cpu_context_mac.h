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

#ifndef CRASHPAD_SNAPSHOT_MAC_CPU_CONTEXT_MAC_H_
#define CRASHPAD_SNAPSHOT_MAC_CPU_CONTEXT_MAC_H_

#include <mach/mach.h>

#include "build/build_config.h"
#include "snapshot/cpu_context.h"
#include "util/mach/mach_extensions.h"

namespace crashpad {
namespace internal {

#if defined(ARCH_CPU_X86_FAMILY) || DOXYGEN

//! \brief Initializes a CPUContextX86 structure from native context structures
//!     on macOS.
//!
//! \a flavor, \a state, and \a state_count may be supplied by exception
//! handlers in order for the \a context parameter to be initialized by the
//! thread state received by the exception handler to the extent possible. In
//! that case, whatever thread state specified by these three parameters will
//! supersede \a x86_thread_state32, \a x86_float_state32, or \a
//! x86_debug_state32. If thread state in this format is not available, \a
//! flavor may be set to `THREAD_STATE_NONE`, and all of \a x86_thread_state32,
//! \a x86_float_state32, and \a x86_debug_state32 will be honored.
//!
//! If \a flavor, \a state, and \a state_count are provided but do not contain
//! valid values, a message will be logged and their values will be ignored as
//! though \a flavor were specified as `THREAD_STATE_NONE`.
//!
//! \param[out] context The CPUContextX86 structure to initialize.
//! \param[in] flavor The native thread state flavor of \a state. This may be
//!     `x86_THREAD_STATE32`, `x86_FLOAT_STATE32`, `x86_DEBUG_STATE32`,
//!     `x86_THREAD_STATE`, `x86_FLOAT_STATE`, or `x86_DEBUG_STATE`. It may also
//!     be `THREAD_STATE_NONE` if \a state is not supplied (and is `nullptr`).
//! \param[in] state The native thread state, which may be a casted pointer to
//!     `x86_thread_state32_t`, `x86_float_state32_t`, `x86_debug_state32_t`,
//!     `x86_thread_state`, `x86_float_state`, or `x86_debug_state`. This
//!     parameter may be `nullptr` to not supply this data, in which case \a
//!     flavor must be `THREAD_STATE_NONE`. If a “universal” structure is used,
//!     it must carry 32-bit state data of the correct type.
//! \param[in] state_count The number of `natural_t`-sized (`int`-sized) units
//!     in \a state. This may be 0 if \a state is `nullptr`.
//! \param[in] x86_thread_state32 The state of the thread’s integer registers.
//! \param[in] x86_float_state32 The state of the thread’s floating-point
//!     registers.
//! \param[in] x86_debug_state32 The state of the thread’s debug registers.
void InitializeCPUContextX86(CPUContextX86* context,
                             thread_state_flavor_t flavor,
                             ConstThreadState state,
                             mach_msg_type_number_t state_count,
                             const x86_thread_state32_t* x86_thread_state32,
                             const x86_float_state32_t* x86_float_state32,
                             const x86_debug_state32_t* x86_debug_state32);

//! \brief Initializes a CPUContextX86_64 structure from native context
//!     structures on macOS.
//!
//! \a flavor, \a state, and \a state_count may be supplied by exception
//! handlers in order for the \a context parameter to be initialized by the
//! thread state received by the exception handler to the extent possible. In
//! that case, whatever thread state specified by these three parameters will
//! supersede \a x86_thread_state64, \a x86_float_state64, or \a
//! x86_debug_state64. If thread state in this format is not available, \a
//! flavor may be set to `THREAD_STATE_NONE`, and all of \a x86_thread_state64,
//! \a x86_float_state64, and \a x86_debug_state64 will be honored.
//!
//! If \a flavor, \a state, and \a state_count are provided but do not contain
//! valid values, a message will be logged and their values will be ignored as
//! though \a flavor were specified as `THREAD_STATE_NONE`.
//!
//! \param[out] context The CPUContextX86_64 structure to initialize.
//! \param[in] flavor The native thread state flavor of \a state. This may be
//!     `x86_THREAD_STATE64`, `x86_FLOAT_STATE64`, `x86_DEBUG_STATE64`,
//!     `x86_THREAD_STATE`, `x86_FLOAT_STATE`, or `x86_DEBUG_STATE`. It may also
//!     be `THREAD_STATE_NONE` if \a state is not supplied (and is `nullptr`).
//! \param[in] state The native thread state, which may be a casted pointer to
//!     `x86_thread_state64_t`, `x86_float_state64_t`, `x86_debug_state64_t`,
//!     `x86_thread_state`, `x86_float_state`, or `x86_debug_state`. This
//!     parameter may be `nullptr` to not supply this data, in which case \a
//!     flavor must be `THREAD_STATE_NONE`. If a “universal” structure is used,
//!     it must carry 64-bit state data of the correct type.
//! \param[in] state_count The number of `int`-sized units in \a state. This may
//!     be 0 if \a state is `nullptr`.
//! \param[in] x86_thread_state64 The state of the thread’s integer registers.
//! \param[in] x86_float_state64 The state of the thread’s floating-point
//!     registers.
//! \param[in] x86_debug_state64 The state of the thread’s debug registers.
void InitializeCPUContextX86_64(CPUContextX86_64* context,
                                thread_state_flavor_t flavor,
                                ConstThreadState state,
                                mach_msg_type_number_t state_count,
                                const x86_thread_state64_t* x86_thread_state64,
                                const x86_float_state64_t* x86_float_state64,
                                const x86_debug_state64_t* x86_debug_state64);

#endif

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MAC_CPU_CONTEXT_MAC_H_
