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

#ifndef CRASHPAD_SNAPSHOT_LINUX_CPU_CONTEXT_LINUX_H_
#define CRASHPAD_SNAPSHOT_LINUX_CPU_CONTEXT_LINUX_H_

#include "build/build_config.h"
#include "snapshot/cpu_context.h"
#include "snapshot/linux/signal_context.h"
#include "util/linux/thread_info.h"

namespace crashpad {
namespace internal {

#if defined(ARCH_CPU_X86_FAMILY) || DOXYGEN

//! \{
//! \brief Initializes a CPUContextX86 structure from native context structures
//!     on Linux.
//!
//! \param[in] thread_context The native thread context.
//! \param[in] float_context The native float context.
//! \param[out] context The CPUContextX86 structure to initialize.
void InitializeCPUContextX86(const ThreadContext::t32_t& thread_context,
                             const FloatContext::f32_t& float_context,
                             CPUContextX86* context);

void InitializeCPUContextX86(const SignalThreadContext32& thread_context,
                             const SignalFloatContext32& float_context,
                             CPUContextX86* context);
//! \}

//! \brief Initializes GPR and debug state in a CPUContextX86 from a native
//!     signal context structure on Linux.
//!
//! Floating point state and debug registers are initialized to zero.
//!
//! \param[in] thread_context The native thread context.
//! \param[out] context The CPUContextX86 structure to initialize.
void InitializeCPUContextX86_NoFloatingPoint(
    const SignalThreadContext32& thread_context,
    CPUContextX86* context);

//! \{
//! \brief Initializes a CPUContextX86_64 structure from native context
//!     structures on Linux.
//!
//! \param[in] thread_context The native thread context.
//! \param[in] float_context The native float context.
//! \param[out] context The CPUContextX86_64 structure to initialize.
void InitializeCPUContextX86_64(const ThreadContext::t64_t& thread_context,
                                const FloatContext::f64_t& float_context,
                                CPUContextX86_64* context);

void InitializeCPUContextX86_64(const SignalThreadContext64& thread_context,
                                const SignalFloatContext64& float_context,
                                CPUContextX86_64* context);
//! \}

//! \brief Initializes GPR and debug state in a CPUContextX86_64 from a native
//!     signal context structure on Linux.
//!
//! Floating point state and debug registers are initialized to zero.
//!
//! \param[in] thread_context The native thread context.
//! \param[out] context The CPUContextX86_64 structure to initialize.
void InitializeCPUContextX86_64_NoFloatingPoint(
    const SignalThreadContext64& thread_context,
    CPUContextX86_64* context);

#endif  // ARCH_CPU_X86_FAMILY || DOXYGEN

#if defined(ARCH_CPU_ARM_FAMILY) || DOXYGEN

//! \brief Initializes a CPUContextARM structure from native context structures
//!     on Linux.
//!
//! \param[in] thread_context The native thread context.
//! \param[in] float_context The native float context.
//! \param[out] context The CPUContextARM structure to initialize.
void InitializeCPUContextARM(const ThreadContext::t32_t& thread_context,
                             const FloatContext::f32_t& float_context,
                             CPUContextARM* context);

//! \brief Initializes GPR state in a CPUContextARM from a native signal context
//!     structure on Linux.
//!
//! Floating point state is initialized to zero.
//!
//! \param[in] thread_context The native thread context.
//! \param[out] context The CPUContextARM structure to initialize.
void InitializeCPUContextARM_NoFloatingPoint(
    const SignalThreadContext32& thread_context,
    CPUContextARM* context);

//! \brief Initializes a CPUContextARM64 structure from native context
//!     structures on Linux.
//!
//! \param[in] thread_context The native thread context.
//! \param[in] float_context The native float context.
//! \param[out] context The CPUContextARM64 structure to initialize.
void InitializeCPUContextARM64(const ThreadContext::t64_t& thread_context,
                               const FloatContext::f64_t& float_context,
                               CPUContextARM64* context);

//! \brief Initializes GPR state in a CPUContextARM64 from a native context
//!     structure on Linux.
//!
//! Floating point state is initialized to zero.
//!
//! \param[in] thread_context The native thread context.
//! \param[out] context The CPUContextARM64 structure to initialize.
void InitializeCPUContextARM64_NoFloatingPoint(
    const ThreadContext::t64_t& thread_context,
    CPUContextARM64* context);

//! \brief Initializes FPSIMD state in a CPUContextARM64 from a native fpsimd
//!     signal context structure on Linux.
//!
//! General purpose registers are not initialized.
//!
//! \param[in] float_context The native fpsimd context.
//! \param[out] context The CPUContextARM64 structure to initialize.
void InitializeCPUContextARM64_OnlyFPSIMD(
    const SignalFPSIMDContext& float_context,
    CPUContextARM64* context);

#endif  // ARCH_CPU_ARM_FAMILY || DOXYGEN

#if defined(ARCH_CPU_MIPS_FAMILY) || DOXYGEN

//! \brief Initializes a CPUContextMIPS structure from native context
//!     structures on Linux.
//!
//! This function has template specializations for MIPSEL and MIPS64EL
//! architecture contexts, using ContextTraits32 or ContextTraits64 as template
//! parameter, respectively.
//!
//! \param[in] thread_context The native thread context.
//! \param[in] float_context The native float context.
//! \param[out] context The CPUContextMIPS structure to initialize.
template <typename Traits>
void InitializeCPUContextMIPS(
    const typename Traits::SignalThreadContext& thread_context,
    const typename Traits::SignalFloatContext& float_context,
    typename Traits::CPUContext* context) {
  static_assert(sizeof(context->regs) == sizeof(thread_context.regs),
                "registers size mismatch");
  static_assert(sizeof(context->fpregs) == sizeof(float_context.fpregs),
                "fp registers size mismatch");
  memcpy(&context->regs, &thread_context.regs, sizeof(context->regs));
  context->mdlo = thread_context.lo;
  context->mdhi = thread_context.hi;
  context->cp0_epc = thread_context.cp0_epc;
  context->cp0_badvaddr = thread_context.cp0_badvaddr;
  context->cp0_status = thread_context.cp0_status;
  context->cp0_cause = thread_context.cp0_cause;

  memcpy(&context->fpregs, &float_context.fpregs, sizeof(context->fpregs));
  context->fpcsr = float_context.fpcsr;
  context->fir = float_context.fpu_id;
};

#endif  // ARCH_CPU_MIPS_FAMILY || DOXYGEN

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_LINUX_CPU_CONTEXT_LINUX_H_
