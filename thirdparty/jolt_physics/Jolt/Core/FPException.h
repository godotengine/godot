// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Core/FPControlWord.h>

JPH_NAMESPACE_BEGIN

#ifdef JPH_FLOATING_POINT_EXCEPTIONS_ENABLED

#if defined(JPH_CPU_WASM)

// Not supported
class FPExceptionsEnable { };
class FPExceptionDisableInvalid { };
class FPExceptionDisableDivByZero { };
class FPExceptionDisableOverflow { };

#elif defined(JPH_USE_SSE)

/// Enable floating point divide by zero exception, overflow exceptions and exceptions on invalid numbers
class FPExceptionsEnable : public FPControlWord<0, _MM_MASK_DIV_ZERO | _MM_MASK_INVALID | _MM_MASK_OVERFLOW> { };

/// Disable invalid floating point value exceptions
class FPExceptionDisableInvalid : public FPControlWord<_MM_MASK_INVALID, _MM_MASK_INVALID> { };

/// Disable division by zero floating point exceptions
class FPExceptionDisableDivByZero : public FPControlWord<_MM_MASK_DIV_ZERO, _MM_MASK_DIV_ZERO> { };

/// Disable floating point overflow exceptions
class FPExceptionDisableOverflow : public FPControlWord<_MM_MASK_OVERFLOW, _MM_MASK_OVERFLOW> { };

#elif defined(JPH_CPU_ARM) && defined(JPH_COMPILER_MSVC)

/// Enable floating point divide by zero exception, overflow exceptions and exceptions on invalid numbers
class FPExceptionsEnable : public FPControlWord<0, _EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW> { };

/// Disable invalid floating point value exceptions
class FPExceptionDisableInvalid : public FPControlWord<_EM_INVALID, _EM_INVALID> { };

/// Disable division by zero floating point exceptions
class FPExceptionDisableDivByZero : public FPControlWord<_EM_ZERODIVIDE, _EM_ZERODIVIDE> { };

/// Disable floating point overflow exceptions
class FPExceptionDisableOverflow : public FPControlWord<_EM_OVERFLOW, _EM_OVERFLOW> { };

#elif defined(JPH_CPU_ARM)

/// Invalid operation exception bit
static constexpr uint64 FP_IOE = 1 << 8;

/// Enable divide by zero exception bit
static constexpr uint64 FP_DZE = 1 << 9;

/// Enable floating point overflow bit
static constexpr uint64 FP_OFE = 1 << 10;

/// Enable floating point divide by zero exception, overflow exceptions and exceptions on invalid numbers
class FPExceptionsEnable : public FPControlWord<FP_IOE | FP_DZE | FP_OFE, FP_IOE | FP_DZE | FP_OFE> { };

/// Disable invalid floating point value exceptions
class FPExceptionDisableInvalid : public FPControlWord<0, FP_IOE> { };

/// Disable division by zero floating point exceptions
class FPExceptionDisableDivByZero : public FPControlWord<0, FP_DZE> { };

/// Disable floating point overflow exceptions
class FPExceptionDisableOverflow : public FPControlWord<0, FP_OFE> { };

#elif defined(JPH_CPU_RISCV)

#error "RISC-V only implements manually checking if exceptions occurred by reading the fcsr register. It doesn't generate exceptions. JPH_FLOATING_POINT_EXCEPTIONS_ENABLED must be disabled."

#elif defined(JPH_CPU_PPC)

#error PowerPC floating point exception handling to be implemented. JPH_FLOATING_POINT_EXCEPTIONS_ENABLED must be disabled.

#else

#error Unsupported CPU architecture

#endif

#else

/// Dummy implementations
class FPExceptionsEnable { };
class FPExceptionDisableInvalid { };
class FPExceptionDisableDivByZero { };
class FPExceptionDisableOverflow { };

#endif

JPH_NAMESPACE_END
