//===- llvm/Support/ErrorHandling.h - Fatal error handling ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an API used to indicate fatal error conditions.  Non-fatal
// errors (most of them) should be handled through LLVMContext.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ERRORHANDLING_H
#define LLVM_SUPPORT_ERRORHANDLING_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include <string>

namespace llvm {
  class Twine;

  /// An error handler callback.
  typedef void (*fatal_error_handler_t)(void *user_data,
                                        const std::string& reason,
                                        bool gen_crash_diag);

  /// install_fatal_error_handler - Installs a new error handler to be used
  /// whenever a serious (non-recoverable) error is encountered by LLVM.
  ///
  /// If no error handler is installed the default is to print the error message
  /// to stderr, and call exit(1).  If an error handler is installed then it is
  /// the handler's responsibility to log the message, it will no longer be
  /// printed to stderr.  If the error handler returns, then exit(1) will be
  /// called.
  ///
  /// It is dangerous to naively use an error handler which throws an exception.
  /// Even though some applications desire to gracefully recover from arbitrary
  /// faults, blindly throwing exceptions through unfamiliar code isn't a way to
  /// achieve this.
  ///
  /// \param user_data - An argument which will be passed to the install error
  /// handler.
  void install_fatal_error_handler(fatal_error_handler_t handler,
                                   void *user_data = nullptr);

  /// Restores default error handling behaviour.
  void remove_fatal_error_handler();

  /// ScopedFatalErrorHandler - This is a simple helper class which just
  /// calls install_fatal_error_handler in its constructor and
  /// remove_fatal_error_handler in its destructor.
  struct ScopedFatalErrorHandler {
    explicit ScopedFatalErrorHandler(fatal_error_handler_t handler,
                                     void *user_data = nullptr) {
      install_fatal_error_handler(handler, user_data);
    }

    ~ScopedFatalErrorHandler() { remove_fatal_error_handler(); }
  };

  /// Reports a serious error, calling any installed error handler. These
  /// functions are intended to be used for error conditions which are outside
  /// the control of the compiler (I/O errors, invalid user input, etc.)
  ///
  /// If no error handler is installed the default is to print the message to
  /// standard error, followed by a newline.
  /// After the error handler is called this function will call exit(1), it 
  /// does not return.
  LLVM_ATTRIBUTE_NORETURN void report_fatal_error(const char *reason,
                                                  bool gen_crash_diag = true);
  LLVM_ATTRIBUTE_NORETURN void report_fatal_error(const std::string &reason,
                                                  bool gen_crash_diag = true);
  LLVM_ATTRIBUTE_NORETURN void report_fatal_error(StringRef reason,
                                                  bool gen_crash_diag = true);
  LLVM_ATTRIBUTE_NORETURN void report_fatal_error(const Twine &reason,
                                                  bool gen_crash_diag = true);

  /// This function calls abort(), and prints the optional message to stderr.
  /// Use the llvm_unreachable macro (that adds location info), instead of
  /// calling this function directly.
  LLVM_ATTRIBUTE_NORETURN void
  llvm_unreachable_internal(const char *msg=nullptr, const char *file=nullptr,
                            unsigned line=0);

  // HLSL Change - throw special exception for cast mismatch
  void llvm_cast_assert_internal(const char *func);
}

/// Marks that the current location is not supposed to be reachable.
/// In !NDEBUG builds, prints the message and location info to stderr.
/// In NDEBUG builds, becomes an optimizer hint that the current location
/// is not supposed to be reachable.  On compilers that don't support
/// such hints, prints a reduced message instead.
///
/// Use this instead of assert(0).  It conveys intent more clearly and
/// allows compilers to omit some unnecessary code.
#if 1 // HLSL Change - always throw exception with message for unreachable
#define llvm_unreachable(msg) \
  ::llvm::llvm_unreachable_internal(msg, __FILE__, __LINE__)
#elif defined(LLVM_BUILTIN_UNREACHABLE)
#define llvm_unreachable(msg) LLVM_BUILTIN_UNREACHABLE
#else
#define llvm_unreachable(msg) ::llvm::llvm_unreachable_internal()
#endif

// HLSL Change - throw special exception for cast type mismatch
#define llvm_cast_assert(X, Val) ((void)( (!!(isa<X>(Val))) || (::llvm::llvm_cast_assert_internal(__FUNCTION__), 0) ))

#endif
