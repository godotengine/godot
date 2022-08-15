//===- llvm/Support/Debug.h - Easy way to add debug output ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a handy way of adding debugging information to your
// code, without it being enabled all of the time, and without having to add
// command line options to enable it.
//
// In particular, just wrap your code with the DEBUG() macro, and it will be
// enabled automatically if you specify '-debug' on the command-line.
// Alternatively, you can also define the DEBUG_TYPE macro to "foo" specify
// that your debug code belongs to class "foo". Be careful that you only do
// this after including Debug.h and not around any #include of headers. Headers
// should define and undef the macro acround the code that needs to use the
// DEBUG() macro. Then, on the command line, you can specify '-debug-only=foo'
// to enable JUST the debug information for the foo class.
//
// When compiling without assertions, the -debug-* options and all code in
// DEBUG() statements disappears, so it does not affect the runtime of the code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DEBUG_H
#define LLVM_SUPPORT_DEBUG_H

namespace llvm {
class raw_ostream;

#ifndef NDEBUG
/// DebugFlag - This boolean is set to true if the '-debug' command line option
/// is specified.  This should probably not be referenced directly, instead, use
/// the DEBUG macro below.
///
extern bool DebugFlag;

/// isCurrentDebugType - Return true if the specified string is the debug type
/// specified on the command line, or if none was specified on the command line
/// with the -debug-only=X option.
///
bool isCurrentDebugType(const char *Type);

/// setCurrentDebugType - Set the current debug type, as if the -debug-only=X
/// option were specified.  Note that DebugFlag also needs to be set to true for
/// debug output to be produced.
///
void setCurrentDebugType(const char *Type);

/// DEBUG_WITH_TYPE macro - This macro should be used by passes to emit debug
/// information.  In the '-debug' option is specified on the commandline, and if
/// this is a debug build, then the code specified as the option to the macro
/// will be executed.  Otherwise it will not be.  Example:
///
/// DEBUG_WITH_TYPE("bitset", dbgs() << "Bitset contains: " << Bitset << "\n");
///
/// This will emit the debug information if -debug is present, and -debug-only
/// is not specified, or is specified as "bitset".
#define DEBUG_WITH_TYPE(TYPE, X)                                        \
  do { if (::llvm::DebugFlag && ::llvm::isCurrentDebugType(TYPE)) { X; } \
  } while (0)

// HLSL Change
#define DEBUG_WITH_TYPE_IF(C, TYPE, X)                                        \
  do { if ((C) && ::llvm::DebugFlag && ::llvm::isCurrentDebugType(TYPE)) { X; __BREAK_ON_FAIL; } \
  } while (0)

#else
#define isCurrentDebugType(X) (false)
#define setCurrentDebugType(X)
#define DEBUG_WITH_TYPE(TYPE, X) do { } while (0)
// HLSL Change
#define DEBUG_WITH_TYPE_IF(C, TYPE, X) do { } while (0)

#endif

/// EnableDebugBuffering - This defaults to false.  If true, the debug
/// stream will install signal handlers to dump any buffered debug
/// output.  It allows clients to selectively allow the debug stream
/// to install signal handlers if they are certain there will be no
/// conflict.
///
extern bool EnableDebugBuffering;

/// dbgs() - This returns a reference to a raw_ostream for debugging
/// messages.  If debugging is disabled it returns errs().  Use it
/// like: dbgs() << "foo" << "bar";
raw_ostream &dbgs();

// DEBUG macro - This macro should be used by passes to emit debug information.
// In the '-debug' option is specified on the commandline, and if this is a
// debug build, then the code specified as the option to the macro will be
// executed.  Otherwise it will not be.  Example:
//
// DEBUG(dbgs() << "Bitset contains: " << Bitset << "\n");
//
#define DEBUG(X) DEBUG_WITH_TYPE(DEBUG_TYPE, X)

// HLSL Change
#define DEBUG_IF(C, X) DEBUG_WITH_TYPE_IF(C, DEBUG_TYPE, X)


} // End llvm namespace

#endif
