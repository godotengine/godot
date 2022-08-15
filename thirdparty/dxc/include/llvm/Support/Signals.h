//===- llvm/Support/Signals.h - Signal Handling support ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// unix signals occurring while your program is running.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SIGNALS_H
#define LLVM_SUPPORT_SIGNALS_H

#include <string>
#include <llvm/ADT/StringRef.h> // HLSL Change - StringRef is, in fact, referenced directly

namespace llvm {
class StringRef;
class raw_ostream;

namespace sys {

  /// This function runs all the registered interrupt handlers, including the
  /// removal of files registered by RemoveFileOnSignal.
  void RunInterruptHandlers();

  /// This function registers signal handlers to ensure that if a signal gets
  /// delivered that the named file is removed.
  /// @brief Remove a file if a fatal signal occurs.
  bool RemoveFileOnSignal(StringRef Filename, std::string* ErrMsg = nullptr);

  /// This function removes a file from the list of files to be removed on
  /// signal delivery.
  void DontRemoveFileOnSignal(StringRef Filename);

  /// When an error signal (such as SIBABRT or SIGSEGV) is delivered to the
  /// process, print a stack trace and then exit.
  /// @brief Print a stack trace if a fatal signal occurs.
  void PrintStackTraceOnErrorSignal(bool DisableCrashReporting = false);

  /// Disable all system dialog boxes that appear when the process crashes.
  void DisableSystemDialogsOnCrash();

  /// \brief Print the stack trace using the given \c raw_ostream object.
  void PrintStackTrace(raw_ostream &OS);

  /// AddSignalHandler - Add a function to be called when an abort/kill signal
  /// is delivered to the process.  The handler can have a cookie passed to it
  /// to identify what instance of the handler it is.
  void AddSignalHandler(void (*FnPtr)(void *), void *Cookie);

  /// This function registers a function to be called when the user "interrupts"
  /// the program (typically by pressing ctrl-c).  When the user interrupts the
  /// program, the specified interrupt function is called instead of the program
  /// being killed, and the interrupt function automatically disabled.  Note
  /// that interrupt functions are not allowed to call any non-reentrant
  /// functions.  An null interrupt function pointer disables the current
  /// installed function.  Note also that the handler may be executed on a
  /// different thread on some platforms.
  /// @brief Register a function to be called when ctrl-c is pressed.
  void SetInterruptFunction(void (*IF)());
} // End sys namespace
} // End llvm namespace

#endif
