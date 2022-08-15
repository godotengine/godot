//===- llvm/Support/PrettyStackTrace.h - Pretty Crash Handling --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PrettyStackTraceEntry class, which is used to make
// crashes give more contextual information about what the program was doing
// when it crashed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PRETTYSTACKTRACE_H
#define LLVM_SUPPORT_PRETTYSTACKTRACE_H

#include "llvm/Support/Compiler.h"

namespace llvm {
  class raw_ostream;

  void EnablePrettyStackTrace();

  /// PrettyStackTraceEntry - This class is used to represent a frame of the
  /// "pretty" stack trace that is dumped when a program crashes. You can define
  /// subclasses of this and declare them on the program stack: when they are
  /// constructed and destructed, they will add their symbolic frames to a
  /// virtual stack trace.  This gets dumped out if the program crashes.
  class PrettyStackTraceEntry {
    const PrettyStackTraceEntry *NextEntry;
    PrettyStackTraceEntry(const PrettyStackTraceEntry &) = delete;
    void operator=(const PrettyStackTraceEntry&) = delete;
  public:
    PrettyStackTraceEntry();
    virtual ~PrettyStackTraceEntry();

    /// print - Emit information about this stack frame to OS.
    virtual void print(raw_ostream &OS) const = 0;

    /// getNextEntry - Return the next entry in the list of frames.
    const PrettyStackTraceEntry *getNextEntry() const { return NextEntry; }
  };

  /// PrettyStackTraceString - This object prints a specified string (which
  /// should not contain newlines) to the stream as the stack trace when a crash
  /// occurs.
  class PrettyStackTraceString : public PrettyStackTraceEntry {
    const char *Str;
  public:
    PrettyStackTraceString(const char *str) : Str(str) {}
    void print(raw_ostream &OS) const override;
  };

  /// PrettyStackTraceProgram - This object prints a specified program arguments
  /// to the stream as the stack trace when a crash occurs.
  class PrettyStackTraceProgram : public PrettyStackTraceEntry {
    int ArgC;
    const char *const *ArgV;
  public:
    PrettyStackTraceProgram(int argc, const char * const*argv)
      : ArgC(argc), ArgV(argv) {
      EnablePrettyStackTrace();
    }
    void print(raw_ostream &OS) const override;
  };

} // end namespace llvm

#endif
