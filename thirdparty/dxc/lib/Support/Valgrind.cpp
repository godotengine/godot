//===-- Valgrind.cpp - Implement Valgrind communication ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Defines Valgrind communication methods, if HAVE_VALGRIND_VALGRIND_H is
//  defined.  If we have valgrind.h but valgrind isn't running, its macros are
//  no-ops.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Valgrind.h"
#include "llvm/Config/config.h"

#if HAVE_VALGRIND_VALGRIND_H
#include <valgrind/valgrind.h>

static bool InitNotUnderValgrind() {
  return !RUNNING_ON_VALGRIND;
}

// This bool is negated from what we'd expect because code may run before it
// gets initialized.  If that happens, it will appear to be 0 (false), and we
// want that to cause the rest of the code in this file to run the
// Valgrind-provided macros.
static const bool NotUnderValgrind = InitNotUnderValgrind();

bool llvm::sys::RunningOnValgrind() {
  if (NotUnderValgrind)
    return false;
  return RUNNING_ON_VALGRIND;
}

void llvm::sys::ValgrindDiscardTranslations(const void *Addr, size_t Len) {
  if (NotUnderValgrind)
    return;

  VALGRIND_DISCARD_TRANSLATIONS(Addr, Len);
}

#else  // !HAVE_VALGRIND_VALGRIND_H

bool llvm::sys::RunningOnValgrind() {
  return false;
}

void llvm::sys::ValgrindDiscardTranslations(const void *Addr, size_t Len) {
}

#endif  // !HAVE_VALGRIND_VALGRIND_H

// These functions require no implementation, tsan just looks at the arguments
// they're called with. However, they are required to be weak as some other
// application or library may already be providing these definitions for the
// same reason we are.
// extern "C" {   // HLSL Change -Don't use c linkage.
LLVM_ATTRIBUTE_WEAK void AnnotateHappensAfter(const char *file, int line,
                                              const volatile void *cv);
void AnnotateHappensAfter(const char *file, int line, const volatile void *cv) {
}              
LLVM_ATTRIBUTE_WEAK void AnnotateHappensBefore(const char *file, int line,
                                               const volatile void *cv);
void AnnotateHappensBefore(const char *file, int line,
                           const volatile void *cv) {}
LLVM_ATTRIBUTE_WEAK void AnnotateIgnoreWritesBegin(const char *file, int line);
void AnnotateIgnoreWritesBegin(const char *file, int line) {}
LLVM_ATTRIBUTE_WEAK void AnnotateIgnoreWritesEnd(const char *file, int line);
void AnnotateIgnoreWritesEnd(const char *file, int line) {}
// }  // HLSL Change -Don't use c linkage.

