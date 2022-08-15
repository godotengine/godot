//===- SearchForAddressOfSpecialSymbol.cpp - Function addresses -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file pulls the addresses of certain symbols out of the linker.  It must
//  include as few header files as possible because it declares the symbols as
//  void*, which would conflict with the actual symbol type if any header
//  declared it.
//
//===----------------------------------------------------------------------===//

#include <string.h>

// Must declare the symbols in the global namespace.
static void *DoSearch(const char* symbolName) {
#define EXPLICIT_SYMBOL(SYM) \
   extern void *SYM; if (!strcmp(symbolName, #SYM)) return &SYM

  // If this is darwin, it has some funky issues, try to solve them here.  Some
  // important symbols are marked 'private external' which doesn't allow
  // SearchForAddressOfSymbol to find them.  As such, we special case them here,
  // there is only a small handful of them.

#ifdef __APPLE__
  {
    // __eprintf is sometimes used for assert() handling on x86.
    //
    // FIXME: Currently disabled when using Clang, as we don't always have our
    // runtime support libraries available.
#ifndef __clang__
#ifdef __i386__
    EXPLICIT_SYMBOL(__eprintf);
#endif
#endif
  }
#endif

#ifdef __CYGWIN__
  {
    EXPLICIT_SYMBOL(_alloca);
    EXPLICIT_SYMBOL(__main);
  }
#endif

#undef EXPLICIT_SYMBOL
  return nullptr;
}

namespace llvm {
void *SearchForAddressOfSpecialSymbol(const char* symbolName) {
  return DoSearch(symbolName);
}
}  // namespace llvm
