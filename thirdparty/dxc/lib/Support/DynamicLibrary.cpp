//===-- DynamicLibrary.cpp - Runtime link/load libraries --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the operating system DynamicLibrary concept.
//
// FIXME: This file leaks ExplicitSymbols and OpenedHandles!
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DynamicLibrary.h"
#include "llvm-c/Support.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Config/config.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include <cstdio>
#include <cstring>

// Collection of symbol name/value pairs to be searched prior to any libraries.
static llvm::ManagedStatic<llvm::StringMap<void *> > ExplicitSymbols;
static llvm::ManagedStatic<llvm::sys::SmartMutex<true> > SymbolsMutex;

void llvm::sys::DynamicLibrary::AddSymbol(StringRef symbolName,
                                          void *symbolValue) {
  SmartScopedLock<true> lock(*SymbolsMutex);
  (*ExplicitSymbols)[symbolName] = symbolValue;
}

char llvm::sys::DynamicLibrary::Invalid = 0;

#ifdef LLVM_ON_WIN32

#include "Windows/DynamicLibrary.inc"

#else

#if HAVE_DLFCN_H
#include <dlfcn.h>
using namespace llvm;
using namespace llvm::sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

static DenseSet<void *> *OpenedHandles = nullptr;

DynamicLibrary DynamicLibrary::getPermanentLibrary(const char *filename,
                                                   std::string *errMsg) {
  SmartScopedLock<true> lock(*SymbolsMutex);

  void *handle = dlopen(filename, RTLD_LAZY|RTLD_GLOBAL);
  if (!handle) {
    if (errMsg) *errMsg = dlerror();
    return DynamicLibrary();
  }

#ifdef __CYGWIN__
  // Cygwin searches symbols only in the main
  // with the handle of dlopen(NULL, RTLD_GLOBAL).
  if (!filename)
    handle = RTLD_DEFAULT;
#endif

  if (!OpenedHandles)
    OpenedHandles = new DenseSet<void *>();

  // If we've already loaded this library, dlclose() the handle in order to
  // keep the internal refcount at +1.
  if (!OpenedHandles->insert(handle).second)
    dlclose(handle);

  return DynamicLibrary(handle);
}

void *DynamicLibrary::getAddressOfSymbol(const char *symbolName) {
  if (!isValid())
    return nullptr;
  return dlsym(Data, symbolName);
}

#else

using namespace llvm;
using namespace llvm::sys;

DynamicLibrary DynamicLibrary::getPermanentLibrary(const char *filename,
                                                   std::string *errMsg) {
  if (errMsg) *errMsg = "dlopen() not supported on this platform";
  return DynamicLibrary();
}

void *DynamicLibrary::getAddressOfSymbol(const char *symbolName) {
  return NULL;
}

#endif

namespace llvm {
void *SearchForAddressOfSpecialSymbol(const char* symbolName);
}

void* DynamicLibrary::SearchForAddressOfSymbol(const char *symbolName) {
  SmartScopedLock<true> Lock(*SymbolsMutex);

  // First check symbols added via AddSymbol().
  if (ExplicitSymbols.isConstructed()) {
    StringMap<void *>::iterator i = ExplicitSymbols->find(symbolName);

    if (i != ExplicitSymbols->end())
      return i->second;
  }

#if HAVE_DLFCN_H
  // Now search the libraries.
  if (OpenedHandles) {
    for (DenseSet<void *>::iterator I = OpenedHandles->begin(),
         E = OpenedHandles->end(); I != E; ++I) {
      //lt_ptr ptr = lt_dlsym(*I, symbolName);
      void *ptr = dlsym(*I, symbolName);
      if (ptr) {
        return ptr;
      }
    }
  }
#endif

  if (void *Result = llvm::SearchForAddressOfSpecialSymbol(symbolName))
    return Result;

// This macro returns the address of a well-known, explicit symbol
#define EXPLICIT_SYMBOL(SYM)                                                   \
  if (!strcmp(symbolName, #SYM))                                               \
    return (void *)&SYM

// On linux we have a weird situation. The stderr/out/in symbols are both
// macros and global variables because of standards requirements. So, we
// boldly use the EXPLICIT_SYMBOL macro without checking for a #define first.
#if defined(__linux__) and !defined(__ANDROID__)
  {
    EXPLICIT_SYMBOL(stderr);
    EXPLICIT_SYMBOL(stdout);
    EXPLICIT_SYMBOL(stdin);
  }
#else
  // For everything else, we want to check to make sure the symbol isn't defined
  // as a macro before using EXPLICIT_SYMBOL.
  {
#ifndef stdin
    EXPLICIT_SYMBOL(stdin);
#endif
#ifndef stdout
    EXPLICIT_SYMBOL(stdout);
#endif
#ifndef stderr
    EXPLICIT_SYMBOL(stderr);
#endif
  }
#endif
#undef EXPLICIT_SYMBOL

  return nullptr;
}

#endif // LLVM_ON_WIN32

//===----------------------------------------------------------------------===//
// C API.
//===----------------------------------------------------------------------===//

LLVMBool LLVMLoadLibraryPermanently(const char* Filename) {
  return llvm::sys::DynamicLibrary::LoadLibraryPermanently(Filename);
}

void *LLVMSearchForAddressOfSymbol(const char *symbolName) {
  return llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(symbolName);
}

void LLVMAddSymbol(const char *symbolName, void *symbolValue) {
  return llvm::sys::DynamicLibrary::AddSymbol(symbolName, symbolValue);
}

