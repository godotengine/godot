//===- Win32/DynamicLibrary.cpp - Win32 DL Implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of DynamicLibrary.
//
//===----------------------------------------------------------------------===//

#include "WindowsSupport.h"

#ifdef __MINGW32__
 #include <imagehlp.h>
#else
 #include <dbghelp.h>
#endif

#ifdef _MSC_VER
 #include <ntverp.h>
#endif

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code
//===          and must not be UNIX code.
//===----------------------------------------------------------------------===//

typedef BOOL (WINAPI *fpEnumerateLoadedModules)(HANDLE,PENUMLOADED_MODULES_CALLBACK64,PVOID);
static fpEnumerateLoadedModules fEnumerateLoadedModules;
static DenseSet<HMODULE> *OpenedHandles;

static bool loadDebugHelp(void) {
  HMODULE hLib = ::LoadLibraryW(L"Dbghelp.dll");
  if (hLib) {
    fEnumerateLoadedModules = (fpEnumerateLoadedModules)
      ::GetProcAddress(hLib, "EnumerateLoadedModules64");
  }
  return fEnumerateLoadedModules != 0;
}

static BOOL CALLBACK
ELM_Callback(WIN32_ELMCB_PCSTR ModuleName, DWORD64 ModuleBase,
             ULONG ModuleSize, PVOID UserContext) {
  OpenedHandles->insert((HMODULE)ModuleBase);
  return TRUE;
}

DynamicLibrary DynamicLibrary::getPermanentLibrary(const char *filename,
                                                   std::string *errMsg) {
  SmartScopedLock<true> lock(*SymbolsMutex);

  if (!filename) {
    // When no file is specified, enumerate all DLLs and EXEs in the process.
    if (OpenedHandles == 0)
      OpenedHandles = new DenseSet<HMODULE>();

    if (!fEnumerateLoadedModules) {
      if (!loadDebugHelp()) {
        assert(false && "These APIs should always be available");
        return DynamicLibrary();
      }
    }

    fEnumerateLoadedModules(GetCurrentProcess(), ELM_Callback, 0);
    // Dummy library that represents "search all handles".
    // This is mostly to ensure that the return value still shows up as "valid".
    return DynamicLibrary(&OpenedHandles);
  }

  SmallVector<wchar_t, MAX_PATH> filenameUnicode;
  if (std::error_code ec = windows::UTF8ToUTF16(filename, filenameUnicode)) {
    SetLastError(ec.value());
    MakeErrMsg(errMsg, std::string(filename) + ": Can't convert to UTF-16: ");
    return DynamicLibrary();
  }
  
  HMODULE a_handle = LoadLibraryW(filenameUnicode.data());

  if (a_handle == 0) {
    MakeErrMsg(errMsg, std::string(filename) + ": Can't open : ");
    return DynamicLibrary();
  }

  if (OpenedHandles == 0)
    OpenedHandles = new DenseSet<HMODULE>();

  // If we've already loaded this library, FreeLibrary() the handle in order to
  // keep the internal refcount at +1.
  if (!OpenedHandles->insert(a_handle).second)
    FreeLibrary(a_handle);

  return DynamicLibrary(a_handle);
}

// Stack probing routines are in the support library (e.g. libgcc), but we don't
// have dynamic linking on windows. Provide a hook.
#define EXPLICIT_SYMBOL(SYM)                    \
  extern "C" { extern void *SYM; }
#define EXPLICIT_SYMBOL2(SYMFROM, SYMTO) EXPLICIT_SYMBOL(SYMTO)

#ifdef _M_IX86
// Win32 on x86 implements certain single-precision math functions as macros.
// These functions are not exported by the DLL, but will still be needed
// for symbol-resolution by the JIT loader. Therefore, this Support libray
// provides helper functions with the same implementation.

#define INLINE_DEF_SYMBOL1(TYP, SYM)                                           \
  extern "C" TYP inline_##SYM(TYP _X) { return SYM(_X); }
#define INLINE_DEF_SYMBOL2(TYP, SYM)                                           \
  extern "C" TYP inline_##SYM(TYP _X, TYP _Y) { return SYM(_X, _Y); }
#endif

#include "explicit_symbols.inc"

#undef EXPLICIT_SYMBOL
#undef EXPLICIT_SYMBOL2
#undef INLINE_DEF_SYMBOL1
#undef INLINE_DEF_SYMBOL2

void* DynamicLibrary::SearchForAddressOfSymbol(const char* symbolName) {
  SmartScopedLock<true> Lock(*SymbolsMutex);

  // First check symbols added via AddSymbol().
  if (ExplicitSymbols.isConstructed()) {
    StringMap<void *>::iterator i = ExplicitSymbols->find(symbolName);

    if (i != ExplicitSymbols->end())
      return i->second;
  }

  // Now search the libraries.
  if (OpenedHandles) {
    for (DenseSet<HMODULE>::iterator I = OpenedHandles->begin(),
         E = OpenedHandles->end(); I != E; ++I) {
      FARPROC ptr = GetProcAddress((HMODULE)*I, symbolName);
      if (ptr) {
        return (void *)(intptr_t)ptr;
      }
    }
  }

#define EXPLICIT_SYMBOL(SYM)                                                   \
  if (!strcmp(symbolName, #SYM))                                               \
    return (void *)&SYM;
#define EXPLICIT_SYMBOL2(SYMFROM, SYMTO)                                       \
  if (!strcmp(symbolName, #SYMFROM))                                           \
    return (void *)&SYMTO;

#ifdef _M_IX86
#define INLINE_DEF_SYMBOL1(TYP, SYM)                                           \
  if (!strcmp(symbolName, #SYM))                                               \
    return (void *)&inline_##SYM;
#define INLINE_DEF_SYMBOL2(TYP, SYM) INLINE_DEF_SYMBOL1(TYP, SYM)
#endif

  {
#include "explicit_symbols.inc"
  }

#undef EXPLICIT_SYMBOL
#undef EXPLICIT_SYMBOL2
#undef INLINE_DEF_SYMBOL1
#undef INLINE_DEF_SYMBOL2

  return 0;
}

void *DynamicLibrary::getAddressOfSymbol(const char *symbolName) {
  if (!isValid())
    return NULL;
  if (Data == &OpenedHandles)
    return SearchForAddressOfSymbol(symbolName);
  return (void *)(intptr_t)GetProcAddress((HMODULE)Data, symbolName);
}

}
