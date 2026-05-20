// Common/MyInitGuid.h

#ifndef ZIP7_INC_COMMON_MY_INITGUID_H
#define ZIP7_INC_COMMON_MY_INITGUID_H

/*
This file must be included only to one C++ file in project before
declarations of COM interfaces with DEFINE_GUID macro.

Each GUID must be initialized exactly once in project.
There are two different versions of the DEFINE_GUID macro in guiddef.h (MyGuidDef.h):
  - if INITGUID is not defined:  DEFINE_GUID declares an external reference to the symbol name.
  - if INITGUID is     defined:  DEFINE_GUID initializes the symbol name to the value of the GUID.

Also we need IID_IUnknown that is initialized in some file for linking:
  MSVC:  by default the linker uses some lib file that contains IID_IUnknown
  MinGW: add -luuid switch for linker
  WinCE: we define IID_IUnknown in this file
  Other: we define IID_IUnknown in this file
*/

// #include "Common.h"
/* vc6 without sdk needs <objbase.h> before <initguid.h>,
   but it doesn't work in new msvc.
   So we include full "MyWindows.h" instead of <objbase.h> */
// #include <objbase.h>
#include "MyWindows.h"

#ifdef _WIN32

#ifdef __clang__
  // #pragma GCC diagnostic ignored "-Wmissing-variable-declarations"
#endif

#ifdef UNDER_CE
#include <basetyps.h>
#endif

// for vc6 without sdk we must define INITGUID here
#define INITGUID
#include <initguid.h>

#ifdef UNDER_CE
DEFINE_GUID(IID_IUnknown,
0x00000000, 0x0000, 0x0000, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46);
#endif

#else // _WIN32

#define INITGUID
#include "MyGuidDef.h"
DEFINE_GUID(IID_IUnknown,
0x00000000, 0x0000, 0x0000, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46);

#endif // _WIN32

#endif
