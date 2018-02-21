/*
 * StdAfx.h
 * --------
 * Purpose: Include file for standard system include files, or project specific include files that are used frequently, but are changed infrequently. Also includes the global build settings from BuildSettings.h.
 * Notes  : (currently none)
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


// has to be first
#include "BuildSettings.h"


#if defined(MODPLUG_TRACKER)

#if MPT_OS_WINDOWS

#if !defined(MPT_BUILD_WINESUPPORT)

#define _AFX_NO_MFC_CONTROLS_IN_DIALOGS	// Do not include support for MFC controls in dialogs (reduces binary bloat; remove this #define if you want to use MFC controls)
#include <afxwin.h>         // MFC core and standard components
#include <afxext.h>         // MFC extensions
#include <afxcmn.h>			// MFC support for Windows Common Controls
#include <afxcview.h>
#include <afxdlgs.h>
#include <afxole.h>

#endif // !MPT_BUILD_WINESUPPORT

#include <windows.h>
#include <windowsx.h>
#include <shlwapi.h>
#include <mmsystem.h>

#endif // MPT_OS_WINDOWS

#endif // MODPLUG_TRACKER


#if MPT_COMPILER_MSVC
#include <intrin.h>
#endif


#if MPT_OS_WINDOWS
#if MPT_COMPILER_MSVCCLANGC2
// windows.h references IUnknown in a template function without having it even forward-declared.
// Clang does not like that. Forward-declaration fixes it.
struct IUnknown;
#endif
#endif


// this will be available everywhere

#include "../common/typedefs.h"
// <memory>
// <new>
// <climits>
// <cstddef>
// <cstdint>
// <stdint.h>

#include "../common/mptTypeTraits.h"
// <type_traits>

#include "../common/mptString.h"
// <algorithm>
// <limits>
// <string>
// <type_traits>
// <cstring>

#include "../common/mptStringFormat.h"

#include "../common/mptPathString.h"

#include "../common/Logging.h"

#include "../common/misc_util.h"
// <algorithm>
// <limits>
// <string>
// <type_traits>
// <vector>
// <cmath>
// <cstdlib>
// <cstring>
// <time.h>

// for std::abs
#include <cstdlib>
#include <stdlib.h>
#include <cmath>
#include <math.h>

#if defined(MPT_ENABLE_FILEIO_STDIO)
// for FILE* definition (which cannot be forward-declared in a portable way)
#include <cstdio>
#include <stdio.h>
#endif

#ifndef NO_VST
// VST SDK includes these headers after messing with default compiler structure
// packing. No problem in practice as VST SDK sets packing matching the default
// packing and we are compiling with default packing and standard headers should
// be careful about structure packing anyway, but it is very much unclean
// nonetheless. Pre-include the affected headers here as a future-proof
// safe-guard and let their own include guards handle the further including by
// VST SDK.
#include <cstdint>
#include <stdint.h>
#include <cstring>
#include <string.h>
#endif


//{{AFX_INSERT_LOCATION}}
// Microsoft Developer Studio will insert additional declarations immediately before the previous line.
