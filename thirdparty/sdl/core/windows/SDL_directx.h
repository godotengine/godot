/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#ifndef SDL_directx_h_
#define SDL_directx_h_

// Include all of the DirectX 8.0 headers and adds any necessary tweaks

#include "SDL_windows.h"
#include <mmsystem.h>
#ifndef WIN32
#define WIN32
#endif
#undef WINNT

// Far pointers don't exist in 32-bit code
#ifndef FAR
#define FAR
#endif

// Error codes not yet included in Win32 API header files
#ifndef MAKE_HRESULT
#define MAKE_HRESULT(sev, fac, code) \
    ((HRESULT)(((unsigned long)(sev) << 31) | ((unsigned long)(fac) << 16) | ((unsigned long)(code))))
#endif

#ifndef S_OK
#define S_OK (HRESULT)0x00000000L
#endif

#ifndef SUCCEEDED
#define SUCCEEDED(x) ((HRESULT)(x) >= 0)
#endif
#ifndef FAILED
#define FAILED(x) ((HRESULT)(x) < 0)
#endif

#ifndef E_FAIL
#define E_FAIL (HRESULT)0x80000008L
#endif
#ifndef E_NOINTERFACE
#define E_NOINTERFACE (HRESULT)0x80004002L
#endif
#ifndef E_OUTOFMEMORY
#define E_OUTOFMEMORY (HRESULT)0x8007000EL
#endif
#ifndef E_INVALIDARG
#define E_INVALIDARG (HRESULT)0x80070057L
#endif
#ifndef E_NOTIMPL
#define E_NOTIMPL (HRESULT)0x80004001L
#endif
#ifndef REGDB_E_CLASSNOTREG
#define REGDB_E_CLASSNOTREG (HRESULT)0x80040154L
#endif

// Severity codes
#ifndef SEVERITY_ERROR
#define SEVERITY_ERROR 1
#endif

// Error facility codes
#ifndef FACILITY_WIN32
#define FACILITY_WIN32 7
#endif

#ifndef FIELD_OFFSET
#define FIELD_OFFSET(type, field) ((LONG) & (((type *)0)->field))
#endif

/* DirectX headers (if it isn't included, I haven't tested it yet)
 */
// We need these defines to mark what version of DirectX API we use
#define DIRECTDRAW_VERSION  0x0700
#define DIRECTSOUND_VERSION 0x0800
#define DIRECTINPUT_VERSION 0x0800 // Need version 7 for force feedback. Need version 8 so IDirectInput8_EnumDevices doesn't leak like a sieve...

#ifdef HAVE_DDRAW_H
#include <ddraw.h>
#endif
#ifdef HAVE_DSOUND_H
#include <dsound.h>
#endif
#ifdef HAVE_DINPUT_H
#include <dinput.h>
#else
typedef struct
{
    int unused;
} DIDEVICEINSTANCE;
#endif

#endif // SDL_directx_h_
