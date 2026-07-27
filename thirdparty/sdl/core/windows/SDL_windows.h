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

// This is an include file for windows.h with the SDL build settings

#ifndef _INCLUDED_WINDOWS_H
#define _INCLUDED_WINDOWS_H

#ifdef SDL_PLATFORM_WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef STRICT
#define STRICT 1
#endif
#ifndef UNICODE
#define UNICODE 1
#endif
#undef WINVER
#undef _WIN32_WINNT
#if defined(SDL_VIDEO_RENDER_D3D12) || defined(HAVE_DXGI1_6_H)
#define _WIN32_WINNT 0xA00 // For D3D12, 0xA00 is required
#elif defined(HAVE_SHELLSCALINGAPI_H)
#define _WIN32_WINNT 0x603 // For DPI support
#else
#define _WIN32_WINNT 0x501 // Need 0x410 for AlphaBlend() and 0x500 for EnumDisplayDevices(), 0x501 for raw input
#endif
#define WINVER _WIN32_WINNT

#elif defined(SDL_PLATFORM_WINGDK)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef STRICT
#define STRICT 1
#endif
#ifndef UNICODE
#define UNICODE 1
#endif
#undef WINVER
#undef _WIN32_WINNT
#define _WIN32_WINNT 0xA00
#define WINVER       _WIN32_WINNT

#elif defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef STRICT
#define STRICT 1
#endif
#ifndef UNICODE
#define UNICODE 1
#endif
#undef WINVER
#undef _WIN32_WINNT
#define _WIN32_WINNT 0xA00
#define WINVER       _WIN32_WINNT
#endif

// See https://github.com/libsdl-org/SDL/pull/7607
// force_align_arg_pointer attribute requires gcc >= 4.2.x.
#if defined(__clang__)
#define HAVE_FORCE_ALIGN_ARG_POINTER
#elif defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
#define HAVE_FORCE_ALIGN_ARG_POINTER
#endif
#if defined(__GNUC__) && defined(__i386__) && defined(HAVE_FORCE_ALIGN_ARG_POINTER)
#define MINGW32_FORCEALIGN __attribute__((force_align_arg_pointer))
#else
#define MINGW32_FORCEALIGN
#endif

#include <windows.h>
#include <basetyps.h> // for REFIID with broken mingw.org headers
#include <mmreg.h>

// Routines to convert from UTF8 to native Windows text
#define WIN_StringToUTF8W(S) SDL_iconv_string("UTF-8", "UTF-16LE", (const char *)(S), (SDL_wcslen(S) + 1) * sizeof(WCHAR))
#define WIN_UTF8ToStringW(S) (WCHAR *)SDL_iconv_string("UTF-16LE", "UTF-8", (const char *)(S), SDL_strlen(S) + 1)
// !!! FIXME: UTF8ToString() can just be a SDL_strdup() here.
#define WIN_StringToUTF8A(S) SDL_iconv_string("UTF-8", "ASCII", (const char *)(S), (SDL_strlen(S) + 1))
#define WIN_UTF8ToStringA(S) SDL_iconv_string("ASCII", "UTF-8", (const char *)(S), SDL_strlen(S) + 1)
#if UNICODE
#define WIN_StringToUTF8 WIN_StringToUTF8W
#define WIN_UTF8ToString WIN_UTF8ToStringW
#define SDL_tcslen       SDL_wcslen
#define SDL_tcsstr       SDL_wcsstr
#else
#define WIN_StringToUTF8 WIN_StringToUTF8A
#define WIN_UTF8ToString WIN_UTF8ToStringA
#define SDL_tcslen       SDL_strlen
#define SDL_tcsstr       SDL_strstr
#endif

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

// Sets an error message based on a given HRESULT
extern bool WIN_SetErrorFromHRESULT(const char *prefix, HRESULT hr);

// Sets an error message based on GetLastError(). Always returns false.
extern bool WIN_SetError(const char *prefix);

// Load a function from combase.dll
FARPROC WIN_LoadComBaseFunction(const char *name);

// Wrap up the oddities of CoInitialize() into a common function.
extern HRESULT WIN_CoInitialize(void);
extern void WIN_CoUninitialize(void);

// Wrap up the oddities of RoInitialize() into a common function.
extern HRESULT WIN_RoInitialize(void);
extern void WIN_RoUninitialize(void);

// Returns true if we're running on Wine
extern BOOL WIN_IsWine(void);

// Returns true if we're running on Windows XP (any service pack). DOES NOT CHECK XP "OR GREATER"!
extern BOOL WIN_IsWindowsXP(void);

// Returns true if we're running on Windows Vista and newer
extern BOOL WIN_IsWindowsVistaOrGreater(void);

// Returns true if we're running on Windows 7 and newer
extern BOOL WIN_IsWindows7OrGreater(void);

// Returns true if we're running on Windows 8 and newer
extern BOOL WIN_IsWindows8OrGreater(void);

// You need to SDL_free() the result of this call.
extern char *WIN_LookupAudioDeviceName(const WCHAR *name, const GUID *guid);

// Checks to see if two GUID are the same.
extern BOOL WIN_IsEqualGUID(const GUID *a, const GUID *b);
extern BOOL WIN_IsEqualIID(REFIID a, REFIID b);

// Convert between SDL_rect and RECT
extern void WIN_RECTToRect(const RECT *winrect, SDL_Rect *sdlrect);
extern void WIN_RectToRECT(const SDL_Rect *sdlrect, RECT *winrect);

// Returns false if a window client rect is not valid
bool WIN_WindowRectValid(const RECT *rect);

extern SDL_AudioFormat SDL_WaveFormatExToSDLFormat(WAVEFORMATEX *waveformat);

// WideCharToMultiByte, but with some WinXP management.
extern int WIN_WideCharToMultiByte(UINT CodePage, DWORD dwFlags, LPCWCH lpWideCharStr, int cchWideChar, LPSTR lpMultiByteStr, int cbMultiByte, LPCCH lpDefaultChar, LPBOOL lpUsedDefaultChar);

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif

#endif // _INCLUDED_WINDOWS_H
