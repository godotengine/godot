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

/* WIKI CATEGORY: Platform */

/*
 * SDL_platform_defines.h tries to get a standard set of platform defines.
 */

#ifndef SDL_platform_defines_h_
#define SDL_platform_defines_h_

#ifdef _AIX

/**
 * A preprocessor macro that is only defined if compiling for AIX.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_AIX 1
#endif

#ifdef __HAIKU__

/**
 * A preprocessor macro that is only defined if compiling for Haiku OS.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_HAIKU 1
#endif

#if defined(bsdi) || defined(__bsdi) || defined(__bsdi__)

/**
 * A preprocessor macro that is only defined if compiling for BSDi
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_BSDI 1
#endif

#if defined(__FreeBSD__) || defined(__FreeBSD_kernel__) || defined(__DragonFly__)

/**
 * A preprocessor macro that is only defined if compiling for FreeBSD.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_FREEBSD 1
#endif

#if defined(hpux) || defined(__hpux) || defined(__hpux__)

/**
 * A preprocessor macro that is only defined if compiling for HP-UX.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_HPUX 1
#endif

#if defined(sgi) || defined(__sgi) || defined(__sgi__) || defined(_SGI_SOURCE)

/**
 * A preprocessor macro that is only defined if compiling for IRIX.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_IRIX 1
#endif

#if (defined(linux) || defined(__linux) || defined(__linux__))

/**
 * A preprocessor macro that is only defined if compiling for Linux.
 *
 * Note that Android, although ostensibly a Linux-based system, will not
 * define this. It defines SDL_PLATFORM_ANDROID instead.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_LINUX 1
#endif

#if defined(ANDROID) || defined(__ANDROID__)

/**
 * A preprocessor macro that is only defined if compiling for Android.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_ANDROID 1
#undef SDL_PLATFORM_LINUX
#endif

#if defined(__unix__) || defined(__unix) || defined(unix)

/**
 * A preprocessor macro that is only defined if compiling for a Unix-like
 * system.
 *
 * Other platforms, like Linux, might define this in addition to their primary
 * define.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_UNIX 1
#endif

#ifdef __APPLE__

/**
 * A preprocessor macro that is only defined if compiling for Apple platforms.
 *
 * iOS, macOS, etc will additionally define a more specific platform macro.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_PLATFORM_MACOS
 * \sa SDL_PLATFORM_IOS
 * \sa SDL_PLATFORM_TVOS
 * \sa SDL_PLATFORM_VISIONOS
 */
#define SDL_PLATFORM_APPLE 1

/* lets us know what version of macOS we're compiling on */
#include <AvailabilityMacros.h>
#ifndef __has_extension /* Older compilers don't support this */
    #define __has_extension(x) 0
    #include <TargetConditionals.h>
    #undef __has_extension
#else
    #include <TargetConditionals.h>
#endif

/* Fix building with older SDKs that don't define these
    See this for more information:
    https://stackoverflow.com/questions/12132933/preprocessor-macro-for-os-x-targets
*/
#ifndef TARGET_OS_MACCATALYST
    #define TARGET_OS_MACCATALYST 0
#endif
#ifndef TARGET_OS_IOS
    #define TARGET_OS_IOS 0
#endif
#ifndef TARGET_OS_IPHONE
    #define TARGET_OS_IPHONE 0
#endif
#ifndef TARGET_OS_TV
    #define TARGET_OS_TV 0
#endif
#ifndef TARGET_OS_SIMULATOR
    #define TARGET_OS_SIMULATOR 0
#endif
#ifndef TARGET_OS_VISION
    #define TARGET_OS_VISION 0
#endif

#if TARGET_OS_TV

/**
 * A preprocessor macro that is only defined if compiling for tvOS.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_PLATFORM_APPLE
 */
#define SDL_PLATFORM_TVOS 1
#endif

#if TARGET_OS_VISION

/**
 * A preprocessor macro that is only defined if compiling for VisionOS.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_PLATFORM_APPLE
 */
#define SDL_PLATFORM_VISIONOS 1
#endif

#if TARGET_OS_IPHONE

/**
 * A preprocessor macro that is only defined if compiling for iOS.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_PLATFORM_APPLE
 */
#define SDL_PLATFORM_IOS 1

#else

/**
 * A preprocessor macro that is only defined if compiling for macOS.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_PLATFORM_APPLE
 */
#define SDL_PLATFORM_MACOS 1

#if MAC_OS_X_VERSION_MIN_REQUIRED < 1070
    #error SDL for macOS only supports deploying on 10.7 and above.
#endif /* MAC_OS_X_VERSION_MIN_REQUIRED < 1070 */
#endif /* TARGET_OS_IPHONE */
#endif /* defined(__APPLE__) */

#ifdef __EMSCRIPTEN__

/**
 * A preprocessor macro that is only defined if compiling for Emscripten.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_EMSCRIPTEN 1
#endif

#ifdef __NetBSD__

/**
 * A preprocessor macro that is only defined if compiling for NetBSD.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_NETBSD 1
#endif

#ifdef __OpenBSD__

/**
 * A preprocessor macro that is only defined if compiling for OpenBSD.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_OPENBSD 1
#endif

#if defined(__OS2__) || defined(__EMX__)

/**
 * A preprocessor macro that is only defined if compiling for OS/2.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_OS2 1
#endif

#if defined(osf) || defined(__osf) || defined(__osf__) || defined(_OSF_SOURCE)

/**
 * A preprocessor macro that is only defined if compiling for Tru64 (OSF/1).
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_OSF 1
#endif

#ifdef __QNXNTO__

/**
 * A preprocessor macro that is only defined if compiling for QNX Neutrino.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_QNXNTO 1
#endif

#if defined(riscos) || defined(__riscos) || defined(__riscos__)

/**
 * A preprocessor macro that is only defined if compiling for RISC OS.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_RISCOS 1
#endif

#if defined(__sun) && defined(__SVR4)

/**
 * A preprocessor macro that is only defined if compiling for SunOS/Solaris.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_SOLARIS 1
#endif

#if defined(__CYGWIN__)

/**
 * A preprocessor macro that is only defined if compiling for Cygwin.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_CYGWIN 1
#endif

#if defined(_WIN32) || defined(SDL_PLATFORM_CYGWIN)

/**
 * A preprocessor macro that is only defined if compiling for Windows.
 *
 * This also covers several other platforms, like Microsoft GDK, Xbox, WinRT,
 * etc. Each will have their own more-specific platform macros, too.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_PLATFORM_WIN32
 * \sa SDL_PLATFORM_XBOXONE
 * \sa SDL_PLATFORM_XBOXSERIES
 * \sa SDL_PLATFORM_WINGDK
 * \sa SDL_PLATFORM_GDK
 */
#define SDL_PLATFORM_WINDOWS 1

/* Try to find out if we're compiling for WinRT, GDK or non-WinRT/GDK */
#if defined(_MSC_VER) && defined(__has_include)
    #if __has_include(<winapifamily.h>)
        #define HAVE_WINAPIFAMILY_H 1
    #else
        #define HAVE_WINAPIFAMILY_H 0
    #endif

    /* If _USING_V110_SDK71_ is defined it means we are using the Windows XP toolset. */
#elif defined(_MSC_VER) && (_MSC_VER >= 1700 && !_USING_V110_SDK71_)    /* _MSC_VER == 1700 for Visual Studio 2012 */
    #define HAVE_WINAPIFAMILY_H 1
#else
    #define HAVE_WINAPIFAMILY_H 0
#endif

#if HAVE_WINAPIFAMILY_H
    #include <winapifamily.h>
    #define WINAPI_FAMILY_WINRT (!WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP) && WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP))
#else
    #define WINAPI_FAMILY_WINRT 0
#endif /* HAVE_WINAPIFAMILY_H */

#ifdef SDL_WIKI_DOCUMENTATION_SECTION

/**
 * A preprocessor macro that defined to 1 if compiling for Windows Phone.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_WINAPI_FAMILY_PHONE (WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP)

#elif defined(HAVE_WINAPIFAMILY_H) && HAVE_WINAPIFAMILY_H
    #define SDL_WINAPI_FAMILY_PHONE (WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP)
#else
    #define SDL_WINAPI_FAMILY_PHONE 0
#endif

#if WINAPI_FAMILY_WINRT
#error Windows RT/UWP is no longer supported in SDL

#elif defined(_GAMING_DESKTOP) /* GDK project configuration always defines _GAMING_XXX */

/**
 * A preprocessor macro that is only defined if compiling for Microsoft GDK
 * for Windows.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_WINGDK 1

#elif defined(_GAMING_XBOX_XBOXONE)

/**
 * A preprocessor macro that is only defined if compiling for Xbox One.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_XBOXONE 1

#elif defined(_GAMING_XBOX_SCARLETT)

/**
 * A preprocessor macro that is only defined if compiling for Xbox Series.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_XBOXSERIES 1

#else

/**
 * A preprocessor macro that is only defined if compiling for desktop Windows.
 *
 * Despite the "32", this also covers 64-bit Windows; as an informal
 * convention, its system layer tends to still be referred to as "the Win32
 * API."
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_WIN32 1

#endif
#endif /* defined(_WIN32) || defined(SDL_PLATFORM_CYGWIN) */


/* This is to support generic "any GDK" separate from a platform-specific GDK */
#if defined(SDL_PLATFORM_WINGDK) || defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)

/**
 * A preprocessor macro that is only defined if compiling for Microsoft GDK on
 * any platform.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_GDK 1
#endif

#if defined(__PSP__) || defined(__psp__)

/**
 * A preprocessor macro that is only defined if compiling for Sony PSP.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_PSP 1
#endif

#if defined(__PS2__) || defined(PS2)

/**
 * A preprocessor macro that is only defined if compiling for Sony PlayStation
 * 2.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_PS2 1
#endif

#if defined(__vita__) || defined(__psp2__)

/**
 * A preprocessor macro that is only defined if compiling for Sony Vita.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_VITA 1
#endif

#ifdef __3DS__

/**
 * A preprocessor macro that is only defined if compiling for Nintendo 3DS.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PLATFORM_3DS 1
#endif

#endif /* SDL_platform_defines_h_ */
