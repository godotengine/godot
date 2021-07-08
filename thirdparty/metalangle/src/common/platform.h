//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// platform.h: Operating system specific includes and defines.

#ifndef COMMON_PLATFORM_H_
#define COMMON_PLATFORM_H_

#if defined(_WIN32)
#    define ANGLE_PLATFORM_WINDOWS 1
#elif defined(__Fuchsia__)
#    define ANGLE_PLATFORM_FUCHSIA 1
#    define ANGLE_PLATFORM_POSIX 1
#elif defined(__APPLE__)
#    define ANGLE_PLATFORM_APPLE 1
#    define ANGLE_PLATFORM_POSIX 1
#elif defined(ANDROID)
#    define ANGLE_PLATFORM_ANDROID 1
#    define ANGLE_PLATFORM_POSIX 1
#elif defined(__linux__) || defined(EMSCRIPTEN)
#    define ANGLE_PLATFORM_LINUX 1
#    define ANGLE_PLATFORM_POSIX 1
#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) ||              \
    defined(__DragonFly__) || defined(__sun) || defined(__GLIBC__) || defined(__GNU__) || \
    defined(__QNX__) || defined(__Fuchsia__) || defined(__HAIKU__)
#    define ANGLE_PLATFORM_POSIX 1
#else
#    error Unsupported platform.
#endif

#ifdef ANGLE_PLATFORM_WINDOWS
#    ifndef STRICT
#        define STRICT 1
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN 1
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX 1
#    endif

#    include <intrin.h>
#    include <windows.h>

#    if defined(WINAPI_FAMILY) && (WINAPI_FAMILY != WINAPI_FAMILY_DESKTOP_APP)
#        define ANGLE_ENABLE_WINDOWS_UWP 1
#    endif

#    if defined(ANGLE_ENABLE_D3D9)
#        include <d3d9.h>
#        include <d3dcompiler.h>
#    endif

// Include D3D11 headers when OpenGL is enabled on Windows for interop extensions.
#    if defined(ANGLE_ENABLE_D3D11) || defined(ANGLE_ENABLE_OPENGL)
#        include <d3d10_1.h>
#        include <d3d11.h>
#        include <d3d11_3.h>
#        include <d3dcompiler.h>
#        include <dxgi.h>
#        include <dxgi1_2.h>
#    endif

#    if defined(ANGLE_ENABLE_D3D9) || defined(ANGLE_ENABLE_D3D11)
#        include <wrl.h>
#    endif

#    if defined(ANGLE_ENABLE_WINDOWS_UWP)
#        include <dxgi1_3.h>
#        if defined(_DEBUG)
#            include <DXProgrammableCapture.h>
#            include <dxgidebug.h>
#        endif
#    endif

#    undef near
#    undef far
#endif

#if defined(_MSC_VER) && !defined(_M_ARM) && !defined(_M_ARM64)
#    include <intrin.h>
#    define ANGLE_USE_SSE
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#    include <x86intrin.h>
#    define ANGLE_USE_SSE
#endif

// Mips and arm devices need to include stddef for size_t.
#if defined(__mips__) || defined(__arm__) || defined(__aarch64__)
#    include <stddef.h>
#endif

// The MemoryBarrier function name collides with a macro under Windows
// We will undef the macro so that the function name does not get replaced
#undef MemoryBarrier

// Macro for hinting that an expression is likely to be true/false.
#if !defined(ANGLE_LIKELY) || !defined(ANGLE_UNLIKELY)
#    if defined(__GNUC__) || defined(__clang__)
#        define ANGLE_LIKELY(x) __builtin_expect(!!(x), 1)
#        define ANGLE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#    else
#        define ANGLE_LIKELY(x) (x)
#        define ANGLE_UNLIKELY(x) (x)
#    endif  // defined(__GNUC__) || defined(__clang__)
#endif      // !defined(ANGLE_LIKELY) || !defined(ANGLE_UNLIKELY)

#if defined(ANGLE_PLATFORM_APPLE)
#    include <TargetConditionals.h>
#    if TARGET_OS_OSX
#        define ANGLE_PLATFORM_MACOS 1
#    elif TARGET_OS_IPHONE
#        define ANGLE_PLATFORM_IOS 1
#        if defined(ANGLE_ENABLE_OPENGL)
#            define GLES_SILENCE_DEPRECATION
#        endif
#        if TARGET_OS_SIMULATOR
#            define ANGLE_PLATFORM_IOS_SIMULATOR 1
#        endif
#        if TARGET_OS_MACCATALYST
#            define ANGLE_PLATFORM_MACCATALYST
#        endif
#    endif
#endif

#endif  // COMMON_PLATFORM_H_
