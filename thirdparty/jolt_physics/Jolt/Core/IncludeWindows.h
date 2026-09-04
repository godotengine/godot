// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2025 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#ifdef JPH_PLATFORM_WINDOWS

JPH_SUPPRESS_WARNING_PUSH
JPH_MSVC_SUPPRESS_WARNING(5039) // winbase.h(13179): warning C5039: 'TpSetCallbackCleanupGroup': pointer or reference to potentially throwing function passed to 'extern "C"' function under -EHc. Undefined behavior may occur if this function throws an exception.
JPH_MSVC2026_PLUS_SUPPRESS_WARNING(4865) // wingdi.h(2806,1): '<unnamed-enum-DISPLAYCONFIG_OUTPUT_TECHNOLOGY_OTHER>': the underlying type will change from 'int' to '__int64' when '/Zc:enumTypes' is specified on the command line
JPH_CLANG_SUPPRESS_WARNING("-Wreserved-macro-identifier") // Complains about _WIN32_WINNT being reserved

#ifndef WINVER
	#define WINVER 0x0A00 // Targeting Windows 10 and above
#endif

#ifndef _WIN32_WINNT
	#define _WIN32_WINNT 0x0A00
#endif

#ifndef WIN32_LEAN_AND_MEAN
	#define WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
	#define NOMINMAX
#endif

#ifndef JPH_COMPILER_MINGW
	#include <Windows.h>
#else
	#include <windows.h>
#endif

JPH_SUPPRESS_WARNING_POP

#endif
