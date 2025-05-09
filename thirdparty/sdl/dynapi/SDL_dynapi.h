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

#ifndef SDL_dynapi_h_
#define SDL_dynapi_h_

/* IMPORTANT:
   This is the master switch to disabling the dynamic API. We made it so you
   have to hand-edit an internal source file in SDL to turn it off; you
   can do it if you want it badly enough, but hopefully you won't want to.
   You should understand the ramifications of turning this off: it makes it
   hard to update your SDL in the field, and impossible if you've statically
   linked SDL into your app. Understand that platforms change, and if we can't
   drop in an updated SDL, your application can definitely break some time
   in the future, even if it's fine today.
   To be sure, as new system-level video and audio APIs are introduced, an
   updated SDL can transparently take advantage of them, but your program will
   not without this feature. Think hard before turning it off.
*/
#ifdef SDL_DYNAMIC_API // Tried to force it on the command line?
#error Nope, you have to edit this file to force this off.
#endif

#ifdef SDL_PLATFORM_APPLE
#include "TargetConditionals.h"
#endif

#if defined(SDL_PLATFORM_PRIVATE) // probably not useful on private platforms.
#define SDL_DYNAMIC_API 0
#elif defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE // probably not useful on iOS.
#define SDL_DYNAMIC_API 0
#elif defined(SDL_PLATFORM_ANDROID) // probably not useful on Android.
#define SDL_DYNAMIC_API 0
#elif defined(SDL_PLATFORM_EMSCRIPTEN) // probably not useful on Emscripten.
#define SDL_DYNAMIC_API 0
#elif defined(SDL_PLATFORM_PS2) && SDL_PLATFORM_PS2
#define SDL_DYNAMIC_API 0
#elif defined(SDL_PLATFORM_PSP) && SDL_PLATFORM_PSP
#define SDL_DYNAMIC_API 0
#elif defined(SDL_PLATFORM_RISCOS) // probably not useful on RISC OS, since dlopen() can't be used when using static linking.
#define SDL_DYNAMIC_API 0
#elif defined(__clang_analyzer__) || defined(__INTELLISENSE__) || defined(SDL_THREAD_SAFETY_ANALYSIS)
#define SDL_DYNAMIC_API 0 // Turn off for static analysis, so reports are more clear.
#elif defined(SDL_PLATFORM_VITA)
#define SDL_DYNAMIC_API 0 // vitasdk doesn't support dynamic linking
#elif defined(SDL_PLATFORM_3DS)
#define SDL_DYNAMIC_API 0 // devkitARM doesn't support dynamic linking
#elif defined(DYNAPI_NEEDS_DLOPEN) && !defined(HAVE_DLOPEN)
#define SDL_DYNAMIC_API 0 // we need dlopen(), but don't have it....
#endif

// everyone else. This is where we turn on the API if nothing forced it off.
#ifndef SDL_DYNAMIC_API
#define SDL_DYNAMIC_API 1
#endif

#endif
