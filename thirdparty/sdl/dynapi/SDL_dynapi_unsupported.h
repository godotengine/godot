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

#ifndef SDL_dynapi_unsupported_h_
#define SDL_dynapi_unsupported_h_


#if !defined(SDL_PLATFORM_WINDOWS)
typedef struct ID3D12Device ID3D12Device;
typedef void *SDL_WindowsMessageHook;
#endif

#if !(defined(SDL_PLATFORM_WIN32) || defined(SDL_PLATFORM_WINGDK))
typedef struct ID3D11Device ID3D11Device;
typedef struct IDirect3DDevice9 IDirect3DDevice9;
#endif

#ifndef SDL_PLATFORM_GDK
typedef struct XTaskQueueHandle XTaskQueueHandle;
#endif

#ifndef SDL_PLATFORM_GDK
typedef struct XUserHandle XUserHandle;
#endif

#ifndef SDL_PLATFORM_ANDROID
typedef void *SDL_RequestAndroidPermissionCallback;
#endif

#ifndef SDL_PLATFORM_IOS
typedef void *SDL_iOSAnimationCallback;
#endif

#endif
