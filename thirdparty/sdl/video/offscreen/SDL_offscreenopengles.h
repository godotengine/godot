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

#ifndef SDL_offscreenopengles_h
#define SDL_offscreenopengles_h

#if defined(SDL_VIDEO_DRIVER_OFFSCREEN) && defined(SDL_VIDEO_OPENGL_EGL)

#include "../SDL_sysvideo.h"
#include "../SDL_egl_c.h"

#define OFFSCREEN_GLES_GetProcAddress  SDL_EGL_GetProcAddressInternal
#define OFFSCREEN_GLES_UnloadLibrary   SDL_EGL_UnloadLibrary
#define OFFSCREEN_GLES_GetSwapInterval SDL_EGL_GetSwapInterval
#define OFFSCREEN_GLES_SetSwapInterval SDL_EGL_SetSwapInterval
#define OFFSCREEN_GLES_DestroyContext   SDL_EGL_DestroyContext

extern bool OFFSCREEN_GLES_LoadLibrary(SDL_VideoDevice *_this, const char *path);
extern SDL_GLContext OFFSCREEN_GLES_CreateContext(SDL_VideoDevice *_this, SDL_Window *window);
extern bool OFFSCREEN_GLES_MakeCurrent(SDL_VideoDevice *_this, SDL_Window *window, SDL_GLContext context);
extern bool OFFSCREEN_GLES_SwapWindow(SDL_VideoDevice *_this, SDL_Window *window);

#endif // SDL_VIDEO_DRIVER_OFFSCREEN && SDL_VIDEO_OPENGL_EGL

#endif // SDL_offscreenopengles_h
