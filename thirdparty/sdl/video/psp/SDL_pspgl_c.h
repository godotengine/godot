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

#ifndef SDL_pspgl_c_h_
#define SDL_pspgl_c_h_

#include <GLES/egl.h>
#include <GLES/gl.h>

#include "SDL_pspvideo.h"

typedef struct SDL_GLDriverData
{
    EGLDisplay display;
    EGLContext context;
    EGLSurface surface;
    uint32_t swapinterval;
} SDL_GLDriverData;

extern SDL_FunctionPointer PSP_GL_GetProcAddress(SDL_VideoDevice *_this, const char *proc);
extern bool PSP_GL_MakeCurrent(SDL_VideoDevice *_this, SDL_Window *window, SDL_GLContext context);
extern void PSP_GL_SwapBuffers(SDL_VideoDevice *_this);

extern bool PSP_GL_SwapWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern SDL_GLContext PSP_GL_CreateContext(SDL_VideoDevice *_this, SDL_Window *window);

extern bool PSP_GL_LoadLibrary(SDL_VideoDevice *_this, const char *path);
extern void PSP_GL_UnloadLibrary(SDL_VideoDevice *_this);
extern bool PSP_GL_SetSwapInterval(SDL_VideoDevice *_this, int interval);
extern bool PSP_GL_GetSwapInterval(SDL_VideoDevice *_this, int *interval);

#endif // SDL_pspgl_c_h_
