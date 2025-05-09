/*
  Simple DirectMedia Layer
  Copyright (C) 2017 BlackBerry Limited

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

#ifndef __SDL_QNX_H__
#define __SDL_QNX_H__

#include "../SDL_sysvideo.h"
#include <screen/screen.h>
#include <EGL/egl.h>

typedef struct
{
    screen_window_t window;
    EGLSurface      surface;
    EGLConfig       conf;
} window_impl_t;

extern void handleKeyboardEvent(screen_event_t event);

extern bool glGetConfig(EGLConfig *pconf, int *pformat);
extern bool glLoadLibrary(SDL_VideoDevice *_this, const char *name);
extern SDL_FunctionPointer glGetProcAddress(SDL_VideoDevice *_this, const char *proc);
extern SDL_GLContext glCreateContext(SDL_VideoDevice *_this, SDL_Window *window);
extern bool glSetSwapInterval(SDL_VideoDevice *_this, int interval);
extern bool glSwapWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern bool glMakeCurrent(SDL_VideoDevice *_this, SDL_Window * window, SDL_GLContext context);
extern void glDeleteContext(SDL_VideoDevice *_this, SDL_GLContext context);
extern void glUnloadLibrary(SDL_VideoDevice *_this);

#endif
