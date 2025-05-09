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

#if defined(SDL_VIDEO_DRIVER_X11) && defined(SDL_VIDEO_OPENGL_EGL)

#include "SDL_x11video.h"
#include "SDL_x11opengles.h"
#include "SDL_x11opengl.h"
#include "SDL_x11xsync.h"

// EGL implementation of SDL OpenGL support

bool X11_GLES_LoadLibrary(SDL_VideoDevice *_this, const char *path)
{
    SDL_VideoData *data = _this->internal;

    // If the profile requested is not GL ES, switch over to X11_GL functions
    if ((_this->gl_config.profile_mask != SDL_GL_CONTEXT_PROFILE_ES) &&
        !SDL_GetHintBoolean(SDL_HINT_VIDEO_FORCE_EGL, false)) {
#ifdef SDL_VIDEO_OPENGL_GLX
        X11_GLES_UnloadLibrary(_this);
        _this->GL_LoadLibrary = X11_GL_LoadLibrary;
        _this->GL_GetProcAddress = X11_GL_GetProcAddress;
        _this->GL_UnloadLibrary = X11_GL_UnloadLibrary;
        _this->GL_CreateContext = X11_GL_CreateContext;
        _this->GL_MakeCurrent = X11_GL_MakeCurrent;
        _this->GL_SetSwapInterval = X11_GL_SetSwapInterval;
        _this->GL_GetSwapInterval = X11_GL_GetSwapInterval;
        _this->GL_SwapWindow = X11_GL_SwapWindow;
        _this->GL_DestroyContext = X11_GL_DestroyContext;
        return X11_GL_LoadLibrary(_this, path);
#else
        return SDL_SetError("SDL not configured with OpenGL/GLX support");
#endif
    }

    return SDL_EGL_LoadLibrary(_this, path, (NativeDisplayType)data->display, _this->gl_config.egl_platform);
}

XVisualInfo *X11_GLES_GetVisual(SDL_VideoDevice *_this, Display *display, int screen, bool transparent)
{

    XVisualInfo *egl_visualinfo = NULL;
    EGLint visual_id = 0;
    XVisualInfo vi_in;
    int out_count = 0;

    if (!_this->egl_data) {
        // The EGL library wasn't loaded, SDL_GetError() should have info
        return NULL;
    }

    if (_this->egl_data->eglGetConfigAttrib(_this->egl_data->egl_display,
                                            _this->egl_data->egl_config,
                                            EGL_NATIVE_VISUAL_ID,
                                            &visual_id) == EGL_FALSE) {
        visual_id = 0;
    }
    if (visual_id != 0) {
        vi_in.screen = screen;
        vi_in.visualid = visual_id;
        egl_visualinfo = X11_XGetVisualInfo(display, VisualScreenMask | VisualIDMask, &vi_in, &out_count);
        if (transparent && egl_visualinfo) {
            Uint32 format = X11_GetPixelFormatFromVisualInfo(display, egl_visualinfo);
            if (!SDL_ISPIXELFORMAT_ALPHA(format)) {
                // not transparent!
                X11_XFree(egl_visualinfo);
                egl_visualinfo = NULL;
            }
        }
    }
    
    if(!egl_visualinfo) {
        // Use the default visual when all else fails
        vi_in.screen = screen;
        egl_visualinfo = X11_XGetVisualInfo(display,
                                            VisualScreenMask,
                                            &vi_in, &out_count);

        // Return the first transparent Visual
        if (transparent) {
            int i;
            for (i = 0; i < out_count; i++) {
                XVisualInfo *v = &egl_visualinfo[i];
                Uint32 format = X11_GetPixelFormatFromVisualInfo(display, v);
                if (SDL_ISPIXELFORMAT_ALPHA(format)) { // found!
                    // re-request it to have a copy that can be X11_XFree'ed later
                    vi_in.screen = screen;
                    vi_in.visualid = v->visualid;
                    X11_XFree(egl_visualinfo);
                    egl_visualinfo = X11_XGetVisualInfo(display, VisualScreenMask | VisualIDMask, &vi_in, &out_count);
                    return egl_visualinfo;
                }
            }
        }
    }
    return egl_visualinfo;
}

SDL_GLContext X11_GLES_CreateContext(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_GLContext context;
    SDL_WindowData *data = window->internal;
    Display *display = data->videodata->display;

    X11_XSync(display, False);
    context = SDL_EGL_CreateContext(_this, data->egl_surface);
    X11_XSync(display, False);

    return context;
}

SDL_EGLSurface X11_GLES_GetEGLSurface(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data = window->internal;
    return data->egl_surface;
}

bool X11_GLES_SwapWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    const bool ret = SDL_EGL_SwapBuffers(_this, window->internal->egl_surface);                       \

#ifdef SDL_VIDEO_DRIVER_X11_XSYNC
    X11_HandlePresent(window);
#endif /* SDL_VIDEO_DRIVER_X11_XSYNC */

    return ret;
}

SDL_EGL_MakeCurrent_impl(X11)

#endif // SDL_VIDEO_DRIVER_X11 && SDL_VIDEO_OPENGL_EGL

