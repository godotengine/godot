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

#if defined(SDL_VIDEO_DRIVER_VITA) && defined(SDL_VIDEO_VITA_PVR) && defined(SDL_VIDEO_VITA_PVR_OGL)
#include <stdlib.h>
#include <string.h>
#include <psp2/kernel/modulemgr.h>
#include <gpu_es4/psp2_pvr_hint.h>
#include <gl4esinit.h>

#include "SDL_vitavideo.h"
#include "../SDL_egl_c.h"

#define MAX_PATH 256 // vita limits are somehow wrong

// Defaults
static int FB_WIDTH = 960;
static int FB_HEIGHT = 544;

static void getFBSize(int *width, int *height)
{
    *width = FB_WIDTH;
    *height = FB_HEIGHT;
}

bool VITA_GL_LoadLibrary(SDL_VideoDevice *_this, const char *path)
{
    PVRSRV_PSP2_APPHINT hint;
    char *default_path = "app0:module";
    char target_path[MAX_PATH];

    if (SDL_GetHintBoolean(SDL_HINT_VITA_PVR_INIT, true)) {
        const char *override = SDL_GetHint(SDL_HINT_VITA_MODULE_PATH);

        if (override && *override) {
            default_path = override;
        }

        sceKernelLoadStartModule("vs0:sys/external/libfios2.suprx", 0, NULL, 0, NULL, NULL);
        sceKernelLoadStartModule("vs0:sys/external/libc.suprx", 0, NULL, 0, NULL, NULL);

        SDL_snprintf(target_path, MAX_PATH, "%s/%s", default_path, "libGL.suprx");
        sceKernelLoadStartModule(target_path, 0, NULL, 0, NULL, NULL);

        SDL_snprintf(target_path, MAX_PATH, "%s/%s", default_path, "libgpu_es4_ext.suprx");
        sceKernelLoadStartModule(target_path, 0, NULL, 0, NULL, NULL);

        SDL_snprintf(target_path, MAX_PATH, "%s/%s", default_path, "libIMGEGL.suprx");
        sceKernelLoadStartModule(target_path, 0, NULL, 0, NULL, NULL);

        PVRSRVInitializeAppHint(&hint);

        SDL_snprintf(hint.szGLES1, MAX_PATH, "%s/%s", default_path, "libGLESv1_CM.suprx");
        SDL_snprintf(hint.szGLES2, MAX_PATH, "%s/%s", default_path, "libGLESv2.suprx");
        SDL_snprintf(hint.szWindowSystem, MAX_PATH, "%s/%s", default_path, "libpvrPSP2_WSEGL.suprx");

        PVRSRVCreateVirtualAppHint(&hint);
    }

    return SDL_EGL_LoadLibrary(_this, path, (NativeDisplayType)0, 0);
}

SDL_GLContext VITA_GL_CreateContext(SDL_VideoDevice *_this, SDL_Window *window)
{
    char gl_version[3];
    SDL_GLContext context = NULL;
    int temp_major = _this->gl_config.major_version;
    int temp_minor = _this->gl_config.minor_version;
    int temp_profile = _this->gl_config.profile_mask;

    // Set version to 2.0 and PROFILE to ES
    _this->gl_config.major_version = 2;
    _this->gl_config.minor_version = 0;
    _this->gl_config.profile_mask = SDL_GL_CONTEXT_PROFILE_ES;

    context = SDL_EGL_CreateContext(_this, window->internal->egl_surface);

    if (context != NULL) {
        FB_WIDTH = window->w;
        FB_HEIGHT = window->h;
        set_getprocaddress((void *(*)(const char *))eglGetProcAddress);
        set_getmainfbsize(getFBSize);
        SDL_snprintf(gl_version, 3, "%d%d", temp_major, temp_minor);
        gl4es_setenv("LIBGL_NOTEXRECT", "1", 1); // Currently broken in driver
        gl4es_setenv("LIBGL_GL", gl_version, 1);
        initialize_gl4es();
    }

    // Restore gl_config
    _this->gl_config.major_version = temp_major;
    _this->gl_config.minor_version = temp_minor;
    _this->gl_config.profile_mask = temp_profile;

    return context;
}

SDL_FunctionPointer VITA_GL_GetProcAddress(SDL_VideoDevice *_this, const char *proc)
{
    return gl4es_GetProcAddress(proc);
}

#endif // SDL_VIDEO_DRIVER_VITA && SDL_VIDEO_VITA_PVR
