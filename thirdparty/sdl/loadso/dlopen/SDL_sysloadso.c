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

#ifdef SDL_LOADSO_DLOPEN

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// System dependent library loading routines

#include <stdio.h>
#include <dlfcn.h>

#ifdef SDL_VIDEO_DRIVER_UIKIT
#include "../../video/uikit/SDL_uikitvideo.h"
#endif

SDL_SharedObject *SDL_LoadObject(const char *sofile)
{
    void *handle;
    const char *loaderror;

#ifdef SDL_VIDEO_DRIVER_UIKIT
    if (!UIKit_IsSystemVersionAtLeast(8.0)) {
        SDL_SetError("SDL_LoadObject requires iOS 8+");
        return NULL;
    }
#endif

    handle = dlopen(sofile, RTLD_NOW | RTLD_LOCAL);
    loaderror = dlerror();
    if (!handle) {
        SDL_SetError("Failed loading %s: %s", sofile, loaderror);
    }
    return (SDL_SharedObject *) handle;
}

SDL_FunctionPointer SDL_LoadFunction(SDL_SharedObject *handle, const char *name)
{
    void *symbol = dlsym(handle, name);
    if (!symbol) {
        // prepend an underscore for platforms that need that.
        bool isstack;
        size_t len = SDL_strlen(name) + 1;
        char *_name = SDL_small_alloc(char, len + 1, &isstack);
        _name[0] = '_';
        SDL_memcpy(&_name[1], name, len);
        symbol = dlsym(handle, _name);
        SDL_small_free(_name, isstack);
        if (!symbol) {
            SDL_SetError("Failed loading %s: %s", name,
                         (const char *)dlerror());
        }
    }
    return symbol;
}

void SDL_UnloadObject(SDL_SharedObject *handle)
{
    if (handle) {
        dlclose(handle);
    }
}

#endif // SDL_LOADSO_DLOPEN
