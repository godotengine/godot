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

#ifdef SDL_LOADSO_WINDOWS

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// System dependent library loading routines

#include "../../core/windows/SDL_windows.h"

SDL_SharedObject *SDL_LoadObject(const char *sofile)
{
    if (!sofile) {
        SDL_InvalidParamError("sofile");
        return NULL;
    }

    LPWSTR wstr = WIN_UTF8ToStringW(sofile);
    HMODULE handle = LoadLibraryW(wstr);
    SDL_free(wstr);

    // Generate an error message if all loads failed
    if (!handle) {
        char errbuf[512];
        SDL_snprintf(errbuf, sizeof (errbuf), "Failed loading %s", sofile);
        WIN_SetError(errbuf);
    }
    return (SDL_SharedObject *) handle;
}

SDL_FunctionPointer SDL_LoadFunction(SDL_SharedObject *handle, const char *name)
{
    SDL_FunctionPointer symbol = (SDL_FunctionPointer)GetProcAddress((HMODULE)handle, name);
    if (!symbol) {
        char errbuf[512];
        SDL_snprintf(errbuf, sizeof (errbuf), "Failed loading %s", name);
        WIN_SetError(errbuf);
    }
    return symbol;
}

void SDL_UnloadObject(SDL_SharedObject *handle)
{
    if (handle) {
        FreeLibrary((HMODULE)handle);
    }
}

#endif // SDL_LOADSO_WINDOWS
