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

#ifdef SDL_THREAD_WINDOWS

#include "../../core/windows/SDL_windows.h"

#include "../SDL_thread_c.h"

static DWORD thread_local_storage = TLS_OUT_OF_INDEXES;
static bool generic_local_storage = false;

void SDL_SYS_InitTLSData(void)
{
    if (thread_local_storage == TLS_OUT_OF_INDEXES && !generic_local_storage) {
        thread_local_storage = TlsAlloc();
        if (thread_local_storage == TLS_OUT_OF_INDEXES) {
            SDL_Generic_InitTLSData();
            generic_local_storage = true;
        }
    }
}

SDL_TLSData *SDL_SYS_GetTLSData(void)
{
    if (generic_local_storage) {
        return SDL_Generic_GetTLSData();
    }

    if (thread_local_storage != TLS_OUT_OF_INDEXES) {
        return (SDL_TLSData *)TlsGetValue(thread_local_storage);
    }
    return NULL;
}

bool SDL_SYS_SetTLSData(SDL_TLSData *data)
{
    if (generic_local_storage) {
        return SDL_Generic_SetTLSData(data);
    }

    if (!TlsSetValue(thread_local_storage, data)) {
        return WIN_SetError("TlsSetValue()");
    }
    return true;
}

void SDL_SYS_QuitTLSData(void)
{
    if (generic_local_storage) {
        SDL_Generic_QuitTLSData();
        generic_local_storage = false;
    } else {
        if (thread_local_storage != TLS_OUT_OF_INDEXES) {
            TlsFree(thread_local_storage);
            thread_local_storage = TLS_OUT_OF_INDEXES;
        }
    }
}

#endif // SDL_THREAD_WINDOWS
