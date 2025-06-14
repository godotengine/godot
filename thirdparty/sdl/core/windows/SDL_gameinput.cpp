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

#ifdef HAVE_GAMEINPUT_H

#include "SDL_windows.h"
#include "SDL_gameinput.h"

static SDL_SharedObject *g_hGameInputDLL;
static IGameInput *g_pGameInput;
static int g_nGameInputRefCount;


bool SDL_InitGameInput(IGameInput **ppGameInput)
{
    if (g_nGameInputRefCount == 0) {
        g_hGameInputDLL = SDL_LoadObject("gameinput.dll");
        if (!g_hGameInputDLL) {
            return false;
        }

        typedef HRESULT (WINAPI *GameInputCreate_t)(IGameInput **gameInput);
        GameInputCreate_t GameInputCreateFunc = (GameInputCreate_t)SDL_LoadFunction(g_hGameInputDLL, "GameInputCreate");
        if (!GameInputCreateFunc) {
            SDL_UnloadObject(g_hGameInputDLL);
            return false;
        }

        IGameInput *pGameInput = NULL;
        HRESULT hr = GameInputCreateFunc(&pGameInput);
        if (FAILED(hr)) {
            SDL_UnloadObject(g_hGameInputDLL);
            return WIN_SetErrorFromHRESULT("GameInputCreate failed", hr);
        }

#ifdef SDL_PLATFORM_WIN32
#if GAMEINPUT_API_VERSION >= 1
        hr = pGameInput->QueryInterface(IID_IGameInput, (void **)&g_pGameInput);
#else
        // We require GameInput v1.1 or newer
        hr = E_NOINTERFACE;
#endif
        pGameInput->Release();
        if (FAILED(hr)) {
            SDL_UnloadObject(g_hGameInputDLL);
            return WIN_SetErrorFromHRESULT("GameInput QueryInterface failed", hr);
        }
#else
        // Assume that the version we get is compatible with the current SDK
        g_pGameInput = pGameInput;
#endif
    }
    ++g_nGameInputRefCount;

    if (ppGameInput) {
        *ppGameInput = g_pGameInput;
    }
    return true;
}

void SDL_QuitGameInput(void)
{
    SDL_assert(g_nGameInputRefCount > 0);

    --g_nGameInputRefCount;
    if (g_nGameInputRefCount == 0) {
        if (g_pGameInput) {
            g_pGameInput->Release();
            g_pGameInput = NULL;
        }
        if (g_hGameInputDLL) {
            SDL_UnloadObject(g_hGameInputDLL);
            g_hGameInputDLL = NULL;
        }
    }
}

#endif // HAVE_GAMEINPUT_H
