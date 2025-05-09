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
#include "../SDL_main_callbacks.h"

#include <emscripten.h>

static void EmscriptenInternalMainloop(void)
{
    const SDL_AppResult rc = SDL_IterateMainCallbacks(true);
    if (rc != SDL_APP_CONTINUE) {
        SDL_QuitMainCallbacks(rc);
        emscripten_cancel_main_loop();  // kill" the mainloop, so it stops calling back into it.
        exit((rc == SDL_APP_FAILURE) ? 1 : 0);  // hopefully this takes down everything else, too.
    }
}

int SDL_EnterAppMainCallbacks(int argc, char* argv[], SDL_AppInit_func appinit, SDL_AppIterate_func appiter, SDL_AppEvent_func appevent, SDL_AppQuit_func appquit)
{
    const SDL_AppResult rc = SDL_InitMainCallbacks(argc, argv, appinit, appiter, appevent, appquit);
    if (rc == SDL_APP_CONTINUE) {
        emscripten_set_main_loop(EmscriptenInternalMainloop, 0, 0);  // run at refresh rate, don't throw an exception since we do an orderly return.
    } else {
        SDL_QuitMainCallbacks(rc);
    }
    return (rc == SDL_APP_FAILURE) ? 1 : 0;
}

