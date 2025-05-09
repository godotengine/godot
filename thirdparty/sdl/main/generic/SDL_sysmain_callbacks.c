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
#include "../../video/SDL_sysvideo.h"

#ifndef SDL_PLATFORM_IOS

static int callback_rate_increment = 0;
static bool iterate_after_waitevent = false;

static void SDLCALL MainCallbackRateHintChanged(void *userdata, const char *name, const char *oldValue, const char *newValue)
{
    iterate_after_waitevent = newValue && (SDL_strcmp(newValue, "waitevent") == 0);
    if (iterate_after_waitevent) {
        callback_rate_increment = 0;
    } else {
        const int callback_rate = newValue ? SDL_atoi(newValue) : 0;
        if (callback_rate > 0) {
            callback_rate_increment = ((Uint64) 1000000000) / ((Uint64) callback_rate);
        } else {
            callback_rate_increment = 0;
        }
    }
}

static SDL_AppResult GenericIterateMainCallbacks(void)
{
    if (iterate_after_waitevent) {
        SDL_WaitEvent(NULL);
    }
    return SDL_IterateMainCallbacks(!iterate_after_waitevent);
}

int SDL_EnterAppMainCallbacks(int argc, char* argv[], SDL_AppInit_func appinit, SDL_AppIterate_func appiter, SDL_AppEvent_func appevent, SDL_AppQuit_func appquit)
{
    SDL_AppResult rc = SDL_InitMainCallbacks(argc, argv, appinit, appiter, appevent, appquit);
    if (rc == 0) {
        SDL_AddHintCallback(SDL_HINT_MAIN_CALLBACK_RATE, MainCallbackRateHintChanged, NULL);

        Uint64 next_iteration = callback_rate_increment ? (SDL_GetTicksNS() + callback_rate_increment) : 0;

        while ((rc = GenericIterateMainCallbacks()) == SDL_APP_CONTINUE) {
            // !!! FIXME: this can be made more complicated if we decide to
            // !!! FIXME: optionally hand off callback responsibility to the
            // !!! FIXME: video subsystem (for example, if Wayland has a
            // !!! FIXME: protocol to drive an animation loop, maybe we hand
            // !!! FIXME: off to them here if/when the video subsystem becomes
            // !!! FIXME: initialized).

            // Try to run at whatever rate the hint requested. This makes this
            //  not eat all the CPU in simple things like loopwave. By
            //  default, we run as fast as possible, which means we'll clamp to
            //  vsync in common cases, and won't be restrained to vsync if the
            //  app is doing a benchmark or doesn't want to be, based on how
            // they've set up that window.
            if (callback_rate_increment == 0) {
                next_iteration = 0; // just clear the timer and run at the pace the video subsystem allows.
            } else {
                const Uint64 now = SDL_GetTicksNS();
                if (next_iteration > now) {  // Running faster than the limit, sleep a little.
                    SDL_DelayPrecise(next_iteration - now);
                } else {
                    next_iteration = now;  // if running behind, reset the timer. If right on time, `next_iteration` already equals `now`.
                }
                next_iteration += callback_rate_increment;
            }
        }

        SDL_RemoveHintCallback(SDL_HINT_MAIN_CALLBACK_RATE, MainCallbackRateHintChanged, NULL);
    }
    SDL_QuitMainCallbacks(rc);

    return (rc == SDL_APP_FAILURE) ? 1 : 0;
}

#endif // !SDL_PLATFORM_IOS
