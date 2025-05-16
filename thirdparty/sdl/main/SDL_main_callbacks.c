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
#include "SDL_main_callbacks.h"

static SDL_AppEvent_func SDL_main_event_callback;
static SDL_AppIterate_func SDL_main_iteration_callback;
static SDL_AppQuit_func SDL_main_quit_callback;
static SDL_AtomicInt apprc;  // use an atomic, since events might land from any thread and we don't want to wrap this all in a mutex. A CAS makes sure we only move from zero once.
static void *SDL_main_appstate = NULL;

// Return true if this event needs to be processed before returning from the event watcher
static bool ShouldDispatchImmediately(SDL_Event *event)
{
    switch (event->type) {
    case SDL_EVENT_TERMINATING:
    case SDL_EVENT_LOW_MEMORY:
    case SDL_EVENT_WILL_ENTER_BACKGROUND:
    case SDL_EVENT_DID_ENTER_BACKGROUND:
    case SDL_EVENT_WILL_ENTER_FOREGROUND:
    case SDL_EVENT_DID_ENTER_FOREGROUND:
        return true;
    default:
        return false;
    }
}

static void SDL_DispatchMainCallbackEvent(SDL_Event *event)
{
    if (SDL_GetAtomicInt(&apprc) == SDL_APP_CONTINUE) { // if already quitting, don't send the event to the app.
        SDL_CompareAndSwapAtomicInt(&apprc, SDL_APP_CONTINUE, SDL_main_event_callback(SDL_main_appstate, event));
    }
}

static void SDL_DispatchMainCallbackEvents(void)
{
    SDL_Event events[16];

    for (;;) {
        int count = SDL_PeepEvents(events, SDL_arraysize(events), SDL_GETEVENT, SDL_EVENT_FIRST, SDL_EVENT_LAST);
        if (count <= 0) {
            break;
        }
        for (int i = 0; i < count; ++i) {
            SDL_Event *event = &events[i];
            if (!ShouldDispatchImmediately(event)) {
                SDL_DispatchMainCallbackEvent(event);
            }
        }
    }
}

static bool SDLCALL SDL_MainCallbackEventWatcher(void *userdata, SDL_Event *event)
{
    if (ShouldDispatchImmediately(event)) {
        // Make sure any currently queued events are processed then dispatch this before continuing
        SDL_DispatchMainCallbackEvents();
        SDL_DispatchMainCallbackEvent(event);

        // Make sure that we quit if we get a terminating event
        if (event->type == SDL_EVENT_TERMINATING) {
            SDL_CompareAndSwapAtomicInt(&apprc, SDL_APP_CONTINUE, SDL_APP_SUCCESS);
        }
    } else {
        // We'll process this event later from the main event queue
    }
    return true;
}

bool SDL_HasMainCallbacks(void)
{
    if (SDL_main_iteration_callback) {
        return true;
    }
    return false;
}

SDL_AppResult SDL_InitMainCallbacks(int argc, char* argv[], SDL_AppInit_func appinit, SDL_AppIterate_func appiter, SDL_AppEvent_func appevent, SDL_AppQuit_func appquit)
{
    SDL_main_iteration_callback = appiter;
    SDL_main_event_callback = appevent;
    SDL_main_quit_callback = appquit;
    SDL_SetAtomicInt(&apprc, SDL_APP_CONTINUE);

    const SDL_AppResult rc = appinit(&SDL_main_appstate, argc, argv);
    if (SDL_CompareAndSwapAtomicInt(&apprc, SDL_APP_CONTINUE, rc) && (rc == SDL_APP_CONTINUE)) { // bounce if SDL_AppInit already said abort, otherwise...
        // make sure we definitely have events initialized, even if the app didn't do it.
        if (!SDL_InitSubSystem(SDL_INIT_EVENTS)) {
            SDL_SetAtomicInt(&apprc, SDL_APP_FAILURE);
            return SDL_APP_FAILURE;
        }

        if (!SDL_AddEventWatch(SDL_MainCallbackEventWatcher, NULL)) {
            SDL_SetAtomicInt(&apprc, SDL_APP_FAILURE);
            return SDL_APP_FAILURE;
        }
    }

    return (SDL_AppResult)SDL_GetAtomicInt(&apprc);
}

SDL_AppResult SDL_IterateMainCallbacks(bool pump_events)
{
    if (pump_events) {
        SDL_PumpEvents();
    }
    SDL_DispatchMainCallbackEvents();

    SDL_AppResult rc = (SDL_AppResult)SDL_GetAtomicInt(&apprc);
    if (rc == SDL_APP_CONTINUE) {
        rc = SDL_main_iteration_callback(SDL_main_appstate);
        if (!SDL_CompareAndSwapAtomicInt(&apprc, SDL_APP_CONTINUE, rc)) {
            rc = (SDL_AppResult)SDL_GetAtomicInt(&apprc); // something else already set a quit result, keep that.
        }
    }
    return rc;
}

void SDL_QuitMainCallbacks(SDL_AppResult result)
{
    SDL_RemoveEventWatch(SDL_MainCallbackEventWatcher, NULL);
    SDL_main_quit_callback(SDL_main_appstate, result);
    SDL_main_appstate = NULL;  // just in case.

    // for symmetry, you should explicitly Quit what you Init, but we might come through here uninitialized and SDL_Quit() will clear everything anyhow.
    //SDL_QuitSubSystem(SDL_INIT_EVENTS);

    SDL_Quit();
}

