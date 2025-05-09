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

// Drag and drop event handling code for SDL

#include "SDL_events_c.h"
#include "SDL_dropevents_c.h"

#include "../video/SDL_sysvideo.h" // for SDL_Window internals.

static bool SDL_SendDrop(SDL_Window *window, const SDL_EventType evtype, const char *source, const char *data, float x, float y)
{
    static bool app_is_dropping = false;
    static float last_drop_x = 0;
    static float last_drop_y = 0;
    bool posted = false;

    // Post the event, if desired
    if (SDL_EventEnabled(evtype)) {
        const bool need_begin = window ? !window->is_dropping : !app_is_dropping;
        SDL_Event event;

        if (need_begin) {
            SDL_zero(event);
            event.type = SDL_EVENT_DROP_BEGIN;
            event.common.timestamp = 0;
            event.drop.windowID = window ? window->id : 0;
            posted = SDL_PushEvent(&event);
            if (!posted) {
                return false;
            }
            if (window) {
                window->is_dropping = true;
            } else {
                app_is_dropping = true;
            }
        }

        SDL_zero(event);
        event.type = evtype;
        event.common.timestamp = 0;
        if (source) {
            event.drop.source = SDL_CreateTemporaryString(source);
            if (!event.drop.source) {
                return false;
            }
        }
        if (data) {
            event.drop.data = SDL_CreateTemporaryString(data);
            if (!event.drop.data) {
                return false;
            }
        }
        event.drop.windowID = window ? window->id : 0;

        if (evtype == SDL_EVENT_DROP_POSITION) {
            last_drop_x = x;
            last_drop_y = y;
        }
        event.drop.x = last_drop_x;
        event.drop.y = last_drop_y;
        posted = SDL_PushEvent(&event);

        if (posted && (evtype == SDL_EVENT_DROP_COMPLETE)) {
            if (window) {
                window->is_dropping = false;
            } else {
                app_is_dropping = false;
            }

            last_drop_x = 0;
            last_drop_y = 0;
        }
    }
    return posted;
}

bool SDL_SendDropFile(SDL_Window *window, const char *source, const char *file)
{
    return SDL_SendDrop(window, SDL_EVENT_DROP_FILE, source, file, 0, 0);
}

bool SDL_SendDropPosition(SDL_Window *window, float x, float y)
{
    return SDL_SendDrop(window, SDL_EVENT_DROP_POSITION, NULL, NULL, x, y);
}

bool SDL_SendDropText(SDL_Window *window, const char *text)
{
    return SDL_SendDrop(window, SDL_EVENT_DROP_TEXT, NULL, text, 0, 0);
}

bool SDL_SendDropComplete(SDL_Window *window)
{
    return SDL_SendDrop(window, SDL_EVENT_DROP_COMPLETE, NULL, NULL, 0, 0);
}
