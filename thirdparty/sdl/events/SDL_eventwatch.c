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

#include "SDL_eventwatch_c.h"


bool SDL_InitEventWatchList(SDL_EventWatchList *list)
{
    if (list->lock == NULL) {
        list->lock = SDL_CreateMutex();
        if (list->lock == NULL) {
            return false;
        }
    }
    return true;
}

void SDL_QuitEventWatchList(SDL_EventWatchList *list)
{
    if (list->lock) {
        SDL_DestroyMutex(list->lock);
        list->lock = NULL;
    }
    if (list->watchers) {
        SDL_free(list->watchers);
        list->watchers = NULL;
        list->count = 0;
    }
    SDL_zero(list->filter);
}

bool SDL_DispatchEventWatchList(SDL_EventWatchList *list, SDL_Event *event)
{
    SDL_EventWatcher *filter = &list->filter;

    if (!filter->callback && list->count == 0) {
        return true;
    }

    SDL_LockMutex(list->lock);
    {
        // Make sure we only dispatch the current watcher list
        int i, count = list->count;

        if (filter->callback && !filter->callback(filter->userdata, event)) {
            SDL_UnlockMutex(list->lock);
            return false;
        }

        list->dispatching = true;
        for (i = 0; i < count; ++i) {
            if (!list->watchers[i].removed) {
                list->watchers[i].callback(list->watchers[i].userdata, event);
            }
        }
        list->dispatching = false;

        if (list->removed) {
            for (i = list->count; i--;) {
                if (list->watchers[i].removed) {
                    --list->count;
                    if (i < list->count) {
                        SDL_memmove(&list->watchers[i], &list->watchers[i + 1], (list->count - i) * sizeof(list->watchers[i]));
                    }
                }
            }
            list->removed = false;
        }
    }
    SDL_UnlockMutex(list->lock);

    return true;
}

bool SDL_AddEventWatchList(SDL_EventWatchList *list, SDL_EventFilter filter, void *userdata)
{
    bool result = true;

    SDL_LockMutex(list->lock);
    {
        SDL_EventWatcher *watchers;

        watchers = (SDL_EventWatcher *)SDL_realloc(list->watchers, (list->count + 1) * sizeof(*watchers));
        if (watchers) {
            SDL_EventWatcher *watcher;

            list->watchers = watchers;
            watcher = &list->watchers[list->count];
            watcher->callback = filter;
            watcher->userdata = userdata;
            watcher->removed = false;
            ++list->count;
        } else {
            result = false;
        }
    }
    SDL_UnlockMutex(list->lock);

    return result;
}

void SDL_RemoveEventWatchList(SDL_EventWatchList *list, SDL_EventFilter filter, void *userdata)
{
    SDL_LockMutex(list->lock);
    {
        int i;

        for (i = 0; i < list->count; ++i) {
            if (list->watchers[i].callback == filter && list->watchers[i].userdata == userdata) {
                if (list->dispatching) {
                    list->watchers[i].removed = true;
                    list->removed = true;
                } else {
                    --list->count;
                    if (i < list->count) {
                        SDL_memmove(&list->watchers[i], &list->watchers[i + 1], (list->count - i) * sizeof(list->watchers[i]));
                    }
                }
                break;
            }
        }
    }
    SDL_UnlockMutex(list->lock);
}
