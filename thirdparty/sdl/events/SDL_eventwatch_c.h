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

typedef struct SDL_EventWatcher
{
    SDL_EventFilter callback;
    void *userdata;
    bool removed;
} SDL_EventWatcher;

typedef struct SDL_EventWatchList
{
    SDL_Mutex *lock;
    SDL_EventWatcher filter;
    SDL_EventWatcher *watchers;
    int count;
    bool dispatching;
    bool removed;
} SDL_EventWatchList;


extern bool SDL_InitEventWatchList(SDL_EventWatchList *list);
extern void SDL_QuitEventWatchList(SDL_EventWatchList *list);
extern bool SDL_DispatchEventWatchList(SDL_EventWatchList *list, SDL_Event *event);
extern bool SDL_AddEventWatchList(SDL_EventWatchList *list, SDL_EventFilter filter, void *userdata);
extern void SDL_RemoveEventWatchList(SDL_EventWatchList *list, SDL_EventFilter filter, void *userdata);
