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

typedef struct SDL_ProcessData SDL_ProcessData;

struct SDL_Process
{
    bool alive;
    bool background;
    int exitcode;
    SDL_PropertiesID props;
    SDL_ProcessData *internal;
};

bool SDL_SYS_CreateProcessWithProperties(SDL_Process *process, SDL_PropertiesID props);
bool SDL_SYS_KillProcess(SDL_Process *process, bool force);
bool SDL_SYS_WaitProcess(SDL_Process *process, bool block, int *exitcode);
void SDL_SYS_DestroyProcess(SDL_Process *process);
