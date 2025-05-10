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

#include "SDL_sysprocess.h"


SDL_Process *SDL_CreateProcess(const char * const *args, bool pipe_stdio)
{
    if (!args || !args[0] || !args[0][0]) {
        SDL_InvalidParamError("args");
        return NULL;
    }

    SDL_Process *process;
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, (void *)args);
    if (pipe_stdio) {
        SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDIN_NUMBER, SDL_PROCESS_STDIO_APP);
        SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_APP);
    }
    process = SDL_CreateProcessWithProperties(props);
    SDL_DestroyProperties(props);
    return process;
}

SDL_Process *SDL_CreateProcessWithProperties(SDL_PropertiesID props)
{
    const char * const *args = SDL_GetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, NULL);
    if (!args || !args[0] || !args[0][0]) {
        SDL_InvalidParamError("SDL_PROP_PROCESS_CREATE_ARGS_POINTER");
        return NULL;
    }

    SDL_Process *process = (SDL_Process *)SDL_calloc(1, sizeof(*process));
    if (!process) {
        return NULL;
    }
    process->background = SDL_GetBooleanProperty(props, SDL_PROP_PROCESS_CREATE_BACKGROUND_BOOLEAN, false);

    process->props = SDL_CreateProperties();
    if (!process->props) {
        SDL_DestroyProcess(process);
        return NULL;
    }
    SDL_SetBooleanProperty(process->props, SDL_PROP_PROCESS_BACKGROUND_BOOLEAN, process->background);

    if (!SDL_SYS_CreateProcessWithProperties(process, props)) {
        SDL_DestroyProcess(process);
        return NULL;
    }
    process->alive = true;
    return process;
}

SDL_PropertiesID SDL_GetProcessProperties(SDL_Process *process)
{
    if (!process) {
        return SDL_InvalidParamError("process");
    }
    return process->props;
}

void *SDL_ReadProcess(SDL_Process *process, size_t *datasize, int *exitcode)
{
    void *result;

    if (datasize) {
        *datasize = 0;
    }
    if (exitcode) {
        *exitcode = -1;
    }

    if (!process) {
        SDL_InvalidParamError("process");
        return NULL;
    }

    SDL_IOStream *io = (SDL_IOStream *)SDL_GetPointerProperty(process->props, SDL_PROP_PROCESS_STDOUT_POINTER, NULL);
    if (!io) {
        SDL_SetError("Process not created with I/O enabled");
        return NULL;
    }

    result = SDL_LoadFile_IO(io, datasize, false);

    SDL_WaitProcess(process, true, exitcode);

    return result;
}

SDL_IOStream *SDL_GetProcessInput(SDL_Process *process)
{
    if (!process) {
        SDL_InvalidParamError("process");
        return NULL;
    }

    SDL_IOStream *io = (SDL_IOStream *)SDL_GetPointerProperty(process->props, SDL_PROP_PROCESS_STDIN_POINTER, NULL);
    if (!io) {
        SDL_SetError("Process not created with standard input available");
        return NULL;
    }

    return io;
}

SDL_IOStream *SDL_GetProcessOutput(SDL_Process *process)
{
    if (!process) {
        SDL_InvalidParamError("process");
        return NULL;
    }

    SDL_IOStream *io = (SDL_IOStream *)SDL_GetPointerProperty(process->props, SDL_PROP_PROCESS_STDOUT_POINTER, NULL);
    if (!io) {
        SDL_SetError("Process not created with standard output available");
        return NULL;
    }

    return io;
}

bool SDL_KillProcess(SDL_Process *process, bool force)
{
    if (!process) {
        return SDL_InvalidParamError("process");
    }

    if (!process->alive) {
        return SDL_SetError("Process isn't running");
    }

    return SDL_SYS_KillProcess(process, force);
}

bool SDL_WaitProcess(SDL_Process *process, bool block, int *exitcode)
{
    if (!process) {
        return SDL_InvalidParamError("process");
    }

    if (!process->alive) {
        if (exitcode) {
            *exitcode = process->exitcode;
        }
        return true;
    }

    if (SDL_SYS_WaitProcess(process, block, &process->exitcode)) {
        process->alive = false;
        if (exitcode) {
            if (process->background) {
                process->exitcode = 0;
            }
            *exitcode = process->exitcode;
        }
        return true;
    }
    return false;
}

void SDL_DestroyProcess(SDL_Process *process)
{
    if (!process) {
        return;
    }

    // Check to see if the process has exited, will reap zombies on POSIX platforms
    if (process->alive) {
        SDL_WaitProcess(process, false, NULL);
    }

    SDL_SYS_DestroyProcess(process);
    SDL_DestroyProperties(process->props);
    SDL_free(process);
}
