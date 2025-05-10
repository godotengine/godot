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

#ifdef SDL_PLATFORM_PSP

// SDL_RunApp() for PSP based on SDL_psp_main.c, placed in the public domain by Sam Lantinga  3/13/14

#include <pspkernel.h>
#include <pspthreadman.h>
#include "../../events/SDL_events_c.h"

/* If application's main() is redefined as SDL_main, and libSDL_main is
   linked, then this file will create the standard exit callback,
   define the PSP_MODULE_INFO macro, and exit back to the browser when
   the program is finished.

   You can still override other parameters in your own code if you
   desire, such as PSP_HEAP_SIZE_KB, PSP_MAIN_THREAD_ATTR,
   PSP_MAIN_THREAD_STACK_SIZE, etc.
*/

PSP_MODULE_INFO("SDL App", 0, 1, 0);
PSP_MAIN_THREAD_ATTR(THREAD_ATTR_VFPU | THREAD_ATTR_USER);

int sdl_psp_exit_callback(int arg1, int arg2, void *common)
{
    SDL_SendQuit();
    return 0;
}

int sdl_psp_callback_thread(SceSize args, void *argp)
{
    int cbid;
    cbid = sceKernelCreateCallback("Exit Callback",
                                   sdl_psp_exit_callback, NULL);
    sceKernelRegisterExitCallback(cbid);
    sceKernelSleepThreadCB();
    return 0;
}

int sdl_psp_setup_callbacks(void)
{
    int thid;
    thid = sceKernelCreateThread("update_thread",
                                 sdl_psp_callback_thread, 0x11, 0xFA0, 0, 0);
    if (thid >= 0) {
        sceKernelStartThread(thid, 0, 0);
    }
    return thid;
}

int SDL_RunApp(int argc, char* argv[], SDL_main_func mainFunction, void * reserved)
{
    (void)reserved;
    sdl_psp_setup_callbacks();

    SDL_SetMainReady();

    return mainFunction(argc, argv);
}

#endif // SDL_PLATFORM_PSP
