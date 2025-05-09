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

extern "C" {
#include "../../core/gdk/SDL_gdk.h"
#include "../../core/windows/SDL_windows.h"
#include "../../events/SDL_events_c.h"
}
#include <XGameRuntime.h>
#include <xsapi-c/services_c.h>
#include <shellapi.h> // CommandLineToArgvW()
#include <appnotify.h>

// Pop up an out of memory message, returns to Windows
static BOOL OutOfMemory(void)
{
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fatal Error", "Out of memory - aborting", NULL);
    return FALSE;
}

/* Gets the arguments with GetCommandLine, converts them to argc and argv
   and calls SDL_main */
extern "C"
int SDL_RunApp(int, char**, SDL_main_func mainFunction, void *reserved)
{
    LPWSTR *argvw;
    char **argv;
    int i, argc, result;
    HRESULT hr;
    XTaskQueueHandle taskQueue;

    argvw = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (argvw == NULL) {
        return OutOfMemory();
    }

    /* Note that we need to be careful about how we allocate/free memory here.
     * If the application calls SDL_SetMemoryFunctions(), we can't rely on
     * SDL_free() to use the same allocator after SDL_main() returns.
     */

    // Parse it into argv and argc
    argv = (char **)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, (argc + 1) * sizeof(*argv));
    if (argv == NULL) {
        return OutOfMemory();
    }
    for (i = 0; i < argc; ++i) {
        const int utf8size = WideCharToMultiByte(CP_UTF8, 0, argvw[i], -1, NULL, 0, NULL, NULL);
        if (!utf8size) {  // uhoh?
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fatal Error", "Error processing command line arguments", NULL);
            return -1;
        }

        argv[i] = (char *)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, utf8size);  // this size includes the null-terminator character.
        if (!argv[i]) {
            return OutOfMemory();
        }

        if (WideCharToMultiByte(CP_UTF8, 0, argvw[i], -1, argv[i], utf8size, NULL, NULL) == 0) {  // failed? uhoh!
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fatal Error", "Error processing command line arguments", NULL);
            return -1;
        }
    }
    argv[i] = NULL;
    LocalFree(argvw);

    hr = XGameRuntimeInitialize();

    if (SUCCEEDED(hr) && SDL_GetGDKTaskQueue(&taskQueue)) {
        Uint32 titleid = 0;
        char scidBuffer[64];
        XblInitArgs xblArgs;

        XTaskQueueSetCurrentProcessTaskQueue(taskQueue);

        // Try to get the title ID and initialize Xbox Live
        hr = XGameGetXboxTitleId(&titleid);
        if (SUCCEEDED(hr)) {
            SDL_zero(xblArgs);
            xblArgs.queue = taskQueue;
            SDL_snprintf(scidBuffer, 64, "00000000-0000-0000-0000-0000%08X", titleid);
            xblArgs.scid = scidBuffer;
            hr = XblInitialize(&xblArgs);
        } else {
            SDL_SetError("[GDK] Unable to get titleid. Will not call XblInitialize. Check MicrosoftGame.config!");
        }

        SDL_SetMainReady();

        if (!GDK_RegisterChangeNotifications()) {
            return -1;
        }

        // Run the application main() code
        result = mainFunction(argc, argv);

        GDK_UnregisterChangeNotifications();

        // !!! FIXME: This follows the docs exactly, but for some reason still leaks handles on exit?
        // Terminate the task queue and dispatch any pending tasks
        XTaskQueueTerminate(taskQueue, false, nullptr, nullptr);
        while (XTaskQueueDispatch(taskQueue, XTaskQueuePort::Completion, 0))
            ;

        XTaskQueueCloseHandle(taskQueue);

        XGameRuntimeUninitialize();
    } else {
#ifdef SDL_PLATFORM_WINGDK
        if (hr == E_GAMERUNTIME_DLL_NOT_FOUND) {
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fatal Error", "[GDK] Gaming Runtime library not found (xgameruntime.dll)", NULL);
        } else {
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fatal Error", "[GDK] Could not initialize - aborting", NULL);
        }
#else
        SDL_assert_always(0 && "[GDK] Could not initialize - aborting");
#endif
        result = -1;
    }

    // Free argv, to avoid memory leak
    for (i = 0; i < argc; ++i) {
        HeapFree(GetProcessHeap(), 0, argv[i]);
    }
    HeapFree(GetProcessHeap(), 0, argv);

    return result;
}

