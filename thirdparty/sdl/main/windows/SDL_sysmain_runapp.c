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

#ifdef SDL_PLATFORM_WIN32

#include "../../core/windows/SDL_windows.h"

/* Win32-specific SDL_RunApp(), which does most of the SDL_main work,
  based on SDL_windows_main.c, placed in the public domain by Sam Lantinga  4/13/98 */

#include <shellapi.h> // CommandLineToArgvW()

// Pop up an out of memory message, returns to Windows
static int OutOfMemory(void)
{
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fatal Error", "Out of memory - aborting", NULL);
    return -1;
}

int MINGW32_FORCEALIGN SDL_RunApp(int _argc, char* _argv[], SDL_main_func mainFunction, void * reserved)
{
    /* Gets the arguments with GetCommandLine, converts them to argc and argv
       and calls SDL_main */

    LPWSTR *argvw;
    char **argv;
    int i, argc, result;

    (void)_argc; (void)_argv; (void)reserved;

    argvw = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argvw) {
        return OutOfMemory();
    }

    /* Note that we need to be careful about how we allocate/free memory here.
     * If the application calls SDL_SetMemoryFunctions(), we can't rely on
     * SDL_free() to use the same allocator after SDL_main() returns.
     */

    // Parse it into argv and argc
    argv = (char **)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, (argc + 1) * sizeof(*argv));
    if (!argv) {
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

    SDL_SetMainReady();

    // Run the application main() code
    result = mainFunction(argc, argv);

    // Free argv, to avoid memory leak
    for (i = 0; i < argc; ++i) {
        HeapFree(GetProcessHeap(), 0, argv[i]);
    }
    HeapFree(GetProcessHeap(), 0, argv);

    return result;
}

#endif // SDL_PLATFORM_WIN32
