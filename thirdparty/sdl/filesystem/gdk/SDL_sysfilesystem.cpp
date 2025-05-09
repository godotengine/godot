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

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// System dependent filesystem routines

extern "C" {
#include "../SDL_sysfilesystem.h"
}

#include "../../core/windows/SDL_windows.h"
#include <SDL3/SDL_hints.h>
#include <SDL3/SDL_system.h>
#include <SDL3/SDL_filesystem.h>
#include <XGameSaveFiles.h>

char *
SDL_SYS_GetBasePath(void)
{
    /* NOTE: This function is a UTF8 version of the Win32 SDL_GetBasePath()!
     * The GDK actually _recommends_ the 'A' functions over the 'W' functions :o
     */
    DWORD buflen = 128;
    CHAR *path = NULL;
    DWORD len = 0;
    int i;

    while (true) {
        void *ptr = SDL_realloc(path, buflen * sizeof(CHAR));
        if (!ptr) {
            SDL_free(path);
            return NULL;
        }

        path = (CHAR *)ptr;

        len = GetModuleFileNameA(NULL, path, buflen);
        // if it truncated, then len >= buflen - 1
        // if there was enough room (or failure), len < buflen - 1
        if (len < buflen - 1) {
            break;
        }

        // buffer too small? Try again.
        buflen *= 2;
    }

    if (len == 0) {
        SDL_free(path);
        WIN_SetError("Couldn't locate our .exe");
        return NULL;
    }

    for (i = len - 1; i > 0; i--) {
        if (path[i] == '\\') {
            break;
        }
    }

    SDL_assert(i > 0);  // Should have been an absolute path.
    path[i + 1] = '\0'; // chop off filename.

    return path;
}

char *SDL_SYS_GetPrefPath(const char *org, const char *app)
{
    XUserHandle user = NULL;
    XAsyncBlock block = { 0 };
    char *folderPath;
    HRESULT result;
    const char *csid = SDL_GetHint("SDL_GDK_SERVICE_CONFIGURATION_ID");

    if (!app) {
        SDL_InvalidParamError("app");
        return NULL;
    }

    // This should be set before calling SDL_GetPrefPath!
    if (!csid) {
        SDL_LogWarn(SDL_LOG_CATEGORY_SYSTEM, "Set SDL_GDK_SERVICE_CONFIGURATION_ID before calling SDL_GetPrefPath!");
        return SDL_strdup("T:\\");
    }

    if (!SDL_GetGDKDefaultUser(&user)) {
        // Error already set, just return
        return NULL;
    }

    if (FAILED(result = XGameSaveFilesGetFolderWithUiAsync(user, csid, &block))) {
        WIN_SetErrorFromHRESULT("XGameSaveFilesGetFolderWithUiAsync", result);
        return NULL;
    }

    folderPath = (char*) SDL_malloc(MAX_PATH);
    do {
        result = XGameSaveFilesGetFolderWithUiResult(&block, MAX_PATH, folderPath);
    } while (result == E_PENDING);
    if (FAILED(result)) {
        WIN_SetErrorFromHRESULT("XGameSaveFilesGetFolderWithUiResult", result);
        SDL_free(folderPath);
        return NULL;
    }

    /* We aren't using 'app' here because the container rules are a lot more
     * strict than the NTFS rules, so it will most likely be invalid :(
     */
    SDL_strlcat(folderPath, "\\SDLPrefPath\\", MAX_PATH);
    if (CreateDirectoryA(folderPath, NULL) == FALSE) {
        if (GetLastError() != ERROR_ALREADY_EXISTS) {
            WIN_SetError("CreateDirectoryA");
            SDL_free(folderPath);
            return NULL;
        }
    }
    return folderPath;
}

// TODO
char *SDL_SYS_GetUserFolder(SDL_Folder folder)
{
    SDL_Unsupported();
    return NULL;
}

// TODO
char *SDL_SYS_GetCurrentDirectory(void)
{
    SDL_Unsupported();
    return NULL;
}
