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

#ifdef SDL_FILESYSTEM_WINDOWS

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// System dependent filesystem routines

#include "../SDL_sysfilesystem.h"

#include "../../core/windows/SDL_windows.h"
#include <shlobj.h>
#include <initguid.h>

// These aren't all defined in older SDKs, so define them here
DEFINE_GUID(SDL_FOLDERID_Profile, 0x5E6C858F, 0x0E22, 0x4760, 0x9A, 0xFE, 0xEA, 0x33, 0x17, 0xB6, 0x71, 0x73);
DEFINE_GUID(SDL_FOLDERID_Desktop, 0xB4BFCC3A, 0xDB2C, 0x424C, 0xB0, 0x29, 0x7F, 0xE9, 0x9A, 0x87, 0xC6, 0x41);
DEFINE_GUID(SDL_FOLDERID_Documents, 0xFDD39AD0, 0x238F, 0x46AF, 0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7);
DEFINE_GUID(SDL_FOLDERID_Downloads, 0x374de290, 0x123f, 0x4565, 0x91, 0x64, 0x39, 0xc4, 0x92, 0x5e, 0x46, 0x7b);
DEFINE_GUID(SDL_FOLDERID_Music, 0x4BD8D571, 0x6D19, 0x48D3, 0xBE, 0x97, 0x42, 0x22, 0x20, 0x08, 0x0E, 0x43);
DEFINE_GUID(SDL_FOLDERID_Pictures, 0x33E28130, 0x4E1E, 0x4676, 0x83, 0x5A, 0x98, 0x39, 0x5C, 0x3B, 0xC3, 0xBB);
DEFINE_GUID(SDL_FOLDERID_SavedGames, 0x4c5c32ff, 0xbb9d, 0x43b0, 0xb5, 0xb4, 0x2d, 0x72, 0xe5, 0x4e, 0xaa, 0xa4);
DEFINE_GUID(SDL_FOLDERID_Screenshots, 0xb7bede81, 0xdf94, 0x4682, 0xa7, 0xd8, 0x57, 0xa5, 0x26, 0x20, 0xb8, 0x6f);
DEFINE_GUID(SDL_FOLDERID_Templates, 0xA63293E8, 0x664E, 0x48DB, 0xA0, 0x79, 0xDF, 0x75, 0x9E, 0x05, 0x09, 0xF7);
DEFINE_GUID(SDL_FOLDERID_Videos, 0x18989B1D, 0x99B5, 0x455B, 0x84, 0x1C, 0xAB, 0x7C, 0x74, 0xE4, 0xDD, 0xFC);

char *SDL_SYS_GetBasePath(void)
{
    DWORD buflen = 128;
    WCHAR *path = NULL;
    char *result = NULL;
    DWORD len = 0;
    int i;

    while (true) {
        void *ptr = SDL_realloc(path, buflen * sizeof(WCHAR));
        if (!ptr) {
            SDL_free(path);
            return NULL;
        }

        path = (WCHAR *)ptr;

        len = GetModuleFileNameW(NULL, path, buflen);
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

    result = WIN_StringToUTF8W(path);
    SDL_free(path);

    return result;
}

char *SDL_SYS_GetPrefPath(const char *org, const char *app)
{
    /*
     * Vista and later has a new API for this, but SHGetFolderPath works there,
     *  and apparently just wraps the new API. This is the new way to do it:
     *
     *     SHGetKnownFolderPath(SDL_FOLDERID_RoamingAppData, KF_FLAG_CREATE,
     *                          NULL, &wszPath);
     */

    HRESULT hr = E_FAIL;
    WCHAR path[MAX_PATH];
    char *result = NULL;
    WCHAR *worg = NULL;
    WCHAR *wapp = NULL;
    size_t new_wpath_len = 0;
    BOOL api_result = FALSE;

    if (!app) {
        SDL_InvalidParamError("app");
        return NULL;
    }
    if (!org) {
        org = "";
    }

    hr = SHGetFolderPathW(NULL, CSIDL_APPDATA | CSIDL_FLAG_CREATE, NULL, 0, path);
    if (!SUCCEEDED(hr)) {
        WIN_SetErrorFromHRESULT("Couldn't locate our prefpath", hr);
        return NULL;
    }

    worg = WIN_UTF8ToStringW(org);
    if (!worg) {
        return NULL;
    }

    wapp = WIN_UTF8ToStringW(app);
    if (!wapp) {
        SDL_free(worg);
        return NULL;
    }

    new_wpath_len = SDL_wcslen(worg) + SDL_wcslen(wapp) + SDL_wcslen(path) + 3;

    if ((new_wpath_len + 1) > MAX_PATH) {
        SDL_free(worg);
        SDL_free(wapp);
        WIN_SetError("Path too long.");
        return NULL;
    }

    if (*worg) {
        SDL_wcslcat(path, L"\\", SDL_arraysize(path));
        SDL_wcslcat(path, worg, SDL_arraysize(path));
    }
    SDL_free(worg);

    api_result = CreateDirectoryW(path, NULL);
    if (api_result == FALSE) {
        if (GetLastError() != ERROR_ALREADY_EXISTS) {
            SDL_free(wapp);
            WIN_SetError("Couldn't create a prefpath.");
            return NULL;
        }
    }

    SDL_wcslcat(path, L"\\", SDL_arraysize(path));
    SDL_wcslcat(path, wapp, SDL_arraysize(path));
    SDL_free(wapp);

    api_result = CreateDirectoryW(path, NULL);
    if (api_result == FALSE) {
        if (GetLastError() != ERROR_ALREADY_EXISTS) {
            WIN_SetError("Couldn't create a prefpath.");
            return NULL;
        }
    }

    SDL_wcslcat(path, L"\\", SDL_arraysize(path));

    result = WIN_StringToUTF8W(path);

    return result;
}

char *SDL_SYS_GetUserFolder(SDL_Folder folder)
{
    typedef HRESULT (WINAPI *pfnSHGetKnownFolderPath)(REFGUID /* REFKNOWNFOLDERID */, DWORD, HANDLE, PWSTR*);
    HMODULE lib = LoadLibrary(L"Shell32.dll");
    pfnSHGetKnownFolderPath pSHGetKnownFolderPath = NULL;
    char *result = NULL;

    if (lib) {
        pSHGetKnownFolderPath = (pfnSHGetKnownFolderPath)GetProcAddress(lib, "SHGetKnownFolderPath");
    }

    if (pSHGetKnownFolderPath) {
        GUID type; // KNOWNFOLDERID
        HRESULT hr;
        wchar_t *path;

        switch (folder) {
        case SDL_FOLDER_HOME:
            type = SDL_FOLDERID_Profile;
            break;

        case SDL_FOLDER_DESKTOP:
            type = SDL_FOLDERID_Desktop;
            break;

        case SDL_FOLDER_DOCUMENTS:
            type = SDL_FOLDERID_Documents;
            break;

        case SDL_FOLDER_DOWNLOADS:
            type = SDL_FOLDERID_Downloads;
            break;

        case SDL_FOLDER_MUSIC:
            type = SDL_FOLDERID_Music;
            break;

        case SDL_FOLDER_PICTURES:
            type = SDL_FOLDERID_Pictures;
            break;

        case SDL_FOLDER_PUBLICSHARE:
            SDL_SetError("Public share unavailable on Windows");
            goto done;

        case SDL_FOLDER_SAVEDGAMES:
            type = SDL_FOLDERID_SavedGames;
            break;

        case SDL_FOLDER_SCREENSHOTS:
            type = SDL_FOLDERID_Screenshots;
            break;

        case SDL_FOLDER_TEMPLATES:
            type = SDL_FOLDERID_Templates;
            break;

        case SDL_FOLDER_VIDEOS:
            type = SDL_FOLDERID_Videos;
            break;

        default:
            SDL_SetError("Invalid SDL_Folder: %d", (int)folder);
            goto done;
        };

        hr = pSHGetKnownFolderPath(&type, 0x00008000 /* KF_FLAG_CREATE */, NULL, &path);
        if (SUCCEEDED(hr)) {
            result = WIN_StringToUTF8W(path);
        } else {
            WIN_SetErrorFromHRESULT("Couldn't get folder", hr);
        }

    } else {
        int type;
        HRESULT hr;
        wchar_t path[MAX_PATH];

        switch (folder) {
        case SDL_FOLDER_HOME:
            type = CSIDL_PROFILE;
            break;

        case SDL_FOLDER_DESKTOP:
            type = CSIDL_DESKTOP;
            break;

        case SDL_FOLDER_DOCUMENTS:
            type = CSIDL_MYDOCUMENTS;
            break;

        case SDL_FOLDER_DOWNLOADS:
            SDL_SetError("Downloads folder unavailable before Vista");
            goto done;

        case SDL_FOLDER_MUSIC:
            type = CSIDL_MYMUSIC;
            break;

        case SDL_FOLDER_PICTURES:
            type = CSIDL_MYPICTURES;
            break;

        case SDL_FOLDER_PUBLICSHARE:
            SDL_SetError("Public share unavailable on Windows");
            goto done;

        case SDL_FOLDER_SAVEDGAMES:
            SDL_SetError("Saved games unavailable before Vista");
            goto done;

        case SDL_FOLDER_SCREENSHOTS:
            SDL_SetError("Screenshots folder unavailable before Vista");
            goto done;

        case SDL_FOLDER_TEMPLATES:
            type = CSIDL_TEMPLATES;
            break;

        case SDL_FOLDER_VIDEOS:
            type = CSIDL_MYVIDEO;
            break;

        default:
            SDL_SetError("Unsupported SDL_Folder on Windows before Vista: %d", (int)folder);
            goto done;
        };

        // Create the OS-specific folder if it doesn't already exist
        type |= CSIDL_FLAG_CREATE;

#if 0
        // Apparently the oldest, but not supported in modern Windows
        HRESULT hr = SHGetSpecialFolderPath(NULL, path, type, TRUE);
#endif

        /* Windows 2000/XP and later, deprecated as of Windows 10 (still
           available), available in Wine (tested 6.0.3) */
        hr = SHGetFolderPathW(NULL, type, NULL, SHGFP_TYPE_CURRENT, path);

        // use `== TRUE` for SHGetSpecialFolderPath
        if (SUCCEEDED(hr)) {
            result = WIN_StringToUTF8W(path);
        } else {
            WIN_SetErrorFromHRESULT("Couldn't get folder", hr);
        }
    }

    if (result) {
        char *newresult = (char *) SDL_realloc(result, SDL_strlen(result) + 2);

        if (!newresult) {
            SDL_free(result);
            result = NULL; // will be returned
            goto done;
        }

        result = newresult;
        SDL_strlcat(result, "\\", SDL_strlen(result) + 2);
    }

done:
    if (lib) {
        FreeLibrary(lib);
    }
    return result;
}

char *SDL_SYS_GetCurrentDirectory(void)
{
    WCHAR *wstr = NULL;
    DWORD buflen = 0;
    while (true) {
        const DWORD bw = GetCurrentDirectoryW(buflen, wstr);
        if (bw == 0) {
            WIN_SetError("GetCurrentDirectoryW failed");
            return NULL;
        } else if (bw < buflen) {  // we got it!
            // make sure there's a path separator at the end.
            SDL_assert(bw < (buflen + 2));
            if ((bw == 0) || (wstr[bw-1] != '\\')) {
                wstr[bw] = '\\';
                wstr[bw + 1] = '\0';
            }
            break;
        }

        void *ptr = SDL_realloc(wstr, (bw + 1) * sizeof (WCHAR));
        if (!ptr) {
            SDL_free(wstr);
            return NULL;
        }
        wstr = (WCHAR *) ptr;
        buflen = bw;
    }

    char *retval = WIN_StringToUTF8W(wstr);
    SDL_free(wstr);
    return retval;
}

#endif // SDL_FILESYSTEM_WINDOWS
