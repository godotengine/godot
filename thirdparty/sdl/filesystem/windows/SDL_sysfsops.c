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

#if defined(SDL_FSOPS_WINDOWS)

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// System dependent filesystem routines

#include "../../core/windows/SDL_windows.h"
#include "../SDL_sysfilesystem.h"

bool SDL_SYS_EnumerateDirectory(const char *path, SDL_EnumerateDirectoryCallback cb, void *userdata)
{
    SDL_EnumerationResult result = SDL_ENUM_CONTINUE;
    if (*path == '\0') {  // if empty (completely at the root), we need to enumerate drive letters.
        const DWORD drives = GetLogicalDrives();
        char name[] = { 0, ':', '\\', '\0' };
        for (int i = 'A'; (result == SDL_ENUM_CONTINUE) && (i <= 'Z'); i++) {
            if (drives & (1 << (i - 'A'))) {
                name[0] = (char) i;
                result = cb(userdata, "", name);
            }
        }
    } else {
        // you need a wildcard to enumerate through FindFirstFileEx(), but the wildcard is only checked in the
        // filename element at the end of the path string, so always tack on a "\\*" to get everything, and
        // also prevent any wildcards inserted by the app from being respected.
        char *pattern = NULL;
        int patternlen = SDL_asprintf(&pattern, "%s\\\\", path);  // we'll replace that second '\\' in the trimdown.
        if ((patternlen == -1) || (!pattern)) {
            return false;
        }

        // trim down to a single path separator at the end, in case the caller added one or more.
        patternlen--;
        while ((patternlen >= 0) && ((pattern[patternlen] == '\\') || (pattern[patternlen] == '/'))) {
            pattern[patternlen--] ='\0';
        }
        pattern[++patternlen] = '\\';
        pattern[++patternlen] = '*';
        pattern[++patternlen] = '\0';

        WCHAR *wpattern = WIN_UTF8ToStringW(pattern);
        if (!wpattern) {
            SDL_free(pattern);
            return false;
        }

        pattern[--patternlen] = '\0';  // chop off the '*' so we just have the dirname with a path separator.

        WIN32_FIND_DATAW entw;
        HANDLE dir = FindFirstFileExW(wpattern, FindExInfoStandard, &entw, FindExSearchNameMatch, NULL, 0);
        SDL_free(wpattern);
        if (dir == INVALID_HANDLE_VALUE) {
            SDL_free(pattern);
            return WIN_SetError("Failed to enumerate directory");
        }

        do {
            const WCHAR *fn = entw.cFileName;

            if (fn[0] == '.') {  // ignore "." and ".."
                if ((fn[1] == '\0') || ((fn[1] == '.') && (fn[2] == '\0'))) {
                    continue;
                }
            }

            char *utf8fn = WIN_StringToUTF8W(fn);
            if (!utf8fn) {
                result = SDL_ENUM_FAILURE;
            } else {
                result = cb(userdata, pattern, utf8fn);
                SDL_free(utf8fn);
            }
        } while ((result == SDL_ENUM_CONTINUE) && (FindNextFileW(dir, &entw) != 0));

        FindClose(dir);
        SDL_free(pattern);
    }

    return (result != SDL_ENUM_FAILURE);
}

bool SDL_SYS_RemovePath(const char *path)
{
    WCHAR *wpath = WIN_UTF8ToStringW(path);
    if (!wpath) {
        return false;
    }

    WIN32_FILE_ATTRIBUTE_DATA info;
    if (!GetFileAttributesExW(wpath, GetFileExInfoStandard, &info)) {
        SDL_free(wpath);
        if (GetLastError() == ERROR_FILE_NOT_FOUND) {
            // Note that ERROR_PATH_NOT_FOUND means a parent dir is missing, and we consider that an error.
            return true;  // thing is already gone, call it a success.
        }
        return WIN_SetError("Couldn't get path's attributes");
    }

    const int isdir = (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
    const BOOL rc = isdir ? RemoveDirectoryW(wpath) : DeleteFileW(wpath);
    SDL_free(wpath);
    if (!rc) {
        return WIN_SetError("Couldn't remove path");
    }
    return true;
}

bool SDL_SYS_RenamePath(const char *oldpath, const char *newpath)
{
    WCHAR *woldpath = WIN_UTF8ToStringW(oldpath);
    if (!woldpath) {
        return false;
    }

    WCHAR *wnewpath = WIN_UTF8ToStringW(newpath);
    if (!wnewpath) {
        SDL_free(woldpath);
        return false;
    }

    const BOOL rc = MoveFileExW(woldpath, wnewpath, MOVEFILE_REPLACE_EXISTING);
    SDL_free(wnewpath);
    SDL_free(woldpath);
    if (!rc) {
        return WIN_SetError("Couldn't rename path");
    }
    return true;
}

bool SDL_SYS_CopyFile(const char *oldpath, const char *newpath)
{
    WCHAR *woldpath = WIN_UTF8ToStringW(oldpath);
    if (!woldpath) {
        return false;
    }

    WCHAR *wnewpath = WIN_UTF8ToStringW(newpath);
    if (!wnewpath) {
        SDL_free(woldpath);
        return false;
    }

    const BOOL rc = CopyFileExW(woldpath, wnewpath, NULL, NULL, NULL, COPY_FILE_ALLOW_DECRYPTED_DESTINATION|COPY_FILE_NO_BUFFERING);
    SDL_free(wnewpath);
    SDL_free(woldpath);
    if (!rc) {
        return WIN_SetError("Couldn't copy path");
    }
    return true;
}

bool SDL_SYS_CreateDirectory(const char *path)
{
    WCHAR *wpath = WIN_UTF8ToStringW(path);
    if (!wpath) {
        return false;
    }

    DWORD rc = CreateDirectoryW(wpath, NULL);
    if (!rc && (GetLastError() == ERROR_ALREADY_EXISTS)) {
        WIN32_FILE_ATTRIBUTE_DATA winstat;
        if (GetFileAttributesExW(wpath, GetFileExInfoStandard, &winstat)) {
            if (winstat.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                rc = 1;  // exists and is already a directory: cool.
            }
        }
    }

    SDL_free(wpath);
    if (!rc) {
        return WIN_SetError("Couldn't create directory");
    }
    return true;
}

bool SDL_SYS_GetPathInfo(const char *path, SDL_PathInfo *info)
{
    WCHAR *wpath = WIN_UTF8ToStringW(path);
    if (!wpath) {
        return false;
    }

    WIN32_FILE_ATTRIBUTE_DATA winstat;
    const BOOL rc = GetFileAttributesExW(wpath, GetFileExInfoStandard, &winstat);
    SDL_free(wpath);
    if (!rc) {
        return WIN_SetError("Can't stat");
    }

    if (winstat.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        info->type = SDL_PATHTYPE_DIRECTORY;
        info->size = 0;
    } else if (winstat.dwFileAttributes & (FILE_ATTRIBUTE_OFFLINE | FILE_ATTRIBUTE_DEVICE)) {
        info->type = SDL_PATHTYPE_OTHER;
        info->size = ((((Uint64) winstat.nFileSizeHigh) << 32) | winstat.nFileSizeLow);
    } else {
        info->type = SDL_PATHTYPE_FILE;
        info->size = ((((Uint64) winstat.nFileSizeHigh) << 32) | winstat.nFileSizeLow);
    }

    info->create_time = SDL_TimeFromWindows(winstat.ftCreationTime.dwLowDateTime, winstat.ftCreationTime.dwHighDateTime);
    info->modify_time = SDL_TimeFromWindows(winstat.ftLastWriteTime.dwLowDateTime, winstat.ftLastWriteTime.dwHighDateTime);
    info->access_time = SDL_TimeFromWindows(winstat.ftLastAccessTime.dwLowDateTime, winstat.ftLastAccessTime.dwHighDateTime);

    return true;
}

#endif // SDL_FSOPS_WINDOWS

