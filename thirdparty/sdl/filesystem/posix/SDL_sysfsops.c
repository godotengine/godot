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

#if defined(SDL_FSOPS_POSIX)

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// System dependent filesystem routines

#include "../SDL_sysfilesystem.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

bool SDL_SYS_EnumerateDirectory(const char *path, SDL_EnumerateDirectoryCallback cb, void *userdata)
{
    char *pathwithsep = NULL;
    int pathwithseplen = SDL_asprintf(&pathwithsep, "%s/", path);
    if ((pathwithseplen == -1) || (!pathwithsep)) {
        return false;
    }

    // trim down to a single path separator at the end, in case the caller added one or more.
    pathwithseplen--;
    while ((pathwithseplen >= 0) && (pathwithsep[pathwithseplen] == '/')) {
        pathwithsep[pathwithseplen--] = '\0';
    }

    DIR *dir = opendir(pathwithsep);
    if (!dir) {
        SDL_free(pathwithsep);
        return SDL_SetError("Can't open directory: %s", strerror(errno));
    }

    // make sure there's a path separator at the end now for the actual callback.
    pathwithsep[++pathwithseplen] = '/';
    pathwithsep[++pathwithseplen] = '\0';

    SDL_EnumerationResult result = SDL_ENUM_CONTINUE;
    struct dirent *ent;
    while ((result == SDL_ENUM_CONTINUE) && ((ent = readdir(dir)) != NULL)) {
        const char *name = ent->d_name;
        if ((SDL_strcmp(name, ".") == 0) || (SDL_strcmp(name, "..") == 0)) {
            continue;
        }
        result = cb(userdata, pathwithsep, name);
    }

    closedir(dir);

    SDL_free(pathwithsep);

    return (result != SDL_ENUM_FAILURE);
}

bool SDL_SYS_RemovePath(const char *path)
{
    int rc = remove(path);
    if (rc < 0) {
        if (errno == ENOENT) {
            // It's already gone, this is a success
            return true;
        }
        return SDL_SetError("Can't remove path: %s", strerror(errno));
    }
    return true;
}

bool SDL_SYS_RenamePath(const char *oldpath, const char *newpath)
{
    if (rename(oldpath, newpath) < 0) {
        return SDL_SetError("Can't rename path: %s", strerror(errno));
    }
    return true;
}

bool SDL_SYS_CopyFile(const char *oldpath, const char *newpath)
{
    char *buffer = NULL;
    SDL_IOStream *input = NULL;
    SDL_IOStream *output = NULL;
    const size_t maxlen = 4096;
    size_t len;
    bool result = false;

    input = SDL_IOFromFile(oldpath, "rb");
    if (!input) {
        goto done;
    }

    output = SDL_IOFromFile(newpath, "wb");
    if (!output) {
        goto done;
    }

    buffer = (char *)SDL_malloc(maxlen);
    if (!buffer) {
        goto done;
    }

    while ((len = SDL_ReadIO(input, buffer, maxlen)) > 0) {
        if (SDL_WriteIO(output, buffer, len) < len) {
            goto done;
        }
    }
    if (SDL_GetIOStatus(input) != SDL_IO_STATUS_EOF) {
        goto done;
    }

    SDL_CloseIO(input);
    input = NULL;

    if (!SDL_FlushIO(output)) {
        goto done;
    }

    result = SDL_CloseIO(output);
    output = NULL;  // it's gone, even if it failed.

done:
    if (output) {
        SDL_CloseIO(output);
    }
    if (input) {
        SDL_CloseIO(input);
    }
    SDL_free(buffer);

    return result;
}

bool SDL_SYS_CreateDirectory(const char *path)
{
    const int rc = mkdir(path, 0770);
    if (rc < 0) {
        const int origerrno = errno;
        if (origerrno == EEXIST) {
            struct stat statbuf;
            if ((stat(path, &statbuf) == 0) && (S_ISDIR(statbuf.st_mode))) {
                return true;  // it already exists and it's a directory, consider it success.
            }
        }
        return SDL_SetError("Can't create directory: %s", strerror(origerrno));
    }
    return true;
}

bool SDL_SYS_GetPathInfo(const char *path, SDL_PathInfo *info)
{
    struct stat statbuf;
    const int rc = stat(path, &statbuf);
    if (rc < 0) {
        return SDL_SetError("Can't stat: %s", strerror(errno));
    } else if (S_ISREG(statbuf.st_mode)) {
        info->type = SDL_PATHTYPE_FILE;
        info->size = (Uint64) statbuf.st_size;
    } else if (S_ISDIR(statbuf.st_mode)) {
        info->type = SDL_PATHTYPE_DIRECTORY;
        info->size = 0;
    } else {
        info->type = SDL_PATHTYPE_OTHER;
        info->size = (Uint64) statbuf.st_size;
    }

#if defined(HAVE_ST_MTIM)
    // POSIX.1-2008 standard
    info->create_time = (SDL_Time)SDL_SECONDS_TO_NS(statbuf.st_ctim.tv_sec) + statbuf.st_ctim.tv_nsec;
    info->modify_time = (SDL_Time)SDL_SECONDS_TO_NS(statbuf.st_mtim.tv_sec) + statbuf.st_mtim.tv_nsec;
    info->access_time = (SDL_Time)SDL_SECONDS_TO_NS(statbuf.st_atim.tv_sec) + statbuf.st_atim.tv_nsec;
#elif defined(SDL_PLATFORM_APPLE)
    /* Apple platform stat structs use 'st_*timespec' naming. */
    info->create_time = (SDL_Time)SDL_SECONDS_TO_NS(statbuf.st_ctimespec.tv_sec) + statbuf.st_ctimespec.tv_nsec;
    info->modify_time = (SDL_Time)SDL_SECONDS_TO_NS(statbuf.st_mtimespec.tv_sec) + statbuf.st_mtimespec.tv_nsec;
    info->access_time = (SDL_Time)SDL_SECONDS_TO_NS(statbuf.st_atimespec.tv_sec) + statbuf.st_atimespec.tv_nsec;
#else
    info->create_time = (SDL_Time)SDL_SECONDS_TO_NS(statbuf.st_ctime);
    info->modify_time = (SDL_Time)SDL_SECONDS_TO_NS(statbuf.st_mtime);
    info->access_time = (SDL_Time)SDL_SECONDS_TO_NS(statbuf.st_atime);
#endif
    return true;
}

// Note that this isn't actually part of filesystem, not fsops, but everything that uses posix fsops uses this implementation, even with separate filesystem code.
char *SDL_SYS_GetCurrentDirectory(void)
{
    size_t buflen = 64;
    char *buf = NULL;

    while (true) {
        void *ptr = SDL_realloc(buf, buflen);
        if (!ptr) {
            SDL_free(buf);
            return NULL;
        }
        buf = (char *) ptr;

        if (getcwd(buf, buflen-1) != NULL) {
            break;  // we got it!
        }

        if (errno == ERANGE) {
            buflen *= 2;  // try again with a bigger buffer.
            continue;
        }

        SDL_free(buf);
        SDL_SetError("getcwd failed: %s", strerror(errno));
        return NULL;
    }

    // make sure there's a path separator at the end.
    SDL_assert(SDL_strlen(buf) < (buflen + 2));
    buflen = SDL_strlen(buf);
    if ((buflen == 0) || (buf[buflen-1] != '/')) {
        buf[buflen] = '/';
        buf[buflen + 1] = '\0';
    }

    return buf;
}

#endif // SDL_FSOPS_POSIX

