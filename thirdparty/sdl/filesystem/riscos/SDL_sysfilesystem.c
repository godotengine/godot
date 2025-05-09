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

#ifdef SDL_FILESYSTEM_RISCOS

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// System dependent filesystem routines

#include "../SDL_sysfilesystem.h"

#include <kernel.h>
#include <swis.h>
#include <unixlib/local.h>

// Wrapper around __unixify_std that uses SDL's memory allocators
static char *SDL_unixify_std(const char *ro_path, char *buffer, size_t buf_len, int filetype)
{
    const char *const in_buf = buffer; // = NULL if we allocate the buffer.

    if (!buffer) {
        /* This matches the logic in __unixify, with an additional byte for the
         * extra path separator.
         */
        buf_len = SDL_strlen(ro_path) + 14 + 1;
        buffer = SDL_malloc(buf_len);

        if (!buffer) {
            return NULL;
        }
    }

    if (!__unixify_std(ro_path, buffer, buf_len, filetype)) {
        if (!in_buf) {
            SDL_free(buffer);
        }

        SDL_SetError("Could not convert '%s' to a Unix-style path", ro_path);
        return NULL;
    }

    /* HACK: It's necessary to add an extra path separator here since SDL's API
     * requires it, however paths with trailing separators aren't normally valid
     * on RISC OS.
     */
    if (__get_riscosify_control() & __RISCOSIFY_NO_PROCESS)
        SDL_strlcat(buffer, ".", buf_len);
    else
        SDL_strlcat(buffer, "/", buf_len);

    return buffer;
}

static char *canonicalisePath(const char *path, const char *pathVar)
{
    _kernel_oserror *error;
    _kernel_swi_regs regs;
    char *buf;

    regs.r[0] = 37;
    regs.r[1] = (int)path;
    regs.r[2] = 0;
    regs.r[3] = (int)pathVar;
    regs.r[4] = 0;
    regs.r[5] = 0;
    error = _kernel_swi(OS_FSControl, &regs, &regs);
    if (error) {
        SDL_SetError("Couldn't canonicalise path: %s", error->errmess);
        return NULL;
    }

    regs.r[5] = 1 - regs.r[5];
    buf = SDL_malloc(regs.r[5]);
    if (!buf) {
        return NULL;
    }
    regs.r[2] = (int)buf;
    error = _kernel_swi(OS_FSControl, &regs, &regs);
    if (error) {
        SDL_SetError("Couldn't canonicalise path: %s", error->errmess);
        SDL_free(buf);
        return NULL;
    }

    return buf;
}

static _kernel_oserror *createDirectoryRecursive(char *path)
{
    char *ptr = NULL;
    _kernel_oserror *error;
    _kernel_swi_regs regs;
    regs.r[0] = 8;
    regs.r[1] = (int)path;
    regs.r[2] = 0;

    for (ptr = path + 1; *ptr; ptr++) {
        if (*ptr == '.') {
            *ptr = '\0';
            error = _kernel_swi(OS_File, &regs, &regs);
            *ptr = '.';
            if (error) {
                return error;
            }
        }
    }
    return _kernel_swi(OS_File, &regs, &regs);
}

char *SDL_SYS_GetBasePath(void)
{
    _kernel_swi_regs regs;
    _kernel_oserror *error;
    char *canon, *ptr, *result;

    error = _kernel_swi(OS_GetEnv, &regs, &regs);
    if (error) {
        return NULL;
    }

    canon = canonicalisePath((const char *)regs.r[0], "Run$Path");
    if (!canon) {
        return NULL;
    }

    // chop off filename.
    ptr = SDL_strrchr(canon, '.');
    if (ptr) {
        *ptr = '\0';
    }

    result = SDL_unixify_std(canon, NULL, 0, __RISCOSIFY_FILETYPE_NOTSPECIFIED);
    SDL_free(canon);
    return result;
}

char *SDL_SYS_GetPrefPath(const char *org, const char *app)
{
    char *canon, *dir, *result;
    size_t len;
    _kernel_oserror *error;

    if (!app) {
        SDL_InvalidParamError("app");
        return NULL;
    }
    if (!org) {
        org = "";
    }

    canon = canonicalisePath("<Choices$Write>", "Run$Path");
    if (!canon) {
        return NULL;
    }

    len = SDL_strlen(canon) + SDL_strlen(org) + SDL_strlen(app) + 4;
    dir = (char *)SDL_malloc(len);
    if (!dir) {
        SDL_free(canon);
        return NULL;
    }

    if (*org) {
        SDL_snprintf(dir, len, "%s.%s.%s", canon, org, app);
    } else {
        SDL_snprintf(dir, len, "%s.%s", canon, app);
    }

    SDL_free(canon);

    error = createDirectoryRecursive(dir);
    if (error) {
        SDL_SetError("Couldn't create directory: %s", error->errmess);
        SDL_free(dir);
        return NULL;
    }

    result = SDL_unixify_std(dir, NULL, 0, __RISCOSIFY_FILETYPE_NOTSPECIFIED);
    SDL_free(dir);
    return result;
}

// TODO
char *SDL_SYS_GetUserFolder(SDL_Folder folder)
{
    SDL_Unsupported();
    return NULL;
}

#endif // SDL_FILESYSTEM_RISCOS
