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

#include "SDL_iostreamromfs.h"

// Checks if the mode is a kind of reading
static bool IsReadMode(const char *mode);

// Checks if the file starts with the given prefix
static bool HasPrefix(const char *file, const char *prefix);

static FILE *TryOpenFile(const char *file, const char *mode);
static FILE *TryOpenInRomfs(const char *file, const char *mode);

/* Nintendo 3DS applications may embed resources in the executable. The
  resources are stored in a special read-only partition prefixed with
  'romfs:/'. As such, when opening a file, we should first try the romfs
  unless sdmc is specifically mentioned.
*/
FILE *N3DS_FileOpen(const char *file, const char *mode)
{
    // romfs are read-only
    if (!IsReadMode(mode)) {
        return fopen(file, mode);
    }

    // If the path has an explicit prefix, we skip the guess work
    if (HasPrefix(file, "romfs:/") || HasPrefix(file, "sdmc:/")) {
        return fopen(file, mode);
    }

    return TryOpenFile(file, mode);
}

static bool IsReadMode(const char *mode)
{
    return SDL_strchr(mode, 'r') != NULL;
}

static bool HasPrefix(const char *file, const char *prefix)
{
    return SDL_strncmp(prefix, file, SDL_strlen(prefix)) == 0;
}

static FILE *TryOpenFile(const char *file, const char *mode)
{
    FILE *fp = NULL;

    fp = TryOpenInRomfs(file, mode);
    if (!fp) {
        fp = fopen(file, mode);
    }

    return fp;
}

FILE *TryOpenInRomfs(const char *file, const char *mode)
{
    FILE *fp = NULL;
    char *prefixed_filepath = NULL;

    if (SDL_asprintf(&prefixed_filepath, "romfs:/%s", file) < 0) {
        return NULL;
    }

    fp = fopen(prefixed_filepath, mode);

    SDL_free(prefixed_filepath);
    return fp;
}
