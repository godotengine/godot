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

#include "SDL_sysstorage.h"
#include "../filesystem/SDL_sysfilesystem.h"

// Available title storage drivers
static TitleStorageBootStrap *titlebootstrap[] = {
    &GENERIC_titlebootstrap,
    NULL
};

// Available user storage drivers
static UserStorageBootStrap *userbootstrap[] = {
#ifdef SDL_STORAGE_STEAM
    &STEAM_userbootstrap,
#endif
#ifdef SDL_STORAGE_PRIVATE
    &PRIVATE_userbootstrap,
#endif
    &GENERIC_userbootstrap,
    NULL
};

struct SDL_Storage
{
    SDL_StorageInterface iface;
    void *userdata;
};

#define CHECK_STORAGE_MAGIC()                             \
    if (!storage) {                                       \
        return SDL_SetError("Invalid storage container"); \
    }

#define CHECK_STORAGE_MAGIC_RET(result)            \
    if (!storage) {                                \
        SDL_SetError("Invalid storage container"); \
        return result;                             \
    }

// we don't make any effort to convert path separators here, because a)
// everything including Windows will accept a '/' separator and b) that
// conversion should probably happen in the storage backend anyhow.

static bool ValidateStoragePath(const char *path)
{
    if (SDL_strchr(path, '\\')) {
        return SDL_SetError("Windows-style path separators ('\\') not permitted, use '/' instead.");
    }

    const char *ptr;
    const char *prev = path;
    while ((ptr = SDL_strchr(prev, '/')) != NULL) {
        if ((SDL_strncmp(prev, "./", 2) == 0) || (SDL_strncmp(prev, "../", 3) == 0)) {
            return SDL_SetError("Relative paths not permitted");
        }
        prev = ptr + 1;
    }

    // check the last path element (or the only path element).
    if ((SDL_strcmp(prev, ".") == 0) || (SDL_strcmp(prev, "..") == 0)) {
        return SDL_SetError("Relative paths not permitted");
    }

    return true;
}

SDL_Storage *SDL_OpenTitleStorage(const char *override, SDL_PropertiesID props)
{
    SDL_Storage *storage = NULL;
    int i = 0;

    // Select the proper storage driver
    const char *driver_name = SDL_GetHint(SDL_HINT_STORAGE_TITLE_DRIVER);
    if (driver_name && *driver_name != 0) {
        const char *driver_attempt = driver_name;
        while (driver_attempt && *driver_attempt != 0 && !storage) {
            const char *driver_attempt_end = SDL_strchr(driver_attempt, ',');
            size_t driver_attempt_len = (driver_attempt_end) ? (driver_attempt_end - driver_attempt)
                                                                     : SDL_strlen(driver_attempt);

            for (i = 0; titlebootstrap[i]; ++i) {
                if ((driver_attempt_len == SDL_strlen(titlebootstrap[i]->name)) &&
                    (SDL_strncasecmp(titlebootstrap[i]->name, driver_attempt, driver_attempt_len) == 0)) {
                    storage = titlebootstrap[i]->create(override, props);
                    break;
                }
            }

            driver_attempt = (driver_attempt_end) ? (driver_attempt_end + 1) : NULL;
        }
    } else {
        for (i = 0; titlebootstrap[i]; ++i) {
            storage = titlebootstrap[i]->create(override, props);
            if (storage) {
                break;
            }
        }
    }
    if (!storage) {
        if (driver_name) {
            SDL_SetError("%s not available", driver_name);
        } else {
            SDL_SetError("No available title storage driver");
        }
    }
    return storage;
}

SDL_Storage *SDL_OpenUserStorage(const char *org, const char *app, SDL_PropertiesID props)
{
    SDL_Storage *storage = NULL;
    int i = 0;

    // Select the proper storage driver
    const char *driver_name = SDL_GetHint(SDL_HINT_STORAGE_USER_DRIVER);
    if (driver_name && *driver_name != 0) {
        const char *driver_attempt = driver_name;
        while (driver_attempt && *driver_attempt != 0 && !storage) {
            const char *driver_attempt_end = SDL_strchr(driver_attempt, ',');
            size_t driver_attempt_len = (driver_attempt_end) ? (driver_attempt_end - driver_attempt)
                                                                     : SDL_strlen(driver_attempt);

            for (i = 0; userbootstrap[i]; ++i) {
                if ((driver_attempt_len == SDL_strlen(userbootstrap[i]->name)) &&
                    (SDL_strncasecmp(userbootstrap[i]->name, driver_attempt, driver_attempt_len) == 0)) {
                    storage = userbootstrap[i]->create(org, app, props);
                    break;
                }
            }

            driver_attempt = (driver_attempt_end) ? (driver_attempt_end + 1) : NULL;
        }
    } else {
        for (i = 0; userbootstrap[i]; ++i) {
            storage = userbootstrap[i]->create(org, app, props);
            if (storage) {
                break;
            }
        }
    }
    if (!storage) {
        if (driver_name) {
            SDL_SetError("%s not available", driver_name);
        } else {
            SDL_SetError("No available user storage driver");
        }
    }
    return storage;
}

SDL_Storage *SDL_OpenFileStorage(const char *path)
{
    return GENERIC_OpenFileStorage(path);
}

SDL_Storage *SDL_OpenStorage(const SDL_StorageInterface *iface, void *userdata)
{
    SDL_Storage *storage;

    if (!iface) {
        SDL_InvalidParamError("iface");
        return NULL;
    }
    if (iface->version < sizeof(*iface)) {
        // Update this to handle older versions of this interface
        SDL_SetError("Invalid interface, should be initialized with SDL_INIT_INTERFACE()");
        return NULL;
    }

    storage = (SDL_Storage *)SDL_calloc(1, sizeof(*storage));
    if (storage) {
        SDL_copyp(&storage->iface, iface);
        storage->userdata = userdata;
    }
    return storage;
}

bool SDL_CloseStorage(SDL_Storage *storage)
{
    bool result = true;

    CHECK_STORAGE_MAGIC()

    if (storage->iface.close) {
        result = storage->iface.close(storage->userdata);
    }
    SDL_free(storage);
    return result;
}

bool SDL_StorageReady(SDL_Storage *storage)
{
    CHECK_STORAGE_MAGIC_RET(false)

    if (storage->iface.ready) {
        return storage->iface.ready(storage->userdata);
    }
    return true;
}

bool SDL_GetStorageFileSize(SDL_Storage *storage, const char *path, Uint64 *length)
{
    SDL_PathInfo info;

    if (SDL_GetStoragePathInfo(storage, path, &info)) {
        if (length) {
            *length = info.size;
        }
        return true;
    } else {
        if (length) {
            *length = 0;
        }
        return false;
    }
}

bool SDL_ReadStorageFile(SDL_Storage *storage, const char *path, void *destination, Uint64 length)
{
    CHECK_STORAGE_MAGIC()

    if (!path) {
        return SDL_InvalidParamError("path");
    } else if (!ValidateStoragePath(path)) {
        return false;
    } else if (!storage->iface.read_file) {
        return SDL_Unsupported();
    }

    return storage->iface.read_file(storage->userdata, path, destination, length);
}

bool SDL_WriteStorageFile(SDL_Storage *storage, const char *path, const void *source, Uint64 length)
{
    CHECK_STORAGE_MAGIC()

    if (!path) {
        return SDL_InvalidParamError("path");
    } else if (!ValidateStoragePath(path)) {
        return false;
    } else if (!storage->iface.write_file) {
        return SDL_Unsupported();
    }

    return storage->iface.write_file(storage->userdata, path, source, length);
}

bool SDL_CreateStorageDirectory(SDL_Storage *storage, const char *path)
{
    CHECK_STORAGE_MAGIC()

    if (!path) {
        return SDL_InvalidParamError("path");
    } else if (!ValidateStoragePath(path)) {
        return false;
    } else if (!storage->iface.mkdir) {
        return SDL_Unsupported();
    }

    return storage->iface.mkdir(storage->userdata, path);
}

bool SDL_EnumerateStorageDirectory(SDL_Storage *storage, const char *path, SDL_EnumerateDirectoryCallback callback, void *userdata)
{
    CHECK_STORAGE_MAGIC()

    if (!path) {
        path = "";  // we allow NULL to mean "root of the storage tree".
    }

    if (!ValidateStoragePath(path)) {
        return false;
    } else if (!storage->iface.enumerate) {
        return SDL_Unsupported();
    }

    return storage->iface.enumerate(storage->userdata, path, callback, userdata);
}

bool SDL_RemoveStoragePath(SDL_Storage *storage, const char *path)
{
    CHECK_STORAGE_MAGIC()

    if (!path) {
        return SDL_InvalidParamError("path");
    } else if (!ValidateStoragePath(path)) {
        return false;
    } else if (!storage->iface.remove) {
        return SDL_Unsupported();
    }

    return storage->iface.remove(storage->userdata, path);
}

bool SDL_RenameStoragePath(SDL_Storage *storage, const char *oldpath, const char *newpath)
{
    CHECK_STORAGE_MAGIC()

    if (!oldpath) {
        return SDL_InvalidParamError("oldpath");
    } else if (!newpath) {
        return SDL_InvalidParamError("newpath");
    } else if (!ValidateStoragePath(oldpath)) {
        return false;
    } else if (!ValidateStoragePath(newpath)) {
        return false;
    } else if (!storage->iface.rename) {
        return SDL_Unsupported();
    }

    return storage->iface.rename(storage->userdata, oldpath, newpath);
}

bool SDL_CopyStorageFile(SDL_Storage *storage, const char *oldpath, const char *newpath)
{
    CHECK_STORAGE_MAGIC()

    if (!oldpath) {
        return SDL_InvalidParamError("oldpath");
    } else if (!newpath) {
        return SDL_InvalidParamError("newpath");
    } else if (!ValidateStoragePath(oldpath)) {
        return false;
    } else if (!ValidateStoragePath(newpath)) {
        return false;
    } else if (!storage->iface.copy) {
        return SDL_Unsupported();
    }

    return storage->iface.copy(storage->userdata, oldpath, newpath);
}

bool SDL_GetStoragePathInfo(SDL_Storage *storage, const char *path, SDL_PathInfo *info)
{
    SDL_PathInfo dummy;

    if (!info) {
        info = &dummy;
    }
    SDL_zerop(info);

    CHECK_STORAGE_MAGIC()

    if (!path) {
        return SDL_InvalidParamError("path");
    } else if (!ValidateStoragePath(path)) {
        return false;
    } else if (!storage->iface.info) {
        return SDL_Unsupported();
    }

    return storage->iface.info(storage->userdata, path, info);
}

Uint64 SDL_GetStorageSpaceRemaining(SDL_Storage *storage)
{
    CHECK_STORAGE_MAGIC_RET(0)

    if (!storage->iface.space_remaining) {
        SDL_Unsupported();
        return 0;
    }

    return storage->iface.space_remaining(storage->userdata);
}

static bool GlobStorageDirectoryGetPathInfo(const char *path, SDL_PathInfo *info, void *userdata)
{
    return SDL_GetStoragePathInfo((SDL_Storage *) userdata, path, info);
}

static bool GlobStorageDirectoryEnumerator(const char *path, SDL_EnumerateDirectoryCallback cb, void *cbuserdata, void *userdata)
{
    return SDL_EnumerateStorageDirectory((SDL_Storage *) userdata, path, cb, cbuserdata);
}

char **SDL_GlobStorageDirectory(SDL_Storage *storage, const char *path, const char *pattern, SDL_GlobFlags flags, int *count)
{
    CHECK_STORAGE_MAGIC_RET(NULL)

    if (!path) {
        path = "";  // we allow NULL to mean "root of the storage tree".
    }

    if (!ValidateStoragePath(path)) {
        return NULL;
    }

    return SDL_InternalGlobDirectory(path, pattern, flags, count, GlobStorageDirectoryEnumerator, GlobStorageDirectoryGetPathInfo, storage);
}

