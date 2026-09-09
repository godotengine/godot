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

/**
 * # CategoryStorage
 *
 * The storage API is a high-level API designed to abstract away the
 * portability issues that come up when using something lower-level (in SDL's
 * case, this sits on top of the [Filesystem](CategoryFilesystem) and
 * [IOStream](CategoryIOStream) subsystems). It is significantly more
 * restrictive than a typical filesystem API, for a number of reasons:
 *
 * 1. **What to Access:** A common pitfall with existing filesystem APIs is
 * the assumption that all storage is monolithic. However, many other
 * platforms (game consoles in particular) are more strict about what _type_
 * of filesystem is being accessed; for example, game content and user data
 * are usually two separate storage devices with entirely different
 * characteristics (and possibly different low-level APIs altogether!).
 *
 * 2. **How to Access:** Another common mistake is applications assuming that
 * all storage is universally writeable - again, many platforms treat game
 * content and user data as two separate storage devices, and only user data
 * is writeable while game content is read-only.
 *
 * 3. **When to Access:** The most common portability issue with filesystem
 * access is _timing_ - you cannot always assume that the storage device is
 * always accessible all of the time, nor can you assume that there are no
 * limits to how long you have access to a particular device.
 *
 * Consider the following example:
 *
 * ```c
 * void ReadGameData(void)
 * {
 *     extern char** fileNames;
 *     extern size_t numFiles;
 *     for (size_t i = 0; i < numFiles; i += 1) {
 *         FILE *data = fopen(fileNames[i], "rwb");
 *         if (data == NULL) {
 *             // Something bad happened!
 *         } else {
 *             // A bunch of stuff happens here
 *             fclose(data);
 *         }
 *     }
 * }
 *
 * void ReadSave(void)
 * {
 *     FILE *save = fopen("saves/save0.sav", "rb");
 *     if (save == NULL) {
 *         // Something bad happened!
 *     } else {
 *         // A bunch of stuff happens here
 *         fclose(save);
 *     }
 * }
 *
 * void WriteSave(void)
 * {
 *     FILE *save = fopen("saves/save0.sav", "wb");
 *     if (save == NULL) {
 *         // Something bad happened!
 *     } else {
 *         // A bunch of stuff happens here
 *         fclose(save);
 *     }
 * }
 * ```
 *
 * Going over the bullet points again:
 *
 * 1. **What to Access:** This code accesses a global filesystem; game data
 * and saves are all presumed to be in the current working directory (which
 * may or may not be the game's installation folder!).
 *
 * 2. **How to Access:** This code assumes that content paths are writeable,
 * and that save data is also writeable despite being in the same location as
 * the game data.
 *
 * 3. **When to Access:** This code assumes that they can be called at any
 * time, since the filesystem is always accessible and has no limits on how
 * long the filesystem is being accessed.
 *
 * Due to these assumptions, the filesystem code is not portable and will fail
 * under these common scenarios:
 *
 * - The game is installed on a device that is read-only, both content loading
 *   and game saves will fail or crash outright
 * - Game/User storage is not implicitly mounted, so no files will be found
 *   for either scenario when a platform requires explicitly mounting
 *   filesystems
 * - Save data may not be safe since the I/O is not being flushed or
 *   validated, so an error occurring elsewhere in the program may result in
 *   missing/corrupted save data
 *
 * When using SDL_Storage, these types of problems are virtually impossible to
 * trip over:
 *
 * ```c
 * void ReadGameData(void)
 * {
 *     extern char** fileNames;
 *     extern size_t numFiles;
 *
 *     SDL_Storage *title = SDL_OpenTitleStorage(NULL, 0);
 *     if (title == NULL) {
 *         // Something bad happened!
 *     }
 *     while (!SDL_StorageReady(title)) {
 *         SDL_Delay(1);
 *     }
 *
 *     for (size_t i = 0; i < numFiles; i += 1) {
 *         void* dst;
 *         Uint64 dstLen = 0;
 *
 *         if (SDL_GetStorageFileSize(title, fileNames[i], &dstLen) && dstLen > 0) {
 *             dst = SDL_malloc(dstLen);
 *             if (SDL_ReadStorageFile(title, fileNames[i], dst, dstLen)) {
 *                 // A bunch of stuff happens here
 *             } else {
 *                 // Something bad happened!
 *             }
 *             SDL_free(dst);
 *         } else {
 *             // Something bad happened!
 *         }
 *     }
 *
 *     SDL_CloseStorage(title);
 * }
 *
 * void ReadSave(void)
 * {
 *     SDL_Storage *user = SDL_OpenUserStorage("libsdl", "Storage Example", 0);
 *     if (user == NULL) {
 *         // Something bad happened!
 *     }
 *     while (!SDL_StorageReady(user)) {
 *         SDL_Delay(1);
 *     }
 *
 *     Uint64 saveLen = 0;
 *     if (SDL_GetStorageFileSize(user, "save0.sav", &saveLen) && saveLen > 0) {
 *         void* dst = SDL_malloc(saveLen);
 *         if (SDL_ReadStorageFile(user, "save0.sav", dst, saveLen)) {
 *             // A bunch of stuff happens here
 *         } else {
 *             // Something bad happened!
 *         }
 *         SDL_free(dst);
 *     } else {
 *         // Something bad happened!
 *     }
 *
 *     SDL_CloseStorage(user);
 * }
 *
 * void WriteSave(void)
 * {
 *     SDL_Storage *user = SDL_OpenUserStorage("libsdl", "Storage Example", 0);
 *     if (user == NULL) {
 *         // Something bad happened!
 *     }
 *     while (!SDL_StorageReady(user)) {
 *         SDL_Delay(1);
 *     }
 *
 *     extern void *saveData; // A bunch of stuff happened here...
 *     extern Uint64 saveLen;
 *     if (!SDL_WriteStorageFile(user, "save0.sav", saveData, saveLen)) {
 *         // Something bad happened!
 *     }
 *
 *     SDL_CloseStorage(user);
 * }
 * ```
 *
 * Note the improvements that SDL_Storage makes:
 *
 * 1. **What to Access:** This code explicitly reads from a title or user
 * storage device based on the context of the function.
 *
 * 2. **How to Access:** This code explicitly uses either a read or write
 * function based on the context of the function.
 *
 * 3. **When to Access:** This code explicitly opens the device when it needs
 * to, and closes it when it is finished working with the filesystem.
 *
 * The result is an application that is significantly more robust against the
 * increasing demands of platforms and their filesystems!
 *
 * A publicly available example of an SDL_Storage backend is the
 * [Steam Cloud](https://partner.steamgames.com/doc/features/cloud)
 * backend - you can initialize Steamworks when starting the program, and then
 * SDL will recognize that Steamworks is initialized and automatically use
 * ISteamRemoteStorage when the application opens user storage. More
 * importantly, when you _open_ storage it knows to begin a "batch" of
 * filesystem operations, and when you _close_ storage it knows to end and
 * flush the batch. This is used by Steam to support
 * [Dynamic Cloud Sync](https://steamcommunity.com/groups/steamworks/announcements/detail/3142949576401813670)
 * ; users can save data on one PC, put the device to sleep, and then continue
 * playing on another PC (and vice versa) with the save data fully
 * synchronized across all devices, allowing for a seamless experience without
 * having to do full restarts of the program.
 *
 * ## Notes on valid paths
 *
 * All paths in the Storage API use Unix-style path separators ('/'). Using a
 * different path separator will not work, even if the underlying platform
 * would otherwise accept it. This is to keep code using the Storage API
 * portable between platforms and Storage implementations and simplify app
 * code.
 *
 * Paths with relative directories ("." and "..") are forbidden by the Storage
 * API.
 *
 * All valid UTF-8 strings (discounting the NULL terminator character and the
 * '/' path separator) are usable for filenames, however, an underlying
 * Storage implementation may not support particularly strange sequences and
 * refuse to create files with those names, etc.
 */

#ifndef SDL_storage_h_
#define SDL_storage_h_

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_properties.h>

#include <SDL3/SDL_begin_code.h>

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Function interface for SDL_Storage.
 *
 * Apps that want to supply a custom implementation of SDL_Storage will fill
 * in all the functions in this struct, and then pass it to SDL_OpenStorage to
 * create a custom SDL_Storage object.
 *
 * It is not usually necessary to do this; SDL provides standard
 * implementations for many things you might expect to do with an SDL_Storage.
 *
 * This structure should be initialized using SDL_INIT_INTERFACE()
 *
 * \since This struct is available since SDL 3.2.0.
 *
 * \sa SDL_INIT_INTERFACE
 */
typedef struct SDL_StorageInterface
{
    /* The version of this interface */
    Uint32 version;

    /* Called when the storage is closed */
    bool (SDLCALL *close)(void *userdata);

    /* Optional, returns whether the storage is currently ready for access */
    bool (SDLCALL *ready)(void *userdata);

    /* Enumerate a directory, optional for write-only storage */
    bool (SDLCALL *enumerate)(void *userdata, const char *path, SDL_EnumerateDirectoryCallback callback, void *callback_userdata);

    /* Get path information, optional for write-only storage */
    bool (SDLCALL *info)(void *userdata, const char *path, SDL_PathInfo *info);

    /* Read a file from storage, optional for write-only storage */
    bool (SDLCALL *read_file)(void *userdata, const char *path, void *destination, Uint64 length);

    /* Write a file to storage, optional for read-only storage */
    bool (SDLCALL *write_file)(void *userdata, const char *path, const void *source, Uint64 length);

    /* Create a directory, optional for read-only storage */
    bool (SDLCALL *mkdir)(void *userdata, const char *path);

    /* Remove a file or empty directory, optional for read-only storage */
    bool (SDLCALL *remove)(void *userdata, const char *path);

    /* Rename a path, optional for read-only storage */
    bool (SDLCALL *rename)(void *userdata, const char *oldpath, const char *newpath);

    /* Copy a file, optional for read-only storage */
    bool (SDLCALL *copy)(void *userdata, const char *oldpath, const char *newpath);

    /* Get the space remaining, optional for read-only storage */
    Uint64 (SDLCALL *space_remaining)(void *userdata);
} SDL_StorageInterface;

/* Check the size of SDL_StorageInterface
 *
 * If this assert fails, either the compiler is padding to an unexpected size,
 * or the interface has been updated and this should be updated to match and
 * the code using this interface should be updated to handle the old version.
 */
SDL_COMPILE_TIME_ASSERT(SDL_StorageInterface_SIZE,
    (sizeof(void *) == 4 && sizeof(SDL_StorageInterface) == 48) ||
    (sizeof(void *) == 8 && sizeof(SDL_StorageInterface) == 96));

/**
 * An abstract interface for filesystem access.
 *
 * This is an opaque datatype. One can create this object using standard SDL
 * functions like SDL_OpenTitleStorage or SDL_OpenUserStorage, etc, or create
 * an object with a custom implementation using SDL_OpenStorage.
 *
 * \since This struct is available since SDL 3.2.0.
 */
typedef struct SDL_Storage SDL_Storage;

/**
 * Opens up a read-only container for the application's filesystem.
 *
 * \param override a path to override the backend's default title root.
 * \param props a property list that may contain backend-specific information.
 * \returns a title storage container on success or NULL on failure; call
 *          SDL_GetError() for more information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CloseStorage
 * \sa SDL_GetStorageFileSize
 * \sa SDL_OpenUserStorage
 * \sa SDL_ReadStorageFile
 */
extern SDL_DECLSPEC SDL_Storage * SDLCALL SDL_OpenTitleStorage(const char *override, SDL_PropertiesID props);

/**
 * Opens up a container for a user's unique read/write filesystem.
 *
 * While title storage can generally be kept open throughout runtime, user
 * storage should only be opened when the client is ready to read/write files.
 * This allows the backend to properly batch file operations and flush them
 * when the container has been closed; ensuring safe and optimal save I/O.
 *
 * \param org the name of your organization.
 * \param app the name of your application.
 * \param props a property list that may contain backend-specific information.
 * \returns a user storage container on success or NULL on failure; call
 *          SDL_GetError() for more information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CloseStorage
 * \sa SDL_GetStorageFileSize
 * \sa SDL_GetStorageSpaceRemaining
 * \sa SDL_OpenTitleStorage
 * \sa SDL_ReadStorageFile
 * \sa SDL_StorageReady
 * \sa SDL_WriteStorageFile
 */
extern SDL_DECLSPEC SDL_Storage * SDLCALL SDL_OpenUserStorage(const char *org, const char *app, SDL_PropertiesID props);

/**
 * Opens up a container for local filesystem storage.
 *
 * This is provided for development and tools. Portable applications should
 * use SDL_OpenTitleStorage() for access to game data and
 * SDL_OpenUserStorage() for access to user data.
 *
 * \param path the base path prepended to all storage paths, or NULL for no
 *             base path.
 * \returns a filesystem storage container on success or NULL on failure; call
 *          SDL_GetError() for more information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CloseStorage
 * \sa SDL_GetStorageFileSize
 * \sa SDL_GetStorageSpaceRemaining
 * \sa SDL_OpenTitleStorage
 * \sa SDL_OpenUserStorage
 * \sa SDL_ReadStorageFile
 * \sa SDL_WriteStorageFile
 */
extern SDL_DECLSPEC SDL_Storage * SDLCALL SDL_OpenFileStorage(const char *path);

/**
 * Opens up a container using a client-provided storage interface.
 *
 * Applications do not need to use this function unless they are providing
 * their own SDL_Storage implementation. If you just need an SDL_Storage, you
 * should use the built-in implementations in SDL, like SDL_OpenTitleStorage()
 * or SDL_OpenUserStorage().
 *
 * This function makes a copy of `iface` and the caller does not need to keep
 * it around after this call.
 *
 * \param iface the interface that implements this storage, initialized using
 *              SDL_INIT_INTERFACE().
 * \param userdata the pointer that will be passed to the interface functions.
 * \returns a storage container on success or NULL on failure; call
 *          SDL_GetError() for more information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CloseStorage
 * \sa SDL_GetStorageFileSize
 * \sa SDL_GetStorageSpaceRemaining
 * \sa SDL_INIT_INTERFACE
 * \sa SDL_ReadStorageFile
 * \sa SDL_StorageReady
 * \sa SDL_WriteStorageFile
 */
extern SDL_DECLSPEC SDL_Storage * SDLCALL SDL_OpenStorage(const SDL_StorageInterface *iface, void *userdata);

/**
 * Closes and frees a storage container.
 *
 * \param storage a storage container to close.
 * \returns true if the container was freed with no errors, false otherwise;
 *          call SDL_GetError() for more information. Even if the function
 *          returns an error, the container data will be freed; the error is
 *          only for informational purposes.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_OpenFileStorage
 * \sa SDL_OpenStorage
 * \sa SDL_OpenTitleStorage
 * \sa SDL_OpenUserStorage
 */
extern SDL_DECLSPEC bool SDLCALL SDL_CloseStorage(SDL_Storage *storage);

/**
 * Checks if the storage container is ready to use.
 *
 * This function should be called in regular intervals until it returns true -
 * however, it is not recommended to spinwait on this call, as the backend may
 * depend on a synchronous message loop. You might instead poll this in your
 * game's main loop while processing events and drawing a loading screen.
 *
 * \param storage a storage container to query.
 * \returns true if the container is ready, false otherwise.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_StorageReady(SDL_Storage *storage);

/**
 * Query the size of a file within a storage container.
 *
 * \param storage a storage container to query.
 * \param path the relative path of the file to query.
 * \param length a pointer to be filled with the file's length.
 * \returns true if the file could be queried or false on failure; call
 *          SDL_GetError() for more information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_ReadStorageFile
 * \sa SDL_StorageReady
 */
extern SDL_DECLSPEC bool SDLCALL SDL_GetStorageFileSize(SDL_Storage *storage, const char *path, Uint64 *length);

/**
 * Synchronously read a file from a storage container into a client-provided
 * buffer.
 *
 * The value of `length` must match the length of the file exactly; call
 * SDL_GetStorageFileSize() to get this value. This behavior may be relaxed in
 * a future release.
 *
 * \param storage a storage container to read from.
 * \param path the relative path of the file to read.
 * \param destination a client-provided buffer to read the file into.
 * \param length the length of the destination buffer.
 * \returns true if the file was read or false on failure; call SDL_GetError()
 *          for more information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetStorageFileSize
 * \sa SDL_StorageReady
 * \sa SDL_WriteStorageFile
 */
extern SDL_DECLSPEC bool SDLCALL SDL_ReadStorageFile(SDL_Storage *storage, const char *path, void *destination, Uint64 length);

/**
 * Synchronously write a file from client memory into a storage container.
 *
 * \param storage a storage container to write to.
 * \param path the relative path of the file to write.
 * \param source a client-provided buffer to write from.
 * \param length the length of the source buffer.
 * \returns true if the file was written or false on failure; call
 *          SDL_GetError() for more information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetStorageSpaceRemaining
 * \sa SDL_ReadStorageFile
 * \sa SDL_StorageReady
 */
extern SDL_DECLSPEC bool SDLCALL SDL_WriteStorageFile(SDL_Storage *storage, const char *path, const void *source, Uint64 length);

/**
 * Create a directory in a writable storage container.
 *
 * \param storage a storage container.
 * \param path the path of the directory to create.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_StorageReady
 */
extern SDL_DECLSPEC bool SDLCALL SDL_CreateStorageDirectory(SDL_Storage *storage, const char *path);

/**
 * Enumerate a directory in a storage container through a callback function.
 *
 * This function provides every directory entry through an app-provided
 * callback, called once for each directory entry, until all results have been
 * provided or the callback returns either SDL_ENUM_SUCCESS or
 * SDL_ENUM_FAILURE.
 *
 * This will return false if there was a system problem in general, or if a
 * callback returns SDL_ENUM_FAILURE. A successful return means a callback
 * returned SDL_ENUM_SUCCESS to halt enumeration, or all directory entries
 * were enumerated.
 *
 * If `path` is NULL, this is treated as a request to enumerate the root of
 * the storage container's tree. An empty string also works for this.
 *
 * \param storage a storage container.
 * \param path the path of the directory to enumerate, or NULL for the root.
 * \param callback a function that is called for each entry in the directory.
 * \param userdata a pointer that is passed to `callback`.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_StorageReady
 */
extern SDL_DECLSPEC bool SDLCALL SDL_EnumerateStorageDirectory(SDL_Storage *storage, const char *path, SDL_EnumerateDirectoryCallback callback, void *userdata);

/**
 * Remove a file or an empty directory in a writable storage container.
 *
 * \param storage a storage container.
 * \param path the path of the directory to enumerate.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_StorageReady
 */
extern SDL_DECLSPEC bool SDLCALL SDL_RemoveStoragePath(SDL_Storage *storage, const char *path);

/**
 * Rename a file or directory in a writable storage container.
 *
 * \param storage a storage container.
 * \param oldpath the old path.
 * \param newpath the new path.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_StorageReady
 */
extern SDL_DECLSPEC bool SDLCALL SDL_RenameStoragePath(SDL_Storage *storage, const char *oldpath, const char *newpath);

/**
 * Copy a file in a writable storage container.
 *
 * \param storage a storage container.
 * \param oldpath the old path.
 * \param newpath the new path.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_StorageReady
 */
extern SDL_DECLSPEC bool SDLCALL SDL_CopyStorageFile(SDL_Storage *storage, const char *oldpath, const char *newpath);

/**
 * Get information about a filesystem path in a storage container.
 *
 * \param storage a storage container.
 * \param path the path to query.
 * \param info a pointer filled in with information about the path, or NULL to
 *             check for the existence of a file.
 * \returns true on success or false if the file doesn't exist, or another
 *          failure; call SDL_GetError() for more information.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_StorageReady
 */
extern SDL_DECLSPEC bool SDLCALL SDL_GetStoragePathInfo(SDL_Storage *storage, const char *path, SDL_PathInfo *info);

/**
 * Queries the remaining space in a storage container.
 *
 * \param storage a storage container to query.
 * \returns the amount of remaining space, in bytes.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_StorageReady
 * \sa SDL_WriteStorageFile
 */
extern SDL_DECLSPEC Uint64 SDLCALL SDL_GetStorageSpaceRemaining(SDL_Storage *storage);

/**
 * Enumerate a directory tree, filtered by pattern, and return a list.
 *
 * Files are filtered out if they don't match the string in `pattern`, which
 * may contain wildcard characters `*` (match everything) and `?` (match one
 * character). If pattern is NULL, no filtering is done and all results are
 * returned. Subdirectories are permitted, and are specified with a path
 * separator of '/'. Wildcard characters `*` and `?` never match a path
 * separator.
 *
 * `flags` may be set to SDL_GLOB_CASEINSENSITIVE to make the pattern matching
 * case-insensitive.
 *
 * The returned array is always NULL-terminated, for your iterating
 * convenience, but if `count` is non-NULL, on return it will contain the
 * number of items in the array, not counting the NULL terminator.
 *
 * If `path` is NULL, this is treated as a request to enumerate the root of
 * the storage container's tree. An empty string also works for this.
 *
 * \param storage a storage container.
 * \param path the path of the directory to enumerate, or NULL for the root.
 * \param pattern the pattern that files in the directory must match. Can be
 *                NULL.
 * \param flags `SDL_GLOB_*` bitflags that affect this search.
 * \param count on return, will be set to the number of items in the returned
 *              array. Can be NULL.
 * \returns an array of strings on success or NULL on failure; call
 *          SDL_GetError() for more information. The caller should pass the
 *          returned pointer to SDL_free when done with it. This is a single
 *          allocation that should be freed with SDL_free() when it is no
 *          longer needed.
 *
 * \threadsafety It is safe to call this function from any thread, assuming
 *               the `storage` object is thread-safe.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC char ** SDLCALL SDL_GlobStorageDirectory(SDL_Storage *storage, const char *path, const char *pattern, SDL_GlobFlags flags, int *count);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_storage_h_ */
