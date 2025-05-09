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

#ifndef SDL_sysfilesystem_h_
#define SDL_sysfilesystem_h_

// return a string that we can SDL_free(). It will be cached at the higher level.
extern char *SDL_SYS_GetBasePath(void);
extern char *SDL_SYS_GetPrefPath(const char *org, const char *app);
extern char *SDL_SYS_GetUserFolder(SDL_Folder folder);
extern char *SDL_SYS_GetCurrentDirectory(void);

extern bool SDL_SYS_EnumerateDirectory(const char *path, SDL_EnumerateDirectoryCallback cb, void *userdata);
extern bool SDL_SYS_RemovePath(const char *path);
extern bool SDL_SYS_RenamePath(const char *oldpath, const char *newpath);
extern bool SDL_SYS_CopyFile(const char *oldpath, const char *newpath);
extern bool SDL_SYS_CreateDirectory(const char *path);
extern bool SDL_SYS_GetPathInfo(const char *path, SDL_PathInfo *info);

typedef bool (*SDL_GlobEnumeratorFunc)(const char *path, SDL_EnumerateDirectoryCallback cb, void *cbuserdata, void *userdata);
typedef bool (*SDL_GlobGetPathInfoFunc)(const char *path, SDL_PathInfo *info, void *userdata);
extern char **SDL_InternalGlobDirectory(const char *path, const char *pattern, SDL_GlobFlags flags, int *count, SDL_GlobEnumeratorFunc enumerator, SDL_GlobGetPathInfoFunc getpathinfo, void *userdata);

#endif

