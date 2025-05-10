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

#ifdef SDL_FILESYSTEM_COCOA

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// System dependent filesystem routines

#include "../SDL_sysfilesystem.h"

#include <Foundation/Foundation.h>
#include <sys/stat.h>
#include <sys/types.h>

char *SDL_SYS_GetBasePath(void)
{
    @autoreleasepool {
        NSBundle *bundle = [NSBundle mainBundle];
        const char *baseType = [[[bundle infoDictionary] objectForKey:@"SDL_FILESYSTEM_BASE_DIR_TYPE"] UTF8String];
        const char *base = NULL;
        char *result = NULL;

        if (baseType == NULL) {
            baseType = "resource";
        }
        if (SDL_strcasecmp(baseType, "bundle") == 0) {
            base = [[bundle bundlePath] fileSystemRepresentation];
        } else if (SDL_strcasecmp(baseType, "parent") == 0) {
            base = [[[bundle bundlePath] stringByDeletingLastPathComponent] fileSystemRepresentation];
        } else {
            // this returns the exedir for non-bundled  and the resourceDir for bundled apps
            base = [[bundle resourcePath] fileSystemRepresentation];
        }

        if (base) {
            const size_t len = SDL_strlen(base) + 2;
            result = (char *)SDL_malloc(len);
            if (result != NULL) {
                SDL_snprintf(result, len, "%s/", base);
            }
        }

        return result;
    }
}

char *SDL_SYS_GetPrefPath(const char *org, const char *app)
{
    @autoreleasepool {
        char *result = NULL;
        NSArray *array;

        if (!app) {
            SDL_InvalidParamError("app");
            return NULL;
        }
        if (!org) {
            org = "";
        }

#ifndef SDL_PLATFORM_TVOS
        array = NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory, NSUserDomainMask, YES);
#else
        /* tvOS does not have persistent local storage!
         * The only place on-device where we can store data is
         * a cache directory that the OS can empty at any time.
         *
         * It's therefore very likely that save data will be erased
         * between sessions. If you want your app's save data to
         * actually stick around, you'll need to use iCloud storage.
         */
        {
            static bool shown = false;
            if (!shown) {
                shown = true;
                SDL_LogCritical(SDL_LOG_CATEGORY_SYSTEM, "tvOS does not have persistent local storage! Use iCloud storage if you want your data to persist between sessions.");
            }
        }

        array = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
#endif // !SDL_PLATFORM_TVOS

        if ([array count] > 0) { // we only want the first item in the list.
            NSString *str = [array objectAtIndex:0];
            const char *base = [str fileSystemRepresentation];
            if (base) {
                const size_t len = SDL_strlen(base) + SDL_strlen(org) + SDL_strlen(app) + 4;
                result = (char *)SDL_malloc(len);
                if (result != NULL) {
                    char *ptr;
                    if (*org) {
                        SDL_snprintf(result, len, "%s/%s/%s/", base, org, app);
                    } else {
                        SDL_snprintf(result, len, "%s/%s/", base, app);
                    }
                    for (ptr = result + 1; *ptr; ptr++) {
                        if (*ptr == '/') {
                            *ptr = '\0';
                            mkdir(result, 0700);
                            *ptr = '/';
                        }
                    }
                    mkdir(result, 0700);
                }
            }
        }

        return result;
    }
}

char *SDL_SYS_GetUserFolder(SDL_Folder folder)
{
    @autoreleasepool {
#ifdef SDL_PLATFORM_TVOS
        SDL_SetError("tvOS does not have persistent storage");
        return NULL;
#else
        char *result = NULL;
        const char* base;
        NSArray *array;
        NSSearchPathDirectory dir;
        NSString *str;
        char *ptr;

        switch (folder) {
        case SDL_FOLDER_HOME:
            base = SDL_getenv("HOME");

            if (!base) {
                SDL_SetError("No $HOME environment variable available");
                return NULL;
            }

            goto append_slash;

        case SDL_FOLDER_DESKTOP:
            dir = NSDesktopDirectory;
            break;

        case SDL_FOLDER_DOCUMENTS:
            dir = NSDocumentDirectory;
            break;

        case SDL_FOLDER_DOWNLOADS:
            dir = NSDownloadsDirectory;
            break;

        case SDL_FOLDER_MUSIC:
            dir = NSMusicDirectory;
            break;

        case SDL_FOLDER_PICTURES:
            dir = NSPicturesDirectory;
            break;

        case SDL_FOLDER_PUBLICSHARE:
            dir = NSSharedPublicDirectory;
            break;

        case SDL_FOLDER_SAVEDGAMES:
            SDL_SetError("Saved games folder not supported on Cocoa");
            return NULL;

        case SDL_FOLDER_SCREENSHOTS:
            SDL_SetError("Screenshots folder not supported on Cocoa");
            return NULL;

        case SDL_FOLDER_TEMPLATES:
            SDL_SetError("Templates folder not supported on Cocoa");
            return NULL;

        case SDL_FOLDER_VIDEOS:
            dir = NSMoviesDirectory;
            break;

        default:
            SDL_SetError("Invalid SDL_Folder: %d", (int) folder);
            return NULL;
        };

        array = NSSearchPathForDirectoriesInDomains(dir, NSUserDomainMask, YES);

        if ([array count] <= 0) {
            SDL_SetError("Directory not found");
            return NULL;
        }

        str = [array objectAtIndex:0];
        base = [str fileSystemRepresentation];
        if (!base) {
            SDL_SetError("Couldn't get folder path");
            return NULL;
        }

append_slash:
        result = SDL_malloc(SDL_strlen(base) + 2);
        if (result == NULL) {
            return NULL;
        }

        if (SDL_snprintf(result, SDL_strlen(base) + 2, "%s/", base) < 0) {
            SDL_SetError("Couldn't snprintf folder path for Cocoa: %s", base);
            SDL_free(result);
            return NULL;
        }

        for (ptr = result + 1; *ptr; ptr++) {
            if (*ptr == '/') {
                *ptr = '\0';
                mkdir(result, 0700);
                *ptr = '/';
            }
        }

        return result;
#endif // SDL_PLATFORM_TVOS
    }
}

#endif // SDL_FILESYSTEM_COCOA
