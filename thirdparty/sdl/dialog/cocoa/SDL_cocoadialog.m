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
#include "../SDL_dialog.h"
#include "../SDL_dialog_utils.h"

#ifdef SDL_PLATFORM_MACOS

#import <Cocoa/Cocoa.h>
#import <UniformTypeIdentifiers/UTType.h>

void SDL_SYS_ShowFileDialogWithProperties(SDL_FileDialogType type, SDL_DialogFileCallback callback, void *userdata, SDL_PropertiesID props)
{
    SDL_Window* window = SDL_GetPointerProperty(props, SDL_PROP_FILE_DIALOG_WINDOW_POINTER, NULL);
    SDL_DialogFileFilter *filters = SDL_GetPointerProperty(props, SDL_PROP_FILE_DIALOG_FILTERS_POINTER, NULL);
    int nfilters = (int) SDL_GetNumberProperty(props, SDL_PROP_FILE_DIALOG_NFILTERS_NUMBER, 0);
    bool allow_many = SDL_GetBooleanProperty(props, SDL_PROP_FILE_DIALOG_MANY_BOOLEAN, false);
    const char* default_location = SDL_GetStringProperty(props, SDL_PROP_FILE_DIALOG_LOCATION_STRING, NULL);
    const char* title = SDL_GetStringProperty(props, SDL_PROP_FILE_DIALOG_TITLE_STRING, NULL);
    const char* accept = SDL_GetStringProperty(props, SDL_PROP_FILE_DIALOG_ACCEPT_STRING, NULL);

    if (filters) {
        const char *msg = validate_filters(filters, nfilters);

        if (msg) {
            SDL_SetError("%s", msg);
            callback(userdata, NULL, -1);
            return;
        }
    }

    if (SDL_GetHint(SDL_HINT_FILE_DIALOG_DRIVER) != NULL) {
        SDL_SetError("File dialog driver unsupported (don't set SDL_HINT_FILE_DIALOG_DRIVER)");
        callback(userdata, NULL, -1);
        return;
    }

    // NSOpenPanel inherits from NSSavePanel
    NSSavePanel *dialog;
    NSOpenPanel *dialog_as_open;

    switch (type) {
    case SDL_FILEDIALOG_SAVEFILE:
        dialog = [NSSavePanel savePanel];
        break;

    case SDL_FILEDIALOG_OPENFILE:
        dialog_as_open = [NSOpenPanel openPanel];
        [dialog_as_open setAllowsMultipleSelection:((allow_many == true) ? YES : NO)];
        dialog = dialog_as_open;
        break;

    case SDL_FILEDIALOG_OPENFOLDER:
        dialog_as_open = [NSOpenPanel openPanel];
        [dialog_as_open setCanChooseFiles:NO];
        [dialog_as_open setCanChooseDirectories:YES];
        [dialog_as_open setAllowsMultipleSelection:((allow_many == true) ? YES : NO)];
        dialog = dialog_as_open;
        break;
    };

    if (title) {
        [dialog setTitle:[NSString stringWithUTF8String:title]];
    }

    if (accept) {
        [dialog setPrompt:[NSString stringWithUTF8String:accept]];
    }

    if (filters) {
        // On macOS 11.0 and up, this is an array of UTType. Prior to that, it's an array of NSString
        NSMutableArray *types = [[NSMutableArray alloc] initWithCapacity:nfilters ];

        int has_all_files = 0;
        for (int i = 0; i < nfilters; i++) {
            char *pattern = SDL_strdup(filters[i].pattern);
            char *pattern_ptr = pattern;

            if (!pattern_ptr) {
                callback(userdata, NULL, -1);
                return;
            }

            for (char *c = pattern; *c; c++) {
                if (*c == ';') {
                    *c = '\0';
                    if(@available(macOS 11.0, *)) {
                        [types addObject: [UTType typeWithFilenameExtension:[NSString stringWithFormat: @"%s", pattern_ptr]]];
                    } else {
                        [types addObject: [NSString stringWithFormat: @"%s", pattern_ptr]];
                    }
                    pattern_ptr = c + 1;
                } else if (*c == '*') {
                    has_all_files = 1;
                }
            }
            if(@available(macOS 11.0, *)) {
                [types addObject: [UTType typeWithFilenameExtension:[NSString stringWithFormat: @"%s", pattern_ptr]]];
            } else {
                [types addObject: [NSString stringWithFormat: @"%s", pattern_ptr]];
            }

            SDL_free(pattern);
        }

        if (!has_all_files) {
            if (@available(macOS 11.0, *)) {
                [dialog setAllowedContentTypes:types];
            } else {
                [dialog setAllowedFileTypes:types];
            }
        }
    }

    // Keep behavior consistent with other platforms
    [dialog setAllowsOtherFileTypes:YES];

    if (default_location) {
        [dialog setDirectoryURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:default_location]]];
    }

    NSWindow *w = NULL;

    if (window) {
        w = (__bridge NSWindow *)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, NULL);
    }

    if (w) {
        // [dialog beginWithCompletionHandler:^(NSInteger result) {
        [dialog beginSheetModalForWindow:w completionHandler:^(NSInteger result) {
            if (result == NSModalResponseOK) {
                if (dialog_as_open) {
                    NSArray* urls = [dialog_as_open URLs];
                    const char *files[[urls count] + 1];
                    for (int i = 0; i < [urls count]; i++) {
                        files[i] = [[[urls objectAtIndex:i] path] UTF8String];
                    }
                    files[[urls count]] = NULL;
                    callback(userdata, files, -1);
                } else {
                    const char *files[2] = { [[[dialog URL] path] UTF8String], NULL };
                    callback(userdata, files, -1);
                }
            } else if (result == NSModalResponseCancel) {
                const char *files[1] = { NULL };
                callback(userdata, files, -1);
            }
        }];
    } else {
        if ([dialog runModal] == NSModalResponseOK) {
            if (dialog_as_open) {
                NSArray* urls = [dialog_as_open URLs];
                const char *files[[urls count] + 1];
                for (int i = 0; i < [urls count]; i++) {
                    files[i] = [[[urls objectAtIndex:i] path] UTF8String];
                }
                files[[urls count]] = NULL;
                callback(userdata, files, -1);
            } else {
                const char *files[2] = { [[[dialog URL] path] UTF8String], NULL };
                callback(userdata, files, -1);
            }
        } else {
            const char *files[1] = { NULL };
            callback(userdata, files, -1);
        }
    }
}

#endif // SDL_PLATFORM_MACOS
