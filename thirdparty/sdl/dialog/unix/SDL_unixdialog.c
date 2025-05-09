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
#include "./SDL_portaldialog.h"
#include "./SDL_zenitydialog.h"

static void (*detected_function)(SDL_FileDialogType type, SDL_DialogFileCallback callback, void *userdata, SDL_PropertiesID props) = NULL;

void SDLCALL hint_callback(void *userdata, const char *name, const char *oldValue, const char *newValue);

static void set_callback(void)
{
    static bool is_set = false;

    if (is_set == false) {
        is_set = true;
        SDL_AddHintCallback(SDL_HINT_FILE_DIALOG_DRIVER, hint_callback, NULL);
    }
}

// Returns non-zero on success, 0 on failure
static int detect_available_methods(const char *value)
{
    const char *driver = value ? value : SDL_GetHint(SDL_HINT_FILE_DIALOG_DRIVER);

    set_callback();

    if (driver == NULL || SDL_strcmp(driver, "portal") == 0) {
        if (SDL_Portal_detect()) {
            detected_function = SDL_Portal_ShowFileDialogWithProperties;
            return 1;
        }
    }

    if (driver == NULL || SDL_strcmp(driver, "zenity") == 0) {
        if (SDL_Zenity_detect()) {
            detected_function = SDL_Zenity_ShowFileDialogWithProperties;
            return 2;
        }
    }

    SDL_SetError("File dialog driver unsupported (supported values for SDL_HINT_FILE_DIALOG_DRIVER are 'zenity' and 'portal')");
    return 0;
}

void SDLCALL hint_callback(void *userdata, const char *name, const char *oldValue, const char *newValue)
{
    detect_available_methods(newValue);
}

void SDL_SYS_ShowFileDialogWithProperties(SDL_FileDialogType type, SDL_DialogFileCallback callback, void *userdata, SDL_PropertiesID props)
{
    // Call detect_available_methods() again each time in case the situation changed
    if (!detected_function && !detect_available_methods(NULL)) {
        // SetError() done by detect_available_methods()
        callback(userdata, NULL, -1);
        return;
    }

    detected_function(type, callback, userdata, props);
}
