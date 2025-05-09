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

#include "SDL_dialog.h"
#include "SDL_dialog_utils.h"

void SDL_ShowFileDialogWithProperties(SDL_FileDialogType type, SDL_DialogFileCallback callback, void *userdata, SDL_PropertiesID props)
{
    if (!callback) {
        return;
    }
#ifdef SDL_DIALOG_DISABLED
    SDL_SetError("SDL not built with dialog support");
    callback(userdata, NULL, -1);
#else
    SDL_DialogFileFilter *filters = SDL_GetPointerProperty(props, SDL_PROP_FILE_DIALOG_FILTERS_POINTER, NULL);
    int nfilters = (int) SDL_GetNumberProperty(props, SDL_PROP_FILE_DIALOG_NFILTERS_NUMBER, -1);

    if (filters && nfilters == -1) {
        SDL_SetError("Set filter pointers, but didn't set number of filters (SDL_PROP_FILE_DIALOG_NFILTERS_NUMBER)");
        callback(userdata, NULL, -1);
        return;
    }

    const char *msg = validate_filters(filters, nfilters);

    if (msg) {
        SDL_SetError("Invalid dialog file filters: %s", msg);
        callback(userdata, NULL, -1);
        return;
    }

    switch (type) {
    case SDL_FILEDIALOG_OPENFILE:
    case SDL_FILEDIALOG_SAVEFILE:
    case SDL_FILEDIALOG_OPENFOLDER:
        SDL_SYS_ShowFileDialogWithProperties(type, callback, userdata, props);
        break;

    default:
        SDL_SetError("Unsupported file dialog type: %d", (int) type);
        callback(userdata, NULL, -1);
        break;
    };
#endif
}

void SDL_ShowOpenFileDialog(SDL_DialogFileCallback callback, void *userdata, SDL_Window *window, const SDL_DialogFileFilter *filters, int nfilters, const char *default_location, bool allow_many)
{
#ifdef SDL_DIALOG_DISABLED
    if (!callback) {
        return;
    }
    SDL_SetError("SDL not built with dialog support");
    callback(userdata, NULL, -1);
#else
    SDL_PropertiesID props = SDL_CreateProperties();

    SDL_SetPointerProperty(props, SDL_PROP_FILE_DIALOG_FILTERS_POINTER, (void *) filters);
    SDL_SetNumberProperty(props, SDL_PROP_FILE_DIALOG_NFILTERS_NUMBER, nfilters);
    SDL_SetPointerProperty(props, SDL_PROP_FILE_DIALOG_WINDOW_POINTER, window);
    SDL_SetStringProperty(props, SDL_PROP_FILE_DIALOG_LOCATION_STRING, default_location);
    SDL_SetBooleanProperty(props, SDL_PROP_FILE_DIALOG_MANY_BOOLEAN, allow_many);

    SDL_ShowFileDialogWithProperties(SDL_FILEDIALOG_OPENFILE, callback, userdata, props);

    SDL_DestroyProperties(props);
#endif
}

void SDL_ShowSaveFileDialog(SDL_DialogFileCallback callback, void *userdata, SDL_Window *window, const SDL_DialogFileFilter *filters, int nfilters, const char *default_location)
{
#ifdef SDL_DIALOG_DISABLED
    if (!callback) {
        return;
    }
    SDL_SetError("SDL not built with dialog support");
    callback(userdata, NULL, -1);
#else
    SDL_PropertiesID props = SDL_CreateProperties();

    SDL_SetPointerProperty(props, SDL_PROP_FILE_DIALOG_FILTERS_POINTER, (void *) filters);
    SDL_SetNumberProperty(props, SDL_PROP_FILE_DIALOG_NFILTERS_NUMBER, nfilters);
    SDL_SetPointerProperty(props, SDL_PROP_FILE_DIALOG_WINDOW_POINTER, window);
    SDL_SetStringProperty(props, SDL_PROP_FILE_DIALOG_LOCATION_STRING, default_location);

    SDL_ShowFileDialogWithProperties(SDL_FILEDIALOG_SAVEFILE, callback, userdata, props);

    SDL_DestroyProperties(props);
#endif
}

void SDL_ShowOpenFolderDialog(SDL_DialogFileCallback callback, void *userdata, SDL_Window *window, const char *default_location, bool allow_many)
{
#ifdef SDL_DIALOG_DISABLED
    if (!callback) {
        return;
    }
    SDL_SetError("SDL not built with dialog support");
    callback(userdata, NULL, -1);
#else
    SDL_PropertiesID props = SDL_CreateProperties();

    SDL_SetPointerProperty(props, SDL_PROP_FILE_DIALOG_WINDOW_POINTER, window);
    SDL_SetStringProperty(props, SDL_PROP_FILE_DIALOG_LOCATION_STRING, default_location);
    SDL_SetBooleanProperty(props, SDL_PROP_FILE_DIALOG_MANY_BOOLEAN, allow_many);

    SDL_ShowFileDialogWithProperties(SDL_FILEDIALOG_OPENFOLDER, callback, userdata, props);

    SDL_DestroyProperties(props);
#endif
}
