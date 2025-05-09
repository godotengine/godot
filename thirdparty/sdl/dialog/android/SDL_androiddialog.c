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
#include "../../core/android/SDL_android.h"

void SDL_SYS_ShowFileDialogWithProperties(SDL_FileDialogType type, SDL_DialogFileCallback callback, void *userdata, SDL_PropertiesID props)
{
    SDL_DialogFileFilter *filters = SDL_GetPointerProperty(props, SDL_PROP_FILE_DIALOG_FILTERS_POINTER, NULL);
    int nfilters = (int) SDL_GetNumberProperty(props, SDL_PROP_FILE_DIALOG_NFILTERS_NUMBER, 0);
    bool allow_many = SDL_GetBooleanProperty(props, SDL_PROP_FILE_DIALOG_MANY_BOOLEAN, false);
    bool is_save;

    if (SDL_GetHint(SDL_HINT_FILE_DIALOG_DRIVER) != NULL) {
        SDL_SetError("File dialog driver unsupported (don't set SDL_HINT_FILE_DIALOG_DRIVER)");
        callback(userdata, NULL, -1);
        return;
    }

    switch (type) {
    case SDL_FILEDIALOG_OPENFILE:
        is_save = false;
        break;

    case SDL_FILEDIALOG_SAVEFILE:
        is_save = true;
        break;

    case SDL_FILEDIALOG_OPENFOLDER:
        SDL_Unsupported();
        callback(userdata, NULL, -1);
        return;
    };

    if (!Android_JNI_OpenFileDialog(callback, userdata, filters, nfilters, is_save, allow_many)) {
        // SDL_SetError is already called when it fails
        callback(userdata, NULL, -1);
    }
}
