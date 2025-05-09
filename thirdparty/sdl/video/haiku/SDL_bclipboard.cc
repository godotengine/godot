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

#ifdef SDL_VIDEO_DRIVER_HAIKU

// BWindow based clipboard implementation

#include <unistd.h>
#include <TypeConstants.h>

#include "SDL_BWin.h"
#include "../SDL_sysvideo.h"

#ifdef __cplusplus
extern "C" {
#endif

bool HAIKU_SetClipboardText(SDL_VideoDevice *_this, const char *text)
{
    BMessage *clip = NULL;
    if (be_clipboard->Lock()) {
        be_clipboard->Clear();
        if ((clip = be_clipboard->Data())) {
            // Presumably the string of characters is ascii-format
            ssize_t asciiLength = 0;
            for (; text[asciiLength] != 0; ++asciiLength) {}
            clip->AddData("text/plain", B_MIME_TYPE, text, asciiLength);
            be_clipboard->Commit();
        }
        be_clipboard->Unlock();
    }
    return true;
}

char *HAIKU_GetClipboardText(SDL_VideoDevice *_this) {
    BMessage *clip = NULL;
    const char *text = NULL;
    ssize_t length;
    char *result;
    if (be_clipboard->Lock()) {
        if ((clip = be_clipboard->Data())) {
            // Presumably the string of characters is ascii-format
            clip->FindData("text/plain", B_MIME_TYPE, (const void**)&text,
                &length);
        }
        be_clipboard->Unlock();
    }

    if (!text) {
        result = SDL_strdup("");
    } else {
        // Copy the data and pass on to SDL
        result = (char *)SDL_malloc((length + 1) * sizeof(char));
        SDL_strlcpy(result, text, length + 1);
    }

    return result;
}

bool HAIKU_HasClipboardText(SDL_VideoDevice *_this) {
    bool result = false;
    char *text = HAIKU_GetClipboardText(_this);
    if (text) {
        result = (text[0] != '\0');
        SDL_free(text);
    }
    return result;
}

#ifdef __cplusplus
}
#endif

#endif // SDL_VIDEO_DRIVER_HAIKU
