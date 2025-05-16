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

#include "SDL_clipboard_c.h"
#include "SDL_sysvideo.h"
#include "../events/SDL_events_c.h"
#include "../events/SDL_clipboardevents_c.h"

void SDL_FreeClipboardMimeTypes(SDL_VideoDevice *_this)
{
    if (_this->clipboard_mime_types) {
        for (size_t i = 0; i < _this->num_clipboard_mime_types; ++i) {
            SDL_free(_this->clipboard_mime_types[i]);
        }
        SDL_free(_this->clipboard_mime_types);
        _this->clipboard_mime_types = NULL;
        _this->num_clipboard_mime_types = 0;
    }
}


void SDL_CancelClipboardData(Uint32 sequence)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

    if (sequence && sequence != _this->clipboard_sequence) {
        // This clipboard data was already canceled
        return;
    }

    if (_this->clipboard_cleanup) {
        _this->clipboard_cleanup(_this->clipboard_userdata);
    }

    SDL_FreeClipboardMimeTypes(_this);

    _this->clipboard_callback = NULL;
    _this->clipboard_cleanup = NULL;
    _this->clipboard_userdata = NULL;
}

bool SDL_SaveClipboardMimeTypes(const char **mime_types, size_t num_mime_types)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

    SDL_FreeClipboardMimeTypes(_this);

    if (mime_types && num_mime_types > 0) {
        size_t num_allocated = 0;

        _this->clipboard_mime_types = (char **)SDL_malloc(num_mime_types * sizeof(char *));
        if (_this->clipboard_mime_types) {
            for (size_t i = 0; i < num_mime_types; ++i) {
                _this->clipboard_mime_types[i] = SDL_strdup(mime_types[i]);
                if (_this->clipboard_mime_types[i]) {
                    ++num_allocated;
                }
            }
        }
        if (num_allocated < num_mime_types) {
            SDL_FreeClipboardMimeTypes(_this);
            return false;
        }
        _this->num_clipboard_mime_types = num_mime_types;
    }
    return true;
}

bool SDL_SetClipboardData(SDL_ClipboardDataCallback callback, SDL_ClipboardCleanupCallback cleanup, void *userdata, const char **mime_types, size_t num_mime_types)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

    if (!_this) {
        return SDL_UninitializedVideo();
    }

    // Parameter validation
    if (!((callback && mime_types && num_mime_types > 0) ||
          (!callback && !mime_types && num_mime_types == 0))) {
        return SDL_SetError("Invalid parameters");
    }

    SDL_CancelClipboardData(0);

    ++_this->clipboard_sequence;
    if (!_this->clipboard_sequence) {
        _this->clipboard_sequence = 1;
    }
    _this->clipboard_callback = callback;
    _this->clipboard_cleanup = cleanup;
    _this->clipboard_userdata = userdata;

    if (!SDL_SaveClipboardMimeTypes(mime_types, num_mime_types)) {
        SDL_ClearClipboardData();
        return false;
    }

    if (_this->SetClipboardData) {
        if (!_this->SetClipboardData(_this)) {
            return false;
        }
    } else if (_this->SetClipboardText) {
        char *text = NULL;
        size_t size;

        for (size_t i = 0; i < num_mime_types; ++i) {
            const char *mime_type = _this->clipboard_mime_types[i];
            if (SDL_IsTextMimeType(mime_type)) {
                const void *data = _this->clipboard_callback(_this->clipboard_userdata, mime_type, &size);
                if (data) {
                    text = (char *)SDL_malloc(size + 1);
                    SDL_memcpy(text, data, size);
                    text[size] = '\0';
                    if (!_this->SetClipboardText(_this, text)) {
                        SDL_free(text);
                        return false;
                    }
                    break;
                }
            }
        }
        if (text) {
            SDL_free(text);
        } else {
            if (!_this->SetClipboardText(_this, "")) {
                return false;
            }
        }
    }

    char **mime_types_copy = SDL_CopyClipboardMimeTypes(mime_types, num_mime_types, true);
    if (!mime_types_copy)
        return SDL_SetError("unable to copy current mime types");

    SDL_SendClipboardUpdate(true, mime_types_copy, num_mime_types);
    return true;
}

bool SDL_ClearClipboardData(void)
{
    return SDL_SetClipboardData(NULL, NULL, NULL, NULL, 0);
}

void *SDL_GetInternalClipboardData(SDL_VideoDevice *_this, const char *mime_type, size_t *size)
{
    void *data = NULL;

    if (_this->clipboard_callback) {
        const void *provided_data = _this->clipboard_callback(_this->clipboard_userdata, mime_type, size);
        if (provided_data) {
            // Make a copy of it for the caller and guarantee null termination
            data = SDL_malloc(*size + sizeof(Uint32));
            if (data) {
                SDL_memcpy(data, provided_data, *size);
                SDL_memset((Uint8 *)data + *size, 0, sizeof(Uint32));
            }
        }
    }
    return data;
}

void *SDL_GetClipboardData(const char *mime_type, size_t *size)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    size_t unused;

    if (!_this) {
        SDL_UninitializedVideo();
        return NULL;
    }

    if (!mime_type) {
        SDL_InvalidParamError("mime_type");
        return NULL;
    }
    if (!size) {
        size = &unused;
    }

    // Initialize size to empty, so implementations don't have to worry about it
    *size = 0;

    if (_this->GetClipboardData) {
        return _this->GetClipboardData(_this, mime_type, size);
    } else if (_this->GetClipboardText && SDL_IsTextMimeType(mime_type)) {
        char *text = _this->GetClipboardText(_this);
        if (text) {
            if (*text == '\0') {
                SDL_free(text);
                text = NULL;
            } else {
                *size = SDL_strlen(text);
            }
        }
        return text;
    } else {
        return SDL_GetInternalClipboardData(_this, mime_type, size);
    }
}

bool SDL_HasInternalClipboardData(SDL_VideoDevice *_this, const char *mime_type)
{
    size_t i;

    for (i = 0; i < _this->num_clipboard_mime_types; ++i) {
        if (SDL_strcmp(mime_type, _this->clipboard_mime_types[i]) == 0) {
            return true;
        }
    }
    return false;
}

bool SDL_HasClipboardData(const char *mime_type)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

    if (!_this) {
        SDL_UninitializedVideo();
        return false;
    }

    if (!mime_type) {
        SDL_InvalidParamError("mime_type");
        return false;
    }

    if (_this->HasClipboardData) {
        return _this->HasClipboardData(_this, mime_type);
    } else if (_this->HasClipboardText && SDL_IsTextMimeType(mime_type)) {
        return _this->HasClipboardText(_this);
    } else {
        return SDL_HasInternalClipboardData(_this, mime_type);
    }
}

char **SDL_CopyClipboardMimeTypes(const char **clipboard_mime_types, size_t num_mime_types, bool temporary)
{
    size_t allocSize = sizeof(char *);
    for (size_t i = 0; i < num_mime_types; i++) {
        allocSize += sizeof(char *) + SDL_strlen(clipboard_mime_types[i]) + 1;
    }

    char *ret;
    if (temporary)
        ret = (char *)SDL_AllocateTemporaryMemory(allocSize);
    else
        ret = (char *)SDL_malloc(allocSize);
    if (!ret) {
        return NULL;
    }

    char **result = (char **)ret;
    ret += sizeof(char *) * (num_mime_types + 1);

    for (size_t i = 0; i < num_mime_types; i++) {
        result[i] = ret;

        const char *mime_type = clipboard_mime_types[i];
        // Copy the whole string including the terminating null char
        char c;
        do {
            c = *ret++ = *mime_type++;
        } while (c != '\0');
    }
    result[num_mime_types] = NULL;

    return result;

}

char **SDL_GetClipboardMimeTypes(size_t *num_mime_types)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

    if (num_mime_types) {
        *num_mime_types = 0;
    }

    if (!_this) {
        SDL_UninitializedVideo();
        return NULL;
    }

    if (num_mime_types) {
        *num_mime_types = _this->num_clipboard_mime_types;
    }
    return SDL_CopyClipboardMimeTypes((const char **)_this->clipboard_mime_types, _this->num_clipboard_mime_types, false);
}

// Clipboard text

bool SDL_IsTextMimeType(const char *mime_type)
{
    return (SDL_strncmp(mime_type, "text", 4) == 0);
}

static const char **SDL_GetTextMimeTypes(SDL_VideoDevice *_this, size_t *num_mime_types)
{
    if (_this->GetTextMimeTypes) {
        return _this->GetTextMimeTypes(_this, num_mime_types);
    } else {
        static const char *text_mime_types[] = {
            "text/plain;charset=utf-8"
        };

        *num_mime_types = SDL_arraysize(text_mime_types);
        return text_mime_types;
    }
}

const void * SDLCALL SDL_ClipboardTextCallback(void *userdata, const char *mime_type, size_t *size)
{
    char *text = (char *)userdata;
    if (text) {
        *size = SDL_strlen(text);
    } else {
        *size = 0;
    }
    return text;
}

bool SDL_SetClipboardText(const char *text)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    size_t num_mime_types;
    const char **text_mime_types;

    if (!_this) {
        return SDL_UninitializedVideo();
    }

    if (text && *text) {
        text_mime_types = SDL_GetTextMimeTypes(_this, &num_mime_types);

        return SDL_SetClipboardData(SDL_ClipboardTextCallback, SDL_free, SDL_strdup(text), text_mime_types, num_mime_types);
    }
    return SDL_ClearClipboardData();
}

char *SDL_GetClipboardText(void)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    size_t i, num_mime_types;
    const char **text_mime_types;
    size_t length;
    char *text = NULL;

    if (!_this) {
        SDL_UninitializedVideo();
        return SDL_strdup("");
    }

    text_mime_types = SDL_GetTextMimeTypes(_this, &num_mime_types);
    for (i = 0; i < num_mime_types; ++i) {
        void *clipdata = SDL_GetClipboardData(text_mime_types[i], &length);
        if (clipdata) {
            text = (char *)clipdata;
            break;
        }
    }

    if (!text) {
        text = SDL_strdup("");
    }
    return text;
}

bool SDL_HasClipboardText(void)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    size_t i, num_mime_types;
    const char **text_mime_types;

    if (!_this) {
        return SDL_UninitializedVideo();
    }

    text_mime_types = SDL_GetTextMimeTypes(_this, &num_mime_types);
    for (i = 0; i < num_mime_types; ++i) {
        if (SDL_HasClipboardData(text_mime_types[i])) {
            return true;
        }
    }
    return false;
}

// Primary selection text

bool SDL_SetPrimarySelectionText(const char *text)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

    if (!_this) {
        return SDL_UninitializedVideo();
    }

    if (!text) {
        text = "";
    }
    if (_this->SetPrimarySelectionText) {
        if (!_this->SetPrimarySelectionText(_this, text)) {
            return false;
        }
    } else {
        SDL_free(_this->primary_selection_text);
        _this->primary_selection_text = SDL_strdup(text);
    }

    char **mime_types = SDL_CopyClipboardMimeTypes((const char **)_this->clipboard_mime_types, _this->num_clipboard_mime_types, true);
    if (!mime_types)
        return SDL_SetError("unable to copy current mime types");

    SDL_SendClipboardUpdate(true, mime_types, _this->num_clipboard_mime_types);
    return true;
}

char *SDL_GetPrimarySelectionText(void)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

    if (!_this) {
        SDL_UninitializedVideo();
        return SDL_strdup("");
    }

    if (_this->GetPrimarySelectionText) {
        return _this->GetPrimarySelectionText(_this);
    } else {
        const char *text = _this->primary_selection_text;
        if (!text) {
            text = "";
        }
        return SDL_strdup(text);
    }
}

bool SDL_HasPrimarySelectionText(void)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

    if (!_this) {
        return SDL_UninitializedVideo();
    }

    if (_this->HasPrimarySelectionText) {
        return _this->HasPrimarySelectionText(_this);
    } else {
        if (_this->primary_selection_text && _this->primary_selection_text[0] != '\0') {
            return true;
        } else {
            return false;
        }
    }
}

