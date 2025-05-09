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

#ifdef SDL_VIDEO_DRIVER_X11

#include <limits.h> // For INT_MAX

#include "SDL_x11video.h"
#include "SDL_x11clipboard.h"
#include "../SDL_clipboard_c.h"
#include "../../events/SDL_events_c.h"

static const char *text_mime_types[] = {
    "UTF8_STRING",
    "text/plain;charset=utf-8",
    "text/plain",
    "TEXT",
    "STRING"
};

// Get any application owned window handle for clipboard association
Window GetWindow(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;

    /* We create an unmapped window that exists just to manage the clipboard,
       since X11 selection data is tied to a specific window and dies with it.
       We create the window on demand, so apps that don't use the clipboard
       don't have to keep an unnecessary resource around. */
    if (data->clipboard_window == None) {
        Display *dpy = data->display;
        Window parent = RootWindow(dpy, DefaultScreen(dpy));
        XSetWindowAttributes xattr;
        data->clipboard_window = X11_XCreateWindow(dpy, parent, -10, -10, 1, 1, 0,
                                                   CopyFromParent, InputOnly,
                                                   CopyFromParent, 0, &xattr);

        X11_XSelectInput(dpy, data->clipboard_window, PropertyChangeMask);
        X11_XFlush(data->display);
    }

    return data->clipboard_window;
}

static bool SetSelectionData(SDL_VideoDevice *_this, Atom selection, SDL_ClipboardDataCallback callback,
                            void *userdata, const char **mime_types, size_t mime_count, Uint32 sequence)
{
    SDL_VideoData *videodata = _this->internal;
    Display *display = videodata->display;
    Window window;
    SDLX11_ClipboardData *clipboard;
    bool clipboard_owner = false;

    window = GetWindow(_this);
    if (window == None) {
        return SDL_SetError("Couldn't find a window to own the selection");
    }

    if (selection == XA_PRIMARY) {
        clipboard = &videodata->primary_selection;
    } else {
        clipboard = &videodata->clipboard;
    }

    clipboard_owner = X11_XGetSelectionOwner(display, selection) == window;

    // If we are canceling our own data we need to clean it up
    if (clipboard_owner && clipboard->sequence == 0) {
        SDL_free(clipboard->userdata);
    }

    clipboard->callback = callback;
    clipboard->userdata = userdata;
    clipboard->mime_types = mime_types;
    clipboard->mime_count = mime_count;
    clipboard->sequence = sequence;

    X11_XSetSelectionOwner(display, selection, window, CurrentTime);
    return true;
}

static void *CloneDataBuffer(const void *buffer, const size_t len)
{
    void *clone = NULL;
    if (len > 0 && buffer) {
        clone = SDL_malloc(len + sizeof(Uint32));
        if (clone) {
            SDL_memcpy(clone, buffer, len);
            SDL_memset((Uint8 *)clone + len, 0, sizeof(Uint32));
        }
    }
    return clone;
}

/*
 * original_buffer is considered unusable after the function is called.
 */
static void *AppendDataBuffer(void *original_buffer, const size_t old_len, const void *buffer, const size_t buffer_len)
{
    void *resized_buffer;

    if (buffer_len > 0 && buffer) {
        resized_buffer = SDL_realloc(original_buffer, old_len + buffer_len + sizeof(Uint32));
        if (resized_buffer) {
            SDL_memcpy((Uint8 *)resized_buffer + old_len, buffer, buffer_len);
            SDL_memset((Uint8 *)resized_buffer + old_len + buffer_len, 0, sizeof(Uint32));
        }

        return resized_buffer;
    } else {
        return original_buffer;
    }
}

static bool WaitForSelection(SDL_VideoDevice *_this, Atom selection_type, bool *flag)
{
    Uint64 waitStart;
    Uint64 waitElapsed;

    waitStart = SDL_GetTicks();
    *flag = true;
    while (*flag) {
        SDL_PumpEvents();
        waitElapsed = SDL_GetTicks() - waitStart;
        // Wait one second for a selection response.
        if (waitElapsed > 1000) {
            *flag = false;
            SDL_SetError("Selection timeout");
            /* We need to set the selection text so that next time we won't
               timeout, otherwise we will hang on every call to this function. */
            SetSelectionData(_this, selection_type, SDL_ClipboardTextCallback, NULL,
                             text_mime_types, SDL_arraysize(text_mime_types), 0);
            return false;
        }
    }

    return true;
}

static void *GetSelectionData(SDL_VideoDevice *_this, Atom selection_type,
                              const char *mime_type, size_t *length)
{
    SDL_VideoData *videodata = _this->internal;
    Display *display = videodata->display;
    Window window;
    Window owner;
    Atom selection;
    Atom seln_type;
    int seln_format;
    unsigned long count;
    unsigned long overflow;

    SDLX11_ClipboardData *clipboard;
    void *data = NULL;
    unsigned char *src = NULL;
    bool incr_success = false;
    Atom XA_MIME = X11_XInternAtom(display, mime_type, False);

    *length = 0;

    // Get the window that holds the selection
    window = GetWindow(_this);
    owner = X11_XGetSelectionOwner(display, selection_type);
    if (owner == None) {
        // This requires a fallback to ancient X10 cut-buffers. We will just skip those for now
        data = NULL;
    } else if (owner == window) {
        owner = DefaultRootWindow(display);
        if (selection_type == XA_PRIMARY) {
            clipboard = &videodata->primary_selection;
        } else {
            clipboard = &videodata->clipboard;
        }

        if (clipboard->callback) {
            const void *clipboard_data = clipboard->callback(clipboard->userdata, mime_type, length);
            data = CloneDataBuffer(clipboard_data, *length);
        }
    } else {
        // Request that the selection owner copy the data to our window
        owner = window;
        selection = videodata->atoms.SDL_SELECTION;
        X11_XConvertSelection(display, selection_type, XA_MIME, selection, owner,
                              CurrentTime);

        if (WaitForSelection(_this, selection_type, &videodata->selection_waiting) == false) {
            data = NULL;
            *length = 0;
        }

        if (X11_XGetWindowProperty(display, owner, selection, 0, INT_MAX / 4, False,
                                   XA_MIME, &seln_type, &seln_format, &count, &overflow, &src) == Success) {
            if (seln_type == XA_MIME) {
                *length = (size_t)count;
                data = CloneDataBuffer(src, count);
            } else if (seln_type == videodata->atoms.INCR) {
                while (1) {
                    // Only delete the property after being done with the previous "chunk".
                    X11_XDeleteProperty(display, owner, selection);
                    X11_XFlush(display);

                    if (WaitForSelection(_this, selection_type, &videodata->selection_incr_waiting) == false) {
                        break;
                    }

                    X11_XFree(src);
                    if (X11_XGetWindowProperty(display, owner, selection, 0, INT_MAX / 4, False,
                                           XA_MIME, &seln_type, &seln_format, &count, &overflow, &src) != Success) {
                        break;
                    }

                    if (count == 0) {
                        incr_success = true;
                        break;
                    }

                    if (*length == 0) {
                        *length = (size_t)count;
                        data = CloneDataBuffer(src, count);
                    } else {
                        data = AppendDataBuffer(data, *length, src, count);
                        *length += (size_t)count;
                    }

                    if (data == NULL) {
                        break;
                    }
                }

                if (incr_success == false) {
                    SDL_free(data);
                    data = 0;
                    *length = 0;
                }
            }
            X11_XFree(src);
        }
    }
    return data;
}

const char **X11_GetTextMimeTypes(SDL_VideoDevice *_this, size_t *num_mime_types)
{
    *num_mime_types = SDL_arraysize(text_mime_types);
    return text_mime_types;
}

bool X11_SetClipboardData(SDL_VideoDevice *_this)
{
    SDL_VideoData *videodata = _this->internal;
    return SetSelectionData(_this, videodata->atoms.CLIPBOARD, _this->clipboard_callback, _this->clipboard_userdata, (const char **)_this->clipboard_mime_types, _this->num_clipboard_mime_types, _this->clipboard_sequence);
}

void *X11_GetClipboardData(SDL_VideoDevice *_this, const char *mime_type, size_t *length)
{
    SDL_VideoData *videodata = _this->internal;
    return GetSelectionData(_this, videodata->atoms.CLIPBOARD, mime_type, length);
}

bool X11_HasClipboardData(SDL_VideoDevice *_this, const char *mime_type)
{
    size_t length;
    void *data;
    data = X11_GetClipboardData(_this, mime_type, &length);
    if (data) {
        SDL_free(data);
    }
    return length > 0;
}

bool X11_SetPrimarySelectionText(SDL_VideoDevice *_this, const char *text)
{
    return SetSelectionData(_this, XA_PRIMARY, SDL_ClipboardTextCallback, SDL_strdup(text), text_mime_types, SDL_arraysize(text_mime_types), 0);
}

char *X11_GetPrimarySelectionText(SDL_VideoDevice *_this)
{
    size_t length;
    char *text = GetSelectionData(_this, XA_PRIMARY, text_mime_types[0], &length);
    if (!text) {
        text = SDL_strdup("");
    }
    return text;
}

bool X11_HasPrimarySelectionText(SDL_VideoDevice *_this)
{
    bool result = false;
    char *text = X11_GetPrimarySelectionText(_this);
    if (text) {
        if (text[0] != '\0') {
            result = true;
        }
        SDL_free(text);
    }
    return result;
}

void X11_QuitClipboard(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;
    if (data->primary_selection.sequence == 0) {
        SDL_free(data->primary_selection.userdata);
    }
    if (data->clipboard.sequence == 0) {
        SDL_free(data->clipboard.userdata);
    }
}

#endif // SDL_VIDEO_DRIVER_X11
