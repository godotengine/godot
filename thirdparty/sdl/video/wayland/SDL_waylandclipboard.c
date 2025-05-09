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

#ifdef SDL_VIDEO_DRIVER_WAYLAND

#include "SDL_waylanddatamanager.h"
#include "SDL_waylandevents_c.h"
#include "SDL_waylandclipboard.h"
#include "../SDL_clipboard_c.h"
#include "../../events/SDL_events_c.h"


bool Wayland_SetClipboardData(SDL_VideoDevice *_this)
{
    SDL_VideoData *video_data = _this->internal;
    SDL_WaylandSeat *seat = video_data->last_implicit_grab_seat;
    bool result = false;

    // If no implicit grab is available yet, just attach it to the first available seat.
    if (!seat && !WAYLAND_wl_list_empty(&video_data->seat_list)) {
        seat = wl_container_of(video_data->seat_list.next, seat, link);
    }

    if (seat && seat->data_device) {
        SDL_WaylandDataDevice *data_device = seat->data_device;

        if (_this->clipboard_callback && _this->clipboard_mime_types) {
            SDL_WaylandDataSource *source = Wayland_data_source_create(_this);
            Wayland_data_source_set_callback(source, _this->clipboard_callback, _this->clipboard_userdata, _this->clipboard_sequence);

            result = Wayland_data_device_set_selection(data_device, source, (const char **)_this->clipboard_mime_types, _this->num_clipboard_mime_types);
            if (!result) {
                Wayland_data_source_destroy(source);
            }
        } else {
            result = Wayland_data_device_clear_selection(data_device);
        }
    }

    return result;
}

void *Wayland_GetClipboardData(SDL_VideoDevice *_this, const char *mime_type, size_t *length)
{
    SDL_VideoData *video_data = _this->internal;
    SDL_WaylandSeat *seat = video_data->last_incoming_data_offer_seat;
    void *buffer = NULL;

    if (seat && seat->data_device) {
        SDL_WaylandDataDevice *data_device = seat->data_device;
        if (data_device->selection_source) {
            buffer = SDL_GetInternalClipboardData(_this, mime_type, length);
        } else if (Wayland_data_offer_has_mime(data_device->selection_offer, mime_type)) {
            buffer = Wayland_data_offer_receive(data_device->selection_offer, mime_type, length);
        }
    }

    return buffer;
}

bool Wayland_HasClipboardData(SDL_VideoDevice *_this, const char *mime_type)
{
    SDL_VideoData *video_data = _this->internal;
    SDL_WaylandSeat *seat = video_data->last_incoming_data_offer_seat;
    bool result = false;

    if (seat && seat->data_device) {
        SDL_WaylandDataDevice *data_device = seat->data_device;
        if (data_device->selection_source) {
            result = SDL_HasInternalClipboardData(_this, mime_type);
        } else {
            result = Wayland_data_offer_has_mime(data_device->selection_offer, mime_type);
        }
    }
    return result;
}

static const char *text_mime_types[] = {
    TEXT_MIME,
    "text/plain",
    "TEXT",
    "UTF8_STRING",
    "STRING"
};

const char **Wayland_GetTextMimeTypes(SDL_VideoDevice *_this, size_t *num_mime_types)
{
    *num_mime_types = SDL_arraysize(text_mime_types);
    return text_mime_types;
}

bool Wayland_SetPrimarySelectionText(SDL_VideoDevice *_this, const char *text)
{
    SDL_VideoData *video_data = _this->internal;
    SDL_WaylandSeat *seat = video_data->last_implicit_grab_seat;
    bool result;

    // If no implicit grab is available yet, just attach it to the first available seat.
    if (!seat && !WAYLAND_wl_list_empty(&video_data->seat_list)) {
        seat = wl_container_of(video_data->seat_list.next, seat, link);
    }

    if (seat && seat->primary_selection_device) {
        SDL_WaylandPrimarySelectionDevice *primary_selection_device = seat->primary_selection_device;
        if (text[0] != '\0') {
            SDL_WaylandPrimarySelectionSource *source = Wayland_primary_selection_source_create(_this);
            Wayland_primary_selection_source_set_callback(source, SDL_ClipboardTextCallback, SDL_strdup(text));

            result = Wayland_primary_selection_device_set_selection(primary_selection_device,
                                                                    source,
                                                                    text_mime_types,
                                                                    SDL_arraysize(text_mime_types));
            if (!result) {
                Wayland_primary_selection_source_destroy(source);
            }
        } else {
            result = Wayland_primary_selection_device_clear_selection(primary_selection_device);
        }
    } else {
        result = SDL_SetError("Primary selection not supported");
    }
    return result;
}

char *Wayland_GetPrimarySelectionText(SDL_VideoDevice *_this)
{
    SDL_VideoData *video_data = _this->internal;
    SDL_WaylandSeat *seat = video_data->last_incoming_primary_selection_seat;
    char *text = NULL;
    size_t length = 0;

    if (seat && seat->primary_selection_device) {
        SDL_WaylandPrimarySelectionDevice *primary_selection_device = seat->primary_selection_device;
        if (primary_selection_device->selection_source) {
            text = Wayland_primary_selection_source_get_data(primary_selection_device->selection_source, TEXT_MIME, &length);
        } else {
            for (size_t i = 0; i < SDL_arraysize(text_mime_types); i++) {
                if (Wayland_primary_selection_offer_has_mime(primary_selection_device->selection_offer, text_mime_types[i])) {
                    text = Wayland_primary_selection_offer_receive(primary_selection_device->selection_offer, text_mime_types[i], &length);
                    break;
                }
            }
        }
    }

    if (!text) {
        text = SDL_strdup("");
    }

    return text;
}

bool Wayland_HasPrimarySelectionText(SDL_VideoDevice *_this)
{
    SDL_VideoData *video_data = _this->internal;
    SDL_WaylandSeat *seat = video_data->last_incoming_primary_selection_seat;
    bool result = false;

    if (seat && seat->primary_selection_device) {
        SDL_WaylandPrimarySelectionDevice *primary_selection_device = seat->primary_selection_device;
        if (primary_selection_device->selection_source) {
            result = true;
        } else {
            size_t mime_count = 0;
            const char **mime_types = Wayland_GetTextMimeTypes(_this, &mime_count);
            for (size_t i = 0; i < mime_count; i++) {
                if (Wayland_primary_selection_offer_has_mime(primary_selection_device->selection_offer, mime_types[i])) {
                    result = true;
                    break;
                }
            }
        }
    }
    return result;
}

#endif // SDL_VIDEO_DRIVER_WAYLAND
