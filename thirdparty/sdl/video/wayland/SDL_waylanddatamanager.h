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
#include "SDL_waylandevents_c.h"

#ifndef SDL_waylanddatamanager_h_
#define SDL_waylanddatamanager_h_

#include "SDL_waylandvideo.h"
#include "SDL_waylandwindow.h"

#define TEXT_MIME "text/plain;charset=utf-8"
#define FILE_MIME "text/uri-list"
#define FILE_PORTAL_MIME "application/vnd.portal.filetransfer"
#define SDL_DATA_ORIGIN_MIME "application/x-sdl3-source-id"

typedef struct SDL_WaylandDataDevice SDL_WaylandDataDevice;
typedef struct SDL_WaylandPrimarySelectionDevice SDL_WaylandPrimarySelectionDevice;

typedef struct
{
    char *mime_type;
    void *data;
    size_t length;
    struct wl_list link;
} SDL_MimeDataList;

typedef struct SDL_WaylandUserdata
{
    Uint32 sequence;
    void *data;
} SDL_WaylandUserdata;

typedef struct
{
    struct wl_data_source *source;
    SDL_WaylandDataDevice *data_device;
    SDL_ClipboardDataCallback callback;
    SDL_WaylandUserdata userdata;
} SDL_WaylandDataSource;

typedef struct
{
    struct zwp_primary_selection_source_v1 *source;
    SDL_WaylandDataDevice *data_device;
    SDL_WaylandPrimarySelectionDevice *primary_selection_device;
    SDL_ClipboardDataCallback callback;
    SDL_WaylandUserdata userdata;
} SDL_WaylandPrimarySelectionSource;

typedef struct
{
    struct wl_data_offer *offer;
    struct wl_list mimes;
    SDL_WaylandDataDevice *data_device;

    // Callback data for queued receive.
    struct wl_callback *callback;
    int read_fd;
} SDL_WaylandDataOffer;

typedef struct
{
    struct zwp_primary_selection_offer_v1 *offer;
    struct wl_list mimes;
    SDL_WaylandPrimarySelectionDevice *primary_selection_device;
} SDL_WaylandPrimarySelectionOffer;

struct SDL_WaylandDataDevice
{
    struct wl_data_device *data_device;
    struct SDL_WaylandSeat *seat;
    char *id_str;

    // Drag and Drop
    uint32_t drag_serial;
    SDL_WaylandDataOffer *drag_offer;
    SDL_WaylandDataOffer *selection_offer;
    const char *mime_type;
    bool has_mime_file, has_mime_text;
    SDL_Window *dnd_window;

    // Clipboard and Primary Selection
    uint32_t selection_serial;
    SDL_WaylandDataSource *selection_source;
};

struct SDL_WaylandPrimarySelectionDevice
{
    struct zwp_primary_selection_device_v1 *primary_selection_device;
    struct SDL_WaylandSeat *seat;

    uint32_t selection_serial;
    SDL_WaylandPrimarySelectionSource *selection_source;
    SDL_WaylandPrimarySelectionOffer *selection_offer;
};

// Wayland Data Source / Primary Selection Source - (Sending)
extern SDL_WaylandDataSource *Wayland_data_source_create(SDL_VideoDevice *_this);
extern SDL_WaylandPrimarySelectionSource *Wayland_primary_selection_source_create(SDL_VideoDevice *_this);
extern ssize_t Wayland_data_source_send(SDL_WaylandDataSource *source,
                                        const char *mime_type, int fd);
extern ssize_t Wayland_primary_selection_source_send(SDL_WaylandPrimarySelectionSource *source,
                                                     const char *mime_type, int fd);
extern void Wayland_data_source_set_callback(SDL_WaylandDataSource *source,
                                            SDL_ClipboardDataCallback callback,
                                            void *userdata,
                                            Uint32 sequence);
extern void Wayland_primary_selection_source_set_callback(SDL_WaylandPrimarySelectionSource *source,
                                                          SDL_ClipboardDataCallback callback,
                                                          void *userdata);
extern void *Wayland_data_source_get_data(SDL_WaylandDataSource *source,
                                          const char *mime_type,
                                          size_t *length);
extern void *Wayland_primary_selection_source_get_data(SDL_WaylandPrimarySelectionSource *source,
                                                       const char *mime_type,
                                                       size_t *length);
extern void Wayland_data_source_destroy(SDL_WaylandDataSource *source);
extern void Wayland_primary_selection_source_destroy(SDL_WaylandPrimarySelectionSource *source);

// Wayland Data / Primary Selection Offer - (Receiving)
extern void *Wayland_data_offer_receive(SDL_WaylandDataOffer *offer,
                                        const char *mime_type,
                                        size_t *length);
extern void *Wayland_primary_selection_offer_receive(SDL_WaylandPrimarySelectionOffer *offer,
                                                     const char *mime_type,
                                                     size_t *length);
extern bool Wayland_data_offer_has_mime(SDL_WaylandDataOffer *offer,
                                        const char *mime_type);
extern void Wayland_data_offer_notify_from_mimes(SDL_WaylandDataOffer *offer,
                                                 bool check_origin);
extern bool Wayland_primary_selection_offer_has_mime(SDL_WaylandPrimarySelectionOffer *offer,
                                                     const char *mime_type);
extern bool Wayland_data_offer_add_mime(SDL_WaylandDataOffer *offer,
                                        const char *mime_type);
extern bool Wayland_primary_selection_offer_add_mime(SDL_WaylandPrimarySelectionOffer *offer,
                                                     const char *mime_type);
extern void Wayland_data_offer_destroy(SDL_WaylandDataOffer *offer);
extern void Wayland_primary_selection_offer_destroy(SDL_WaylandPrimarySelectionOffer *offer);

// Clipboard / Primary Selection
extern bool Wayland_data_device_clear_selection(SDL_WaylandDataDevice *device);
extern bool Wayland_primary_selection_device_clear_selection(SDL_WaylandPrimarySelectionDevice *device);
extern bool Wayland_data_device_set_selection(SDL_WaylandDataDevice *device,
                                              SDL_WaylandDataSource *source,
                                              const char **mime_types,
                                              size_t mime_count);
extern bool Wayland_primary_selection_device_set_selection(SDL_WaylandPrimarySelectionDevice *device,
                                                           SDL_WaylandPrimarySelectionSource *source,
                                                           const char **mime_types,
                                                           size_t mime_count);
extern void Wayland_data_device_set_serial(SDL_WaylandDataDevice *device,
                                           uint32_t serial);
extern void Wayland_primary_selection_device_set_serial(SDL_WaylandPrimarySelectionDevice *device,
                                                        uint32_t serial);
#endif // SDL_waylanddatamanager_h_
