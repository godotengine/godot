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

#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <signal.h>

#include "../../core/unix/SDL_poll.h"
#include "../../events/SDL_events_c.h"
#include "../SDL_clipboard_c.h"

#include "SDL_waylandvideo.h"
#include "SDL_waylandevents_c.h"
#include "SDL_waylanddatamanager.h"
#include "primary-selection-unstable-v1-client-protocol.h"

/* FIXME: This is arbitrary, but we want this to be less than a frame because
 * any longer can potentially spin an infinite loop of PumpEvents (!)
 */
#define PIPE_TIMEOUT_NS SDL_MS_TO_NS(14)

static ssize_t write_pipe(int fd, const void *buffer, size_t total_length, size_t *pos)
{
    int ready = 0;
    ssize_t bytes_written = 0;
    ssize_t length = total_length - *pos;

    sigset_t sig_set;
    sigset_t old_sig_set;
    struct timespec zerotime = { 0 };

    ready = SDL_IOReady(fd, SDL_IOR_WRITE, PIPE_TIMEOUT_NS);

    sigemptyset(&sig_set);
    sigaddset(&sig_set, SIGPIPE);

#ifdef SDL_THREADS_DISABLED
    sigprocmask(SIG_BLOCK, &sig_set, &old_sig_set);
#else
    pthread_sigmask(SIG_BLOCK, &sig_set, &old_sig_set);
#endif

    if (ready == 0) {
        bytes_written = SDL_SetError("Pipe timeout");
    } else if (ready < 0) {
        bytes_written = SDL_SetError("Pipe select error");
    } else {
        if (length > 0) {
            bytes_written = write(fd, (Uint8 *)buffer + *pos, SDL_min(length, PIPE_BUF));
        }

        if (bytes_written > 0) {
            *pos += bytes_written;
        }
    }

    sigtimedwait(&sig_set, 0, &zerotime);

#ifdef SDL_THREADS_DISABLED
    sigprocmask(SIG_SETMASK, &old_sig_set, NULL);
#else
    pthread_sigmask(SIG_SETMASK, &old_sig_set, NULL);
#endif

    return bytes_written;
}

static ssize_t read_pipe(int fd, void **buffer, size_t *total_length)
{
    int ready = 0;
    void *output_buffer = NULL;
    char temp[PIPE_BUF];
    size_t new_buffer_length = 0;
    ssize_t bytes_read = 0;
    size_t pos = 0;

    ready = SDL_IOReady(fd, SDL_IOR_READ, PIPE_TIMEOUT_NS);

    if (ready == 0) {
        bytes_read = SDL_SetError("Pipe timeout");
    } else if (ready < 0) {
        bytes_read = SDL_SetError("Pipe select error");
    } else {
        bytes_read = read(fd, temp, sizeof(temp));
    }

    if (bytes_read > 0) {
        pos = *total_length;
        *total_length += bytes_read;

        new_buffer_length = *total_length + sizeof(Uint32);

        if (!*buffer) {
            output_buffer = SDL_malloc(new_buffer_length);
        } else {
            output_buffer = SDL_realloc(*buffer, new_buffer_length);
        }

        if (!output_buffer) {
            bytes_read = -1;
        } else {
            SDL_memcpy((Uint8 *)output_buffer + pos, temp, bytes_read);
            SDL_memset((Uint8 *)output_buffer + (new_buffer_length - sizeof(Uint32)), 0, sizeof(Uint32));

            *buffer = output_buffer;
        }
    }

    return bytes_read;
}

static SDL_MimeDataList *mime_data_list_find(struct wl_list *list,
                                             const char *mime_type)
{
    SDL_MimeDataList *found = NULL;

    SDL_MimeDataList *mime_list = NULL;
    wl_list_for_each (mime_list, list, link) {
        if (SDL_strcmp(mime_list->mime_type, mime_type) == 0) {
            found = mime_list;
            break;
        }
    }
    return found;
}

static bool mime_data_list_add(struct wl_list *list,
                               const char *mime_type,
                               const void *buffer, size_t length)
{
    bool result = true;
    size_t mime_type_length = 0;
    SDL_MimeDataList *mime_data = NULL;
    void *internal_buffer = NULL;

    if (buffer) {
        internal_buffer = SDL_malloc(length);
        if (!internal_buffer) {
            return false;
        }
        SDL_memcpy(internal_buffer, buffer, length);
    }

    mime_data = mime_data_list_find(list, mime_type);

    if (!mime_data) {
        mime_data = SDL_calloc(1, sizeof(*mime_data));
        if (!mime_data) {
            result = false;
        } else {
            WAYLAND_wl_list_insert(list, &(mime_data->link));

            mime_type_length = SDL_strlen(mime_type) + 1;
            mime_data->mime_type = SDL_malloc(mime_type_length);
            if (!mime_data->mime_type) {
                result = false;
            } else {
                SDL_memcpy(mime_data->mime_type, mime_type, mime_type_length);
            }
        }
    }

    if (mime_data && buffer && length > 0) {
        if (mime_data->data) {
            SDL_free(mime_data->data);
        }
        mime_data->data = internal_buffer;
        mime_data->length = length;
    } else {
        SDL_free(internal_buffer);
    }

    return result;
}

static void mime_data_list_free(struct wl_list *list)
{
    SDL_MimeDataList *mime_data = NULL;
    SDL_MimeDataList *next = NULL;

    wl_list_for_each_safe (mime_data, next, list, link) {
        if (mime_data->data) {
            SDL_free(mime_data->data);
        }
        if (mime_data->mime_type) {
            SDL_free(mime_data->mime_type);
        }
        SDL_free(mime_data);
    }
}

static size_t Wayland_send_data(const void *data, size_t length, int fd)
{
    size_t result = 0;

    if (length > 0 && data) {
        while (write_pipe(fd, data, length, &result) > 0) {
            // Just keep spinning
        }
    }
    close(fd);

    return result;
}

ssize_t Wayland_data_source_send(SDL_WaylandDataSource *source, const char *mime_type, int fd)
{
    const void *data = NULL;
    size_t length = 0;

    if (SDL_strcmp(mime_type, SDL_DATA_ORIGIN_MIME) == 0) {
        data = source->data_device->id_str;
        length = SDL_strlen(source->data_device->id_str);
    } else if (source->callback) {
        data = source->callback(source->userdata.data, mime_type, &length);
    }

    return Wayland_send_data(data, length, fd);
}

ssize_t Wayland_primary_selection_source_send(SDL_WaylandPrimarySelectionSource *source, const char *mime_type, int fd)
{
    const void *data = NULL;
    size_t length = 0;

    if (source->callback) {
        data = source->callback(source->userdata.data, mime_type, &length);
    }

    return Wayland_send_data(data, length, fd);
}

void Wayland_data_source_set_callback(SDL_WaylandDataSource *source,
                                      SDL_ClipboardDataCallback callback,
                                      void *userdata,
                                      Uint32 sequence)
{
    if (source) {
        source->callback = callback;
        source->userdata.sequence = sequence;
        source->userdata.data = userdata;
    }
}

void Wayland_primary_selection_source_set_callback(SDL_WaylandPrimarySelectionSource *source,
                                                   SDL_ClipboardDataCallback callback,
                                                   void *userdata)
{
    if (source) {
        source->callback = callback;
        source->userdata.sequence = 0;
        source->userdata.data = userdata;
    }
}

static void *Wayland_clone_data_buffer(const void *buffer, const size_t *len)
{
    void *clone = NULL;
    if (*len > 0 && buffer) {
        clone = SDL_malloc((*len)+sizeof(Uint32));
        if (clone) {
            SDL_memcpy(clone, buffer, *len);
            SDL_memset((Uint8 *)clone + *len, 0, sizeof(Uint32));
        }
    }
    return clone;
}

void *Wayland_data_source_get_data(SDL_WaylandDataSource *source,
                                   const char *mime_type, size_t *length)
{
    void *buffer = NULL;
    const void *internal_buffer;
    *length = 0;

    if (!source) {
        SDL_SetError("Invalid data source");
    } else if (source->callback) {
        internal_buffer = source->callback(source->userdata.data, mime_type, length);
        buffer = Wayland_clone_data_buffer(internal_buffer, length);
    }

    return buffer;
}

void *Wayland_primary_selection_source_get_data(SDL_WaylandPrimarySelectionSource *source,
                                                const char *mime_type, size_t *length)
{
    void *buffer = NULL;
    const void *internal_buffer;
    *length = 0;

    if (!source) {
        SDL_SetError("Invalid primary selection source");
    } else if (source->callback) {
        internal_buffer = source->callback(source->userdata.data, mime_type, length);
        buffer = Wayland_clone_data_buffer(internal_buffer, length);
    }

    return buffer;
}

void Wayland_data_source_destroy(SDL_WaylandDataSource *source)
{
    if (source) {
        SDL_WaylandDataDevice *data_device = (SDL_WaylandDataDevice *)source->data_device;
        if (data_device && (data_device->selection_source == source)) {
            data_device->selection_source = NULL;
        }
        wl_data_source_destroy(source->source);
        if (source->userdata.sequence) {
            SDL_CancelClipboardData(source->userdata.sequence);
        } else {
            SDL_free(source->userdata.data);
        }
        SDL_free(source);
    }
}

void Wayland_primary_selection_source_destroy(SDL_WaylandPrimarySelectionSource *source)
{
    if (source) {
        SDL_WaylandPrimarySelectionDevice *primary_selection_device = (SDL_WaylandPrimarySelectionDevice *)source->primary_selection_device;
        if (primary_selection_device && (primary_selection_device->selection_source == source)) {
            primary_selection_device->selection_source = NULL;
        }
        zwp_primary_selection_source_v1_destroy(source->source);
        if (source->userdata.sequence == 0) {
            SDL_free(source->userdata.data);
        }
        SDL_free(source);
    }
}

static void offer_source_done_handler(void *data, struct wl_callback *callback, uint32_t callback_data)
{
    if (!callback) {
        return;
    }

    SDL_WaylandDataOffer *offer = data;
    char *id = NULL;
    size_t length = 0;

    wl_callback_destroy(offer->callback);
    offer->callback = NULL;

    while (read_pipe(offer->read_fd, (void **)&id, &length) > 0) {
    }
    close(offer->read_fd);
    offer->read_fd = -1;

    if (id) {
        const bool source_is_external = SDL_strncmp(offer->data_device->id_str, id, length) != 0;
        SDL_free(id);
        if (source_is_external) {
            Wayland_data_offer_notify_from_mimes(offer, false);
        }
    }
}

static struct wl_callback_listener offer_source_listener = {
    offer_source_done_handler
};

static void Wayland_data_offer_check_source(SDL_WaylandDataOffer *offer, const char *mime_type)
{
    SDL_WaylandDataDevice *data_device = NULL;
    int pipefd[2];

    if (!offer) {
        SDL_SetError("Invalid data offer");
    }
    data_device = offer->data_device;
    if (!data_device) {
        SDL_SetError("Data device not initialized");
    } else if (pipe2(pipefd, O_CLOEXEC | O_NONBLOCK) == -1) {
        SDL_SetError("Could not read pipe");
    } else {
        if (offer->callback) {
            wl_callback_destroy(offer->callback);
        }
        if (offer->read_fd >= 0) {
            close(offer->read_fd);
        }

        offer->read_fd = pipefd[0];

        wl_data_offer_receive(offer->offer, mime_type, pipefd[1]);
        close(pipefd[1]);

        offer->callback = wl_display_sync(offer->data_device->seat->display->display);
        wl_callback_add_listener(offer->callback, &offer_source_listener, offer);

        WAYLAND_wl_display_flush(data_device->seat->display->display);
    }
}

void Wayland_data_offer_notify_from_mimes(SDL_WaylandDataOffer *offer, bool check_origin)
{
    int nformats = 0;
    char **new_mime_types = NULL;
    if (offer) {
        size_t alloc_size = 0;

        // Do a first pass to compute allocation size.
        SDL_MimeDataList *item = NULL;
        wl_list_for_each(item, &offer->mimes, link) {
            // If origin metadata is found, queue a check and wait for confirmation that this offer isn't recursive.
            if (check_origin && SDL_strcmp(item->mime_type, SDL_DATA_ORIGIN_MIME) == 0) {
                Wayland_data_offer_check_source(offer, item->mime_type);
                return;
            }

            ++nformats;
            alloc_size += SDL_strlen(item->mime_type) + 1;
        }

        alloc_size += (nformats + 1) * sizeof(char *);

        new_mime_types = SDL_AllocateTemporaryMemory(alloc_size);
        if (!new_mime_types) {
            SDL_LogError(SDL_LOG_CATEGORY_INPUT, "unable to allocate new_mime_types");
            return;
        }

        // Second pass to fill.
        char *strPtr = (char *)(new_mime_types + nformats + 1);
        item = NULL;
        int i = 0;
        wl_list_for_each(item, &offer->mimes, link) {
            new_mime_types[i] = strPtr;
            strPtr = stpcpy(strPtr, item->mime_type) + 1;
            i++;
        }
        new_mime_types[nformats] = NULL;
    }

    SDL_SendClipboardUpdate(false, new_mime_types, nformats);
}

void *Wayland_data_offer_receive(SDL_WaylandDataOffer *offer,
                                 const char *mime_type, size_t *length)
{
    SDL_WaylandDataDevice *data_device = NULL;

    int pipefd[2];
    void *buffer = NULL;
    *length = 0;

    if (!offer) {
        SDL_SetError("Invalid data offer");
        return NULL;
    }
    data_device = offer->data_device;
    if (!data_device) {
        SDL_SetError("Data device not initialized");
    } else if (pipe2(pipefd, O_CLOEXEC | O_NONBLOCK) == -1) {
        SDL_SetError("Could not read pipe");
    } else {
        wl_data_offer_receive(offer->offer, mime_type, pipefd[1]);
        close(pipefd[1]);

        WAYLAND_wl_display_flush(data_device->seat->display->display);

        while (read_pipe(pipefd[0], &buffer, length) > 0) {
        }
        close(pipefd[0]);
    }
    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". In Wayland_data_offer_receive for '%s', buffer (%zu) at %p",
                 mime_type, *length, buffer);
    return buffer;
}

void *Wayland_primary_selection_offer_receive(SDL_WaylandPrimarySelectionOffer *offer,
                                              const char *mime_type, size_t *length)
{
    SDL_WaylandPrimarySelectionDevice *primary_selection_device = NULL;

    int pipefd[2];
    void *buffer = NULL;
    *length = 0;

    if (!offer) {
        SDL_SetError("Invalid data offer");
        return NULL;
    }
    primary_selection_device = offer->primary_selection_device;
    if (!primary_selection_device) {
        SDL_SetError("Primary selection device not initialized");
    } else if (pipe2(pipefd, O_CLOEXEC | O_NONBLOCK) == -1) {
        SDL_SetError("Could not read pipe");
    } else {
        zwp_primary_selection_offer_v1_receive(offer->offer, mime_type, pipefd[1]);
        close(pipefd[1]);

        WAYLAND_wl_display_flush(primary_selection_device->seat->display->display);

        while (read_pipe(pipefd[0], &buffer, length) > 0) {
        }
        close(pipefd[0]);
    }
    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". In Wayland_primary_selection_offer_receive for '%s', buffer (%zu) at %p",
                 mime_type, *length, buffer);
    return buffer;
}

bool Wayland_data_offer_add_mime(SDL_WaylandDataOffer *offer,
                                const char *mime_type)
{
    return mime_data_list_add(&offer->mimes, mime_type, NULL, 0);
}

bool Wayland_primary_selection_offer_add_mime(SDL_WaylandPrimarySelectionOffer *offer,
                                             const char *mime_type)
{
    return mime_data_list_add(&offer->mimes, mime_type, NULL, 0);
}

bool Wayland_data_offer_has_mime(SDL_WaylandDataOffer *offer,
                                 const char *mime_type)
{
    bool found = false;

    if (offer) {
        found = mime_data_list_find(&offer->mimes, mime_type) != NULL;
    }
    return found;
}

bool Wayland_primary_selection_offer_has_mime(SDL_WaylandPrimarySelectionOffer *offer,
                                              const char *mime_type)
{
    bool found = false;

    if (offer) {
        found = mime_data_list_find(&offer->mimes, mime_type) != NULL;
    }
    return found;
}

void Wayland_data_offer_destroy(SDL_WaylandDataOffer *offer)
{
    if (offer) {
        if (offer->callback) {
            wl_callback_destroy(offer->callback);
        }
        if (offer->read_fd >= 0) {
            close(offer->read_fd);
        }
        wl_data_offer_destroy(offer->offer);
        mime_data_list_free(&offer->mimes);
        SDL_free(offer);
    }
}

void Wayland_primary_selection_offer_destroy(SDL_WaylandPrimarySelectionOffer *offer)
{
    if (offer) {
        zwp_primary_selection_offer_v1_destroy(offer->offer);
        mime_data_list_free(&offer->mimes);
        SDL_free(offer);
    }
}

bool Wayland_data_device_clear_selection(SDL_WaylandDataDevice *data_device)
{
    bool result = true;

    if (!data_device || !data_device->data_device) {
        result = SDL_SetError("Invalid Data Device");
    } else if (data_device->selection_source) {
        wl_data_device_set_selection(data_device->data_device, NULL, 0);
        Wayland_data_source_destroy(data_device->selection_source);
        data_device->selection_source = NULL;
    }
    return result;
}

bool Wayland_primary_selection_device_clear_selection(SDL_WaylandPrimarySelectionDevice *primary_selection_device)
{
    bool result = true;

    if (!primary_selection_device || !primary_selection_device->primary_selection_device) {
        result = SDL_SetError("Invalid Primary Selection Device");
    } else if (primary_selection_device->selection_source) {
        zwp_primary_selection_device_v1_set_selection(primary_selection_device->primary_selection_device,
                                                      NULL, 0);
        Wayland_primary_selection_source_destroy(primary_selection_device->selection_source);
        primary_selection_device->selection_source = NULL;
    }
    return result;
}

bool Wayland_data_device_set_selection(SDL_WaylandDataDevice *data_device,
                                       SDL_WaylandDataSource *source,
                                       const char **mime_types,
                                       size_t mime_count)
{
    bool result = true;

    if (!data_device) {
        result = SDL_SetError("Invalid Data Device");
    } else if (!source) {
        result = SDL_SetError("Invalid source");
    } else {
        size_t index = 0;
        const char *mime_type;

        for (index = 0; index < mime_count; ++index) {
            mime_type = mime_types[index];
            wl_data_source_offer(source->source,
                                 mime_type);
        }

        // Advertise the data origin MIME
        wl_data_source_offer(source->source, SDL_DATA_ORIGIN_MIME);

        if (index == 0) {
            Wayland_data_device_clear_selection(data_device);
            result = SDL_SetError("No mime data");
        } else {
            // Only set if there is a valid serial if not set it later
            if (data_device->selection_serial != 0) {
                wl_data_device_set_selection(data_device->data_device,
                                             source->source,
                                             data_device->selection_serial);
            }
            if (data_device->selection_source) {
                Wayland_data_source_destroy(data_device->selection_source);
            }
            data_device->selection_source = source;
            source->data_device = data_device;
        }
    }

    return result;
}

bool Wayland_primary_selection_device_set_selection(SDL_WaylandPrimarySelectionDevice *primary_selection_device,
                                                   SDL_WaylandPrimarySelectionSource *source,
                                                   const char **mime_types,
                                                   size_t mime_count)
{
    bool result = true;

    if (!primary_selection_device) {
        result = SDL_SetError("Invalid Primary Selection Device");
    } else if (!source) {
        result = SDL_SetError("Invalid source");
    } else {
        size_t index = 0;
        const char *mime_type = mime_types[index];

        for (index = 0; index < mime_count; ++index) {
            mime_type = mime_types[index];
            zwp_primary_selection_source_v1_offer(source->source, mime_type);
        }

        if (index == 0) {
            Wayland_primary_selection_device_clear_selection(primary_selection_device);
            result = SDL_SetError("No mime data");
        } else {
            // Only set if there is a valid serial if not set it later
            if (primary_selection_device->selection_serial != 0) {
                zwp_primary_selection_device_v1_set_selection(primary_selection_device->primary_selection_device,
                                                              source->source,
                                                              primary_selection_device->selection_serial);
            }
            if (primary_selection_device->selection_source) {
                Wayland_primary_selection_source_destroy(primary_selection_device->selection_source);
            }
            primary_selection_device->selection_source = source;
            source->primary_selection_device = primary_selection_device;
        }
    }

    return result;
}

void Wayland_data_device_set_serial(SDL_WaylandDataDevice *data_device, uint32_t serial)
{
    if (data_device) {
        data_device->selection_serial = serial;

        // If there was no serial and there is a pending selection set it now.
        if (data_device->selection_serial == 0 && data_device->selection_source) {
            wl_data_device_set_selection(data_device->data_device,
                                         data_device->selection_source->source,
                                         data_device->selection_serial);
        }
    }
}

void Wayland_primary_selection_device_set_serial(SDL_WaylandPrimarySelectionDevice *primary_selection_device,
                                                 uint32_t serial)
{
    if (primary_selection_device) {
        primary_selection_device->selection_serial = serial;

        // If there was no serial and there is a pending selection set it now.
        if (primary_selection_device->selection_serial == 0 && primary_selection_device->selection_source) {
            zwp_primary_selection_device_v1_set_selection(primary_selection_device->primary_selection_device,
                                                          primary_selection_device->selection_source->source,
                                                          primary_selection_device->selection_serial);
        }
    }
}

#endif // SDL_VIDEO_DRIVER_WAYLAND
