/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>
  Copyright (C) 2024 Wim Taymans <wtaymans@redhat.com>

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

#ifdef SDL_CAMERA_DRIVER_PIPEWIRE

#include "../SDL_syscamera.h"

#ifdef HAVE_DBUS_DBUS_H
#include "../../core/linux/SDL_dbus.h"
#endif

#include <spa/utils/type.h>
#include <spa/pod/builder.h>
#include <spa/pod/iter.h>
#include <spa/param/video/raw.h>
#include <spa/param/video/format.h>
#include <spa/utils/result.h>
#include <spa/utils/json.h>

#include <pipewire/pipewire.h>
#include <pipewire/extensions/metadata.h>

#define PW_POD_BUFFER_LENGTH         1024
#define PW_THREAD_NAME_BUFFER_LENGTH 128
#define PW_MAX_IDENTIFIER_LENGTH     256

#define PW_REQUIRED_MAJOR       1
#define PW_REQUIRED_MINOR       0
#define PW_REQUIRED_PATCH       0

enum PW_READY_FLAGS
{
    PW_READY_FLAG_BUFFER_ADDED = 0x1,
    PW_READY_FLAG_STREAM_READY = 0x2,
    PW_READY_FLAG_ALL_BITS = 0x3
};

#define PW_ID_TO_HANDLE(x) (void *)((uintptr_t)x)
#define PW_HANDLE_TO_ID(x) (uint32_t)((uintptr_t)x)

static bool pipewire_initialized = false;

// Pipewire entry points
static const char *(*PIPEWIRE_pw_get_library_version)(void);
#if PW_CHECK_VERSION(0, 3, 75)
static bool (*PIPEWIRE_pw_check_library_version)(int major, int minor, int micro);
#endif
static void (*PIPEWIRE_pw_init)(int *, char ***);
static void (*PIPEWIRE_pw_deinit)(void);
static struct pw_main_loop *(*PIPEWIRE_pw_main_loop_new)(const struct spa_dict *loop);
static struct pw_loop *(*PIPEWIRE_pw_main_loop_get_loop)(struct pw_main_loop *loop);
static int (*PIPEWIRE_pw_main_loop_run)(struct pw_main_loop *loop);
static int (*PIPEWIRE_pw_main_loop_quit)(struct pw_main_loop *loop);
static void(*PIPEWIRE_pw_main_loop_destroy)(struct pw_main_loop *loop);
static struct pw_thread_loop *(*PIPEWIRE_pw_thread_loop_new)(const char *, const struct spa_dict *);
static void (*PIPEWIRE_pw_thread_loop_destroy)(struct pw_thread_loop *);
static void (*PIPEWIRE_pw_thread_loop_stop)(struct pw_thread_loop *);
static struct pw_loop *(*PIPEWIRE_pw_thread_loop_get_loop)(struct pw_thread_loop *);
static void (*PIPEWIRE_pw_thread_loop_lock)(struct pw_thread_loop *);
static void (*PIPEWIRE_pw_thread_loop_unlock)(struct pw_thread_loop *);
static void (*PIPEWIRE_pw_thread_loop_signal)(struct pw_thread_loop *, bool);
static void (*PIPEWIRE_pw_thread_loop_wait)(struct pw_thread_loop *);
static int (*PIPEWIRE_pw_thread_loop_start)(struct pw_thread_loop *);
static struct pw_context *(*PIPEWIRE_pw_context_new)(struct pw_loop *, struct pw_properties *, size_t);
static void (*PIPEWIRE_pw_context_destroy)(struct pw_context *);
static struct pw_core *(*PIPEWIRE_pw_context_connect)(struct pw_context *, struct pw_properties *, size_t);
#ifdef SDL_USE_LIBDBUS
static struct pw_core *(*PIPEWIRE_pw_context_connect_fd)(struct pw_context *, int, struct pw_properties *, size_t);
#endif
static void (*PIPEWIRE_pw_proxy_add_object_listener)(struct pw_proxy *, struct spa_hook *, const void *, void *);
static void (*PIPEWIRE_pw_proxy_add_listener)(struct pw_proxy *, struct spa_hook *, const struct pw_proxy_events *, void *);
static void *(*PIPEWIRE_pw_proxy_get_user_data)(struct pw_proxy *);
static void (*PIPEWIRE_pw_proxy_destroy)(struct pw_proxy *);
static int (*PIPEWIRE_pw_core_disconnect)(struct pw_core *);
static struct pw_node_info * (*PIPEWIRE_pw_node_info_merge)(struct pw_node_info *info, const struct pw_node_info *update, bool reset);
static void (*PIPEWIRE_pw_node_info_free)(struct pw_node_info *info);
static struct pw_stream *(*PIPEWIRE_pw_stream_new)(struct pw_core *, const char *, struct pw_properties *);
static void (*PIPEWIRE_pw_stream_add_listener)(struct pw_stream *stream, struct spa_hook *listener, const struct pw_stream_events *events, void *data);
static void (*PIPEWIRE_pw_stream_destroy)(struct pw_stream *);
static int (*PIPEWIRE_pw_stream_connect)(struct pw_stream *, enum pw_direction, uint32_t, enum pw_stream_flags,
                                         const struct spa_pod **, uint32_t);
static enum pw_stream_state (*PIPEWIRE_pw_stream_get_state)(struct pw_stream *stream, const char **error);
static struct pw_buffer *(*PIPEWIRE_pw_stream_dequeue_buffer)(struct pw_stream *);
static int (*PIPEWIRE_pw_stream_queue_buffer)(struct pw_stream *, struct pw_buffer *);
static struct pw_properties *(*PIPEWIRE_pw_properties_new)(const char *, ...)SPA_SENTINEL;
static struct pw_properties *(*PIPEWIRE_pw_properties_new_dict)(const struct spa_dict *dict);
static int (*PIPEWIRE_pw_properties_set)(struct pw_properties *, const char *, const char *);
static int (*PIPEWIRE_pw_properties_setf)(struct pw_properties *, const char *, const char *, ...) SPA_PRINTF_FUNC(3, 4);

#ifdef SDL_CAMERA_DRIVER_PIPEWIRE_DYNAMIC

static const char *pipewire_library = SDL_CAMERA_DRIVER_PIPEWIRE_DYNAMIC;
static SDL_SharedObject *pipewire_handle = NULL;

static bool pipewire_dlsym(const char *fn, void **addr)
{
    *addr = SDL_LoadFunction(pipewire_handle, fn);
    if (!*addr) {
        // Don't call SDL_SetError(): SDL_LoadFunction already did.
        return false;
    }

    return true;
}

#define SDL_PIPEWIRE_SYM(x)                                     \
    if (!pipewire_dlsym(#x, (void **)(char *)&PIPEWIRE_##x))    \
        return false

static bool load_pipewire_library(void)
{
    pipewire_handle = SDL_LoadObject(pipewire_library);
    return pipewire_handle ? true : false;
}

static void unload_pipewire_library(void)
{
    if (pipewire_handle) {
        SDL_UnloadObject(pipewire_handle);
        pipewire_handle = NULL;
    }
}

#else

#define SDL_PIPEWIRE_SYM(x) PIPEWIRE_##x = x

static bool load_pipewire_library(void)
{
    return true;
}

static void unload_pipewire_library(void)
{
    // Nothing to do
}

#endif // SDL_CAMERA_DRIVER_PIPEWIRE_DYNAMIC

static bool load_pipewire_syms(void)
{
    SDL_PIPEWIRE_SYM(pw_get_library_version);
#if PW_CHECK_VERSION(0, 3, 75)
    SDL_PIPEWIRE_SYM(pw_check_library_version);
#endif
    SDL_PIPEWIRE_SYM(pw_init);
    SDL_PIPEWIRE_SYM(pw_deinit);
    SDL_PIPEWIRE_SYM(pw_main_loop_new);
    SDL_PIPEWIRE_SYM(pw_main_loop_get_loop);
    SDL_PIPEWIRE_SYM(pw_main_loop_run);
    SDL_PIPEWIRE_SYM(pw_main_loop_quit);
    SDL_PIPEWIRE_SYM(pw_main_loop_destroy);
    SDL_PIPEWIRE_SYM(pw_thread_loop_new);
    SDL_PIPEWIRE_SYM(pw_thread_loop_destroy);
    SDL_PIPEWIRE_SYM(pw_thread_loop_stop);
    SDL_PIPEWIRE_SYM(pw_thread_loop_get_loop);
    SDL_PIPEWIRE_SYM(pw_thread_loop_lock);
    SDL_PIPEWIRE_SYM(pw_thread_loop_unlock);
    SDL_PIPEWIRE_SYM(pw_thread_loop_signal);
    SDL_PIPEWIRE_SYM(pw_thread_loop_wait);
    SDL_PIPEWIRE_SYM(pw_thread_loop_start);
    SDL_PIPEWIRE_SYM(pw_context_new);
    SDL_PIPEWIRE_SYM(pw_context_destroy);
    SDL_PIPEWIRE_SYM(pw_context_connect);
#ifdef SDL_USE_LIBDBUS
    SDL_PIPEWIRE_SYM(pw_context_connect_fd);
#endif
    SDL_PIPEWIRE_SYM(pw_proxy_add_listener);
    SDL_PIPEWIRE_SYM(pw_proxy_add_object_listener);
    SDL_PIPEWIRE_SYM(pw_proxy_get_user_data);
    SDL_PIPEWIRE_SYM(pw_proxy_destroy);
    SDL_PIPEWIRE_SYM(pw_core_disconnect);
    SDL_PIPEWIRE_SYM(pw_node_info_merge);
    SDL_PIPEWIRE_SYM(pw_node_info_free);
    SDL_PIPEWIRE_SYM(pw_stream_new);
    SDL_PIPEWIRE_SYM(pw_stream_add_listener);
    SDL_PIPEWIRE_SYM(pw_stream_destroy);
    SDL_PIPEWIRE_SYM(pw_stream_connect);
    SDL_PIPEWIRE_SYM(pw_stream_get_state);
    SDL_PIPEWIRE_SYM(pw_stream_dequeue_buffer);
    SDL_PIPEWIRE_SYM(pw_stream_queue_buffer);
    SDL_PIPEWIRE_SYM(pw_properties_new);
    SDL_PIPEWIRE_SYM(pw_properties_new_dict);
    SDL_PIPEWIRE_SYM(pw_properties_set);
    SDL_PIPEWIRE_SYM(pw_properties_setf);

    return true;
}

static bool init_pipewire_library(void)
{
    if (load_pipewire_library()) {
        if (load_pipewire_syms()) {
            PIPEWIRE_pw_init(NULL, NULL);
            return true;
        }
    }
    return false;
}

static void deinit_pipewire_library(void)
{
    PIPEWIRE_pw_deinit();
    unload_pipewire_library();
}

// The global hotplug thread and associated objects.
static struct
{
    struct pw_thread_loop *loop;

    struct pw_context *context;

    struct pw_core *core;
    struct spa_hook core_listener;
    int server_major;
    int server_minor;
    int server_patch;
    int last_seq;
    int pending_seq;

    struct pw_registry *registry;
    struct spa_hook registry_listener;

    struct spa_list global_list;

    bool have_1_0_5;
    bool init_complete;
    bool events_enabled;
} hotplug;

struct global
{
    struct spa_list link;

    const struct global_class *class;

    uint32_t id;
    uint32_t permissions;
    struct pw_properties *props;

    char *name;

    struct pw_proxy *proxy;
    struct spa_hook proxy_listener;
    struct spa_hook object_listener;

    int changed;
    void *info;
    struct spa_list pending_list;
    struct spa_list param_list;

    bool added;
};

struct global_class
{
    const char *type;
    uint32_t version;
    const void *events;
    int (*init) (struct global *g);
    void (*destroy) (struct global *g);
};

struct param {
    uint32_t id;
    int32_t seq;
    struct spa_list link;
    struct spa_pod *param;
};

static uint32_t param_clear(struct spa_list *param_list, uint32_t id)
{
    struct param *p, *t;
    uint32_t count = 0;

    spa_list_for_each_safe(p, t, param_list, link) {
        if (id == SPA_ID_INVALID || p->id == id) {
            spa_list_remove(&p->link);
            free(p); // This should NOT be SDL_free()
            count++;
        }
    }
    return count;
}

#if PW_CHECK_VERSION(0,3,60)
#define SPA_PARAMS_INFO_SEQ(p)  ((p).seq)
#else
#define SPA_PARAMS_INFO_SEQ(p)  ((p).padding[0])
#endif

static struct param *param_add(struct spa_list *params,
                int seq, uint32_t id, const struct spa_pod *param)
{
    struct param *p;

    if (id == SPA_ID_INVALID) {
        if (param == NULL || !spa_pod_is_object(param)) {
            errno = EINVAL;
            return NULL;
        }
        id = SPA_POD_OBJECT_ID(param);
    }

    p = malloc(sizeof(*p) + (param != NULL ? SPA_POD_SIZE(param) : 0));
    if (p == NULL)
        return NULL;

    p->id = id;
    p->seq = seq;
    if (param != NULL) {
        p->param = SPA_PTROFF(p, sizeof(*p), struct spa_pod);
        SDL_memcpy(p->param, param, SPA_POD_SIZE(param));
    } else {
        param_clear(params, id);
        p->param = NULL;
    }
    spa_list_append(params, &p->link);

    return p;
}

static void param_update(struct spa_list *param_list, struct spa_list *pending_list,
                        uint32_t n_params, struct spa_param_info *params)
{
    struct param *p, *t;
    uint32_t i;

    for (i = 0; i < n_params; i++) {
        spa_list_for_each_safe(p, t, pending_list, link) {
            if (p->id == params[i].id &&
                p->seq != SPA_PARAMS_INFO_SEQ(params[i]) &&
                p->param != NULL) {
                    spa_list_remove(&p->link);
                    free(p); // This should NOT be SDL_free()
            }
        }
    }
    spa_list_consume(p, pending_list, link) {
        spa_list_remove(&p->link);
        if (p->param == NULL) {
            param_clear(param_list, p->id);
            free(p); // This should NOT be SDL_free()
        } else {
            spa_list_append(param_list, &p->link);
        }
    }
}

static struct sdl_video_format {
    SDL_PixelFormat format;
    SDL_Colorspace colorspace;
    uint32_t id;
} sdl_video_formats[] = {
    { SDL_PIXELFORMAT_RGBX32, SDL_COLORSPACE_SRGB, SPA_VIDEO_FORMAT_RGBx },
    { SDL_PIXELFORMAT_XRGB32, SDL_COLORSPACE_SRGB, SPA_VIDEO_FORMAT_xRGB },
    { SDL_PIXELFORMAT_BGRX32, SDL_COLORSPACE_SRGB, SPA_VIDEO_FORMAT_BGRx },
    { SDL_PIXELFORMAT_XBGR32, SDL_COLORSPACE_SRGB, SPA_VIDEO_FORMAT_xBGR },
    { SDL_PIXELFORMAT_RGBA32, SDL_COLORSPACE_SRGB, SPA_VIDEO_FORMAT_RGBA },
    { SDL_PIXELFORMAT_ARGB32, SDL_COLORSPACE_SRGB, SPA_VIDEO_FORMAT_ARGB },
    { SDL_PIXELFORMAT_BGRA32, SDL_COLORSPACE_SRGB, SPA_VIDEO_FORMAT_BGRA },
    { SDL_PIXELFORMAT_ABGR32, SDL_COLORSPACE_SRGB, SPA_VIDEO_FORMAT_ABGR },
    { SDL_PIXELFORMAT_RGB24, SDL_COLORSPACE_SRGB, SPA_VIDEO_FORMAT_RGB },
    { SDL_PIXELFORMAT_BGR24, SDL_COLORSPACE_SRGB, SPA_VIDEO_FORMAT_BGR },
    { SDL_PIXELFORMAT_YV12, SDL_COLORSPACE_BT709_LIMITED, SPA_VIDEO_FORMAT_YV12 },
    { SDL_PIXELFORMAT_IYUV, SDL_COLORSPACE_BT709_LIMITED, SPA_VIDEO_FORMAT_I420 },
    { SDL_PIXELFORMAT_YUY2, SDL_COLORSPACE_BT709_LIMITED, SPA_VIDEO_FORMAT_YUY2 },
    { SDL_PIXELFORMAT_UYVY, SDL_COLORSPACE_BT709_LIMITED, SPA_VIDEO_FORMAT_UYVY },
    { SDL_PIXELFORMAT_YVYU, SDL_COLORSPACE_BT709_LIMITED, SPA_VIDEO_FORMAT_YVYU },
    { SDL_PIXELFORMAT_NV12, SDL_COLORSPACE_BT709_LIMITED, SPA_VIDEO_FORMAT_NV12 },
    { SDL_PIXELFORMAT_NV21, SDL_COLORSPACE_BT709_LIMITED, SPA_VIDEO_FORMAT_NV21 }
};

static uint32_t sdl_format_to_id(SDL_PixelFormat format)
{
    struct sdl_video_format *f;
    SPA_FOR_EACH_ELEMENT(sdl_video_formats, f) {
        if (f->format == format)
            return f->id;
    }
    return SPA_VIDEO_FORMAT_UNKNOWN;
}

static void id_to_sdl_format(uint32_t id, SDL_PixelFormat *format, SDL_Colorspace *colorspace)
{
    struct sdl_video_format *f;
    SPA_FOR_EACH_ELEMENT(sdl_video_formats, f) {
        if (f->id == id) {
            *format = f->format;
            *colorspace = f->colorspace;
            return;
        }
    }
    *format = SDL_PIXELFORMAT_UNKNOWN;
    *colorspace = SDL_COLORSPACE_UNKNOWN;
}

struct SDL_PrivateCameraData
{
    struct pw_stream *stream;
    struct spa_hook stream_listener;

    struct pw_array buffers;
};

static void on_process(void *data)
{
    PIPEWIRE_pw_thread_loop_signal(hotplug.loop, false);
}

static void on_stream_state_changed(void *data, enum pw_stream_state old,
                enum pw_stream_state state, const char *error)
{
    SDL_Camera *device = data;
    switch (state) {
    case PW_STREAM_STATE_UNCONNECTED:
        break;
    case PW_STREAM_STATE_STREAMING:
        SDL_CameraPermissionOutcome(device, true);
        break;
    default:
        break;
    }
}

static void on_stream_param_changed(void *data, uint32_t id, const struct spa_pod *param)
{
}

static void on_add_buffer(void *data, struct pw_buffer *buffer)
{
    SDL_Camera *device = data;
    pw_array_add_ptr(&device->hidden->buffers, buffer);
}

static void on_remove_buffer(void *data, struct pw_buffer *buffer)
{
    SDL_Camera *device = data;
    struct pw_buffer **p;
    pw_array_for_each(p, &device->hidden->buffers) {
        if (*p == buffer) {
            pw_array_remove(&device->hidden->buffers, p);
            return;
        }
    }
}

static const struct pw_stream_events stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .add_buffer = on_add_buffer,
    .remove_buffer = on_remove_buffer,
    .state_changed = on_stream_state_changed,
    .param_changed = on_stream_param_changed,
    .process = on_process,
};

static bool PIPEWIRECAMERA_OpenDevice(SDL_Camera *device, const SDL_CameraSpec *spec)
{
    struct pw_properties *props;
    const struct spa_pod *params[3];
    int res, n_params = 0;
    uint8_t buffer[1024];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

    if (!device) {
        return false;
    }
    device->hidden = (struct SDL_PrivateCameraData *) SDL_calloc(1, sizeof (struct SDL_PrivateCameraData));
    if (device->hidden == NULL) {
        return false;
    }
    pw_array_init(&device->hidden->buffers, 64);

    PIPEWIRE_pw_thread_loop_lock(hotplug.loop);

    props = PIPEWIRE_pw_properties_new(PW_KEY_MEDIA_TYPE, "Video",
                    PW_KEY_MEDIA_CATEGORY, "Capture",
                    PW_KEY_MEDIA_ROLE, "Camera",
                    PW_KEY_TARGET_OBJECT, device->name,
                    NULL);
    if (props == NULL) {
        return false;
    }

    device->hidden->stream = PIPEWIRE_pw_stream_new(hotplug.core, "SDL PipeWire Camera", props);
    if (device->hidden->stream == NULL) {
        return false;
    }

    PIPEWIRE_pw_stream_add_listener(device->hidden->stream,
                    &device->hidden->stream_listener,
                    &stream_events, device);

    if (spec->format == SDL_PIXELFORMAT_MJPG) {
        params[n_params++] = spa_pod_builder_add_object(&b,
                SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
                SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
                        SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_mjpg),
                        SPA_FORMAT_VIDEO_size, SPA_POD_Rectangle(&SPA_RECTANGLE(spec->width, spec->height)),
                        SPA_FORMAT_VIDEO_framerate,
                    SPA_POD_Fraction(&SPA_FRACTION(spec->framerate_numerator, spec->framerate_denominator)));
    } else {
        params[n_params++] = spa_pod_builder_add_object(&b,
                SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
                SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
                        SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
                        SPA_FORMAT_VIDEO_format, SPA_POD_Id(sdl_format_to_id(spec->format)),
                        SPA_FORMAT_VIDEO_size, SPA_POD_Rectangle(&SPA_RECTANGLE(spec->width, spec->height)),
                        SPA_FORMAT_VIDEO_framerate,
                    SPA_POD_Fraction(&SPA_FRACTION(spec->framerate_numerator, spec->framerate_denominator)));
    }

    if ((res = PIPEWIRE_pw_stream_connect(device->hidden->stream,
                                    PW_DIRECTION_INPUT,
                                    PW_ID_ANY,
                                    PW_STREAM_FLAG_AUTOCONNECT |
                                    PW_STREAM_FLAG_MAP_BUFFERS,
                                    params, n_params)) < 0) {
        return false;
    }

    PIPEWIRE_pw_thread_loop_unlock(hotplug.loop);

    return true;
}

static void PIPEWIRECAMERA_CloseDevice(SDL_Camera *device)
{
    if (!device) {
        return;
    }

    PIPEWIRE_pw_thread_loop_lock(hotplug.loop);
    if (device->hidden) {
        if (device->hidden->stream)
            PIPEWIRE_pw_stream_destroy(device->hidden->stream);
        pw_array_clear(&device->hidden->buffers);
        SDL_free(device->hidden);
        device->hidden = NULL;
    }
    PIPEWIRE_pw_thread_loop_unlock(hotplug.loop);
}

static bool PIPEWIRECAMERA_WaitDevice(SDL_Camera *device)
{
    PIPEWIRE_pw_thread_loop_lock(hotplug.loop);
    PIPEWIRE_pw_thread_loop_wait(hotplug.loop);
    PIPEWIRE_pw_thread_loop_unlock(hotplug.loop);
    return true;
}

static SDL_CameraFrameResult PIPEWIRECAMERA_AcquireFrame(SDL_Camera *device, SDL_Surface *frame, Uint64 *timestampNS)
{
    struct pw_buffer *b;

    PIPEWIRE_pw_thread_loop_lock(hotplug.loop);
    b = NULL;
    while (true) {
        struct pw_buffer *t;
        if ((t = PIPEWIRE_pw_stream_dequeue_buffer(device->hidden->stream)) == NULL)
            break;
        if (b)
            PIPEWIRE_pw_stream_queue_buffer(device->hidden->stream, b);
        b = t;
    }
    if (b == NULL) {
        PIPEWIRE_pw_thread_loop_unlock(hotplug.loop);
        return SDL_CAMERA_FRAME_SKIP;
    }

#if PW_CHECK_VERSION(1,0,5)
    *timestampNS = hotplug.have_1_0_5 ? b->time : SDL_GetTicksNS();
#else
    *timestampNS = SDL_GetTicksNS();
#endif
    frame->pixels = b->buffer->datas[0].data;
    if (frame->format == SDL_PIXELFORMAT_MJPG) {
        frame->pitch = b->buffer->datas[0].chunk->size;
    } else {
        frame->pitch = b->buffer->datas[0].chunk->stride;
    }

    PIPEWIRE_pw_thread_loop_unlock(hotplug.loop);

    return SDL_CAMERA_FRAME_READY;
}

static void PIPEWIRECAMERA_ReleaseFrame(SDL_Camera *device, SDL_Surface *frame)
{
    struct pw_buffer **p;
    PIPEWIRE_pw_thread_loop_lock(hotplug.loop);
    pw_array_for_each(p, &device->hidden->buffers) {
        if ((*p)->buffer->datas[0].data == frame->pixels) {
            PIPEWIRE_pw_stream_queue_buffer(device->hidden->stream, (*p));
            break;
        }
    }
    PIPEWIRE_pw_thread_loop_unlock(hotplug.loop);
}

static void collect_rates(CameraFormatAddData *data, struct param *p, SDL_PixelFormat sdlfmt, SDL_Colorspace colorspace, const struct spa_rectangle *size)
{
    const struct spa_pod_prop *prop;
    struct spa_pod * values;
    uint32_t i, n_vals, choice;
    struct spa_fraction *rates;

    prop = spa_pod_find_prop(p->param, NULL, SPA_FORMAT_VIDEO_framerate);
    if (prop == NULL)
        return;

    values = spa_pod_get_values(&prop->value, &n_vals, &choice);
    if (values->type != SPA_TYPE_Fraction || n_vals == 0)
        return;

    rates = SPA_POD_BODY(values);
    switch (choice) {
    case SPA_CHOICE_None:
        n_vals = 1;
    SDL_FALLTHROUGH;
    case SPA_CHOICE_Enum:
        for (i = 0; i < n_vals; i++) {
            if (!SDL_AddCameraFormat(data, sdlfmt, colorspace, size->width, size->height, rates[i].num, rates[i].denom)) {
                return;  // Probably out of memory; we'll go with what we have, if anything.
            }
        }
        break;
    default:
        SDL_Log("CAMERA: unimplemented choice:%d", choice);
        break;
    }
}

static void collect_size(CameraFormatAddData *data, struct param *p, SDL_PixelFormat sdlfmt, SDL_Colorspace colorspace)
{
    const struct spa_pod_prop *prop;
    struct spa_pod * values;
    uint32_t i, n_vals, choice;
    struct spa_rectangle *rectangles;

    prop = spa_pod_find_prop(p->param, NULL, SPA_FORMAT_VIDEO_size);
    if (prop == NULL)
        return;

    values = spa_pod_get_values(&prop->value, &n_vals, &choice);
    if (values->type != SPA_TYPE_Rectangle || n_vals == 0)
        return;

    rectangles = SPA_POD_BODY(values);
    switch (choice) {
    case SPA_CHOICE_None:
        n_vals = 1;
    SDL_FALLTHROUGH;
    case SPA_CHOICE_Enum:
        for (i = 0; i < n_vals; i++) {
            collect_rates(data, p, sdlfmt, colorspace, &rectangles[i]);
        }
        break;
    default:
        SDL_Log("CAMERA: unimplemented choice:%d", choice);
        break;
    }
}

static void collect_raw(CameraFormatAddData *data, struct param *p)
{
    const struct spa_pod_prop *prop;
    SDL_PixelFormat sdlfmt;
    SDL_Colorspace colorspace;
    struct spa_pod * values;
    uint32_t i, n_vals, choice, *ids;

    prop = spa_pod_find_prop(p->param, NULL, SPA_FORMAT_VIDEO_format);
    if (prop == NULL)
        return;

    values = spa_pod_get_values(&prop->value, &n_vals, &choice);
    if (values->type != SPA_TYPE_Id || n_vals == 0)
        return;

    ids = SPA_POD_BODY(values);
    switch (choice) {
    case SPA_CHOICE_None:
        n_vals = 1;
	SDL_FALLTHROUGH;
    case SPA_CHOICE_Enum:
        for (i = 0; i < n_vals; i++) {
            id_to_sdl_format(ids[i], &sdlfmt, &colorspace);
            if (sdlfmt == SDL_PIXELFORMAT_UNKNOWN) {
                continue;
            }
            collect_size(data, p, sdlfmt, colorspace);
        }
        break;
    default:
        SDL_Log("CAMERA: unimplemented choice: %d", choice);
        break;
    }
}

static void collect_format(CameraFormatAddData *data, struct param *p)
{
    const struct spa_pod_prop *prop;
    struct spa_pod * values;
    uint32_t i, n_vals, choice, *ids;

    prop = spa_pod_find_prop(p->param, NULL, SPA_FORMAT_mediaSubtype);
    if (prop == NULL)
        return;

    values = spa_pod_get_values(&prop->value, &n_vals, &choice);
    if (values->type != SPA_TYPE_Id || n_vals == 0)
        return;

    ids = SPA_POD_BODY(values);
    switch (choice) {
    case SPA_CHOICE_None:
        n_vals = 1;
    SDL_FALLTHROUGH;
    case SPA_CHOICE_Enum:
        for (i = 0; i < n_vals; i++) {
            switch (ids[i]) {
            case SPA_MEDIA_SUBTYPE_raw:
                collect_raw(data, p);
                break;
            case SPA_MEDIA_SUBTYPE_mjpg:
                collect_size(data, p, SDL_PIXELFORMAT_MJPG, SDL_COLORSPACE_JPEG);
                break;
            default:
                // Unsupported format
                break;
            }
        }
        break;
    default:
        SDL_Log("CAMERA: unimplemented choice: %d", choice);
        break;
    }
}

static void add_device(struct global *g)
{
    struct param *p;
    CameraFormatAddData data;

    SDL_zero(data);

    spa_list_for_each(p, &g->param_list, link) {
        if (p->id != SPA_PARAM_EnumFormat)
            continue;

        collect_format(&data, p);
    }
    if (data.num_specs > 0) {
        SDL_AddCamera(g->name, SDL_CAMERA_POSITION_UNKNOWN,
				    data.num_specs, data.specs, g);
    }
    SDL_free(data.specs);

    g->added = true;
}

static void PIPEWIRECAMERA_DetectDevices(void)
{
    struct global *g;

    PIPEWIRE_pw_thread_loop_lock(hotplug.loop);

    // Wait until the initial registry enumeration is complete
    while (!hotplug.init_complete) {
        PIPEWIRE_pw_thread_loop_wait(hotplug.loop);
    }

    spa_list_for_each (g, &hotplug.global_list, link) {
	    if (!g->added) {
                add_device(g);
	    }
    }

    hotplug.events_enabled = true;

    PIPEWIRE_pw_thread_loop_unlock(hotplug.loop);
}

static void PIPEWIRECAMERA_FreeDeviceHandle(SDL_Camera *device)
{
}

static void do_resync(void)
{
    hotplug.pending_seq = pw_core_sync(hotplug.core, PW_ID_CORE, 0);
}

/** node */
static void node_event_info(void *object, const struct pw_node_info *info)
{
    struct global *g = object;
    uint32_t i;

    info = g->info = PIPEWIRE_pw_node_info_merge(g->info, info, g->changed == 0);
    if (info == NULL)
        return;

    if (info->change_mask & PW_NODE_CHANGE_MASK_PARAMS) {
        for (i = 0; i < info->n_params; i++) {
            uint32_t id = info->params[i].id;
            int res;

            if (info->params[i].user == 0)
                continue;
            info->params[i].user = 0;

	    if (id != SPA_PARAM_EnumFormat)
		    continue;

            param_add(&g->pending_list, SPA_PARAMS_INFO_SEQ(info->params[i]), id, NULL);
            if (!(info->params[i].flags & SPA_PARAM_INFO_READ))
                continue;

            res = pw_node_enum_params((struct pw_node*)g->proxy,
                        ++SPA_PARAMS_INFO_SEQ(info->params[i]), id, 0, -1, NULL);
            if (SPA_RESULT_IS_ASYNC(res))
                SPA_PARAMS_INFO_SEQ(info->params[i]) = res;

	    g->changed++;
        }
    }
    do_resync();
}

static void node_event_param(void *object, int seq,
                uint32_t id, uint32_t index, uint32_t next,
                const struct spa_pod *param)
{
    struct global *g = object;
    param_add(&g->pending_list, seq, id, param);
}

static const struct pw_node_events node_events = {
    .version = PW_VERSION_NODE_EVENTS,
    .info = node_event_info,
    .param = node_event_param,
};

static void node_destroy(struct global *g)
{
    if (g->info) {
        PIPEWIRE_pw_node_info_free(g->info);
        g->info = NULL;
    }
}


static const struct global_class node_class = {
    .type = PW_TYPE_INTERFACE_Node,
    .version = PW_VERSION_NODE,
    .events = &node_events,
    .destroy = node_destroy,
};

/** proxy */
static void proxy_removed(void *data)
{
    struct global *g = data;
    PIPEWIRE_pw_proxy_destroy(g->proxy);
}

static void proxy_destroy(void *data)
{
    struct global *g = data;
    spa_list_remove(&g->link);
    g->proxy = NULL;
    if (g->class) {
        if (g->class->events)
            spa_hook_remove(&g->object_listener);
        if (g->class->destroy)
            g->class->destroy(g);
    }
    param_clear(&g->param_list, SPA_ID_INVALID);
    param_clear(&g->pending_list, SPA_ID_INVALID);
    free(g->name); // This should NOT be SDL_free()
}

static const struct pw_proxy_events proxy_events = {
    .version = PW_VERSION_PROXY_EVENTS,
    .removed = proxy_removed,
    .destroy = proxy_destroy
};

// called with thread_loop lock
static void hotplug_registry_global_callback(void *object, uint32_t id,
		uint32_t permissions, const char *type, uint32_t version,
		const struct spa_dict *props)
{
    const struct global_class *class = NULL;
    struct pw_proxy *proxy;
    const char *str, *name = NULL;

    if (spa_streq(type, PW_TYPE_INTERFACE_Node)) {
        if (props == NULL)
            return;
        if (((str = spa_dict_lookup(props, PW_KEY_MEDIA_CLASS)) == NULL) ||
            (!spa_streq(str, "Video/Source")))
            return;

        if ((name = spa_dict_lookup(props, PW_KEY_NODE_DESCRIPTION)) == NULL &&
            (name = spa_dict_lookup(props, PW_KEY_NODE_NAME)) == NULL)
		name = "unnamed camera";

        class = &node_class;
    }
    if (class) {
        struct global *g;

        proxy = pw_registry_bind(hotplug.registry,
                            id, class->type, class->version,
                            sizeof(struct global));

        g = PIPEWIRE_pw_proxy_get_user_data(proxy);
        g->class = class;
        g->id = id;
        g->permissions = permissions;
        g->props = props ? PIPEWIRE_pw_properties_new_dict(props) : NULL;
        g->proxy = proxy;
        g->name = strdup(name);
        spa_list_init(&g->pending_list);
        spa_list_init(&g->param_list);
        spa_list_append(&hotplug.global_list, &g->link);

        PIPEWIRE_pw_proxy_add_listener(proxy,
                            &g->proxy_listener,
                            &proxy_events, g);

        if (class->events) {
            PIPEWIRE_pw_proxy_add_object_listener(proxy,
                                &g->object_listener,
                                class->events, g);
        }
        if (class->init)
            class->init(g);

        do_resync();
    }
}

// called with thread_loop lock
static void hotplug_registry_global_remove_callback(void *object, uint32_t id)
{
}

static const struct pw_registry_events hotplug_registry_events =
{
    .version = PW_VERSION_REGISTRY_EVENTS,
    .global = hotplug_registry_global_callback,
    .global_remove = hotplug_registry_global_remove_callback
};

static void parse_version(const char *str, int *major, int *minor, int *patch)
{
    if (SDL_sscanf(str, "%d.%d.%d", major, minor, patch) < 3) {
        *major = 0;
        *minor = 0;
        *patch = 0;
    }
}

// Core info, called with thread_loop lock
static void hotplug_core_info_callback(void *data, const struct pw_core_info *info)
{
    parse_version(info->version, &hotplug.server_major, &hotplug.server_minor, &hotplug.server_patch);
}

// Core sync points, called with thread_loop lock
static void hotplug_core_done_callback(void *object, uint32_t id, int seq)
{
    hotplug.last_seq = seq;
    if (id == PW_ID_CORE && seq == hotplug.pending_seq) {
        struct global *g;
        struct pw_node_info *info;

        spa_list_for_each(g, &hotplug.global_list, link) {
             if (!g->changed)
		     continue;

	     info = g->info;
             param_update(&g->param_list, &g->pending_list, info->n_params, info->params);

	     if (!g->added && hotplug.events_enabled) {
                 add_device(g);
	     }
        }
	hotplug.init_complete = true;
        PIPEWIRE_pw_thread_loop_signal(hotplug.loop, false);
    }
}
static const struct pw_core_events hotplug_core_events =
{
    .version = PW_VERSION_CORE_EVENTS,
    .info = hotplug_core_info_callback,
    .done = hotplug_core_done_callback
};

/* When in a container, the library version can differ from the underlying core version,
 * so make sure the underlying Pipewire implementation meets the version requirement.
 */
static bool pipewire_server_version_at_least(int major, int minor, int patch)
{
    return (hotplug.server_major >= major) &&
           (hotplug.server_major > major || hotplug.server_minor >= minor) &&
           (hotplug.server_major > major || hotplug.server_minor > minor || hotplug.server_patch >= patch);
}

// The hotplug thread
static bool hotplug_loop_init(void)
{
    int res;
#ifdef SDL_USE_LIBDBUS
    int fd;

    fd = SDL_DBus_CameraPortalRequestAccess();
    if (fd == -1)
        return false;
#endif

    spa_list_init(&hotplug.global_list);

#if PW_CHECK_VERSION(0, 3, 75)
    hotplug.have_1_0_5 = PIPEWIRE_pw_check_library_version(1,0,5);
#else
    hotplug.have_1_0_5 = false;
#endif

    hotplug.loop = PIPEWIRE_pw_thread_loop_new("SDLPwCameraPlug", NULL);
    if (!hotplug.loop) {
        return SDL_SetError("Pipewire: Failed to create hotplug detection loop (%i)", errno);
    }

    hotplug.context = PIPEWIRE_pw_context_new(PIPEWIRE_pw_thread_loop_get_loop(hotplug.loop), NULL, 0);
    if (!hotplug.context) {
        return SDL_SetError("Pipewire: Failed to create hotplug detection context (%i)", errno);
    }
#ifdef SDL_USE_LIBDBUS
    if (fd >= 0) {
        hotplug.core = PIPEWIRE_pw_context_connect_fd(hotplug.context, fd, NULL, 0);
    } else {
        hotplug.core = PIPEWIRE_pw_context_connect(hotplug.context, NULL, 0);
    }
#else
    hotplug.core = PIPEWIRE_pw_context_connect(hotplug.context, NULL, 0);
#endif
    if (!hotplug.core) {
        return SDL_SetError("Pipewire: Failed to connect hotplug detection context (%i)", errno);
    }
    spa_zero(hotplug.core_listener);
    pw_core_add_listener(hotplug.core, &hotplug.core_listener, &hotplug_core_events, NULL);

    hotplug.registry = pw_core_get_registry(hotplug.core, PW_VERSION_REGISTRY, 0);
    if (!hotplug.registry) {
        return SDL_SetError("Pipewire: Failed to acquire hotplug detection registry (%i)", errno);
    }

    spa_zero(hotplug.registry_listener);
    pw_registry_add_listener(hotplug.registry, &hotplug.registry_listener, &hotplug_registry_events, NULL);

    do_resync();

    res = PIPEWIRE_pw_thread_loop_start(hotplug.loop);
    if (res != 0) {
        return SDL_SetError("Pipewire: Failed to start hotplug detection loop");
    }

    PIPEWIRE_pw_thread_loop_lock(hotplug.loop);
    while (!hotplug.init_complete) {
        PIPEWIRE_pw_thread_loop_wait(hotplug.loop);
    }
    PIPEWIRE_pw_thread_loop_unlock(hotplug.loop);

    if (!pipewire_server_version_at_least(PW_REQUIRED_MAJOR, PW_REQUIRED_MINOR, PW_REQUIRED_PATCH)) {
        return SDL_SetError("Pipewire: server version is too old %d.%d.%d < %d.%d.%d",
			hotplug.server_major, hotplug.server_minor, hotplug.server_patch,
                    PW_REQUIRED_MAJOR, PW_REQUIRED_MINOR, PW_REQUIRED_PATCH);
    }

    return true;
}


static void PIPEWIRECAMERA_Deinitialize(void)
{
    if (pipewire_initialized) {
        if (hotplug.loop) {
            PIPEWIRE_pw_thread_loop_lock(hotplug.loop);
        }
        if (hotplug.registry) {
            spa_hook_remove(&hotplug.registry_listener);
            PIPEWIRE_pw_proxy_destroy((struct pw_proxy *)hotplug.registry);
        }
        if (hotplug.core) {
            spa_hook_remove(&hotplug.core_listener);
            PIPEWIRE_pw_core_disconnect(hotplug.core);
        }
        if (hotplug.context) {
            PIPEWIRE_pw_context_destroy(hotplug.context);
        }
        if (hotplug.loop) {
            PIPEWIRE_pw_thread_loop_unlock(hotplug.loop);
            PIPEWIRE_pw_thread_loop_destroy(hotplug.loop);
        }
        deinit_pipewire_library();
        spa_zero(hotplug);
        pipewire_initialized = false;
    }
}

static bool PIPEWIRECAMERA_Init(SDL_CameraDriverImpl *impl)
{
    if (!pipewire_initialized) {

        if (!init_pipewire_library()) {
            return false;
        }

        pipewire_initialized = true;

        if (!hotplug_loop_init()) {
            PIPEWIRECAMERA_Deinitialize();
            return false;
        }
    }

    impl->DetectDevices = PIPEWIRECAMERA_DetectDevices;
    impl->OpenDevice = PIPEWIRECAMERA_OpenDevice;
    impl->CloseDevice = PIPEWIRECAMERA_CloseDevice;
    impl->WaitDevice = PIPEWIRECAMERA_WaitDevice;
    impl->AcquireFrame = PIPEWIRECAMERA_AcquireFrame;
    impl->ReleaseFrame = PIPEWIRECAMERA_ReleaseFrame;
    impl->FreeDeviceHandle = PIPEWIRECAMERA_FreeDeviceHandle;
    impl->Deinitialize = PIPEWIRECAMERA_Deinitialize;

    return true;
}

CameraBootStrap PIPEWIRECAMERA_bootstrap = {
    "pipewire", "SDL PipeWire camera driver", PIPEWIRECAMERA_Init, false
};

#endif  // SDL_CAMERA_DRIVER_PIPEWIRE
