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

#include "SDL_syscamera.h"
#include "SDL_camera_c.h"
#include "../video/SDL_pixels_c.h"
#include "../video/SDL_surface_c.h"
#include "../thread/SDL_systhread.h"


// A lot of this is a simplified version of SDL_audio.c; if fixing stuff here,
//  maybe check that file, too.

// Available camera drivers
static const CameraBootStrap *const bootstrap[] = {
#ifdef SDL_CAMERA_DRIVER_V4L2
    &V4L2_bootstrap,
#endif
#ifdef SDL_CAMERA_DRIVER_PIPEWIRE
    &PIPEWIRECAMERA_bootstrap,
#endif
#ifdef SDL_CAMERA_DRIVER_COREMEDIA
    &COREMEDIA_bootstrap,
#endif
#ifdef SDL_CAMERA_DRIVER_ANDROID
    &ANDROIDCAMERA_bootstrap,
#endif
#ifdef SDL_CAMERA_DRIVER_EMSCRIPTEN
    &EMSCRIPTENCAMERA_bootstrap,
#endif
#ifdef SDL_CAMERA_DRIVER_MEDIAFOUNDATION
    &MEDIAFOUNDATION_bootstrap,
#endif
#ifdef SDL_CAMERA_DRIVER_VITA
    &VITACAMERA_bootstrap,
#endif
#ifdef SDL_CAMERA_DRIVER_DUMMY
    &DUMMYCAMERA_bootstrap,
#endif
    NULL
};

static SDL_CameraDriver camera_driver;


int SDL_GetNumCameraDrivers(void)
{
    return SDL_arraysize(bootstrap) - 1;
}

const char *SDL_GetCameraDriver(int index)
{
    if (index >= 0 && index < SDL_GetNumCameraDrivers()) {
        return bootstrap[index]->name;
    }
    SDL_InvalidParamError("index");
    return NULL;
}

const char *SDL_GetCurrentCameraDriver(void)
{
    return camera_driver.name;
}

char *SDL_GetCameraThreadName(SDL_Camera *device, char *buf, size_t buflen)
{
    (void)SDL_snprintf(buf, buflen, "SDLCamera%d", (int) device->instance_id);
    return buf;
}

bool SDL_AddCameraFormat(CameraFormatAddData *data, SDL_PixelFormat format, SDL_Colorspace colorspace, int w, int h, int framerate_numerator, int framerate_denominator)
{
    SDL_assert(data != NULL);
    if (data->allocated_specs <= data->num_specs) {
        const int newalloc = data->allocated_specs ? (data->allocated_specs * 2) : 16;
        void *ptr = SDL_realloc(data->specs, sizeof (SDL_CameraSpec) * newalloc);
        if (!ptr) {
            return false;
        }
        data->specs = (SDL_CameraSpec *) ptr;
        data->allocated_specs = newalloc;
    }

    SDL_CameraSpec *spec = &data->specs[data->num_specs];
    spec->format = format;
    spec->colorspace = colorspace;
    spec->width = w;
    spec->height = h;
    spec->framerate_numerator = framerate_numerator;
    spec->framerate_denominator = framerate_denominator;

    data->num_specs++;

    return true;
}


// Zombie device implementation...

// These get used when a device is disconnected or fails. Apps that ignore the
//  loss notifications will get black frames but otherwise keep functioning.
static bool ZombieWaitDevice(SDL_Camera *device)
{
    if (!SDL_GetAtomicInt(&device->shutdown)) {
        // !!! FIXME: this is bad for several reasons (uses double, could be precalculated, doesn't track elapsed time).
        const double duration = ((double) device->actual_spec.framerate_denominator / ((double) device->actual_spec.framerate_numerator));
        SDL_Delay((Uint32) (duration * 1000.0));
    }
    return true;
}

static size_t GetFrameBufLen(const SDL_CameraSpec *spec)
{
    const size_t w = (const size_t) spec->width;
    const size_t h = (const size_t) spec->height;
    const size_t wxh = w * h;
    const SDL_PixelFormat fmt = spec->format;

    switch (fmt) {
        // Some YUV formats have a larger Y plane than their U or V planes.
        case SDL_PIXELFORMAT_YV12:
        case SDL_PIXELFORMAT_IYUV:
        case SDL_PIXELFORMAT_NV12:
        case SDL_PIXELFORMAT_NV21:
            return wxh + (wxh / 2);

        default: break;
    }

    // this is correct for most things.
    return wxh * SDL_BYTESPERPIXEL(fmt);
}

static SDL_CameraFrameResult ZombieAcquireFrame(SDL_Camera *device, SDL_Surface *frame, Uint64 *timestampNS)
{
    const SDL_CameraSpec *spec = &device->actual_spec;

    if (!device->zombie_pixels) {
        // attempt to allocate and initialize a fake frame of pixels.
        const size_t buflen = GetFrameBufLen(&device->actual_spec);
        device->zombie_pixels = (Uint8 *)SDL_aligned_alloc(SDL_GetSIMDAlignment(), buflen);
        if (!device->zombie_pixels) {
            *timestampNS = 0;
            return SDL_CAMERA_FRAME_SKIP; // oh well, say there isn't a frame yet, so we'll go back to waiting. Maybe allocation will succeed later...?
        }

        Uint8 *dst = device->zombie_pixels;
        switch (spec->format) {
            // in YUV formats, the U and V values must be 128 to get a black frame. If set to zero, it'll be bright green.
            case SDL_PIXELFORMAT_YV12:
            case SDL_PIXELFORMAT_IYUV:
            case SDL_PIXELFORMAT_NV12:
            case SDL_PIXELFORMAT_NV21:
                SDL_memset(dst, 0, spec->width * spec->height);  // set Y to zero.
                SDL_memset(dst + (spec->width * spec->height), 128, (spec->width * spec->height) / 2); // set U and V to 128.
                break;

            case SDL_PIXELFORMAT_YUY2:
            case SDL_PIXELFORMAT_YVYU:
                // Interleaved Y1[U1|V1]Y2[U2|V2].
                for (size_t i = 0; i < buflen; i += 4) {
                    dst[i] = 0;
                    dst[i+1] = 128;
                    dst[i+2] = 0;
                    dst[i+3] = 128;
                }
                break;


            case SDL_PIXELFORMAT_UYVY:
                 // Interleaved [U1|V1]Y1[U2|V2]Y2.
                for (size_t i = 0; i < buflen; i += 4) {
                    dst[i] = 128;
                    dst[i+1] = 0;
                    dst[i+2] = 128;
                    dst[i+3] = 0;
                }
                break;

            default:
                // just zero everything else, it'll _probably_ be okay.
                SDL_memset(dst, 0, buflen);
                break;
        }
    }


    *timestampNS = SDL_GetTicksNS();
    frame->pixels = device->zombie_pixels;

    // SDL (currently) wants the pitch of YUV formats to be the pitch of the (1-byte-per-pixel) Y plane.
    frame->pitch = spec->width;
    if (!SDL_ISPIXELFORMAT_FOURCC(spec->format)) {  // checking if it's not FOURCC to only do this for non-YUV data is good enough for now.
        frame->pitch *= SDL_BYTESPERPIXEL(spec->format);
    }

    #if DEBUG_CAMERA
    SDL_Log("CAMERA: dev[%p] Acquired Zombie frame, timestamp %llu", device, (unsigned long long) *timestampNS);
    #endif

    return SDL_CAMERA_FRAME_READY; // frame is available.
}

static void ZombieReleaseFrame(SDL_Camera *device, SDL_Surface *frame) // Reclaim frame->pixels and frame->pitch!
{
    if (frame->pixels != device->zombie_pixels) {
        // this was a frame from before the disconnect event; let the backend make an attempt to free it.
        camera_driver.impl.ReleaseFrame(device, frame);
    }
    // we just leave zombie_pixels alone, as we'll reuse it for every new frame until the camera is closed.
}

static void ClosePhysicalCamera(SDL_Camera *device)
{
    if (!device) {
        return;
    }

    SDL_SetAtomicInt(&device->shutdown, 1);

// !!! FIXME: the close_cond stuff from audio might help the race condition here.

    if (device->thread != NULL) {
        SDL_WaitThread(device->thread, NULL);
        device->thread = NULL;
    }

    // release frames that are queued up somewhere...
    if (!device->needs_conversion && !device->needs_scaling) {
        for (SurfaceList *i = device->filled_output_surfaces.next; i != NULL; i = i->next) {
            device->ReleaseFrame(device, i->surface);
        }
        for (SurfaceList *i = device->app_held_output_surfaces.next; i != NULL; i = i->next) {
            device->ReleaseFrame(device, i->surface);
        }
    }

    camera_driver.impl.CloseDevice(device);

    SDL_DestroyProperties(device->props);

    SDL_DestroySurface(device->acquire_surface);
    device->acquire_surface = NULL;
    SDL_DestroySurface(device->conversion_surface);
    device->conversion_surface = NULL;

    for (int i = 0; i < SDL_arraysize(device->output_surfaces); i++) {
        SDL_DestroySurface(device->output_surfaces[i].surface);
    }
    SDL_zeroa(device->output_surfaces);

    SDL_aligned_free(device->zombie_pixels);

    device->permission = 0;
    device->zombie_pixels = NULL;
    device->filled_output_surfaces.next = NULL;
    device->empty_output_surfaces.next = NULL;
    device->app_held_output_surfaces.next = NULL;

    device->base_timestamp = 0;
    device->adjust_timestamp = 0;

    SDL_zero(device->spec);
}

// Don't hold the device lock when calling this, as we may destroy the device!
void UnrefPhysicalCamera(SDL_Camera *device)
{
    if (SDL_AtomicDecRef(&device->refcount)) {
        // take it out of the device list.
        SDL_LockRWLockForWriting(camera_driver.device_hash_lock);
        if (SDL_RemoveFromHashTable(camera_driver.device_hash, (const void *) (uintptr_t) device->instance_id)) {
            SDL_AddAtomicInt(&camera_driver.device_count, -1);
        }
        SDL_UnlockRWLock(camera_driver.device_hash_lock);
    }
}

void RefPhysicalCamera(SDL_Camera *device)
{
    SDL_AtomicIncRef(&device->refcount);
}

static void ObtainPhysicalCameraObj(SDL_Camera *device) SDL_NO_THREAD_SAFETY_ANALYSIS  // !!! FIXME: SDL_ACQUIRE
{
    if (device) {
        RefPhysicalCamera(device);
        SDL_LockMutex(device->lock);
    }
}

static SDL_Camera *ObtainPhysicalCamera(SDL_CameraID devid)  // !!! FIXME: SDL_ACQUIRE
{
    if (!SDL_GetCurrentCameraDriver()) {
        SDL_SetError("Camera subsystem is not initialized");
        return NULL;
    }

    SDL_Camera *device = NULL;
    SDL_LockRWLockForReading(camera_driver.device_hash_lock);
    SDL_FindInHashTable(camera_driver.device_hash, (const void *) (uintptr_t) devid, (const void **) &device);
    SDL_UnlockRWLock(camera_driver.device_hash_lock);
    if (!device) {
        SDL_SetError("Invalid camera device instance ID");
    } else {
        ObtainPhysicalCameraObj(device);
    }
    return device;
}

static void ReleaseCamera(SDL_Camera *device) SDL_NO_THREAD_SAFETY_ANALYSIS  // !!! FIXME: SDL_RELEASE
{
    if (device) {
        SDL_UnlockMutex(device->lock);
        UnrefPhysicalCamera(device);
    }
}

// we want these sorted by format first, so you can find a block of all
// resolutions that are supported for a format. The formats are sorted in
// "best" order, but that's subjective: right now, we prefer planar
// formats, since they're likely what the cameras prefer to produce
// anyhow, and they basically send the same information in less space
// than an RGB-style format. After that, sort by bits-per-pixel.

// we want specs sorted largest to smallest dimensions, larger width taking precedence over larger height.
static int SDLCALL CameraSpecCmp(const void *vpa, const void *vpb)
{
    const SDL_CameraSpec *a = (const SDL_CameraSpec *) vpa;
    const SDL_CameraSpec *b = (const SDL_CameraSpec *) vpb;

    // driver shouldn't send specs like this, check here since we're eventually going to sniff the whole array anyhow.
    SDL_assert(a->format != SDL_PIXELFORMAT_UNKNOWN);
    SDL_assert(a->width > 0);
    SDL_assert(a->height > 0);
    SDL_assert(b->format != SDL_PIXELFORMAT_UNKNOWN);
    SDL_assert(b->width > 0);
    SDL_assert(b->height > 0);

    const SDL_PixelFormat afmt = a->format;
    const SDL_PixelFormat bfmt = b->format;
    if (SDL_ISPIXELFORMAT_FOURCC(afmt) && !SDL_ISPIXELFORMAT_FOURCC(bfmt)) {
        return -1;
    } else if (!SDL_ISPIXELFORMAT_FOURCC(afmt) && SDL_ISPIXELFORMAT_FOURCC(bfmt)) {
        return 1;
    } else if (SDL_BITSPERPIXEL(afmt) > SDL_BITSPERPIXEL(bfmt)) {
        return -1;
    } else if (SDL_BITSPERPIXEL(bfmt) > SDL_BITSPERPIXEL(afmt)) {
        return 1;
    } else if (a->width > b->width) {
        return -1;
    } else if (b->width > a->width) {
        return 1;
    } else if (a->height > b->height) {
        return -1;
    } else if (b->height > a->height) {
        return 1;
    }

    // still here? We care about framerate less than format or size, but faster is better than slow.
    if (a->framerate_numerator && !b->framerate_numerator) {
        return -1;
    } else if (!a->framerate_numerator && b->framerate_numerator) {
        return 1;
    }

    const float fpsa = ((float)a->framerate_numerator / a->framerate_denominator);
    const float fpsb = ((float)b->framerate_numerator / b->framerate_denominator);
    if (fpsa > fpsb) {
        return -1;
    } else if (fpsb > fpsa) {
        return 1;
    }

    if (SDL_COLORSPACERANGE(a->colorspace) == SDL_COLOR_RANGE_FULL &&
        SDL_COLORSPACERANGE(b->colorspace) != SDL_COLOR_RANGE_FULL) {
        return -1;
    }
    if (SDL_COLORSPACERANGE(a->colorspace) != SDL_COLOR_RANGE_FULL &&
        SDL_COLORSPACERANGE(b->colorspace) == SDL_COLOR_RANGE_FULL) {
        return 1;
    }

    return 0;  // apparently, they're equal.
}

// The camera backends call this when a new device is plugged in.
SDL_Camera *SDL_AddCamera(const char *name, SDL_CameraPosition position, int num_specs, const SDL_CameraSpec *specs, void *handle)
{
    SDL_assert(name != NULL);
    SDL_assert(num_specs >= 0);
    SDL_assert((specs != NULL) == (num_specs > 0));
    SDL_assert(handle != NULL);

    SDL_LockRWLockForReading(camera_driver.device_hash_lock);
    const int shutting_down = SDL_GetAtomicInt(&camera_driver.shutting_down);
    SDL_UnlockRWLock(camera_driver.device_hash_lock);
    if (shutting_down) {
        return NULL;  // we're shutting down, don't add any devices that are hotplugged at the last possible moment.
    }

    SDL_Camera *device = (SDL_Camera *)SDL_calloc(1, sizeof(SDL_Camera));
    if (!device) {
        return NULL;
    }

    device->name = SDL_strdup(name);
    if (!device->name) {
        SDL_free(device);
        return NULL;
    }

    device->position = position;

    device->lock = SDL_CreateMutex();
    if (!device->lock) {
        SDL_free(device->name);
        SDL_free(device);
        return NULL;
    }

    device->all_specs = (SDL_CameraSpec *)SDL_calloc(num_specs + 1, sizeof (*specs));
    if (!device->all_specs) {
        SDL_DestroyMutex(device->lock);
        SDL_free(device->name);
        SDL_free(device);
        return NULL;
    }

    if (num_specs > 0) {
        SDL_memcpy(device->all_specs, specs, sizeof (*specs) * num_specs);
        SDL_qsort(device->all_specs, num_specs, sizeof (*specs), CameraSpecCmp);

        // weed out duplicates, just in case.
        for (int i = 0; i < num_specs; i++) {
            SDL_CameraSpec *a = &device->all_specs[i];
            SDL_CameraSpec *b = &device->all_specs[i + 1];
            if (SDL_memcmp(a, b, sizeof (*a)) == 0) {
                SDL_memmove(a, b, sizeof (*specs) * (num_specs - i));
                i--;
                num_specs--;
            }
        }
    }

    #if DEBUG_CAMERA
    const char *posstr = "unknown position";
    if (position == SDL_CAMERA_POSITION_FRONT_FACING) {
        posstr = "front-facing";
    } else if (position == SDL_CAMERA_POSITION_BACK_FACING) {
        posstr = "back-facing";
    }
    SDL_Log("CAMERA: Adding device '%s' (%s) with %d spec%s%s", name, posstr, num_specs, (num_specs == 1) ? "" : "s", (num_specs == 0) ? "" : ":");
    for (int i = 0; i < num_specs; i++) {
        const SDL_CameraSpec *spec = &device->all_specs[i];
        SDL_Log("CAMERA:   - fmt=%s, w=%d, h=%d, numerator=%d, denominator=%d", SDL_GetPixelFormatName(spec->format), spec->width, spec->height, spec->framerate_numerator, spec->framerate_denominator);
    }
    #endif

    device->num_specs = num_specs;
    device->handle = handle;
    device->instance_id = SDL_GetNextObjectID();
    SDL_SetAtomicInt(&device->shutdown, 0);
    SDL_SetAtomicInt(&device->zombie, 0);
    RefPhysicalCamera(device);

    SDL_LockRWLockForWriting(camera_driver.device_hash_lock);
    if (SDL_InsertIntoHashTable(camera_driver.device_hash, (const void *) (uintptr_t) device->instance_id, device, false)) {
        SDL_AddAtomicInt(&camera_driver.device_count, 1);
    } else {
        SDL_DestroyMutex(device->lock);
        SDL_free(device->all_specs);
        SDL_free(device->name);
        SDL_free(device);
        device = NULL;
    }

    // Add a device add event to the pending list, to be pushed when the event queue is pumped (away from any of our internal threads).
    if (device) {
        SDL_PendingCameraEvent *p = (SDL_PendingCameraEvent *) SDL_malloc(sizeof (SDL_PendingCameraEvent));
        if (p) {  // if allocation fails, you won't get an event, but we can't help that.
            p->type = SDL_EVENT_CAMERA_DEVICE_ADDED;
            p->devid = device->instance_id;
            p->next = NULL;
            SDL_assert(camera_driver.pending_events_tail != NULL);
            SDL_assert(camera_driver.pending_events_tail->next == NULL);
            camera_driver.pending_events_tail->next = p;
            camera_driver.pending_events_tail = p;
        }
    }
    SDL_UnlockRWLock(camera_driver.device_hash_lock);

    return device;
}

// Called when a device is removed from the system, or it fails unexpectedly, from any thread, possibly even the camera device's thread.
void SDL_CameraDisconnected(SDL_Camera *device)
{
    if (!device) {
        return;
    }

    #if DEBUG_CAMERA
    SDL_Log("CAMERA: DISCONNECTED! dev[%p]", device);
    #endif

    // Save off removal info in a list so we can send events for each, next
    //  time the event queue pumps, in case something tries to close a device
    //  from an event filter, as this would risk deadlocks and other disasters
    //  if done from the device thread.
    SDL_PendingCameraEvent pending;
    pending.next = NULL;
    SDL_PendingCameraEvent *pending_tail = &pending;

    ObtainPhysicalCameraObj(device);

    const bool first_disconnect = SDL_CompareAndSwapAtomicInt(&device->zombie, 0, 1);
    if (first_disconnect) {   // if already disconnected this device, don't do it twice.
        // Swap in "Zombie" versions of the usual platform interfaces, so the device will keep
        // making progress until the app closes it. Otherwise, streams might continue to
        // accumulate waste data that never drains, apps that depend on audio callbacks to
        // progress will freeze, etc.
        device->WaitDevice = ZombieWaitDevice;
        device->AcquireFrame = ZombieAcquireFrame;
        device->ReleaseFrame = ZombieReleaseFrame;

        // Zombie functions will just report the timestamp as SDL_GetTicksNS(), so we don't need to adjust anymore to get it to match.
        device->adjust_timestamp = 0;
        device->base_timestamp = 0;

        SDL_PendingCameraEvent *p = (SDL_PendingCameraEvent *) SDL_malloc(sizeof (SDL_PendingCameraEvent));
        if (p) {  // if this failed, no event for you, but you have deeper problems anyhow.
            p->type = SDL_EVENT_CAMERA_DEVICE_REMOVED;
            p->devid = device->instance_id;
            p->next = NULL;
            pending_tail->next = p;
            pending_tail = p;
        }
    }

    ReleaseCamera(device);

    if (first_disconnect) {
        if (pending.next) {  // NULL if event is disabled or disaster struck.
            SDL_LockRWLockForWriting(camera_driver.device_hash_lock);
            SDL_assert(camera_driver.pending_events_tail != NULL);
            SDL_assert(camera_driver.pending_events_tail->next == NULL);
            camera_driver.pending_events_tail->next = pending.next;
            camera_driver.pending_events_tail = pending_tail;
            SDL_UnlockRWLock(camera_driver.device_hash_lock);
        }
    }
}

void SDL_CameraPermissionOutcome(SDL_Camera *device, bool approved)
{
    if (!device) {
        return;
    }

    SDL_PendingCameraEvent pending;
    pending.next = NULL;
    SDL_PendingCameraEvent *pending_tail = &pending;

    const int permission = approved ? 1 : -1;

    ObtainPhysicalCameraObj(device);
    if (device->permission != permission) {
        device->permission = permission;
        SDL_PendingCameraEvent *p = (SDL_PendingCameraEvent *) SDL_malloc(sizeof (SDL_PendingCameraEvent));
        if (p) {  // if this failed, no event for you, but you have deeper problems anyhow.
            p->type = approved ? SDL_EVENT_CAMERA_DEVICE_APPROVED : SDL_EVENT_CAMERA_DEVICE_DENIED;
            p->devid = device->instance_id;
            p->next = NULL;
            pending_tail->next = p;
            pending_tail = p;
        }
    }

    ReleaseCamera(device);

    if (pending.next) {  // NULL if event is disabled or disaster struck.
        SDL_LockRWLockForWriting(camera_driver.device_hash_lock);
        SDL_assert(camera_driver.pending_events_tail != NULL);
        SDL_assert(camera_driver.pending_events_tail->next == NULL);
        camera_driver.pending_events_tail->next = pending.next;
        camera_driver.pending_events_tail = pending_tail;
        SDL_UnlockRWLock(camera_driver.device_hash_lock);
    }
}

typedef struct FindOnePhysicalCameraByCallbackData
{
    bool (*callback)(SDL_Camera *device, void *userdata);
    void *userdata;
    SDL_Camera *device;
} FindOnePhysicalCameraByCallbackData;

static bool SDLCALL FindOnePhysicalCameraByCallback(void *userdata, const SDL_HashTable *table, const void *key, const void *value)
{
    FindOnePhysicalCameraByCallbackData *data = (FindOnePhysicalCameraByCallbackData *) userdata;
    SDL_Camera *device = (SDL_Camera *) value;
    if (data->callback(device, data->userdata)) {
        data->device = device;
        return false;  // stop iterating.
    }
    return true;  // keep iterating.
}

// !!! FIXME: this doesn't follow SDL convention of `userdata` being the first param of the callback.
SDL_Camera *SDL_FindPhysicalCameraByCallback(bool (*callback)(SDL_Camera *device, void *userdata), void *userdata)
{
    if (!SDL_GetCurrentCameraDriver()) {
        SDL_SetError("Camera subsystem is not initialized");
        return NULL;
    }


    FindOnePhysicalCameraByCallbackData data = { callback, userdata, NULL };
    SDL_LockRWLockForReading(camera_driver.device_hash_lock);
    SDL_IterateHashTable(camera_driver.device_hash, FindOnePhysicalCameraByCallback, &data);
    SDL_UnlockRWLock(camera_driver.device_hash_lock);

    if (!data.device) {
        SDL_SetError("Device not found");
    }

    return data.device;
}

void SDL_CloseCamera(SDL_Camera *camera)
{
    SDL_Camera *device = camera;  // currently there's no separation between physical and logical device.
    ClosePhysicalCamera(device);
}

bool SDL_GetCameraFormat(SDL_Camera *camera, SDL_CameraSpec *spec)
{
    bool result;

    if (!camera) {
        return SDL_InvalidParamError("camera");
    } else if (!spec) {
        return SDL_InvalidParamError("spec");
    }

    SDL_Camera *device = camera;  // currently there's no separation between physical and logical device.
    ObtainPhysicalCameraObj(device);
    if (device->permission > 0) {
        SDL_copyp(spec, &device->spec);
        result = true;
    } else {
        SDL_zerop(spec);
        result = SDL_SetError("Camera permission has not been granted");
    }
    ReleaseCamera(device);

    return result;
}

const char *SDL_GetCameraName(SDL_CameraID instance_id)
{
    const char *result = NULL;
    SDL_Camera *device = ObtainPhysicalCamera(instance_id);
    if (device) {
        result = SDL_GetPersistentString(device->name);
        ReleaseCamera(device);
    }
    return result;
}

SDL_CameraPosition SDL_GetCameraPosition(SDL_CameraID instance_id)
{
    SDL_CameraPosition result = SDL_CAMERA_POSITION_UNKNOWN;
    SDL_Camera *device = ObtainPhysicalCamera(instance_id);
    if (device) {
        result = device->position;
        ReleaseCamera(device);
    }
    return result;
}


typedef struct GetOneCameraData
{
    SDL_CameraID *result;
    int devs_seen;
} GetOneCameraData;

static bool SDLCALL GetOneCamera(void *userdata, const SDL_HashTable *table, const void *key, const void *value)
{
    GetOneCameraData *data = (GetOneCameraData *) userdata;
    data->result[data->devs_seen++] = (SDL_CameraID) (uintptr_t) key;
    return true;  // keep iterating.
}

SDL_CameraID *SDL_GetCameras(int *count)
{
    int dummy_count;
    if (!count) {
        count = &dummy_count;
    }

    if (!SDL_GetCurrentCameraDriver()) {
        *count = 0;
        SDL_SetError("Camera subsystem is not initialized");
        return NULL;
    }

    SDL_CameraID *result = NULL;

    SDL_LockRWLockForReading(camera_driver.device_hash_lock);
    int num_devices = SDL_GetAtomicInt(&camera_driver.device_count);
    result = (SDL_CameraID *) SDL_malloc((num_devices + 1) * sizeof (SDL_CameraID));
    if (!result) {
        num_devices = 0;
    } else {
        GetOneCameraData data = { result, 0 };
        SDL_IterateHashTable(camera_driver.device_hash, GetOneCamera, &data);
        SDL_assert(data.devs_seen == num_devices);
        result[num_devices] = 0;  // null-terminated.
    }
    SDL_UnlockRWLock(camera_driver.device_hash_lock);

    *count = num_devices;

    return result;
}

SDL_CameraSpec **SDL_GetCameraSupportedFormats(SDL_CameraID instance_id, int *count)
{
    if (count) {
        *count = 0;
    }

    SDL_Camera *device = ObtainPhysicalCamera(instance_id);
    if (!device) {
        return NULL;
    }

    int i;
    int num_specs = device->num_specs;
    SDL_CameraSpec **result = (SDL_CameraSpec **) SDL_malloc(((num_specs + 1) * sizeof(*result)) + (num_specs * sizeof (**result)));
    if (result) {
        SDL_CameraSpec *specs = (SDL_CameraSpec *)(result + (num_specs + 1));
        SDL_memcpy(specs, device->all_specs, num_specs * sizeof(*specs));
        for (i = 0; i < num_specs; ++i) {
            result[i] = specs++;
        }
        result[i] = NULL;

        if (count) {
            *count = num_specs;
        }
    }

    ReleaseCamera(device);

    return result;
}


// Camera device thread. This is split into chunks, so drivers that need to control this directly can use the pieces they need without duplicating effort.

void SDL_CameraThreadSetup(SDL_Camera *device)
{
    //camera_driver.impl.ThreadInit(device);
#ifdef SDL_VIDEO_DRIVER_ANDROID
    // TODO
    /*
    {
        // Set thread priority to THREAD_PRIORITY_VIDEO
        extern void Android_JNI_CameraSetThreadPriority(int, int);
        Android_JNI_CameraSetThreadPriority(device->recording, device);
    }*/
#else
    // The camera capture is always a high priority thread
    SDL_SetCurrentThreadPriority(SDL_THREAD_PRIORITY_HIGH);
#endif
}

bool SDL_CameraThreadIterate(SDL_Camera *device)
{
    SDL_LockMutex(device->lock);

    if (SDL_GetAtomicInt(&device->shutdown)) {
        SDL_UnlockMutex(device->lock);
        return false;  // we're done, shut it down.
    }

    const int permission = device->permission;
    if (permission <= 0) {
        SDL_UnlockMutex(device->lock);
        return (permission < 0) ? false : true;  // if permission was denied, shut it down. if undecided, we're done for now.
    }

    bool failed = false;  // set to true if disaster worthy of treating the device as lost has happened.
    SDL_Surface *acquired = NULL;
    SDL_Surface *output_surface = NULL;
    SurfaceList *slist = NULL;
    Uint64 timestampNS = 0;

    // AcquireFrame SHOULD NOT BLOCK, as we are holding a lock right now. Block in WaitDevice instead!
    const SDL_CameraFrameResult rc = device->AcquireFrame(device, device->acquire_surface, &timestampNS);

    if (rc == SDL_CAMERA_FRAME_READY) {  // new frame acquired!
        #if DEBUG_CAMERA
        SDL_Log("CAMERA: New frame available! pixels=%p pitch=%d", device->acquire_surface->pixels, device->acquire_surface->pitch);
        #endif

        if (device->drop_frames > 0) {
            #if DEBUG_CAMERA
            SDL_Log("CAMERA: Dropping an initial frame");
            #endif
            device->drop_frames--;
            device->ReleaseFrame(device, device->acquire_surface);
            device->acquire_surface->pixels = NULL;
            device->acquire_surface->pitch = 0;
        } else if (device->empty_output_surfaces.next == NULL) {
            // uhoh, no output frames available! Either the app is slow, or it forgot to release frames when done with them. Drop this new frame.
            #if DEBUG_CAMERA
            SDL_Log("CAMERA: No empty output surfaces! Dropping frame!");
            #endif
            device->ReleaseFrame(device, device->acquire_surface);
            device->acquire_surface->pixels = NULL;
            device->acquire_surface->pitch = 0;
        } else {
            if (!device->adjust_timestamp) {
                device->adjust_timestamp = SDL_GetTicksNS();
                device->base_timestamp = timestampNS;
            }
            timestampNS = (timestampNS - device->base_timestamp) + device->adjust_timestamp;

            slist = device->empty_output_surfaces.next;
            output_surface = slist->surface;
            device->empty_output_surfaces.next = slist->next;
            acquired = device->acquire_surface;
            slist->timestampNS = timestampNS;
        }
    } else if (rc == SDL_CAMERA_FRAME_SKIP) {  // no frame available yet; not an error.
        #if 0 //DEBUG_CAMERA
        SDL_Log("CAMERA: No frame available yet.");
        #endif
    } else {  // fatal error!
        #if DEBUG_CAMERA
        SDL_Log("CAMERA: dev[%p] error AcquireFrame: %s", device, SDL_GetError());
        #endif
        failed = true;
    }

    // we can let go of the lock once we've tried to grab a frame of video and maybe moved the output frame off the empty list.
    // this lets us chew up the CPU for conversion and scaling without blocking other threads.
    SDL_UnlockMutex(device->lock);

    if (failed) {
        SDL_assert(slist == NULL);
        SDL_assert(acquired == NULL);
        SDL_CameraDisconnected(device);  // doh.
    } else if (acquired) {  // we have a new frame, scale/convert if necessary and queue it for the app!
        SDL_assert(slist != NULL);
        if (!device->needs_scaling && !device->needs_conversion) {  // no conversion needed? Just move the pointer/pitch into the output surface.
            #if DEBUG_CAMERA
            SDL_Log("CAMERA: Frame is going through without conversion!");
            #endif
            output_surface->w = acquired->w;
            output_surface->h = acquired->h;
            output_surface->pixels = acquired->pixels;
            output_surface->pitch = acquired->pitch;
        } else {  // convert/scale into a different surface.
            #if DEBUG_CAMERA
            SDL_Log("CAMERA: Frame is getting converted!");
            #endif
            SDL_Surface *srcsurf = acquired;
            if (device->needs_scaling == -1) {  // downscaling? Do it first.  -1: downscale, 0: no scaling, 1: upscale
                SDL_Surface *dstsurf = device->needs_conversion ? device->conversion_surface : output_surface;
                SDL_StretchSurface(srcsurf, NULL, dstsurf, NULL, SDL_SCALEMODE_NEAREST);  // !!! FIXME: linear scale? letterboxing?
                srcsurf = dstsurf;
            }
            if (device->needs_conversion) {
                SDL_Surface *dstsurf = (device->needs_scaling == 1) ? device->conversion_surface : output_surface;
                SDL_ConvertPixels(srcsurf->w, srcsurf->h,
                                  srcsurf->format, srcsurf->pixels, srcsurf->pitch,
                                  dstsurf->format, dstsurf->pixels, dstsurf->pitch);
                srcsurf = dstsurf;
            }
            if (device->needs_scaling == 1) {  // upscaling? Do it last.  -1: downscale, 0: no scaling, 1: upscale
                SDL_StretchSurface(srcsurf, NULL, output_surface, NULL, SDL_SCALEMODE_NEAREST);  // !!! FIXME: linear scale? letterboxing?
            }

            // we made a copy, so we can give the driver back its resources.
            device->ReleaseFrame(device, acquired);
        }

        // we either released these already after we copied the data, or the pointer was migrated to output_surface.
        acquired->pixels = NULL;
        acquired->pitch = 0;

        // make the filled output surface available to the app.
        SDL_LockMutex(device->lock);
        slist->next = device->filled_output_surfaces.next;
        device->filled_output_surfaces.next = slist;
        SDL_UnlockMutex(device->lock);
    }

    return true;  // always go on if not shutting down, even if device failed.
}

void SDL_CameraThreadShutdown(SDL_Camera *device)
{
    //device->FlushRecording(device);
    //camera_driver.impl.ThreadDeinit(device);
    //SDL_CameraThreadFinalize(device);
}

// Actual thread entry point, if driver didn't handle this itself.
static int SDLCALL CameraThread(void *devicep)
{
    SDL_Camera *device = (SDL_Camera *) devicep;

    #if DEBUG_CAMERA
    SDL_Log("CAMERA: dev[%p] Start thread 'CameraThread'", devicep);
    #endif

    SDL_assert(device != NULL);
    SDL_CameraThreadSetup(device);

    do {
        if (!device->WaitDevice(device)) {
            SDL_CameraDisconnected(device);  // doh. (but don't break out of the loop, just be a zombie for now!)
        }
    } while (SDL_CameraThreadIterate(device));

    SDL_CameraThreadShutdown(device);

    #if DEBUG_CAMERA
    SDL_Log("CAMERA: dev[%p] End thread 'CameraThread'", devicep);
    #endif

    return 0;
}

bool SDL_PrepareCameraSurfaces(SDL_Camera *device)
{
    SDL_CameraSpec *appspec = &device->spec;           // the app wants this format.
    const SDL_CameraSpec *devspec = &device->actual_spec;  // the hardware is set to this format.

    SDL_assert(device->acquire_surface == NULL);   // shouldn't call this function twice on an opened camera!
    SDL_assert(devspec->format != SDL_PIXELFORMAT_UNKNOWN);  // fix the backend, it should have an actual format by now.
    SDL_assert(devspec->width >= 0);  // fix the backend, it should have an actual format by now.
    SDL_assert(devspec->height >= 0);  // fix the backend, it should have an actual format by now.

    if (appspec->width <= 0 || appspec->height <= 0) {
        appspec->width = devspec->width;
        appspec->height = devspec->height;
    }

    if (appspec->format == SDL_PIXELFORMAT_UNKNOWN) {
        appspec->format = devspec->format;
    }

    if (appspec->framerate_denominator == 0) {
        appspec->framerate_numerator = devspec->framerate_numerator;
        appspec->framerate_denominator = devspec->framerate_denominator;
    }

    if ((devspec->width == appspec->width) && (devspec->height == appspec->height)) {
        device->needs_scaling = 0;
    } else {
        const Uint64 srcarea = ((Uint64) devspec->width) * ((Uint64) devspec->height);
        const Uint64 dstarea = ((Uint64) appspec->width) * ((Uint64) appspec->height);
        if (dstarea <= srcarea) {
            device->needs_scaling = -1;  // downscaling (or changing to new aspect ratio with same area)
        } else {
            device->needs_scaling = 1;  // upscaling
        }
    }

    device->needs_conversion = (devspec->format != appspec->format);

    device->acquire_surface = SDL_CreateSurfaceFrom(devspec->width, devspec->height, devspec->format, NULL, 0);
    if (!device->acquire_surface) {
        goto failed;
    }
    SDL_SetSurfaceColorspace(device->acquire_surface, devspec->colorspace);

    // if we have to scale _and_ convert, we need a middleman surface, since we can't do both changes at once.
    if (device->needs_scaling && device->needs_conversion) {
        const bool downscaling_first = (device->needs_scaling < 0);
        const SDL_CameraSpec *s = downscaling_first ? appspec : devspec;
        const SDL_PixelFormat fmt = downscaling_first ? devspec->format : appspec->format;
        device->conversion_surface = SDL_CreateSurface(s->width, s->height, fmt);
        if (!device->conversion_surface) {
            goto failed;
        }
        SDL_SetSurfaceColorspace(device->conversion_surface, devspec->colorspace);
    }

    // output surfaces are in the app-requested format. If no conversion is necessary, we'll just use the pointers
    // the backend fills into acquired_surface, and you can get all the way from DMA access in the camera hardware
    // to the app without a single copy. Otherwise, these will be full surfaces that hold converted/scaled copies.

    for (int i = 0; i < (SDL_arraysize(device->output_surfaces) - 1); i++) {
        device->output_surfaces[i].next = &device->output_surfaces[i + 1];
    }
    device->empty_output_surfaces.next = device->output_surfaces;

    for (int i = 0; i < SDL_arraysize(device->output_surfaces); i++) {
        SDL_Surface *surf;
        if (device->needs_scaling || device->needs_conversion) {
            surf = SDL_CreateSurface(appspec->width, appspec->height, appspec->format);
        } else {
            surf = SDL_CreateSurfaceFrom(appspec->width, appspec->height, appspec->format, NULL, 0);
        }
        if (!surf) {
            goto failed;
        }
        SDL_SetSurfaceColorspace(surf, devspec->colorspace);

        device->output_surfaces[i].surface = surf;
    }

    return true;

failed:
    if (device->acquire_surface) {
        SDL_DestroySurface(device->acquire_surface);
        device->acquire_surface = NULL;
    }

    if (device->conversion_surface) {
        SDL_DestroySurface(device->conversion_surface);
        device->conversion_surface = NULL;
    }

    for (int i = 0; i < SDL_arraysize(device->output_surfaces); i++) {
        SDL_Surface *surf = device->output_surfaces[i].surface;
        if (surf) {
            SDL_DestroySurface(surf);
        }
    }
    SDL_zeroa(device->output_surfaces);

    return false;
}

static void ChooseBestCameraSpec(SDL_Camera *device, const SDL_CameraSpec *spec, SDL_CameraSpec *closest)
{
    // Find the closest available native format/size...
    //
    // We want the exact size if possible, even if we have
    // to convert formats, because we can _probably_ do that
    // conversion losslessly at less expense verses scaling.
    //
    // Failing that, we want the size that's closest to the
    // requested aspect ratio, then the closest size within
    // that.

    SDL_zerop(closest);
    SDL_assert(((Uint32) SDL_PIXELFORMAT_UNKNOWN) == 0);  // since we SDL_zerop'd to this value.

    if (device->num_specs == 0) {  // device listed no specs! You get whatever you want!
        if (spec) {
            SDL_copyp(closest, spec);
        }
        return;
    } else if (!spec) {  // nothing specifically requested, get the best format we can...
        // we sorted this into the "best" format order when adding the camera.
        SDL_copyp(closest, &device->all_specs[0]);
    } else {  // specific thing requested, try to get as close to that as possible...
        const int num_specs = device->num_specs;
        int wantw = spec->width;
        int wanth = spec->height;

        if (wantw > 0 && wanth > 0) {
            // Find the sizes with the closest aspect ratio and then find the best fit of those.
            const float wantaspect = ((float)wantw) / ((float)wanth);
            const float epsilon = 1e-6f;
            float closestaspect = -9999999.0f;
            float closestdiff = 999999.0f;
            int closestdiffw = 9999999;

            for (int i = 0; i < num_specs; i++) {
                const SDL_CameraSpec *thisspec = &device->all_specs[i];
                const int thisw = thisspec->width;
                const int thish = thisspec->height;
                const float thisaspect = ((float)thisw) / ((float)thish);
                const float aspectdiff = SDL_fabsf(wantaspect - thisaspect);
                const float diff = SDL_fabsf(closestaspect - thisaspect);
                const int diffw = SDL_abs(thisw - wantw);
                if (diff < epsilon) { // matches current closestaspect? See if resolution is closer in size.
                    if (diffw < closestdiffw) {
                        closestdiffw = diffw;
                        closest->width = thisw;
                        closest->height = thish;
                    }
                } else if (aspectdiff < closestdiff) { // this is a closer aspect ratio? Take it, reset resolution checks.
                    closestdiff = aspectdiff;
                    closestaspect = thisaspect;
                    closestdiffw = diffw;
                    closest->width = thisw;
                    closest->height = thish;
                }
            }
        } else {
            SDL_copyp(closest, &device->all_specs[0]);
        }

        SDL_assert(closest->width > 0);
        SDL_assert(closest->height > 0);

        // okay, we have what we think is the best resolution, now we just need the best format that supports it...
        const SDL_PixelFormat wantfmt = spec->format;
        SDL_PixelFormat best_format = SDL_PIXELFORMAT_UNKNOWN;
        SDL_Colorspace best_colorspace = SDL_COLORSPACE_UNKNOWN;
        for (int i = 0; i < num_specs; i++) {
            const SDL_CameraSpec *thisspec = &device->all_specs[i];
            if ((thisspec->width == closest->width) && (thisspec->height == closest->height)) {
                if (best_format == SDL_PIXELFORMAT_UNKNOWN) {
                    best_format = thisspec->format;  // spec list is sorted by what we consider "best" format, so unless we find an exact match later, first size match is the one!
                    best_colorspace = thisspec->colorspace;
                }
                if (thisspec->format == wantfmt) {
                    best_format = thisspec->format;
                    best_colorspace = thisspec->colorspace;
                    break;  // exact match, stop looking.
                }
            }
        }

        SDL_assert(best_format != SDL_PIXELFORMAT_UNKNOWN);
        SDL_assert(best_colorspace != SDL_COLORSPACE_UNKNOWN);
        closest->format = best_format;
        closest->colorspace = best_colorspace;

        // We have a resolution and a format, find the closest framerate...
        const float wantfps = spec->framerate_denominator ? ((float)spec->framerate_numerator / spec->framerate_denominator) : 0.0f;
        float closestfps = 9999999.0f;
        for (int i = 0; i < num_specs; i++) {
            const SDL_CameraSpec *thisspec = &device->all_specs[i];
            if ((thisspec->format == closest->format) && (thisspec->width == closest->width) && (thisspec->height == closest->height)) {
                if ((thisspec->framerate_numerator == spec->framerate_numerator) && (thisspec->framerate_denominator == spec->framerate_denominator)) {
                    closest->framerate_numerator = thisspec->framerate_numerator;
                    closest->framerate_denominator = thisspec->framerate_denominator;
                    break;  // exact match, stop looking.
                }

                const float thisfps = thisspec->framerate_denominator ? ((float)thisspec->framerate_numerator / thisspec->framerate_denominator) : 0.0f;
                const float fpsdiff = SDL_fabsf(wantfps - thisfps);
                if (fpsdiff < closestfps) {  // this is a closest FPS? Take it until something closer arrives.
                    closestfps = fpsdiff;
                    closest->framerate_numerator = thisspec->framerate_numerator;
                    closest->framerate_denominator = thisspec->framerate_denominator;
                }
            }
        }
    }

    SDL_assert(closest->width > 0);
    SDL_assert(closest->height > 0);
    SDL_assert(closest->format != SDL_PIXELFORMAT_UNKNOWN);
}

SDL_Camera *SDL_OpenCamera(SDL_CameraID instance_id, const SDL_CameraSpec *spec)
{
    SDL_Camera *device = ObtainPhysicalCamera(instance_id);
    if (!device) {
        return NULL;
    }

    if (device->hidden != NULL) {
        ReleaseCamera(device);
        SDL_SetError("Camera already opened");  // we may remove this limitation at some point.
        return NULL;
    }

    SDL_SetAtomicInt(&device->shutdown, 0);

    // These start with the backend's implementation, but we might swap them out with zombie versions later.
    device->WaitDevice = camera_driver.impl.WaitDevice;
    device->AcquireFrame = camera_driver.impl.AcquireFrame;
    device->ReleaseFrame = camera_driver.impl.ReleaseFrame;

    SDL_CameraSpec closest;
    ChooseBestCameraSpec(device, spec, &closest);

    #if DEBUG_CAMERA
    SDL_Log("CAMERA: App wanted [(%dx%d) fmt=%s framerate=%d/%d], chose [(%dx%d) fmt=%s framerate=%d/%d]",
            spec ? spec->width : -1, spec ? spec->height : -1, spec ? SDL_GetPixelFormatName(spec->format) : "(null)", spec ? spec->framerate_numerator : -1, spec ? spec->framerate_denominator : -1,
            closest.width, closest.height, SDL_GetPixelFormatName(closest.format), closest.framerate_numerator, closest.framerate_denominator);
    #endif

    if (!camera_driver.impl.OpenDevice(device, &closest)) {
        ClosePhysicalCamera(device);  // in case anything is half-initialized.
        ReleaseCamera(device);
        return NULL;
    }

    SDL_copyp(&device->spec, spec ? spec : &closest);
    SDL_copyp(&device->actual_spec, &closest);

    // SDL_PIXELFORMAT_UNKNOWN here is taken as a signal that the backend
    //  doesn't know its format yet (Emscripten waiting for user permission,
    //  in this case), and the backend will call SDL_PrepareCameraSurfaces()
    //  itself, later but before the app is allowed to acquire images.
    if (closest.format != SDL_PIXELFORMAT_UNKNOWN) {
        if (!SDL_PrepareCameraSurfaces(device)) {
            ClosePhysicalCamera(device);
            ReleaseCamera(device);
            return NULL;
        }
    }

    device->drop_frames = 1;

    // Start the camera thread if necessary
    if (!camera_driver.impl.ProvidesOwnCallbackThread) {
        char threadname[64];
        SDL_GetCameraThreadName(device, threadname, sizeof (threadname));
        device->thread = SDL_CreateThread(CameraThread, threadname, device);
        if (!device->thread) {
            ClosePhysicalCamera(device);
            ReleaseCamera(device);
            SDL_SetError("Couldn't create camera thread");
            return NULL;
        }
    }

    ReleaseCamera(device);  // unlock, we're good to go!

    return device;  // currently there's no separation between physical and logical device.
}

SDL_Surface *SDL_AcquireCameraFrame(SDL_Camera *camera, Uint64 *timestampNS)
{
    if (timestampNS) {
        *timestampNS = 0;
    }

    if (!camera) {
        SDL_InvalidParamError("camera");
        return NULL;
    }

    SDL_Camera *device = camera;  // currently there's no separation between physical and logical device.

    ObtainPhysicalCameraObj(device);

    if (device->permission <= 0) {
        ReleaseCamera(device);
        SDL_SetError("Camera permission has not been granted");
        return NULL;
    }

    SDL_Surface *result = NULL;

    // frames are in this list from newest to oldest, so find the end of the list...
    SurfaceList *slistprev = &device->filled_output_surfaces;
    SurfaceList *slist = slistprev;
    while (slist->next) {
        slistprev = slist;
        slist = slist->next;
    }

    const bool list_is_empty = (slist == slistprev);
    if (!list_is_empty) { // report the oldest frame.
        if (timestampNS) {
            *timestampNS = slist->timestampNS;
        }
        result = slist->surface;
        slistprev->next = slist->next;  // remove from filled list.
        slist->next = device->app_held_output_surfaces.next;  // add to app_held list.
        device->app_held_output_surfaces.next = slist;
    }

    ReleaseCamera(device);

    return result;
}

void SDL_ReleaseCameraFrame(SDL_Camera *camera, SDL_Surface *frame)
{
    if (!camera || !frame) {
        return;
    }

    SDL_Camera *device = camera;  // currently there's no separation between physical and logical device.
    ObtainPhysicalCameraObj(device);

    SurfaceList *slistprev = &device->app_held_output_surfaces;
    SurfaceList *slist;
    for (slist = slistprev->next; slist != NULL; slist = slist->next) {
        if (slist->surface == frame) {
            break;
        }
        slistprev = slist;
    }

    if (!slist) {
        ReleaseCamera(device);
        return;
    }

    // this pointer was owned by the backend (DMA memory or whatever), clear it out.
    if (!device->needs_conversion && !device->needs_scaling) {
        device->ReleaseFrame(device, frame);
        frame->pixels = NULL;
        frame->pitch = 0;
    }

    slist->timestampNS = 0;

    // remove from app_held list...
    slistprev->next = slist->next;

    // insert at front of empty list (and we'll use it first when we need to fill a new frame).
    slist->next = device->empty_output_surfaces.next;
    device->empty_output_surfaces.next = slist;

    ReleaseCamera(device);
}

SDL_CameraID SDL_GetCameraID(SDL_Camera *camera)
{
    SDL_CameraID result = 0;
    if (!camera) {
        SDL_InvalidParamError("camera");
    } else {
        SDL_Camera *device = camera;  // currently there's no separation between physical and logical device.
        ObtainPhysicalCameraObj(device);
        result = device->instance_id;
        ReleaseCamera(device);
    }

    return result;
}

SDL_PropertiesID SDL_GetCameraProperties(SDL_Camera *camera)
{
    SDL_PropertiesID result = 0;
    if (!camera) {
        SDL_InvalidParamError("camera");
    } else {
        SDL_Camera *device = camera;  // currently there's no separation between physical and logical device.
        ObtainPhysicalCameraObj(device);
        if (device->props == 0) {
            device->props = SDL_CreateProperties();
        }
        result = device->props;
        ReleaseCamera(device);
    }

    return result;
}

int SDL_GetCameraPermissionState(SDL_Camera *camera)
{
    int result;
    if (!camera) {
        SDL_InvalidParamError("camera");
        result = -1;
    } else {
        SDL_Camera *device = camera;  // currently there's no separation between physical and logical device.
        ObtainPhysicalCameraObj(device);
        result = device->permission;
        ReleaseCamera(device);
    }
    return result;
}


static void CompleteCameraEntryPoints(void)
{
    // this doesn't currently fill in stub implementations, it just asserts the backend filled them all in.
    #define FILL_STUB(x) SDL_assert(camera_driver.impl.x != NULL)
    FILL_STUB(DetectDevices);
    FILL_STUB(OpenDevice);
    FILL_STUB(CloseDevice);
    FILL_STUB(AcquireFrame);
    FILL_STUB(ReleaseFrame);
    FILL_STUB(FreeDeviceHandle);
    FILL_STUB(Deinitialize);
    #undef FILL_STUB
}

void SDL_QuitCamera(void)
{
    if (!camera_driver.name) {  // not initialized?!
        return;
    }

    SDL_LockRWLockForWriting(camera_driver.device_hash_lock);
    SDL_SetAtomicInt(&camera_driver.shutting_down, 1);
    SDL_HashTable *device_hash = camera_driver.device_hash;
    camera_driver.device_hash = NULL;
    SDL_PendingCameraEvent *pending_events = camera_driver.pending_events.next;
    camera_driver.pending_events.next = NULL;
    SDL_SetAtomicInt(&camera_driver.device_count, 0);
    SDL_UnlockRWLock(camera_driver.device_hash_lock);

    SDL_PendingCameraEvent *pending_next = NULL;
    for (SDL_PendingCameraEvent *i = pending_events; i; i = pending_next) {
        pending_next = i->next;
        SDL_free(i);
    }

    SDL_DestroyHashTable(device_hash);

    // Free the driver data
    camera_driver.impl.Deinitialize();

    SDL_DestroyRWLock(camera_driver.device_hash_lock);

    SDL_zero(camera_driver);
}

// Physical camera objects are only destroyed when removed from the device hash.
static void SDLCALL DestroyCameraHashItem(void *userdata, const void *key, const void *value)
{
    SDL_Camera *device = (SDL_Camera *) value;
    ClosePhysicalCamera(device);
    camera_driver.impl.FreeDeviceHandle(device);
    SDL_DestroyMutex(device->lock);
    SDL_free(device->all_specs);
    SDL_free(device->name);
    SDL_free(device);
}

bool SDL_CameraInit(const char *driver_name)
{
    if (SDL_GetCurrentCameraDriver()) {
        SDL_QuitCamera(); // shutdown driver if already running.
    }

    SDL_RWLock *device_hash_lock = SDL_CreateRWLock();  // create this early, so if it fails we don't have to tear down the whole camera subsystem.
    if (!device_hash_lock) {
        return false;
    }

    SDL_HashTable *device_hash = SDL_CreateHashTable(0, false, SDL_HashID, SDL_KeyMatchID, DestroyCameraHashItem, NULL);
    if (!device_hash) {
        SDL_DestroyRWLock(device_hash_lock);
        return false;
    }

    // Select the proper camera driver
    if (!driver_name) {
        driver_name = SDL_GetHint(SDL_HINT_CAMERA_DRIVER);
    }

    bool initialized = false;
    bool tried_to_init = false;

    if (driver_name && (*driver_name != 0)) {
        char *driver_name_copy = SDL_strdup(driver_name);
        const char *driver_attempt = driver_name_copy;

        if (!driver_name_copy) {
            SDL_DestroyRWLock(device_hash_lock);
            SDL_DestroyHashTable(device_hash);
            return false;
        }

        while (driver_attempt && (*driver_attempt != 0) && !initialized) {
            char *driver_attempt_end = SDL_strchr(driver_attempt, ',');
            if (driver_attempt_end) {
                *driver_attempt_end = '\0';
            }

            for (int i = 0; bootstrap[i]; i++) {
                if (SDL_strcasecmp(bootstrap[i]->name, driver_attempt) == 0) {
                    tried_to_init = true;
                    SDL_zero(camera_driver);
                    camera_driver.pending_events_tail = &camera_driver.pending_events;
                    camera_driver.device_hash_lock = device_hash_lock;
                    camera_driver.device_hash = device_hash;
                    if (bootstrap[i]->init(&camera_driver.impl)) {
                        camera_driver.name = bootstrap[i]->name;
                        camera_driver.desc = bootstrap[i]->desc;
                        initialized = true;
                    }
                    break;
                }
            }

            driver_attempt = (driver_attempt_end) ? (driver_attempt_end + 1) : NULL;
        }

        SDL_free(driver_name_copy);
    } else {
        for (int i = 0; !initialized && bootstrap[i]; i++) {
            if (bootstrap[i]->demand_only) {
                continue;
            }

            tried_to_init = true;
            SDL_zero(camera_driver);
            camera_driver.pending_events_tail = &camera_driver.pending_events;
            camera_driver.device_hash_lock = device_hash_lock;
            camera_driver.device_hash = device_hash;
            if (bootstrap[i]->init(&camera_driver.impl)) {
                camera_driver.name = bootstrap[i]->name;
                camera_driver.desc = bootstrap[i]->desc;
                initialized = true;
            }
        }
    }

    if (!initialized) {
        // specific drivers will set the error message if they fail, but otherwise we do it here.
        if (!tried_to_init) {
            if (driver_name) {
                SDL_SetError("Camera driver '%s' not available", driver_name);
            } else {
                SDL_SetError("No available camera driver");
            }
        }

        SDL_zero(camera_driver);
        SDL_DestroyRWLock(device_hash_lock);
        SDL_DestroyHashTable(device_hash);
        return false;  // No driver was available, so fail.
    }

    CompleteCameraEntryPoints();

    // Make sure we have a list of devices available at startup...
    camera_driver.impl.DetectDevices();

    return true;
}

// This is an internal function, so SDL_PumpEvents() can check for pending camera device events.
// ("UpdateSubsystem" is the same naming that the other things that hook into PumpEvents use.)
void SDL_UpdateCamera(void)
{
    SDL_LockRWLockForReading(camera_driver.device_hash_lock);
    SDL_PendingCameraEvent *pending_events = camera_driver.pending_events.next;
    SDL_UnlockRWLock(camera_driver.device_hash_lock);

    if (!pending_events) {
        return;  // nothing to do, check next time.
    }

    // okay, let's take this whole list of events so we can dump the lock, and new ones can queue up for a later update.
    SDL_LockRWLockForWriting(camera_driver.device_hash_lock);
    pending_events = camera_driver.pending_events.next;  // in case this changed...
    camera_driver.pending_events.next = NULL;
    camera_driver.pending_events_tail = &camera_driver.pending_events;
    SDL_UnlockRWLock(camera_driver.device_hash_lock);

    SDL_PendingCameraEvent *pending_next = NULL;
    for (SDL_PendingCameraEvent *i = pending_events; i; i = pending_next) {
        pending_next = i->next;
        if (SDL_EventEnabled(i->type)) {
            SDL_Event event;
            SDL_zero(event);
            event.type = i->type;
            event.cdevice.which = (Uint32) i->devid;
            SDL_PushEvent(&event);
        }
        SDL_free(i);
    }
}

