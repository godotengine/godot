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

#include "../SDL_syscamera.h"
#include "../SDL_camera_c.h"
#include "../../video/SDL_pixels_c.h"
#include "../../video/SDL_surface_c.h"
#include "../../thread/SDL_systhread.h"

#ifdef SDL_CAMERA_DRIVER_ANDROID

/*
 * AndroidManifest.xml:
 *   <uses-permission android:name="android.permission.CAMERA"></uses-permission>
 *   <uses-feature android:name="android.hardware.camera" />
 *
 * Very likely SDL must be build with YUV support (done by default)
 *
 * https://developer.android.com/reference/android/hardware/camera2/CameraManager
 * "All camera devices intended to be operated concurrently, must be opened using openCamera(String, CameraDevice.StateCallback, Handler),
 * before configuring sessions on any of the camera devices."
 */

// this is kinda gross, but on older NDK headers all the camera stuff is
//  gated behind __ANDROID_API__. We'll dlopen() it at runtime, so we'll do
//  the right thing on pre-Android 7.0 devices, but we still
//  need the struct declarations and such in those headers.
// The other option is to make a massive jump in minimum Android version we
//  support--going from ancient to merely really old--but this seems less
//  distasteful and using dlopen matches practices on other SDL platforms.
//  We'll see if it works out.
#if __ANDROID_API__ < 24
#undef __ANDROID_API__
#define __ANDROID_API__ 24
#endif

#include <dlfcn.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>

#include "../../core/android/SDL_android.h"

static void *libcamera2ndk = NULL;
typedef ACameraManager* (*pfnACameraManager_create)(void);
typedef camera_status_t (*pfnACameraManager_registerAvailabilityCallback)(ACameraManager*, const ACameraManager_AvailabilityCallbacks*);
typedef camera_status_t (*pfnACameraManager_unregisterAvailabilityCallback)(ACameraManager*, const ACameraManager_AvailabilityCallbacks*);
typedef camera_status_t (*pfnACameraManager_getCameraIdList)(ACameraManager*, ACameraIdList**);
typedef void (*pfnACameraManager_deleteCameraIdList)(ACameraIdList*);
typedef void (*pfnACameraCaptureSession_close)(ACameraCaptureSession*);
typedef void (*pfnACaptureRequest_free)(ACaptureRequest*);
typedef void (*pfnACameraOutputTarget_free)(ACameraOutputTarget*);
typedef camera_status_t (*pfnACameraDevice_close)(ACameraDevice*);
typedef void (*pfnACameraManager_delete)(ACameraManager*);
typedef void (*pfnACaptureSessionOutputContainer_free)(ACaptureSessionOutputContainer*);
typedef void (*pfnACaptureSessionOutput_free)(ACaptureSessionOutput*);
typedef camera_status_t (*pfnACameraManager_openCamera)(ACameraManager*, const char*, ACameraDevice_StateCallbacks*, ACameraDevice**);
typedef camera_status_t (*pfnACameraDevice_createCaptureRequest)(const ACameraDevice*, ACameraDevice_request_template, ACaptureRequest**);
typedef camera_status_t (*pfnACameraDevice_createCaptureSession)(ACameraDevice*, const ACaptureSessionOutputContainer*, const ACameraCaptureSession_stateCallbacks*,ACameraCaptureSession**);
typedef camera_status_t (*pfnACameraManager_getCameraCharacteristics)(ACameraManager*, const char*, ACameraMetadata**);
typedef void (*pfnACameraMetadata_free)(ACameraMetadata*);
typedef camera_status_t (*pfnACameraMetadata_getConstEntry)(const ACameraMetadata*, uint32_t tag, ACameraMetadata_const_entry*);
typedef camera_status_t (*pfnACameraCaptureSession_setRepeatingRequest)(ACameraCaptureSession*, ACameraCaptureSession_captureCallbacks*, int numRequests, ACaptureRequest**, int*);
typedef camera_status_t (*pfnACameraOutputTarget_create)(ACameraWindowType*,ACameraOutputTarget**);
typedef camera_status_t (*pfnACaptureRequest_addTarget)(ACaptureRequest*, const ACameraOutputTarget*);
typedef camera_status_t (*pfnACaptureSessionOutputContainer_add)(ACaptureSessionOutputContainer*, const ACaptureSessionOutput*);
typedef camera_status_t (*pfnACaptureSessionOutputContainer_create)(ACaptureSessionOutputContainer**);
typedef camera_status_t (*pfnACaptureSessionOutput_create)(ACameraWindowType*, ACaptureSessionOutput**);
static pfnACameraManager_create pACameraManager_create = NULL;
static pfnACameraManager_registerAvailabilityCallback pACameraManager_registerAvailabilityCallback = NULL;
static pfnACameraManager_unregisterAvailabilityCallback pACameraManager_unregisterAvailabilityCallback = NULL;
static pfnACameraManager_getCameraIdList pACameraManager_getCameraIdList = NULL;
static pfnACameraManager_deleteCameraIdList pACameraManager_deleteCameraIdList = NULL;
static pfnACameraCaptureSession_close pACameraCaptureSession_close = NULL;
static pfnACaptureRequest_free pACaptureRequest_free = NULL;
static pfnACameraOutputTarget_free pACameraOutputTarget_free = NULL;
static pfnACameraDevice_close pACameraDevice_close = NULL;
static pfnACameraManager_delete pACameraManager_delete = NULL;
static pfnACaptureSessionOutputContainer_free pACaptureSessionOutputContainer_free = NULL;
static pfnACaptureSessionOutput_free pACaptureSessionOutput_free = NULL;
static pfnACameraManager_openCamera pACameraManager_openCamera = NULL;
static pfnACameraDevice_createCaptureRequest pACameraDevice_createCaptureRequest = NULL;
static pfnACameraDevice_createCaptureSession pACameraDevice_createCaptureSession = NULL;
static pfnACameraManager_getCameraCharacteristics pACameraManager_getCameraCharacteristics = NULL;
static pfnACameraMetadata_free pACameraMetadata_free = NULL;
static pfnACameraMetadata_getConstEntry pACameraMetadata_getConstEntry = NULL;
static pfnACameraCaptureSession_setRepeatingRequest pACameraCaptureSession_setRepeatingRequest = NULL;
static pfnACameraOutputTarget_create pACameraOutputTarget_create = NULL;
static pfnACaptureRequest_addTarget pACaptureRequest_addTarget = NULL;
static pfnACaptureSessionOutputContainer_add pACaptureSessionOutputContainer_add = NULL;
static pfnACaptureSessionOutputContainer_create pACaptureSessionOutputContainer_create = NULL;
static pfnACaptureSessionOutput_create pACaptureSessionOutput_create = NULL;

static void *libmediandk = NULL;
typedef void (*pfnAImage_delete)(AImage*);
typedef media_status_t (*pfnAImage_getTimestamp)(const AImage*, int64_t*);
typedef media_status_t (*pfnAImage_getNumberOfPlanes)(const AImage*, int32_t*);
typedef media_status_t (*pfnAImage_getPlaneRowStride)(const AImage*, int, int32_t*);
typedef media_status_t (*pfnAImage_getPlaneData)(const AImage*, int, uint8_t**, int*);
typedef media_status_t (*pfnAImageReader_acquireNextImage)(AImageReader*, AImage**);
typedef void (*pfnAImageReader_delete)(AImageReader*);
typedef media_status_t (*pfnAImageReader_setImageListener)(AImageReader*, AImageReader_ImageListener*);
typedef media_status_t (*pfnAImageReader_getWindow)(AImageReader*, ANativeWindow**);
typedef media_status_t (*pfnAImageReader_new)(int32_t, int32_t, int32_t, int32_t, AImageReader**);
static pfnAImage_delete pAImage_delete = NULL;
static pfnAImage_getTimestamp pAImage_getTimestamp = NULL;
static pfnAImage_getNumberOfPlanes pAImage_getNumberOfPlanes = NULL;
static pfnAImage_getPlaneRowStride pAImage_getPlaneRowStride = NULL;
static pfnAImage_getPlaneData pAImage_getPlaneData = NULL;
static pfnAImageReader_acquireNextImage pAImageReader_acquireNextImage = NULL;
static pfnAImageReader_delete pAImageReader_delete = NULL;
static pfnAImageReader_setImageListener pAImageReader_setImageListener = NULL;
static pfnAImageReader_getWindow pAImageReader_getWindow = NULL;
static pfnAImageReader_new pAImageReader_new = NULL;

typedef media_status_t (*pfnAImage_getWidth)(const AImage*, int32_t*);
typedef media_status_t (*pfnAImage_getHeight)(const AImage*, int32_t*);
static pfnAImage_getWidth pAImage_getWidth = NULL;
static pfnAImage_getHeight pAImage_getHeight = NULL;

struct SDL_PrivateCameraData
{
    ACameraDevice *device;
    AImageReader *reader;
    ANativeWindow *window;
    ACaptureSessionOutput *sessionOutput;
    ACaptureSessionOutputContainer *sessionOutputContainer;
    ACameraOutputTarget *outputTarget;
    ACaptureRequest *request;
    ACameraCaptureSession *session;
    SDL_CameraSpec requested_spec;
};

static bool SetErrorStr(const char *what, const char *errstr, const int rc)
{
    char errbuf[128];
    if (!errstr) {
        SDL_snprintf(errbuf, sizeof (errbuf), "Unknown error #%d", rc);
        errstr = errbuf;
    }
    return SDL_SetError("%s: %s", what, errstr);
}

static const char *CameraStatusStr(const camera_status_t rc)
{
    switch (rc) {
        case ACAMERA_OK: return "no error";
        case ACAMERA_ERROR_UNKNOWN: return "unknown error";
        case ACAMERA_ERROR_INVALID_PARAMETER: return "invalid parameter";
        case ACAMERA_ERROR_CAMERA_DISCONNECTED: return "camera disconnected";
        case ACAMERA_ERROR_NOT_ENOUGH_MEMORY: return "not enough memory";
        case ACAMERA_ERROR_METADATA_NOT_FOUND: return "metadata not found";
        case ACAMERA_ERROR_CAMERA_DEVICE: return "camera device error";
        case ACAMERA_ERROR_CAMERA_SERVICE: return "camera service error";
        case ACAMERA_ERROR_SESSION_CLOSED: return "session closed";
        case ACAMERA_ERROR_INVALID_OPERATION: return "invalid operation";
        case ACAMERA_ERROR_STREAM_CONFIGURE_FAIL: return "configure failure";
        case ACAMERA_ERROR_CAMERA_IN_USE: return "camera in use";
        case ACAMERA_ERROR_MAX_CAMERA_IN_USE: return "max cameras in use";
        case ACAMERA_ERROR_CAMERA_DISABLED: return "camera disabled";
        case ACAMERA_ERROR_PERMISSION_DENIED: return "permission denied";
        case ACAMERA_ERROR_UNSUPPORTED_OPERATION: return "unsupported operation";
        default: break;
    }

    return NULL;  // unknown error
}

static bool SetCameraError(const char *what, const camera_status_t rc)
{
    return SetErrorStr(what, CameraStatusStr(rc), (int) rc);
}

static const char *MediaStatusStr(const media_status_t rc)
{
    switch (rc) {
        case AMEDIA_OK: return "no error";
        case AMEDIACODEC_ERROR_INSUFFICIENT_RESOURCE: return "insufficient resources";
        case AMEDIACODEC_ERROR_RECLAIMED: return "reclaimed";
        case AMEDIA_ERROR_UNKNOWN: return "unknown error";
        case AMEDIA_ERROR_MALFORMED: return "malformed";
        case AMEDIA_ERROR_UNSUPPORTED: return "unsupported";
        case AMEDIA_ERROR_INVALID_OBJECT: return "invalid object";
        case AMEDIA_ERROR_INVALID_PARAMETER: return "invalid parameter";
        case AMEDIA_ERROR_INVALID_OPERATION: return "invalid operation";
        case AMEDIA_ERROR_END_OF_STREAM: return "end of stream";
        case AMEDIA_ERROR_IO: return "i/o error";
        case AMEDIA_ERROR_WOULD_BLOCK: return "operation would block";
        case AMEDIA_DRM_NOT_PROVISIONED: return "DRM not provisioned";
        case AMEDIA_DRM_RESOURCE_BUSY: return "DRM resource busy";
        case AMEDIA_DRM_DEVICE_REVOKED: return "DRM device revoked";
        case AMEDIA_DRM_SHORT_BUFFER: return "DRM short buffer";
        case AMEDIA_DRM_SESSION_NOT_OPENED: return "DRM session not opened";
        case AMEDIA_DRM_TAMPER_DETECTED: return "DRM tampering detected";
        case AMEDIA_DRM_VERIFY_FAILED: return "DRM verify failed";
        case AMEDIA_DRM_NEED_KEY: return "DRM need key";
        case AMEDIA_DRM_LICENSE_EXPIRED: return "DRM license expired";
        case AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE: return "no buffer available";
        case AMEDIA_IMGREADER_MAX_IMAGES_ACQUIRED: return "maximum images acquired";
        case AMEDIA_IMGREADER_CANNOT_LOCK_IMAGE: return "cannot lock image";
        case AMEDIA_IMGREADER_CANNOT_UNLOCK_IMAGE: return "cannot unlock image";
        case AMEDIA_IMGREADER_IMAGE_NOT_LOCKED: return "image not locked";
        default: break;
    }

    return NULL;  // unknown error
}

static bool SetMediaError(const char *what, const media_status_t rc)
{
    return SetErrorStr(what, MediaStatusStr(rc), (int) rc);
}


static ACameraManager *cameraMgr = NULL;

static bool CreateCameraManager(void)
{
    SDL_assert(cameraMgr == NULL);

    cameraMgr = pACameraManager_create();
    if (!cameraMgr) {
        return SDL_SetError("Error creating ACameraManager");
    }
    return true;
}

static void DestroyCameraManager(void)
{
    if (cameraMgr) {
        pACameraManager_delete(cameraMgr);
        cameraMgr = NULL;
    }
}

static void format_android_to_sdl(Uint32 fmt, SDL_PixelFormat *format, SDL_Colorspace *colorspace)
{
    switch (fmt) {
        #define CASE(x, y, z)  case x: *format = y; *colorspace = z; return
        CASE(AIMAGE_FORMAT_YUV_420_888, SDL_PIXELFORMAT_NV12, SDL_COLORSPACE_BT709_LIMITED);
        CASE(AIMAGE_FORMAT_RGB_565,     SDL_PIXELFORMAT_RGB565, SDL_COLORSPACE_SRGB);
        CASE(AIMAGE_FORMAT_RGB_888,     SDL_PIXELFORMAT_XRGB8888, SDL_COLORSPACE_SRGB);
        CASE(AIMAGE_FORMAT_RGBA_8888,   SDL_PIXELFORMAT_RGBA8888, SDL_COLORSPACE_SRGB);
        CASE(AIMAGE_FORMAT_RGBX_8888,   SDL_PIXELFORMAT_RGBX8888, SDL_COLORSPACE_SRGB);
        CASE(AIMAGE_FORMAT_RGBA_FP16,   SDL_PIXELFORMAT_RGBA64_FLOAT, SDL_COLORSPACE_SRGB);
        #undef CASE
        default: break;
    }

    #if DEBUG_CAMERA
    //SDL_Log("Unknown format AIMAGE_FORMAT '%d'", fmt);
    #endif

    *format = SDL_PIXELFORMAT_UNKNOWN;
    *colorspace = SDL_COLORSPACE_UNKNOWN;
}

static Uint32 format_sdl_to_android(SDL_PixelFormat fmt)
{
    switch (fmt) {
        #define CASE(x, y)  case y: return x
        CASE(AIMAGE_FORMAT_YUV_420_888, SDL_PIXELFORMAT_NV12);
        CASE(AIMAGE_FORMAT_RGB_565,     SDL_PIXELFORMAT_RGB565);
        CASE(AIMAGE_FORMAT_RGB_888,     SDL_PIXELFORMAT_XRGB8888);
        CASE(AIMAGE_FORMAT_RGBA_8888,   SDL_PIXELFORMAT_RGBA8888);
        CASE(AIMAGE_FORMAT_RGBX_8888,   SDL_PIXELFORMAT_RGBX8888);
        #undef CASE
        default:
            return 0;
    }
}

static bool ANDROIDCAMERA_WaitDevice(SDL_Camera *device)
{
    return true;  // this isn't used atm, since we run our own thread via onImageAvailable callbacks.
}

static SDL_CameraFrameResult ANDROIDCAMERA_AcquireFrame(SDL_Camera *device, SDL_Surface *frame, Uint64 *timestampNS)
{
    SDL_CameraFrameResult result = SDL_CAMERA_FRAME_READY;
    media_status_t res;
    AImage *image = NULL;

    res = pAImageReader_acquireNextImage(device->hidden->reader, &image);
    // We could also use this one:
    //res = AImageReader_acquireLatestImage(device->hidden->reader, &image);

    SDL_assert(res != AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE);  // we should only be here if onImageAvailable was called.

    if (res != AMEDIA_OK) {
        SetMediaError("Error AImageReader_acquireNextImage", res);
        return SDL_CAMERA_FRAME_ERROR;
    }

    int64_t atimestamp = 0;
    if (pAImage_getTimestamp(image, &atimestamp) == AMEDIA_OK) {
        *timestampNS = (Uint64) atimestamp;
    } else {
        *timestampNS = 0;
    }

    // !!! FIXME: this currently copies the data to the surface (see FIXME about non-contiguous planar surfaces, but in theory we could just keep this locked until ReleaseFrame...
    int32_t num_planes = 0;
    pAImage_getNumberOfPlanes(image, &num_planes);

    if ((num_planes == 3) && (device->spec.format == SDL_PIXELFORMAT_NV12)) {
        num_planes--;   // treat the interleaved planes as one.
    }

    size_t buflen = 0;
    pAImage_getPlaneRowStride(image, 0, &frame->pitch);
    for (int i = 0; (i < num_planes) && (i < 3); i++) {
        int32_t expected;
        if (i == 0) {
            expected = frame->pitch * frame->h;
        } else {
            expected = frame->pitch * (frame->h + 1) / 2;
        }
        buflen += expected;
    }

    frame->pixels = SDL_aligned_alloc(SDL_GetSIMDAlignment(), buflen);
    if (frame->pixels == NULL) {
        result = SDL_CAMERA_FRAME_ERROR;
    } else {
        Uint8 *dst = frame->pixels;

        for (int i = 0; (i < num_planes) && (i < 3); i++) {
            uint8_t *data = NULL;
            int32_t datalen = 0;
            int32_t expected;
            if (i == 0) {
                expected = frame->pitch * frame->h;
            } else {
                expected = frame->pitch * (frame->h + 1) / 2;
            }
            pAImage_getPlaneData(image, i, &data, &datalen);

            int32_t row_stride = 0;
            pAImage_getPlaneRowStride(image, i, &row_stride);
            SDL_assert(row_stride == frame->pitch);
            SDL_memcpy(dst, data, SDL_min(expected, datalen));
            dst += expected;
        }
    }

    pAImage_delete(image);

    return result;
}

static void ANDROIDCAMERA_ReleaseFrame(SDL_Camera *device, SDL_Surface *frame)
{
    // !!! FIXME: this currently copies the data to the surface, but in theory we could just keep the AImage until ReleaseFrame...
    SDL_aligned_free(frame->pixels);
}

static void onImageAvailable(void *context, AImageReader *reader)
{
    #if DEBUG_CAMERA
    SDL_Log("CAMERA: CB onImageAvailable");
    #endif
    SDL_Camera *device = (SDL_Camera *) context;
    SDL_CameraThreadIterate(device);
}

static void onDisconnected(void *context, ACameraDevice *device)
{
    #if DEBUG_CAMERA
    SDL_Log("CAMERA: CB onDisconnected");
    #endif
    SDL_CameraDisconnected((SDL_Camera *) context);
}

static void onError(void *context, ACameraDevice *device, int error)
{
    #if DEBUG_CAMERA
    SDL_Log("CAMERA: CB onError");
    #endif
    SDL_CameraDisconnected((SDL_Camera *) context);
}

static void onClosed(void* context, ACameraCaptureSession *session)
{
    // SDL_Camera *_this = (SDL_Camera *) context;
    #if DEBUG_CAMERA
    SDL_Log("CAMERA: CB onClosed");
    #endif
}

static void onReady(void* context, ACameraCaptureSession *session)
{
    // SDL_Camera *_this = (SDL_Camera *) context;
    #if DEBUG_CAMERA
    SDL_Log("CAMERA: CB onReady");
    #endif
}

static void onActive(void* context, ACameraCaptureSession *session)
{
    // SDL_Camera *_this = (SDL_Camera *) context;
    #if DEBUG_CAMERA
    SDL_Log("CAMERA: CB onActive");
    #endif
}

static void ANDROIDCAMERA_CloseDevice(SDL_Camera *device)
{
    if (device && device->hidden) {
        struct SDL_PrivateCameraData *hidden = device->hidden;
        device->hidden = NULL;

        if (hidden->reader) {
            pAImageReader_setImageListener(hidden->reader, NULL);
        }

        if (hidden->session) {
            pACameraCaptureSession_close(hidden->session);
        }

        if (hidden->request) {
            pACaptureRequest_free(hidden->request);
        }

        if (hidden->outputTarget) {
            pACameraOutputTarget_free(hidden->outputTarget);
        }

        if (hidden->sessionOutputContainer) {
            pACaptureSessionOutputContainer_free(hidden->sessionOutputContainer);
        }

        if (hidden->sessionOutput) {
            pACaptureSessionOutput_free(hidden->sessionOutput);
        }

        // we don't free hidden->window here, it'll be cleaned up by AImageReader_delete.

        if (hidden->reader) {
            pAImageReader_delete(hidden->reader);
        }

        if (hidden->device) {
            pACameraDevice_close(hidden->device);
        }

        SDL_free(hidden);
    }
}

// this is where the "opening" of the camera happens, after permission is granted.
static bool PrepareCamera(SDL_Camera *device)
{
    SDL_assert(device->hidden != NULL);

    camera_status_t res;
    media_status_t res2;

    ACameraDevice_StateCallbacks dev_callbacks;
    SDL_zero(dev_callbacks);
    dev_callbacks.context = device;
    dev_callbacks.onDisconnected = onDisconnected;
    dev_callbacks.onError = onError;

    ACameraCaptureSession_stateCallbacks capture_callbacks;
    SDL_zero(capture_callbacks);
    capture_callbacks.context = device;
    capture_callbacks.onClosed = onClosed;
    capture_callbacks.onReady = onReady;
    capture_callbacks.onActive = onActive;

    AImageReader_ImageListener imglistener;
    SDL_zero(imglistener);
    imglistener.context = device;
    imglistener.onImageAvailable = onImageAvailable;

    // just in case SDL_OpenCamera is overwriting device->spec as CameraPermissionCallback runs, we work from a different copy.
    const SDL_CameraSpec *spec = &device->hidden->requested_spec;

    if ((res = pACameraManager_openCamera(cameraMgr, (const char *) device->handle, &dev_callbacks, &device->hidden->device)) != ACAMERA_OK) {
        return SetCameraError("Failed to open camera", res);
    } else if ((res2 = pAImageReader_new(spec->width, spec->height, format_sdl_to_android(spec->format), 10 /* nb buffers */, &device->hidden->reader)) != AMEDIA_OK) {
        return SetMediaError("Error AImageReader_new", res2);
    } else if ((res2 = pAImageReader_getWindow(device->hidden->reader, &device->hidden->window)) != AMEDIA_OK) {
        return SetMediaError("Error AImageReader_getWindow", res2);
    } else if ((res = pACaptureSessionOutput_create(device->hidden->window, &device->hidden->sessionOutput)) != ACAMERA_OK) {
        return SetCameraError("Error ACaptureSessionOutput_create", res);
    } else if ((res = pACaptureSessionOutputContainer_create(&device->hidden->sessionOutputContainer)) != ACAMERA_OK) {
        return SetCameraError("Error ACaptureSessionOutputContainer_create", res);
    } else if ((res = pACaptureSessionOutputContainer_add(device->hidden->sessionOutputContainer, device->hidden->sessionOutput)) != ACAMERA_OK) {
        return SetCameraError("Error ACaptureSessionOutputContainer_add", res);
    } else if ((res = pACameraOutputTarget_create(device->hidden->window, &device->hidden->outputTarget)) != ACAMERA_OK) {
        return SetCameraError("Error ACameraOutputTarget_create", res);
    } else if ((res = pACameraDevice_createCaptureRequest(device->hidden->device, TEMPLATE_RECORD, &device->hidden->request)) != ACAMERA_OK) {
        return SetCameraError("Error ACameraDevice_createCaptureRequest", res);
    } else if ((res = pACaptureRequest_addTarget(device->hidden->request, device->hidden->outputTarget)) != ACAMERA_OK) {
        return SetCameraError("Error ACaptureRequest_addTarget", res);
    } else if ((res = pACameraDevice_createCaptureSession(device->hidden->device, device->hidden->sessionOutputContainer, &capture_callbacks, &device->hidden->session)) != ACAMERA_OK) {
        return SetCameraError("Error ACameraDevice_createCaptureSession", res);
    } else if ((res = pACameraCaptureSession_setRepeatingRequest(device->hidden->session, NULL, 1, &device->hidden->request, NULL)) != ACAMERA_OK) {
        return SetCameraError("Error ACameraCaptureSession_setRepeatingRequest", res);
    } else if ((res2 = pAImageReader_setImageListener(device->hidden->reader, &imglistener)) != AMEDIA_OK) {
        return SetMediaError("Error AImageReader_setImageListener", res2);
    }

    return true;
}

static void SDLCALL CameraPermissionCallback(void *userdata, const char *permission, bool granted)
{
    SDL_Camera *device = (SDL_Camera *) userdata;
    if (device->hidden != NULL) {   // if device was already closed, don't send an event.
        if (!granted) {
            SDL_CameraPermissionOutcome(device, false);  // sorry, permission denied.
        } else if (!PrepareCamera(device)) {  // permission given? Actually open the camera now.
            // uhoh, setup failed; since the app thinks we already "opened" the device, mark it as disconnected and don't report the permission.
            SDL_CameraDisconnected(device);
        } else {
            // okay! We have permission to use the camera _and_ opening the hardware worked out, report that the camera is usable!
            SDL_CameraPermissionOutcome(device, true);  // go go go!
        }
    }

    UnrefPhysicalCamera(device);   // we ref'd this in OpenDevice, release the extra reference.
}


static bool ANDROIDCAMERA_OpenDevice(SDL_Camera *device, const SDL_CameraSpec *spec)
{
#if 0  // !!! FIXME: for now, we'll just let this fail if it is going to fail, without checking for this
    /* Cannot open a second camera, while the first one is opened.
     * If you want to play several camera, they must all be opened first, then played.
     *
     * https://developer.android.com/reference/android/hardware/camera2/CameraManager
     * "All camera devices intended to be operated concurrently, must be opened using openCamera(String, CameraDevice.StateCallback, Handler),
     * before configuring sessions on any of the camera devices.  * "
     *
     */
    if (CheckDevicePlaying()) {
        return SDL_SetError("A camera is already playing");
    }
#endif

    device->hidden = (struct SDL_PrivateCameraData *) SDL_calloc(1, sizeof (struct SDL_PrivateCameraData));
    if (device->hidden == NULL) {
        return false;
    }

    RefPhysicalCamera(device);  // ref'd until permission callback fires.

    // just in case SDL_OpenCamera is overwriting device->spec as CameraPermissionCallback runs, we work from a different copy.
    SDL_copyp(&device->hidden->requested_spec, spec);
    if (!SDL_RequestAndroidPermission("android.permission.CAMERA", CameraPermissionCallback, device)) {
        UnrefPhysicalCamera(device);
        return false;
    }

    return true;  // we don't open the camera until permission is granted, so always succeed for now.
}

static void ANDROIDCAMERA_FreeDeviceHandle(SDL_Camera *device)
{
    if (device) {
        SDL_free(device->handle);
    }
}

static void GatherCameraSpecs(const char *devid, CameraFormatAddData *add_data, char **fullname, SDL_CameraPosition *position)
{
    SDL_zerop(add_data);

    ACameraMetadata *metadata = NULL;
    ACameraMetadata_const_entry cfgentry;
    ACameraMetadata_const_entry durentry;
    ACameraMetadata_const_entry infoentry;

    // This can fail with an "unknown error" (with `adb logcat` reporting "no such file or directory")
    // for "LEGACY" level cameras. I saw this happen on a 30-dollar budget phone I have for testing
    // (but a different brand budget phone worked, so it's not strictly the low-end of Android devices).
    // LEGACY devices are seen by onCameraAvailable, but are not otherwise accessible through
    // libcamera2ndk. The Java camera2 API apparently _can_ access these cameras, but we're going on
    // without them here for now, in hopes that such hardware is a dying breed.
    if (pACameraManager_getCameraCharacteristics(cameraMgr, devid, &metadata) != ACAMERA_OK) {
        return;  // oh well.
    } else if (pACameraMetadata_getConstEntry(metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &cfgentry) != ACAMERA_OK) {
        pACameraMetadata_free(metadata);
        return;  // oh well.
    } else if (pACameraMetadata_getConstEntry(metadata, ACAMERA_SCALER_AVAILABLE_MIN_FRAME_DURATIONS, &durentry) != ACAMERA_OK) {
        pACameraMetadata_free(metadata);
        return;  // oh well.
    }

    *fullname = NULL;
    if (pACameraMetadata_getConstEntry(metadata, ACAMERA_INFO_VERSION, &infoentry) == ACAMERA_OK) {
        *fullname = (char *) SDL_malloc(infoentry.count + 1);
        if (*fullname) {
            SDL_strlcpy(*fullname, (const char *) infoentry.data.u8, infoentry.count + 1);
        }
    }

    ACameraMetadata_const_entry posentry;
    if (pACameraMetadata_getConstEntry(metadata, ACAMERA_LENS_FACING, &posentry) == ACAMERA_OK) {  // ignore this if it fails.
        if (*posentry.data.u8 == ACAMERA_LENS_FACING_FRONT) {
            *position = SDL_CAMERA_POSITION_FRONT_FACING;
            if (!*fullname) {
                *fullname = SDL_strdup("Front-facing camera");
            }
        } else if (*posentry.data.u8 == ACAMERA_LENS_FACING_BACK) {
            *position = SDL_CAMERA_POSITION_BACK_FACING;
            if (!*fullname) {
                *fullname = SDL_strdup("Back-facing camera");
            }
        }
    }

    if (!*fullname) {
        *fullname = SDL_strdup("Generic camera");   // we tried.
    }

    const int32_t *i32ptr = cfgentry.data.i32;
    for (int i = 0; i < cfgentry.count; i++, i32ptr += 4) {
        const int32_t fmt = i32ptr[0];
        const int w = i32ptr[1];
        const int h = i32ptr[2];
        const int32_t type = i32ptr[3];
        SDL_PixelFormat sdlfmt = SDL_PIXELFORMAT_UNKNOWN;
        SDL_Colorspace colorspace = SDL_COLORSPACE_UNKNOWN;

        if (type == ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS_INPUT) {
            continue;
        } else if ((w <= 0) || (h <= 0)) {
            continue;
        } else {
            format_android_to_sdl(fmt, &sdlfmt, &colorspace);
            if (sdlfmt == SDL_PIXELFORMAT_UNKNOWN) {
                continue;
            }
        }

#if 0 // !!! FIXME: these all come out with 0 durations on my test phone.  :(
        const int64_t *i64ptr = durentry.data.i64;
        for (int j = 0; j < durentry.count; j++, i64ptr += 4) {
            const int32_t fpsfmt = (int32_t) i64ptr[0];
            const int fpsw = (int) i64ptr[1];
            const int fpsh = (int) i64ptr[2];
            const long long duration = (long long) i64ptr[3];
            SDL_Log("CAMERA: possible fps %s %dx%d duration=%lld", SDL_GetPixelFormatName(sdlfmt), fpsw, fpsh, duration);
            if ((duration > 0) && (fpsfmt == fmt) && (fpsw == w) && (fpsh == h)) {
                SDL_AddCameraFormat(add_data, sdlfmt, colorspace, w, h, 1000000000, duration);
            }
        }
#else
        SDL_AddCameraFormat(add_data, sdlfmt, colorspace, w, h, 30, 1);
#endif
    }

    pACameraMetadata_free(metadata);
}

static bool FindAndroidCameraByID(SDL_Camera *device, void *userdata)
{
    const char *devid = (const char *) userdata;
    return (SDL_strcmp(devid, (const char *) device->handle) == 0);
}

static void MaybeAddDevice(const char *devid)
{
    #if DEBUG_CAMERA
    SDL_Log("CAMERA: MaybeAddDevice('%s')", devid);
    #endif

    if (SDL_FindPhysicalCameraByCallback(FindAndroidCameraByID, (void *) devid)) {
        return;  // already have this one.
    }

    SDL_CameraPosition position = SDL_CAMERA_POSITION_UNKNOWN;
    char *fullname = NULL;
    CameraFormatAddData add_data;
    GatherCameraSpecs(devid, &add_data, &fullname, &position);
    if (add_data.num_specs > 0) {
        char *namecpy = SDL_strdup(devid);
        if (namecpy) {
            SDL_Camera *device = SDL_AddCamera(fullname, position, add_data.num_specs, add_data.specs, namecpy);
            if (!device) {
                SDL_free(namecpy);
            }
        }
    }

    SDL_free(fullname);
    SDL_free(add_data.specs);
}

// note that camera "availability" covers both hotplugging and whether another
//  has the device opened, but for something like Android, it's probably fine
//  to treat both unplugging and loss of access as disconnection events. When
//  the other app closes the camera, we get an available event as if it was
//  just plugged back in.

static void onCameraAvailable(void *context, const char *cameraId)
{
    #if DEBUG_CAMERA
    SDL_Log("CAMERA: CB onCameraAvailable('%s')", cameraId);
    #endif
    SDL_assert(cameraId != NULL);
    MaybeAddDevice(cameraId);
}

static void onCameraUnavailable(void *context, const char *cameraId)
{
    #if DEBUG_CAMERA
    SDL_Log("CAMERA: CB onCameraUnvailable('%s')", cameraId);
    #endif

    SDL_assert(cameraId != NULL);

    // THIS CALLBACK FIRES WHEN YOU OPEN THE DEVICE YOURSELF.  :(
    // Make sure we don't have the device opened, in which case onDisconnected will fire instead if actually lost.
    SDL_Camera *device = SDL_FindPhysicalCameraByCallback(FindAndroidCameraByID, (void *) cameraId);
    if (device && !device->hidden) {
        SDL_CameraDisconnected(device);
    }
}

static const ACameraManager_AvailabilityCallbacks camera_availability_listener = {
    NULL,
    onCameraAvailable,
    onCameraUnavailable
};

static void ANDROIDCAMERA_DetectDevices(void)
{
    ACameraIdList *list = NULL;
    camera_status_t res = pACameraManager_getCameraIdList(cameraMgr, &list);

    if ((res == ACAMERA_OK) && list) {
        const int total = list->numCameras;
        for (int i = 0; i < total; i++) {
            MaybeAddDevice(list->cameraIds[i]);
        }

        pACameraManager_deleteCameraIdList(list);
    }

    pACameraManager_registerAvailabilityCallback(cameraMgr, &camera_availability_listener);
}

static void ANDROIDCAMERA_Deinitialize(void)
{
    pACameraManager_unregisterAvailabilityCallback(cameraMgr, &camera_availability_listener);
    DestroyCameraManager();

    dlclose(libcamera2ndk);
    libcamera2ndk = NULL;
    pACameraManager_create = NULL;
    pACameraManager_registerAvailabilityCallback = NULL;
    pACameraManager_unregisterAvailabilityCallback = NULL;
    pACameraManager_getCameraIdList = NULL;
    pACameraManager_deleteCameraIdList = NULL;
    pACameraCaptureSession_close = NULL;
    pACaptureRequest_free = NULL;
    pACameraOutputTarget_free = NULL;
    pACameraDevice_close = NULL;
    pACameraManager_delete = NULL;
    pACaptureSessionOutputContainer_free = NULL;
    pACaptureSessionOutput_free = NULL;
    pACameraManager_openCamera = NULL;
    pACameraDevice_createCaptureRequest = NULL;
    pACameraDevice_createCaptureSession = NULL;
    pACameraManager_getCameraCharacteristics = NULL;
    pACameraMetadata_free = NULL;
    pACameraMetadata_getConstEntry = NULL;
    pACameraCaptureSession_setRepeatingRequest = NULL;
    pACameraOutputTarget_create = NULL;
    pACaptureRequest_addTarget = NULL;
    pACaptureSessionOutputContainer_add = NULL;
    pACaptureSessionOutputContainer_create = NULL;
    pACaptureSessionOutput_create = NULL;

    dlclose(libmediandk);
    libmediandk = NULL;
    pAImage_delete = NULL;
    pAImage_getTimestamp = NULL;
    pAImage_getNumberOfPlanes = NULL;
    pAImage_getPlaneRowStride = NULL;
    pAImage_getPlaneData = NULL;
    pAImageReader_acquireNextImage = NULL;
    pAImageReader_delete = NULL;
    pAImageReader_setImageListener = NULL;
    pAImageReader_getWindow = NULL;
    pAImageReader_new = NULL;
}

static bool ANDROIDCAMERA_Init(SDL_CameraDriverImpl *impl)
{
    // !!! FIXME: slide this off into a subroutine
    // system libraries are in android-24 and later; we currently target android-16 and later, so check if they exist at runtime.
    void *libcamera2 = dlopen("libcamera2ndk.so", RTLD_NOW | RTLD_LOCAL);
    if (!libcamera2) {
        SDL_Log("CAMERA: libcamera2ndk.so can't be loaded: %s", dlerror());
        return false;
    }

    void *libmedia = dlopen("libmediandk.so", RTLD_NOW | RTLD_LOCAL);
    if (!libmedia) {
        SDL_Log("CAMERA: libmediandk.so can't be loaded: %s", dlerror());
        dlclose(libcamera2);
        return false;
    }

    bool okay = true;
    #define LOADSYM(lib, fn) if (okay) { p##fn = (pfn##fn) dlsym(lib, #fn); if (!p##fn) { SDL_Log("CAMERA: symbol '%s' can't be found in %s: %s", #fn, #lib "ndk.so", dlerror()); okay = false; } }
    //#define LOADSYM(lib, fn) p##fn = (pfn##fn) fn
    LOADSYM(libcamera2, ACameraManager_create);
    LOADSYM(libcamera2, ACameraManager_registerAvailabilityCallback);
    LOADSYM(libcamera2, ACameraManager_unregisterAvailabilityCallback);
    LOADSYM(libcamera2, ACameraManager_getCameraIdList);
    LOADSYM(libcamera2, ACameraManager_deleteCameraIdList);
    LOADSYM(libcamera2, ACameraCaptureSession_close);
    LOADSYM(libcamera2, ACaptureRequest_free);
    LOADSYM(libcamera2, ACameraOutputTarget_free);
    LOADSYM(libcamera2, ACameraDevice_close);
    LOADSYM(libcamera2, ACameraManager_delete);
    LOADSYM(libcamera2, ACaptureSessionOutputContainer_free);
    LOADSYM(libcamera2, ACaptureSessionOutput_free);
    LOADSYM(libcamera2, ACameraManager_openCamera);
    LOADSYM(libcamera2, ACameraDevice_createCaptureRequest);
    LOADSYM(libcamera2, ACameraDevice_createCaptureSession);
    LOADSYM(libcamera2, ACameraManager_getCameraCharacteristics);
    LOADSYM(libcamera2, ACameraMetadata_free);
    LOADSYM(libcamera2, ACameraMetadata_getConstEntry);
    LOADSYM(libcamera2, ACameraCaptureSession_setRepeatingRequest);
    LOADSYM(libcamera2, ACameraOutputTarget_create);
    LOADSYM(libcamera2, ACaptureRequest_addTarget);
    LOADSYM(libcamera2, ACaptureSessionOutputContainer_add);
    LOADSYM(libcamera2, ACaptureSessionOutputContainer_create);
    LOADSYM(libcamera2, ACaptureSessionOutput_create);
    LOADSYM(libmedia, AImage_delete);
    LOADSYM(libmedia, AImage_getTimestamp);
    LOADSYM(libmedia, AImage_getNumberOfPlanes);
    LOADSYM(libmedia, AImage_getPlaneRowStride);
    LOADSYM(libmedia, AImage_getPlaneData);
    LOADSYM(libmedia, AImageReader_acquireNextImage);
    LOADSYM(libmedia, AImageReader_delete);
    LOADSYM(libmedia, AImageReader_setImageListener);
    LOADSYM(libmedia, AImageReader_getWindow);
    LOADSYM(libmedia, AImageReader_new);
    LOADSYM(libmedia, AImage_getWidth);
    LOADSYM(libmedia, AImage_getHeight);

    #undef LOADSYM

    if (!okay) {
        dlclose(libmedia);
        dlclose(libcamera2);
    }

    if (!CreateCameraManager()) {
        dlclose(libmedia);
        dlclose(libcamera2);
        return false;
    }

    libcamera2ndk = libcamera2;
    libmediandk = libmedia;

    impl->DetectDevices = ANDROIDCAMERA_DetectDevices;
    impl->OpenDevice = ANDROIDCAMERA_OpenDevice;
    impl->CloseDevice = ANDROIDCAMERA_CloseDevice;
    impl->WaitDevice = ANDROIDCAMERA_WaitDevice;
    impl->AcquireFrame = ANDROIDCAMERA_AcquireFrame;
    impl->ReleaseFrame = ANDROIDCAMERA_ReleaseFrame;
    impl->FreeDeviceHandle = ANDROIDCAMERA_FreeDeviceHandle;
    impl->Deinitialize = ANDROIDCAMERA_Deinitialize;

    impl->ProvidesOwnCallbackThread = true;

    return true;
}

CameraBootStrap ANDROIDCAMERA_bootstrap = {
    "android", "SDL Android camera driver", ANDROIDCAMERA_Init, false
};

#endif
