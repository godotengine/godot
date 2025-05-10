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

#ifdef SDL_CAMERA_DRIVER_VITA

#include "../SDL_syscamera.h"
#include <psp2/camera.h>
#include <psp2/kernel/sysmem.h>

static struct  {
    Sint32 w;
    Sint32 h;
    Sint32 res;
} resolutions[] = {
    {640, 480, SCE_CAMERA_RESOLUTION_640_480},
    {320, 240, SCE_CAMERA_RESOLUTION_320_240},
    {160, 120, SCE_CAMERA_RESOLUTION_160_120},
    {352, 288, SCE_CAMERA_RESOLUTION_352_288},
    {176, 144, SCE_CAMERA_RESOLUTION_176_144},
    {480, 272, SCE_CAMERA_RESOLUTION_480_272},
    {640, 360, SCE_CAMERA_RESOLUTION_640_360},
    {0, 0, 0}
};

static Sint32 fps[] = {5, 10, 15, 20, 24, 25, 30, 60, 0};

static void GatherCameraSpecs(Sint32 devid, CameraFormatAddData *add_data, char **fullname, SDL_CameraPosition *position)
{
    SDL_zerop(add_data);

    if (devid == SCE_CAMERA_DEVICE_FRONT) {
        *position = SDL_CAMERA_POSITION_FRONT_FACING;
        *fullname = SDL_strdup("Front-facing camera");
    } else if (devid == SCE_CAMERA_DEVICE_BACK) {
        *position = SDL_CAMERA_POSITION_BACK_FACING;
        *fullname = SDL_strdup("Back-facing camera");
    }

    if (!*fullname) {
        *fullname = SDL_strdup("Generic camera");
    }

    // Note: there are actually more fps and pixelformats. Planar YUV is fastest. Support only YUV and integer fps for now
    Sint32 idx = 0;
    while (resolutions[idx].res > 0) {
        Sint32 fps_idx = 0;
        while (fps[fps_idx] > 0) {
            SDL_AddCameraFormat(add_data, SDL_PIXELFORMAT_IYUV, SDL_COLORSPACE_BT601_LIMITED, resolutions[idx].w, resolutions[idx].h, fps[fps_idx], 1); /* SCE_CAMERA_FORMAT_ARGB */
            fps_idx++;
        }
        idx++;
    }
}

static bool FindVitaCameraByID(SDL_Camera *device, void *userdata)
{
    Sint32 devid = (Sint32) userdata;
    return (devid == (Sint32)device->handle);
}

static void MaybeAddDevice(Sint32 devid)
{
    #if DEBUG_CAMERA
    SDL_Log("CAMERA: MaybeAddDevice('%d')", devid);
    #endif

    if (SDL_FindPhysicalCameraByCallback(FindVitaCameraByID, (void *) devid)) {
        return;  // already have this one.
    }

    SDL_CameraPosition position = SDL_CAMERA_POSITION_UNKNOWN;
    char *fullname = NULL;
    CameraFormatAddData add_data;
    GatherCameraSpecs(devid, &add_data, &fullname, &position);

    if (add_data.num_specs > 0) {
        SDL_AddCamera(fullname, position, add_data.num_specs, add_data.specs, (void*)devid);
    }

    SDL_free(fullname);
    SDL_free(add_data.specs);
}

static SceUID imbUid = -1;

static void freeBuffers(SceCameraInfo* info)
{
    if (imbUid != -1) {
        sceKernelFreeMemBlock(imbUid);
        info->pIBase = NULL;
        imbUid = -1;
    }
}

static bool VITACAMERA_OpenDevice(SDL_Camera *device, const SDL_CameraSpec *spec)
{
    // we can't open more than one camera, so error-out early
    if (imbUid != -1) {
        return SDL_SetError("Only one camera can be active");
    }

    SceCameraInfo* info = (SceCameraInfo*)SDL_calloc(1, sizeof(SceCameraInfo));

    info->size = sizeof(SceCameraInfo);
    info->priority = SCE_CAMERA_PRIORITY_SHARE;
    info->buffer = 0; // target buffer set by sceCameraOpen

    info->framerate = spec->framerate_numerator / spec->framerate_denominator;

    Sint32 idx = 0;
    while (resolutions[idx].res > 0) {
        if (spec->width == resolutions[idx].w && spec->height == resolutions[idx].h) {
            info->resolution = resolutions[idx].res;
            break;
        }
        idx++;
    }

    info->range = 1;
    info->format = SCE_CAMERA_FORMAT_YUV420_PLANE;
    info->pitch = 0; // same size surface

    info->sizeIBase =  spec->width*spec->height;;
    info->sizeUBase =  ((spec->width+1)/2) * ((spec->height+1) / 2);
    info->sizeVBase =  ((spec->width+1)/2) * ((spec->height+1) / 2);

    // PHYCONT memory size *must* be a multiple of 1MB, we can just always spend 2MB, since we don't use PHYCONT anywhere else
    imbUid = sceKernelAllocMemBlock("CameraI", SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_NC_RW, 2*1024*1024 , NULL);
    if (imbUid < 0)
    {
        return SDL_SetError("sceKernelAllocMemBlock error: 0x%08X", imbUid);
    }
    sceKernelGetMemBlockBase(imbUid, &(info->pIBase));

    info->pUBase = info->pIBase + info->sizeIBase;
    info->pVBase = info->pIBase + (info->sizeIBase + info->sizeUBase);

    device->hidden = (struct SDL_PrivateCameraData *)info;

    int ret = sceCameraOpen((int)device->handle, info);
    if (ret == 0) {
        ret = sceCameraStart((int)device->handle);
        if (ret == 0) {
            SDL_CameraPermissionOutcome(device, true);
            return true;
        } else {
            SDL_SetError("sceCameraStart error: 0x%08X", imbUid);
        }
    } else {
        SDL_SetError("sceCameraOpen error: 0x%08X", imbUid);
    }

    freeBuffers(info);

    return false;
}

static void VITACAMERA_CloseDevice(SDL_Camera *device)
{
    if (device->hidden) {
        sceCameraStop((int)device->handle);
        sceCameraClose((int)device->handle);
        freeBuffers((SceCameraInfo*)device->hidden);
        SDL_free(device->hidden);
    }
}

static bool VITACAMERA_WaitDevice(SDL_Camera *device)
{
    while(!sceCameraIsActive((int)device->handle)) {}
    return true;
}

static SDL_CameraFrameResult VITACAMERA_AcquireFrame(SDL_Camera *device, SDL_Surface *frame, Uint64 *timestampNS)
{
    SceCameraRead read = {0};
    read.size = sizeof(SceCameraRead);
    read.mode = 1; // don't wait next frame

    int ret = sceCameraRead((int)device->handle, &read);

    if (ret < 0) {
        SDL_SetError("sceCameraRead error: 0x%08X", ret);
        return SDL_CAMERA_FRAME_ERROR;
    }

    *timestampNS = read.timestamp;

    SceCameraInfo* info = (SceCameraInfo*)(device->hidden);

    frame->pitch = info->width;
    frame->pixels = SDL_aligned_alloc(SDL_GetSIMDAlignment(), info->sizeIBase + info->sizeUBase + info->sizeVBase);

    if (frame->pixels) {
        SDL_memcpy(frame->pixels, info->pIBase, info->sizeIBase + info->sizeUBase + info->sizeVBase);
        return SDL_CAMERA_FRAME_READY;
    }

    return SDL_CAMERA_FRAME_ERROR;
}

static void VITACAMERA_ReleaseFrame(SDL_Camera *device, SDL_Surface *frame)
{
    SDL_aligned_free(frame->pixels);
}

static void VITACAMERA_DetectDevices(void)
{
    MaybeAddDevice(SCE_CAMERA_DEVICE_FRONT);
    MaybeAddDevice(SCE_CAMERA_DEVICE_BACK);
}

static void VITACAMERA_FreeDeviceHandle(SDL_Camera *device)
{
}

static void VITACAMERA_Deinitialize(void)
{
}

static bool VITACAMERA_Init(SDL_CameraDriverImpl *impl)
{
    impl->DetectDevices = VITACAMERA_DetectDevices;
    impl->OpenDevice = VITACAMERA_OpenDevice;
    impl->CloseDevice = VITACAMERA_CloseDevice;
    impl->WaitDevice = VITACAMERA_WaitDevice;
    impl->AcquireFrame = VITACAMERA_AcquireFrame;
    impl->ReleaseFrame = VITACAMERA_ReleaseFrame;
    impl->FreeDeviceHandle = VITACAMERA_FreeDeviceHandle;
    impl->Deinitialize = VITACAMERA_Deinitialize;

    return true;
}

CameraBootStrap VITACAMERA_bootstrap = {
    "vita", "SDL PSVita camera driver", VITACAMERA_Init, false
};

#endif  // SDL_CAMERA_DRIVER_VITA
