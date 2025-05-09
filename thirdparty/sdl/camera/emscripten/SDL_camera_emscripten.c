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

#ifdef SDL_CAMERA_DRIVER_EMSCRIPTEN

#include "../SDL_syscamera.h"
#include "../SDL_camera_c.h"
#include "../../video/SDL_pixels_c.h"
#include "../../video/SDL_surface_c.h"

#include <emscripten/emscripten.h>

// just turn off clang-format for this whole file, this INDENT_OFF stuff on
//  each EM_ASM section is ugly.
/* *INDENT-OFF* */ // clang-format off

EM_JS_DEPS(sdlcamera, "$dynCall");

static bool EMSCRIPTENCAMERA_WaitDevice(SDL_Camera *device)
{
    SDL_assert(!"This shouldn't be called");  // we aren't using SDL's internal thread.
    return false;
}

static SDL_CameraFrameResult EMSCRIPTENCAMERA_AcquireFrame(SDL_Camera *device, SDL_Surface *frame, Uint64 *timestampNS)
{
    void *rgba = SDL_malloc(device->actual_spec.width * device->actual_spec.height * 4);
    if (!rgba) {
        return SDL_CAMERA_FRAME_ERROR;
    }

    *timestampNS = SDL_GetTicksNS();  // best we can do here.

    const int rc = MAIN_THREAD_EM_ASM_INT({
        const w = $0;
        const h = $1;
        const rgba = $2;
        const SDL3 = Module['SDL3'];
        if ((typeof(SDL3) === 'undefined') || (typeof(SDL3.camera) === 'undefined') || (typeof(SDL3.camera.ctx2d) === 'undefined')) {
            return 0;  // don't have something we need, oh well.
        }

        SDL3.camera.ctx2d.drawImage(SDL3.camera.video, 0, 0, w, h);
        const imgrgba = SDL3.camera.ctx2d.getImageData(0, 0, w, h).data;
        Module.HEAPU8.set(imgrgba, rgba);

        return 1;
    }, device->actual_spec.width, device->actual_spec.height, rgba);

    if (!rc) {
        SDL_free(rgba);
        return SDL_CAMERA_FRAME_ERROR;  // something went wrong, maybe shutting down; just don't return a frame.
    }

    frame->pixels = rgba;
    frame->pitch = device->actual_spec.width * 4;

    return SDL_CAMERA_FRAME_READY;
}

static void EMSCRIPTENCAMERA_ReleaseFrame(SDL_Camera *device, SDL_Surface *frame)
{
    SDL_free(frame->pixels);
}

static void EMSCRIPTENCAMERA_CloseDevice(SDL_Camera *device)
{
    if (device) {
        MAIN_THREAD_EM_ASM({
            const SDL3 = Module['SDL3'];
            if ((typeof(SDL3) === 'undefined') || (typeof(SDL3.camera) === 'undefined') || (typeof(SDL3.camera.stream) === 'undefined')) {
                return;  // camera was closed and/or subsystem was shut down, we're already done.
            }
            SDL3.camera.stream.getTracks().forEach(track => track.stop());  // stop all recording.
            SDL3.camera = {};  // dump our references to everything.
        });
        SDL_free(device->hidden);
        device->hidden = NULL;
    }
}

static int SDLEmscriptenCameraPermissionOutcome(SDL_Camera *device, int approved, int w, int h, int fps)
{
    if (approved) {
        device->actual_spec.format = SDL_PIXELFORMAT_RGBA32;
        device->actual_spec.width = w;
        device->actual_spec.height = h;
        device->actual_spec.framerate_numerator = fps;
        device->actual_spec.framerate_denominator = 1;

        if (!SDL_PrepareCameraSurfaces(device)) {
            // uhoh, we're in trouble. Probably ran out of memory.
            SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Camera could not prepare surfaces: %s ... revoking approval!", SDL_GetError());
            approved = 0;  // disconnecting the SDL camera might not be safe here, just mark it as denied by user.
        }
    }

    SDL_CameraPermissionOutcome(device, approved ? true : false);
    return approved;
}

static bool EMSCRIPTENCAMERA_OpenDevice(SDL_Camera *device, const SDL_CameraSpec *spec)
{
    MAIN_THREAD_EM_ASM({
        // Since we can't get actual specs until we make a move that prompts the user for
        // permission, we don't list any specs for the device and wrangle it during device open.
        const device = $0;
        const w = $1;
        const h = $2;
        const framerate_numerator = $3;
        const framerate_denominator = $4;
        const outcome = $5;
        const iterate = $6;

        const constraints = {};
        if ((w <= 0) || (h <= 0)) {
            constraints.video = true;   // didn't ask for anything, let the system choose.
        } else {
            constraints.video = {};  // asked for a specific thing: request it as "ideal" but take closest hardware will offer.
            constraints.video.width = w;
            constraints.video.height = h;
        }

        if ((framerate_numerator > 0) && (framerate_denominator > 0)) {
            var fps = framerate_numerator / framerate_denominator;
            constraints.video.frameRate = { ideal: fps };
        }

        function grabNextCameraFrame() {  // !!! FIXME: this (currently) runs as a requestAnimationFrame callback, for lack of a better option.
            const SDL3 = Module['SDL3'];
            if ((typeof(SDL3) === 'undefined') || (typeof(SDL3.camera) === 'undefined') || (typeof(SDL3.camera.stream) === 'undefined')) {
                return;  // camera was closed and/or subsystem was shut down, stop iterating here.
            }

            // time for a new frame from the camera?
            const nextframems = SDL3.camera.next_frame_time;
            const now = performance.now();
            if (now >= nextframems) {
                dynCall('vi', iterate, [device]);  // calls SDL_CameraThreadIterate, which will call our AcquireFrame implementation.

                // bump ahead but try to stay consistent on timing, in case we dropped frames.
                while (SDL3.camera.next_frame_time < now) {
                    SDL3.camera.next_frame_time += SDL3.camera.fpsincrms;
                }
            }

            requestAnimationFrame(grabNextCameraFrame);  // run this function again at the display framerate.  (!!! FIXME: would this be better as requestIdleCallback?)
        }

        navigator.mediaDevices.getUserMedia(constraints)
            .then((stream) => {
                const settings = stream.getVideoTracks()[0].getSettings();
                const actualw = settings.width;
                const actualh = settings.height;
                const actualfps = settings.frameRate;
                console.log("Camera is opened! Actual spec: (" + actualw + "x" + actualh + "), fps=" + actualfps);

                if (dynCall('iiiiii', outcome, [device, 1, actualw, actualh, actualfps])) {
                    const video = document.createElement("video");
                    video.width = actualw;
                    video.height = actualh;
                    video.style.display = 'none';    // we need to attach this to a hidden video node so we can read it as pixels.
                    video.srcObject = stream;

                    const canvas = document.createElement("canvas");
                    canvas.width = actualw;
                    canvas.height = actualh;
                    canvas.style.display = 'none';    // we need to attach this to a hidden video node so we can read it as pixels.

                    const ctx2d = canvas.getContext('2d');

                    const SDL3 = Module['SDL3'];
                    SDL3.camera.width = actualw;
                    SDL3.camera.height = actualh;
                    SDL3.camera.fps = actualfps;
                    SDL3.camera.fpsincrms = 1000.0 / actualfps;
                    SDL3.camera.stream = stream;
                    SDL3.camera.video = video;
                    SDL3.camera.canvas = canvas;
                    SDL3.camera.ctx2d = ctx2d;
                    SDL3.camera.next_frame_time = performance.now();

                    video.play();
                    video.addEventListener('loadedmetadata', () => {
                        grabNextCameraFrame();  // start this loop going.
                    });
                }
            })
            .catch((err) => {
                console.error("Tried to open camera but it threw an error! " + err.name + ": " +  err.message);
                dynCall('iiiiii', outcome, [device, 0, 0, 0, 0]);   // we call this a permission error, because it probably is.
            });
    }, device, spec->width, spec->height, spec->framerate_numerator, spec->framerate_denominator, SDLEmscriptenCameraPermissionOutcome, SDL_CameraThreadIterate);

    return true;  // the real work waits until the user approves a camera.
}

static void EMSCRIPTENCAMERA_FreeDeviceHandle(SDL_Camera *device)
{
    // no-op.
}

static void EMSCRIPTENCAMERA_Deinitialize(void)
{
    MAIN_THREAD_EM_ASM({
        if (typeof(Module['SDL3']) !== 'undefined') {
            Module['SDL3'].camera = undefined;
        }
    });
}

static void EMSCRIPTENCAMERA_DetectDevices(void)
{
    // `navigator.mediaDevices` is not defined if unsupported or not in a secure context!
    const int supported = MAIN_THREAD_EM_ASM_INT({ return (navigator.mediaDevices === undefined) ? 0 : 1; });

    // if we have support at all, report a single generic camera with no specs.
    //  We'll find out if there really _is_ a camera when we try to open it, but querying it for real here
    //  will pop up a user permission dialog warning them we're trying to access the camera, and we generally
    //  don't want that during SDL_Init().
    if (supported) {
        SDL_AddCamera("Web browser's camera", SDL_CAMERA_POSITION_UNKNOWN, 0, NULL, (void *) (size_t) 0x1);
    }
}

static bool EMSCRIPTENCAMERA_Init(SDL_CameraDriverImpl *impl)
{
    MAIN_THREAD_EM_ASM({
        if (typeof(Module['SDL3']) === 'undefined') {
            Module['SDL3'] = {};
        }
        Module['SDL3'].camera = {};
    });

    impl->DetectDevices = EMSCRIPTENCAMERA_DetectDevices;
    impl->OpenDevice = EMSCRIPTENCAMERA_OpenDevice;
    impl->CloseDevice = EMSCRIPTENCAMERA_CloseDevice;
    impl->WaitDevice = EMSCRIPTENCAMERA_WaitDevice;
    impl->AcquireFrame = EMSCRIPTENCAMERA_AcquireFrame;
    impl->ReleaseFrame = EMSCRIPTENCAMERA_ReleaseFrame;
    impl->FreeDeviceHandle = EMSCRIPTENCAMERA_FreeDeviceHandle;
    impl->Deinitialize = EMSCRIPTENCAMERA_Deinitialize;

    impl->ProvidesOwnCallbackThread = true;

    return true;
}

CameraBootStrap EMSCRIPTENCAMERA_bootstrap = {
    "emscripten", "SDL Emscripten MediaStream camera driver", EMSCRIPTENCAMERA_Init, false
};

/* *INDENT-ON* */ // clang-format on

#endif // SDL_CAMERA_DRIVER_EMSCRIPTEN

