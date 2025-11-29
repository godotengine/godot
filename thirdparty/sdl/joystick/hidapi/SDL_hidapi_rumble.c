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

#ifdef SDL_JOYSTICK_HIDAPI

// Handle rumble on a separate thread so it doesn't block the application

#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"
#include "../../thread/SDL_systhread.h"

typedef struct SDL_HIDAPI_RumbleRequest
{
    SDL_HIDAPI_Device *device;
    Uint8 data[2 * USB_PACKET_LENGTH]; // need enough space for the biggest report: dualshock4 is 78 bytes
    int size;
    SDL_HIDAPI_RumbleSentCallback callback;
    void *userdata;
    struct SDL_HIDAPI_RumbleRequest *prev;

} SDL_HIDAPI_RumbleRequest;

typedef struct SDL_HIDAPI_RumbleContext
{
    SDL_AtomicInt initialized;
    SDL_AtomicInt running;
    SDL_Thread *thread;
    SDL_Semaphore *request_sem;
    SDL_HIDAPI_RumbleRequest *requests_head;
    SDL_HIDAPI_RumbleRequest *requests_tail;
} SDL_HIDAPI_RumbleContext;

#ifndef SDL_THREAD_SAFETY_ANALYSIS
static
#endif
SDL_Mutex *SDL_HIDAPI_rumble_lock;
static SDL_HIDAPI_RumbleContext rumble_context SDL_GUARDED_BY(SDL_HIDAPI_rumble_lock);

static int SDLCALL SDL_HIDAPI_RumbleThread(void *data)
{
    SDL_HIDAPI_RumbleContext *ctx = (SDL_HIDAPI_RumbleContext *)data;

    SDL_SetCurrentThreadPriority(SDL_THREAD_PRIORITY_HIGH);

    while (SDL_GetAtomicInt(&ctx->running)) {
        SDL_HIDAPI_RumbleRequest *request = NULL;

        SDL_WaitSemaphore(ctx->request_sem);

        SDL_LockMutex(SDL_HIDAPI_rumble_lock);
        request = ctx->requests_tail;
        if (request) {
            if (request == ctx->requests_head) {
                ctx->requests_head = NULL;
            }
            ctx->requests_tail = request->prev;
        }
        SDL_UnlockMutex(SDL_HIDAPI_rumble_lock);

        if (request) {
            SDL_LockMutex(request->device->dev_lock);
            if (request->device->dev) {
#ifdef DEBUG_RUMBLE
                HIDAPI_DumpPacket("Rumble packet: size = %d", request->data, request->size);
#endif
                SDL_hid_write(request->device->dev, request->data, request->size);
            }
            SDL_UnlockMutex(request->device->dev_lock);
            if (request->callback) {
                request->callback(request->userdata);
            }
            (void)SDL_AtomicDecRef(&request->device->rumble_pending);
            SDL_free(request);

            // Make sure we're not starving report reads when there's lots of rumble
            SDL_Delay(10);
        }
    }
    return 0;
}

static void SDL_HIDAPI_StopRumbleThread(SDL_HIDAPI_RumbleContext *ctx)
{
    SDL_HIDAPI_RumbleRequest *request;

    SDL_SetAtomicInt(&ctx->running, false);

    if (ctx->thread) {
        int result;

        SDL_SignalSemaphore(ctx->request_sem);
        SDL_WaitThread(ctx->thread, &result);
        ctx->thread = NULL;
    }

    SDL_LockMutex(SDL_HIDAPI_rumble_lock);
    while (ctx->requests_tail) {
        request = ctx->requests_tail;
        if (request == ctx->requests_head) {
            ctx->requests_head = NULL;
        }
        ctx->requests_tail = request->prev;

        if (request->callback) {
            request->callback(request->userdata);
        }
        (void)SDL_AtomicDecRef(&request->device->rumble_pending);
        SDL_free(request);
    }
    SDL_UnlockMutex(SDL_HIDAPI_rumble_lock);

    if (ctx->request_sem) {
        SDL_DestroySemaphore(ctx->request_sem);
        ctx->request_sem = NULL;
    }

    if (SDL_HIDAPI_rumble_lock) {
        SDL_DestroyMutex(SDL_HIDAPI_rumble_lock);
        SDL_HIDAPI_rumble_lock = NULL;
    }

    SDL_SetAtomicInt(&ctx->initialized, false);
}

static bool SDL_HIDAPI_StartRumbleThread(SDL_HIDAPI_RumbleContext *ctx)
{
    SDL_HIDAPI_rumble_lock = SDL_CreateMutex();
    if (!SDL_HIDAPI_rumble_lock) {
        SDL_HIDAPI_StopRumbleThread(ctx);
        return false;
    }

    ctx->request_sem = SDL_CreateSemaphore(0);
    if (!ctx->request_sem) {
        SDL_HIDAPI_StopRumbleThread(ctx);
        return false;
    }

    SDL_SetAtomicInt(&ctx->running, true);
    ctx->thread = SDL_CreateThread(SDL_HIDAPI_RumbleThread, "HIDAPI Rumble", ctx);
    if (!ctx->thread) {
        SDL_HIDAPI_StopRumbleThread(ctx);
        return false;
    }
    return true;
}

bool SDL_HIDAPI_LockRumble(void)
{
    SDL_HIDAPI_RumbleContext *ctx = &rumble_context;

    if (SDL_CompareAndSwapAtomicInt(&ctx->initialized, false, true)) {
        if (!SDL_HIDAPI_StartRumbleThread(ctx)) {
            return false;
        }
    }

    SDL_LockMutex(SDL_HIDAPI_rumble_lock);
    return true;
}

bool SDL_HIDAPI_GetPendingRumbleLocked(SDL_HIDAPI_Device *device, Uint8 **data, int **size, int *maximum_size)
{
    SDL_HIDAPI_RumbleContext *ctx = &rumble_context;
    SDL_HIDAPI_RumbleRequest *request, *found;

    found = NULL;
    for (request = ctx->requests_tail; request; request = request->prev) {
        if (request->device == device) {
            found = request;
        }
    }
    if (found) {
        *data = found->data;
        *size = &found->size;
        *maximum_size = sizeof(found->data);
        return true;
    }
    return false;
}

int SDL_HIDAPI_SendRumbleAndUnlock(SDL_HIDAPI_Device *device, const Uint8 *data, int size)
{
    return SDL_HIDAPI_SendRumbleWithCallbackAndUnlock(device, data, size, NULL, NULL);
}

int SDL_HIDAPI_SendRumbleWithCallbackAndUnlock(SDL_HIDAPI_Device *device, const Uint8 *data, int size, SDL_HIDAPI_RumbleSentCallback callback, void *userdata)
{
    SDL_HIDAPI_RumbleContext *ctx = &rumble_context;
    SDL_HIDAPI_RumbleRequest *request;

    if (size > sizeof(request->data)) {
        SDL_HIDAPI_UnlockRumble();
        SDL_SetError("Couldn't send rumble, size %d is greater than %d", size, (int)sizeof(request->data));
        return -1;
    }

    request = (SDL_HIDAPI_RumbleRequest *)SDL_calloc(1, sizeof(*request));
    if (!request) {
        SDL_HIDAPI_UnlockRumble();
        return -1;
    }
    request->device = device;
    SDL_memcpy(request->data, data, size);
    request->size = size;
    request->callback = callback;
    request->userdata = userdata;

    SDL_AtomicIncRef(&device->rumble_pending);

    if (ctx->requests_head) {
        ctx->requests_head->prev = request;
    } else {
        ctx->requests_tail = request;
    }
    ctx->requests_head = request;

    // Make sure we unlock before posting the semaphore so the rumble thread can run immediately
    SDL_HIDAPI_UnlockRumble();

    SDL_SignalSemaphore(ctx->request_sem);

    return size;
}

void SDL_HIDAPI_UnlockRumble(void)
{
    SDL_UnlockMutex(SDL_HIDAPI_rumble_lock);
}

int SDL_HIDAPI_SendRumble(SDL_HIDAPI_Device *device, const Uint8 *data, int size)
{
    Uint8 *pending_data;
    int *pending_size;
    int maximum_size;

    if (size <= 0) {
        SDL_SetError("Tried to send rumble with invalid size");
        return -1;
    }

    if (!SDL_HIDAPI_LockRumble()) {
        return -1;
    }

    // check if there is a pending request for the device and update it
    if (SDL_HIDAPI_GetPendingRumbleLocked(device, &pending_data, &pending_size, &maximum_size) &&
        size == *pending_size && data[0] == pending_data[0]) {
        SDL_memcpy(pending_data, data, size);
        SDL_HIDAPI_UnlockRumble();
        return size;
    }

    return SDL_HIDAPI_SendRumbleAndUnlock(device, data, size);
}

void SDL_HIDAPI_QuitRumble(void)
{
    SDL_HIDAPI_RumbleContext *ctx = &rumble_context;

    if (SDL_GetAtomicInt(&ctx->running)) {
        SDL_HIDAPI_StopRumbleThread(ctx);
    }
}

#endif // SDL_JOYSTICK_HIDAPI
