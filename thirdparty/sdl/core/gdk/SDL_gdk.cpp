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

extern "C" {
#include "../windows/SDL_windows.h"
#include "../../events/SDL_events_c.h"
}
#include <XGameRuntime.h>
#include <xsapi-c/services_c.h>
#include <appnotify.h>

static XTaskQueueHandle GDK_GlobalTaskQueue;

PAPPSTATE_REGISTRATION hPLM = {};
PAPPCONSTRAIN_REGISTRATION hCPLM = {};
HANDLE plmSuspendComplete = nullptr;

extern "C"
bool SDL_GetGDKTaskQueue(XTaskQueueHandle *outTaskQueue)
{
    // If this is the first call, first create the global task queue.
    if (!GDK_GlobalTaskQueue) {
        HRESULT hr;

        hr = XTaskQueueCreate(XTaskQueueDispatchMode::ThreadPool,
                              XTaskQueueDispatchMode::Manual,
                              &GDK_GlobalTaskQueue);
        if (FAILED(hr)) {
            return SDL_SetError("[GDK] Could not create global task queue");
        }

        // The initial call gets the non-duplicated handle so they can clean it up
        *outTaskQueue = GDK_GlobalTaskQueue;
    } else {
        // Duplicate the global task queue handle into outTaskQueue
        if (FAILED(XTaskQueueDuplicateHandle(GDK_GlobalTaskQueue, outTaskQueue))) {
            return SDL_SetError("[GDK] Unable to acquire global task queue");
        }
    }

    return true;
}

extern "C"
void GDK_DispatchTaskQueue(void)
{
    /* If there is no global task queue, don't do anything.
     * This gives the option to opt-out for those who want to handle everything themselves.
     */
    if (GDK_GlobalTaskQueue) {
        // Dispatch any callbacks which are ready.
        while (XTaskQueueDispatch(GDK_GlobalTaskQueue, XTaskQueuePort::Completion, 0))
            ;
    }
}

extern "C"
bool GDK_RegisterChangeNotifications(void)
{
    // Register suspend/resume handling
    plmSuspendComplete = CreateEventEx(nullptr, nullptr, 0, EVENT_MODIFY_STATE | SYNCHRONIZE);
    if (!plmSuspendComplete) {
        return SDL_SetError("[GDK] Unable to create plmSuspendComplete event");
    }
    auto rascn = [](BOOLEAN quiesced, PVOID context) {
        SDL_LogDebug(SDL_LOG_CATEGORY_APPLICATION, "[GDK] in RegisterAppStateChangeNotification handler");
        if (quiesced) {
            ResetEvent(plmSuspendComplete);
            SDL_SendAppEvent(SDL_EVENT_DID_ENTER_BACKGROUND);

            // To defer suspension, we must wait to exit this callback.
            // IMPORTANT: The app must call SDL_GDKSuspendComplete() to release this lock.
            (void)WaitForSingleObject(plmSuspendComplete, INFINITE);

            SDL_LogDebug(SDL_LOG_CATEGORY_APPLICATION, "[GDK] in RegisterAppStateChangeNotification handler: plmSuspendComplete event signaled.");
        } else {
            SDL_SendAppEvent(SDL_EVENT_WILL_ENTER_FOREGROUND);
        }
    };
    if (RegisterAppStateChangeNotification(rascn, NULL, &hPLM)) {
        return SDL_SetError("[GDK] Unable to call RegisterAppStateChangeNotification");
    }

    // Register constrain/unconstrain handling
    auto raccn = [](BOOLEAN constrained, PVOID context) {
        SDL_LogDebug(SDL_LOG_CATEGORY_APPLICATION, "[GDK] in RegisterAppConstrainedChangeNotification handler");
        SDL_VideoDevice *_this = SDL_GetVideoDevice();
        if (_this) {
            if (constrained) {
                SDL_SetKeyboardFocus(NULL);
            } else {
                SDL_SetKeyboardFocus(_this->windows);
            }
        }
    };
    if (RegisterAppConstrainedChangeNotification(raccn, NULL, &hCPLM)) {
        return SDL_SetError("[GDK] Unable to call RegisterAppConstrainedChangeNotification");
    }

    return true;
}

extern "C"
void GDK_UnregisterChangeNotifications(void)
{
    // Unregister suspend/resume handling
    UnregisterAppStateChangeNotification(hPLM);
    CloseHandle(plmSuspendComplete);

    // Unregister constrain/unconstrain handling
    UnregisterAppConstrainedChangeNotification(hCPLM);
}

extern "C"
void SDL_GDKSuspendComplete()
{
    if (plmSuspendComplete) {
        SetEvent(plmSuspendComplete);
    }
}

extern "C"
bool SDL_GetGDKDefaultUser(XUserHandle *outUserHandle)
{
    XAsyncBlock block = { 0 };
    HRESULT result;

    if (FAILED(result = XUserAddAsync(XUserAddOptions::AddDefaultUserAllowingUI, &block))) {
        return WIN_SetErrorFromHRESULT("XUserAddAsync", result);
    }

    do {
        result = XUserAddResult(&block, outUserHandle);
    } while (result == E_PENDING);
    if (FAILED(result)) {
        return WIN_SetErrorFromHRESULT("XUserAddResult", result);
    }

    return true;
}
