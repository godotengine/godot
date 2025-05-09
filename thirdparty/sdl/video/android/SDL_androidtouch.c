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

#ifdef SDL_VIDEO_DRIVER_ANDROID

#include <android/log.h>

#include "SDL_androidtouch.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_touch_c.h"
#include "../../core/android/SDL_android.h"

#define ACTION_DOWN 0
#define ACTION_UP   1
#define ACTION_MOVE 2
#define ACTION_CANCEL 3
// #define ACTION_OUTSIDE 4
#define ACTION_POINTER_DOWN 5
#define ACTION_POINTER_UP   6

void Android_InitTouch(void)
{
    // Add all touch devices
    Android_JNI_InitTouch();
}

void Android_QuitTouch(void)
{
}

void Android_OnTouch(SDL_Window *window, int touch_device_id_in, int pointer_finger_id_in, int action, float x, float y, float p)
{
    SDL_TouchID touchDeviceId = 0;
    SDL_FingerID fingerId = 0;

    if (!window) {
        return;
    }

    /* Touch device -1 appears when using Android emulator, eg:
     *  adb shell input mouse tap 100 100
     *  adb shell input touchscreen tap 100 100
     */
    touchDeviceId = (SDL_TouchID)(touch_device_id_in + 2);

    // Finger ID should be greater than 0
    fingerId = (SDL_FingerID)(pointer_finger_id_in + 1);

    if (SDL_AddTouch(touchDeviceId, SDL_TOUCH_DEVICE_DIRECT, "") < 0) {
        SDL_Log("error: can't add touch %s, %d", __FILE__, __LINE__);
    }

    switch (action) {
    case ACTION_DOWN:
    case ACTION_POINTER_DOWN:
        SDL_SendTouch(0, touchDeviceId, fingerId, window, SDL_EVENT_FINGER_DOWN, x, y, p);
        break;

    case ACTION_MOVE:
        SDL_SendTouchMotion(0, touchDeviceId, fingerId, window, x, y, p);
        break;

    case ACTION_UP:
    case ACTION_POINTER_UP:
        SDL_SendTouch(0, touchDeviceId, fingerId, window, SDL_EVENT_FINGER_UP, x, y, p);
        break;

    case ACTION_CANCEL:
        SDL_SendTouch(0, touchDeviceId, fingerId, window, SDL_EVENT_FINGER_CANCELED, x, y, p);
        break;

    default:
        break;
    }
}

#endif // SDL_VIDEO_DRIVER_ANDROID
