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

#if defined(SDL_JOYSTICK_DINPUT) || defined(SDL_JOYSTICK_XINPUT)

/* DirectInput joystick driver; written by Glenn Maynard, based on Andrei de
 * A. Formiga's WINMM driver.
 *
 * Hats and sliders are completely untested; the app I'm writing this for mostly
 * doesn't use them and I don't own any joysticks with them.
 *
 * We don't bother to use event notification here.  It doesn't seem to work
 * with polled devices, and it's fine to call IDirectInputDevice8_GetDeviceData and
 * let it return 0 events. */

#include "../SDL_sysjoystick.h"
#include "../../thread/SDL_systhread.h"
#include "../../core/windows/SDL_windows.h"
#include "../../core/windows/SDL_hid.h"
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
#include <dbt.h>
#endif

#define INITGUID // Only set here, if set twice will cause mingw32 to break.
#include "SDL_windowsjoystick_c.h"
#include "SDL_dinputjoystick_c.h"
#include "SDL_xinputjoystick_c.h"
#include "SDL_rawinputjoystick_c.h"

#include "../../haptic/windows/SDL_dinputhaptic_c.h" // For haptic hot plugging

#ifndef DEVICE_NOTIFY_WINDOW_HANDLE
#define DEVICE_NOTIFY_WINDOW_HANDLE 0x00000000
#endif

// local variables
static bool s_bJoystickThread = false;
static SDL_Condition *s_condJoystickThread = NULL;
static SDL_Mutex *s_mutexJoyStickEnum = NULL;
static SDL_Thread *s_joystickThread = NULL;
static bool s_bJoystickThreadQuit = false;
static Uint64 s_lastDeviceChange = 0;
static GUID GUID_DEVINTERFACE_HID = { 0x4D1E55B2L, 0xF16F, 0x11CF, { 0x88, 0xCB, 0x00, 0x11, 0x11, 0x00, 0x00, 0x30 } };

JoyStick_DeviceData *SYS_Joystick; // array to hold joystick ID values


static bool WindowsDeviceChanged(void)
{
    return (s_lastDeviceChange != WIN_GetLastDeviceNotification());
}

static void SetWindowsDeviceChanged(void)
{
    s_lastDeviceChange = 0;
}

void WINDOWS_RAWINPUTEnabledChanged(void)
{
    SetWindowsDeviceChanged();
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

typedef struct
{
    HRESULT coinitialized;
    WNDCLASSEX wincl;
    HWND messageWindow;
    HDEVNOTIFY hNotify;
} SDL_DeviceNotificationData;

#define IDT_SDL_DEVICE_CHANGE_TIMER_1 1200
#define IDT_SDL_DEVICE_CHANGE_TIMER_2 1201

// windowproc for our joystick detect thread message only window, to detect any USB device addition/removal
static LRESULT CALLBACK SDL_PrivateJoystickDetectProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_DEVICECHANGE:
        switch (wParam) {
        case DBT_DEVICEARRIVAL:
        case DBT_DEVICEREMOVECOMPLETE:
            if (((DEV_BROADCAST_HDR *)lParam)->dbch_devicetype == DBT_DEVTYP_DEVICEINTERFACE) {
                // notify 300ms and 2 seconds later to ensure all APIs have updated status
                SetTimer(hwnd, IDT_SDL_DEVICE_CHANGE_TIMER_1, 300, NULL);
                SetTimer(hwnd, IDT_SDL_DEVICE_CHANGE_TIMER_2, 2000, NULL);
            }
            break;
        }
        return true;
    case WM_TIMER:
        if (wParam == IDT_SDL_DEVICE_CHANGE_TIMER_1 ||
            wParam == IDT_SDL_DEVICE_CHANGE_TIMER_2) {
            KillTimer(hwnd, wParam);
            SetWindowsDeviceChanged();
            return true;
        }
        break;
    }

#ifdef SDL_JOYSTICK_RAWINPUT
    return CallWindowProc(RAWINPUT_WindowProc, hwnd, msg, wParam, lParam);
#else
    return CallWindowProc(DefWindowProc, hwnd, msg, wParam, lParam);
#endif
}

static void SDL_CleanupDeviceNotification(SDL_DeviceNotificationData *data)
{
#ifdef SDL_JOYSTICK_RAWINPUT
    RAWINPUT_UnregisterNotifications();
#endif

    if (data->hNotify) {
        UnregisterDeviceNotification(data->hNotify);
    }

    if (data->messageWindow) {
        DestroyWindow(data->messageWindow);
    }

    UnregisterClass(data->wincl.lpszClassName, data->wincl.hInstance);

    if (data->coinitialized == S_OK) {
        WIN_CoUninitialize();
    }
}

static bool SDL_CreateDeviceNotification(SDL_DeviceNotificationData *data)
{
    DEV_BROADCAST_DEVICEINTERFACE dbh;

    SDL_zerop(data);

    data->coinitialized = WIN_CoInitialize();

    data->wincl.hInstance = GetModuleHandle(NULL);
    data->wincl.lpszClassName = TEXT("Message");
    data->wincl.lpfnWndProc = SDL_PrivateJoystickDetectProc; // This function is called by windows
    data->wincl.cbSize = sizeof(WNDCLASSEX);

    if (!RegisterClassEx(&data->wincl)) {
        WIN_SetError("Failed to create register class for joystick autodetect");
        SDL_CleanupDeviceNotification(data);
        return false;
    }

    data->messageWindow = CreateWindowEx(0, TEXT("Message"), NULL, 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, NULL, NULL);
    if (!data->messageWindow) {
        WIN_SetError("Failed to create message window for joystick autodetect");
        SDL_CleanupDeviceNotification(data);
        return false;
    }

    SDL_zero(dbh);
    dbh.dbcc_size = sizeof(dbh);
    dbh.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE;
    dbh.dbcc_classguid = GUID_DEVINTERFACE_HID;

    data->hNotify = RegisterDeviceNotification(data->messageWindow, &dbh, DEVICE_NOTIFY_WINDOW_HANDLE);
    if (!data->hNotify) {
        WIN_SetError("Failed to create notify device for joystick autodetect");
        SDL_CleanupDeviceNotification(data);
        return false;
    }

#ifdef SDL_JOYSTICK_RAWINPUT
    RAWINPUT_RegisterNotifications(data->messageWindow);
#endif
    return true;
}

static bool SDL_WaitForDeviceNotification(SDL_DeviceNotificationData *data, SDL_Mutex *mutex)
{
    MSG msg;
    int lastret = 1;

    if (!data->messageWindow) {
        return false; // device notifications require a window
    }

    SDL_UnlockMutex(mutex);
    while (lastret > 0 && !WindowsDeviceChanged()) {
        lastret = GetMessage(&msg, NULL, 0, 0); // WM_QUIT causes return value of 0
        if (lastret > 0) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }
    SDL_LockMutex(mutex);
    return (lastret != -1);
}

#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
static SDL_DeviceNotificationData s_notification_data;
#endif

// Function/thread to scan the system for joysticks.
static int SDLCALL SDL_JoystickThread(void *_data)
{
#ifdef SDL_JOYSTICK_XINPUT
    bool bOpenedXInputDevices[XUSER_MAX_COUNT];
    SDL_zeroa(bOpenedXInputDevices);
#endif

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    if (!SDL_CreateDeviceNotification(&s_notification_data)) {
        return 0;
    }
#endif

    SDL_LockMutex(s_mutexJoyStickEnum);
    while (s_bJoystickThreadQuit == false) {
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
        if (SDL_WaitForDeviceNotification(&s_notification_data, s_mutexJoyStickEnum) == false) {
#else
        {
#endif
#ifdef SDL_JOYSTICK_XINPUT
            // WM_DEVICECHANGE not working, poll for new XINPUT controllers
            SDL_WaitConditionTimeout(s_condJoystickThread, s_mutexJoyStickEnum, 1000);
            if (SDL_XINPUT_Enabled()) {
                // scan for any change in XInput devices
                Uint8 userId;
                for (userId = 0; userId < XUSER_MAX_COUNT; userId++) {
                    XINPUT_CAPABILITIES capabilities;
                    const DWORD result = XINPUTGETCAPABILITIES(userId, XINPUT_FLAG_GAMEPAD, &capabilities);
                    const bool available = (result == ERROR_SUCCESS);
                    if (bOpenedXInputDevices[userId] != available) {
                        SetWindowsDeviceChanged();
                        bOpenedXInputDevices[userId] = available;
                    }
                }
            }
#else
            // WM_DEVICECHANGE not working, no XINPUT, no point in keeping thread alive
            break;
#endif // SDL_JOYSTICK_XINPUT
        }
    }

    SDL_UnlockMutex(s_mutexJoyStickEnum);

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    SDL_CleanupDeviceNotification(&s_notification_data);
#endif

    return 1;
}

// spin up the thread to detect hotplug of devices
static bool SDL_StartJoystickThread(void)
{
    s_mutexJoyStickEnum = SDL_CreateMutex();
    if (!s_mutexJoyStickEnum) {
        return false;
    }

    s_condJoystickThread = SDL_CreateCondition();
    if (!s_condJoystickThread) {
        return false;
    }

    s_bJoystickThreadQuit = false;
    s_joystickThread = SDL_CreateThread(SDL_JoystickThread, "SDL_joystick", NULL);
    if (!s_joystickThread) {
        return false;
    }
    return true;
}

static void SDL_StopJoystickThread(void)
{
    if (!s_joystickThread) {
        return;
    }

    SDL_LockMutex(s_mutexJoyStickEnum);
    s_bJoystickThreadQuit = true;
    SDL_BroadcastCondition(s_condJoystickThread); // signal the joystick thread to quit
    SDL_UnlockMutex(s_mutexJoyStickEnum);
    PostThreadMessage((DWORD)SDL_GetThreadID(s_joystickThread), WM_QUIT, 0, 0);

    // Unlock joysticks while the joystick thread finishes processing messages
    SDL_AssertJoysticksLocked();
    SDL_UnlockJoysticks();
    SDL_WaitThread(s_joystickThread, NULL); // wait for it to bugger off
    SDL_LockJoysticks();

    SDL_DestroyCondition(s_condJoystickThread);
    s_condJoystickThread = NULL;

    SDL_DestroyMutex(s_mutexJoyStickEnum);
    s_mutexJoyStickEnum = NULL;

    s_joystickThread = NULL;
}

void WINDOWS_AddJoystickDevice(JoyStick_DeviceData *device)
{
    device->send_add_event = true;
    device->nInstanceID = SDL_GetNextObjectID();
    device->pNext = SYS_Joystick;
    SYS_Joystick = device;
}

void WINDOWS_JoystickDetect(void);
void WINDOWS_JoystickQuit(void);

static bool WINDOWS_JoystickInit(void)
{
    if (!SDL_XINPUT_JoystickInit()) {
        WINDOWS_JoystickQuit();
        return false;
    }

    if (!SDL_DINPUT_JoystickInit()) {
        WINDOWS_JoystickQuit();
        return false;
    }

    WIN_InitDeviceNotification();

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    s_bJoystickThread = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_THREAD, true);
    if (s_bJoystickThread) {
        if (!SDL_StartJoystickThread()) {
            return false;
        }
    } else {
        if (!SDL_CreateDeviceNotification(&s_notification_data)) {
            return false;
        }
    }
#endif

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    // On Xbox, force create the joystick thread for device detection (since other methods don't work
    s_bJoystickThread = true;
    if (!SDL_StartJoystickThread()) {
        return false;
    }
#endif

    SetWindowsDeviceChanged(); // force a scan of the system for joysticks this first time

    WINDOWS_JoystickDetect();

    return true;
}

// return the number of joysticks that are connected right now
static int WINDOWS_JoystickGetCount(void)
{
    int nJoysticks = 0;
    JoyStick_DeviceData *device = SYS_Joystick;
    while (device) {
        nJoysticks++;
        device = device->pNext;
    }

    return nJoysticks;
}

// detect any new joysticks being inserted into the system
void WINDOWS_JoystickDetect(void)
{
    JoyStick_DeviceData *pCurList = NULL;

    // only enum the devices if the joystick thread told us something changed
    if (!WindowsDeviceChanged()) {
        return; // thread hasn't signaled, nothing to do right now.
    }

    if (s_mutexJoyStickEnum) {
        SDL_LockMutex(s_mutexJoyStickEnum);
    }

    s_lastDeviceChange = WIN_GetLastDeviceNotification();

    pCurList = SYS_Joystick;
    SYS_Joystick = NULL;

    // Look for DirectInput joysticks, wheels, head trackers, gamepads, etc..
    SDL_DINPUT_JoystickDetect(&pCurList);

    // Look for XInput devices. Do this last, so they're first in the final list.
    SDL_XINPUT_JoystickDetect(&pCurList);

    if (s_mutexJoyStickEnum) {
        SDL_UnlockMutex(s_mutexJoyStickEnum);
    }

    while (pCurList) {
        JoyStick_DeviceData *pListNext = NULL;

        if (!pCurList->bXInputDevice) {
#ifdef SDL_HAPTIC_DINPUT
            SDL_DINPUT_HapticMaybeRemoveDevice(&pCurList->dxdevice);
#endif
        }

        SDL_PrivateJoystickRemoved(pCurList->nInstanceID);

        pListNext = pCurList->pNext;
        SDL_free(pCurList->joystickname);
        SDL_free(pCurList);
        pCurList = pListNext;
    }

    for (pCurList = SYS_Joystick; pCurList; pCurList = pCurList->pNext) {
        if (pCurList->send_add_event) {
            if (!pCurList->bXInputDevice) {
#ifdef SDL_HAPTIC_DINPUT
                SDL_DINPUT_HapticMaybeAddDevice(&pCurList->dxdevice);
#endif
            }

            SDL_PrivateJoystickAdded(pCurList->nInstanceID);

            pCurList->send_add_event = false;
        }
    }
}

static bool WINDOWS_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    if (SDL_DINPUT_JoystickPresent(vendor_id, product_id, version)) {
        return true;
    }
    if (SDL_XINPUT_JoystickPresent(vendor_id, product_id, version)) {
        return true;
    }
    return false;
}

static const char *WINDOWS_JoystickGetDeviceName(int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    return device->joystickname;
}

static const char *WINDOWS_JoystickGetDevicePath(int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    return device->path;
}

static int WINDOWS_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    if (device->bXInputDevice) {
        // The slot for XInput devices can change as controllers are seated
        return SDL_XINPUT_GetSteamVirtualGamepadSlot(device->XInputUserId);
    } else {
        return device->steam_virtual_gamepad_slot;
    }
}

static int WINDOWS_JoystickGetDevicePlayerIndex(int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    return device->bXInputDevice ? (int)device->XInputUserId : -1;
}

static void WINDOWS_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

// return the stable device guid for this device index
static SDL_GUID WINDOWS_JoystickGetDeviceGUID(int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    return device->guid;
}

// Function to perform the mapping between current device instance and this joysticks instance id
static SDL_JoystickID WINDOWS_JoystickGetDeviceInstanceID(int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    return device->nInstanceID;
}

/* Function to open a joystick for use.
   The joystick to open is specified by the device index.
   This should fill the nbuttons and naxes fields of the joystick structure.
   It returns 0, or -1 if there is an error.
 */
static bool WINDOWS_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    // allocate memory for system specific hardware data
    joystick->hwdata = (struct joystick_hwdata *)SDL_calloc(1, sizeof(struct joystick_hwdata));
    if (!joystick->hwdata) {
        return false;
    }
    joystick->hwdata->guid = device->guid;

    if (device->bXInputDevice) {
        return SDL_XINPUT_JoystickOpen(joystick, device);
    } else {
        return SDL_DINPUT_JoystickOpen(joystick, device);
    }
}

static bool WINDOWS_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    if (joystick->hwdata->bXInputDevice) {
        return SDL_XINPUT_JoystickRumble(joystick, low_frequency_rumble, high_frequency_rumble);
    } else {
        return SDL_DINPUT_JoystickRumble(joystick, low_frequency_rumble, high_frequency_rumble);
    }
}

static bool WINDOWS_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static bool WINDOWS_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool WINDOWS_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool WINDOWS_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

static void WINDOWS_JoystickUpdate(SDL_Joystick *joystick)
{
    if (!joystick->hwdata) {
        return;
    }

    if (joystick->hwdata->bXInputDevice) {
        SDL_XINPUT_JoystickUpdate(joystick);
    } else {
        SDL_DINPUT_JoystickUpdate(joystick);
    }
}

// Function to close a joystick after use
static void WINDOWS_JoystickClose(SDL_Joystick *joystick)
{
    if (joystick->hwdata->bXInputDevice) {
        SDL_XINPUT_JoystickClose(joystick);
    } else {
        SDL_DINPUT_JoystickClose(joystick);
    }

    SDL_free(joystick->hwdata);
}

// Function to perform any system-specific joystick related cleanup
void WINDOWS_JoystickQuit(void)
{
    JoyStick_DeviceData *device = SYS_Joystick;

    while (device) {
        JoyStick_DeviceData *device_next = device->pNext;
        SDL_free(device->joystickname);
        SDL_free(device);
        device = device_next;
    }
    SYS_Joystick = NULL;

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    if (s_bJoystickThread) {
        SDL_StopJoystickThread();
    } else {
        SDL_CleanupDeviceNotification(&s_notification_data);
    }
#endif

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    if (s_bJoystickThread) {
        SDL_StopJoystickThread();
    }
#endif

    SDL_DINPUT_JoystickQuit();
    SDL_XINPUT_JoystickQuit();

    WIN_QuitDeviceNotification();
}

static bool WINDOWS_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return false;
}

SDL_JoystickDriver SDL_WINDOWS_JoystickDriver = {
    WINDOWS_JoystickInit,
    WINDOWS_JoystickGetCount,
    WINDOWS_JoystickDetect,
    WINDOWS_JoystickIsDevicePresent,
    WINDOWS_JoystickGetDeviceName,
    WINDOWS_JoystickGetDevicePath,
    WINDOWS_JoystickGetDeviceSteamVirtualGamepadSlot,
    WINDOWS_JoystickGetDevicePlayerIndex,
    WINDOWS_JoystickSetDevicePlayerIndex,
    WINDOWS_JoystickGetDeviceGUID,
    WINDOWS_JoystickGetDeviceInstanceID,
    WINDOWS_JoystickOpen,
    WINDOWS_JoystickRumble,
    WINDOWS_JoystickRumbleTriggers,
    WINDOWS_JoystickSetLED,
    WINDOWS_JoystickSendEffect,
    WINDOWS_JoystickSetSensorsEnabled,
    WINDOWS_JoystickUpdate,
    WINDOWS_JoystickClose,
    WINDOWS_JoystickQuit,
    WINDOWS_JoystickGetGamepadMapping
};

#else

#ifdef SDL_JOYSTICK_RAWINPUT
// The RAWINPUT driver needs the device notification setup above
#error SDL_JOYSTICK_RAWINPUT requires SDL_JOYSTICK_DINPUT || SDL_JOYSTICK_XINPUT
#endif

#endif // SDL_JOYSTICK_DINPUT || SDL_JOYSTICK_XINPUT
