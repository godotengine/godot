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

#ifdef SDL_JOYSTICK_WGI

#include "../SDL_sysjoystick.h"
#include "../hidapi/SDL_hidapijoystick_c.h"
#include "SDL_rawinputjoystick_c.h"

#include "../../core/windows/SDL_windows.h"
#define COBJMACROS
#include "windows.gaming.input.h"
#include <cfgmgr32.h>
#include <objidlbase.h>
#include <roapi.h>
#include <initguid.h>

#ifdef ____FIReference_1_INT32_INTERFACE_DEFINED__
// MinGW-64 uses __FIReference_1_INT32 instead of Microsoft's __FIReference_1_int
#define __FIReference_1_int           __FIReference_1_INT32
#define __FIReference_1_int_get_Value __FIReference_1_INT32_get_Value
#define __FIReference_1_int_Release   __FIReference_1_INT32_Release
#endif

struct joystick_hwdata
{
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController *controller;
    __x_ABI_CWindows_CGaming_CInput_CIGameController *game_controller;
    __x_ABI_CWindows_CGaming_CInput_CIGameControllerBatteryInfo *battery;
    __x_ABI_CWindows_CGaming_CInput_CIGamepad *gamepad;
    __x_ABI_CWindows_CGaming_CInput_CGamepadVibration vibration;
    UINT64 timestamp;
};

typedef struct WindowsGamingInputControllerState
{
    SDL_JoystickID instance_id;
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController *controller;
    char *name;
    SDL_GUID guid;
    SDL_JoystickType type;
    int steam_virtual_gamepad_slot;
} WindowsGamingInputControllerState;

typedef HRESULT(WINAPI *CoIncrementMTAUsage_t)(CO_MTA_USAGE_COOKIE *pCookie);
typedef HRESULT(WINAPI *RoGetActivationFactory_t)(HSTRING activatableClassId, REFIID iid, void **factory);
typedef HRESULT(WINAPI *WindowsCreateStringReference_t)(PCWSTR sourceString, UINT32 length, HSTRING_HEADER *hstringHeader, HSTRING *string);
typedef HRESULT(WINAPI *WindowsDeleteString_t)(HSTRING string);
typedef PCWSTR(WINAPI *WindowsGetStringRawBuffer_t)(HSTRING string, UINT32 *length);

static struct
{
    CoIncrementMTAUsage_t CoIncrementMTAUsage;
    RoGetActivationFactory_t RoGetActivationFactory;
    WindowsCreateStringReference_t WindowsCreateStringReference;
    WindowsDeleteString_t WindowsDeleteString;
    WindowsGetStringRawBuffer_t WindowsGetStringRawBuffer;
    __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics *controller_statics;
    __x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics *arcade_stick_statics;
    __x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics2 *arcade_stick_statics2;
    __x_ABI_CWindows_CGaming_CInput_CIFlightStickStatics *flight_stick_statics;
    __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics *gamepad_statics;
    __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics2 *gamepad_statics2;
    __x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics *racing_wheel_statics;
    __x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics2 *racing_wheel_statics2;
    EventRegistrationToken controller_added_token;
    EventRegistrationToken controller_removed_token;
    int controller_count;
    WindowsGamingInputControllerState *controllers;
} wgi;

// WinRT headers in official Windows SDK contain only declarations, and we have to define these GUIDs ourselves.
// https://stackoverflow.com/a/55605485/1795050
DEFINE_GUID(IID___FIEventHandler_1_Windows__CGaming__CInput__CRawGameController, 0x00621c22, 0x42e8, 0x529f, 0x92, 0x70, 0x83, 0x6b, 0x32, 0x93, 0x1d, 0x72);
DEFINE_GUID(IID___x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics, 0x5c37b8c8, 0x37b1, 0x4ad8, 0x94, 0x58, 0x20, 0x0f, 0x1a, 0x30, 0x01, 0x8e);
DEFINE_GUID(IID___x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics2, 0x52b5d744, 0xbb86, 0x445a, 0xb5, 0x9c, 0x59, 0x6f, 0x0e, 0x2a, 0x49, 0xdf);
DEFINE_GUID(IID___x_ABI_CWindows_CGaming_CInput_CIFlightStickStatics, 0x5514924a, 0xfecc, 0x435e, 0x83, 0xdc, 0x5c, 0xec, 0x8a, 0x18, 0xa5, 0x20);
DEFINE_GUID(IID___x_ABI_CWindows_CGaming_CInput_CIGameController, 0x1baf6522, 0x5f64, 0x42c5, 0x82, 0x67, 0xb9, 0xfe, 0x22, 0x15, 0xbf, 0xbd);
DEFINE_GUID(IID___x_ABI_CWindows_CGaming_CInput_CIGameControllerBatteryInfo, 0xdcecc681, 0x3963, 0x4da6, 0x95, 0x5d, 0x55, 0x3f, 0x3b, 0x6f, 0x61, 0x61);
DEFINE_GUID(IID___x_ABI_CWindows_CGaming_CInput_CIGamepadStatics, 0x8bbce529, 0xd49c, 0x39e9, 0x95, 0x60, 0xe4, 0x7d, 0xde, 0x96, 0xb7, 0xc8);
DEFINE_GUID(IID___x_ABI_CWindows_CGaming_CInput_CIGamepadStatics2, 0x42676dc5, 0x0856, 0x47c4, 0x92, 0x13, 0xb3, 0x95, 0x50, 0x4c, 0x3a, 0x3c);
DEFINE_GUID(IID___x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics, 0x3ac12cd5, 0x581b, 0x4936, 0x9f, 0x94, 0x69, 0xf1, 0xe6, 0x51, 0x4c, 0x7d);
DEFINE_GUID(IID___x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics2, 0xe666bcaa, 0xedfd, 0x4323, 0xa9, 0xf6, 0x3c, 0x38, 0x40, 0x48, 0xd1, 0xed);
DEFINE_GUID(IID___x_ABI_CWindows_CGaming_CInput_CIRawGameController, 0x7cad6d91, 0xa7e1, 0x4f71, 0x9a, 0x78, 0x33, 0xe9, 0xc5, 0xdf, 0xea, 0x62);
DEFINE_GUID(IID___x_ABI_CWindows_CGaming_CInput_CIRawGameController2, 0x43c0c035, 0xbb73, 0x4756, 0xa7, 0x87, 0x3e, 0xd6, 0xbe, 0xa6, 0x17, 0xbd);
DEFINE_GUID(IID___x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics, 0xeb8d0792, 0xe95a, 0x4b19, 0xaf, 0xc7, 0x0a, 0x59, 0xf8, 0xbf, 0x75, 0x9e);

extern bool SDL_XINPUT_Enabled(void);


static bool SDL_IsXInputDevice(Uint16 vendor, Uint16 product, const char *name)
{
#if defined(SDL_JOYSTICK_XINPUT) || defined(SDL_JOYSTICK_RAWINPUT)
    PRAWINPUTDEVICELIST raw_devices = NULL;
    UINT i, raw_device_count = 0;
    LONG vidpid = MAKELONG(vendor, product);

    // XInput and RawInput backends will pick up XInput-compatible devices
    if (!SDL_XINPUT_Enabled()
#ifdef SDL_JOYSTICK_RAWINPUT
        && !RAWINPUT_IsEnabled()
#endif
    ) {
        return false;
    }

    // Sometimes we'll get a Windows.Gaming.Input callback before the raw input device is even in the list,
    // so try to do some checks up front to catch these cases.
    if (SDL_IsJoystickXboxOne(vendor, product) ||
        (name && SDL_strncmp(name, "Xbox ", 5) == 0)) {
        return true;
    }

    // Go through RAWINPUT (WinXP and later) to find HID devices.
    if ((GetRawInputDeviceList(NULL, &raw_device_count, sizeof(RAWINPUTDEVICELIST)) == -1) || (!raw_device_count)) {
        return false; // oh well.
    }

    raw_devices = (PRAWINPUTDEVICELIST)SDL_malloc(sizeof(RAWINPUTDEVICELIST) * raw_device_count);
    if (!raw_devices) {
        return false;
    }

    raw_device_count = GetRawInputDeviceList(raw_devices, &raw_device_count, sizeof(RAWINPUTDEVICELIST));
    if (raw_device_count == (UINT)-1) {
        SDL_free(raw_devices);
        raw_devices = NULL;
        return false; // oh well.
    }

    for (i = 0; i < raw_device_count; i++) {
        RID_DEVICE_INFO rdi;
        char devName[MAX_PATH] = { 0 };
        UINT rdiSize = sizeof(rdi);
        UINT nameSize = SDL_arraysize(devName);
        DEVINST devNode;
        char devVidPidString[32];
        int j;

        rdi.cbSize = sizeof(rdi);

        if ((raw_devices[i].dwType != RIM_TYPEHID) ||
            (GetRawInputDeviceInfoA(raw_devices[i].hDevice, RIDI_DEVICEINFO, &rdi, &rdiSize) == ((UINT)-1)) ||
            (GetRawInputDeviceInfoA(raw_devices[i].hDevice, RIDI_DEVICENAME, devName, &nameSize) == ((UINT)-1)) ||
            (SDL_strstr(devName, "IG_") == NULL)) {
            // Skip non-XInput devices
            continue;
        }

        // First check for a simple VID/PID match. This will work for Xbox 360 controllers.
        if (MAKELONG(rdi.hid.dwVendorId, rdi.hid.dwProductId) == vidpid) {
            SDL_free(raw_devices);
            return true;
        }

        /* For Xbox One controllers, Microsoft doesn't propagate the VID/PID down to the HID stack.
         * We'll have to walk the device tree upwards searching for a match for our VID/PID. */

        // Make sure the device interface string is something we know how to parse
        // Example: \\?\HID#VID_045E&PID_02FF&IG_00#9&2c203035&2&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}
        if ((SDL_strstr(devName, "\\\\?\\") != devName) || (SDL_strstr(devName, "#{") == NULL)) {
            continue;
        }

        // Unescape the backslashes in the string and terminate before the GUID portion
        for (j = 0; devName[j] != '\0'; j++) {
            if (devName[j] == '#') {
                if (devName[j + 1] == '{') {
                    devName[j] = '\0';
                    break;
                } else {
                    devName[j] = '\\';
                }
            }
        }

        /* We'll be left with a string like this: \\?\HID\VID_045E&PID_02FF&IG_00\9&2c203035&2&0000
         * Simply skip the \\?\ prefix and we'll have a properly formed device instance ID */
        if (CM_Locate_DevNodeA(&devNode, &devName[4], CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS) {
            continue;
        }

        (void)SDL_snprintf(devVidPidString, sizeof(devVidPidString), "VID_%04X&PID_%04X", vendor, product);

        while (CM_Get_Parent(&devNode, devNode, 0) == CR_SUCCESS) {
            char deviceId[MAX_DEVICE_ID_LEN];

            if ((CM_Get_Device_IDA(devNode, deviceId, SDL_arraysize(deviceId), 0) == CR_SUCCESS) &&
                (SDL_strstr(deviceId, devVidPidString) != NULL)) {
                // The VID/PID matched a parent device
                SDL_free(raw_devices);
                return true;
            }
        }
    }

    SDL_free(raw_devices);
#endif // SDL_JOYSTICK_XINPUT || SDL_JOYSTICK_RAWINPUT

    return false;
}

static void WGI_LoadRawGameControllerStatics(void)
{
    HRESULT hr;
    HSTRING_HEADER class_name_header;
    HSTRING class_name;

    hr = wgi.WindowsCreateStringReference(RuntimeClass_Windows_Gaming_Input_RawGameController, (UINT32)SDL_wcslen(RuntimeClass_Windows_Gaming_Input_RawGameController), &class_name_header, &class_name);
    if (SUCCEEDED(hr)) {
        hr = wgi.RoGetActivationFactory(class_name, &IID___x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics, (void **)&wgi.controller_statics);
        if (!SUCCEEDED(hr)) {
            WIN_SetErrorFromHRESULT("Couldn't find Windows.Gaming.Input.IRawGameControllerStatics", hr);
        }
    }
}

static void WGI_LoadOtherControllerStatics(void)
{
    HRESULT hr;
    HSTRING_HEADER class_name_header;
    HSTRING class_name;

    if (!wgi.arcade_stick_statics) {
        hr = wgi.WindowsCreateStringReference(RuntimeClass_Windows_Gaming_Input_ArcadeStick, (UINT32)SDL_wcslen(RuntimeClass_Windows_Gaming_Input_ArcadeStick), &class_name_header, &class_name);
        if (SUCCEEDED(hr)) {
            hr = wgi.RoGetActivationFactory(class_name, &IID___x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics, (void **)&wgi.arcade_stick_statics);
            if (SUCCEEDED(hr)) {
                __x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics_QueryInterface(wgi.arcade_stick_statics, &IID___x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics2, (void **)&wgi.arcade_stick_statics2);
            } else {
                WIN_SetErrorFromHRESULT("Couldn't find Windows.Gaming.Input.IArcadeStickStatics", hr);
            }
        }
    }

    if (!wgi.flight_stick_statics) {
        hr = wgi.WindowsCreateStringReference(RuntimeClass_Windows_Gaming_Input_FlightStick, (UINT32)SDL_wcslen(RuntimeClass_Windows_Gaming_Input_FlightStick), &class_name_header, &class_name);
        if (SUCCEEDED(hr)) {
            hr = wgi.RoGetActivationFactory(class_name, &IID___x_ABI_CWindows_CGaming_CInput_CIFlightStickStatics, (void **)&wgi.flight_stick_statics);
            if (!SUCCEEDED(hr)) {
                WIN_SetErrorFromHRESULT("Couldn't find Windows.Gaming.Input.IFlightStickStatics", hr);
            }
        }
    }

    if (!wgi.gamepad_statics) {
        hr = wgi.WindowsCreateStringReference(RuntimeClass_Windows_Gaming_Input_Gamepad, (UINT32)SDL_wcslen(RuntimeClass_Windows_Gaming_Input_Gamepad), &class_name_header, &class_name);
        if (SUCCEEDED(hr)) {
            hr = wgi.RoGetActivationFactory(class_name, &IID___x_ABI_CWindows_CGaming_CInput_CIGamepadStatics, (void **)&wgi.gamepad_statics);
            if (SUCCEEDED(hr)) {
                __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics_QueryInterface(wgi.gamepad_statics, &IID___x_ABI_CWindows_CGaming_CInput_CIGamepadStatics2, (void **)&wgi.gamepad_statics2);
            } else {
                WIN_SetErrorFromHRESULT("Couldn't find Windows.Gaming.Input.IGamepadStatics", hr);
            }
        }
    }

    if (!wgi.racing_wheel_statics) {
        hr = wgi.WindowsCreateStringReference(RuntimeClass_Windows_Gaming_Input_RacingWheel, (UINT32)SDL_wcslen(RuntimeClass_Windows_Gaming_Input_RacingWheel), &class_name_header, &class_name);
        if (SUCCEEDED(hr)) {
            hr = wgi.RoGetActivationFactory(class_name, &IID___x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics, (void **)&wgi.racing_wheel_statics);
            if (SUCCEEDED(hr)) {
                __x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics_QueryInterface(wgi.racing_wheel_statics, &IID___x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics2, (void **)&wgi.racing_wheel_statics2);
            } else {
                WIN_SetErrorFromHRESULT("Couldn't find Windows.Gaming.Input.IRacingWheelStatics", hr);
            }
        }
    }
}

static SDL_JoystickType GetGameControllerType(__x_ABI_CWindows_CGaming_CInput_CIGameController *game_controller)
{
    __x_ABI_CWindows_CGaming_CInput_CIArcadeStick *arcade_stick = NULL;
    __x_ABI_CWindows_CGaming_CInput_CIFlightStick *flight_stick = NULL;
    __x_ABI_CWindows_CGaming_CInput_CIGamepad *gamepad = NULL;
    __x_ABI_CWindows_CGaming_CInput_CIRacingWheel *racing_wheel = NULL;

    /* Wait to initialize these interfaces until we need them.
     * Initializing the gamepad interface will switch Bluetooth PS4 controllers into enhanced mode, breaking DirectInput
     */
    WGI_LoadOtherControllerStatics();

    if (wgi.gamepad_statics2 && SUCCEEDED(__x_ABI_CWindows_CGaming_CInput_CIGamepadStatics2_FromGameController(wgi.gamepad_statics2, game_controller, &gamepad)) && gamepad) {
        __x_ABI_CWindows_CGaming_CInput_CIGamepad_Release(gamepad);
        return SDL_JOYSTICK_TYPE_GAMEPAD;
    }

    if (wgi.arcade_stick_statics2 && SUCCEEDED(__x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics2_FromGameController(wgi.arcade_stick_statics2, game_controller, &arcade_stick)) && arcade_stick) {
        __x_ABI_CWindows_CGaming_CInput_CIArcadeStick_Release(arcade_stick);
        return SDL_JOYSTICK_TYPE_ARCADE_STICK;
    }

    if (wgi.flight_stick_statics && SUCCEEDED(__x_ABI_CWindows_CGaming_CInput_CIFlightStickStatics_FromGameController(wgi.flight_stick_statics, game_controller, &flight_stick)) && flight_stick) {
        __x_ABI_CWindows_CGaming_CInput_CIFlightStick_Release(flight_stick);
        return SDL_JOYSTICK_TYPE_FLIGHT_STICK;
    }

    if (wgi.racing_wheel_statics2 && SUCCEEDED(__x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics2_FromGameController(wgi.racing_wheel_statics2, game_controller, &racing_wheel)) && racing_wheel) {
        __x_ABI_CWindows_CGaming_CInput_CIRacingWheel_Release(racing_wheel);
        return SDL_JOYSTICK_TYPE_WHEEL;
    }

    return SDL_JOYSTICK_TYPE_UNKNOWN;
}

typedef struct RawGameControllerDelegate
{
    __FIEventHandler_1_Windows__CGaming__CInput__CRawGameController iface;
    SDL_AtomicInt refcount;
} RawGameControllerDelegate;

static HRESULT STDMETHODCALLTYPE IEventHandler_CRawGameControllerVtbl_QueryInterface(__FIEventHandler_1_Windows__CGaming__CInput__CRawGameController *This, REFIID riid, void **ppvObject)
{
    if (!ppvObject) {
        return E_INVALIDARG;
    }

    *ppvObject = NULL;
    if (WIN_IsEqualIID(riid, &IID_IUnknown) || WIN_IsEqualIID(riid, &IID_IAgileObject) || WIN_IsEqualIID(riid, &IID___FIEventHandler_1_Windows__CGaming__CInput__CRawGameController)) {
        *ppvObject = This;
        __FIEventHandler_1_Windows__CGaming__CInput__CRawGameController_AddRef(This);
        return S_OK;
    } else if (WIN_IsEqualIID(riid, &IID_IMarshal)) {
        // This seems complicated. Let's hope it doesn't happen.
        return E_OUTOFMEMORY;
    } else {
        return E_NOINTERFACE;
    }
}

static ULONG STDMETHODCALLTYPE IEventHandler_CRawGameControllerVtbl_AddRef(__FIEventHandler_1_Windows__CGaming__CInput__CRawGameController *This)
{
    RawGameControllerDelegate *self = (RawGameControllerDelegate *)This;
    return SDL_AddAtomicInt(&self->refcount, 1) + 1UL;
}

static ULONG STDMETHODCALLTYPE IEventHandler_CRawGameControllerVtbl_Release(__FIEventHandler_1_Windows__CGaming__CInput__CRawGameController *This)
{
    RawGameControllerDelegate *self = (RawGameControllerDelegate *)This;
    int rc = SDL_AddAtomicInt(&self->refcount, -1) - 1;
    // Should never free the static delegate objects
    SDL_assert(rc > 0);
    return rc;
}

static int GetSteamVirtualGamepadSlot(__x_ABI_CWindows_CGaming_CInput_CIRawGameController *controller, Uint16 vendor_id, Uint16 product_id)
{
    int slot = -1;

    if (vendor_id == USB_VENDOR_VALVE &&
        product_id == USB_PRODUCT_STEAM_VIRTUAL_GAMEPAD) {
        __x_ABI_CWindows_CGaming_CInput_CIRawGameController2 *controller2 = NULL;
        HRESULT hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(controller, &IID___x_ABI_CWindows_CGaming_CInput_CIRawGameController2, (void **)&controller2);
        if (SUCCEEDED(hr)) {
            HSTRING hString;
            hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController2_get_NonRoamableId(controller2, &hString);
            if (SUCCEEDED(hr)) {
                PCWSTR string = wgi.WindowsGetStringRawBuffer(hString, NULL);
                if (string) {
                    char *id = WIN_StringToUTF8W(string);
                    if (id) {
                        (void)SDL_sscanf(id, "{wgi/nrid/:steam-%*X&%*X&%*X#%d#%*u}", &slot);
                        SDL_free(id);
                    }
                }
                wgi.WindowsDeleteString(hString);
            }
            __x_ABI_CWindows_CGaming_CInput_CIRawGameController2_Release(controller2);
        }
    }
    return slot;
}

static HRESULT STDMETHODCALLTYPE IEventHandler_CRawGameControllerVtbl_InvokeAdded(__FIEventHandler_1_Windows__CGaming__CInput__CRawGameController *This, IInspectable *sender, __x_ABI_CWindows_CGaming_CInput_CIRawGameController *e)
{
    HRESULT hr;
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController *controller = NULL;

    SDL_LockJoysticks();

    // We can get delayed calls to InvokeAdded() after WGI_JoystickQuit()
    if (SDL_JoysticksQuitting() || !SDL_JoysticksInitialized()) {
        SDL_UnlockJoysticks();
        return S_OK;
    }

    hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(e, &IID___x_ABI_CWindows_CGaming_CInput_CIRawGameController, (void **)&controller);
    if (SUCCEEDED(hr)) {
        char *name = NULL;
        Uint16 bus = SDL_HARDWARE_BUS_USB;
        Uint16 vendor = 0;
        Uint16 product = 0;
        Uint16 version = 0;
        SDL_JoystickType type = SDL_JOYSTICK_TYPE_UNKNOWN;
        __x_ABI_CWindows_CGaming_CInput_CIRawGameController2 *controller2 = NULL;
        __x_ABI_CWindows_CGaming_CInput_CIGameController *game_controller = NULL;
        bool ignore_joystick = false;

        __x_ABI_CWindows_CGaming_CInput_CIRawGameController_get_HardwareVendorId(controller, &vendor);
        __x_ABI_CWindows_CGaming_CInput_CIRawGameController_get_HardwareProductId(controller, &product);

        hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(controller, &IID___x_ABI_CWindows_CGaming_CInput_CIGameController, (void **)&game_controller);
        if (SUCCEEDED(hr)) {
            boolean wireless = 0;
            hr = __x_ABI_CWindows_CGaming_CInput_CIGameController_get_IsWireless(game_controller, &wireless);
            if (SUCCEEDED(hr) && wireless) {
                bus = SDL_HARDWARE_BUS_BLUETOOTH;

                // Fixup for Wireless Xbox 360 Controller
                if (product == 0) {
                    vendor = USB_VENDOR_MICROSOFT;
                    product = USB_PRODUCT_XBOX360_XUSB_CONTROLLER;
                }
            }

            __x_ABI_CWindows_CGaming_CInput_CIGameController_Release(game_controller);
        }

        hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(controller, &IID___x_ABI_CWindows_CGaming_CInput_CIRawGameController2, (void **)&controller2);
        if (SUCCEEDED(hr)) {
            HSTRING hString;
            hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController2_get_DisplayName(controller2, &hString);
            if (SUCCEEDED(hr)) {
                PCWSTR string = wgi.WindowsGetStringRawBuffer(hString, NULL);
                if (string) {
                    name = WIN_StringToUTF8W(string);
                }
                wgi.WindowsDeleteString(hString);
            }
            __x_ABI_CWindows_CGaming_CInput_CIRawGameController2_Release(controller2);
        }
        if (!name) {
            name = SDL_strdup("");
        }

        if (!ignore_joystick && SDL_ShouldIgnoreJoystick(vendor, product, version, name)) {
            ignore_joystick = true;
        }

        if (!ignore_joystick && SDL_JoystickHandledByAnotherDriver(&SDL_WGI_JoystickDriver, vendor, product, version, name)) {
            ignore_joystick = true;
        }

        if (!ignore_joystick && SDL_IsXInputDevice(vendor, product, name)) {
            // This hasn't been detected by the RAWINPUT driver yet, but it will be picked up later.
            ignore_joystick = true;
        }

        if (!ignore_joystick) {
            // New device, add it
            WindowsGamingInputControllerState *controllers = SDL_realloc(wgi.controllers, sizeof(wgi.controllers[0]) * (wgi.controller_count + 1));
            if (controllers) {
                WindowsGamingInputControllerState *state = &controllers[wgi.controller_count];
                SDL_JoystickID joystickID = SDL_GetNextObjectID();

                if (game_controller) {
                    type = GetGameControllerType(game_controller);
                }

                SDL_zerop(state);
                state->instance_id = joystickID;
                state->controller = controller;
                state->name = name;
                state->guid = SDL_CreateJoystickGUID(bus, vendor, product, version, NULL, name, 'w', (Uint8)type);
                state->type = type;
                state->steam_virtual_gamepad_slot = GetSteamVirtualGamepadSlot(controller, vendor, product);

                __x_ABI_CWindows_CGaming_CInput_CIRawGameController_AddRef(controller);

                ++wgi.controller_count;
                wgi.controllers = controllers;

                SDL_PrivateJoystickAdded(joystickID);
            } else {
                SDL_free(name);
            }
        } else {
            SDL_free(name);
        }

        __x_ABI_CWindows_CGaming_CInput_CIRawGameController_Release(controller);
    }

    SDL_UnlockJoysticks();

    return S_OK;
}

static HRESULT STDMETHODCALLTYPE IEventHandler_CRawGameControllerVtbl_InvokeRemoved(__FIEventHandler_1_Windows__CGaming__CInput__CRawGameController *This, IInspectable *sender, __x_ABI_CWindows_CGaming_CInput_CIRawGameController *e)
{
    HRESULT hr;
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController *controller = NULL;

    SDL_LockJoysticks();

    // Can we get delayed calls to InvokeRemoved() after WGI_JoystickQuit()?
    if (!SDL_JoysticksInitialized()) {
        SDL_UnlockJoysticks();
        return S_OK;
    }

    hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(e, &IID___x_ABI_CWindows_CGaming_CInput_CIRawGameController, (void **)&controller);
    if (SUCCEEDED(hr)) {
        int i;

        for (i = 0; i < wgi.controller_count; i++) {
            if (wgi.controllers[i].controller == controller) {
                WindowsGamingInputControllerState *state = &wgi.controllers[i];
                SDL_JoystickID joystickID = state->instance_id;

                __x_ABI_CWindows_CGaming_CInput_CIRawGameController_Release(state->controller);

                SDL_free(state->name);

                --wgi.controller_count;
                if (i < wgi.controller_count) {
                    SDL_memmove(&wgi.controllers[i], &wgi.controllers[i + 1], (wgi.controller_count - i) * sizeof(wgi.controllers[i]));
                }

                SDL_PrivateJoystickRemoved(joystickID);
                break;
            }
        }

        __x_ABI_CWindows_CGaming_CInput_CIRawGameController_Release(controller);
    }

    SDL_UnlockJoysticks();

    return S_OK;
}

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4028) // formal parameter 3 different from declaration, when using older buggy WGI headers
#pragma warning(disable : 4113) // formal parameter 3 different from declaration (a more specific warning added in VS 2022), when using older buggy WGI headers
#endif

static __FIEventHandler_1_Windows__CGaming__CInput__CRawGameControllerVtbl controller_added_vtbl = {
    IEventHandler_CRawGameControllerVtbl_QueryInterface,
    IEventHandler_CRawGameControllerVtbl_AddRef,
    IEventHandler_CRawGameControllerVtbl_Release,
    IEventHandler_CRawGameControllerVtbl_InvokeAdded
};
static RawGameControllerDelegate controller_added = {
    { &controller_added_vtbl },
    { 1 }
};

static __FIEventHandler_1_Windows__CGaming__CInput__CRawGameControllerVtbl controller_removed_vtbl = {
    IEventHandler_CRawGameControllerVtbl_QueryInterface,
    IEventHandler_CRawGameControllerVtbl_AddRef,
    IEventHandler_CRawGameControllerVtbl_Release,
    IEventHandler_CRawGameControllerVtbl_InvokeRemoved
};
static RawGameControllerDelegate controller_removed = {
    { &controller_removed_vtbl },
    { 1 }
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

static bool WGI_JoystickInit(void)
{
    HRESULT hr;

    if (!SDL_GetHintBoolean(SDL_HINT_JOYSTICK_WGI, true)) {
        return true;
    }

    if (FAILED(WIN_RoInitialize())) {
        return SDL_SetError("RoInitialize() failed");
    }

#define RESOLVE(x) wgi.x = (x##_t)WIN_LoadComBaseFunction(#x); if (!wgi.x) return WIN_SetError("GetProcAddress failed for " #x);
    RESOLVE(CoIncrementMTAUsage);
    RESOLVE(RoGetActivationFactory);
    RESOLVE(WindowsCreateStringReference);
    RESOLVE(WindowsDeleteString);
    RESOLVE(WindowsGetStringRawBuffer);
#undef RESOLVE

    {
        /* There seems to be a bug in Windows where a dependency of WGI can be unloaded from memory prior to WGI itself.
         * This results in Windows_Gaming_Input!GameController::~GameController() invoking an unloaded DLL and crashing.
         * As a workaround, we will keep a reference to the MTA to prevent COM from unloading DLLs later.
         * See https://github.com/libsdl-org/SDL/issues/5552 for more details.
         */
        static CO_MTA_USAGE_COOKIE cookie = NULL;
        if (!cookie) {
            hr = wgi.CoIncrementMTAUsage(&cookie);
            if (FAILED(hr)) {
                return WIN_SetErrorFromHRESULT("CoIncrementMTAUsage() failed", hr);
            }
        }
    }

    WGI_LoadRawGameControllerStatics();

    if (wgi.controller_statics) {
        __FIVectorView_1_Windows__CGaming__CInput__CRawGameController *controllers;

        hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics_add_RawGameControllerAdded(wgi.controller_statics, &controller_added.iface, &wgi.controller_added_token);
        if (!SUCCEEDED(hr)) {
            WIN_SetErrorFromHRESULT("Windows.Gaming.Input.IRawGameControllerStatics.add_RawGameControllerAdded failed", hr);
        }

        hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics_add_RawGameControllerRemoved(wgi.controller_statics, &controller_removed.iface, &wgi.controller_removed_token);
        if (!SUCCEEDED(hr)) {
            WIN_SetErrorFromHRESULT("Windows.Gaming.Input.IRawGameControllerStatics.add_RawGameControllerRemoved failed", hr);
        }

        hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics_get_RawGameControllers(wgi.controller_statics, &controllers);
        if (SUCCEEDED(hr)) {
            unsigned i, count = 0;

            hr = __FIVectorView_1_Windows__CGaming__CInput__CRawGameController_get_Size(controllers, &count);
            if (SUCCEEDED(hr)) {
                for (i = 0; i < count; ++i) {
                    __x_ABI_CWindows_CGaming_CInput_CIRawGameController *controller = NULL;

                    hr = __FIVectorView_1_Windows__CGaming__CInput__CRawGameController_GetAt(controllers, i, &controller);
                    if (SUCCEEDED(hr) && controller) {
                        IEventHandler_CRawGameControllerVtbl_InvokeAdded(&controller_added.iface, NULL, controller);
                        __x_ABI_CWindows_CGaming_CInput_CIRawGameController_Release(controller);
                    }
                }
            }

            __FIVectorView_1_Windows__CGaming__CInput__CRawGameController_Release(controllers);
        }
    }

    return true;
}

static int WGI_JoystickGetCount(void)
{
    return wgi.controller_count;
}

static void WGI_JoystickDetect(void)
{
}

static bool WGI_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    // We don't override any other drivers
    return false;
}

static const char *WGI_JoystickGetDeviceName(int device_index)
{
    return wgi.controllers[device_index].name;
}

static const char *WGI_JoystickGetDevicePath(int device_index)
{
    return NULL;
}

static int WGI_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return wgi.controllers[device_index].steam_virtual_gamepad_slot;
}

static int WGI_JoystickGetDevicePlayerIndex(int device_index)
{
    return false;
}

static void WGI_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

static SDL_GUID WGI_JoystickGetDeviceGUID(int device_index)
{
    return wgi.controllers[device_index].guid;
}

static SDL_JoystickID WGI_JoystickGetDeviceInstanceID(int device_index)
{
    return wgi.controllers[device_index].instance_id;
}

static bool WGI_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    WindowsGamingInputControllerState *state = &wgi.controllers[device_index];
    struct joystick_hwdata *hwdata;
    boolean wireless = false;

    hwdata = (struct joystick_hwdata *)SDL_calloc(1, sizeof(*hwdata));
    if (!hwdata) {
        return false;
    }
    joystick->hwdata = hwdata;

    hwdata->controller = state->controller;
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController_AddRef(hwdata->controller);
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(hwdata->controller, &IID___x_ABI_CWindows_CGaming_CInput_CIGameController, (void **)&hwdata->game_controller);
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(hwdata->controller, &IID___x_ABI_CWindows_CGaming_CInput_CIGameControllerBatteryInfo, (void **)&hwdata->battery);

    if (wgi.gamepad_statics2) {
        __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics2_FromGameController(wgi.gamepad_statics2, hwdata->game_controller, &hwdata->gamepad);
    }

    if (hwdata->game_controller) {
        __x_ABI_CWindows_CGaming_CInput_CIGameController_get_IsWireless(hwdata->game_controller, &wireless);
    }

    // Initialize the joystick capabilities
    if (wireless) {
        joystick->connection_state = SDL_JOYSTICK_CONNECTION_WIRELESS;
    } else {
        joystick->connection_state = SDL_JOYSTICK_CONNECTION_WIRED;
    }
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController_get_ButtonCount(hwdata->controller, &joystick->nbuttons);
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController_get_AxisCount(hwdata->controller, &joystick->naxes);
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController_get_SwitchCount(hwdata->controller, &joystick->nhats);

    if (hwdata->gamepad) {
        // FIXME: Can WGI even tell us if trigger rumble is supported?
        SDL_SetBooleanProperty(SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_RUMBLE_BOOLEAN, true);
        SDL_SetBooleanProperty(SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_TRIGGER_RUMBLE_BOOLEAN, true);
    }
    return true;
}

static bool WGI_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    struct joystick_hwdata *hwdata = joystick->hwdata;

    if (hwdata->gamepad) {
        HRESULT hr;

        // Note: reusing partially filled vibration data struct
        hwdata->vibration.LeftMotor = (DOUBLE)low_frequency_rumble / SDL_MAX_UINT16;
        hwdata->vibration.RightMotor = (DOUBLE)high_frequency_rumble / SDL_MAX_UINT16;
        hr = __x_ABI_CWindows_CGaming_CInput_CIGamepad_put_Vibration(hwdata->gamepad, hwdata->vibration);
        if (SUCCEEDED(hr)) {
            return true;
        } else {
            return WIN_SetErrorFromHRESULT("Windows.Gaming.Input.IGamepad.put_Vibration failed", hr);
        }
    } else {
        return SDL_Unsupported();
    }
}

static bool WGI_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    struct joystick_hwdata *hwdata = joystick->hwdata;

    if (hwdata->gamepad) {
        HRESULT hr;

        // Note: reusing partially filled vibration data struct
        hwdata->vibration.LeftTrigger = (DOUBLE)left_rumble / SDL_MAX_UINT16;
        hwdata->vibration.RightTrigger = (DOUBLE)right_rumble / SDL_MAX_UINT16;
        hr = __x_ABI_CWindows_CGaming_CInput_CIGamepad_put_Vibration(hwdata->gamepad, hwdata->vibration);
        if (SUCCEEDED(hr)) {
            return true;
        } else {
            return WIN_SetErrorFromHRESULT("Windows.Gaming.Input.IGamepad.put_Vibration failed", hr);
        }
    } else {
        return SDL_Unsupported();
    }
}

static bool WGI_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool WGI_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool WGI_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

static Uint8 ConvertHatValue(__x_ABI_CWindows_CGaming_CInput_CGameControllerSwitchPosition value)
{
    switch (value) {
    case GameControllerSwitchPosition_Up:
        return SDL_HAT_UP;
    case GameControllerSwitchPosition_UpRight:
        return SDL_HAT_RIGHTUP;
    case GameControllerSwitchPosition_Right:
        return SDL_HAT_RIGHT;
    case GameControllerSwitchPosition_DownRight:
        return SDL_HAT_RIGHTDOWN;
    case GameControllerSwitchPosition_Down:
        return SDL_HAT_DOWN;
    case GameControllerSwitchPosition_DownLeft:
        return SDL_HAT_LEFTDOWN;
    case GameControllerSwitchPosition_Left:
        return SDL_HAT_LEFT;
    case GameControllerSwitchPosition_UpLeft:
        return SDL_HAT_LEFTUP;
    default:
        return SDL_HAT_CENTERED;
    }
}

static void WGI_JoystickUpdate(SDL_Joystick *joystick)
{
    struct joystick_hwdata *hwdata = joystick->hwdata;
    HRESULT hr;
    UINT32 nbuttons = SDL_min(joystick->nbuttons, SDL_MAX_UINT8);
    boolean *buttons = NULL;
    UINT32 nhats = SDL_min(joystick->nhats, SDL_MAX_UINT8);
    __x_ABI_CWindows_CGaming_CInput_CGameControllerSwitchPosition *hats = NULL;
    UINT32 naxes = SDL_min(joystick->naxes, SDL_MAX_UINT8);
    DOUBLE *axes = NULL;
    UINT64 timestamp;

    if (nbuttons > 0) {
        buttons = SDL_stack_alloc(boolean, nbuttons);
    }
    if (nhats > 0) {
        hats = SDL_stack_alloc(__x_ABI_CWindows_CGaming_CInput_CGameControllerSwitchPosition, nhats);
    }
    if (naxes > 0) {
        axes = SDL_stack_alloc(DOUBLE, naxes);
    }

    hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController_GetCurrentReading(hwdata->controller, nbuttons, buttons, nhats, hats, naxes, axes, &timestamp);
    if (SUCCEEDED(hr) && (!timestamp || timestamp != hwdata->timestamp)) {
        UINT32 i;
        bool all_zero = false;

        hwdata->timestamp = timestamp;

        // The axes are all zero when the application loses focus
        if (naxes > 0) {
            all_zero = true;
            for (i = 0; i < naxes; ++i) {
                if (axes[i] != 0.0f) {
                    all_zero = false;
                    break;
                }
            }
        }
        if (all_zero) {
            SDL_PrivateJoystickForceRecentering(joystick);
        } else {
            // FIXME: What units are the timestamp we get from GetCurrentReading()?
            timestamp = SDL_GetTicksNS();
            for (i = 0; i < nbuttons; ++i) {
                SDL_SendJoystickButton(timestamp, joystick, (Uint8)i, buttons[i]);
            }
            for (i = 0; i < nhats; ++i) {
                SDL_SendJoystickHat(timestamp, joystick, (Uint8)i, ConvertHatValue(hats[i]));
            }
            for (i = 0; i < naxes; ++i) {
                SDL_SendJoystickAxis(timestamp, joystick, (Uint8)i, (Sint16)((int)(axes[i] * 65535) - 32768));
            }
        }
    }

    SDL_stack_free(buttons);
    SDL_stack_free(hats);
    SDL_stack_free(axes);

    if (hwdata->battery) {
        __x_ABI_CWindows_CDevices_CPower_CIBatteryReport *report = NULL;

        hr = __x_ABI_CWindows_CGaming_CInput_CIGameControllerBatteryInfo_TryGetBatteryReport(hwdata->battery, &report);
        if (SUCCEEDED(hr) && report) {
            SDL_PowerState state = SDL_POWERSTATE_UNKNOWN;
            int percent = 0;
            __x_ABI_CWindows_CSystem_CPower_CBatteryStatus status;
            int full_capacity = 0, curr_capacity = 0;
            __FIReference_1_int *full_capacityP, *curr_capacityP;

            hr = __x_ABI_CWindows_CDevices_CPower_CIBatteryReport_get_Status(report, &status);
            if (SUCCEEDED(hr)) {
                switch (status) {
                case BatteryStatus_NotPresent:
                    state = SDL_POWERSTATE_NO_BATTERY;
                    break;
                case BatteryStatus_Discharging:
                    state = SDL_POWERSTATE_ON_BATTERY;
                    break;
                case BatteryStatus_Idle:
                    state = SDL_POWERSTATE_CHARGED;
                    break;
                case BatteryStatus_Charging:
                    state = SDL_POWERSTATE_CHARGING;
                    break;
                default:
                    state = SDL_POWERSTATE_UNKNOWN;
                    break;
                }
            }

            hr = __x_ABI_CWindows_CDevices_CPower_CIBatteryReport_get_FullChargeCapacityInMilliwattHours(report, &full_capacityP);
            if (SUCCEEDED(hr)) {
                __FIReference_1_int_get_Value(full_capacityP, &full_capacity);
                __FIReference_1_int_Release(full_capacityP);
            }

            hr = __x_ABI_CWindows_CDevices_CPower_CIBatteryReport_get_RemainingCapacityInMilliwattHours(report, &curr_capacityP);
            if (SUCCEEDED(hr)) {
                __FIReference_1_int_get_Value(curr_capacityP, &curr_capacity);
                __FIReference_1_int_Release(curr_capacityP);
            }

            if (full_capacity > 0) {
                percent = (int)SDL_roundf(((float)curr_capacity / full_capacity) * 100.0f);
            }

            SDL_SendJoystickPowerInfo(joystick, state, percent);

            __x_ABI_CWindows_CDevices_CPower_CIBatteryReport_Release(report);
        }
    }
}

static void WGI_JoystickClose(SDL_Joystick *joystick)
{
    struct joystick_hwdata *hwdata = joystick->hwdata;

    if (hwdata) {
        if (hwdata->controller) {
            __x_ABI_CWindows_CGaming_CInput_CIRawGameController_Release(hwdata->controller);
        }
        if (hwdata->game_controller) {
            __x_ABI_CWindows_CGaming_CInput_CIGameController_Release(hwdata->game_controller);
        }
        if (hwdata->battery) {
            __x_ABI_CWindows_CGaming_CInput_CIGameControllerBatteryInfo_Release(hwdata->battery);
        }
        if (hwdata->gamepad) {
            __x_ABI_CWindows_CGaming_CInput_CIGamepad_Release(hwdata->gamepad);
        }
        SDL_free(hwdata);
    }
    joystick->hwdata = NULL;
}

static void WGI_JoystickQuit(void)
{
    if (wgi.controller_statics) {
        while (wgi.controller_count > 0) {
            IEventHandler_CRawGameControllerVtbl_InvokeRemoved(&controller_removed.iface, NULL, wgi.controllers[wgi.controller_count - 1].controller);
        }
        if (wgi.controllers) {
            SDL_free(wgi.controllers);
        }

        if (wgi.arcade_stick_statics) {
            __x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics_Release(wgi.arcade_stick_statics);
        }
        if (wgi.arcade_stick_statics2) {
            __x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics2_Release(wgi.arcade_stick_statics2);
        }
        if (wgi.flight_stick_statics) {
            __x_ABI_CWindows_CGaming_CInput_CIFlightStickStatics_Release(wgi.flight_stick_statics);
        }
        if (wgi.gamepad_statics) {
            __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics_Release(wgi.gamepad_statics);
        }
        if (wgi.gamepad_statics2) {
            __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics2_Release(wgi.gamepad_statics2);
        }
        if (wgi.racing_wheel_statics) {
            __x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics_Release(wgi.racing_wheel_statics);
        }
        if (wgi.racing_wheel_statics2) {
            __x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics2_Release(wgi.racing_wheel_statics2);
        }

        __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics_remove_RawGameControllerAdded(wgi.controller_statics, wgi.controller_added_token);
        __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics_remove_RawGameControllerRemoved(wgi.controller_statics, wgi.controller_removed_token);
        __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics_Release(wgi.controller_statics);
    }

    WIN_RoUninitialize();

    SDL_zero(wgi);
}

static bool WGI_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return false;
}

SDL_JoystickDriver SDL_WGI_JoystickDriver = {
    WGI_JoystickInit,
    WGI_JoystickGetCount,
    WGI_JoystickDetect,
    WGI_JoystickIsDevicePresent,
    WGI_JoystickGetDeviceName,
    WGI_JoystickGetDevicePath,
    WGI_JoystickGetDeviceSteamVirtualGamepadSlot,
    WGI_JoystickGetDevicePlayerIndex,
    WGI_JoystickSetDevicePlayerIndex,
    WGI_JoystickGetDeviceGUID,
    WGI_JoystickGetDeviceInstanceID,
    WGI_JoystickOpen,
    WGI_JoystickRumble,
    WGI_JoystickRumbleTriggers,
    WGI_JoystickSetLED,
    WGI_JoystickSendEffect,
    WGI_JoystickSetSensorsEnabled,
    WGI_JoystickUpdate,
    WGI_JoystickClose,
    WGI_JoystickQuit,
    WGI_JoystickGetGamepadMapping
};

#endif // SDL_JOYSTICK_WGI
