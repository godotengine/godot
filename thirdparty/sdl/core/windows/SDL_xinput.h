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

#ifndef SDL_xinput_h_
#define SDL_xinput_h_

#include "SDL_windows.h"

#ifdef HAVE_XINPUT_H
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
// Xbox supports an XInput wrapper which is a C++-only header...
#include <math.h> // Required to compile with recent MSVC...
#include <XInputOnGameInput.h>
using namespace XInputOnGameInput;
#else
#include <xinput.h>
#endif
#endif // HAVE_XINPUT_H

#ifndef XUSER_MAX_COUNT
#define XUSER_MAX_COUNT 4
#endif
#ifndef XUSER_INDEX_ANY
#define XUSER_INDEX_ANY 0x000000FF
#endif
#ifndef XINPUT_CAPS_FFB_SUPPORTED
#define XINPUT_CAPS_FFB_SUPPORTED 0x0001
#endif
#ifndef XINPUT_CAPS_WIRELESS
#define XINPUT_CAPS_WIRELESS 0x0002
#endif

#ifndef XINPUT_DEVSUBTYPE_UNKNOWN
#define XINPUT_DEVSUBTYPE_UNKNOWN 0x00
#endif
#ifndef XINPUT_DEVSUBTYPE_GAMEPAD
#define XINPUT_DEVSUBTYPE_GAMEPAD 0x01
#endif
#ifndef XINPUT_DEVSUBTYPE_WHEEL
#define XINPUT_DEVSUBTYPE_WHEEL 0x02
#endif
#ifndef XINPUT_DEVSUBTYPE_ARCADE_STICK
#define XINPUT_DEVSUBTYPE_ARCADE_STICK 0x03
#endif
#ifndef XINPUT_DEVSUBTYPE_FLIGHT_STICK
#define XINPUT_DEVSUBTYPE_FLIGHT_STICK 0x04
#endif
#ifndef XINPUT_DEVSUBTYPE_DANCE_PAD
#define XINPUT_DEVSUBTYPE_DANCE_PAD 0x05
#endif
#ifndef XINPUT_DEVSUBTYPE_GUITAR
#define XINPUT_DEVSUBTYPE_GUITAR 0x06
#endif
#ifndef XINPUT_DEVSUBTYPE_GUITAR_ALTERNATE
#define XINPUT_DEVSUBTYPE_GUITAR_ALTERNATE 0x07
#endif
#ifndef XINPUT_DEVSUBTYPE_DRUM_KIT
#define XINPUT_DEVSUBTYPE_DRUM_KIT 0x08
#endif
#ifndef XINPUT_DEVSUBTYPE_GUITAR_BASS
#define XINPUT_DEVSUBTYPE_GUITAR_BASS 0x0B
#endif
#ifndef XINPUT_DEVSUBTYPE_ARCADE_PAD
#define XINPUT_DEVSUBTYPE_ARCADE_PAD 0x13
#endif

#ifndef XINPUT_FLAG_GAMEPAD
#define XINPUT_FLAG_GAMEPAD 0x01
#endif

#ifndef XINPUT_GAMEPAD_DPAD_UP
#define XINPUT_GAMEPAD_DPAD_UP 0x0001
#endif
#ifndef XINPUT_GAMEPAD_DPAD_DOWN
#define XINPUT_GAMEPAD_DPAD_DOWN 0x0002
#endif
#ifndef XINPUT_GAMEPAD_DPAD_LEFT
#define XINPUT_GAMEPAD_DPAD_LEFT 0x0004
#endif
#ifndef XINPUT_GAMEPAD_DPAD_RIGHT
#define XINPUT_GAMEPAD_DPAD_RIGHT 0x0008
#endif
#ifndef XINPUT_GAMEPAD_START
#define XINPUT_GAMEPAD_START 0x0010
#endif
#ifndef XINPUT_GAMEPAD_BACK
#define XINPUT_GAMEPAD_BACK 0x0020
#endif
#ifndef XINPUT_GAMEPAD_LEFT_THUMB
#define XINPUT_GAMEPAD_LEFT_THUMB 0x0040
#endif
#ifndef XINPUT_GAMEPAD_RIGHT_THUMB
#define XINPUT_GAMEPAD_RIGHT_THUMB 0x0080
#endif
#ifndef XINPUT_GAMEPAD_LEFT_SHOULDER
#define XINPUT_GAMEPAD_LEFT_SHOULDER 0x0100
#endif
#ifndef XINPUT_GAMEPAD_RIGHT_SHOULDER
#define XINPUT_GAMEPAD_RIGHT_SHOULDER 0x0200
#endif
#ifndef XINPUT_GAMEPAD_A
#define XINPUT_GAMEPAD_A 0x1000
#endif
#ifndef XINPUT_GAMEPAD_B
#define XINPUT_GAMEPAD_B 0x2000
#endif
#ifndef XINPUT_GAMEPAD_X
#define XINPUT_GAMEPAD_X 0x4000
#endif
#ifndef XINPUT_GAMEPAD_Y
#define XINPUT_GAMEPAD_Y 0x8000
#endif

#ifndef XINPUT_GAMEPAD_GUIDE
#define XINPUT_GAMEPAD_GUIDE 0x0400
#endif

#ifndef BATTERY_DEVTYPE_GAMEPAD
#define BATTERY_DEVTYPE_GAMEPAD 0x00
#endif

#ifndef BATTERY_TYPE_DISCONNECTED
#define BATTERY_TYPE_DISCONNECTED 0x00
#endif
#ifndef BATTERY_TYPE_WIRED
#define BATTERY_TYPE_WIRED 0x01
#endif
#ifndef BATTERY_TYPE_UNKNOWN
#define BATTERY_TYPE_UNKNOWN 0xFF
#endif
#ifndef BATTERY_LEVEL_EMPTY
#define BATTERY_LEVEL_EMPTY 0x00
#endif
#ifndef BATTERY_LEVEL_LOW
#define BATTERY_LEVEL_LOW 0x01
#endif
#ifndef BATTERY_LEVEL_MEDIUM
#define BATTERY_LEVEL_MEDIUM 0x02
#endif
#ifndef BATTERY_LEVEL_FULL
#define BATTERY_LEVEL_FULL 0x03
#endif

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

// typedef's for XInput structs we use


// This is the same as XINPUT_BATTERY_INFORMATION, but always defined instead of just if WIN32_WINNT >= _WIN32_WINNT_WIN8
typedef struct
{
    BYTE BatteryType;
    BYTE BatteryLevel;
} XINPUT_BATTERY_INFORMATION_EX;

#ifndef HAVE_XINPUT_H

typedef struct
{
    WORD wButtons;
    BYTE bLeftTrigger;
    BYTE bRightTrigger;
    SHORT sThumbLX;
    SHORT sThumbLY;
    SHORT sThumbRX;
    SHORT sThumbRY;
} XINPUT_GAMEPAD;

typedef struct
{
    DWORD dwPacketNumber;
    XINPUT_GAMEPAD Gamepad;
} XINPUT_STATE;

typedef struct
{
    WORD wLeftMotorSpeed;
    WORD wRightMotorSpeed;
} XINPUT_VIBRATION;

typedef struct
{
    BYTE Type;
    BYTE SubType;
    WORD Flags;
    XINPUT_GAMEPAD Gamepad;
    XINPUT_VIBRATION Vibration;
} XINPUT_CAPABILITIES;

#endif // HAVE_XINPUT_H

// This struct is not defined in XInput headers.
typedef struct
{
    XINPUT_CAPABILITIES Capabilities;
    WORD VendorId;
    WORD ProductId;
    WORD ProductVersion;
    WORD unk1;
    DWORD unk2;
} SDL_XINPUT_CAPABILITIES_EX;

// Forward decl's for XInput API's we load dynamically and use if available
typedef DWORD(WINAPI *XInputGetState_t)(
    DWORD dwUserIndex,      // [in] Index of the gamer associated with the device
    XINPUT_STATE *pState    // [out] Receives the current state
);

typedef DWORD(WINAPI *XInputSetState_t)(
    DWORD dwUserIndex,           // [in] Index of the gamer associated with the device
    XINPUT_VIBRATION *pVibration // [in, out] The vibration information to send to the controller
);

typedef DWORD(WINAPI *XInputGetCapabilities_t)(
    DWORD dwUserIndex,                 // [in] Index of the gamer associated with the device
    DWORD dwFlags,                     // [in] Input flags that identify the device type
    XINPUT_CAPABILITIES *pCapabilities // [out] Receives the capabilities
);

// Only available in XInput 1.4 that is shipped with Windows 8 and newer.
typedef DWORD(WINAPI *XInputGetCapabilitiesEx_t)(
    DWORD dwReserved,                       // [in] Must be 1
    DWORD dwUserIndex,                      // [in] Index of the gamer associated with the device
    DWORD dwFlags,                          // [in] Input flags that identify the device type
    SDL_XINPUT_CAPABILITIES_EX *pCapabilitiesEx // [out] Receives the capabilities
);

typedef DWORD(WINAPI *XInputGetBatteryInformation_t)(
    DWORD dwUserIndex,
    BYTE devType,
    XINPUT_BATTERY_INFORMATION_EX *pBatteryInformation);

extern bool WIN_LoadXInputDLL(void);
extern void WIN_UnloadXInputDLL(void);

extern XInputGetState_t SDL_XInputGetState;
extern XInputSetState_t SDL_XInputSetState;
extern XInputGetCapabilities_t SDL_XInputGetCapabilities;
extern XInputGetCapabilitiesEx_t SDL_XInputGetCapabilitiesEx;
extern XInputGetBatteryInformation_t SDL_XInputGetBatteryInformation;

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif

#define XINPUTGETSTATE              SDL_XInputGetState
#define XINPUTSETSTATE              SDL_XInputSetState
#define XINPUTGETCAPABILITIES       SDL_XInputGetCapabilities
#define XINPUTGETCAPABILITIESEX     SDL_XInputGetCapabilitiesEx
#define XINPUTGETBATTERYINFORMATION SDL_XInputGetBatteryInformation

#endif // SDL_xinput_h_
