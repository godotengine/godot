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

#include "../SDL_sysjoystick.h"
#include "../../core/windows/SDL_windows.h"
#include "../../core/windows/SDL_directx.h"

#define MAX_INPUTS 256 // each joystick can have up to 256 inputs

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

typedef struct JoyStick_DeviceData
{
    SDL_GUID guid;
    char *joystickname;
    Uint8 send_add_event;
    SDL_JoystickID nInstanceID;
    bool bXInputDevice;
    BYTE SubType;
    Uint8 XInputUserId;
    DIDEVICEINSTANCE dxdevice;
    char path[MAX_PATH];
    int steam_virtual_gamepad_slot;
    struct JoyStick_DeviceData *pNext;
} JoyStick_DeviceData;

extern JoyStick_DeviceData *SYS_Joystick; // array to hold joystick ID values

typedef enum Type
{
    BUTTON,
    AXIS,
    HAT
} Type;

typedef struct input_t
{
    // DirectInput offset for this input type:
    DWORD ofs;

    // Button, axis or hat:
    Type type;

    // SDL input offset:
    Uint8 num;
} input_t;

// The private structure used to keep track of a joystick
struct joystick_hwdata
{
    SDL_GUID guid;

#ifdef SDL_JOYSTICK_DINPUT
    LPDIRECTINPUTDEVICE8 InputDevice;
    DIDEVCAPS Capabilities;
    bool buffered;
    bool first_update;
    input_t Inputs[MAX_INPUTS];
    int NumInputs;
    int NumSliders;
    bool ff_initialized;
    DIEFFECT *ffeffect;
    LPDIRECTINPUTEFFECT ffeffect_ref;
#endif

    bool bXInputDevice; // true if this device supports using the xinput API rather than DirectInput
    bool bXInputHaptic; // Supports force feedback via XInput.
    Uint8 userid;           // XInput userid index for this joystick
    DWORD dwPacketNumber;
};

#ifdef SDL_JOYSTICK_DINPUT
extern const DIDATAFORMAT SDL_c_dfDIJoystick2;
#endif

extern void WINDOWS_AddJoystickDevice(JoyStick_DeviceData *device);

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif
