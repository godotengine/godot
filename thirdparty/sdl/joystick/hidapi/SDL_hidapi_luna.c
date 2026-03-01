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

#include "../SDL_sysjoystick.h"
#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"

#ifdef SDL_JOYSTICK_HIDAPI_LUNA

// Define this if you want to log all packets from the controller
// #define DEBUG_LUNA_PROTOCOL

// Sending rumble on macOS blocks for a long time and eventually fails
#ifndef SDL_PLATFORM_MACOS
#define ENABLE_LUNA_BLUETOOTH_RUMBLE
#endif

enum
{
    SDL_GAMEPAD_BUTTON_LUNA_MICROPHONE = 11,
    SDL_GAMEPAD_NUM_LUNA_BUTTONS,
};

typedef struct
{
    Uint8 last_state[USB_PACKET_LENGTH];
} SDL_DriverLuna_Context;

static void HIDAPI_DriverLuna_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_LUNA, callback, userdata);
}

static void HIDAPI_DriverLuna_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_LUNA, callback, userdata);
}

static bool HIDAPI_DriverLuna_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_LUNA, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));
}

static bool HIDAPI_DriverLuna_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    return SDL_IsJoystickAmazonLunaController(vendor_id, product_id);
}

static bool HIDAPI_DriverLuna_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverLuna_Context *ctx;

    ctx = (SDL_DriverLuna_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    device->context = ctx;

    HIDAPI_SetDeviceName(device, "Amazon Luna Controller");

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverLuna_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverLuna_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static bool HIDAPI_DriverLuna_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverLuna_Context *ctx = (SDL_DriverLuna_Context *)device->context;

    SDL_AssertJoysticksLocked();

    SDL_zeroa(ctx->last_state);

    // Initialize the joystick capabilities
    joystick->nbuttons = SDL_GAMEPAD_NUM_LUNA_BUTTONS;
    joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
    joystick->nhats = 1;

    return true;
}

static bool HIDAPI_DriverLuna_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
#ifdef ENABLE_LUNA_BLUETOOTH_RUMBLE
    if (device->product_id == BLUETOOTH_PRODUCT_LUNA_CONTROLLER) {
        // Same packet as on Xbox One controllers connected via Bluetooth
        Uint8 rumble_packet[] = { 0x03, 0x0F, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0xEB };

        // Magnitude is 1..100 so scale the 16-bit input here
        rumble_packet[4] = (Uint8)(low_frequency_rumble / 655);
        rumble_packet[5] = (Uint8)(high_frequency_rumble / 655);

        if (SDL_HIDAPI_SendRumble(device, rumble_packet, sizeof(rumble_packet)) != sizeof(rumble_packet)) {
            return SDL_SetError("Couldn't send rumble packet");
        }

        return true;
    }
#endif // ENABLE_LUNA_BLUETOOTH_RUMBLE

    // There is currently no rumble packet over USB
    return SDL_Unsupported();
}

static bool HIDAPI_DriverLuna_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverLuna_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    Uint32 result = 0;

#ifdef ENABLE_LUNA_BLUETOOTH_RUMBLE
    if (device->product_id == BLUETOOTH_PRODUCT_LUNA_CONTROLLER) {
        result |= SDL_JOYSTICK_CAP_RUMBLE;
    }
#endif

    return result;
}

static bool HIDAPI_DriverLuna_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverLuna_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverLuna_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

static void HIDAPI_DriverLuna_HandleUSBStatePacket(SDL_Joystick *joystick, SDL_DriverLuna_Context *ctx, Uint8 *data, int size)
{
    Uint64 timestamp = SDL_GetTicksNS();

    if (ctx->last_state[1] != data[1]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[1] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[1] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[1] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[1] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[1] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[1] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[1] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[1] & 0x80) != 0));
    }
    if (ctx->last_state[2] != data[2]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[2] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LUNA_MICROPHONE, ((data[2] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[2] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[2] & 0x08) != 0));
    }

    if (ctx->last_state[3] != data[3]) {
        Uint8 hat;

        switch (data[3] & 0x0f) {
        case 0:
            hat = SDL_HAT_UP;
            break;
        case 1:
            hat = SDL_HAT_RIGHTUP;
            break;
        case 2:
            hat = SDL_HAT_RIGHT;
            break;
        case 3:
            hat = SDL_HAT_RIGHTDOWN;
            break;
        case 4:
            hat = SDL_HAT_DOWN;
            break;
        case 5:
            hat = SDL_HAT_LEFTDOWN;
            break;
        case 6:
            hat = SDL_HAT_LEFT;
            break;
        case 7:
            hat = SDL_HAT_LEFTUP;
            break;
        default:
            hat = SDL_HAT_CENTERED;
            break;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);
    }

#define READ_STICK_AXIS(offset) \
    (data[offset] == 0x7f ? 0 : (Sint16)HIDAPI_RemapVal((float)data[offset], 0x00, 0xff, SDL_MIN_SINT16, SDL_MAX_SINT16))
    {
        Sint16 axis = READ_STICK_AXIS(4);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
        axis = READ_STICK_AXIS(5);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
        axis = READ_STICK_AXIS(6);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
        axis = READ_STICK_AXIS(7);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);
    }
#undef READ_STICK_AXIS

#define READ_TRIGGER_AXIS(offset) \
    (Sint16) HIDAPI_RemapVal((float)data[offset], 0x00, 0xff, SDL_MIN_SINT16, SDL_MAX_SINT16)
    {
        Sint16 axis = READ_TRIGGER_AXIS(8);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);
        axis = READ_TRIGGER_AXIS(9);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);
    }
#undef READ_TRIGGER_AXIS

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static void HIDAPI_DriverLuna_HandleBluetoothStatePacket(SDL_Joystick *joystick, SDL_DriverLuna_Context *ctx, Uint8 *data, int size)
{
    Uint64 timestamp = SDL_GetTicksNS();

    if (size >= 2 && data[0] == 0x02) {
        // Home button has dedicated report
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[1] & 0x1) != 0));
        return;
    }

    if (size >= 2 && data[0] == 0x04) {
        // Battery level report
        int percent = (int)SDL_roundf((data[1] / 255.0f) * 100.0f);
        SDL_SendJoystickPowerInfo(joystick, SDL_POWERSTATE_ON_BATTERY, percent);
        return;
    }

    if (size < 17 || data[0] != 0x01) {
        // We don't know how to handle this report
        return;
    }

    if (ctx->last_state[13] != data[13]) {
        Uint8 hat;

        switch (data[13] & 0x0f) {
        case 1:
            hat = SDL_HAT_UP;
            break;
        case 2:
            hat = SDL_HAT_RIGHTUP;
            break;
        case 3:
            hat = SDL_HAT_RIGHT;
            break;
        case 4:
            hat = SDL_HAT_RIGHTDOWN;
            break;
        case 5:
            hat = SDL_HAT_DOWN;
            break;
        case 6:
            hat = SDL_HAT_LEFTDOWN;
            break;
        case 7:
            hat = SDL_HAT_LEFT;
            break;
        case 8:
            hat = SDL_HAT_LEFTUP;
            break;
        default:
            hat = SDL_HAT_CENTERED;
            break;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);
    }

    if (ctx->last_state[14] != data[14]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[14] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[14] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[14] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[14] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[14] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[14] & 0x80) != 0));
    }
    if (ctx->last_state[15] != data[15]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[15] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[15] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[15] & 0x40) != 0));
    }
    if (ctx->last_state[16] != data[16]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[16] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LUNA_MICROPHONE, ((data[16] & 0x02) != 0));
    }

#define READ_STICK_AXIS(offset) \
    (data[offset] == 0x7f ? 0 : (Sint16)HIDAPI_RemapVal((float)data[offset], 0x00, 0xff, SDL_MIN_SINT16, SDL_MAX_SINT16))
    {
        Sint16 axis = READ_STICK_AXIS(2);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
        axis = READ_STICK_AXIS(4);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
        axis = READ_STICK_AXIS(6);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
        axis = READ_STICK_AXIS(8);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);
    }
#undef READ_STICK_AXIS

#define READ_TRIGGER_AXIS(offset) \
    (Sint16) HIDAPI_RemapVal((float)((int)(((data[offset] | (data[offset + 1] << 8)) & 0x3ff) - 0x200)), 0x00 - 0x200, 0x3ff - 0x200, SDL_MIN_SINT16, SDL_MAX_SINT16)
    {
        Sint16 axis = READ_TRIGGER_AXIS(9);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);
        axis = READ_TRIGGER_AXIS(11);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);
    }
#undef READ_TRIGGER_AXIS

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static bool HIDAPI_DriverLuna_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverLuna_Context *ctx = (SDL_DriverLuna_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size = 0;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    } else {
        return false;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_LUNA_PROTOCOL
        HIDAPI_DumpPacket("Amazon Luna packet: size = %d", data, size);
#endif
        if (!joystick) {
            continue;
        }

        switch (size) {
        case 10:
            HIDAPI_DriverLuna_HandleUSBStatePacket(joystick, ctx, data, size);
            break;
        default:
            HIDAPI_DriverLuna_HandleBluetoothStatePacket(joystick, ctx, data, size);
            break;
        }
    }

    if (size < 0) {
        // Read error, device is disconnected
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return (size >= 0);
}

static void HIDAPI_DriverLuna_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
}

static void HIDAPI_DriverLuna_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverLuna = {
    SDL_HINT_JOYSTICK_HIDAPI_LUNA,
    true,
    HIDAPI_DriverLuna_RegisterHints,
    HIDAPI_DriverLuna_UnregisterHints,
    HIDAPI_DriverLuna_IsEnabled,
    HIDAPI_DriverLuna_IsSupportedDevice,
    HIDAPI_DriverLuna_InitDevice,
    HIDAPI_DriverLuna_GetDevicePlayerIndex,
    HIDAPI_DriverLuna_SetDevicePlayerIndex,
    HIDAPI_DriverLuna_UpdateDevice,
    HIDAPI_DriverLuna_OpenJoystick,
    HIDAPI_DriverLuna_RumbleJoystick,
    HIDAPI_DriverLuna_RumbleJoystickTriggers,
    HIDAPI_DriverLuna_GetJoystickCapabilities,
    HIDAPI_DriverLuna_SetJoystickLED,
    HIDAPI_DriverLuna_SendJoystickEffect,
    HIDAPI_DriverLuna_SetJoystickSensorsEnabled,
    HIDAPI_DriverLuna_CloseJoystick,
    HIDAPI_DriverLuna_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_LUNA

#endif // SDL_JOYSTICK_HIDAPI
