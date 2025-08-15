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

#ifdef SDL_JOYSTICK_HIDAPI_SHIELD

// Define this if you want to log all packets from the controller
// #define DEBUG_SHIELD_PROTOCOL

#define CMD_BATTERY_STATE 0x07
#define CMD_RUMBLE        0x39
#define CMD_CHARGE_STATE  0x3A

// Milliseconds between polls of battery state
#define BATTERY_POLL_INTERVAL_MS 60000

// Milliseconds between retransmission of rumble to keep motors running
#define RUMBLE_REFRESH_INTERVAL_MS 500

// Reports that are too small are dropped over Bluetooth
#define HID_REPORT_SIZE 33

enum
{
    SDL_GAMEPAD_BUTTON_SHIELD_SHARE = 11,
    SDL_GAMEPAD_BUTTON_SHIELD_V103_TOUCHPAD,
    SDL_GAMEPAD_BUTTON_SHIELD_V103_MINUS,
    SDL_GAMEPAD_BUTTON_SHIELD_V103_PLUS,
    SDL_GAMEPAD_NUM_SHIELD_V103_BUTTONS,

    SDL_GAMEPAD_NUM_SHIELD_V104_BUTTONS = SDL_GAMEPAD_BUTTON_SHIELD_SHARE + 1,
};

typedef enum
{
    k_ShieldReportIdControllerState = 0x01,
    k_ShieldReportIdControllerTouch = 0x02,
    k_ShieldReportIdCommandResponse = 0x03,
    k_ShieldReportIdCommandRequest = 0x04,
} EShieldReportId;

// This same report structure is used for both requests and responses
typedef struct
{
    Uint8 report_id;
    Uint8 cmd;
    Uint8 seq_num;
    Uint8 payload[HID_REPORT_SIZE - 3];
} ShieldCommandReport_t;
SDL_COMPILE_TIME_ASSERT(ShieldCommandReport_t, sizeof(ShieldCommandReport_t) == HID_REPORT_SIZE);

typedef struct
{
    Uint8 seq_num;

    bool has_charging;
    Uint8 charging;
    bool has_battery_level;
    Uint8 battery_level;
    Uint64 last_battery_query_time;

    bool rumble_report_pending;
    bool rumble_update_pending;
    Uint8 left_motor_amplitude;
    Uint8 right_motor_amplitude;
    Uint64 last_rumble_time;

    Uint8 last_state[USB_PACKET_LENGTH];
} SDL_DriverShield_Context;

static void HIDAPI_DriverShield_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_SHIELD, callback, userdata);
}

static void HIDAPI_DriverShield_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_SHIELD, callback, userdata);
}

static bool HIDAPI_DriverShield_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_SHIELD, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));
}

static bool HIDAPI_DriverShield_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    return SDL_IsJoystickNVIDIASHIELDController(vendor_id, product_id);
}

static bool HIDAPI_DriverShield_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverShield_Context *ctx;

    ctx = (SDL_DriverShield_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    device->context = ctx;

    HIDAPI_SetDeviceName(device, "NVIDIA SHIELD Controller");

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverShield_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverShield_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static bool HIDAPI_DriverShield_SendCommand(SDL_HIDAPI_Device *device, Uint8 cmd, const void *data, int size)
{
    SDL_DriverShield_Context *ctx = (SDL_DriverShield_Context *)device->context;
    ShieldCommandReport_t cmd_pkt;

    if (size > sizeof(cmd_pkt.payload)) {
        return SDL_SetError("Command data exceeds HID report size");
    }

    if (!SDL_HIDAPI_LockRumble()) {
        return false;
    }

    cmd_pkt.report_id = k_ShieldReportIdCommandRequest;
    cmd_pkt.cmd = cmd;
    cmd_pkt.seq_num = ctx->seq_num++;
    if (data) {
        SDL_memcpy(cmd_pkt.payload, data, size);
    }

    // Zero unused data in the payload
    if (size != sizeof(cmd_pkt.payload)) {
        SDL_memset(&cmd_pkt.payload[size], 0, sizeof(cmd_pkt.payload) - size);
    }

    if (SDL_HIDAPI_SendRumbleAndUnlock(device, (Uint8 *)&cmd_pkt, sizeof(cmd_pkt)) != sizeof(cmd_pkt)) {
        return SDL_SetError("Couldn't send command packet");
    }

    return true;
}

static bool HIDAPI_DriverShield_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverShield_Context *ctx = (SDL_DriverShield_Context *)device->context;

    SDL_AssertJoysticksLocked();

    ctx->rumble_report_pending = false;
    ctx->rumble_update_pending = false;
    ctx->left_motor_amplitude = 0;
    ctx->right_motor_amplitude = 0;
    ctx->last_rumble_time = 0;
    SDL_zeroa(ctx->last_state);

    // Initialize the joystick capabilities
    if (device->product_id == USB_PRODUCT_NVIDIA_SHIELD_CONTROLLER_V103) {
        joystick->nbuttons = SDL_GAMEPAD_NUM_SHIELD_V103_BUTTONS;
        joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
        joystick->nhats = 1;

        SDL_PrivateJoystickAddTouchpad(joystick, 1);
    } else {
        joystick->nbuttons = SDL_GAMEPAD_NUM_SHIELD_V104_BUTTONS;
        joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
        joystick->nhats = 1;
    }

    // Request battery and charging info
    ctx->last_battery_query_time = SDL_GetTicks();
    HIDAPI_DriverShield_SendCommand(device, CMD_CHARGE_STATE, NULL, 0);
    HIDAPI_DriverShield_SendCommand(device, CMD_BATTERY_STATE, NULL, 0);

    return true;
}

static bool HIDAPI_DriverShield_SendNextRumble(SDL_HIDAPI_Device *device)
{
    SDL_DriverShield_Context *ctx = (SDL_DriverShield_Context *)device->context;
    Uint8 rumble_data[3];

    if (!ctx->rumble_update_pending) {
        return true;
    }

    rumble_data[0] = 0x01; // enable
    rumble_data[1] = ctx->left_motor_amplitude;
    rumble_data[2] = ctx->right_motor_amplitude;

    ctx->rumble_update_pending = false;
    ctx->last_rumble_time = SDL_GetTicks();

    return HIDAPI_DriverShield_SendCommand(device, CMD_RUMBLE, rumble_data, sizeof(rumble_data));
}

static bool HIDAPI_DriverShield_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    if (device->product_id == USB_PRODUCT_NVIDIA_SHIELD_CONTROLLER_V103) {
        Uint8 rumble_packet[] = { 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

        rumble_packet[2] = (low_frequency_rumble >> 8);
        rumble_packet[4] = (high_frequency_rumble >> 8);

        if (SDL_HIDAPI_SendRumble(device, rumble_packet, sizeof(rumble_packet)) != sizeof(rumble_packet)) {
            return SDL_SetError("Couldn't send rumble packet");
        }
        return true;

    } else {
        SDL_DriverShield_Context *ctx = (SDL_DriverShield_Context *)device->context;

        // The rumble motors are quite intense, so tone down the intensity like the official driver does
        ctx->left_motor_amplitude = low_frequency_rumble >> 11;
        ctx->right_motor_amplitude = high_frequency_rumble >> 11;
        ctx->rumble_update_pending = true;

        if (ctx->rumble_report_pending) {
            // We will service this after the hardware acknowledges the previous request
            return true;
        }

        return HIDAPI_DriverShield_SendNextRumble(device);
    }
}

static bool HIDAPI_DriverShield_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverShield_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    return SDL_JOYSTICK_CAP_RUMBLE;
}

static bool HIDAPI_DriverShield_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverShield_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    const Uint8 *data_bytes = (const Uint8 *)data;

    if (size > 1) {
        // Single command byte followed by a variable length payload
        return HIDAPI_DriverShield_SendCommand(device, data_bytes[0], &data_bytes[1], size - 1);
    } else if (size == 1) {
        // Single command byte with no payload
        return HIDAPI_DriverShield_SendCommand(device, data_bytes[0], NULL, 0);
    } else {
        return SDL_SetError("Effect data must at least contain a command byte");
    }
}

static bool HIDAPI_DriverShield_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

static void HIDAPI_DriverShield_HandleStatePacketV103(SDL_Joystick *joystick, SDL_DriverShield_Context *ctx, Uint8 *data, int size)
{
    Uint64 timestamp = SDL_GetTicksNS();

    if (ctx->last_state[3] != data[3]) {
        Uint8 hat;

        switch (data[3]) {
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

    if (ctx->last_state[1] != data[1]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[1] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[1] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[1] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[1] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[1] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[1] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[1] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[1] & 0x80) != 0));
    }

    if (ctx->last_state[2] != data[2]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[2] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SHIELD_V103_PLUS, ((data[2] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SHIELD_V103_MINUS, ((data[2] & 0x10) != 0));
        //SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[2] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[2] & 0x40) != 0));
        //SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SHIELD_SHARE, ((data[2] & 0x80) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[2] & 0x80) != 0));
    }

    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_Swap16LE(*(Sint16 *)&data[4]) - 0x8000);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_Swap16LE(*(Sint16 *)&data[6]) - 0x8000);

    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, SDL_Swap16LE(*(Sint16 *)&data[8]) - 0x8000);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, SDL_Swap16LE(*(Sint16 *)&data[10]) - 0x8000);

    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, SDL_Swap16LE(*(Sint16 *)&data[12]) - 0x8000);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, SDL_Swap16LE(*(Sint16 *)&data[14]) - 0x8000);

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

#undef clamp
#define clamp(val, min, max) (((val) > (max)) ? (max) : (((val) < (min)) ? (min) : (val)))

static void HIDAPI_DriverShield_HandleTouchPacketV103(SDL_Joystick *joystick, SDL_DriverShield_Context *ctx, const Uint8 *data, int size)
{
    bool touchpad_down;
    float touchpad_x, touchpad_y;
    Uint64 timestamp = SDL_GetTicksNS();

    SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SHIELD_V103_TOUCHPAD, ((data[1] & 0x01) != 0));

    // It's a triangular pad, but just use the center as the usable touch area
    touchpad_down = ((data[1] & 0x80) == 0);
    touchpad_x = clamp((float)(data[2] - 0x70) / 0x50, 0.0f, 1.0f);
    touchpad_y = clamp((float)(data[4] - 0x40) / 0x15, 0.0f, 1.0f);
    SDL_SendJoystickTouchpad(timestamp, joystick, 0, 0, touchpad_down, touchpad_x, touchpad_y, touchpad_down ? 1.0f : 0.0f);
}

static void HIDAPI_DriverShield_HandleStatePacketV104(SDL_Joystick *joystick, SDL_DriverShield_Context *ctx, Uint8 *data, int size)
{
    Uint64 timestamp = SDL_GetTicksNS();

    if (size < 23) {
        return;
    }

    if (ctx->last_state[2] != data[2]) {
        Uint8 hat;

        switch (data[2]) {
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

    if (ctx->last_state[3] != data[3]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[3] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[3] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[3] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[3] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[3] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[3] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[3] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[3] & 0x80) != 0));
    }

    if (ctx->last_state[4] != data[4]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[4] & 0x01) != 0));
    }

    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_Swap16LE(*(Sint16 *)&data[9]) - 0x8000);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_Swap16LE(*(Sint16 *)&data[11]) - 0x8000);

    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, SDL_Swap16LE(*(Sint16 *)&data[13]) - 0x8000);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, SDL_Swap16LE(*(Sint16 *)&data[15]) - 0x8000);

    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, SDL_Swap16LE(*(Sint16 *)&data[19]) - 0x8000);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, SDL_Swap16LE(*(Sint16 *)&data[21]) - 0x8000);

    if (ctx->last_state[17] != data[17]) {
        //SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SHIELD_SHARE, ((data[17] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[17] & 0x02) != 0));
        //SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[17] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[17] & 0x01) != 0));
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static void HIDAPI_DriverShield_UpdatePowerInfo(SDL_Joystick *joystick, SDL_DriverShield_Context *ctx)
{
    if (!ctx->has_charging || !ctx->has_battery_level) {
        return;
    }

    SDL_PowerState state = ctx->charging ? SDL_POWERSTATE_CHARGING : SDL_POWERSTATE_ON_BATTERY;
    int percent = ctx->battery_level * 20;
    SDL_SendJoystickPowerInfo(joystick, state, percent);
}

static bool HIDAPI_DriverShield_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverShield_Context *ctx = (SDL_DriverShield_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size = 0;
    ShieldCommandReport_t *cmd_resp_report;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    } else {
        return false;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_SHIELD_PROTOCOL
        HIDAPI_DumpPacket("NVIDIA SHIELD packet: size = %d", data, size);
#endif

        // Byte 0 is HID report ID
        switch (data[0]) {
        case k_ShieldReportIdControllerState:
            if (!joystick) {
                break;
            }
            if (size == 16) {
                HIDAPI_DriverShield_HandleStatePacketV103(joystick, ctx, data, size);
            } else {
                HIDAPI_DriverShield_HandleStatePacketV104(joystick, ctx, data, size);
            }
            break;
        case k_ShieldReportIdControllerTouch:
            if (!joystick) {
                break;
            }
            HIDAPI_DriverShield_HandleTouchPacketV103(joystick, ctx, data, size);
            break;
        case k_ShieldReportIdCommandResponse:
            cmd_resp_report = (ShieldCommandReport_t *)data;
            switch (cmd_resp_report->cmd) {
            case CMD_RUMBLE:
                ctx->rumble_report_pending = false;
                HIDAPI_DriverShield_SendNextRumble(device);
                break;
            case CMD_CHARGE_STATE:
                ctx->has_charging = true;
                ctx->charging = cmd_resp_report->payload[0];
                HIDAPI_DriverShield_UpdatePowerInfo(joystick, ctx);
                break;
            case CMD_BATTERY_STATE:
                ctx->has_battery_level = true;
                ctx->battery_level = cmd_resp_report->payload[2];
                HIDAPI_DriverShield_UpdatePowerInfo(joystick, ctx);
                break;
            }
            break;
        }
    }

    // Ask for battery state again if we're due for an update
    if (joystick && SDL_GetTicks() >= (ctx->last_battery_query_time + BATTERY_POLL_INTERVAL_MS)) {
        ctx->last_battery_query_time = SDL_GetTicks();
        HIDAPI_DriverShield_SendCommand(device, CMD_BATTERY_STATE, NULL, 0);
    }

    // Retransmit rumble packets if they've lasted longer than the hardware supports
    if ((ctx->left_motor_amplitude != 0 || ctx->right_motor_amplitude != 0) &&
        SDL_GetTicks() >= (ctx->last_rumble_time + RUMBLE_REFRESH_INTERVAL_MS)) {
        ctx->rumble_update_pending = true;
        HIDAPI_DriverShield_SendNextRumble(device);
    }

    if (size < 0) {
        // Read error, device is disconnected
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return (size >= 0);
}

static void HIDAPI_DriverShield_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
}

static void HIDAPI_DriverShield_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverShield = {
    SDL_HINT_JOYSTICK_HIDAPI_SHIELD,
    true,
    HIDAPI_DriverShield_RegisterHints,
    HIDAPI_DriverShield_UnregisterHints,
    HIDAPI_DriverShield_IsEnabled,
    HIDAPI_DriverShield_IsSupportedDevice,
    HIDAPI_DriverShield_InitDevice,
    HIDAPI_DriverShield_GetDevicePlayerIndex,
    HIDAPI_DriverShield_SetDevicePlayerIndex,
    HIDAPI_DriverShield_UpdateDevice,
    HIDAPI_DriverShield_OpenJoystick,
    HIDAPI_DriverShield_RumbleJoystick,
    HIDAPI_DriverShield_RumbleJoystickTriggers,
    HIDAPI_DriverShield_GetJoystickCapabilities,
    HIDAPI_DriverShield_SetJoystickLED,
    HIDAPI_DriverShield_SendJoystickEffect,
    HIDAPI_DriverShield_SetJoystickSensorsEnabled,
    HIDAPI_DriverShield_CloseJoystick,
    HIDAPI_DriverShield_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_SHIELD

#endif // SDL_JOYSTICK_HIDAPI
