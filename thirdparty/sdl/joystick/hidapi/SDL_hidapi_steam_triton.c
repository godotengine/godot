/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

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

#ifdef SDL_JOYSTICK_HIDAPI_STEAM_TRITON

// Define this if you want to log all packets from the controller
#if 0
#define DEBUG_STEAM_PROTOCOL
#endif

/*****************************************************************************************************/

#include "steam/controller_constants.h"
#include "steam/controller_structs.h"

// Always 1kHz according to USB descriptor, but actually about 4 ms.
#define TRITON_SENSOR_UPDATE_INTERVAL_US 4032

// Steam Controller hardware safety timeout is around 50ms, so we resend rumble every 40ms
#define TRITON_RUMBLE_RESEND_INTERVAL_MS 40

enum
{
    SDL_GAMEPAD_BUTTON_TRITON_QAM = 11,
    SDL_GAMEPAD_BUTTON_TRITON_RIGHT_PADDLE1,
    SDL_GAMEPAD_BUTTON_TRITON_LEFT_PADDLE1,
    SDL_GAMEPAD_BUTTON_TRITON_RIGHT_PADDLE2,
    SDL_GAMEPAD_BUTTON_TRITON_LEFT_PADDLE2,
    SDL_GAMEPAD_BUTTON_TRITON_RIGHT_TOUCHPAD,
    SDL_GAMEPAD_BUTTON_TRITON_LEFT_TOUCHPAD,
    SDL_GAMEPAD_BUTTON_TRITON_RIGHT_JOYSTICK_TOUCH,
    SDL_GAMEPAD_BUTTON_TRITON_LEFT_JOYSTICK_TOUCH,
    SDL_GAMEPAD_BUTTON_TRITON_RIGHT_GRIP_TOUCH,
    SDL_GAMEPAD_BUTTON_TRITON_LEFT_GRIP_TOUCH,
    SDL_GAMEPAD_NUM_TRITON_BUTTONS,
};

typedef enum
{

    TRITON_LBUTTON_A            = 0x00000001,
    TRITON_LBUTTON_B            = 0x00000002,
    TRITON_LBUTTON_X            = 0x00000004,
    TRITON_LBUTTON_Y            = 0x00000008,

    TRITON_HBUTTON_QAM          = 0x00000010,
    TRITON_LBUTTON_R3           = 0x00000020,
    TRITON_LBUTTON_VIEW         = 0x00000040,
    TRITON_HBUTTON_R4           = 0x00000080,

    TRITON_LBUTTON_R5           = 0x00000100,
    TRITON_LBUTTON_R            = 0x00000200,
    TRITON_LBUTTON_DPAD_DOWN    = 0x00000400,
    TRITON_LBUTTON_DPAD_RIGHT   = 0x00000800,

    TRITON_LBUTTON_DPAD_LEFT    = 0x00001000,
    TRITON_LBUTTON_DPAD_UP      = 0x00002000,
    TRITON_LBUTTON_MENU         = 0x00004000,
    TRITON_LBUTTON_L3           = 0x00008000,

    TRITON_LBUTTON_STEAM        = 0x00010000,
    TRITON_HBUTTON_L4           = 0x00020000,
    TRITON_LBUTTON_L5           = 0x00040000,
    TRITON_LBUTTON_L            = 0x00080000,

    TRITON_RIGHT_JOYSTICK_TOUCH = 0x00100000,
    TRITON_RIGHT_TOUCHPAD_TOUCH = 0x00200000,
    TRITON_RIGHT_TOUCHPAD_CLICK = 0x00400000,
    TRITON_RIGHT_TRIGGER_CLICK  = 0x00800000,

    TRITON_LEFT_JOYSTICK_TOUCH  = 0x01000000,
    TRITON_LEFT_TOUCHPAD_TOUCH  = 0x02000000,
    TRITON_LEFT_TOUCHPAD_CLICK  = 0x04000000,
    TRITON_LEFT_TRIGGER_CLICK   = 0x08000000,

    TRITON_RIGHT_GRIP_TOUCH     = 0x10000000,
    TRITON_LEFT_GRIP_TOUCH      = 0x20000000,
} TritonButtons;

typedef struct
{
    bool connected;
    bool report_sensors;
    Uint16 last_sensor_tick16;
    Uint32 last_sensor_tick32;
    Uint64 sensor_timestamp_ns;
    Uint64 last_button_state;
    Uint64 last_lizard_update;
    Uint16 low_frequency_rumble;
    Uint16 high_frequency_rumble;
    Uint64 last_rumble_time;

    bool left_touch_down;
    float left_touch_x;
    float left_touch_y;
    bool right_touch_down;
    float right_touch_x;
    float right_touch_y;
} SDL_DriverSteamTriton_Context;

static bool IsProteusDongle(Uint16 product_id)
{
    return (product_id == USB_PRODUCT_VALVE_STEAM_PROTEUS_DONGLE ||
            product_id == USB_PRODUCT_VALVE_STEAM_NEREID_DONGLE);
}

static bool DisableSteamTritonLizardMode(SDL_hid_device *dev)
{
    int rc;
    Uint8 buffer[HID_FEATURE_REPORT_BYTES] = { 1 };
    FeatureReportMsg *msg = (FeatureReportMsg *)(buffer + 1);

    msg->header.type = ID_SET_SETTINGS_VALUES;
    msg->header.length = 1 * sizeof(ControllerSetting);
    msg->payload.setSettingsValues.settings[0].settingNum = SETTING_LIZARD_MODE;
    msg->payload.setSettingsValues.settings[0].settingValue = LIZARD_MODE_OFF;

    rc = SDL_hid_send_feature_report(dev, buffer, sizeof(buffer));
    if (rc != sizeof(buffer)) {
        return false;
    }

    return true;
}

// Triton newer state MTUs are identical until touchpads. Parse them using this routine.
// Expects report to be a TritonMTUNoQuat_t, so cast as needed
static void HIDAPI_DriverSteamTriton_HandleGenericState(SDL_DriverSteamTriton_Context *ctx, SDL_Joystick *joystick, Uint64 timestamp, TritonMTUNoQuat_t *pTritonReport)
{
    if (pTritonReport->buttons != ctx->last_button_state) {
        Uint8 hat = 0;

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH,
                               ((pTritonReport->buttons & TRITON_LBUTTON_A) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST,
                               ((pTritonReport->buttons & TRITON_LBUTTON_B) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST,
                               ((pTritonReport->buttons & TRITON_LBUTTON_X) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH,
                               ((pTritonReport->buttons & TRITON_LBUTTON_Y) != 0));

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER,
                               ((pTritonReport->buttons & TRITON_LBUTTON_L) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER,
                               ((pTritonReport->buttons & TRITON_LBUTTON_R) != 0));

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK,
                               ((pTritonReport->buttons & TRITON_LBUTTON_MENU) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START,
                               ((pTritonReport->buttons & TRITON_LBUTTON_VIEW) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE,
                               ((pTritonReport->buttons & TRITON_LBUTTON_STEAM) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_TRITON_QAM,
                               ((pTritonReport->buttons & TRITON_HBUTTON_QAM) != 0));

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK,
                               ((pTritonReport->buttons & TRITON_LBUTTON_L3) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK,
                               ((pTritonReport->buttons & TRITON_LBUTTON_R3) != 0));

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_TRITON_RIGHT_PADDLE1,
                               ((pTritonReport->buttons & TRITON_HBUTTON_R4) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_TRITON_LEFT_PADDLE1,
                               ((pTritonReport->buttons & TRITON_HBUTTON_L4) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_TRITON_RIGHT_PADDLE2,
                               ((pTritonReport->buttons & TRITON_LBUTTON_R5) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_TRITON_LEFT_PADDLE2,
                               ((pTritonReport->buttons & TRITON_LBUTTON_L5) != 0));

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_TRITON_RIGHT_TOUCHPAD,
                               ((pTritonReport->buttons & TRITON_RIGHT_TOUCHPAD_CLICK) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_TRITON_LEFT_TOUCHPAD,
                               ((pTritonReport->buttons & TRITON_LEFT_TOUCHPAD_CLICK) != 0));

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_TRITON_RIGHT_JOYSTICK_TOUCH,
                               ((pTritonReport->buttons & TRITON_RIGHT_JOYSTICK_TOUCH) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_TRITON_LEFT_JOYSTICK_TOUCH,
                               ((pTritonReport->buttons & TRITON_LEFT_JOYSTICK_TOUCH) != 0));

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_TRITON_RIGHT_GRIP_TOUCH,
                               ((pTritonReport->buttons & TRITON_RIGHT_GRIP_TOUCH) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_TRITON_LEFT_GRIP_TOUCH,
                               ((pTritonReport->buttons & TRITON_LEFT_GRIP_TOUCH) != 0));

        if (pTritonReport->buttons & TRITON_LBUTTON_DPAD_UP) {
            hat |= SDL_HAT_UP;
        }
        if (pTritonReport->buttons & TRITON_LBUTTON_DPAD_DOWN) {
            hat |= SDL_HAT_DOWN;
        }
        if (pTritonReport->buttons & TRITON_LBUTTON_DPAD_LEFT) {
            hat |= SDL_HAT_LEFT;
        }
        if (pTritonReport->buttons & TRITON_LBUTTON_DPAD_RIGHT) {
            hat |= SDL_HAT_RIGHT;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);

        ctx->last_button_state = pTritonReport->buttons;
    }

    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER,
                         (int)pTritonReport->sTriggerLeft * 2 - 32768);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER,
                         (int)pTritonReport->sTriggerRight * 2 - 32768);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX,
                         pTritonReport->sLeftStickX);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY,
                         -pTritonReport->sLeftStickY);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX,
                         pTritonReport->sRightStickX);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY,
                         -pTritonReport->sRightStickY);
}

static void HIDAPI_DriverSteamTriton_HandleState(SDL_HIDAPI_Device *device,
                                                 SDL_Joystick *joystick,
                                                 TritonMTUNoQuat_t *pTritonReport)
{
    float values[3];
    SDL_DriverSteamTriton_Context *ctx = (SDL_DriverSteamTriton_Context *)device->context;
    Uint64 timestamp = SDL_GetTicksNS();

    HIDAPI_DriverSteamTriton_HandleGenericState(ctx, joystick, timestamp, pTritonReport);

    bool left_touch_down = (pTritonReport->buttons & TRITON_LEFT_TOUCHPAD_TOUCH) ? true : false;
    bool right_touch_down = (pTritonReport->buttons & TRITON_RIGHT_TOUCHPAD_TOUCH) ? true : false;
 
    if (left_touch_down || ctx->left_touch_down) {
        if (left_touch_down) {
            ctx->left_touch_x = pTritonReport->sLeftPadX / 65536.0f + 0.5f;
            ctx->left_touch_y = -(float)pTritonReport->sLeftPadY / 65536.0f + 0.5f;
        }
        SDL_SendJoystickTouchpad(timestamp, joystick, 0, 0,
                                    left_touch_down,
                                    ctx->left_touch_x,
                                    ctx->left_touch_y,
                                    pTritonReport->unPressureLeft / 32768.0f);
        ctx->left_touch_down = left_touch_down;
    }
    if (right_touch_down || ctx->right_touch_down) {
        if (right_touch_down) {
            ctx->right_touch_x = pTritonReport->sRightPadX / 65536.0f + 0.5f;
            ctx->right_touch_y = -(float)pTritonReport->sRightPadY / 65536.0f + 0.5f;
        }
        SDL_SendJoystickTouchpad(timestamp, joystick, 1, 0,
                                    right_touch_down,
                                    ctx->right_touch_x,
                                    ctx->right_touch_y,
                                    pTritonReport->unPressureRight / 32768.0f);
        ctx->right_touch_down = right_touch_down;
    }

    if (ctx->report_sensors && pTritonReport->imu.timestamp != ctx->last_sensor_tick32) {
        Uint32 delta_us = (pTritonReport->imu.timestamp - ctx->last_sensor_tick32);

        ctx->sensor_timestamp_ns += SDL_US_TO_NS(delta_us);

        values[0] = (pTritonReport->imu.sGyroX / 32768.0f) * (2000.0f * (SDL_PI_F / 180.0f));
        values[1] = (pTritonReport->imu.sGyroZ / 32768.0f) * (2000.0f * (SDL_PI_F / 180.0f));
        values[2] = (-pTritonReport->imu.sGyroY / 32768.0f) * (2000.0f * (SDL_PI_F / 180.0f));
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, ctx->sensor_timestamp_ns, values, 3);

        values[0] = (pTritonReport->imu.sAccelX / 32768.0f) * 2.0f * SDL_STANDARD_GRAVITY;
        values[1] = (pTritonReport->imu.sAccelZ / 32768.0f) * 2.0f * SDL_STANDARD_GRAVITY;
        values[2] = (-pTritonReport->imu.sAccelY / 32768.0f) * 2.0f * SDL_STANDARD_GRAVITY;
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, ctx->sensor_timestamp_ns, values, 3);

        ctx->last_sensor_tick32 = pTritonReport->imu.timestamp;
    }
}

static void HIDAPI_DriverSteamTriton_HandleState_Timestamp(SDL_HIDAPI_Device *device,
                                                           SDL_Joystick *joystick,
                                                           TritonMTUNoQuat32TS_t *pTritonReport)
{
    float values[3];
    SDL_DriverSteamTriton_Context *ctx = (SDL_DriverSteamTriton_Context *)device->context;
    Uint64 timestamp = SDL_GetTicksNS();

    HIDAPI_DriverSteamTriton_HandleGenericState(ctx, joystick, timestamp, (TritonMTUNoQuat_t *) pTritonReport);

    bool left_touch_down = (pTritonReport->buttons & TRITON_LEFT_TOUCHPAD_TOUCH) ? true : false;
    bool right_touch_down = (pTritonReport->buttons & TRITON_RIGHT_TOUCHPAD_TOUCH) ? true : false;

    if (left_touch_down || ctx->left_touch_down) {
        if (left_touch_down) {
            ctx->left_touch_x = pTritonReport->sLeftPadX / 65536.0f + 0.5f;
            ctx->left_touch_y = -(float)pTritonReport->sLeftPadY / 65536.0f + 0.5f;
        }
        SDL_SendJoystickTouchpad(timestamp, joystick, 0, 0,
                                 left_touch_down,
                                 ctx->left_touch_x,
                                 ctx->left_touch_y,
                                 pTritonReport->unPressureLeft / 32768.0f);
        ctx->left_touch_down = left_touch_down;
    }
    if (right_touch_down || ctx->right_touch_down) {
        if (right_touch_down) {
            ctx->right_touch_x = pTritonReport->sRightPadX / 65536.0f + 0.5f;
            ctx->right_touch_y = -(float)pTritonReport->sRightPadY / 65536.0f + 0.5f;
        }
        SDL_SendJoystickTouchpad(timestamp, joystick, 1, 0,
                                 right_touch_down,
                                 ctx->right_touch_x,
                                 ctx->right_touch_y,
                                 pTritonReport->unPressureRight / 32768.0f);
        ctx->right_touch_down = right_touch_down;
    }

    if (ctx->report_sensors && pTritonReport->imu.timestamp != ctx->last_sensor_tick16) {
        // The timestamp is in units of 32 microseconds
        Uint32 delta_us = (Uint32)(pTritonReport->imu.timestamp - ctx->last_sensor_tick16) * 32;

        ctx->sensor_timestamp_ns += SDL_US_TO_NS(delta_us);

        values[0] = (pTritonReport->imu.sGyroX / 32768.0f) * (2000.0f * (SDL_PI_F / 180.0f));
        values[1] = (pTritonReport->imu.sGyroZ / 32768.0f) * (2000.0f * (SDL_PI_F / 180.0f));
        values[2] = (-pTritonReport->imu.sGyroY / 32768.0f) * (2000.0f * (SDL_PI_F / 180.0f));
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, ctx->sensor_timestamp_ns, values, 3);

        values[0] = (pTritonReport->imu.sAccelX / 32768.0f) * 2.0f * SDL_STANDARD_GRAVITY;
        values[1] = (pTritonReport->imu.sAccelZ / 32768.0f) * 2.0f * SDL_STANDARD_GRAVITY;
        values[2] = (-pTritonReport->imu.sAccelY / 32768.0f) * 2.0f * SDL_STANDARD_GRAVITY;
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, ctx->sensor_timestamp_ns, values, 3);

        ctx->last_sensor_tick16 = pTritonReport->imu.timestamp;
    }
}

static void HIDAPI_DriverSteamTriton_HandleBatteryStatus(SDL_HIDAPI_Device *device,
                                                         SDL_Joystick *joystick,
                                                         TritonBatteryStatus_t *pTritonBatteryStatus)
{
    SDL_PowerState state;

    switch (pTritonBatteryStatus->ucChargeState) {
    case k_EChargeStateDischarging:
        state = SDL_POWERSTATE_ON_BATTERY;
        break;
    case k_EChargeStateCharging:
        state = SDL_POWERSTATE_CHARGING;
        break;
    case k_EChargeStateChargingDone:
        state = SDL_POWERSTATE_CHARGED;
        break;
    default:
        // Error state?
        state = SDL_POWERSTATE_UNKNOWN;
        break;
    }
    SDL_SendJoystickPowerInfo(joystick, state, pTritonBatteryStatus->ucBatteryLevel);
}

static bool HIDAPI_DriverSteamTriton_SetControllerConnected(SDL_HIDAPI_Device *device, bool connected)
{
    SDL_DriverSteamTriton_Context *ctx = (SDL_DriverSteamTriton_Context *)device->context;

    if (ctx->connected != connected) {
        ctx->connected = connected;

        if (connected) {
            SDL_JoystickID joystickID;
            if (!HIDAPI_JoystickConnected(device, &joystickID)) {
                return false;
            }
        } else {
            if (device->num_joysticks > 0) {
                HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
            }
        }
    }
    return true;
}

static void HIDAPI_DriverSteamTriton_HandleWirelessStatus(SDL_HIDAPI_Device *device,
                                                          TritonWirelessStatus_t *pTritonWirelessStatus)
{
    switch (pTritonWirelessStatus->state) {
    case k_ETritonWirelessStateConnect:
        HIDAPI_DriverSteamTriton_SetControllerConnected(device, true);
        break;
    case k_ETritonWirelessStateDisconnect:
        HIDAPI_DriverSteamTriton_SetControllerConnected(device, false);
        break;
    default:
        break;
    }
}

/*****************************************************************************************************/

static void HIDAPI_DriverSteamTriton_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_STEAM, callback, userdata);
}

static void HIDAPI_DriverSteamTriton_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_STEAM, callback, userdata);
}

static bool HIDAPI_DriverSteamTriton_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_STEAM,
                              SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));
}

static bool HIDAPI_DriverSteamTriton_IsSupportedDevice(
    SDL_HIDAPI_Device *device,
    const char *name,
    SDL_GamepadType type,
    Uint16 vendor_id,
    Uint16 product_id,
    Uint16 version,
    int interface_number,
    int interface_class,
    int interface_subclass,
    int interface_protocol)
{

    if (IsProteusDongle(product_id)) {
        if (interface_number >= 2 && interface_number <= 5) {
            // The set of controller interfaces for Proteus & Nereid...currently
            return true;
        }
    } else if (SDL_IsJoystickSteamTriton(vendor_id, product_id)) {
        return true;
    }
    return false;
}

static bool HIDAPI_DriverSteamTriton_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverSteamTriton_Context *ctx;

    ctx = (SDL_DriverSteamTriton_Context *)SDL_calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return false;
    }

    device->context = ctx;

    HIDAPI_SetDeviceName(device, "Steam Controller");

    if (IsProteusDongle(device->product_id)) {
        return true;
    }

    // Wired controller, connected!
    return HIDAPI_DriverSteamTriton_SetControllerConnected(device, true);
}

static int HIDAPI_DriverSteamTriton_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverSteamTriton_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static bool HIDAPI_DriverSteamTriton_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble);

static bool HIDAPI_DriverSteamTriton_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverSteamTriton_Context *ctx = (SDL_DriverSteamTriton_Context *)device->context;
    SDL_Joystick *joystick = NULL;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    }

    if (ctx->connected && joystick) {
        Uint64 now = SDL_GetTicks();
        if (!ctx->last_lizard_update || (now - ctx->last_lizard_update) >= 3000) {
            DisableSteamTritonLizardMode(device->dev);
            ctx->last_lizard_update = now;
        }

        if (ctx->low_frequency_rumble || ctx->high_frequency_rumble) {
            if ((now - ctx->last_rumble_time) >= TRITON_RUMBLE_RESEND_INTERVAL_MS) {
                HIDAPI_DriverSteamTriton_RumbleJoystick(device, joystick, ctx->low_frequency_rumble, ctx->high_frequency_rumble);
            }
        }
    }

    for (;;) {
        uint8_t data[64];
        int r = SDL_hid_read(device->dev, data, sizeof(data));

        if (r == 0) {
            return true;
        }
        if (r < 0) {
            // Failed to read from controller
            HIDAPI_DriverSteamTriton_SetControllerConnected(device, false);
            return false;
        }

#ifdef DEBUG_STEAM_PROTOCOL
        HIDAPI_DumpPacket("Steam Controller packet: size = %d", data, r);
#endif

        switch (data[0]) {
        case ID_TRITON_CONTROLLER_STATE:
        case ID_TRITON_CONTROLLER_STATE_BLE:
            if (!joystick) {
                HIDAPI_DriverSteamTriton_SetControllerConnected(device, true);
                if (device->num_joysticks > 0) {
                    joystick = SDL_GetJoystickFromID(device->joysticks[0]);
                }
            }
            if (joystick && r >= (1 + sizeof(TritonMTUNoQuat_t))) {
                TritonMTUNoQuat_t *pTritonReport = (TritonMTUNoQuat_t *)&data[1];
                HIDAPI_DriverSteamTriton_HandleState(device, joystick, pTritonReport);
            }
            break;
        case ID_TRITON_CONTROLLER_STATE_TIMESTAMP:
            if (!joystick) {
                HIDAPI_DriverSteamTriton_SetControllerConnected(device, true);
                if (device->num_joysticks > 0) {
                    joystick = SDL_GetJoystickFromID(device->joysticks[0]);
                }
            }
            if (joystick && r >= (1 + sizeof(TritonMTUNoQuat32TS_t))) {
                TritonMTUNoQuat32TS_t *pTritonReport = (TritonMTUNoQuat32TS_t *)&data[1];
                HIDAPI_DriverSteamTriton_HandleState_Timestamp(device, joystick, pTritonReport);
            }
            break;
        case ID_TRITON_BATTERY_STATUS:
            if (joystick && r >= (1 + sizeof(TritonBatteryStatus_t))) {
                TritonBatteryStatus_t *pTritonBatteryStatus = (TritonBatteryStatus_t *)&data[1];
                HIDAPI_DriverSteamTriton_HandleBatteryStatus(device, joystick, pTritonBatteryStatus);
            }
            break;
        case ID_TRITON_WIRELESS_STATUS_X:
        case ID_TRITON_WIRELESS_STATUS:
            if (r >= (1 + sizeof(TritonWirelessStatus_t))) {
                TritonWirelessStatus_t *pTritonWirelessStatus = (TritonWirelessStatus_t *)&data[1];
                HIDAPI_DriverSteamTriton_HandleWirelessStatus(device, pTritonWirelessStatus);
            }
            break;
        default:
            break;
        }
    }
}

static bool HIDAPI_DriverSteamTriton_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    float update_rate_in_hz = 1000000.0f / TRITON_SENSOR_UPDATE_INTERVAL_US;

    SDL_AssertJoysticksLocked();

    // Initialize the joystick capabilities
    joystick->nbuttons = SDL_GAMEPAD_NUM_TRITON_BUTTONS;
    joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
    joystick->nhats = 1;

    SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_GYRO, update_rate_in_hz);
    SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, update_rate_in_hz);

    SDL_PrivateJoystickAddTouchpad(joystick, 1);
    SDL_PrivateJoystickAddTouchpad(joystick, 1);

    return true;
}

static bool HIDAPI_DriverSteamTriton_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    SDL_DriverSteamTriton_Context *ctx = (SDL_DriverSteamTriton_Context *)device->context;
    int rc;

    ctx->low_frequency_rumble = low_frequency_rumble;
    ctx->high_frequency_rumble = high_frequency_rumble;
    ctx->last_rumble_time = SDL_GetTicks();

    Uint8 buffer[HID_RUMBLE_OUTPUT_REPORT_BYTES] = { 0 };
    OutputReportMsg *msg = (OutputReportMsg *)(buffer);

    msg->report_id = ID_OUT_REPORT_HAPTIC_RUMBLE;
    msg->payload.hapticRumble.type = 0;
    msg->payload.hapticRumble.intensity = 0;
    msg->payload.hapticRumble.left.speed = low_frequency_rumble;
    msg->payload.hapticRumble.left.gain = 0;
    msg->payload.hapticRumble.right.speed = high_frequency_rumble;
    msg->payload.hapticRumble.right.gain = 0;

    rc = SDL_hid_write(device->dev, buffer, sizeof(buffer));
    if (rc < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_INPUT, 
            "Steam Controller HID Write FAILED! rc: %d. SDL_Error: %s", 
            rc, SDL_GetError());

        return false;
    }
    return true;
}

static bool HIDAPI_DriverSteamTriton_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverSteamTriton_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    return SDL_JOYSTICK_CAP_RUMBLE;
}

static bool HIDAPI_DriverSteamTriton_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverSteamTriton_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    if (size == HID_FEATURE_REPORT_BYTES) {
        int rc = SDL_hid_send_feature_report(device->dev, data, size);
        if (rc != size) {
            return false;
        }
        return true;
    }
    return SDL_Unsupported();
}

static bool HIDAPI_DriverSteamTriton_SetSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    SDL_DriverSteamTriton_Context *ctx = (SDL_DriverSteamTriton_Context *)device->context;
    int rc;
    Uint8 buffer[HID_FEATURE_REPORT_BYTES] = { 1 };
    FeatureReportMsg *msg = (FeatureReportMsg *)(buffer + 1);

    msg->header.type = ID_SET_SETTINGS_VALUES;
    msg->header.length = 1 * sizeof(ControllerSetting);
    msg->payload.setSettingsValues.settings[0].settingNum = SETTING_IMU_MODE;
    if (enabled) {
        msg->payload.setSettingsValues.settings[0].settingValue = (SETTING_GYRO_MODE_SEND_RAW_ACCEL | SETTING_GYRO_MODE_SEND_RAW_GYRO);
    } else {
        msg->payload.setSettingsValues.settings[0].settingValue = SETTING_GYRO_MODE_OFF;
    }

    rc = SDL_hid_send_feature_report(device->dev, buffer, sizeof(buffer));
    if (rc != sizeof(buffer)) {
        return false;
    }

    ctx->report_sensors = enabled;

    return true;
}

static void HIDAPI_DriverSteamTriton_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    // Lizard mode id automatically re-enabled by watchdog. Nothing to do here.
}

static void HIDAPI_DriverSteamTriton_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverSteamTriton = {
    SDL_HINT_JOYSTICK_HIDAPI_STEAM,
    true,
    HIDAPI_DriverSteamTriton_RegisterHints,
    HIDAPI_DriverSteamTriton_UnregisterHints,
    HIDAPI_DriverSteamTriton_IsEnabled,
    HIDAPI_DriverSteamTriton_IsSupportedDevice,
    HIDAPI_DriverSteamTriton_InitDevice,
    HIDAPI_DriverSteamTriton_GetDevicePlayerIndex,
    HIDAPI_DriverSteamTriton_SetDevicePlayerIndex,
    HIDAPI_DriverSteamTriton_UpdateDevice,
    HIDAPI_DriverSteamTriton_OpenJoystick,
    HIDAPI_DriverSteamTriton_RumbleJoystick,
    HIDAPI_DriverSteamTriton_RumbleJoystickTriggers,
    HIDAPI_DriverSteamTriton_GetJoystickCapabilities,
    HIDAPI_DriverSteamTriton_SetJoystickLED,
    HIDAPI_DriverSteamTriton_SendJoystickEffect,
    HIDAPI_DriverSteamTriton_SetSensorsEnabled,
    HIDAPI_DriverSteamTriton_CloseJoystick,
    HIDAPI_DriverSteamTriton_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_STEAM_TRITON

#endif // SDL_JOYSTICK_HIDAPI
