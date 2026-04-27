/*
  Simple DirectMedia Layer
  Copyright (C) 2023 Max Maisel <max.maisel@posteo.de>

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

/*****************************************************************************************************/

#include "steam/controller_constants.h"
#include "steam/controller_structs.h"

// Always 1kHz according to USB descriptor, but actually about 4 ms.
#define TRITON_SENSOR_UPDATE_INTERVAL_US 4032

enum
{
    SDL_GAMEPAD_BUTTON_STEAM_DECK_QAM = 11,
    SDL_GAMEPAD_BUTTON_STEAM_DECK_RIGHT_PADDLE1,
    SDL_GAMEPAD_BUTTON_STEAM_DECK_LEFT_PADDLE1,
    SDL_GAMEPAD_BUTTON_STEAM_DECK_RIGHT_PADDLE2,
    SDL_GAMEPAD_BUTTON_STEAM_DECK_LEFT_PADDLE2,
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

    /*
	STEAM_RIGHTSTICK_FINGERDOWN_MASK,   // Right Stick Touch    0x00100000
	STEAM_RIGHTPAD_FINGERDOWN_MASK,     // Right Pad Touch      0x00200000
	STEAM_BUTTON_RIGHTPAD_CLICKED_MASK, // Right Pressure Click 0x00400000
	STEAM_RIGHT_TRIGGER_MASK,           // Right Trigger Click  0x00800000

	STEAM_LEFTSTICK_FINGERDOWN_MASK,    // Left Stick Touch     0x01000000
	STEAM_LEFTPAD_FINGERDOWN_MASK,      // Left Pad Touch       0x02000000
	STEAM_BUTTON_LEFTPAD_CLICKED_MASK,  // Left Pressure Click  0x04000000
	STEAM_LEFT_TRIGGER_MASK,            // Left Trigger Click   0x08000000
    STEAM_RIGHT_AUX_MASK,               // Right Pinky Touch   0x10000000
	STEAM_LEFT_AUX_MASK,                // Left Pinky Touch    0x20000000 
    */
} TritonButtons;

typedef struct
{
    bool connected;
    bool report_sensors;
    Uint32 last_sensor_tick;
    Uint64 sensor_timestamp_ns;
    Uint64 last_button_state;
    Uint64 last_lizard_update;
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

static void HIDAPI_DriverSteamTriton_HandleState(SDL_HIDAPI_Device *device,
                                               SDL_Joystick *joystick,
                                               TritonMTUFull_t *pTritonReport)
{
    float values[3];
    SDL_DriverSteamTriton_Context *ctx = (SDL_DriverSteamTriton_Context *)device->context;
    Uint64 timestamp = SDL_GetTicksNS();

    if (pTritonReport->uButtons != ctx->last_button_state) {
        Uint8 hat = 0;

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_A) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_B) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_X) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_Y) != 0));

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_L) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_R) != 0));

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_MENU) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_VIEW) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_STEAM) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_STEAM_DECK_QAM,
                               ((pTritonReport->uButtons & TRITON_HBUTTON_QAM) != 0));

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_L3) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_R3) != 0));

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_STEAM_DECK_RIGHT_PADDLE1,
                               ((pTritonReport->uButtons & TRITON_HBUTTON_R4) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_STEAM_DECK_LEFT_PADDLE1,
                               ((pTritonReport->uButtons & TRITON_HBUTTON_L4) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_STEAM_DECK_RIGHT_PADDLE2,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_R5) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_STEAM_DECK_LEFT_PADDLE2,
                               ((pTritonReport->uButtons & TRITON_LBUTTON_L5) != 0));

        if (pTritonReport->uButtons & TRITON_LBUTTON_DPAD_UP) {
            hat |= SDL_HAT_UP;
        }
        if (pTritonReport->uButtons & TRITON_LBUTTON_DPAD_DOWN) {
            hat |= SDL_HAT_DOWN;
        }
        if (pTritonReport->uButtons & TRITON_LBUTTON_DPAD_LEFT) {
            hat |= SDL_HAT_LEFT;
        }
        if (pTritonReport->uButtons & TRITON_LBUTTON_DPAD_RIGHT) {
            hat |= SDL_HAT_RIGHT;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);

        ctx->last_button_state = pTritonReport->uButtons;
    }

    // RKRK There're button bits for this if you so choose.
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

    if (ctx->report_sensors && pTritonReport->imu.uTimestamp != ctx->last_sensor_tick) {
        Uint32 delta_us = (pTritonReport->imu.uTimestamp - ctx->last_sensor_tick);

        ctx->sensor_timestamp_ns += SDL_US_TO_NS(delta_us);

        values[0] = (pTritonReport->imu.sGyroX / 32768.0f) * (2000.0f * (SDL_PI_F / 180.0f));
        values[1] = (pTritonReport->imu.sGyroZ / 32768.0f) * (2000.0f * (SDL_PI_F / 180.0f));
        values[2] = (-pTritonReport->imu.sGyroY / 32768.0f) * (2000.0f * (SDL_PI_F / 180.0f));
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, ctx->sensor_timestamp_ns, values, 3);

        values[0] = (pTritonReport->imu.sAccelX / 32768.0f) * 2.0f * SDL_STANDARD_GRAVITY;
        values[1] = (pTritonReport->imu.sAccelZ / 32768.0f) * 2.0f * SDL_STANDARD_GRAVITY;
        values[2] = (-pTritonReport->imu.sAccelY / 32768.0f) * 2.0f * SDL_STANDARD_GRAVITY;
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, ctx->sensor_timestamp_ns, values, 3);

        ctx->last_sensor_tick = pTritonReport->imu.uTimestamp;
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

        switch (data[0]) {
        case ID_TRITON_CONTROLLER_STATE:
        case ID_TRITON_CONTROLLER_STATE_BLE:
            if (!joystick) {
                HIDAPI_DriverSteamTriton_SetControllerConnected(device, true);
                if (device->num_joysticks > 0) {
                    joystick = SDL_GetJoystickFromID(device->joysticks[0]);
                }
            }
            if (joystick && r >= (1 + sizeof(TritonMTUFull_t))) {
                TritonMTUFull_t *pTritonReport = (TritonMTUFull_t *)&data[1];
                HIDAPI_DriverSteamTriton_HandleState(device, joystick, pTritonReport);
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

    return true;
}

static bool HIDAPI_DriverSteamTriton_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    int rc;

    //RKRK Not sure about size. Probably 64+1 is OK for ORs
    Uint8 buffer[HID_RUMBLE_OUTPUT_REPORT_BYTES];
    OutputReportMsg *msg = (OutputReportMsg *)(buffer);

	msg->report_id = ID_OUT_REPORT_HAPTIC_RUMBLE;
    msg->payload.hapticRumble.type = 0;
    msg->payload.hapticRumble.intensity = 0;
    msg->payload.hapticRumble.left.speed = low_frequency_rumble;
    msg->payload.hapticRumble.left.gain = 0;
    msg->payload.hapticRumble.right.speed = high_frequency_rumble;
    msg->payload.hapticRumble.right.gain = 0;


    rc = SDL_hid_write(device->dev, buffer, sizeof(buffer));
    if (rc != sizeof(buffer)) {
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
