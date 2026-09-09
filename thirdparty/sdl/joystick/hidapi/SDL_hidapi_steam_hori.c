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
#include "../SDL_joystick_c.h"

#ifdef SDL_JOYSTICK_HIDAPI_STEAM_HORI

/* Define this if you want to log all packets from the controller */
/*#define DEBUG_HORI_PROTOCOL*/

#define LOAD16(A, B) (Sint16)((Uint16)(A) | (((Uint16)(B)) << 8))

enum
{
    SDL_GAMEPAD_BUTTON_HORI_QAM = 11,
    SDL_GAMEPAD_BUTTON_HORI_FR,
    SDL_GAMEPAD_BUTTON_HORI_FL,
    SDL_GAMEPAD_BUTTON_HORI_M1,
    SDL_GAMEPAD_BUTTON_HORI_M2,
    SDL_GAMEPAD_BUTTON_HORI_JOYSTICK_TOUCH_L,
    SDL_GAMEPAD_BUTTON_HORI_JOYSTICK_TOUCH_R,
    SDL_GAMEPAD_NUM_HORI_BUTTONS
};

typedef struct
{
    Uint8 last_state[USB_PACKET_LENGTH];
    Uint64 sensor_ticks;
    Uint32 last_tick;
    bool wireless;
    bool serial_needs_init;
} SDL_DriverSteamHori_Context;

static bool HIDAPI_DriverSteamHori_UpdateDevice(SDL_HIDAPI_Device *device);

static void HIDAPI_DriverSteamHori_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_STEAM_HORI, callback, userdata);
}

static void HIDAPI_DriverSteamHori_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_STEAM_HORI, callback, userdata);
}

static bool HIDAPI_DriverSteamHori_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_STEAM_HORI, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));
}

static bool HIDAPI_DriverSteamHori_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    return SDL_IsJoystickHoriSteamController(vendor_id, product_id);
}

static bool HIDAPI_DriverSteamHori_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverSteamHori_Context *ctx;

    ctx = (SDL_DriverSteamHori_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }

    device->context = ctx;
    ctx->serial_needs_init = true;

    HIDAPI_SetDeviceName(device, "Wireless HORIPAD For Steam");

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverSteamHori_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverSteamHori_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static bool HIDAPI_DriverSteamHori_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverSteamHori_Context *ctx = (SDL_DriverSteamHori_Context *)device->context;

    SDL_AssertJoysticksLocked();

    SDL_zeroa(ctx->last_state);

    /* Initialize the joystick capabilities */
    joystick->nbuttons = SDL_GAMEPAD_NUM_HORI_BUTTONS;
    joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
	joystick->nhats = 1;

	ctx->wireless = device->product_id == USB_PRODUCT_HORI_STEAM_CONTROLLER_BT;

	if (ctx->wireless && device->serial) {
		joystick->serial = SDL_strdup(device->serial);
		ctx->serial_needs_init = false;
	} else if (!ctx->wireless) {
		// Need to actual read from the device to init the serial
		HIDAPI_DriverSteamHori_UpdateDevice(device);
	}
       
    SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_GYRO, 250.0f);
    SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, 250.0f);

    return true;
}

static bool HIDAPI_DriverSteamHori_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    // Device doesn't support rumble
    return SDL_Unsupported();
}

static bool HIDAPI_DriverSteamHori_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverSteamHori_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    return 0;
}

static bool HIDAPI_DriverSteamHori_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverSteamHori_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverSteamHori_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    return true;
}

#undef clamp
#define clamp(val, min, max) (((val) > (max)) ? (max) : (((val) < (min)) ? (min) : (val)))

#ifndef DEG2RAD
#define DEG2RAD(x) ((float)(x) * (float)(SDL_PI_F / 180.f))
#endif

//---------------------------------------------------------------------------
// Scale and clamp values to a range
//---------------------------------------------------------------------------
static float RemapValClamped(float val, float A, float B, float C, float D)
{
    if (A == B) {
        return (val - B) >= 0.0f ? D : C;
    } else {
        float cVal = (val - A) / (B - A);
        cVal = clamp(cVal, 0.0f, 1.0f);

        return C + (D - C) * cVal;
    }
}

#define REPORT_HEADER_USB 0x07
#define REPORT_HEADER_BT  0x00

static void HIDAPI_DriverSteamHori_HandleStatePacket(SDL_Joystick *joystick, SDL_DriverSteamHori_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;
    Uint64 timestamp = SDL_GetTicksNS();

    // Make sure it's gamepad state and not OTA FW update info
    if (data[0] != REPORT_HEADER_USB && data[0] != REPORT_HEADER_BT) {
        /* We don't know how to handle this report */
        return;
    }

    #define READ_STICK_AXIS(offset) \
    (data[offset] == 0x80 ? 0 : (Sint16)HIDAPI_RemapVal((float)((int)data[offset] - 0x80), -0x80, 0xff - 0x80, SDL_MIN_SINT16, SDL_MAX_SINT16))
    {
        axis = READ_STICK_AXIS(1);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
        axis = READ_STICK_AXIS(2);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
        axis = READ_STICK_AXIS(3);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
        axis = READ_STICK_AXIS(4);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);
    }
#undef READ_STICK_AXIS

    if (ctx->last_state[5] != data[5]) {
        Uint8 hat;

        switch (data[5] & 0xF) {
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
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[5] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[5] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_HORI_QAM, ((data[5] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[5] & 0x80) != 0));

    }

    if (ctx->last_state[6] != data[6]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[6] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_HORI_M1 /* M1 */, ((data[6] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[6] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[6] & 0x08) != 0));

        // TODO: can we handle the digital trigger mode? The data seems to come through analog regardless of the trigger state
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[6] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[6] & 0x80) != 0));
    }

    if (ctx->last_state[7] != data[7]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[7] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[7] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[7] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_HORI_M2, ((data[7] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_HORI_JOYSTICK_TOUCH_L, ((data[7] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_HORI_JOYSTICK_TOUCH_R, ((data[7] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_HORI_FR, ((data[7] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_HORI_FL, ((data[7] & 0x80) != 0));
    }

	if (!ctx->wireless && ctx->serial_needs_init) {
        char serial[18];
		(void)SDL_snprintf(serial, sizeof(serial), "%.2x-%.2x-%.2x-%.2x-%.2x-%.2x",
								data[38], data[39], data[40], data[41], data[42], data[43]);

		joystick->serial = SDL_strdup(serial);
        ctx->serial_needs_init = false;
    }

#define READ_TRIGGER_AXIS(offset) \
    (Sint16)(((int)data[offset] * 257) - 32768)
    {
        axis = READ_TRIGGER_AXIS(8);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);
        axis = READ_TRIGGER_AXIS(9);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);
    }
#undef READ_TRIGGER_AXIS

    if (1) {
        Uint64 sensor_timestamp;
        float imu_data[3];

        /* 16-bit timestamp */
        Uint32 delta;
        Uint16 tick = LOAD16(data[10],
                             data[11]);
        if (ctx->last_tick < tick) {
            delta = (tick - ctx->last_tick);
        } else {
            delta = (SDL_MAX_UINT16 - ctx->last_tick + tick + 1);
        }

        ctx->last_tick = tick;
        ctx->sensor_ticks += delta;

        /* Sensor timestamp is in 1us units, but there seems to be some issues with the values reported from the device */
        sensor_timestamp = timestamp; // if the values were good we woudl call SDL_US_TO_NS(ctx->sensor_ticks);

        const float accelScale = SDL_STANDARD_GRAVITY * 8 / 32768.0f;
        const float gyroScale = DEG2RAD(2048);

        imu_data[1] = RemapValClamped(-1.0f * LOAD16(data[12], data[13]), INT16_MIN, INT16_MAX, -gyroScale, gyroScale);
        imu_data[2] = RemapValClamped(-1.0f * LOAD16(data[14], data[15]), INT16_MIN, INT16_MAX, -gyroScale, gyroScale);
        imu_data[0] = RemapValClamped(-1.0f * LOAD16(data[16], data[17]), INT16_MIN, INT16_MAX, -gyroScale, gyroScale);
		

        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, sensor_timestamp, imu_data, 3);

      //  SDL_Log("%u %f, %f, %f ", data[0], imu_data[0], imu_data[1], imu_data[2] );
        imu_data[2] = LOAD16(data[18], data[19]) * accelScale;
        imu_data[1] = -1 * LOAD16(data[20], data[21]) * accelScale;
        imu_data[0] = LOAD16(data[22], data[23]) * accelScale;
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, sensor_timestamp, imu_data, 3);
    }

    if (ctx->last_state[24] != data[24]) {
        bool bCharging = (data[24] & 0x10) != 0;
        int percent = (data[24] & 0xF) * 10;
		SDL_PowerState state;
         if (bCharging) {
            state = SDL_POWERSTATE_CHARGING;
        } else if (ctx->wireless) {
             state = SDL_POWERSTATE_ON_BATTERY;
        } else {
            state = SDL_POWERSTATE_CHARGED;
        }

        SDL_SendJoystickPowerInfo(joystick, state, percent);
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static bool HIDAPI_DriverSteamHori_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverSteamHori_Context *ctx = (SDL_DriverSteamHori_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size = 0;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    } else {
        return false;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_HORI_PROTOCOL
        HIDAPI_DumpPacket("Google Hori packet: size = %d", data, size);
#endif
        if (!joystick) {
            continue;
        }

        HIDAPI_DriverSteamHori_HandleStatePacket(joystick, ctx, data, size);
    }

    if (size < 0) {
        /* Read error, device is disconnected */
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return (size >= 0);
}

static void HIDAPI_DriverSteamHori_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
}

static void HIDAPI_DriverSteamHori_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverSteamHori = {
    SDL_HINT_JOYSTICK_HIDAPI_STEAM_HORI,
    true,
    HIDAPI_DriverSteamHori_RegisterHints,
    HIDAPI_DriverSteamHori_UnregisterHints,
    HIDAPI_DriverSteamHori_IsEnabled,
    HIDAPI_DriverSteamHori_IsSupportedDevice,
    HIDAPI_DriverSteamHori_InitDevice,
    HIDAPI_DriverSteamHori_GetDevicePlayerIndex,
    HIDAPI_DriverSteamHori_SetDevicePlayerIndex,
    HIDAPI_DriverSteamHori_UpdateDevice,
    HIDAPI_DriverSteamHori_OpenJoystick,
    HIDAPI_DriverSteamHori_RumbleJoystick,
    HIDAPI_DriverSteamHori_RumbleJoystickTriggers,
    HIDAPI_DriverSteamHori_GetJoystickCapabilities,
    HIDAPI_DriverSteamHori_SetJoystickLED,
    HIDAPI_DriverSteamHori_SendJoystickEffect,
    HIDAPI_DriverSteamHori_SetJoystickSensorsEnabled,
    HIDAPI_DriverSteamHori_CloseJoystick,
    HIDAPI_DriverSteamHori_FreeDevice,
};

#endif /* SDL_JOYSTICK_HIDAPI_STEAM_HORI */

#endif /* SDL_JOYSTICK_HIDAPI */
