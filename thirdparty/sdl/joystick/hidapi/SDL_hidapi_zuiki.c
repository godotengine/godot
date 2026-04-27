/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>
  Copyright (C) 2025 Zuiki Inc.

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

#ifdef SDL_JOYSTICK_HIDAPI_ZUIKI

#define GYRO_SCALE   (1024.0f / 32768.0f * SDL_PI_F / 180.0f) // Calculate scaling factor based on gyroscope data range and radians
#define ACCEL_SCALE  (8.0f / 32768.0f * SDL_STANDARD_GRAVITY) // Calculate acceleration scaling factor based on gyroscope data range and standard gravity
#define FILTER_SIZE 11  // Must be an odd number
#define MAX_RETRY_COUNT 10 // zuiki device initialization retry count

// Define this if you want to log all packets from the controller
#if 0
#define DEBUG_ZUIKI_PROTOCOL
#endif

typedef struct {
    float buffer[FILTER_SIZE];
    uint8_t index;
    uint8_t count;
} MedianFilter_t;

typedef struct
{
    Uint8 last_state[USB_PACKET_LENGTH];
    bool sensors_supported;     // Sensor enabled status flag
    Uint64 sensor_timestamp_ns; // Sensor timestamp (nanoseconds, cumulative update)
    float sensor_rate;
    MedianFilter_t filter_gyro_x;
    MedianFilter_t filter_gyro_y;
    MedianFilter_t filter_gyro_z;
} SDL_DriverZUIKI_Context;

static float median_filter_update(MedianFilter_t* mf, float input) {
    mf->buffer[mf->index] = input;
    mf->index = (mf->index + 1) % FILTER_SIZE;
    if (mf->count < FILTER_SIZE) mf->count++;
    float temp[FILTER_SIZE];
    SDL_memcpy(temp, mf->buffer, sizeof(temp));
    for (int i = 0; i < mf->count - 1; i++) {
        for (int j = i + 1; j < mf->count; j++) {
            if (temp[i] > temp[j]) {
                float t = temp[i];
                temp[i] = temp[j];
                temp[j] = t;
            }
        }
    }
    return temp[mf->count / 2];
}


static void HIDAPI_DriverZUIKI_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_ZUIKI, callback, userdata);
}

static void HIDAPI_DriverZUIKI_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_ZUIKI, callback, userdata);
}

static bool HIDAPI_DriverZUIKI_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_ZUIKI, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));
}

static bool HIDAPI_DriverZUIKI_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    if (vendor_id == USB_VENDOR_ZUIKI) {
        switch (product_id) {
        case USB_PRODUCT_ZUIKI_MASCON_PRO:
        case USB_PRODUCT_ZUIKI_EVOTOP_UWB_DINPUT:
        case USB_PRODUCT_ZUIKI_EVOTOP_PC_DINPUT:
        case USB_PRODUCT_ZUIKI_EVOTOP_PC_BT:
            return true;
        default:
            break;
        }
    }
    return false;
}

static bool HIDAPI_DriverZUIKI_InitDevice(SDL_HIDAPI_Device *device)
{
    Uint8 data[USB_PACKET_LENGTH * 2];
    SDL_DriverZUIKI_Context *ctx = (SDL_DriverZUIKI_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    device->context = ctx;
    ctx->sensors_supported = false;

    // Read report data once for device initialization
    int size = -1;
    Uint8 retry_count = 0;
    while (retry_count < MAX_RETRY_COUNT) {
        size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 10);
        if (size > 0) {
            break;
        }
        retry_count++;
    }
    if (size <= 0) {
        return false;
    }

    switch (device->product_id) {
        case USB_PRODUCT_ZUIKI_MASCON_PRO:
            HIDAPI_SetDeviceName(device, "ZUIKI MASCON PRO");
            break;
        case USB_PRODUCT_ZUIKI_EVOTOP_PC_DINPUT:
            ctx->sensors_supported = true;
            ctx->sensor_rate = 200.0f;
            break;
        case USB_PRODUCT_ZUIKI_EVOTOP_UWB_DINPUT:
            ctx->sensors_supported = true;
            ctx->sensor_rate = 100.0f;
            break;
        case USB_PRODUCT_ZUIKI_EVOTOP_PC_BT:
            if (size > 0 && data[16] != 0) {
                ctx->sensors_supported = true;
                ctx->sensor_rate = 50.0f;
            }
            HIDAPI_SetDeviceName(device, "ZUIKI EVOTOP");
            break;
        default:
            break;
    }

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverZUIKI_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverZUIKI_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

#ifndef DEG2RAD
#define DEG2RAD(x) ((float)(x) * (float)(SDL_PI_F / 180.f))
#endif

static bool HIDAPI_DriverZUIKI_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverZUIKI_Context *ctx = (SDL_DriverZUIKI_Context *)device->context;

    SDL_AssertJoysticksLocked();

    SDL_zeroa(ctx->last_state);

    joystick->nbuttons = 11;
    joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
    joystick->nhats = 1;
    if (ctx->sensors_supported) {
        SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_GYRO, ctx->sensor_rate);
        SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, ctx->sensor_rate);
    }

    return true;
}

static bool HIDAPI_DriverZUIKI_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    Uint8 rumble_packet[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
    rumble_packet[4] = low_frequency_rumble >> 8;
    rumble_packet[5] = high_frequency_rumble >> 8;
    if (SDL_HIDAPI_SendRumble(device, rumble_packet, sizeof(rumble_packet)) != sizeof(rumble_packet)) {
        return SDL_SetError("Couldn't send rumble packet");
    }
    return true;
}

static bool HIDAPI_DriverZUIKI_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverZUIKI_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    Uint32 caps = 0;
    caps |= SDL_JOYSTICK_CAP_RUMBLE;
    return caps;
}

static bool HIDAPI_DriverZUIKI_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverZUIKI_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    if (SDL_HIDAPI_SendRumble(device, data, size) != size) {
        return SDL_SetError("Couldn't send rumble packet");
    }
    return true;
}

static bool HIDAPI_DriverZUIKI_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    SDL_DriverZUIKI_Context *ctx = (SDL_DriverZUIKI_Context *)device->context;
    if (ctx->sensors_supported) {
        return true;
    }
    return SDL_Unsupported();
}

static void HIDAPI_DriverZUIKI_HandleOldStatePacket(SDL_Joystick *joystick, SDL_DriverZUIKI_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;
    Uint64 timestamp = SDL_GetTicksNS();

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

    if (ctx->last_state[0] != data[0]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[0] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[0] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[0] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[0] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[0] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[0] & 0x20) != 0));
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, (data[0] & 0x40) ? SDL_MAX_SINT16 : SDL_MIN_SINT16);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, (data[0] & 0x80) ? SDL_MAX_SINT16 : SDL_MIN_SINT16);
    }

    if (ctx->last_state[1] != data[1]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[1] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[1] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[1] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[1] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[1] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_MISC1, ((data[1] & 0x20) != 0));
        /* todo for switch C key */
    }

#define READ_STICK_AXIS(offset) \
    (data[offset] == 0x7f ? 0 : (Sint16)HIDAPI_RemapVal((float)((int)data[offset] - 0x7f), -0x7f, 0xff - 0x7f, SDL_MIN_SINT16, SDL_MAX_SINT16))
    {
        axis = READ_STICK_AXIS(3);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
        axis = READ_STICK_AXIS(4);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
        axis = READ_STICK_AXIS(5);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
        axis = READ_STICK_AXIS(6);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);
    }
#undef READ_STICK_AXIS

    if (ctx->sensors_supported) {
        Uint64 sensor_timestamp = timestamp;
        float gyro_values[3];
        gyro_values[0] = median_filter_update(&ctx->filter_gyro_x, LOAD16(data[8], data[9]) * GYRO_SCALE);
        gyro_values[1] = median_filter_update(&ctx->filter_gyro_y, LOAD16(data[12], data[13]) * GYRO_SCALE);
        gyro_values[2] = median_filter_update(&ctx->filter_gyro_z, -LOAD16(data[10], data[11]) * GYRO_SCALE);
        float accel_values[3];
        accel_values[0] = LOAD16(data[14], data[15]) * ACCEL_SCALE;
        accel_values[2] = -LOAD16(data[16], data[17]) * ACCEL_SCALE;
        accel_values[1] = LOAD16(data[18], data[19]) * ACCEL_SCALE;
#ifdef DEBUG_ZUIKI_PROTOCOL
        SDL_Log("Gyro raw: %d, %d, %d -> scaled: %.2f, %.2f, %.2f rad/s",
                LOAD16(data[8], data[9]), LOAD16(data[10], data[11]), LOAD16(data[12], data[13]),
                gyro_values[0], gyro_values[1], gyro_values[2]);
        SDL_Log("Accel raw: %d, %d, %d -> scaled: %.2f, %.2f, %.2f m/s²",
                LOAD16(data[14], data[15]), LOAD16(data[16], data[17]), LOAD16(data[18], data[19]),
                accel_values[0], accel_values[1], accel_values[2]);
#endif

        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, sensor_timestamp, gyro_values, 3);
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, sensor_timestamp, accel_values, 3);
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static void HIDAPI_DriverZUIKI_Handle_EVOTOP_PCBT_StatePacket(SDL_Joystick *joystick, SDL_DriverZUIKI_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;
    Uint64 timestamp = SDL_GetTicksNS();

    axis = (Sint16)HIDAPI_RemapVal((float)(data[2] << 8 | data[1]), 0x0000, 0xffff, SDL_MIN_SINT16, SDL_MAX_SINT16);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
    axis = (Sint16)HIDAPI_RemapVal((float)(data[4] << 8 | data[3]), 0x0000, 0xffff, SDL_MIN_SINT16, SDL_MAX_SINT16);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
    axis = (Sint16)HIDAPI_RemapVal((float)(data[6] << 8 | data[5]), 0x0000, 0xffff, SDL_MIN_SINT16, SDL_MAX_SINT16);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
    axis = (Sint16)HIDAPI_RemapVal((float)(data[8] << 8 | data[7]), 0x0000, 0xffff, SDL_MIN_SINT16, SDL_MAX_SINT16);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);

    axis = (Sint16)HIDAPI_RemapVal((float)(data[10] << 8 | data[9]), 0x0000, 0x03ff, SDL_MIN_SINT16, SDL_MAX_SINT16);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);
    axis = (Sint16)HIDAPI_RemapVal((float)(data[12] << 8 | data[11]), 0x0000, 0x03ff, SDL_MIN_SINT16, SDL_MAX_SINT16);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);

    if (ctx->last_state[13] != data[13]) {
        Uint8 hat;
        switch (data[13]) {
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
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[15] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[15] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[15] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[15] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[15] & 0x40) != 0));
    }

    if (ctx->sensors_supported) {
        Uint64 sensor_timestamp = timestamp;
        float gyro_values[3];
        gyro_values[0] = median_filter_update(&ctx->filter_gyro_x, LOAD16(data[17], data[18]) * GYRO_SCALE);
        gyro_values[1] = median_filter_update(&ctx->filter_gyro_y, LOAD16(data[21], data[22]) * GYRO_SCALE);
        gyro_values[2] = median_filter_update(&ctx->filter_gyro_z, -LOAD16(data[19], data[20]) * GYRO_SCALE);
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, sensor_timestamp, gyro_values, 3);
        float accel_values[3];
        accel_values[0] = LOAD16(data[23], data[24]) * ACCEL_SCALE;
        accel_values[2] = -LOAD16(data[25], data[26]) * ACCEL_SCALE;
        accel_values[1] = LOAD16(data[27], data[28]) * ACCEL_SCALE;
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, sensor_timestamp, accel_values, 3);
#ifdef DEBUG_ZUIKI_PROTOCOL
        SDL_Log("Gyro raw: %d, %d, %d -> scaled: %.2f, %.2f, %.2f rad/s",
                LOAD16(data[17], data[18]), LOAD16(data[19], data[20]), LOAD16(data[21], data[22]),
                gyro_values[0], gyro_values[1], gyro_values[2]);
        SDL_Log("Accel raw: %d, %d, %d -> scaled: %.2f, %.2f, %.2f m/s²",
                LOAD16(data[23], data[24]), LOAD16(data[25], data[26]), LOAD16(data[27], data[28]),
                accel_values[0], accel_values[1], accel_values[2]);
#endif
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static bool HIDAPI_DriverZUIKI_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverZUIKI_Context *ctx = (SDL_DriverZUIKI_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size = 0;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    } else {
        return false;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_ZUIKI_PROTOCOL
        HIDAPI_DumpPacket("ZUIKI packet: size = %d", data, size);
#endif
        if (!joystick) {
            continue;
        }

        if (device->product_id == USB_PRODUCT_ZUIKI_EVOTOP_PC_BT) {
            HIDAPI_DriverZUIKI_Handle_EVOTOP_PCBT_StatePacket(joystick, ctx, data, size);
        } else if (device->product_id == USB_PRODUCT_ZUIKI_EVOTOP_PC_DINPUT
            || device->product_id == USB_PRODUCT_ZUIKI_MASCON_PRO
            || device->product_id == USB_PRODUCT_ZUIKI_EVOTOP_UWB_DINPUT) {
            HIDAPI_DriverZUIKI_HandleOldStatePacket(joystick, ctx, data, size);
        }
    }

    if (size < 0) {
        // Read error, device is disconnected
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return (size >= 0);
}

static void HIDAPI_DriverZUIKI_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
}

static void HIDAPI_DriverZUIKI_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverZUIKI = {
    SDL_HINT_JOYSTICK_HIDAPI_ZUIKI,
    true,
    HIDAPI_DriverZUIKI_RegisterHints,
    HIDAPI_DriverZUIKI_UnregisterHints,
    HIDAPI_DriverZUIKI_IsEnabled,
    HIDAPI_DriverZUIKI_IsSupportedDevice,
    HIDAPI_DriverZUIKI_InitDevice,
    HIDAPI_DriverZUIKI_GetDevicePlayerIndex,
    HIDAPI_DriverZUIKI_SetDevicePlayerIndex,
    HIDAPI_DriverZUIKI_UpdateDevice,
    HIDAPI_DriverZUIKI_OpenJoystick,
    HIDAPI_DriverZUIKI_RumbleJoystick,
    HIDAPI_DriverZUIKI_RumbleJoystickTriggers,
    HIDAPI_DriverZUIKI_GetJoystickCapabilities,
    HIDAPI_DriverZUIKI_SetJoystickLED,
    HIDAPI_DriverZUIKI_SendJoystickEffect,
    HIDAPI_DriverZUIKI_SetJoystickSensorsEnabled,
    HIDAPI_DriverZUIKI_CloseJoystick,
    HIDAPI_DriverZUIKI_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_ZUIKI

#endif // SDL_JOYSTICK_HIDAPI
