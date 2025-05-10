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

#ifdef SDL_SENSOR_N3DS

#include <3ds.h>

#include "../SDL_syssensor.h"

// 1 accelerometer and 1 gyroscope
#define N3DS_SENSOR_COUNT 2

typedef struct
{
    SDL_SensorType type;
    SDL_SensorID instance_id;
} SDL_N3DSSensor;

static SDL_N3DSSensor N3DS_sensors[N3DS_SENSOR_COUNT];

static bool InitN3DSServices(void);
static void UpdateN3DSAccelerometer(SDL_Sensor *sensor);
static void UpdateN3DSGyroscope(SDL_Sensor *sensor);

static bool IsDeviceIndexValid(int device_index)
{
    return device_index >= 0 && device_index < N3DS_SENSOR_COUNT;
}

static bool N3DS_SensorInit(void)
{
    if (!InitN3DSServices()) {
        return SDL_SetError("Failed to initialise N3DS services");
    }

    N3DS_sensors[0].type = SDL_SENSOR_ACCEL;
    N3DS_sensors[0].instance_id = SDL_GetNextObjectID();
    N3DS_sensors[1].type = SDL_SENSOR_GYRO;
    N3DS_sensors[1].instance_id = SDL_GetNextObjectID();
    return true;
}

static bool InitN3DSServices(void)
{
    if (R_FAILED(hidInit())) {
        return false;
    }

    if (R_FAILED(HIDUSER_EnableAccelerometer())) {
        return false;
    }

    if (R_FAILED(HIDUSER_EnableGyroscope())) {
        return false;
    }
    return true;
}

static int N3DS_SensorGetCount(void)
{
    return N3DS_SENSOR_COUNT;
}

static void N3DS_SensorDetect(void)
{
}

static const char *N3DS_SensorGetDeviceName(int device_index)
{
    if (IsDeviceIndexValid(device_index)) {
        switch (N3DS_sensors[device_index].type) {
        case SDL_SENSOR_ACCEL:
            return "Accelerometer";
        case SDL_SENSOR_GYRO:
            return "Gyroscope";
        default:
            return "Unknown";
        }
    }

    return NULL;
}

static SDL_SensorType N3DS_SensorGetDeviceType(int device_index)
{
    if (IsDeviceIndexValid(device_index)) {
        return N3DS_sensors[device_index].type;
    }
    return SDL_SENSOR_INVALID;
}

static int N3DS_SensorGetDeviceNonPortableType(int device_index)
{
    return (int)N3DS_SensorGetDeviceType(device_index);
}

static SDL_SensorID N3DS_SensorGetDeviceInstanceID(int device_index)
{
    if (IsDeviceIndexValid(device_index)) {
        return N3DS_sensors[device_index].instance_id;
    }
    return -1;
}

static bool N3DS_SensorOpen(SDL_Sensor *sensor, int device_index)
{
    return true;
}

static void N3DS_SensorUpdate(SDL_Sensor *sensor)
{
    switch (sensor->type) {
    case SDL_SENSOR_ACCEL:
        UpdateN3DSAccelerometer(sensor);
        break;
    case SDL_SENSOR_GYRO:
        UpdateN3DSGyroscope(sensor);
        break;
    default:
        break;
    }
}

static void UpdateN3DSAccelerometer(SDL_Sensor *sensor)
{
    static accelVector previous_state = { 0, 0, 0 };
    accelVector current_state;
    float data[3];
    Uint64 timestamp = SDL_GetTicksNS();

    hidAccelRead(&current_state);
    if (SDL_memcmp(&previous_state, &current_state, sizeof(accelVector)) != 0) {
        SDL_memcpy(&previous_state, &current_state, sizeof(accelVector));
        data[0] = (float)current_state.x * SDL_STANDARD_GRAVITY;
        data[1] = (float)current_state.y * SDL_STANDARD_GRAVITY;
        data[2] = (float)current_state.z * SDL_STANDARD_GRAVITY;
        SDL_SendSensorUpdate(timestamp, sensor, timestamp, data, sizeof(data));
    }
}

static void UpdateN3DSGyroscope(SDL_Sensor *sensor)
{
    static angularRate previous_state = { 0, 0, 0 };
    angularRate current_state;
    float data[3];
    Uint64 timestamp = SDL_GetTicksNS();

    hidGyroRead(&current_state);
    if (SDL_memcmp(&previous_state, &current_state, sizeof(angularRate)) != 0) {
        SDL_memcpy(&previous_state, &current_state, sizeof(angularRate));
        data[0] = (float)current_state.x;
        data[1] = (float)current_state.y;
        data[2] = (float)current_state.z;
        SDL_SendSensorUpdate(timestamp, sensor, timestamp, data, sizeof(data));
    }
}

static void N3DS_SensorClose(SDL_Sensor *sensor)
{
}

static void N3DS_SensorQuit(void)
{
    HIDUSER_DisableGyroscope();
    HIDUSER_DisableAccelerometer();
    hidExit();
}

SDL_SensorDriver SDL_N3DS_SensorDriver = {
    .Init = N3DS_SensorInit,
    .GetCount = N3DS_SensorGetCount,
    .Detect = N3DS_SensorDetect,
    .GetDeviceName = N3DS_SensorGetDeviceName,
    .GetDeviceType = N3DS_SensorGetDeviceType,
    .GetDeviceNonPortableType = N3DS_SensorGetDeviceNonPortableType,
    .GetDeviceInstanceID = N3DS_SensorGetDeviceInstanceID,
    .Open = N3DS_SensorOpen,
    .Update = N3DS_SensorUpdate,
    .Close = N3DS_SensorClose,
    .Quit = N3DS_SensorQuit,
};

#endif // SDL_SENSOR_N3DS
