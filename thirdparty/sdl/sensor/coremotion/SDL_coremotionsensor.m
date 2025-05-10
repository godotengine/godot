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

#ifdef SDL_SENSOR_COREMOTION

// This is the system specific header for the SDL sensor API
#include <CoreMotion/CoreMotion.h>

#include "SDL_coremotionsensor.h"
#include "../SDL_syssensor.h"
#include "../SDL_sensor_c.h"

typedef struct
{
    SDL_SensorType type;
    SDL_SensorID instance_id;
} SDL_CoreMotionSensor;

static CMMotionManager *SDL_motion_manager;
static SDL_CoreMotionSensor *SDL_sensors;
static int SDL_sensors_count;

static bool SDL_COREMOTION_SensorInit(void)
{
    int i, sensors_count = 0;

    if (!SDL_motion_manager) {
        SDL_motion_manager = [[CMMotionManager alloc] init];
    }

    if (SDL_motion_manager.accelerometerAvailable) {
        ++sensors_count;
    }
    if (SDL_motion_manager.gyroAvailable) {
        ++sensors_count;
    }

    if (sensors_count > 0) {
        SDL_sensors = (SDL_CoreMotionSensor *)SDL_calloc(sensors_count, sizeof(*SDL_sensors));
        if (!SDL_sensors) {
            return false;
        }

        i = 0;
        if (SDL_motion_manager.accelerometerAvailable) {
            SDL_sensors[i].type = SDL_SENSOR_ACCEL;
            SDL_sensors[i].instance_id = SDL_GetNextObjectID();
            ++i;
        }
        if (SDL_motion_manager.gyroAvailable) {
            SDL_sensors[i].type = SDL_SENSOR_GYRO;
            SDL_sensors[i].instance_id = SDL_GetNextObjectID();
            ++i;
        }
        SDL_sensors_count = sensors_count;
    }
    return true;
}

static int SDL_COREMOTION_SensorGetCount(void)
{
    return SDL_sensors_count;
}

static void SDL_COREMOTION_SensorDetect(void)
{
}

static const char *SDL_COREMOTION_SensorGetDeviceName(int device_index)
{
    switch (SDL_sensors[device_index].type) {
    case SDL_SENSOR_ACCEL:
        return "Accelerometer";
    case SDL_SENSOR_GYRO:
        return "Gyro";
    default:
        return "Unknown";
    }
}

static SDL_SensorType SDL_COREMOTION_SensorGetDeviceType(int device_index)
{
    return SDL_sensors[device_index].type;
}

static int SDL_COREMOTION_SensorGetDeviceNonPortableType(int device_index)
{
    return SDL_sensors[device_index].type;
}

static SDL_SensorID SDL_COREMOTION_SensorGetDeviceInstanceID(int device_index)
{
    return SDL_sensors[device_index].instance_id;
}

static bool SDL_COREMOTION_SensorOpen(SDL_Sensor *sensor, int device_index)
{
    struct sensor_hwdata *hwdata;

    hwdata = (struct sensor_hwdata *)SDL_calloc(1, sizeof(*hwdata));
    if (hwdata == NULL) {
        return false;
    }
    sensor->hwdata = hwdata;

    switch (sensor->type) {
    case SDL_SENSOR_ACCEL:
        [SDL_motion_manager startAccelerometerUpdates];
        break;
    case SDL_SENSOR_GYRO:
        [SDL_motion_manager startGyroUpdates];
        break;
    default:
        break;
    }
    return true;
}

static void SDL_COREMOTION_SensorUpdate(SDL_Sensor *sensor)
{
    Uint64 timestamp = SDL_GetTicksNS();

    switch (sensor->type) {
    case SDL_SENSOR_ACCEL:
    {
        CMAccelerometerData *accelerometerData = SDL_motion_manager.accelerometerData;
        if (accelerometerData) {
            CMAcceleration acceleration = accelerometerData.acceleration;
            float data[3];
            data[0] = -acceleration.x * SDL_STANDARD_GRAVITY;
            data[1] = -acceleration.y * SDL_STANDARD_GRAVITY;
            data[2] = -acceleration.z * SDL_STANDARD_GRAVITY;
            if (SDL_memcmp(data, sensor->hwdata->data, sizeof(data)) != 0) {
                SDL_SendSensorUpdate(timestamp, sensor, timestamp, data, SDL_arraysize(data));
                SDL_memcpy(sensor->hwdata->data, data, sizeof(data));
            }
        }
    } break;
    case SDL_SENSOR_GYRO:
    {
        CMGyroData *gyroData = SDL_motion_manager.gyroData;
        if (gyroData) {
            CMRotationRate rotationRate = gyroData.rotationRate;
            float data[3];
            data[0] = rotationRate.x;
            data[1] = rotationRate.y;
            data[2] = rotationRate.z;
            if (SDL_memcmp(data, sensor->hwdata->data, sizeof(data)) != 0) {
                SDL_SendSensorUpdate(timestamp, sensor, timestamp, data, SDL_arraysize(data));
                SDL_memcpy(sensor->hwdata->data, data, sizeof(data));
            }
        }
    } break;
    default:
        break;
    }
}

static void SDL_COREMOTION_SensorClose(SDL_Sensor *sensor)
{
    if (sensor->hwdata) {
        switch (sensor->type) {
        case SDL_SENSOR_ACCEL:
            [SDL_motion_manager stopAccelerometerUpdates];
            break;
        case SDL_SENSOR_GYRO:
            [SDL_motion_manager stopGyroUpdates];
            break;
        default:
            break;
        }
        SDL_free(sensor->hwdata);
        sensor->hwdata = NULL;
    }
}

static void SDL_COREMOTION_SensorQuit(void)
{
}

SDL_SensorDriver SDL_COREMOTION_SensorDriver = {
    SDL_COREMOTION_SensorInit,
    SDL_COREMOTION_SensorGetCount,
    SDL_COREMOTION_SensorDetect,
    SDL_COREMOTION_SensorGetDeviceName,
    SDL_COREMOTION_SensorGetDeviceType,
    SDL_COREMOTION_SensorGetDeviceNonPortableType,
    SDL_COREMOTION_SensorGetDeviceInstanceID,
    SDL_COREMOTION_SensorOpen,
    SDL_COREMOTION_SensorUpdate,
    SDL_COREMOTION_SensorClose,
    SDL_COREMOTION_SensorQuit,
};

#endif // SDL_SENSOR_COREMOTION
