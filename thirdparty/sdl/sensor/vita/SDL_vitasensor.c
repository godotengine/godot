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

#ifdef SDL_SENSOR_VITA

#include "SDL_vitasensor.h"
#include "../SDL_syssensor.h"
#include <psp2/motion.h>

#ifndef SCE_MOTION_MAX_NUM_STATES
#define SCE_MOTION_MAX_NUM_STATES 64
#endif

typedef struct
{
    SDL_SensorType type;
    SDL_SensorID instance_id;
} SDL_VitaSensor;

static SDL_VitaSensor *SDL_sensors;
static int SDL_sensors_count;

static bool SDL_VITA_SensorInit(void)
{
    sceMotionReset();
    sceMotionStartSampling();
    // not sure if these are needed, we are reading unfiltered state
    sceMotionSetAngleThreshold(0);
    sceMotionSetDeadband(SCE_FALSE);
    sceMotionSetTiltCorrection(SCE_FALSE);

    SDL_sensors_count = 2;

    SDL_sensors = (SDL_VitaSensor *)SDL_calloc(SDL_sensors_count, sizeof(*SDL_sensors));
    if (!SDL_sensors) {
        return false;
    }

    SDL_sensors[0].type = SDL_SENSOR_ACCEL;
    SDL_sensors[0].instance_id = SDL_GetNextObjectID();
    SDL_sensors[1].type = SDL_SENSOR_GYRO;
    SDL_sensors[1].instance_id = SDL_GetNextObjectID();

    return true;
}

static int SDL_VITA_SensorGetCount(void)
{
    return SDL_sensors_count;
}

static void SDL_VITA_SensorDetect(void)
{
}

static const char *SDL_VITA_SensorGetDeviceName(int device_index)
{
    if (device_index < SDL_sensors_count) {
        switch (SDL_sensors[device_index].type) {
        case SDL_SENSOR_ACCEL:
            return "Accelerometer";
        case SDL_SENSOR_GYRO:
            return "Gyro";
        default:
            return "Unknown";
        }
    }

    return NULL;
}

static SDL_SensorType SDL_VITA_SensorGetDeviceType(int device_index)
{
    if (device_index < SDL_sensors_count) {
        return SDL_sensors[device_index].type;
    }

    return SDL_SENSOR_INVALID;
}

static int SDL_VITA_SensorGetDeviceNonPortableType(int device_index)
{
    if (device_index < SDL_sensors_count) {
        return SDL_sensors[device_index].type;
    }
    return -1;
}

static SDL_SensorID SDL_VITA_SensorGetDeviceInstanceID(int device_index)
{
    if (device_index < SDL_sensors_count) {
        return SDL_sensors[device_index].instance_id;
    }
    return -1;
}

static bool SDL_VITA_SensorOpen(SDL_Sensor *sensor, int device_index)
{
    struct sensor_hwdata *hwdata;

    hwdata = (struct sensor_hwdata *)SDL_calloc(1, sizeof(*hwdata));
    if (!hwdata) {
        return false;
    }
    sensor->hwdata = hwdata;

    return true;
}

static void SDL_VITA_SensorUpdate(SDL_Sensor *sensor)
{
    int err = 0;
    SceMotionSensorState motionState[SCE_MOTION_MAX_NUM_STATES];
    Uint64 timestamp = SDL_GetTicksNS();

    SDL_zero(motionState);
    err = sceMotionGetSensorState(motionState, SCE_MOTION_MAX_NUM_STATES);
    if (err != 0) {
        return;
    }

    for (int i = 0; i < SCE_MOTION_MAX_NUM_STATES; i++) {
        if (sensor->hwdata->counter < motionState[i].counter) {
            unsigned int tick = motionState[i].timestamp;
            unsigned int delta;

            sensor->hwdata->counter = motionState[i].counter;

            if (sensor->hwdata->last_tick > tick) {
                SDL_COMPILE_TIME_ASSERT(tick, sizeof(tick) == sizeof(Uint32));
                delta = (SDL_MAX_UINT32 - sensor->hwdata->last_tick + tick + 1);
            } else {
                delta = (tick - sensor->hwdata->last_tick);
            }
            sensor->hwdata->sensor_timestamp += SDL_US_TO_NS(delta);
            sensor->hwdata->last_tick = tick;

            switch (sensor->type) {
            case SDL_SENSOR_ACCEL:
            {
                float data[3];
                data[0] = motionState[i].accelerometer.x * SDL_STANDARD_GRAVITY;
                data[1] = motionState[i].accelerometer.y * SDL_STANDARD_GRAVITY;
                data[2] = motionState[i].accelerometer.z * SDL_STANDARD_GRAVITY;
                SDL_SendSensorUpdate(timestamp, sensor, sensor->hwdata->sensor_timestamp, data, SDL_arraysize(data));
            } break;
            case SDL_SENSOR_GYRO:
            {
                float data[3];
                data[0] = motionState[i].gyro.x;
                data[1] = motionState[i].gyro.y;
                data[2] = motionState[i].gyro.z;
                SDL_SendSensorUpdate(timestamp, sensor, sensor->hwdata->sensor_timestamp, data, SDL_arraysize(data));
            } break;
            default:
                break;
            }
        }
    }
}

static void SDL_VITA_SensorClose(SDL_Sensor *sensor)
{
}

static void SDL_VITA_SensorQuit(void)
{
    sceMotionStopSampling();
}

SDL_SensorDriver SDL_VITA_SensorDriver = {
    SDL_VITA_SensorInit,
    SDL_VITA_SensorGetCount,
    SDL_VITA_SensorDetect,
    SDL_VITA_SensorGetDeviceName,
    SDL_VITA_SensorGetDeviceType,
    SDL_VITA_SensorGetDeviceNonPortableType,
    SDL_VITA_SensorGetDeviceInstanceID,
    SDL_VITA_SensorOpen,
    SDL_VITA_SensorUpdate,
    SDL_VITA_SensorClose,
    SDL_VITA_SensorQuit,
};

#endif // SDL_SENSOR_VITA
