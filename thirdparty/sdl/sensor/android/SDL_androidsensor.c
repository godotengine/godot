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

#ifdef SDL_SENSOR_ANDROID

// This is the system specific header for the SDL sensor API
#include <android/sensor.h>

#include "SDL_androidsensor.h"
#include "../SDL_syssensor.h"
#include "../SDL_sensor_c.h"
#include "../../thread/SDL_systhread.h"

#ifndef LOOPER_ID_USER
#define LOOPER_ID_USER 3
#endif

typedef struct
{
    ASensorRef asensor;
    SDL_SensorID instance_id;
    ASensorEventQueue *event_queue;
    SDL_Sensor *sensor;
} SDL_AndroidSensor;

typedef struct
{
    SDL_AtomicInt running;
    SDL_Thread *thread;
    SDL_Semaphore *sem;
} SDL_AndroidSensorThreadContext;

static ASensorManager *SDL_sensor_manager;
static ALooper *SDL_sensor_looper;
static SDL_AndroidSensorThreadContext SDL_sensor_thread_context;
static SDL_AndroidSensor *SDL_sensors SDL_GUARDED_BY(SDL_sensors_lock);
static int SDL_sensors_count;

static int SDLCALL SDL_ANDROID_SensorThread(void *data)
{
    SDL_AndroidSensorThreadContext *ctx = (SDL_AndroidSensorThreadContext *)data;
    int i, events;
    ASensorEvent event;
    struct android_poll_source *source;

    SDL_SetCurrentThreadPriority(SDL_THREAD_PRIORITY_HIGH);

    SDL_sensor_looper = ALooper_prepare(ALOOPER_PREPARE_ALLOW_NON_CALLBACKS);
    SDL_SignalSemaphore(ctx->sem);

    while (SDL_GetAtomicInt(&ctx->running)) {
        Uint64 timestamp = SDL_GetTicksNS();

        if (ALooper_pollOnce(-1, NULL, &events, (void **)&source) == LOOPER_ID_USER) {
            SDL_LockSensors();
            for (i = 0; i < SDL_sensors_count; ++i) {
                if (!SDL_sensors[i].event_queue) {
                    continue;
                }

                SDL_zero(event);
                while (ASensorEventQueue_getEvents(SDL_sensors[i].event_queue, &event, 1) > 0) {
                    SDL_SendSensorUpdate(timestamp, SDL_sensors[i].sensor, timestamp, event.data, SDL_arraysize(event.data));
                }
            }
            SDL_UnlockSensors();
        }
    }

    SDL_sensor_looper = NULL;

    return 0;
}

static void SDL_ANDROID_StopSensorThread(SDL_AndroidSensorThreadContext *ctx)
{
    SDL_SetAtomicInt(&ctx->running, false);

    if (ctx->thread) {
        int result;

        if (SDL_sensor_looper) {
            ALooper_wake(SDL_sensor_looper);
        }
        SDL_WaitThread(ctx->thread, &result);
        ctx->thread = NULL;
    }

    if (ctx->sem) {
        SDL_DestroySemaphore(ctx->sem);
        ctx->sem = NULL;
    }
}

static bool SDL_ANDROID_StartSensorThread(SDL_AndroidSensorThreadContext *ctx)
{
    ctx->sem = SDL_CreateSemaphore(0);
    if (!ctx->sem) {
        SDL_ANDROID_StopSensorThread(ctx);
        return false;
    }

    SDL_SetAtomicInt(&ctx->running, true);
    ctx->thread = SDL_CreateThread(SDL_ANDROID_SensorThread, "Sensors", ctx);
    if (!ctx->thread) {
        SDL_ANDROID_StopSensorThread(ctx);
        return false;
    }

    // Wait for the sensor thread to start
    SDL_WaitSemaphore(ctx->sem);

    return true;
}

static bool SDL_ANDROID_SensorInit(void)
{
    int i, sensors_count;
    ASensorList sensors;

    SDL_sensor_manager = ASensorManager_getInstance();
    if (!SDL_sensor_manager) {
        return SDL_SetError("Couldn't create sensor manager");
    }

    // FIXME: Is the sensor list dynamic?
    sensors_count = ASensorManager_getSensorList(SDL_sensor_manager, &sensors);
    if (sensors_count > 0) {
        SDL_sensors = (SDL_AndroidSensor *)SDL_calloc(sensors_count, sizeof(*SDL_sensors));
        if (!SDL_sensors) {
            return false;
        }

        for (i = 0; i < sensors_count; ++i) {
            SDL_sensors[i].asensor = sensors[i];
            SDL_sensors[i].instance_id = SDL_GetNextObjectID();
        }
        SDL_sensors_count = sensors_count;
    }

    if (!SDL_ANDROID_StartSensorThread(&SDL_sensor_thread_context)) {
        return false;
    }
    return true;
}

static int SDL_ANDROID_SensorGetCount(void)
{
    return SDL_sensors_count;
}

static void SDL_ANDROID_SensorDetect(void)
{
}

static const char *SDL_ANDROID_SensorGetDeviceName(int device_index)
{
    return ASensor_getName(SDL_sensors[device_index].asensor);
}

static SDL_SensorType SDL_ANDROID_SensorGetDeviceType(int device_index)
{
    switch (ASensor_getType(SDL_sensors[device_index].asensor)) {
    case 0x00000001:
        return SDL_SENSOR_ACCEL;
    case 0x00000004:
        return SDL_SENSOR_GYRO;
    default:
        return SDL_SENSOR_UNKNOWN;
    }
}

static int SDL_ANDROID_SensorGetDeviceNonPortableType(int device_index)
{
    return ASensor_getType(SDL_sensors[device_index].asensor);
}

static SDL_SensorID SDL_ANDROID_SensorGetDeviceInstanceID(int device_index)
{
    return SDL_sensors[device_index].instance_id;
}

static bool SDL_ANDROID_SensorOpen(SDL_Sensor *sensor, int device_index)
{
    int delay_us, min_delay_us;

    SDL_LockSensors();
    {
        SDL_sensors[device_index].sensor = sensor;
        SDL_sensors[device_index].event_queue = ASensorManager_createEventQueue(SDL_sensor_manager, SDL_sensor_looper, LOOPER_ID_USER, NULL, NULL);
        if (!SDL_sensors[device_index].event_queue) {
            SDL_UnlockSensors();
            return SDL_SetError("Couldn't create sensor event queue");
        }

        if (ASensorEventQueue_enableSensor(SDL_sensors[device_index].event_queue, SDL_sensors[device_index].asensor) < 0) {
            ASensorManager_destroyEventQueue(SDL_sensor_manager, SDL_sensors[device_index].event_queue);
            SDL_sensors[device_index].event_queue = NULL;
            SDL_UnlockSensors();
            return SDL_SetError("Couldn't enable sensor");
        }

        // Use 60 Hz update rate if possible
        // FIXME: Maybe add a hint for this?
        delay_us = 1000000 / 60;
        min_delay_us = ASensor_getMinDelay(SDL_sensors[device_index].asensor);
        if (delay_us < min_delay_us) {
            delay_us = min_delay_us;
        }
        ASensorEventQueue_setEventRate(SDL_sensors[device_index].event_queue, SDL_sensors[device_index].asensor, delay_us);
    }
    SDL_UnlockSensors();

    return true;
}

static void SDL_ANDROID_SensorUpdate(SDL_Sensor *sensor)
{
}

static void SDL_ANDROID_SensorClose(SDL_Sensor *sensor)
{
    int i;

    for (i = 0; i < SDL_sensors_count; ++i) {
        if (SDL_sensors[i].sensor == sensor) {
            SDL_LockSensors();
            {
                ASensorEventQueue_disableSensor(SDL_sensors[i].event_queue, SDL_sensors[i].asensor);
                ASensorManager_destroyEventQueue(SDL_sensor_manager, SDL_sensors[i].event_queue);
                SDL_sensors[i].event_queue = NULL;
                SDL_sensors[i].sensor = NULL;
            }
            SDL_UnlockSensors();
            break;
        }
    }
}

static void SDL_ANDROID_SensorQuit(void)
{
    // All sensors are closed, but we need to unblock the sensor thread
    SDL_AssertSensorsLocked();
    SDL_UnlockSensors();
    SDL_ANDROID_StopSensorThread(&SDL_sensor_thread_context);
    SDL_LockSensors();

    if (SDL_sensors) {
        SDL_free(SDL_sensors);
        SDL_sensors = NULL;
        SDL_sensors_count = 0;
    }
}

SDL_SensorDriver SDL_ANDROID_SensorDriver = {
    SDL_ANDROID_SensorInit,
    SDL_ANDROID_SensorGetCount,
    SDL_ANDROID_SensorDetect,
    SDL_ANDROID_SensorGetDeviceName,
    SDL_ANDROID_SensorGetDeviceType,
    SDL_ANDROID_SensorGetDeviceNonPortableType,
    SDL_ANDROID_SensorGetDeviceInstanceID,
    SDL_ANDROID_SensorOpen,
    SDL_ANDROID_SensorUpdate,
    SDL_ANDROID_SensorClose,
    SDL_ANDROID_SensorQuit,
};

#endif // SDL_SENSOR_ANDROID
