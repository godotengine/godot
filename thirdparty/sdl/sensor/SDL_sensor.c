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

// This is the sensor API for Simple DirectMedia Layer

#include "SDL_syssensor.h"

#include "../events/SDL_events_c.h"
#include "../joystick/SDL_gamepad_c.h"

static SDL_SensorDriver *SDL_sensor_drivers[] = {
#ifdef SDL_SENSOR_ANDROID
    &SDL_ANDROID_SensorDriver,
#endif
#ifdef SDL_SENSOR_COREMOTION
    &SDL_COREMOTION_SensorDriver,
#endif
#ifdef SDL_SENSOR_WINDOWS
    &SDL_WINDOWS_SensorDriver,
#endif
#ifdef SDL_SENSOR_VITA
    &SDL_VITA_SensorDriver,
#endif
#ifdef SDL_SENSOR_N3DS
    &SDL_N3DS_SensorDriver,
#endif
#if defined(SDL_SENSOR_DUMMY) || defined(SDL_SENSOR_DISABLED)
    &SDL_DUMMY_SensorDriver
#endif
};

#ifndef SDL_THREAD_SAFETY_ANALYSIS
static
#endif
SDL_Mutex *SDL_sensor_lock = NULL; // This needs to support recursive locks
static SDL_AtomicInt SDL_sensor_lock_pending;
static int SDL_sensors_locked;
static bool SDL_sensors_initialized;
static SDL_Sensor *SDL_sensors SDL_GUARDED_BY(SDL_sensor_lock) = NULL;

#define CHECK_SENSOR_MAGIC(sensor, result)                  \
    if (!SDL_ObjectValid(sensor, SDL_OBJECT_TYPE_SENSOR)) { \
        SDL_InvalidParamError("sensor");                    \
        SDL_UnlockSensors();                                \
        return result;                                      \
    }

bool SDL_SensorsInitialized(void)
{
    return SDL_sensors_initialized;
}

void SDL_LockSensors(void)
{
    (void)SDL_AtomicIncRef(&SDL_sensor_lock_pending);
    SDL_LockMutex(SDL_sensor_lock);
    (void)SDL_AtomicDecRef(&SDL_sensor_lock_pending);

    ++SDL_sensors_locked;
}

void SDL_UnlockSensors(void)
{
    bool last_unlock = false;

    --SDL_sensors_locked;

    if (!SDL_sensors_initialized) {
        // NOTE: There's a small window here where another thread could lock the mutex after we've checked for pending locks
        if (!SDL_sensors_locked && SDL_GetAtomicInt(&SDL_sensor_lock_pending) == 0) {
            last_unlock = true;
        }
    }

    /* The last unlock after sensors are uninitialized will cleanup the mutex,
     * allowing applications to lock sensors while reinitializing the system.
     */
    if (last_unlock) {
        SDL_Mutex *sensor_lock = SDL_sensor_lock;

        SDL_LockMutex(sensor_lock);
        {
            SDL_UnlockMutex(SDL_sensor_lock);

            SDL_sensor_lock = NULL;
        }
        SDL_UnlockMutex(sensor_lock);
        SDL_DestroyMutex(sensor_lock);
    } else {
        SDL_UnlockMutex(SDL_sensor_lock);
    }
}

bool SDL_SensorsLocked(void)
{
    return (SDL_sensors_locked > 0);
}

void SDL_AssertSensorsLocked(void)
{
    SDL_assert(SDL_SensorsLocked());
}

bool SDL_InitSensors(void)
{
    int i;
    bool status;

    // Create the sensor list lock
    if (SDL_sensor_lock == NULL) {
        SDL_sensor_lock = SDL_CreateMutex();
    }

    if (!SDL_InitSubSystem(SDL_INIT_EVENTS)) {
        return false;
    }

    SDL_LockSensors();

    SDL_sensors_initialized = true;

    status = false;
    for (i = 0; i < SDL_arraysize(SDL_sensor_drivers); ++i) {
        if (SDL_sensor_drivers[i]->Init()) {
            status = true;
        }
    }

    SDL_UnlockSensors();

    if (!status) {
        SDL_QuitSensors();
    }

    return status;
}

bool SDL_SensorsOpened(void)
{
    bool opened;

    SDL_LockSensors();
    {
        if (SDL_sensors != NULL) {
            opened = true;
        } else {
            opened = false;
        }
    }
    SDL_UnlockSensors();

    return opened;
}

SDL_SensorID *SDL_GetSensors(int *count)
{
    int i, num_sensors, device_index;
    int sensor_index = 0, total_sensors = 0;
    SDL_SensorID *sensors;

    SDL_LockSensors();
    {
        for (i = 0; i < SDL_arraysize(SDL_sensor_drivers); ++i) {
            total_sensors += SDL_sensor_drivers[i]->GetCount();
        }

        sensors = (SDL_SensorID *)SDL_malloc((total_sensors + 1) * sizeof(*sensors));
        if (sensors) {
            if (count) {
                *count = total_sensors;
            }

            for (i = 0; i < SDL_arraysize(SDL_sensor_drivers); ++i) {
                num_sensors = SDL_sensor_drivers[i]->GetCount();
                for (device_index = 0; device_index < num_sensors; ++device_index) {
                    SDL_assert(sensor_index < total_sensors);
                    sensors[sensor_index] = SDL_sensor_drivers[i]->GetDeviceInstanceID(device_index);
                    SDL_assert(sensors[sensor_index] > 0);
                    ++sensor_index;
                }
            }
            SDL_assert(sensor_index == total_sensors);
            sensors[sensor_index] = 0;
        } else {
            if (count) {
                *count = 0;
            }
        }
    }
    SDL_UnlockSensors();

    return sensors;
}

/*
 * Get the driver and device index for a sensor instance ID
 * This should be called while the sensor lock is held, to prevent another thread from updating the list
 */
static bool SDL_GetDriverAndSensorIndex(SDL_SensorID instance_id, SDL_SensorDriver **driver, int *driver_index)
{
    int i, num_sensors, device_index;

    if (instance_id > 0) {
        for (i = 0; i < SDL_arraysize(SDL_sensor_drivers); ++i) {
            num_sensors = SDL_sensor_drivers[i]->GetCount();
            for (device_index = 0; device_index < num_sensors; ++device_index) {
                SDL_SensorID sensor_id = SDL_sensor_drivers[i]->GetDeviceInstanceID(device_index);
                if (sensor_id == instance_id) {
                    *driver = SDL_sensor_drivers[i];
                    *driver_index = device_index;
                    return true;
                }
            }
        }
    }
    SDL_SetError("Sensor %" SDL_PRIu32 " not found", instance_id);
    return false;
}

/*
 * Get the implementation dependent name of a sensor
 */
const char *SDL_GetSensorNameForID(SDL_SensorID instance_id)
{
    SDL_SensorDriver *driver;
    int device_index;
    const char *name = NULL;

    SDL_LockSensors();
    if (SDL_GetDriverAndSensorIndex(instance_id, &driver, &device_index)) {
        name = SDL_GetPersistentString(driver->GetDeviceName(device_index));
    }
    SDL_UnlockSensors();

    return name;
}

SDL_SensorType SDL_GetSensorTypeForID(SDL_SensorID instance_id)
{
    SDL_SensorDriver *driver;
    int device_index;
    SDL_SensorType type = SDL_SENSOR_INVALID;

    SDL_LockSensors();
    if (SDL_GetDriverAndSensorIndex(instance_id, &driver, &device_index)) {
        type = driver->GetDeviceType(device_index);
    }
    SDL_UnlockSensors();

    return type;
}

int SDL_GetSensorNonPortableTypeForID(SDL_SensorID instance_id)
{
    SDL_SensorDriver *driver;
    int device_index;
    int type = -1;

    SDL_LockSensors();
    if (SDL_GetDriverAndSensorIndex(instance_id, &driver, &device_index)) {
        type = driver->GetDeviceNonPortableType(device_index);
    }
    SDL_UnlockSensors();

    return type;
}

/*
 * Open a sensor for use - the index passed as an argument refers to
 * the N'th sensor on the system.  This index is the value which will
 * identify this sensor in future sensor events.
 *
 * This function returns a sensor identifier, or NULL if an error occurred.
 */
SDL_Sensor *SDL_OpenSensor(SDL_SensorID instance_id)
{
    SDL_SensorDriver *driver;
    int device_index;
    SDL_Sensor *sensor;
    SDL_Sensor *sensorlist;
    const char *sensorname = NULL;

    SDL_LockSensors();

    if (!SDL_GetDriverAndSensorIndex(instance_id, &driver, &device_index)) {
        SDL_UnlockSensors();
        return NULL;
    }

    sensorlist = SDL_sensors;
    /* If the sensor is already open, return it
     * it is important that we have a single sensor * for each instance id
     */
    while (sensorlist) {
        if (instance_id == sensorlist->instance_id) {
            sensor = sensorlist;
            ++sensor->ref_count;
            SDL_UnlockSensors();
            return sensor;
        }
        sensorlist = sensorlist->next;
    }

    // Create and initialize the sensor
    sensor = (SDL_Sensor *)SDL_calloc(1, sizeof(*sensor));
    if (!sensor) {
        SDL_UnlockSensors();
        return NULL;
    }
    SDL_SetObjectValid(sensor, SDL_OBJECT_TYPE_SENSOR, true);
    sensor->driver = driver;
    sensor->instance_id = instance_id;
    sensor->type = driver->GetDeviceType(device_index);
    sensor->non_portable_type = driver->GetDeviceNonPortableType(device_index);

    if (!driver->Open(sensor, device_index)) {
        SDL_SetObjectValid(sensor, SDL_OBJECT_TYPE_SENSOR, false);
        SDL_free(sensor);
        SDL_UnlockSensors();
        return NULL;
    }

    sensorname = driver->GetDeviceName(device_index);
    if (sensorname) {
        sensor->name = SDL_strdup(sensorname);
    } else {
        sensor->name = NULL;
    }

    // Add sensor to list
    ++sensor->ref_count;
    // Link the sensor in the list
    sensor->next = SDL_sensors;
    SDL_sensors = sensor;

    driver->Update(sensor);

    SDL_UnlockSensors();

    return sensor;
}

/*
 * Find the SDL_Sensor that owns this instance id
 */
SDL_Sensor *SDL_GetSensorFromID(SDL_SensorID instance_id)
{
    SDL_Sensor *sensor;

    SDL_LockSensors();
    for (sensor = SDL_sensors; sensor; sensor = sensor->next) {
        if (sensor->instance_id == instance_id) {
            break;
        }
    }
    SDL_UnlockSensors();
    return sensor;
}

/*
 * Get the properties associated with a sensor.
 */
SDL_PropertiesID SDL_GetSensorProperties(SDL_Sensor *sensor)
{
    SDL_PropertiesID result;

    SDL_LockSensors();
    {
        CHECK_SENSOR_MAGIC(sensor, 0);

        if (sensor->props == 0) {
            sensor->props = SDL_CreateProperties();
        }
        result = sensor->props;
    }
    SDL_UnlockSensors();

    return result;
}

/*
 * Get the friendly name of this sensor
 */
const char *SDL_GetSensorName(SDL_Sensor *sensor)
{
    const char *result;

    SDL_LockSensors();
    {
        CHECK_SENSOR_MAGIC(sensor, NULL);

        result = SDL_GetPersistentString(sensor->name);
    }
    SDL_UnlockSensors();

    return result;
}

/*
 * Get the type of this sensor
 */
SDL_SensorType SDL_GetSensorType(SDL_Sensor *sensor)
{
    SDL_SensorType result;

    SDL_LockSensors();
    {
        CHECK_SENSOR_MAGIC(sensor, SDL_SENSOR_INVALID);

        result = sensor->type;
    }
    SDL_UnlockSensors();

    return result;
}

/*
 * Get the platform dependent type of this sensor
 */
int SDL_GetSensorNonPortableType(SDL_Sensor *sensor)
{
    int result;

    SDL_LockSensors();
    {
        CHECK_SENSOR_MAGIC(sensor, -1);

        result = sensor->non_portable_type;
    }
    SDL_UnlockSensors();

    return result;
}

/*
 * Get the instance id for this opened sensor
 */
SDL_SensorID SDL_GetSensorID(SDL_Sensor *sensor)
{
    SDL_SensorID result;

    SDL_LockSensors();
    {
        CHECK_SENSOR_MAGIC(sensor, 0);

        result = sensor->instance_id;
    }
    SDL_UnlockSensors();

    return result;
}

/*
 * Get the current state of this sensor
 */
bool SDL_GetSensorData(SDL_Sensor *sensor, float *data, int num_values)
{
    SDL_LockSensors();
    {
        CHECK_SENSOR_MAGIC(sensor, false);

        num_values = SDL_min(num_values, SDL_arraysize(sensor->data));
        SDL_memcpy(data, sensor->data, num_values * sizeof(*data));
    }
    SDL_UnlockSensors();

    return true;
}

/*
 * Close a sensor previously opened with SDL_OpenSensor()
 */
void SDL_CloseSensor(SDL_Sensor *sensor)
{
    SDL_Sensor *sensorlist;
    SDL_Sensor *sensorlistprev;

    SDL_LockSensors();
    {
        CHECK_SENSOR_MAGIC(sensor,);

        // First decrement ref count
        if (--sensor->ref_count > 0) {
            SDL_UnlockSensors();
            return;
        }

        SDL_DestroyProperties(sensor->props);

        sensor->driver->Close(sensor);
        sensor->hwdata = NULL;
        SDL_SetObjectValid(sensor, SDL_OBJECT_TYPE_SENSOR, false);

        sensorlist = SDL_sensors;
        sensorlistprev = NULL;
        while (sensorlist) {
            if (sensor == sensorlist) {
                if (sensorlistprev) {
                    // unlink this entry
                    sensorlistprev->next = sensorlist->next;
                } else {
                    SDL_sensors = sensor->next;
                }
                break;
            }
            sensorlistprev = sensorlist;
            sensorlist = sensorlist->next;
        }

        // Free the data associated with this sensor
        SDL_free(sensor->name);
        SDL_free(sensor);
    }
    SDL_UnlockSensors();
}

void SDL_QuitSensors(void)
{
    int i;

    SDL_LockSensors();

    // Stop the event polling
    while (SDL_sensors) {
        SDL_sensors->ref_count = 1;
        SDL_CloseSensor(SDL_sensors);
    }

    // Quit the sensor setup
    for (i = 0; i < SDL_arraysize(SDL_sensor_drivers); ++i) {
        SDL_sensor_drivers[i]->Quit();
    }

    SDL_QuitSubSystem(SDL_INIT_EVENTS);

    SDL_sensors_initialized = false;

    SDL_UnlockSensors();
}

// These are global for SDL_syssensor.c and SDL_events.c

void SDL_SendSensorUpdate(Uint64 timestamp, SDL_Sensor *sensor, Uint64 sensor_timestamp, float *data, int num_values)
{
    SDL_AssertSensorsLocked();

    // Allow duplicate events, for things like steps and heartbeats

    // Update internal sensor state
    num_values = SDL_min(num_values, SDL_arraysize(sensor->data));
    SDL_memcpy(sensor->data, data, num_values * sizeof(*data));

    // Post the event, if desired
    if (SDL_EventEnabled(SDL_EVENT_SENSOR_UPDATE)) {
        SDL_Event event;
        event.type = SDL_EVENT_SENSOR_UPDATE;
        event.common.timestamp = timestamp;
        event.sensor.which = sensor->instance_id;
        num_values = SDL_min(num_values, SDL_arraysize(event.sensor.data));
        SDL_memset(event.sensor.data, 0, sizeof(event.sensor.data));
        SDL_memcpy(event.sensor.data, data, num_values * sizeof(*data));
        event.sensor.sensor_timestamp = sensor_timestamp;
        SDL_PushEvent(&event);
    }

    SDL_GamepadSensorWatcher(timestamp, sensor->instance_id, sensor_timestamp, data, num_values);
}

void SDL_UpdateSensor(SDL_Sensor *sensor)
{
    SDL_LockSensors();
    {
        CHECK_SENSOR_MAGIC(sensor,);

        sensor->driver->Update(sensor);
    }
    SDL_UnlockSensors();
}

void SDL_UpdateSensors(void)
{
    int i;
    SDL_Sensor *sensor;

    if (!SDL_WasInit(SDL_INIT_SENSOR)) {
        return;
    }

    SDL_LockSensors();

    for (sensor = SDL_sensors; sensor; sensor = sensor->next) {
        sensor->driver->Update(sensor);
    }

    /* this needs to happen AFTER walking the sensor list above, so that any
       dangling hardware data from removed devices can be free'd
     */
    for (i = 0; i < SDL_arraysize(SDL_sensor_drivers); ++i) {
        SDL_sensor_drivers[i]->Detect();
    }

    SDL_UnlockSensors();
}
