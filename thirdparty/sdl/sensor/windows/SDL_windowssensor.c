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

#ifdef SDL_SENSOR_WINDOWS

#include "SDL_windowssensor.h"
#include "../SDL_syssensor.h"
#include "../../core/windows/SDL_windows.h"

#define COBJMACROS
#include <initguid.h>
#include <sensorsapi.h>
#include <sensors.h>

DEFINE_GUID(SDL_CLSID_SensorManager, 0x77A1C827, 0xFCD2, 0x4689, 0x89, 0x15, 0x9D, 0x61, 0x3C, 0xC5, 0xFA, 0x3E);
DEFINE_GUID(SDL_IID_SensorManager, 0xBD77DB67, 0x45A8, 0x42DC, 0x8D, 0x00, 0x6D, 0xCF, 0x15, 0xF8, 0x37, 0x7A);
DEFINE_GUID(SDL_IID_SensorManagerEvents, 0x9B3B0B86, 0x266A, 0x4AAD, 0xB2, 0x1F, 0xFD, 0xE5, 0x50, 0x10, 0x01, 0xB7);
DEFINE_GUID(SDL_IID_SensorEvents, 0x5D8DCC91, 0x4641, 0x47E7, 0xB7, 0xC3, 0xB7, 0x4F, 0x48, 0xA6, 0xC3, 0x91);

// These constants aren't available in Visual Studio 2015 or earlier Windows SDK
DEFINE_PROPERTYKEY(SDL_SENSOR_DATA_TYPE_ANGULAR_VELOCITY_X_DEGREES_PER_SECOND, 0X3F8A69A2, 0X7C5, 0X4E48, 0XA9, 0X65, 0XCD, 0X79, 0X7A, 0XAB, 0X56, 0XD5, 10); //[VT_R8]
DEFINE_PROPERTYKEY(SDL_SENSOR_DATA_TYPE_ANGULAR_VELOCITY_Y_DEGREES_PER_SECOND, 0X3F8A69A2, 0X7C5, 0X4E48, 0XA9, 0X65, 0XCD, 0X79, 0X7A, 0XAB, 0X56, 0XD5, 11); //[VT_R8]
DEFINE_PROPERTYKEY(SDL_SENSOR_DATA_TYPE_ANGULAR_VELOCITY_Z_DEGREES_PER_SECOND, 0X3F8A69A2, 0X7C5, 0X4E48, 0XA9, 0X65, 0XCD, 0X79, 0X7A, 0XAB, 0X56, 0XD5, 12); //[VT_R8]

typedef struct
{
    SDL_SensorID id;
    ISensor *sensor;
    SENSOR_ID sensor_id;
    char *name;
    SDL_SensorType type;
    SDL_Sensor *sensor_opened;

} SDL_Windows_Sensor;

static bool SDL_windowscoinit;
static ISensorManager *SDL_sensor_manager;
static int SDL_num_sensors;
static SDL_Windows_Sensor *SDL_sensors;

static bool ConnectSensor(ISensor *sensor);
static bool DisconnectSensor(ISensor *sensor);

static HRESULT STDMETHODCALLTYPE ISensorManagerEventsVtbl_QueryInterface(ISensorManagerEvents *This, REFIID riid, void **ppvObject)
{
    if (!ppvObject) {
        return E_INVALIDARG;
    }

    *ppvObject = NULL;
    if (WIN_IsEqualIID(riid, &IID_IUnknown) || WIN_IsEqualIID(riid, &SDL_IID_SensorManagerEvents)) {
        *ppvObject = This;
        return S_OK;
    }
    return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE ISensorManagerEventsVtbl_AddRef(ISensorManagerEvents *This)
{
    return 1;
}

static ULONG STDMETHODCALLTYPE ISensorManagerEventsVtbl_Release(ISensorManagerEvents *This)
{
    return 1;
}

static HRESULT STDMETHODCALLTYPE ISensorManagerEventsVtbl_OnSensorEnter(ISensorManagerEvents *This, ISensor *pSensor, SensorState state)
{
    ConnectSensor(pSensor);
    return S_OK;
}

static ISensorManagerEventsVtbl sensor_manager_events_vtbl = {
    ISensorManagerEventsVtbl_QueryInterface,
    ISensorManagerEventsVtbl_AddRef,
    ISensorManagerEventsVtbl_Release,
    ISensorManagerEventsVtbl_OnSensorEnter
};
static ISensorManagerEvents sensor_manager_events = {
    &sensor_manager_events_vtbl
};

static HRESULT STDMETHODCALLTYPE ISensorEventsVtbl_QueryInterface(ISensorEvents *This, REFIID riid, void **ppvObject)
{
    if (!ppvObject) {
        return E_INVALIDARG;
    }

    *ppvObject = NULL;
    if (WIN_IsEqualIID(riid, &IID_IUnknown) || WIN_IsEqualIID(riid, &SDL_IID_SensorEvents)) {
        *ppvObject = This;
        return S_OK;
    }
    return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE ISensorEventsVtbl_AddRef(ISensorEvents *This)
{
    return 1;
}

static ULONG STDMETHODCALLTYPE ISensorEventsVtbl_Release(ISensorEvents *This)
{
    return 1;
}

static HRESULT STDMETHODCALLTYPE ISensorEventsVtbl_OnStateChanged(ISensorEvents *This, ISensor *pSensor, SensorState state)
{
#ifdef DEBUG_SENSORS
    int i;

    SDL_LockSensors();
    for (i = 0; i < SDL_num_sensors; ++i) {
        if (pSensor == SDL_sensors[i].sensor) {
            SDL_Log("Sensor %s state changed to %d", SDL_sensors[i].name, state);
        }
    }
    SDL_UnlockSensors();
#endif
    return S_OK;
}

static HRESULT STDMETHODCALLTYPE ISensorEventsVtbl_OnDataUpdated(ISensorEvents *This, ISensor *pSensor, ISensorDataReport *pNewData)
{
    int i;
    Uint64 timestamp = SDL_GetTicksNS();

    SDL_LockSensors();
    for (i = 0; i < SDL_num_sensors; ++i) {
        if (pSensor == SDL_sensors[i].sensor) {
            if (SDL_sensors[i].sensor_opened) {
                HRESULT hrX, hrY, hrZ;
                PROPVARIANT valueX = { 0 }, valueY = { 0 }, valueZ = { 0 };
                SYSTEMTIME sensor_systemtime;
                FILETIME sensor_filetime;
                Uint64 sensor_timestamp;

#ifdef DEBUG_SENSORS
                SDL_Log("Sensor %s data updated", SDL_sensors[i].name);
#endif
                if (SUCCEEDED(ISensorDataReport_GetTimestamp(pNewData, &sensor_systemtime)) &&
                    SystemTimeToFileTime(&sensor_systemtime, &sensor_filetime)) {
                    ULARGE_INTEGER sensor_time;
                    sensor_time.u.HighPart = sensor_filetime.dwHighDateTime;
                    sensor_time.u.LowPart = sensor_filetime.dwLowDateTime;
                    sensor_timestamp = sensor_time.QuadPart * 100;
                } else {
                    sensor_timestamp = timestamp;
                }

                switch (SDL_sensors[i].type) {
                case SDL_SENSOR_ACCEL:
                    hrX = ISensorDataReport_GetSensorValue(pNewData, &SENSOR_DATA_TYPE_ACCELERATION_X_G, &valueX);
                    hrY = ISensorDataReport_GetSensorValue(pNewData, &SENSOR_DATA_TYPE_ACCELERATION_Y_G, &valueY);
                    hrZ = ISensorDataReport_GetSensorValue(pNewData, &SENSOR_DATA_TYPE_ACCELERATION_Z_G, &valueZ);
                    if (SUCCEEDED(hrX) && SUCCEEDED(hrY) && SUCCEEDED(hrZ) &&
                        valueX.vt == VT_R8 && valueY.vt == VT_R8 && valueZ.vt == VT_R8) {
                        float values[3];

                        values[0] = (float)valueX.dblVal * SDL_STANDARD_GRAVITY;
                        values[1] = (float)valueY.dblVal * SDL_STANDARD_GRAVITY;
                        values[2] = (float)valueZ.dblVal * SDL_STANDARD_GRAVITY;
                        SDL_SendSensorUpdate(timestamp, SDL_sensors[i].sensor_opened, sensor_timestamp, values, 3);
                    }
                    break;
                case SDL_SENSOR_GYRO:
                    hrX = ISensorDataReport_GetSensorValue(pNewData, &SDL_SENSOR_DATA_TYPE_ANGULAR_VELOCITY_X_DEGREES_PER_SECOND, &valueX);
                    hrY = ISensorDataReport_GetSensorValue(pNewData, &SDL_SENSOR_DATA_TYPE_ANGULAR_VELOCITY_Y_DEGREES_PER_SECOND, &valueY);
                    hrZ = ISensorDataReport_GetSensorValue(pNewData, &SDL_SENSOR_DATA_TYPE_ANGULAR_VELOCITY_Z_DEGREES_PER_SECOND, &valueZ);
                    if (SUCCEEDED(hrX) && SUCCEEDED(hrY) && SUCCEEDED(hrZ) &&
                        valueX.vt == VT_R8 && valueY.vt == VT_R8 && valueZ.vt == VT_R8) {
                        const float DEGREES_TO_RADIANS = (SDL_PI_F / 180.0f);
                        float values[3];

                        values[0] = (float)valueX.dblVal * DEGREES_TO_RADIANS;
                        values[1] = (float)valueY.dblVal * DEGREES_TO_RADIANS;
                        values[2] = (float)valueZ.dblVal * DEGREES_TO_RADIANS;
                        SDL_SendSensorUpdate(timestamp, SDL_sensors[i].sensor_opened, sensor_timestamp, values, 3);
                    }
                    break;
                default:
                    // FIXME: Need to know how to interpret the data for this sensor
                    break;
                }
            }
            break;
        }
    }
    SDL_UnlockSensors();

    return S_OK;
}

static HRESULT STDMETHODCALLTYPE ISensorEventsVtbl_OnEvent(ISensorEvents *This, ISensor *pSensor, REFGUID eventID, IPortableDeviceValues *pEventData)
{
#ifdef DEBUG_SENSORS
    int i;

    SDL_LockSensors();
    for (i = 0; i < SDL_num_sensors; ++i) {
        if (pSensor == SDL_sensors[i].sensor) {
            SDL_Log("Sensor %s event occurred", SDL_sensors[i].name);
        }
    }
    SDL_UnlockSensors();
#endif
    return S_OK;
}

static HRESULT STDMETHODCALLTYPE ISensorEventsVtbl_OnLeave(ISensorEvents *This, REFSENSOR_ID ID)
{
    int i;

    SDL_LockSensors();
    for (i = 0; i < SDL_num_sensors; ++i) {
        if (WIN_IsEqualIID(ID, &SDL_sensors[i].sensor_id)) {
#ifdef DEBUG_SENSORS
            SDL_Log("Sensor %s disconnected", SDL_sensors[i].name);
#endif
            DisconnectSensor(SDL_sensors[i].sensor);
        }
    }
    SDL_UnlockSensors();

    return S_OK;
}

static ISensorEventsVtbl sensor_events_vtbl = {
    ISensorEventsVtbl_QueryInterface,
    ISensorEventsVtbl_AddRef,
    ISensorEventsVtbl_Release,
    ISensorEventsVtbl_OnStateChanged,
    ISensorEventsVtbl_OnDataUpdated,
    ISensorEventsVtbl_OnEvent,
    ISensorEventsVtbl_OnLeave
};
static ISensorEvents sensor_events = {
    &sensor_events_vtbl
};

static bool ConnectSensor(ISensor *sensor)
{
    SDL_Windows_Sensor *new_sensor, *new_sensors;
    HRESULT hr;
    SENSOR_ID sensor_id;
    SENSOR_TYPE_ID type_id;
    SDL_SensorType type;
    BSTR bstr_name = NULL;
    char *name;

    hr = ISensor_GetID(sensor, &sensor_id);
    if (FAILED(hr)) {
        return WIN_SetErrorFromHRESULT("Couldn't get sensor ID", hr);
    }

    hr = ISensor_GetType(sensor, &type_id);
    if (FAILED(hr)) {
        return WIN_SetErrorFromHRESULT("Couldn't get sensor type", hr);
    }

    if (WIN_IsEqualIID(&type_id, &SENSOR_TYPE_ACCELEROMETER_3D)) {
        type = SDL_SENSOR_ACCEL;
    } else if (WIN_IsEqualIID(&type_id, &SENSOR_TYPE_GYROMETER_3D)) {
        type = SDL_SENSOR_GYRO;
    } else {
        return SDL_SetError("Unknown sensor type");
    }

    hr = ISensor_GetFriendlyName(sensor, &bstr_name);
    if (SUCCEEDED(hr) && bstr_name) {
        name = WIN_StringToUTF8W(bstr_name);
    } else {
        name = SDL_strdup("Unknown Sensor");
    }
    if (bstr_name != NULL) {
        SysFreeString(bstr_name);
    }
    if (!name) {
        return false;
    }

    SDL_LockSensors();
    new_sensors = (SDL_Windows_Sensor *)SDL_realloc(SDL_sensors, (SDL_num_sensors + 1) * sizeof(SDL_Windows_Sensor));
    if (!new_sensors) {
        SDL_UnlockSensors();
        SDL_free(name);
        return false;
    }

    ISensor_AddRef(sensor);
    ISensor_SetEventSink(sensor, &sensor_events);

    SDL_sensors = new_sensors;
    new_sensor = &SDL_sensors[SDL_num_sensors];
    ++SDL_num_sensors;

    SDL_zerop(new_sensor);
    new_sensor->id = SDL_GetNextObjectID();
    new_sensor->sensor = sensor;
    new_sensor->type = type;
    new_sensor->name = name;

    SDL_UnlockSensors();

    return true;
}

static bool DisconnectSensor(ISensor *sensor)
{
    SDL_Windows_Sensor *old_sensor;
    int i;

    SDL_LockSensors();
    for (i = 0; i < SDL_num_sensors; ++i) {
        old_sensor = &SDL_sensors[i];
        if (sensor == old_sensor->sensor) {
            /* This call hangs for some reason:
             * https://github.com/libsdl-org/SDL/issues/5288
             */
            // ISensor_SetEventSink(sensor, NULL);
            ISensor_Release(sensor);
            SDL_free(old_sensor->name);
            --SDL_num_sensors;
            if (i < SDL_num_sensors) {
                SDL_memmove(&SDL_sensors[i], &SDL_sensors[i + 1], (SDL_num_sensors - i) * sizeof(SDL_sensors[i]));
            }
            break;
        }
    }
    SDL_UnlockSensors();

    return true;
}

static bool SDL_WINDOWS_SensorInit(void)
{
    HRESULT hr;
    ISensorCollection *sensor_collection = NULL;

    if (WIN_CoInitialize() == S_OK) {
        SDL_windowscoinit = true;
    }

    hr = CoCreateInstance(&SDL_CLSID_SensorManager, NULL, CLSCTX_INPROC_SERVER, &SDL_IID_SensorManager, (LPVOID *)&SDL_sensor_manager);
    if (FAILED(hr)) {
        // If we can't create a sensor manager (i.e. on Wine), we won't have any sensors, but don't fail the init
        return true; // WIN_SetErrorFromHRESULT("Couldn't create the sensor manager", hr);
    }

    hr = ISensorManager_SetEventSink(SDL_sensor_manager, &sensor_manager_events);
    if (FAILED(hr)) {
        ISensorManager_Release(SDL_sensor_manager);
        SDL_sensor_manager = NULL;
        return WIN_SetErrorFromHRESULT("Couldn't set the sensor manager event sink", hr);
    }

    hr = ISensorManager_GetSensorsByCategory(SDL_sensor_manager, &SENSOR_CATEGORY_ALL, &sensor_collection);
    if (SUCCEEDED(hr)) {
        ULONG i, count;

        hr = ISensorCollection_GetCount(sensor_collection, &count);
        if (SUCCEEDED(hr)) {
            for (i = 0; i < count; ++i) {
                ISensor *sensor;

                hr = ISensorCollection_GetAt(sensor_collection, i, &sensor);
                if (SUCCEEDED(hr)) {
                    SensorState state;

                    hr = ISensor_GetState(sensor, &state);
                    if (SUCCEEDED(hr)) {
                        ISensorManagerEventsVtbl_OnSensorEnter(&sensor_manager_events, sensor, state);
                    }
                    ISensorManager_Release(sensor);
                }
            }
        }
        ISensorCollection_Release(sensor_collection);
    }
    return true;
}

static int SDL_WINDOWS_SensorGetCount(void)
{
    return SDL_num_sensors;
}

static void SDL_WINDOWS_SensorDetect(void)
{
}

static const char *SDL_WINDOWS_SensorGetDeviceName(int device_index)
{
    return SDL_sensors[device_index].name;
}

static SDL_SensorType SDL_WINDOWS_SensorGetDeviceType(int device_index)
{
    return SDL_sensors[device_index].type;
}

static int SDL_WINDOWS_SensorGetDeviceNonPortableType(int device_index)
{
    return -1;
}

static SDL_SensorID SDL_WINDOWS_SensorGetDeviceInstanceID(int device_index)
{
    return SDL_sensors[device_index].id;
}

static bool SDL_WINDOWS_SensorOpen(SDL_Sensor *sensor, int device_index)
{
    SDL_sensors[device_index].sensor_opened = sensor;
    return true;
}

static void SDL_WINDOWS_SensorUpdate(SDL_Sensor *sensor)
{
}

static void SDL_WINDOWS_SensorClose(SDL_Sensor *sensor)
{
    int i;

    for (i = 0; i < SDL_num_sensors; ++i) {
        if (sensor == SDL_sensors[i].sensor_opened) {
            SDL_sensors[i].sensor_opened = NULL;
            break;
        }
    }
}

static void SDL_WINDOWS_SensorQuit(void)
{
    while (SDL_num_sensors > 0) {
        DisconnectSensor(SDL_sensors[0].sensor);
    }

    if (SDL_sensor_manager) {
        ISensorManager_SetEventSink(SDL_sensor_manager, NULL);
        ISensorManager_Release(SDL_sensor_manager);
        SDL_sensor_manager = NULL;
    }

    if (SDL_windowscoinit) {
        WIN_CoUninitialize();
    }
}

SDL_SensorDriver SDL_WINDOWS_SensorDriver = {
    SDL_WINDOWS_SensorInit,
    SDL_WINDOWS_SensorGetCount,
    SDL_WINDOWS_SensorDetect,
    SDL_WINDOWS_SensorGetDeviceName,
    SDL_WINDOWS_SensorGetDeviceType,
    SDL_WINDOWS_SensorGetDeviceNonPortableType,
    SDL_WINDOWS_SensorGetDeviceInstanceID,
    SDL_WINDOWS_SensorOpen,
    SDL_WINDOWS_SensorUpdate,
    SDL_WINDOWS_SensorClose,
    SDL_WINDOWS_SensorQuit,
};

#endif // SDL_SENSOR_WINDOWS
