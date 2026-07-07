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

#if defined(SDL_SENSOR_DUMMY) || defined(SDL_SENSOR_DISABLED)

#include "SDL_dummysensor.h"
#include "../SDL_syssensor.h"

static bool SDL_DUMMY_SensorInit(void)
{
    return true;
}

static int SDL_DUMMY_SensorGetCount(void)
{
    return 0;
}

static void SDL_DUMMY_SensorDetect(void)
{
}

static const char *SDL_DUMMY_SensorGetDeviceName(int device_index)
{
    return NULL;
}

static SDL_SensorType SDL_DUMMY_SensorGetDeviceType(int device_index)
{
    return SDL_SENSOR_INVALID;
}

static int SDL_DUMMY_SensorGetDeviceNonPortableType(int device_index)
{
    return -1;
}

static SDL_SensorID SDL_DUMMY_SensorGetDeviceInstanceID(int device_index)
{
    return -1;
}

static bool SDL_DUMMY_SensorOpen(SDL_Sensor *sensor, int device_index)
{
    return SDL_Unsupported();
}

static void SDL_DUMMY_SensorUpdate(SDL_Sensor *sensor)
{
}

static void SDL_DUMMY_SensorClose(SDL_Sensor *sensor)
{
}

static void SDL_DUMMY_SensorQuit(void)
{
}

SDL_SensorDriver SDL_DUMMY_SensorDriver = {
    SDL_DUMMY_SensorInit,
    SDL_DUMMY_SensorGetCount,
    SDL_DUMMY_SensorDetect,
    SDL_DUMMY_SensorGetDeviceName,
    SDL_DUMMY_SensorGetDeviceType,
    SDL_DUMMY_SensorGetDeviceNonPortableType,
    SDL_DUMMY_SensorGetDeviceInstanceID,
    SDL_DUMMY_SensorOpen,
    SDL_DUMMY_SensorUpdate,
    SDL_DUMMY_SensorClose,
    SDL_DUMMY_SensorQuit,
};

#endif // SDL_SENSOR_DUMMY || SDL_SENSOR_DISABLED
