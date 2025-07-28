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

#ifndef SDL_sensor_c_h_
#define SDL_sensor_c_h_

#ifdef SDL_THREAD_SAFETY_ANALYSIS
extern SDL_Mutex *SDL_sensor_lock;
#endif

struct SDL_SensorDriver;

// Useful functions and variables from SDL_sensor.c

// Initialization and shutdown functions
extern bool SDL_InitSensors(void);
extern void SDL_QuitSensors(void);

// Return whether the sensor system is currently initialized
extern bool SDL_SensorsInitialized(void);

// Return whether the sensors are currently locked
extern bool SDL_SensorsLocked(void);

// Make sure we currently have the sensors locked
extern void SDL_AssertSensorsLocked(void) SDL_ASSERT_CAPABILITY(SDL_sensor_lock);

extern void SDL_LockSensors(void) SDL_ACQUIRE(SDL_sensor_lock);
extern void SDL_UnlockSensors(void) SDL_RELEASE(SDL_sensor_lock);

// Function to return whether there are any sensors opened by the application
extern bool SDL_SensorsOpened(void);

// Update an individual sensor, used by gamepad sensor fusion
extern void SDL_UpdateSensor(SDL_Sensor *sensor);

// Internal event queueing functions
extern void SDL_SendSensorUpdate(Uint64 timestamp, SDL_Sensor *sensor, Uint64 sensor_timestamp, float *data, int num_values);

#endif // SDL_sensor_c_h_
