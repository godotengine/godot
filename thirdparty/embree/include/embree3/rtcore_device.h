// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "rtcore_common.h"

RTC_NAMESPACE_BEGIN

/* Opaque device type */
typedef struct RTCDeviceTy* RTCDevice;

/* Creates a new Embree device. */
RTC_API RTCDevice rtcNewDevice(const char* config);

/* Retains the Embree device (increments the reference count). */
RTC_API void rtcRetainDevice(RTCDevice device);
  
/* Releases an Embree device (decrements the reference count). */
RTC_API void rtcReleaseDevice(RTCDevice device);

/* Device properties */
enum RTCDeviceProperty
{
  RTC_DEVICE_PROPERTY_VERSION       = 0,
  RTC_DEVICE_PROPERTY_VERSION_MAJOR = 1,
  RTC_DEVICE_PROPERTY_VERSION_MINOR = 2,
  RTC_DEVICE_PROPERTY_VERSION_PATCH = 3,

  RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED  = 32,
  RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED  = 33,
  RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED = 34,
  RTC_DEVICE_PROPERTY_RAY_STREAM_SUPPORTED   = 35,

  RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED          = 64,
  RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED    = 65,
  RTC_DEVICE_PROPERTY_FILTER_FUNCTION_SUPPORTED   = 66,
  RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED = 67,

  RTC_DEVICE_PROPERTY_TRIANGLE_GEOMETRY_SUPPORTED    = 96,
  RTC_DEVICE_PROPERTY_QUAD_GEOMETRY_SUPPORTED        = 97,
  RTC_DEVICE_PROPERTY_SUBDIVISION_GEOMETRY_SUPPORTED = 98,
  RTC_DEVICE_PROPERTY_CURVE_GEOMETRY_SUPPORTED       = 99,
  RTC_DEVICE_PROPERTY_USER_GEOMETRY_SUPPORTED        = 100,
  RTC_DEVICE_PROPERTY_POINT_GEOMETRY_SUPPORTED       = 101,

  RTC_DEVICE_PROPERTY_TASKING_SYSTEM        = 128,
  RTC_DEVICE_PROPERTY_JOIN_COMMIT_SUPPORTED = 129
};

/* Gets a device property. */
RTC_API ssize_t rtcGetDeviceProperty(RTCDevice device, enum RTCDeviceProperty prop);

/* Sets a device property. */
RTC_API void rtcSetDeviceProperty(RTCDevice device, const enum RTCDeviceProperty prop, ssize_t value);
  
/* Error codes */
enum RTCError
{
  RTC_ERROR_NONE              = 0,
  RTC_ERROR_UNKNOWN           = 1,
  RTC_ERROR_INVALID_ARGUMENT  = 2,
  RTC_ERROR_INVALID_OPERATION = 3,
  RTC_ERROR_OUT_OF_MEMORY     = 4,
  RTC_ERROR_UNSUPPORTED_CPU   = 5,
  RTC_ERROR_CANCELLED         = 6
};

/* Returns the error code. */
RTC_API enum RTCError rtcGetDeviceError(RTCDevice device);

/* Error callback function */
typedef void (*RTCErrorFunction)(void* userPtr, enum RTCError code, const char* str);

/* Sets the error callback function. */
RTC_API void rtcSetDeviceErrorFunction(RTCDevice device, RTCErrorFunction error, void* userPtr);

/* Memory monitor callback function */
typedef bool (*RTCMemoryMonitorFunction)(void* ptr, ssize_t bytes, bool post);

/* Sets the memory monitor callback function. */
RTC_API void rtcSetDeviceMemoryMonitorFunction(RTCDevice device, RTCMemoryMonitorFunction memoryMonitor, void* userPtr);

RTC_NAMESPACE_END
