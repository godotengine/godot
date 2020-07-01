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

#include <stddef.h>
#include <sys/types.h>
#include <stdbool.h>

#include "rtcore_version.h"

RTC_NAMESPACE_BEGIN

#if defined(_WIN32)
#if defined(_M_X64)
typedef long long ssize_t;
#else
typedef int ssize_t;
#endif
#endif

#ifdef _WIN32
#  define RTC_ALIGN(...) __declspec(align(__VA_ARGS__))
#else
#  define RTC_ALIGN(...) __attribute__((aligned(__VA_ARGS__)))
#endif

#if !defined (RTC_DEPRECATED)
#ifdef __GNUC__
  #define RTC_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
  #define RTC_DEPRECATED __declspec(deprecated)
#else
  #define RTC_DEPRECATED
#endif
#endif

#if defined(_WIN32) 
#  define RTC_FORCEINLINE __forceinline
#else
#  define RTC_FORCEINLINE inline __attribute__((always_inline))
#endif

/* Invalid geometry ID */
#define RTC_INVALID_GEOMETRY_ID ((unsigned int)-1)

/* Maximum number of time steps */
#define RTC_MAX_TIME_STEP_COUNT 129

/* Maximum number of instancing levels */
#define RTC_MAX_INSTANCE_LEVEL_COUNT 1

/* Formats of buffers and other data structures */
enum RTCFormat
{
  RTC_FORMAT_UNDEFINED = 0,

  /* 8-bit unsigned integer */
  RTC_FORMAT_UCHAR = 0x1001,
  RTC_FORMAT_UCHAR2,
  RTC_FORMAT_UCHAR3,
  RTC_FORMAT_UCHAR4,

  /* 8-bit signed integer */
  RTC_FORMAT_CHAR = 0x2001,
  RTC_FORMAT_CHAR2,
  RTC_FORMAT_CHAR3,
  RTC_FORMAT_CHAR4,

  /* 16-bit unsigned integer */
  RTC_FORMAT_USHORT = 0x3001,
  RTC_FORMAT_USHORT2,
  RTC_FORMAT_USHORT3,
  RTC_FORMAT_USHORT4,

  /* 16-bit signed integer */
  RTC_FORMAT_SHORT = 0x4001,
  RTC_FORMAT_SHORT2,
  RTC_FORMAT_SHORT3,
  RTC_FORMAT_SHORT4,

  /* 32-bit unsigned integer */
  RTC_FORMAT_UINT = 0x5001,
  RTC_FORMAT_UINT2,
  RTC_FORMAT_UINT3,
  RTC_FORMAT_UINT4,

  /* 32-bit signed integer */
  RTC_FORMAT_INT = 0x6001,
  RTC_FORMAT_INT2,
  RTC_FORMAT_INT3,
  RTC_FORMAT_INT4,

  /* 64-bit unsigned integer */
  RTC_FORMAT_ULLONG = 0x7001,
  RTC_FORMAT_ULLONG2,
  RTC_FORMAT_ULLONG3,
  RTC_FORMAT_ULLONG4,

  /* 64-bit signed integer */
  RTC_FORMAT_LLONG = 0x8001,
  RTC_FORMAT_LLONG2,
  RTC_FORMAT_LLONG3,
  RTC_FORMAT_LLONG4,

  /* 32-bit float */
  RTC_FORMAT_FLOAT = 0x9001,
  RTC_FORMAT_FLOAT2,
  RTC_FORMAT_FLOAT3,
  RTC_FORMAT_FLOAT4,
  RTC_FORMAT_FLOAT5,
  RTC_FORMAT_FLOAT6,
  RTC_FORMAT_FLOAT7,
  RTC_FORMAT_FLOAT8,
  RTC_FORMAT_FLOAT9,
  RTC_FORMAT_FLOAT10,
  RTC_FORMAT_FLOAT11,
  RTC_FORMAT_FLOAT12,
  RTC_FORMAT_FLOAT13,
  RTC_FORMAT_FLOAT14,
  RTC_FORMAT_FLOAT15,
  RTC_FORMAT_FLOAT16,

  /* 32-bit float matrix (row-major order) */
  RTC_FORMAT_FLOAT2X2_ROW_MAJOR = 0x9122,
  RTC_FORMAT_FLOAT2X3_ROW_MAJOR = 0x9123,
  RTC_FORMAT_FLOAT2X4_ROW_MAJOR = 0x9124,
  RTC_FORMAT_FLOAT3X2_ROW_MAJOR = 0x9132,
  RTC_FORMAT_FLOAT3X3_ROW_MAJOR = 0x9133,
  RTC_FORMAT_FLOAT3X4_ROW_MAJOR = 0x9134,
  RTC_FORMAT_FLOAT4X2_ROW_MAJOR = 0x9142,
  RTC_FORMAT_FLOAT4X3_ROW_MAJOR = 0x9143,
  RTC_FORMAT_FLOAT4X4_ROW_MAJOR = 0x9144,

  /* 32-bit float matrix (column-major order) */
  RTC_FORMAT_FLOAT2X2_COLUMN_MAJOR = 0x9222,
  RTC_FORMAT_FLOAT2X3_COLUMN_MAJOR = 0x9223,
  RTC_FORMAT_FLOAT2X4_COLUMN_MAJOR = 0x9224,
  RTC_FORMAT_FLOAT3X2_COLUMN_MAJOR = 0x9232,
  RTC_FORMAT_FLOAT3X3_COLUMN_MAJOR = 0x9233,
  RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR = 0x9234,
  RTC_FORMAT_FLOAT4X2_COLUMN_MAJOR = 0x9242,
  RTC_FORMAT_FLOAT4X3_COLUMN_MAJOR = 0x9243,
  RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR = 0x9244,

  /* special 12-byte format for grids */
  RTC_FORMAT_GRID = 0xA001
};

/* Build quality levels */
enum RTCBuildQuality
{
  RTC_BUILD_QUALITY_LOW    = 0,
  RTC_BUILD_QUALITY_MEDIUM = 1,
  RTC_BUILD_QUALITY_HIGH   = 2,
  RTC_BUILD_QUALITY_REFIT  = 3,
};

/* Axis-aligned bounding box representation */
struct RTC_ALIGN(16) RTCBounds
{
  float lower_x, lower_y, lower_z, align0;
  float upper_x, upper_y, upper_z, align1;
};

/* Linear axis-aligned bounding box representation */
struct RTC_ALIGN(16) RTCLinearBounds
{
  struct RTCBounds bounds0;
  struct RTCBounds bounds1;
};

/* Intersection context flags */
enum RTCIntersectContextFlags
{
  RTC_INTERSECT_CONTEXT_FLAG_NONE       = 0,
  RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT = (0 << 0), // optimize for incoherent rays
  RTC_INTERSECT_CONTEXT_FLAG_COHERENT   = (1 << 0)  // optimize for coherent rays
};

/* Arguments for RTCFilterFunctionN */
struct RTCFilterFunctionNArguments
{
  int* valid;
  void* geometryUserPtr;
  const struct RTCIntersectContext* context;
  struct RTCRayN* ray;
  struct RTCHitN* hit;
  unsigned int N;
};

/* Filter callback function */
typedef void (*RTCFilterFunctionN)(const struct RTCFilterFunctionNArguments* args);

/* Intersection context passed to intersect/occluded calls */
struct RTCIntersectContext
{
  enum RTCIntersectContextFlags flags;               // intersection flags
  RTCFilterFunctionN filter;                         // filter function to execute
  unsigned int instID[RTC_MAX_INSTANCE_LEVEL_COUNT]; // will be set to geomID of instance when instance is entered
};

/* Initializes an intersection context. */
RTC_FORCEINLINE void rtcInitIntersectContext(struct RTCIntersectContext* context)
{
  context->flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
  context->filter = NULL;
  context->instID[0] = RTC_INVALID_GEOMETRY_ID;
}
  
RTC_NAMESPACE_END
