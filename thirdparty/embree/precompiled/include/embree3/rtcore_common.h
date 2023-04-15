// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <sys/types.h>
#include <stdbool.h>

#include "rtcore_config.h"

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
  struct RTCIntersectContext* context;
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
  
#if RTC_MAX_INSTANCE_LEVEL_COUNT > 1
  unsigned int instStackSize;                        // Number of instances currently on the stack.
#endif
  unsigned int instID[RTC_MAX_INSTANCE_LEVEL_COUNT]; // The current stack of instance ids.
  
#if RTC_MIN_WIDTH
  float minWidthDistanceFactor;                      // curve radius is set to this factor times distance to ray origin
#endif
};

/* Initializes an intersection context. */
RTC_FORCEINLINE void rtcInitIntersectContext(struct RTCIntersectContext* context)
{
  unsigned l = 0;
  context->flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
  context->filter = NULL;
  
#if RTC_MAX_INSTANCE_LEVEL_COUNT > 1
  context->instStackSize = 0;
#endif
  for (; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l)
    context->instID[l] = RTC_INVALID_GEOMETRY_ID;
  
#if RTC_MIN_WIDTH
  context->minWidthDistanceFactor = 0.0f;
#endif
}

/* Point query structure for closest point query */
struct RTC_ALIGN(16) RTCPointQuery 
{
  float x;                // x coordinate of the query point
  float y;                // y coordinate of the query point
  float z;                // z coordinate of the query point
  float time;             // time of the point query
  float radius;           // radius of the point query 
};

/* Structure of a packet of 4 query points */
struct RTC_ALIGN(16) RTCPointQuery4
{
  float x[4];                // x coordinate of the query point
  float y[4];                // y coordinate of the query point
  float z[4];                // z coordinate of the query point
  float time[4];             // time of the point query
  float radius[4];           // radius of the point query
};

/* Structure of a packet of 8 query points */
struct RTC_ALIGN(32) RTCPointQuery8
{
  float x[8];                // x coordinate of the query point
  float y[8];                // y coordinate of the query point
  float z[8];                // z coordinate of the query point
  float time[8];             // time of the point query
  float radius[8];           // radius ofr the point query 
};

/* Structure of a packet of 16 query points */
struct RTC_ALIGN(64) RTCPointQuery16
{
  float x[16];                // x coordinate of the query point
  float y[16];                // y coordinate of the query point
  float z[16];                // z coordinate of the query point
  float time[16];             // time of the point quey
  float radius[16];           // radius of the point query
};

struct RTCPointQueryN;

struct RTC_ALIGN(16) RTCPointQueryContext
{
  // accumulated 4x4 column major matrices from world space to instance space.
  // undefined if size == 0.
  float world2inst[RTC_MAX_INSTANCE_LEVEL_COUNT][16]; 

  // accumulated 4x4 column major matrices from instance space to world space.
  // undefined if size == 0.
  float inst2world[RTC_MAX_INSTANCE_LEVEL_COUNT][16]; 

  // instance ids.
  unsigned int instID[RTC_MAX_INSTANCE_LEVEL_COUNT];

  // number of instances currently on the stack.
  unsigned int instStackSize;
};

/* Initializes an intersection context. */
RTC_FORCEINLINE void rtcInitPointQueryContext(struct RTCPointQueryContext* context)
{
  context->instStackSize = 0;
  context->instID[0] = RTC_INVALID_GEOMETRY_ID;
}

struct RTC_ALIGN(16) RTCPointQueryFunctionArguments
{
  // The (world space) query object that was passed as an argument of rtcPointQuery. The
  // radius of the query can be decreased inside the callback to shrink the
  // search domain. Increasing the radius or modifying the time or position of
  // the query results in undefined behaviour.
  struct RTCPointQuery* query;

  // Used for user input/output data. Will not be read or modified internally.
  void* userPtr;

  // primitive and geometry ID of primitive
  unsigned int  primID;        
  unsigned int  geomID;    

  // the context with transformation and instance ID stack
  struct RTCPointQueryContext* context;

  // If the current instance transform M (= context->world2inst[context->instStackSize]) 
  // is a similarity matrix, i.e there is a constant factor similarityScale such that,
  //    for all x,y: dist(Mx, My) = similarityScale * dist(x, y),
  // The similarity scale is 0, if the current instance transform is not a
  // similarity transform and vice versa. The similarity scale allows to compute
  // distance information in instance space and scale the distances into world
  // space by dividing with the similarity scale, for example, to update the
  // query radius. If the current instance transform is not a similarity
  // transform (similarityScale = 0), the distance computation has to be
  // performed in world space to ensure correctness. if there is no instance
  // transform (context->instStackSize == 0), the similarity scale is 1.
  float similarityScale;
};

typedef bool (*RTCPointQueryFunction)(struct RTCPointQueryFunctionArguments* args);
  
RTC_NAMESPACE_END
