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

#include "rtcore_device.h"

RTC_NAMESPACE_BEGIN

/* Types of buffers */
enum RTCBufferType
{
  RTC_BUFFER_TYPE_INDEX            = 0,
  RTC_BUFFER_TYPE_VERTEX           = 1,
  RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE = 2,
  RTC_BUFFER_TYPE_NORMAL           = 3,
  RTC_BUFFER_TYPE_TANGENT          = 4,
  RTC_BUFFER_TYPE_NORMAL_DERIVATIVE = 5,

  RTC_BUFFER_TYPE_GRID                 = 8,

  RTC_BUFFER_TYPE_FACE                 = 16,
  RTC_BUFFER_TYPE_LEVEL                = 17,
  RTC_BUFFER_TYPE_EDGE_CREASE_INDEX    = 18,
  RTC_BUFFER_TYPE_EDGE_CREASE_WEIGHT   = 19,
  RTC_BUFFER_TYPE_VERTEX_CREASE_INDEX  = 20,
  RTC_BUFFER_TYPE_VERTEX_CREASE_WEIGHT = 21,
  RTC_BUFFER_TYPE_HOLE                 = 22,

  RTC_BUFFER_TYPE_FLAGS = 32
};

/* Opaque buffer type */
typedef struct RTCBufferTy* RTCBuffer;

/* Creates a new buffer. */
RTC_API RTCBuffer rtcNewBuffer(RTCDevice device, size_t byteSize);

/* Creates a new shared buffer. */
RTC_API RTCBuffer rtcNewSharedBuffer(RTCDevice device, void* ptr, size_t byteSize);

/* Returns a pointer to the buffer data. */
RTC_API void* rtcGetBufferData(RTCBuffer buffer);

/* Retains the buffer (increments the reference count). */
RTC_API void rtcRetainBuffer(RTCBuffer buffer);

/* Releases the buffer (decrements the reference count). */
RTC_API void rtcReleaseBuffer(RTCBuffer buffer);

RTC_NAMESPACE_END
