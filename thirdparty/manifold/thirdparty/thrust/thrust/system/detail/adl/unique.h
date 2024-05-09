/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a fill of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

// the purpose of this header is to #include the unique.h header
// of the sequential, host, and device systems. It should be #included in any
// code which uses adl to dispatch unique

#include <thrust/system/detail/sequential/unique.h>

// SCons can't see through the #defines below to figure out what this header
// includes, so we fake it out by specifying all possible files we might end up
// including inside an #if 0.
#if 0
#include <thrust/system/cpp/detail/unique.h>
#include <thrust/system/cuda/detail/unique.h>
#include <thrust/system/omp/detail/unique.h>
#include <thrust/system/tbb/detail/unique.h>
#endif

#define __THRUST_HOST_SYSTEM_UNIQUE_HEADER <__THRUST_HOST_SYSTEM_ROOT/detail/unique.h>
#include __THRUST_HOST_SYSTEM_UNIQUE_HEADER
#undef __THRUST_HOST_SYSTEM_UNIQUE_HEADER

#define __THRUST_DEVICE_SYSTEM_UNIQUE_HEADER <__THRUST_DEVICE_SYSTEM_ROOT/detail/unique.h>
#include __THRUST_DEVICE_SYSTEM_UNIQUE_HEADER
#undef __THRUST_DEVICE_SYSTEM_UNIQUE_HEADER

