/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// The purpose of this header is to #include the async/sort.h header of the
// sequential, host, and device systems. It should be #included in any code
// which uses ADL to dispatch async sort.

#pragma once

#include <thrust/detail/config.h>

//#include <thrust/system/detail/sequential/async/sort.h>

//#define __THRUST_HOST_SYSTEM_ASYNC_SORT_HEADER <__THRUST_HOST_SYSTEM_ROOT/detail/async/sort.h>
//#include __THRUST_HOST_SYSTEM_ASYNC_SORT_HEADER
//#undef __THRUST_HOST_SYSTEM_ASYNC_SORT_HEADER

#define __THRUST_DEVICE_SYSTEM_ASYNC_SORT_HEADER <__THRUST_DEVICE_SYSTEM_ROOT/detail/async/sort.h>
#include __THRUST_DEVICE_SYSTEM_ASYNC_SORT_HEADER
#undef __THRUST_DEVICE_SYSTEM_ASYNC_SORT_HEADER

