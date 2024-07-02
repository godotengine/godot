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

#pragma once

// reserve 0 for undefined
#define THRUST_DEVICE_SYSTEM_CUDA    1
#define THRUST_DEVICE_SYSTEM_OMP     2
#define THRUST_DEVICE_SYSTEM_TBB     3
#define THRUST_DEVICE_SYSTEM_CPP     4

#ifndef THRUST_DEVICE_SYSTEM
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA
#endif // THRUST_DEVICE_SYSTEM

#ifdef THRUST_DEVICE_BACKEND
#  error THRUST_DEVICE_BACKEND is no longer supported; use THRUST_DEVICE_SYSTEM instead.
#endif // THRUST_DEVICE_BACKEND

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#define __THRUST_DEVICE_SYSTEM_NAMESPACE cuda
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#define __THRUST_DEVICE_SYSTEM_NAMESPACE omp
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
#define __THRUST_DEVICE_SYSTEM_NAMESPACE tbb
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CPP
#define __THRUST_DEVICE_SYSTEM_NAMESPACE cpp
#endif

#define __THRUST_DEVICE_SYSTEM_ROOT thrust/system/__THRUST_DEVICE_SYSTEM_NAMESPACE

