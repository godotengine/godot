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

#include <thrust/detail/config.h>

// the purpose of this header is to #include <driver_types.h> without causing
// warnings from redefinitions of __host__ and __device__.
// carefully save their definitions and restore them
// can't tell exactly when push_macro & pop_macro were introduced to gcc; assume 4.5.0


#if !defined(__GNUC__) || ((10000 * __GNUC__ + 100 * __GNUC_MINOR__ + __GNUC_PATCHLEVEL__) >= 40500)
#  ifdef __host__
#    pragma push_macro("__host__")
#    undef __host__
#    define THRUST_HOST_NEEDS_RESTORATION
#  endif
#  ifdef __device__
#    pragma push_macro("__device__")
#    undef __device__
#    define THRUST_DEVICE_NEEDS_RESTORATION
#  endif
#else // GNUC pre 4.5.0
#  if !defined(__DRIVER_TYPES_H__)
#    ifdef __host__
#      undef __host__
#    endif
#    ifdef __device__
#      undef __device__
#    endif
#  endif // __DRIVER_TYPES_H__
#endif // __GNUC__


#include <driver_types.h>


#if !defined(__GNUC__) || ((10000 * __GNUC__ + 100 * __GNUC_MINOR__ + __GNUC_PATCHLEVEL__) >= 40500)
#  ifdef THRUST_HOST_NEEDS_RESTORATION
#    pragma pop_macro("__host__")
#    undef THRUST_HOST_NEEDS_RESTORATION
#  endif
#  ifdef THRUST_DEVICE_NEEDS_RESTORATION
#    pragma pop_macro("__device__")
#    undef THRUST_DEVICE_NEEDS_RESTORATION
#  endif
#endif // __GNUC__

