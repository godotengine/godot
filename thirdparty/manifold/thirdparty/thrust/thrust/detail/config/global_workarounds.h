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

#include <thrust/detail/config/compiler.h>

// XXX workaround gcc 4.8+'s complaints about unused local typedefs by silencing them globally
#if defined(THRUST_GCC_VERSION) && (THRUST_GCC_VERSION >= 40800)
#  if defined(__NVCC__) && (CUDART_VERSION >= 6000)
#    pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#  endif // nvcc & cuda 6+
#endif // gcc 4.8

