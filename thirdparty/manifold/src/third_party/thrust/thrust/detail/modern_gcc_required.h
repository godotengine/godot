/*
 *  Copyright 2018 NVIDIA Corporation
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

#include <thrust/detail/config/cpp_dialect.h>

#ifndef THRUST_MODERN_GCC_REQUIRED_NO_ERROR
#  if defined(THRUST_GCC_VERSION) && !defined(THRUST_MODERN_GCC)
#    error GCC 5 or later is required for this Thrust feature; please upgrade your compiler.
#  endif
#endif

