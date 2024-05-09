/*
 *  Copyright 2021 NVIDIA Corporation
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

// When a compiler uses Thrust as part of its implementation of Standard C++
// algorithms, a cycle of included files may result when Thrust code tries to
// use a standard algorithm.  Having a macro that is defined only when Thrust
// is including an algorithms-related header gives the compiler a chance to
// detect and break the cycle of includes.

#define THRUST_INCLUDING_ALGORITHMS_HEADER
#include <numeric>
#undef  THRUST_INCLUDING_ALGORITHMS_HEADER
