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

THRUST_NAMESPACE_BEGIN

namespace random
{

namespace detail
{

template<typename T, int w, int i = w-1>
  struct linear_feedback_shift_engine_wordmask
{
  static const T value =
    (T(1u) << i) |
    linear_feedback_shift_engine_wordmask<T, w, i-1>::value;
}; // end linear_feedback_shift_engine_wordmask

template<typename T, int w>
  struct linear_feedback_shift_engine_wordmask<T, w, 0>
{
  static const T value = 0;
}; // end linear_feedback_shift_engine_wordmask

} // end detail

} // end random

THRUST_NAMESPACE_END

