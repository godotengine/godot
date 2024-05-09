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

__thrust_exec_check_disable__
template<typename Assignable1, typename Assignable2>
__host__ __device__
inline void swap(Assignable1 &a, Assignable2 &b)
{
  Assignable1 temp = a;
  a = b;
  b = temp;
} // end swap()

THRUST_NAMESPACE_END

