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


/*! \file find.h
 *  \brief Sequential implementation of find_if. 
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/function.h>
#include <thrust/system/detail/sequential/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename Predicate>
__host__ __device__
InputIterator find_if(execution_policy<DerivedPolicy> &,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<
    Predicate,
    bool
  > wrapped_pred(pred);

  while(first != last)
  {
    if (wrapped_pred(*first))
      return first;

    ++first;
  }

  // return first so zip_iterator works correctly
  return first;
}


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

