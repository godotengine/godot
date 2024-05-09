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
#include <thrust/detail/functional/actor.h>
#include <thrust/detail/functional/composite.h>
#include <thrust/detail/functional/operators/operator_adaptors.h>
#include <thrust/functional.h>

THRUST_NAMESPACE_BEGIN

// XXX WAR circular inclusion with this forward declaration
template<typename,typename,typename> struct binary_function;

namespace detail
{
namespace functional
{

// XXX WAR circular inclusion with this forward declaration
template<typename> struct as_actor;

// there's no standard assign functional, so roll an ad hoc one here
struct assign
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) = THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) = THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) = THRUST_FWD(t2);
  }
};

template<typename Eval, typename T>
  struct assign_result
{
  typedef actor<
    composite<
      transparent_binary_operator<assign>,
      actor<Eval>,
      typename as_actor<T>::type
    >
  > type;
}; // end assign_result

template<typename Eval, typename T>
  __host__ __device__
    typename assign_result<Eval,T>::type
      do_assign(const actor<Eval> &_1, const T &_2)
{
  return compose(transparent_binary_operator<assign>(),
                 _1,
                 as_actor<T>::convert(_2));
} // end do_assign()

} // end functional
} // end detail
THRUST_NAMESPACE_END

