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
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/detail/type_deduction.h>
#include <thrust/type_traits/integer_sequence.h>

#include <tuple>

THRUST_NAMESPACE_BEGIN

template <typename Tuple, std::size_t... Is>
auto tuple_subset(Tuple&& t, index_sequence<Is...>)
THRUST_DECLTYPE_RETURNS(std::make_tuple(std::get<Is>(THRUST_FWD(t))...));

namespace detail
{

template <typename Tuple, typename F, std::size_t... Is>
void tuple_for_each_impl(Tuple&& t, F&& f, index_sequence<Is...>)
{
  auto l = { (f(std::get<Is>(t)), 0)... };
  THRUST_UNUSED_VAR(l);
}

template <typename Tuple, typename F, std::size_t... Is>
auto tuple_transform_impl(Tuple&& t, F&& f, index_sequence<Is...>)
THRUST_DECLTYPE_RETURNS(std::make_tuple(f(std::get<Is>(t))...));

} // namespace detail

template <typename... Ts, typename F>
auto tuple_for_each(std::tuple<Ts...>& t, F&& f)
THRUST_DECLTYPE_RETURNS(
  detail::tuple_for_each_impl(
    t
  , THRUST_FWD(f)
  , make_index_sequence<sizeof...(Ts)>{}
  )
);
template <typename... Ts, typename F>
auto tuple_for_each(std::tuple<Ts...> const& t, F&& f)
THRUST_DECLTYPE_RETURNS(
  detail::tuple_for_each_impl(
    t
  , THRUST_FWD(f)
  , make_index_sequence<sizeof...(Ts)>{}
  )
);
template <typename... Ts, typename F>
auto tuple_for_each(std::tuple<Ts...>&& t, F&& f)
THRUST_DECLTYPE_RETURNS(
  detail::tuple_for_each_impl(
    std::move(t)
  , THRUST_FWD(f)
  , make_index_sequence<sizeof...(Ts)>{}
  )
);

template <typename... Ts, typename F>
auto tuple_transform(std::tuple<Ts...>& t, F&& f)
THRUST_DECLTYPE_RETURNS(
  detail::tuple_transform_impl(
    t
  , THRUST_FWD(f)
  , make_index_sequence<sizeof...(Ts)>{}
  )
);
template <typename... Ts, typename F>
auto tuple_transform(std::tuple<Ts...> const& t, F&& f)
THRUST_DECLTYPE_RETURNS(
  detail::tuple_transform_impl(
    t
  , THRUST_FWD(f)
  , make_index_sequence<sizeof...(Ts)>{}
  )
);
template <typename... Ts, typename F>
auto tuple_transform(std::tuple<Ts...>&& t, F&& f)
THRUST_DECLTYPE_RETURNS(
  detail::tuple_transform_impl(
    std::move(t)
  , THRUST_FWD(f)
  , make_index_sequence<sizeof...(Ts)>{}
  )
);

THRUST_NAMESPACE_END

#endif // THRUST_CPP_DIALECT >= 2011

