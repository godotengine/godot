/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

#include <thrust/functional.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

template<typename Operation>
  struct unary_traits_imp;

template<typename Operation>
  struct unary_traits_imp<Operation*>
{
  typedef Operation                         function_type;
  typedef const function_type &             param_type;
  typedef typename Operation::result_type   result_type;
  typedef typename Operation::argument_type argument_type;
}; // end unary_traits_imp

template<typename Result, typename Argument>
  struct unary_traits_imp<Result(*)(Argument)>
{
  typedef Result   (*function_type)(Argument);
  typedef Result   (*param_type)(Argument);
  typedef Result   result_type;
  typedef Argument argument_type;
}; // end unary_traits_imp

template<typename Operation>
  struct binary_traits_imp;

template<typename Operation>
  struct binary_traits_imp<Operation*>
{
  typedef Operation                                function_type;
  typedef const function_type &                    param_type;
  typedef typename Operation::result_type          result_type;
  typedef typename Operation::first_argument_type  first_argument_type;
  typedef typename Operation::second_argument_type second_argument_type;
}; // end binary_traits_imp

template<typename Result, typename Argument1, typename Argument2>
  struct binary_traits_imp<Result(*)(Argument1, Argument2)>
{
  typedef Result (*function_type)(Argument1, Argument2);
  typedef Result (*param_type)(Argument1, Argument2);
  typedef Result result_type;
  typedef Argument1 first_argument_type;
  typedef Argument2 second_argument_type;
}; // end binary_traits_imp

} // end detail

template<typename Operation>
  struct unary_traits
{
  typedef typename detail::unary_traits_imp<Operation*>::function_type function_type;
  typedef typename detail::unary_traits_imp<Operation*>::param_type    param_type;
  typedef typename detail::unary_traits_imp<Operation*>::result_type   result_type;
  typedef typename detail::unary_traits_imp<Operation*>::argument_type argument_type;
}; // end unary_traits

template<typename Result, typename Argument>
  struct unary_traits<Result(*)(Argument)>
{
  typedef Result   (*function_type)(Argument);
  typedef Result   (*param_type)(Argument);
  typedef Result   result_type;
  typedef Argument argument_type;
}; // end unary_traits

template<typename Operation>
  struct binary_traits
{
  typedef typename detail::binary_traits_imp<Operation*>::function_type        function_type;
  typedef typename detail::binary_traits_imp<Operation*>::param_type           param_type;
  typedef typename detail::binary_traits_imp<Operation*>::result_type          result_type;
  typedef typename detail::binary_traits_imp<Operation*>::first_argument_type  first_argument_type;
  typedef typename detail::binary_traits_imp<Operation*>::second_argument_type second_argument_type;
}; // end binary_traits

template<typename Result, typename Argument1, typename Argument2>
  struct binary_traits<Result(*)(Argument1, Argument2)>
{
  typedef Result (*function_type)(Argument1, Argument2);
  typedef Result (*param_type)(Argument1, Argument2);
  typedef Result result_type;
  typedef Argument1 first_argument_type;
  typedef Argument2 second_argument_type;
}; // end binary_traits

template<typename Predicate>
  __host__ __device__
  unary_negate<Predicate> not1(const Predicate &pred)
{
  return unary_negate<Predicate>(pred);
} // end not1()

template<typename BinaryPredicate>
  __host__ __device__
  binary_negate<BinaryPredicate> not2(const BinaryPredicate &pred)
{
  return binary_negate<BinaryPredicate>(pred);
} // end not2()

THRUST_NAMESPACE_END
