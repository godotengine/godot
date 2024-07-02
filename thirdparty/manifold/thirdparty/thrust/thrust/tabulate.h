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


/*! \file tabulate.h
 *  \brief Fills a range with the tabulation of a function
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup transformations
 *  \{
 */


/*! \p tabulate fills the range <tt>[first, last)</tt> with the value of a function applied to each
 *     element's index.
 *
 *  For each iterator \c i in the range <tt>[first, last)</tt>, \p tabulate performs the assignment
 *  <tt>*i = unary_op(i - first)</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range.
 *  \param last The end of the range.
 *  \param unary_op The unary operation to apply.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and if \c x and \c y are objects of \c ForwardIterator's \c value_type, then <tt>x + y</tt> is defined,
 *          and if \c T is \p ForwardIterator's \c value_type, then <tt>T(0)</tt> is defined.
 *  \tparam UnaryOperation is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>
 *                         and \c UnaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p tabulate to generate the first \c n non-positive integers
 *  using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/tabulate.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 10;
 *  int A[N];
 *  thrust::tabulate(thrust::host, A, A + 10, thrust::negate<int>());
 *  // A is now {0, -1, -2, -3, -4, -5, -6, -7, -8, -9}
 *  \endcode
 *
 *  \see thrust::fill
 *  \see thrust::generate
 *  \see thrust::sequence
 */
template<typename DerivedPolicy, typename ForwardIterator, typename UnaryOperation>
__host__ __device__
  void tabulate(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op);


/*! \p tabulate fills the range <tt>[first, last)</tt> with the value of a function applied to each
 *     element's index.
 *
 *  For each iterator \c i in the range <tt>[first, last)</tt>, \p tabulate performs the assignment
 *  <tt>*i = unary_op(i - first)</tt>.
 *
 *  \param first The beginning of the range.
 *  \param last The end of the range.
 *  \param unary_op The unary operation to apply.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and if \c x and \c y are objects of \c ForwardIterator's \c value_type, then <tt>x + y</tt> is defined,
 *          and if \c T is \p ForwardIterator's \c value_type, then <tt>T(0)</tt> is defined.
 *  \tparam UnaryOperation is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>
 *                         and \c UnaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p tabulate to generate the first \c n non-positive integers:
 *
 *  \code
 *  #include <thrust/tabulate.h>
 *  #include <thrust/functional.h>
 *  ...
 *  const int N = 10;
 *  int A[N];
 *  thrust::tabulate(A, A + 10, thrust::negate<int>());
 *  // A is now {0, -1, -2, -3, -4, -5, -6, -7, -8, -9}
 *  \endcode
 *
 *  \see thrust::fill
 *  \see thrust::generate
 *  \see thrust::sequence
 */
template<typename ForwardIterator, typename UnaryOperation>
  void tabulate(ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op);


/*! \} // end transformations
 */

THRUST_NAMESPACE_END

#include <thrust/detail/tabulate.inl>
