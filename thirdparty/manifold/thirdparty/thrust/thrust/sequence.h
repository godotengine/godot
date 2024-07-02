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


/*! \file sequence.h
 *  \brief Fills a range with a sequence of numbers
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup transformations
 *  \{
 */


/*! \p sequence fills the range <tt>[first, last)</tt> with a sequence of numbers.
 *
 *  For each iterator \c i in the range <tt>[first, last)</tt>, this version of 
 *  \p sequence performs the assignment <tt>*i =  (i - first)</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and if \c x and \c y are objects of \c ForwardIterator's \c value_type, then <tt>x + y</tt> is defined,
 *          and if \c T is \p ForwardIterator's \c value_type, then <tt>T(0)</tt> is defined.
 *
 *  The following code snippet demonstrates how to use \p sequence to fill a range
 *  with a sequence of numbers using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/sequence.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 10;
 *  int A[N];
 *  thrust::sequence(thrust::host, A, A + 10);
 *  // A is now {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
 *  \endcode
 *
 *  \note Unlike the similar C++ STL function \c std::iota, \p sequence offers no
 *        guarantee on order of execution.
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/iota
 */
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  void sequence(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last);


/*! \p sequence fills the range <tt>[first, last)</tt> with a sequence of numbers.
 *
 *  For each iterator \c i in the range <tt>[first, last)</tt>, this version of 
 *  \p sequence performs the assignment <tt>*i =  (i - first)</tt>.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and if \c x and \c y are objects of \c ForwardIterator's \c value_type, then <tt>x + y</tt> is defined,
 *          and if \c T is \p ForwardIterator's \c value_type, then <tt>T(0)</tt> is defined.
 *
 *  The following code snippet demonstrates how to use \p sequence to fill a range
 *  with a sequence of numbers.
 *
 *  \code
 *  #include <thrust/sequence.h>
 *  ...
 *  const int N = 10;
 *  int A[N];
 *  thrust::sequence(A, A + 10);
 *  // A is now {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
 *  \endcode
 *
 *  \note Unlike the similar C++ STL function \c std::iota, \p sequence offers no
 *        guarantee on order of execution.
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/iota
 */
template<typename ForwardIterator>
  void sequence(ForwardIterator first,
                ForwardIterator last);


/*! \p sequence fills the range <tt>[first, last)</tt> with a sequence of numbers.
 *
 *  For each iterator \c i in the range <tt>[first, last)</tt>, this version of 
 *  \p sequence performs the assignment <tt>*i =  init + (i - first)</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param init The first value of the sequence of numbers.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and if \c x and \c y are objects of \c ForwardIterator's \c value_type, then <tt>x + y</tt> is defined,
 *          and if \c T is \p ForwardIterator's \c value_type, then <tt>T(0)</tt> is defined.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and \p T is convertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p sequence to fill a range
 *  with a sequence of numbers starting from the value 1 using the \p thrust::host execution
 *  policy for parallelization:
 *
 *  \code
 *  #include <thrust/sequence.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 10;
 *  int A[N];
 *  thrust::sequence(thrust::host, A, A + 10, 1);
 *  // A is now {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 *  \endcode
 *
 *  \note Unlike the similar C++ STL function \c std::iota, \p sequence offers no
 *        guarantee on order of execution.
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/iota
 */
template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
  void sequence(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                T init);


/*! \p sequence fills the range <tt>[first, last)</tt> with a sequence of numbers.
 *
 *  For each iterator \c i in the range <tt>[first, last)</tt>, this version of 
 *  \p sequence performs the assignment <tt>*i =  init + (i - first)</tt>.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param init The first value of the sequence of numbers.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and if \c x and \c y are objects of \c ForwardIterator's \c value_type, then <tt>x + y</tt> is defined,
 *          and if \c T is \p ForwardIterator's \c value_type, then <tt>T(0)</tt> is defined.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and \p T is convertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p sequence to fill a range
 *  with a sequence of numbers starting from the value 1.
 *
 *  \code
 *  #include <thrust/sequence.h>
 *  ...
 *  const int N = 10;
 *  int A[N];
 *  thrust::sequence(A, A + 10, 1);
 *  // A is now {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 *  \endcode
 *
 *  \note Unlike the similar C++ STL function \c std::iota, \p sequence offers no
 *        guarantee on order of execution.
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/iota
 */
template<typename ForwardIterator, typename T>
  void sequence(ForwardIterator first,
                ForwardIterator last,
                T init);


/*! \p sequence fills the range <tt>[first, last)</tt> with a sequence of numbers.
 *
 *  For each iterator \c i in the range <tt>[first, last)</tt>, this version of 
 *  \p sequence performs the assignment <tt>*i =  init + step * (i - first)</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param init The first value of the sequence of numbers
 *  \param step The difference between consecutive elements.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and if \c x and \c y are objects of \c ForwardIterator's \c value_type, then <tt>x + y</tt> is defined,
 *          and if \c T is \p ForwardIterator's \c value_type, then <tt>T(0)</tt> is defined.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and \p T is convertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p sequence to fill a range
 *  with a sequence of numbers starting from the value 1 with a step size of 3 using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/sequence.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 10;
 *  int A[N];
 *  thrust::sequence(thrust::host, A, A + 10, 1, 3);
 *  // A is now {1, 4, 7, 10, 13, 16, 19, 22, 25, 28}
 *  \endcode
 *
 *  \note Unlike the similar C++ STL function \c std::iota, \p sequence offers no
 *        guarantee on order of execution.
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/iota
 */
template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
  void sequence(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                T init,
                T step);


/*! \p sequence fills the range <tt>[first, last)</tt> with a sequence of numbers.
 *
 *  For each iterator \c i in the range <tt>[first, last)</tt>, this version of 
 *  \p sequence performs the assignment <tt>*i =  init + step * (i - first)</tt>.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param init The first value of the sequence of numbers
 *  \param step The difference between consecutive elements.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable,
 *          and if \c x and \c y are objects of \c ForwardIterator's \c value_type, then <tt>x + y</tt> is defined,
 *          and if \c T is \p ForwardIterator's \c value_type, then <tt>T(0)</tt> is defined.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and \p T is convertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p sequence to fill a range
 *  with a sequence of numbers starting from the value 1 with a step size of 3.
 *
 *  \code
 *  #include <thrust/sequence.h>
 *  ...
 *  const int N = 10;
 *  int A[N];
 *  thrust::sequence(A, A + 10, 1, 3);
 *  // A is now {1, 4, 7, 10, 13, 16, 19, 22, 25, 28}
 *  \endcode
 *
 *  \note Unlike the similar C++ STL function \c std::iota, \p sequence offers no
 *        guarantee on order of execution.
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/iota
 */
template<typename ForwardIterator, typename T>
  void sequence(ForwardIterator first,
                ForwardIterator last,
                T init,
                T step);


/*! \} // end transformations
 */

THRUST_NAMESPACE_END

#include <thrust/detail/sequence.inl>

