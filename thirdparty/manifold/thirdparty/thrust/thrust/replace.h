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


/*! \file replace.h
 *  \brief Functions for replacing elements in a range with a particular value
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup transformations
 *  \addtogroup replacing
 *  \ingroup transformations
 *  \{
 */


/*! \p replace replaces every element in the range [first, last) equal to \p old_value
 *  with \p new_value. That is: for every iterator \c i, if <tt>*i == old_value</tt>
 *  then it performs the <tt>assignment *i = new_value</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence of interest.
 *  \param last The end of the sequence of interest.
 *  \param old_value The value to replace.
 *  \param new_value The new value to replace \p old_value.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable>Assignable">Assignable</a>,
 *          \p T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">EqualityComparable</a>,
 *          objects of \p T may be compared for equality with objects of
 *          \p ForwardIterator's \c value_type,
 *          and \p T is convertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p replace to replace
 *  a value of interest in a \c device_vector with another using the \p thrust::device
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/replace.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *
 *  ...
 *  
 *  thrust::device_vector<int> A(4);
 *  A[0] = 1;
 *  A[1] = 2;
 *  A[2] = 3;
 *  A[3] = 1;
 *
 *  thrust::replace(thrust::device, A.begin(), A.end(), 1, 99);
 *
 *  // A contains [99, 2, 3, 99]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/replace
 *  \see \c replace_if
 *  \see \c replace_copy
 *  \see \c replace_copy_if
 */
template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
  void replace(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
               ForwardIterator first, ForwardIterator last,
               const T &old_value,
               const T &new_value);


/*! \p replace replaces every element in the range [first, last) equal to \p old_value
 *  with \p new_value. That is: for every iterator \c i, if <tt>*i == old_value</tt>
 *  then it performs the <tt>assignment *i = new_value</tt>.
 *
 *  \param first The beginning of the sequence of interest.
 *  \param last The end of the sequence of interest.
 *  \param old_value The value to replace.
 *  \param new_value The new value to replace \p old_value.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable>Assignable">Assignable</a>,
 *          \p T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">EqualityComparable</a>,
 *          objects of \p T may be compared for equality with objects of
 *          \p ForwardIterator's \c value_type,
 *          and \p T is convertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p replace to replace
 *  a value of interest in a \c device_vector with another.
 *
 *  \code
 *  #include <thrust/replace.h>
 *  #include <thrust/device_vector.h>
 *
 *  ...
 *  
 *  thrust::device_vector<int> A(4);
 *  A[0] = 1;
 *  A[1] = 2;
 *  A[2] = 3;
 *  A[3] = 1;
 *
 *  thrust::replace(A.begin(), A.end(), 1, 99);
 *
 *  // A contains [99, 2, 3, 99]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/replace
 *  \see \c replace_if
 *  \see \c replace_copy
 *  \see \c replace_copy_if
 */
template<typename ForwardIterator, typename T>
  void replace(ForwardIterator first, ForwardIterator last, const T &old_value,
               const T &new_value);


/*! \p replace_if replaces every element in the range <tt>[first, last)</tt> for which
 *  \p pred returns \c true with \p new_value. That is: for every iterator \c i, if
 *  <tt>pred(*i)</tt> is \c true then it performs the assignment <tt>*i = new_value</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence of interest.
 *  \param last The end of the sequence of interest.
 *  \param pred The predicate to test on every value of the range <tt>[first,last)</tt>.
 *  \param new_value The new value to replace elements which <tt>pred(*i)</tt> evaluates
 *         to \c true.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          \p ForwardIterator is mutable,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and \p T is convertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p replace_if to replace
 *  a \c device_vector's negative elements with \c 0 using the \p thrust::device execution policy
 *  for parallelization:
 *
 *  \code
 *  #include <thrust/replace.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  struct is_less_than_zero
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x < 0;
 *    }
 *  };
 *
 *  ...
 *  
 *  thrust::device_vector<int> A(4);
 *  A[0] =  1;
 *  A[1] = -3;
 *  A[2] =  2;
 *  A[3] = -1;
 *
 *  is_less_than_zero pred;
 *
 *  thrust::replace_if(thrust::device, A.begin(), A.end(), pred, 0);
 *
 *  // A contains [1, 0, 2, 0]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/replace
 *  \see \c replace
 *  \see \c replace_copy
 *  \see \c replace_copy_if
 */
template<typename DerivedPolicy, typename ForwardIterator, typename Predicate, typename T>
__host__ __device__
  void replace_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                  ForwardIterator first, ForwardIterator last,
                  Predicate pred,
                  const T &new_value);


/*! \p replace_if replaces every element in the range <tt>[first, last)</tt> for which
 *  \p pred returns \c true with \p new_value. That is: for every iterator \c i, if
 *  <tt>pred(*i)</tt> is \c true then it performs the assignment <tt>*i = new_value</tt>.
 *
 *  \param first The beginning of the sequence of interest.
 *  \param last The end of the sequence of interest.
 *  \param pred The predicate to test on every value of the range <tt>[first,last)</tt>.
 *  \param new_value The new value to replace elements which <tt>pred(*i)</tt> evaluates
 *         to \c true.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          \p ForwardIterator is mutable,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and \p T is convertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p replace_if to replace
 *  a \c device_vector's negative elements with \c 0.
 *
 *  \code
 *  #include <thrust/replace.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  struct is_less_than_zero
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x < 0;
 *    }
 *  };
 *
 *  ...
 *  
 *  thrust::device_vector<int> A(4);
 *  A[0] =  1;
 *  A[1] = -3;
 *  A[2] =  2;
 *  A[3] = -1;
 *
 *  is_less_than_zero pred;
 *
 *  thrust::replace_if(A.begin(), A.end(), pred, 0);
 *
 *  // A contains [1, 0, 2, 0]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/replace
 *  \see \c replace
 *  \see \c replace_copy
 *  \see \c replace_copy_if
 */
template<typename ForwardIterator, typename Predicate, typename T>
  void replace_if(ForwardIterator first, ForwardIterator last,
                  Predicate pred,
                  const T &new_value);


/*! \p replace_if replaces every element in the range <tt>[first, last)</tt> for which
 *  <tt>pred(*s)</tt> returns \c true with \p new_value. That is: for every iterator
 *  \c i in the range <tt>[first, last)</tt>, and \c s in the range <tt>[stencil, stencil + (last - first))</tt>,
 *  if <tt>pred(*s)</tt> is \c true then it performs the assignment <tt>*i = new_value</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence of interest.
 *  \param last The end of the sequence of interest.
 *  \param stencil The beginning of the stencil sequence.
 *  \param pred The predicate to test on every value of the range <tt>[first,last)</tt>.
 *  \param new_value The new value to replace elements which <tt>pred(*i)</tt> evaluates
 *         to \c true.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and \p T is convertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p replace_if to replace
 *  a \c device_vector's element with \c 0 when its corresponding stencil element is less than zero
 *  using the \p thrust::device execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/replace.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *
 *  struct is_less_than_zero
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x < 0;
 *    }
 *  };
 *  
 *  ...
 *  
 *  thrust::device_vector<int> A(4);
 *  A[0] =  10;
 *  A[1] =  20;
 *  A[2] =  30;
 *  A[3] =  40;
 *
 *  thrust::device_vector<int> S(4);
 *  S[0] = -1;
 *  S[1] =  0;
 *  S[2] = -1;
 *  S[3] =  0;
 *
 *  is_less_than_zero pred;
 *  thrust::replace_if(thrust::device, A.begin(), A.end(), S.begin(), pred, 0);
 *
 *  // A contains [0, 20, 0, 40]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/replace
 *  \see \c replace
 *  \see \c replace_copy
 *  \see \c replace_copy_if
 */
template<typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
__host__ __device__
  void replace_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                  ForwardIterator first, ForwardIterator last,
                  InputIterator stencil,
                  Predicate pred,
                  const T &new_value);


/*! \p replace_if replaces every element in the range <tt>[first, last)</tt> for which
 *  <tt>pred(*s)</tt> returns \c true with \p new_value. That is: for every iterator
 *  \c i in the range <tt>[first, last)</tt>, and \c s in the range <tt>[stencil, stencil + (last - first))</tt>,
 *  if <tt>pred(*s)</tt> is \c true then it performs the assignment <tt>*i = new_value</tt>.
 *
 *  \param first The beginning of the sequence of interest.
 *  \param last The end of the sequence of interest.
 *  \param stencil The beginning of the stencil sequence.
 *  \param pred The predicate to test on every value of the range <tt>[first,last)</tt>.
 *  \param new_value The new value to replace elements which <tt>pred(*i)</tt> evaluates
 *         to \c true.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and \p T is convertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p replace_if to replace
 *  a \c device_vector's element with \c 0 when its corresponding stencil element is less than zero.
 *
 *  \code
 *  #include <thrust/replace.h>
 *  #include <thrust/device_vector.h>
 *
 *  struct is_less_than_zero
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x < 0;
 *    }
 *  };
 *  
 *  ...
 *  
 *  thrust::device_vector<int> A(4);
 *  A[0] =  10;
 *  A[1] =  20;
 *  A[2] =  30;
 *  A[3] =  40;
 *
 *  thrust::device_vector<int> S(4);
 *  S[0] = -1;
 *  S[1] =  0;
 *  S[2] = -1;
 *  S[3] =  0;
 *
 *  is_less_than_zero pred;
 *  thrust::replace_if(A.begin(), A.end(), S.begin(), pred, 0);
 *
 *  // A contains [0, 20, 0, 40]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/replace
 *  \see \c replace
 *  \see \c replace_copy
 *  \see \c replace_copy_if
 */
template<typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
  void replace_if(ForwardIterator first, ForwardIterator last,
                  InputIterator stencil,
                  Predicate pred,
                  const T &new_value);


/*! \p replace_copy copies elements from the range <tt>[first, last)</tt> to the range
 *  <tt>[result, result + (last-first))</tt>, except that any element equal to \p old_value
 *  is not copied; \p new_value is copied instead.
 *
 *  More precisely, for every integer \c n such that <tt>0 <= n < last-first</tt>, \p replace_copy
 *  performs the assignment <tt>*(result+n) = new_value</tt> if <tt>*(first+n) == old_value</tt>,
 *  and <tt>*(result+n) = *(first+n)</tt> otherwise.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to copy from.
 *  \param last The end of the sequence to copy from.
 *  \param result The beginning of the sequence to copy to.
 *  \param old_value The value to replace.
 *  \param new_value The replacement value for which <tt>*i == old_value</tt> evaluates to \c true.
 *  \return <tt>result + (last-first)</tt>
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          \p T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>,
 *          \p T may be compared for equality with \p InputIterator's \c value_type,
 *          and \p T is convertible to \p OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the ranges <tt>[first, last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  \code
 *  #include <thrust/replace.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> A(4);
 *  A[0] = 1;
 *  A[1] = 2;
 *  A[2] = 3;
 *  A[3] = 1;
 *
 *  thrust::device_vector<int> B(4);
 *
 *  thrust::replace_copy(thrust::device, A.begin(), A.end(), B.begin(), 1, 99);
 *
 *  // B contains [99, 2, 3, 99]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/replace_copy
 *  \see \c copy
 *  \see \c replace
 *  \see \c replace_if
 *  \see \c replace_copy_if
 */
template<typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T>
__host__ __device__
  OutputIterator replace_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                              InputIterator first, InputIterator last,
                              OutputIterator result,
                              const T &old_value,
                              const T &new_value);


/*! \p replace_copy copies elements from the range <tt>[first, last)</tt> to the range
 *  <tt>[result, result + (last-first))</tt>, except that any element equal to \p old_value
 *  is not copied; \p new_value is copied instead.
 *
 *  More precisely, for every integer \c n such that <tt>0 <= n < last-first</tt>, \p replace_copy
 *  performs the assignment <tt>*(result+n) = new_value</tt> if <tt>*(first+n) == old_value</tt>,
 *  and <tt>*(result+n) = *(first+n)</tt> otherwise.
 *
 *  \param first The beginning of the sequence to copy from.
 *  \param last The end of the sequence to copy from.
 *  \param result The beginning of the sequence to copy to.
 *  \param old_value The value to replace.
 *  \param new_value The replacement value for which <tt>*i == old_value</tt> evaluates to \c true.
 *  \return <tt>result + (last-first)</tt>
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          \p T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>,
 *          \p T may be compared for equality with \p InputIterator's \c value_type,
 *          and \p T is convertible to \p OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the ranges <tt>[first, last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  \code
 *  #include <thrust/replace.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> A(4);
 *  A[0] = 1;
 *  A[1] = 2;
 *  A[2] = 3;
 *  A[3] = 1;
 *
 *  thrust::device_vector<int> B(4);
 *
 *  thrust::replace_copy(A.begin(), A.end(), B.begin(), 1, 99);
 *
 *  // B contains [99, 2, 3, 99]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/replace_copy
 *  \see \c copy
 *  \see \c replace
 *  \see \c replace_if
 *  \see \c replace_copy_if
 */
template<typename InputIterator, typename OutputIterator, typename T>
  OutputIterator replace_copy(InputIterator first, InputIterator last,
                              OutputIterator result, const T &old_value,
                              const T &new_value);


/*! \p replace_copy_if copies elements from the range <tt>[first, last)</tt> to the range
 *  <tt>[result, result + (last-first))</tt>, except that any element for which \p pred
 *  is \c true is not copied; \p new_value is copied instead.
 *
 *  More precisely, for every integer \c n such that 0 <= n < last-first,
 *  \p replace_copy_if performs the assignment <tt>*(result+n) = new_value</tt> if
 *  <tt>pred(*(first+n))</tt>, and <tt>*(result+n) = *(first+n)</tt> otherwise.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to copy from.
 *  \param last The end of the sequence to copy from.
 *  \param result The beginning of the sequence to copy to.
 *  \param pred The predicate to test on every value of the range <tt>[first,last)</tt>.
 *  \param new_value The replacement value to assign <tt>pred(*i)</tt> evaluates to \c true.
 *  \return <tt>result + (last-first)</tt>
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and \p T is convertible to \p OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the ranges <tt>[first, last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  \code
 *  #include <thrust/replace.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *
 *  struct is_less_than_zero
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x < 0;
 *    }
 *  };
 *
 *  ...
 *  
 *  thrust::device_vector<int> A(4);
 *  A[0] =  1;
 *  A[1] = -3;
 *  A[2] =  2;
 *  A[3] = -1;
 
 *  thrust::device_vector<int> B(4);
 *  is_less_than_zero pred;
 *
 *  thrust::replace_copy_if(thrust::device, A.begin(), A.end(), B.begin(), pred, 0);
 *
 *  // B contains [1, 0, 2, 0]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/replace_copy
 *  \see \c replace
 *  \see \c replace_if
 *  \see \c replace_copy
 */
template<typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate, typename T>
__host__ __device__
  OutputIterator replace_copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                 InputIterator first, InputIterator last,
                                 OutputIterator result,
                                 Predicate pred,
                                 const T &new_value);


/*! \p replace_copy_if copies elements from the range <tt>[first, last)</tt> to the range
 *  <tt>[result, result + (last-first))</tt>, except that any element for which \p pred
 *  is \c true is not copied; \p new_value is copied instead.
 *
 *  More precisely, for every integer \c n such that 0 <= n < last-first,
 *  \p replace_copy_if performs the assignment <tt>*(result+n) = new_value</tt> if
 *  <tt>pred(*(first+n))</tt>, and <tt>*(result+n) = *(first+n)</tt> otherwise.
 *
 *  \param first The beginning of the sequence to copy from.
 *  \param last The end of the sequence to copy from.
 *  \param result The beginning of the sequence to copy to.
 *  \param pred The predicate to test on every value of the range <tt>[first,last)</tt>.
 *  \param new_value The replacement value to assign <tt>pred(*i)</tt> evaluates to \c true.
 *  \return <tt>result + (last-first)</tt>
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and \p T is convertible to \p OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the ranges <tt>[first, last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  \code
 *  #include <thrust/replace.h>
 *  #include <thrust/device_vector.h>
 *
 *  struct is_less_than_zero
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x < 0;
 *    }
 *  };
 *
 *  ...
 *  
 *  thrust::device_vector<int> A(4);
 *  A[0] =  1;
 *  A[1] = -3;
 *  A[2] =  2;
 *  A[3] = -1;
 
 *  thrust::device_vector<int> B(4);
 *  is_less_than_zero pred;
 *
 *  thrust::replace_copy_if(A.begin(), A.end(), B.begin(), pred, 0);
 *
 *  // B contains [1, 0, 2, 0]
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/replace_copy
 *  \see \c replace
 *  \see \c replace_if
 *  \see \c replace_copy
 */
template<typename InputIterator, typename OutputIterator, typename Predicate, typename T>
  OutputIterator replace_copy_if(InputIterator first, InputIterator last,
                                 OutputIterator result,
                                 Predicate pred,
                                 const T &new_value);


/*! This version of \p replace_copy_if copies elements from the range <tt>[first, last)</tt> to the range
 *  <tt>[result, result + (last-first))</tt>, except that any element whose corresponding stencil
 *  element causes \p pred to be \c true is not copied; \p new_value is copied instead.
 *
 *  More precisely, for every integer \c n such that <tt>0 <= n < last-first</tt>,
 *  \p replace_copy_if performs the assignment <tt>*(result+n) = new_value</tt> if
 *  <tt>pred(*(stencil+n))</tt>, and <tt>*(result+n) = *(first+n)</tt> otherwise.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to copy from.
 *  \param last The end of the sequence to copy from.
 *  \param stencil The beginning of the stencil sequence.
 *  \param result The beginning of the sequence to copy to.
 *  \param pred The predicate to test on every value of the range <tt>[stencil, stencil + (last - first))</tt>.
 *  \param new_value The replacement value to assign when <tt>pred(*s)</tt> evaluates to \c true. 
 *  \return <tt>result + (last-first)</tt>
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \p InputIterator2's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and \p T is convertible to \p OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the ranges <tt>[first, last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *  \pre \p stencil may equal \p result, but the ranges <tt>[stencil, stencil + (last - first))</tt> and <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  \code
 *  #include <thrust/replace.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *
 *  struct is_less_than_zero
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x < 0;
 *    }
 *  };
 *  
 *  ...
 *  
 *  thrust::device_vector<int> A(4);
 *  A[0] =  10;
 *  A[1] =  20;
 *  A[2] =  30;
 *  A[3] =  40;
 *
 *  thrust::device_vector<int> S(4);
 *  S[0] = -1;
 *  S[1] =  0;
 *  S[2] = -1;
 *  S[3] =  0;
 *
 *  thrust::device_vector<int> B(4);
 *  is_less_than_zero pred;
 *
 *  thrust::replace_if(thrust::device, A.begin(), A.end(), S.begin(), B.begin(), pred, 0);
 *
 *  // B contains [0, 20, 0, 40]
 *  \endcode
 *
 *  \see \c replace_copy
 *  \see \c replace_if
 */
template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
__host__ __device__
  OutputIterator replace_copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                 InputIterator1 first, InputIterator1 last,
                                 InputIterator2 stencil,
                                 OutputIterator result,
                                 Predicate pred,
                                 const T &new_value);


/*! This version of \p replace_copy_if copies elements from the range <tt>[first, last)</tt> to the range
 *  <tt>[result, result + (last-first))</tt>, except that any element whose corresponding stencil
 *  element causes \p pred to be \c true is not copied; \p new_value is copied instead.
 *
 *  More precisely, for every integer \c n such that <tt>0 <= n < last-first</tt>,
 *  \p replace_copy_if performs the assignment <tt>*(result+n) = new_value</tt> if
 *  <tt>pred(*(stencil+n))</tt>, and <tt>*(result+n) = *(first+n)</tt> otherwise.
 *
 *  \param first The beginning of the sequence to copy from.
 *  \param last The end of the sequence to copy from.
 *  \param stencil The beginning of the stencil sequence.
 *  \param result The beginning of the sequence to copy to.
 *  \param pred The predicate to test on every value of the range <tt>[stencil, stencil + (last - first))</tt>.
 *  \param new_value The replacement value to assign when <tt>pred(*s)</tt> evaluates to \c true. 
 *  \return <tt>result + (last-first)</tt>
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \p InputIterator2's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and \p T is convertible to \p OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the ranges <tt>[first, last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *  \pre \p stencil may equal \p result, but the ranges <tt>[stencil, stencil + (last - first))</tt> and <tt>[result, result + (last - first))</tt> shall not overlap otherwise.
 *
 *  \code
 *  #include <thrust/replace.h>
 *  #include <thrust/device_vector.h>
 *
 *  struct is_less_than_zero
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x < 0;
 *    }
 *  };
 *  
 *  ...
 *  
 *  thrust::device_vector<int> A(4);
 *  A[0] =  10;
 *  A[1] =  20;
 *  A[2] =  30;
 *  A[3] =  40;
 *
 *  thrust::device_vector<int> S(4);
 *  S[0] = -1;
 *  S[1] =  0;
 *  S[2] = -1;
 *  S[3] =  0;
 *
 *  thrust::device_vector<int> B(4);
 *  is_less_than_zero pred;
 *
 *  thrust::replace_if(A.begin(), A.end(), S.begin(), B.begin(), pred, 0);
 *
 *  // B contains [0, 20, 0, 40]
 *  \endcode
 *
 *  \see \c replace_copy
 *  \see \c replace_if
 */
template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
  OutputIterator replace_copy_if(InputIterator1 first, InputIterator1 last,
                                 InputIterator2 stencil,
                                 OutputIterator result,
                                 Predicate pred,
                                 const T &new_value);


/*! \} // end replacing
 *  \} // transformations
 */

THRUST_NAMESPACE_END

#include <thrust/detail/replace.inl>
