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


/*! \file remove.h
 *  \brief Functions for removing elements from a range
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup stream_compaction Stream Compaction
 *  \ingroup reordering
 *  \{
 *
 */


/*! \p remove removes from the range <tt>[first, last)</tt> all elements that are
 *  equal to \p value. That is, \p remove returns an iterator \p new_last such
 *  that the range <tt>[first, new_last)</tt> contains no elements equal to
 *  \p value. The iterators in the range <tt>[new_first,last)</tt> are all still
 *  dereferenceable, but the elements that they point to are unspecified. \p remove
 *  is stable, meaning that the relative order of elements that are not equal to
 *  \p value is unchanged.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param value The value to remove from the range <tt>[first, last)</tt>.
 *         Elements which are equal to value are removed from the sequence.
 *  \return A \p ForwardIterator pointing to the end of the resulting range of
 *          elements which are not equal to \p value.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>,
 *          and objects of type \p T can be compared for equality with objects of \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p remove to remove a number
 *  of interest from a range using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/remove.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {3, 1, 4, 1, 5, 9};
 *  int *new_end = thrust::remove(A, A + N, 1);
 *  // The first four values of A are now {3, 4, 5, 9}
 *  // Values beyond new_end are unspecified
 *  \endcode
 *
 *  \note The meaning of "removal" is somewhat subtle. \p remove does not destroy any
 *  iterators, and does not change the distance between \p first and \p last.
 *  (There's no way that it could do anything of the sort.) So, for example, if
 *  \c V is a device_vector, <tt>remove(V.begin(), V.end(), 0)</tt> does not
 *  change <tt>V.size()</tt>: \c V will contain just as many elements as it did
 *  before. \p remove returns an iterator that points to the end of the resulting
 *  range after elements have been removed from it; it follows that the elements
 *  after that iterator are of no interest, and may be discarded. If you are
 *  removing elements from a
 *  <a href="https://en.cppreference.com/w/cpp/container">Sequence</a>, you may
 *  simply erase them. That is, a reasonable way of removing elements from a
 *  <a href="https://en.cppreference.com/w/cpp/container">Sequence</a> is
 *  <tt>S.erase(remove(S.begin(), S.end(), x), S.end())</tt>.
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/remove
 *  \see remove_if
 *  \see remove_copy
 *  \see remove_copy_if
 */
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename T>
__host__ __device__
  ForwardIterator remove(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                         ForwardIterator first,
                         ForwardIterator last,
                         const T &value);


/*! \p remove removes from the range <tt>[first, last)</tt> all elements that are
 *  equal to \p value. That is, \p remove returns an iterator \p new_last such
 *  that the range <tt>[first, new_last)</tt> contains no elements equal to
 *  \p value. The iterators in the range <tt>[new_first,last)</tt> are all still
 *  dereferenceable, but the elements that they point to are unspecified. \p remove
 *  is stable, meaning that the relative order of elements that are not equal to
 *  \p value is unchanged.
 *
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param value The value to remove from the range <tt>[first, last)</tt>.
 *         Elements which are equal to value are removed from the sequence.
 *  \return A \p ForwardIterator pointing to the end of the resulting range of
 *          elements which are not equal to \p value.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>,
 *          and objects of type \p T can be compared for equality with objects of \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p remove to remove a number
 *  of interest from a range.
 *
 *  \code
 *  #include <thrust/remove.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {3, 1, 4, 1, 5, 9};
 *  int *new_end = thrust::remove(A, A + N, 1);
 *  // The first four values of A are now {3, 4, 5, 9}
 *  // Values beyond new_end are unspecified
 *  \endcode
 *
 *  \note The meaning of "removal" is somewhat subtle. \p remove does not destroy any
 *  iterators, and does not change the distance between \p first and \p last.
 *  (There's no way that it could do anything of the sort.) So, for example, if
 *  \c V is a device_vector, <tt>remove(V.begin(), V.end(), 0)</tt> does not
 *  change <tt>V.size()</tt>: \c V will contain just as many elements as it did
 *  before. \p remove returns an iterator that points to the end of the resulting
 *  range after elements have been removed from it; it follows that the elements
 *  after that iterator are of no interest, and may be discarded. If you are
 *  removing elements from a
 *  <a href="https://en.cppreference.com/w/cpp/container">Sequence</a>, you may
 *  simply erase them. That is, a reasonable way of removing elements from a
 *  <a href="https://en.cppreference.com/w/cpp/container">Sequence</a> is
 *  <tt>S.erase(remove(S.begin(), S.end(), x), S.end())</tt>.
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/remove
 *  \see remove_if
 *  \see remove_copy
 *  \see remove_copy_if
 */
template<typename ForwardIterator,
         typename T>
  ForwardIterator remove(ForwardIterator first,
                         ForwardIterator last,
                         const T &value);


/*! \p remove_copy copies elements that are not equal to \p value from the range
 *  <tt>[first, last)</tt> to a range beginning at \p result. The return value is
 *  the end of the resulting range. This operation is stable, meaning that the
 *  relative order of the elements that are copied is the same as in
 *  the range <tt>[first, last)</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param result The resulting range is copied to the sequence beginning at this
 *                location.
 *  \param value The value to omit from the copied range.
 *  \return An OutputIterator pointing to the end of the resulting range of elements
 *          which are not equal to \p value.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>,
 *          and objects of type \p T can be compared for equality with objects of \p InputIterator's \c value_type.
 *
 *  \pre The range <tt>[first, last)</tt> shall not overlap the range <tt>[result, result + (last - first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p remove_copy to copy
 *  a sequence of numbers to an output range while omitting a value of interest using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/remove.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int V[N] = {-2, 0, -1, 0, 1, 2};
 *  int result[N-2];
 *  thrust::remove_copy(thrust::host, V, V + N, result, 0);
 *  // V remains {-2, 0, -1, 0, 1, 2}
 *  // result is now {-2, -1, 1, 2}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/remove_copy
 *  \see remove
 *  \see remove_if
 *  \see remove_copy_if
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename T>
__host__ __device__
  OutputIterator remove_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator result,
                             const T &value);


/*! \p remove_copy copies elements that are not equal to \p value from the range
 *  <tt>[first, last)</tt> to a range beginning at \p result. The return value is
 *  the end of the resulting range. This operation is stable, meaning that the
 *  relative order of the elements that are copied is the same as in
 *  the range <tt>[first, last)</tt>.
 *
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param result The resulting range is copied to the sequence beginning at this
 *                location.
 *  \param value The value to omit from the copied range.
 *  \return An OutputIterator pointing to the end of the resulting range of elements
 *          which are not equal to \p value.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to a type in \p OutputIterator's set of \c value_types.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>,
 *          and objects of type \p T can be compared for equality with objects of \p InputIterator's \c value_type.
 *
 *  \pre The range <tt>[first, last)</tt> shall not overlap the range <tt>[result, result + (last - first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p remove_copy to copy
 *  a sequence of numbers to an output range while omitting a value of interest.
 *
 *  \code
 *  #include <thrust/remove.h>
 *  ...
 *  const int N = 6;
 *  int V[N] = {-2, 0, -1, 0, 1, 2};
 *  int result[N-2];
 *  thrust::remove_copy(V, V + N, result, 0);
 *  // V remains {-2, 0, -1, 0, 1, 2}
 *  // result is now {-2, -1, 1, 2}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/remove_copy
 *  \see remove
 *  \see remove_if
 *  \see remove_copy_if
 */
template<typename InputIterator,
         typename OutputIterator,
         typename T>
  OutputIterator remove_copy(InputIterator first,
                             InputIterator last,
                             OutputIterator result,
                             const T &value);


/*! \p remove_if removes from the range <tt>[first, last)</tt> every element \p x
 *  such that <tt>pred(x)</tt> is \c true. That is, \p remove_if returns an
 *  iterator \c new_last such that the range <tt>[first,new_last)</tt> contains
 *  no elements for which \p pred is \c true. The iterators in the range
 *  <tt>[new_last,last)</tt> are all still dereferenceable, but the elements that
 *  they point to are unspecified. \p remove_if is stable, meaning that the
 *  relative order of elements that are not removed is unchanged.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param pred A predicate to evaluate for each element of the range
 *              <tt>[first,last)</tt>. Elements for which \p pred evaluates to
 *              \c true are removed from the sequence.
 *  \return A ForwardIterator pointing to the end of the resulting range of
 *          elements for which \p pred evaluated to \c true.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          \p ForwardIterator is mutable,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p remove_if to remove
 *  all even numbers from an array of integers using the \p thrust::host execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/remove.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  int *new_end = thrust::remove_if(thrust::host, A, A + N, is_even());
 *  // The first three values of A are now {1, 5, 7}
 *  // Values beyond new_end are unspecified
 *  \endcode
 *
 *  \note The meaning of "removal" is somewhat subtle. \p remove_if does not
 *  destroy any iterators, and does not change the distance between \p first and
 *  \p last. (There's no way that it could do anything of the sort.) So, for
 *  example, if \c V is a device_vector,
 *  <tt>remove_if(V.begin(), V.end(), pred)</tt> does not change
 *  <tt>V.size()</tt>: \c V will contain just as many elements as it did before.
 *  \p remove_if returns an iterator that points to the end of the resulting
 *  range after elements have been removed from it; it follows that the elements
 *  after that iterator are of no interest, and may be discarded. If you are
 *  removing elements from a
 *  <a href="https://en.cppreference.com/w/cpp/container">Sequence</a>, you may
 *  simply erase them. That is, a reasonable way of removing elements from a
 *  <a href="https://en.cppreference.com/w/cpp/container">Sequence</a> is
 *  <tt>S.erase(remove_if(S.begin(), S.end(), pred), S.end())</tt>.
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/remove
 *  \see remove
 *  \see remove_copy
 *  \see remove_copy_if
 */
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator remove_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred);


/*! \p remove_if removes from the range <tt>[first, last)</tt> every element \p x
 *  such that <tt>pred(x)</tt> is \c true. That is, \p remove_if returns an
 *  iterator \c new_last such that the range <tt>[first,new_last)</tt> contains
 *  no elements for which \p pred is \c true. The iterators in the range
 *  <tt>[new_last,last)</tt> are all still dereferenceable, but the elements that
 *  they point to are unspecified. \p remove_if is stable, meaning that the
 *  relative order of elements that are not removed is unchanged.
 *
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param pred A predicate to evaluate for each element of the range
 *              <tt>[first,last)</tt>. Elements for which \p pred evaluates to
 *              \c true are removed from the sequence.
 *  \return A ForwardIterator pointing to the end of the resulting range of
 *          elements for which \p pred evaluated to \c true.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          \p ForwardIterator is mutable,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p remove_if to remove
 *  all even numbers from an array of integers.
 *
 *  \code
 *  #include <thrust/remove.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  int *new_end = thrust::remove_if(A, A + N, is_even());
 *  // The first three values of A are now {1, 5, 7}
 *  // Values beyond new_end are unspecified
 *  \endcode
 *
 *  \note The meaning of "removal" is somewhat subtle. \p remove_if does not
 *  destroy any iterators, and does not change the distance between \p first and
 *  \p last. (There's no way that it could do anything of the sort.) So, for
 *  example, if \c V is a device_vector,
 *  <tt>remove_if(V.begin(), V.end(), pred)</tt> does not change
 *  <tt>V.size()</tt>: \c V will contain just as many elements as it did before.
 *  \p remove_if returns an iterator that points to the end of the resulting
 *  range after elements have been removed from it; it follows that the elements
 *  after that iterator are of no interest, and may be discarded. If you are
 *  removing elements from a
 *  <a href="https://en.cppreference.com/w/cpp/container">Sequence</a>, you may
 *  simply erase them. That is, a reasonable way of removing elements from a
 *  <a href="https://en.cppreference.com/w/cpp/container">Sequence</a> is
 *  <tt>S.erase(remove_if(S.begin(), S.end(), pred), S.end())</tt>.
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/remove
 *  \see remove
 *  \see remove_copy
 *  \see remove_copy_if
 */
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred);


/*! \p remove_copy_if copies elements from the range <tt>[first,last)</tt> to a
 *  range beginning at \p result, except that elements for which \p pred is
 *  \c true are not copied. The return value is the end of the resulting range.
 *  This operation is stable, meaning that the relative order of the elements that
 *  are copied is the same as the range <tt>[first,last)</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param result The resulting range is copied to the sequence beginning at this
 *                location.
 *  \param pred A predicate to evaluate for each element of the range <tt>[first,last)</tt>.
 *              Elements for which \p pred evaluates to \c false are not copied
 *              to the resulting sequence.
 *  \return An OutputIterator pointing to the end of the resulting range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator's \c value_type is convertible to a type in \p OutputIterator's set of \c value_types,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The range <tt>[first, last)</tt> shall not overlap the range <tt>[result, result + (last - first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p remove_copy_if to copy
 *  a sequence of numbers to an output range while omitting even numbers using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/remove.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  const int N = 6;
 *  int V[N] = {-2, 0, -1, 0, 1, 2};
 *  int result[2];
 *  thrust::remove_copy_if(thrust::host, V, V + N, result, is_even());
 *  // V remains {-2, 0, -1, 0, 1, 2}
 *  // result is now {-1, 1}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/remove_copy
 *  \see remove
 *  \see remove_copy
 *  \see remove_if
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename Predicate>
__host__ __device__
  OutputIterator remove_copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred);


/*! \p remove_copy_if copies elements from the range <tt>[first,last)</tt> to a
 *  range beginning at \p result, except that elements for which \p pred is
 *  \c true are not copied. The return value is the end of the resulting range.
 *  This operation is stable, meaning that the relative order of the elements that
 *  are copied is the same as the range <tt>[first,last)</tt>.
 *
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param result The resulting range is copied to the sequence beginning at this
 *                location.
 *  \param pred A predicate to evaluate for each element of the range <tt>[first,last)</tt>.
 *              Elements for which \p pred evaluates to \c false are not copied
 *              to the resulting sequence.
 *  \return An OutputIterator pointing to the end of the resulting range.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator's \c value_type is convertible to a type in \p OutputIterator's set of \c value_types,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The range <tt>[first, last)</tt> shall not overlap the range <tt>[result, result + (last - first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p remove_copy_if to copy
 *  a sequence of numbers to an output range while omitting even numbers.
 *
 *  \code
 *  #include <thrust/remove.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  const int N = 6;
 *  int V[N] = {-2, 0, -1, 0, 1, 2};
 *  int result[2];
 *  thrust::remove_copy_if(V, V + N, result, is_even());
 *  // V remains {-2, 0, -1, 0, 1, 2}
 *  // result is now {-1, 1}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/remove_copy
 *  \see remove
 *  \see remove_copy
 *  \see remove_if
 */
template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred);


/*! \p remove_if removes from the range <tt>[first, last)</tt> every element \p x
 *  such that <tt>pred(x)</tt> is \c true. That is, \p remove_if returns an
 *  iterator \c new_last such that the range <tt>[first, new_last)</tt> contains
 *  no elements for which \p pred of the corresponding stencil value is \c true. 
 *  The iterators in the range <tt>[new_last,last)</tt> are all still dereferenceable,
 *  but the elements that they point to are unspecified. \p remove_if is stable,
 *  meaning that the relative order of elements that are not removed is unchanged.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param stencil The beginning of the stencil sequence.
 *  \param pred A predicate to evaluate for each element of the range
 *              <tt>[stencil, stencil + (last - first))</tt>. Elements for which \p pred evaluates to
 *              \c true are removed from the sequence <tt>[first, last)</tt>
 *  \return A ForwardIterator pointing to the end of the resulting range of
 *          elements for which \p pred evaluated to \c true.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>
 *          and \p ForwardIterator is mutable.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The range <tt>[first, last)</tt> shall not overlap the range <tt>[result, result + (last - first))</tt>.
 *  \pre The range <tt>[stencil, stencil + (last - first))</tt> shall not overlap the range <tt>[result, result + (last - first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p remove_if to remove
 *  specific elements from an array of integers using the \p thrust::host execution policy for
 *  parallelization:
 *
 *  \code
 *  #include <thrust/remove.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  int S[N] = {0, 1, 1, 1, 0, 0};
 *
 *  int *new_end = thrust::remove_if(thrust::host, A, A + N, S, thrust::identity<int>());
 *  // The first three values of A are now {1, 5, 7}
 *  // Values beyond new_end are unspecified
 *  \endcode
 *
 *  \note The range <tt>[first, last)</tt> is not permitted to overlap with the range <tt>[stencil, stencil + (last - first))</tt>.
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/remove
 *  \see remove
 *  \see remove_copy
 *  \see remove_copy_if
 */
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator remove_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred);


/*! \p remove_if removes from the range <tt>[first, last)</tt> every element \p x
 *  such that <tt>pred(x)</tt> is \c true. That is, \p remove_if returns an
 *  iterator \c new_last such that the range <tt>[first, new_last)</tt> contains
 *  no elements for which \p pred of the corresponding stencil value is \c true. 
 *  The iterators in the range <tt>[new_last,last)</tt> are all still dereferenceable,
 *  but the elements that they point to are unspecified. \p remove_if is stable,
 *  meaning that the relative order of elements that are not removed is unchanged.
 *
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param stencil The beginning of the stencil sequence.
 *  \param pred A predicate to evaluate for each element of the range
 *              <tt>[stencil, stencil + (last - first))</tt>. Elements for which \p pred evaluates to
 *              \c true are removed from the sequence <tt>[first, last)</tt>
 *  \return A ForwardIterator pointing to the end of the resulting range of
 *          elements for which \p pred evaluated to \c true.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>
 *          and \p ForwardIterator is mutable.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The range <tt>[first, last)</tt> shall not overlap the range <tt>[result, result + (last - first))</tt>.
 *  \pre The range <tt>[stencil, stencil + (last - first))</tt> shall not overlap the range <tt>[result, result + (last - first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p remove_if to remove
 *  specific elements from an array of integers.
 *
 *  \code
 *  #include <thrust/remove.h>
 *  ...
 *  const int N = 6;
 *  int A[N] = {1, 4, 2, 8, 5, 7};
 *  int S[N] = {0, 1, 1, 1, 0, 0};
 *
 *  int *new_end = thrust::remove_if(A, A + N, S, thrust::identity<int>());
 *  // The first three values of A are now {1, 5, 7}
 *  // Values beyond new_end are unspecified
 *  \endcode
 *
 *  \note The range <tt>[first, last)</tt> is not permitted to overlap with the range <tt>[stencil, stencil + (last - first))</tt>.
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/remove
 *  \see remove
 *  \see remove_copy
 *  \see remove_copy_if
 */
template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred);


/*! \p remove_copy_if copies elements from the range <tt>[first,last)</tt> to a
 *  range beginning at \p result, except that elements for which \p pred of the 
 *  corresponding stencil value is \c true are not copied. The return value is 
 *  the end of the resulting range.  This operation is stable, meaning that the
 *  relative order of the elements that are copied is the same as the 
 *  range <tt>[first,last)</tt>.
 *
 *  The algorithm's execution policy is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param stencil The beginning of the stencil sequence.
 *  \param result The resulting range is copied to the sequence beginning at this
 *                location.
 *  \param pred A predicate to evaluate for each element of the range <tt>[first,last)</tt>.
 *              Elements for which \p pred evaluates to \c false are not copied
 *              to the resulting sequence.
 *  \return An OutputIterator pointing to the end of the resulting range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1's \c value_type is convertible to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator2's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The range <tt>[stencil, stencil + (last - first))</tt> shall not overlap the range <tt>[result, result + (last - first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p remove_copy_if to copy
 *  a sequence of numbers to an output range while omitting specific elements using the \p thrust::host
 *  execution policy for parallelization.
 *
 *  \code
 *  #include <thrust/remove.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  const int N = 6;
 *  int V[N] = {-2, 0, -1, 0, 1, 2};
 *  int S[N] = { 1, 1,  0, 1, 0, 1};
 *  int result[2];
 *  thrust::remove_copy_if(thrust::host, V, V + N, S, result, thrust::identity<int>());
 *  // V remains {-2, 0, -1, 0, 1, 2}
 *  // result is now {-1, 1}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/remove_copy
 *  \see remove
 *  \see remove_copy
 *  \see remove_if
 *  \see copy_if
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
__host__ __device__
  OutputIterator remove_copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred);


/*! \p remove_copy_if copies elements from the range <tt>[first,last)</tt> to a
 *  range beginning at \p result, except that elements for which \p pred of the 
 *  corresponding stencil value is \c true are not copied. The return value is 
 *  the end of the resulting range.  This operation is stable, meaning that the
 *  relative order of the elements that are copied is the same as the 
 *  range <tt>[first,last)</tt>.
 *
 *  \param first The beginning of the range of interest.
 *  \param last The end of the range of interest.
 *  \param stencil The beginning of the stencil sequence.
 *  \param result The resulting range is copied to the sequence beginning at this
 *                location.
 *  \param pred A predicate to evaluate for each element of the range <tt>[first,last)</tt>.
 *              Elements for which \p pred evaluates to \c false are not copied
 *              to the resulting sequence.
 *  \return An OutputIterator pointing to the end of the resulting range.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          \p InputIterator1's \c value_type is convertible to a type in \p OutputIterator's set of \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator2's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The range <tt>[stencil, stencil + (last - first))</tt> shall not overlap the range <tt>[result, result + (last - first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p remove_copy_if to copy
 *  a sequence of numbers to an output range while omitting specific elements.
 *
 *  \code
 *  #include <thrust/remove.h>
 *  ...
 *  const int N = 6;
 *  int V[N] = {-2, 0, -1, 0, 1, 2};
 *  int S[N] = { 1, 1,  0, 1, 0, 1};
 *  int result[2];
 *  thrust::remove_copy_if(V, V + N, S, result, thrust::identity<int>());
 *  // V remains {-2, 0, -1, 0, 1, 2}
 *  // result is now {-1, 1}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/remove_copy
 *  \see remove
 *  \see remove_copy
 *  \see remove_if
 *  \see copy_if
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred);


/*! \} // end stream_compaction
 */

THRUST_NAMESPACE_END

#include <thrust/detail/remove.inl>
