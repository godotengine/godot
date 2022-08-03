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


/*! \file partition.h
 *  \brief Reorganizes a range based on a predicate
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup reordering
 *  \ingroup algorithms
 *
 *  \addtogroup partitioning
 *  \ingroup reordering
 *  \{
 */


/*! \p partition reorders the elements <tt>[first, last)</tt> based on the function
 *  object \p pred, such that all of the elements that satisfy \p pred precede the
 *  elements that fail to satisfy it. The postcondition is that, for some iterator
 *  \c middle in the range <tt>[first, last)</tt>, <tt>pred(*i)</tt> is \c true for every
 *  iterator \c i in the range <tt>[first,middle)</tt> and \c false for every iterator
 *  \c i in the range <tt>[middle, last)</tt>. The return value of \p partition is
 *  \c middle.
 *
 *  Note that the relative order of elements in the two reordered sequences is not
 *  necessarily the same as it was in the original sequence. A different algorithm,
 *  \p stable_partition, does guarantee to preserve the relative order.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to reorder.
 *  \param last The end of the sequence to reorder.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return An iterator referring to the first element of the second partition, that is,
 *          the sequence of the elements which do not satisfy \p pred.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type,
 *          and \p ForwardIterator is mutable.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p partition to reorder a
 *  sequence so that even numbers precede odd numbers using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/partition.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::partition(thrust::host,
 *                    A, A + N,
 *                    is_even());
 *  // A is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/partition
 *  \see \p stable_partition
 *  \see \p partition_copy
 */
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator partition(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred);


/*! \p partition reorders the elements <tt>[first, last)</tt> based on the function
 *  object \p pred, such that all of the elements that satisfy \p pred precede the
 *  elements that fail to satisfy it. The postcondition is that, for some iterator
 *  \c middle in the range <tt>[first, last)</tt>, <tt>pred(*i)</tt> is \c true for every
 *  iterator \c i in the range <tt>[first,middle)</tt> and \c false for every iterator
 *  \c i in the range <tt>[middle, last)</tt>. The return value of \p partition is
 *  \c middle.
 *
 *  Note that the relative order of elements in the two reordered sequences is not
 *  necessarily the same as it was in the original sequence. A different algorithm,
 *  \p stable_partition, does guarantee to preserve the relative order.
 *
 *  \param first The beginning of the sequence to reorder.
 *  \param last The end of the sequence to reorder.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return An iterator referring to the first element of the second partition, that is,
 *          the sequence of the elements which do not satisfy \p pred.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type,
 *          and \p ForwardIterator is mutable.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p partition to reorder a
 *  sequence so that even numbers precede odd numbers.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::partition(A, A + N,
 *                     is_even());
 *  // A is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/partition
 *  \see \p stable_partition
 *  \see \p partition_copy
 */
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred);


/*! \p partition reorders the elements <tt>[first, last)</tt> based on the function
 *  object \p pred applied to a stencil range <tt>[stencil, stencil + (last - first))</tt>,
 *  such that all of the elements whose corresponding stencil element satisfies \p pred precede all of the elements whose
 *  corresponding stencil element fails to satisfy it. The postcondition is that, for some iterator
 *  \c middle in the range <tt>[first, last)</tt>, <tt>pred(*stencil_i)</tt> is \c true for every iterator
 *  \c stencil_i in the range <tt>[stencil,stencil + (middle - first))</tt> and \c false for every iterator \c stencil_i
 *  in the range <tt>[stencil + (middle - first), stencil + (last - first))</tt>.
 *  The return value of \p stable_partition is \c middle.
 *
 *  Note that the relative order of elements in the two reordered sequences is not
 *  necessarily the same as it was in the original sequence. A different algorithm,
 *  \p stable_partition, does guarantee to preserve the relative order.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to reorder.
 *  \param last The end of the sequence to reorder.
 *  \param stencil The beginning of the stencil sequence.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return An iterator referring to the first element of the second partition, that is,
 *          the sequence of the elements whose stencil elements do not satisfy \p pred.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[stencil, stencil + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p partition to reorder a
 *  sequence so that even numbers precede odd numbers using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/partition.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
 *  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::partition(thrust::host, A, A + N, S, is_even());
 *  // A is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
 *  // S is unmodified
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/partition
 *  \see \p stable_partition
 *  \see \p partition_copy
 */
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator partition(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred);


/*! \p partition reorders the elements <tt>[first, last)</tt> based on the function
 *  object \p pred applied to a stencil range <tt>[stencil, stencil + (last - first))</tt>,
 *  such that all of the elements whose corresponding stencil element satisfies \p pred precede all of the elements whose
 *  corresponding stencil element fails to satisfy it. The postcondition is that, for some iterator
 *  \c middle in the range <tt>[first, last)</tt>, <tt>pred(*stencil_i)</tt> is \c true for every iterator
 *  \c stencil_i in the range <tt>[stencil,stencil + (middle - first))</tt> and \c false for every iterator \c stencil_i
 *  in the range <tt>[stencil + (middle - first), stencil + (last - first))</tt>.
 *  The return value of \p stable_partition is \c middle.
 *
 *  Note that the relative order of elements in the two reordered sequences is not
 *  necessarily the same as it was in the original sequence. A different algorithm,
 *  \p stable_partition, does guarantee to preserve the relative order.
 *
 *  \param first The beginning of the sequence to reorder.
 *  \param last The end of the sequence to reorder.
 *  \param stencil The beginning of the stencil sequence.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return An iterator referring to the first element of the second partition, that is,
 *          the sequence of the elements whose stencil elements do not satisfy \p pred.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The ranges <tt>[first,last)</tt> and <tt>[stencil, stencil + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p partition to reorder a
 *  sequence so that even numbers precede odd numbers.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
 *  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::partition(A, A + N, S, is_even());
 *  // A is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
 *  // S is unmodified
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/partition
 *  \see \p stable_partition
 *  \see \p partition_copy
 */
template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred);


/*! \p partition_copy differs from \p partition only in that the reordered
 *  sequence is written to difference output sequences, rather than in place.
 *
 *  \p partition_copy copies the elements <tt>[first, last)</tt> based on the
 *  function object \p pred. All of the elements that satisfy \p pred are copied
 *  to the range beginning at \p out_true and all the elements that fail to satisfy it
 *  are copied to the range beginning at \p out_false.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to reorder.
 *  \param last The end of the sequence to reorder.
 *  \param out_true The destination of the resulting sequence of elements which satisfy \p pred.
 *  \param out_false The destination of the resulting sequence of elements which fail to satisfy \p pred.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return A \p pair p such that <tt>p.first</tt> is the end of the output range beginning
 *          at \p out_true and <tt>p.second</tt> is the end of the output range beginning at
 *          \p out_false.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type and \p InputIterator's \c value_type
 *          is convertible to \p OutputIterator1 and \p OutputIterator2's \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The input range shall not overlap with either output range.
 *
 *  The following code snippet demonstrates how to use \p partition_copy to separate a
 *  sequence into two output sequences of even and odd numbers using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/partition.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  int result[10];
 *  const int N = sizeof(A)/sizeof(int);
 *  int *evens = result;
 *  int *odds  = result + 5;
 *  thrust::partition_copy(thrust::host, A, A + N, evens, odds, is_even());
 *  // A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 *  // result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  // evens points to {2, 4, 6, 8, 10}
 *  // odds points to {1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \note The relative order of elements in the two reordered sequences is not
 *  necessarily the same as it was in the original sequence. A different algorithm,
 *  \p stable_partition_copy, does guarantee to preserve the relative order.
 *
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf
 *  \see \p stable_partition_copy
 *  \see \p partition
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   InputIterator first,
                   InputIterator last,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred);


/*! \p partition_copy differs from \p partition only in that the reordered
 *  sequence is written to difference output sequences, rather than in place.
 *
 *  \p partition_copy copies the elements <tt>[first, last)</tt> based on the
 *  function object \p pred. All of the elements that satisfy \p pred are copied
 *  to the range beginning at \p out_true and all the elements that fail to satisfy it
 *  are copied to the range beginning at \p out_false.
 *
 *  \param first The beginning of the sequence to reorder.
 *  \param last The end of the sequence to reorder.
 *  \param out_true The destination of the resulting sequence of elements which satisfy \p pred.
 *  \param out_false The destination of the resulting sequence of elements which fail to satisfy \p pred.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return A \p pair p such that <tt>p.first</tt> is the end of the output range beginning
 *          at \p out_true and <tt>p.second</tt> is the end of the output range beginning at
 *          \p out_false.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type and \p InputIterator's \c value_type
 *          is convertible to \p OutputIterator1 and \p OutputIterator2's \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The input range shall not overlap with either output range.
 *
 *  The following code snippet demonstrates how to use \p partition_copy to separate a
 *  sequence into two output sequences of even and odd numbers.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  int result[10];
 *  const int N = sizeof(A)/sizeof(int);
 *  int *evens = result;
 *  int *odds  = result + 5;
 *  thrust::partition_copy(A, A + N, evens, odds, is_even());
 *  // A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 *  // result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  // evens points to {2, 4, 6, 8, 10}
 *  // odds points to {1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \note The relative order of elements in the two reordered sequences is not
 *  necessarily the same as it was in the original sequence. A different algorithm,
 *  \p stable_partition_copy, does guarantee to preserve the relative order.
 *
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf
 *  \see \p stable_partition_copy
 *  \see \p partition
 */
template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(InputIterator first,
                   InputIterator last,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred);


/*! \p partition_copy differs from \p partition only in that the reordered
 *  sequence is written to difference output sequences, rather than in place.
 *
 *  \p partition_copy copies the elements <tt>[first, last)</tt> based on the
 *  function object \p pred which is applied to a range of stencil elements. All of the elements
 *  whose corresponding stencil element satisfies \p pred are copied to the range beginning at \p out_true
 *  and all the elements whose stencil element fails to satisfy it are copied to the range beginning
 *  at \p out_false.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to reorder.
 *  \param last The end of the sequence to reorder.
 *  \param stencil The beginning of the stencil sequence.
 *  \param out_true The destination of the resulting sequence of elements which satisfy \p pred.
 *  \param out_false The destination of the resulting sequence of elements which fail to satisfy \p pred.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return A \p pair p such that <tt>p.first</tt> is the end of the output range beginning
 *          at \p out_true and <tt>p.second</tt> is the end of the output range beginning at
 *          \p out_false.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p OutputIterator1 and \p OutputIterator2's \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator2's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The input ranges shall not overlap with either output range.
 *
 *  The following code snippet demonstrates how to use \p partition_copy to separate a
 *  sequence into two output sequences of even and odd numbers using the \p thrust::host execution
 *  policy for parallelization.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  int S[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
 *  int result[10];
 *  const int N = sizeof(A)/sizeof(int);
 *  int *evens = result;
 *  int *odds  = result + 5;
 *  thrust::stable_partition_copy(thrust::host, A, A + N, S, evens, odds, thrust::identity<int>());
 *  // A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 *  // S remains {0, 1, 0, 1, 0, 1, 0, 1, 0,  1}
 *  // result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  // evens points to {2, 4, 6, 8, 10}
 *  // odds points to {1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \note The relative order of elements in the two reordered sequences is not
 *  necessarily the same as it was in the original sequence. A different algorithm,
 *  \p stable_partition_copy, does guarantee to preserve the relative order.
 *
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf
 *  \see \p stable_partition_copy
 *  \see \p partition
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   InputIterator1 first,
                   InputIterator1 last,
                   InputIterator2 stencil,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred);


/*! \p partition_copy differs from \p partition only in that the reordered
 *  sequence is written to difference output sequences, rather than in place.
 *
 *  \p partition_copy copies the elements <tt>[first, last)</tt> based on the
 *  function object \p pred which is applied to a range of stencil elements. All of the elements
 *  whose corresponding stencil element satisfies \p pred are copied to the range beginning at \p out_true
 *  and all the elements whose stencil element fails to satisfy it are copied to the range beginning
 *  at \p out_false.
 *
 *  \param first The beginning of the sequence to reorder.
 *  \param last The end of the sequence to reorder.
 *  \param stencil The beginning of the stencil sequence.
 *  \param out_true The destination of the resulting sequence of elements which satisfy \p pred.
 *  \param out_false The destination of the resulting sequence of elements which fail to satisfy \p pred.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return A \p pair p such that <tt>p.first</tt> is the end of the output range beginning
 *          at \p out_true and <tt>p.second</tt> is the end of the output range beginning at
 *          \p out_false.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p OutputIterator1 and \p OutputIterator2's \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator2's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The input ranges shall not overlap with either output range.
 *
 *  The following code snippet demonstrates how to use \p partition_copy to separate a
 *  sequence into two output sequences of even and odd numbers.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  int S[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
 *  int result[10];
 *  const int N = sizeof(A)/sizeof(int);
 *  int *evens = result;
 *  int *odds  = result + 5;
 *  thrust::stable_partition_copy(A, A + N, S, evens, odds, thrust::identity<int>());
 *  // A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 *  // S remains {0, 1, 0, 1, 0, 1, 0, 1, 0,  1}
 *  // result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  // evens points to {2, 4, 6, 8, 10}
 *  // odds points to {1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \note The relative order of elements in the two reordered sequences is not
 *  necessarily the same as it was in the original sequence. A different algorithm,
 *  \p stable_partition_copy, does guarantee to preserve the relative order.
 *
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf
 *  \see \p stable_partition_copy
 *  \see \p partition
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(InputIterator1 first,
                   InputIterator1 last,
                   InputIterator2 stencil,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred);


/*! \p stable_partition is much like \p partition : it reorders the elements in the
 *  range <tt>[first, last)</tt> based on the function object \p pred, such that all of
 *  the elements that satisfy \p pred precede all of the elements that fail to satisfy
 *  it. The postcondition is that, for some iterator \p middle in the range
 *  <tt>[first, last)</tt>, <tt>pred(*i)</tt> is \c true for every iterator \c i in the
 *  range <tt>[first,middle)</tt> and \c false for every iterator \c i in the range
 *  <tt>[middle, last)</tt>. The return value of \p stable_partition is \c middle.
 *
 *  \p stable_partition differs from \p partition in that \p stable_partition is
 *  guaranteed to preserve relative order. That is, if \c x and \c y are elements in
 *  <tt>[first, last)</tt>, and \c stencil_x and \c stencil_y are the stencil elements
 *  in corresponding positions within <tt>[stencil, stencil + (last - first))</tt>,
 *  and <tt>pred(stencil_x) == pred(stencil_y)</tt>, and if \c x precedes
 *  \c y, then it will still be true after \p stable_partition that \c x precedes \c y.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The first element of the sequence to reorder.
 *  \param last One position past the last element of the sequence to reorder.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return An iterator referring to the first element of the second partition, that is,
 *          the sequence of the elements which do not satisfy pred.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type,
 *          and \p ForwardIterator is mutable.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p stable_partition to reorder a
 *  sequence so that even numbers precede odd numbers using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/partition.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::stable_partition(thrust::host,
 *                           A, A + N,
 *                           is_even());
 *  // A is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/stable_partition
 *  \see \p partition
 *  \see \p stable_partition_copy
 */
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator stable_partition(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred);


/*! \p stable_partition is much like \p partition : it reorders the elements in the
 *  range <tt>[first, last)</tt> based on the function object \p pred, such that all of
 *  the elements that satisfy \p pred precede all of the elements that fail to satisfy
 *  it. The postcondition is that, for some iterator \p middle in the range
 *  <tt>[first, last)</tt>, <tt>pred(*i)</tt> is \c true for every iterator \c i in the
 *  range <tt>[first,middle)</tt> and \c false for every iterator \c i in the range
 *  <tt>[middle, last)</tt>. The return value of \p stable_partition is \c middle.
 *
 *  \p stable_partition differs from \p partition in that \p stable_partition is
 *  guaranteed to preserve relative order. That is, if \c x and \c y are elements in
 *  <tt>[first, last)</tt>, and \c stencil_x and \c stencil_y are the stencil elements
 *  in corresponding positions within <tt>[stencil, stencil + (last - first))</tt>,
 *  and <tt>pred(stencil_x) == pred(stencil_y)</tt>, and if \c x precedes
 *  \c y, then it will still be true after \p stable_partition that \c x precedes \c y.
 *
 *  \param first The first element of the sequence to reorder.
 *  \param last One position past the last element of the sequence to reorder.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return An iterator referring to the first element of the second partition, that is,
 *          the sequence of the elements which do not satisfy pred.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type,
 *          and \p ForwardIterator is mutable.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p stable_partition to reorder a
 *  sequence so that even numbers precede odd numbers.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::stable_partition(A, A + N,
 *                            is_even());
 *  // A is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/stable_partition
 *  \see \p partition
 *  \see \p stable_partition_copy
 */
template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred);


/*! \p stable_partition is much like \p partition: it reorders the elements in the
 *  range <tt>[first, last)</tt> based on the function object \p pred applied to a stencil
 *  range <tt>[stencil, stencil + (last - first))</tt>, such that all of
 *  the elements whose corresponding stencil element satisfies \p pred precede all of the elements whose
 *  corresponding stencil element fails to satisfy it. The postcondition is that, for some iterator
 *  \c middle in the range <tt>[first, last)</tt>, <tt>pred(*stencil_i)</tt> is \c true for every iterator
 *  \c stencil_i in the range <tt>[stencil,stencil + (middle - first))</tt> and \c false for every iterator \c stencil_i
 *  in the range <tt>[stencil + (middle - first), stencil + (last - first))</tt>.
 *  The return value of \p stable_partition is \c middle.
 *
 *  \p stable_partition differs from \p partition in that \p stable_partition is
 *  guaranteed to preserve relative order. That is, if \c x and \c y are elements in
 *  <tt>[first, last)</tt>, such that <tt>pred(x) == pred(y)</tt>, and if \c x precedes
 *  \c y, then it will still be true after \p stable_partition that \c x precedes \c y.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The first element of the sequence to reorder.
 *  \param last One position past the last element of the sequence to reorder.
 *  \param stencil The beginning of the stencil sequence.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return An iterator referring to the first element of the second partition, that is,
 *          the sequence of the elements whose stencil elements do not satisfy \p pred.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The range <tt>[first, last)</tt> shall not overlap with the range <tt>[stencil, stencil + (last - first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p stable_partition to reorder a
 *  sequence so that even numbers precede odd numbers using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/partition.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
 *  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::stable_partition(thrust::host, A, A + N, S, is_even());
 *  // A is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
 *  // S is unmodified
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/stable_partition
 *  \see \p partition
 *  \see \p stable_partition_copy
 */
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator stable_partition(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   InputIterator stencil,
                                   Predicate pred);


/*! \p stable_partition is much like \p partition: it reorders the elements in the
 *  range <tt>[first, last)</tt> based on the function object \p pred applied to a stencil
 *  range <tt>[stencil, stencil + (last - first))</tt>, such that all of
 *  the elements whose corresponding stencil element satisfies \p pred precede all of the elements whose
 *  corresponding stencil element fails to satisfy it. The postcondition is that, for some iterator
 *  \c middle in the range <tt>[first, last)</tt>, <tt>pred(*stencil_i)</tt> is \c true for every iterator
 *  \c stencil_i in the range <tt>[stencil,stencil + (middle - first))</tt> and \c false for every iterator \c stencil_i
 *  in the range <tt>[stencil + (middle - first), stencil + (last - first))</tt>.
 *  The return value of \p stable_partition is \c middle.
 *
 *  \p stable_partition differs from \p partition in that \p stable_partition is
 *  guaranteed to preserve relative order. That is, if \c x and \c y are elements in
 *  <tt>[first, last)</tt>, such that <tt>pred(x) == pred(y)</tt>, and if \c x precedes
 *  \c y, then it will still be true after \p stable_partition that \c x precedes \c y.
 *
 *  \param first The first element of the sequence to reorder.
 *  \param last One position past the last element of the sequence to reorder.
 *  \param stencil The beginning of the stencil sequence.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return An iterator referring to the first element of the second partition, that is,
 *          the sequence of the elements whose stencil elements do not satisfy \p pred.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The range <tt>[first, last)</tt> shall not overlap with the range <tt>[stencil, stencil + (last - first))</tt>.
 *
 *  The following code snippet demonstrates how to use \p stable_partition to reorder a
 *  sequence so that even numbers precede odd numbers.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
 *  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::stable_partition(A, A + N, S, is_even());
 *  // A is now {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
 *  // S is unmodified
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/stable_partition
 *  \see \p partition
 *  \see \p stable_partition_copy
 */
template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   InputIterator stencil,
                                   Predicate pred);


/*! \p stable_partition_copy differs from \p stable_partition only in that the reordered
 *  sequence is written to different output sequences, rather than in place.
 *
 *  \p stable_partition_copy copies the elements <tt>[first, last)</tt> based on the
 *  function object \p pred. All of the elements that satisfy \p pred are copied
 *  to the range beginning at \p out_true and all the elements that fail to satisfy it
 *  are copied to the range beginning at \p out_false.
 *
 *  \p stable_partition_copy differs from \p partition_copy in that
 *  \p stable_partition_copy is guaranteed to preserve relative order. That is, if
 *  \c x and \c y are elements in <tt>[first, last)</tt>, such that
 *  <tt>pred(x) == pred(y)</tt>, and if \c x precedes \c y, then it will still be true
 *  after \p stable_partition_copy that \c x precedes \c y in the output.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The first element of the sequence to reorder.
 *  \param last One position past the last element of the sequence to reorder.
 *  \param out_true The destination of the resulting sequence of elements which satisfy \p pred.
 *  \param out_false The destination of the resulting sequence of elements which fail to satisfy \p pred.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return A \p pair p such that <tt>p.first</tt> is the end of the output range beginning
 *          at \p out_true and <tt>p.second</tt> is the end of the output range beginning at
 *          \p out_false.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type and \p InputIterator's \c value_type
 *          is convertible to \p OutputIterator1 and \p OutputIterator2's \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The input ranges shall not overlap with either output range.
 *
 *  The following code snippet demonstrates how to use \p stable_partition_copy to
 *  reorder a sequence so that even numbers precede odd numbers using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/partition.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  int result[10];
 *  const int N = sizeof(A)/sizeof(int);
 *  int *evens = result;
 *  int *odds  = result + 5;
 *  thrust::stable_partition_copy(thrust::host, A, A + N, evens, odds, is_even());
 *  // A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 *  // result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  // evens points to {2, 4, 6, 8, 10}
 *  // odds points to {1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf
 *  \see \p partition_copy
 *  \see \p stable_partition
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred);


/*! \p stable_partition_copy differs from \p stable_partition only in that the reordered
 *  sequence is written to different output sequences, rather than in place.
 *
 *  \p stable_partition_copy copies the elements <tt>[first, last)</tt> based on the
 *  function object \p pred. All of the elements that satisfy \p pred are copied
 *  to the range beginning at \p out_true and all the elements that fail to satisfy it
 *  are copied to the range beginning at \p out_false.
 *
 *  \p stable_partition_copy differs from \p partition_copy in that
 *  \p stable_partition_copy is guaranteed to preserve relative order. That is, if
 *  \c x and \c y are elements in <tt>[first, last)</tt>, such that
 *  <tt>pred(x) == pred(y)</tt>, and if \c x precedes \c y, then it will still be true
 *  after \p stable_partition_copy that \c x precedes \c y in the output.
 *
 *  \param first The first element of the sequence to reorder.
 *  \param last One position past the last element of the sequence to reorder.
 *  \param out_true The destination of the resulting sequence of elements which satisfy \p pred.
 *  \param out_false The destination of the resulting sequence of elements which fail to satisfy \p pred.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return A \p pair p such that <tt>p.first</tt> is the end of the output range beginning
 *          at \p out_true and <tt>p.second</tt> is the end of the output range beginning at
 *          \p out_false.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type and \p InputIterator's \c value_type
 *          is convertible to \p OutputIterator1 and \p OutputIterator2's \c value_types.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The input ranges shall not overlap with either output range.
 *
 *  The following code snippet demonstrates how to use \p stable_partition_copy to
 *  reorder a sequence so that even numbers precede odd numbers.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  ...
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  int result[10];
 *  const int N = sizeof(A)/sizeof(int);
 *  int *evens = result;
 *  int *odds  = result + 5;
 *  thrust::stable_partition_copy(A, A + N, evens, odds, is_even());
 *  // A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 *  // result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  // evens points to {2, 4, 6, 8, 10}
 *  // odds points to {1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf
 *  \see \p partition_copy
 *  \see \p stable_partition
 */
template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred);


/*! \p stable_partition_copy differs from \p stable_partition only in that the reordered
 *  sequence is written to different output sequences, rather than in place.
 *
 *  \p stable_partition_copy copies the elements <tt>[first, last)</tt> based on the
 *  function object \p pred which is applied to a range of stencil elements. All of the elements
 *  whose corresponding stencil element satisfies \p pred are copied to the range beginning at \p out_true
 *  and all the elements whose stencil element fails to satisfy it are copied to the range beginning
 *  at \p out_false.
 *
 *  \p stable_partition_copy differs from \p partition_copy in that
 *  \p stable_partition_copy is guaranteed to preserve relative order. That is, if
 *  \c x and \c y are elements in <tt>[first, last)</tt>, such that
 *  <tt>pred(x) == pred(y)</tt>, and if \c x precedes \c y, then it will still be true
 *  after \p stable_partition_copy that \c x precedes \c y in the output.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The first element of the sequence to reorder.
 *  \param last One position past the last element of the sequence to reorder.
 *  \param stencil The beginning of the stencil sequence.
 *  \param out_true The destination of the resulting sequence of elements which satisfy \p pred.
 *  \param out_false The destination of the resulting sequence of elements which fail to satisfy \p pred.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return A \p pair p such that <tt>p.first</tt> is the end of the output range beginning
 *          at \p out_true and <tt>p.second</tt> is the end of the output range beginning at
 *          \p out_false.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p OutputIterator1 and \p OutputIterator2's \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator2's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The input ranges shall not overlap with either output range.
 *
 *  The following code snippet demonstrates how to use \p stable_partition_copy to
 *  reorder a sequence so that even numbers precede odd numbers using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/partition.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  int S[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
 *  int result[10];
 *  const int N = sizeof(A)/sizeof(int);
 *  int *evens = result;
 *  int *odds  = result + 5;
 *  thrust::stable_partition_copy(thrust::host, A, A + N, S, evens, odds, thrust::identity<int>());
 *  // A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 *  // S remains {0, 1, 0, 1, 0, 1, 0, 1, 0,  1}
 *  // result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  // evens points to {2, 4, 6, 8, 10}
 *  // odds points to {1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf
 *  \see \p partition_copy
 *  \see \p stable_partition
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred);


/*! \p stable_partition_copy differs from \p stable_partition only in that the reordered
 *  sequence is written to different output sequences, rather than in place.
 *
 *  \p stable_partition_copy copies the elements <tt>[first, last)</tt> based on the
 *  function object \p pred which is applied to a range of stencil elements. All of the elements
 *  whose corresponding stencil element satisfies \p pred are copied to the range beginning at \p out_true
 *  and all the elements whose stencil element fails to satisfy it are copied to the range beginning
 *  at \p out_false.
 *
 *  \p stable_partition_copy differs from \p partition_copy in that
 *  \p stable_partition_copy is guaranteed to preserve relative order. That is, if
 *  \c x and \c y are elements in <tt>[first, last)</tt>, such that
 *  <tt>pred(x) == pred(y)</tt>, and if \c x precedes \c y, then it will still be true
 *  after \p stable_partition_copy that \c x precedes \c y in the output.
 *
 *  \param first The first element of the sequence to reorder.
 *  \param last One position past the last element of the sequence to reorder.
 *  \param stencil The beginning of the stencil sequence.
 *  \param out_true The destination of the resulting sequence of elements which satisfy \p pred.
 *  \param out_false The destination of the resulting sequence of elements which fail to satisfy \p pred.
 *  \param pred A function object which decides to which partition each element of the
 *              sequence <tt>[first, last)</tt> belongs.
 *  \return A \p pair p such that <tt>p.first</tt> is the end of the output range beginning
 *          at \p out_true and <tt>p.second</tt> is the end of the output range beginning at
 *          \p out_false.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p OutputIterator1 and \p OutputIterator2's \c value_types.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *          and \p InputIterator2's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam OutputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The input ranges shall not overlap with either output range.
 *
 *  The following code snippet demonstrates how to use \p stable_partition_copy to
 *  reorder a sequence so that even numbers precede odd numbers.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  int S[] = {0, 1, 0, 1, 0, 1, 0, 1, 0,  1};
 *  int result[10];
 *  const int N = sizeof(A)/sizeof(int);
 *  int *evens = result;
 *  int *odds  = result + 5;
 *  thrust::stable_partition_copy(A, A + N, S, evens, odds, thrust::identity<int>());
 *  // A remains {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 *  // S remains {0, 1, 0, 1, 0, 1, 0, 1, 0,  1}
 *  // result is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
 *  // evens points to {2, 4, 6, 8, 10}
 *  // odds points to {1, 3, 5, 7, 9}
 *  \endcode
 *
 *  \see http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2569.pdf
 *  \see \p partition_copy
 *  \see \p stable_partition
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred);


/*! \} // end stream_compaction
 */

/*! \} // end reordering
 */

/*! \addtogroup searching
 *  \{
 */


/*! \p partition_point returns an iterator pointing to the end of the true
 *  partition of a partitioned range. \p partition_point requires the input range
 *  <tt>[first,last)</tt> to be a partition; that is, all elements which satisfy
 *  <tt>pred</tt> shall appear before those that do not.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range to consider.
 *  \param last The end of the range to consider.
 *  \param pred A function object which decides to which partition each element of the
 *              range <tt>[first, last)</tt> belongs.
 *  \return An iterator \c mid such that <tt>all_of(first, mid, pred)</tt>
 *          and <tt>none_of(mid, last, pred)</tt> are both true.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The range <tt>[first, last)</tt> shall be partitioned by \p pred.
 *
 *  \note Though similar, \p partition_point is not redundant with \p find_if_not.
 *        \p partition_point's precondition provides an opportunity for a
 *        faster implemention.
 *
 *  \code
 *  #include <thrust/partition.h>
 *  #include <thrust/execution_policy.h>
 *
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  
 *  ...
 *
 *  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
 *  int * B = thrust::partition_point(thrust::host, A, A + 10, is_even());
 *  // B - A is 5
 *  // [A, B) contains only even values
 *  \endcode
 *
 *  \see \p partition
 *  \see \p find_if_not
 */
template<typename DerivedPolicy, typename ForwardIterator, typename Predicate>
__host__ __device__
  ForwardIterator partition_point(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last,
                                  Predicate pred);


/*! \p partition_point returns an iterator pointing to the end of the true
 *  partition of a partitioned range. \p partition_point requires the input range
 *  <tt>[first,last)</tt> to be a partition; that is, all elements which satisfy
 *  <tt>pred</tt> shall appear before those that do not.
 *  \param first The beginning of the range to consider.
 *  \param last The end of the range to consider.
 *  \param pred A function object which decides to which partition each element of the
 *              range <tt>[first, last)</tt> belongs.
 *  \return An iterator \c mid such that <tt>all_of(first, mid, pred)</tt>
 *          and <tt>none_of(mid, last, pred)</tt> are both true.
 *
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>,
 *          and \p ForwardIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The range <tt>[first, last)</tt> shall be partitioned by \p pred.
 *
 *  \note Though similar, \p partition_point is not redundant with \p find_if_not.
 *        \p partition_point's precondition provides an opportunity for a
 *        faster implemention.
 *
 *  \code
 *  #include <thrust/partition.h>
 *
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  
 *  ...
 *
 *  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
 *  int * B = thrust::partition_point(A, A + 10, is_even());
 *  // B - A is 5
 *  // [A, B) contains only even values
 *  \endcode
 *
 *  \see \p partition
 *  \see \p find_if_not
 */
template<typename ForwardIterator, typename Predicate>
  ForwardIterator partition_point(ForwardIterator first,
                                  ForwardIterator last,
                                  Predicate pred);

/*! \} // searching
 */

/*! \addtogroup reductions
 *  \{
 *  \addtogroup predicates
 *  \{
 */


/*! \p is_partitioned returns \c true if the given range 
 *  is partitioned with respect to a predicate, and \c false otherwise.
 *
 *  Specifically, \p is_partitioned returns \c true if <tt>[first, last)</tt>
 *  is empty of if <tt>[first, last)</tt> is partitioned by \p pred, i.e. if
 *  all elements that satisfy \p pred appear before those that do not.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range to consider.
 *  \param last The end of the range to consider.
 *  \param pred A function object which decides to which partition each element of the
 *         range <tt>[first, last)</tt> belongs.
 *  \return \c true if the range <tt>[first, last)</tt> is partitioned with respect
 *          to \p pred, or if <tt>[first, last)</tt> is empty. \c false, otherwise.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *  
 *  \code
 *  #include <thrust/partition.h>
 *  #include <thrust/execution_policy.h>
 *
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  
 *  ...
 *
 *  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
 *  int B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *
 *  thrust::is_partitioned(thrust::host, A, A + 10, is_even()); // returns true
 *  thrust::is_partitioned(thrust::host, B, B + 10, is_even()); // returns false
 *  \endcode
 *
 *  \see \p partition
 */
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__host__ __device__
  bool is_partitioned(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred);


/*! \p is_partitioned returns \c true if the given range 
 *  is partitioned with respect to a predicate, and \c false otherwise.
 *
 *  Specifically, \p is_partitioned returns \c true if <tt>[first, last)</tt>
 *  is empty of if <tt>[first, last)</tt> is partitioned by \p pred, i.e. if
 *  all elements that satisfy \p pred appear before those that do not.
 *
 *  \param first The beginning of the range to consider.
 *  \param last The end of the range to consider.
 *  \param pred A function object which decides to which partition each element of the
 *         range <tt>[first, last)</tt> belongs.
 *  \return \c true if the range <tt>[first, last)</tt> is partitioned with respect
 *          to \p pred, or if <tt>[first, last)</tt> is empty. \c false, otherwise.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Input Iterator</a>,
 *          and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *  
 *  \code
 *  #include <thrust/partition.h>
 *
 *  struct is_even
 *  {
 *    __host__ __device__
 *    bool operator()(const int &x)
 *    {
 *      return (x % 2) == 0;
 *    }
 *  };
 *  
 *  ...
 *
 *  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
 *  int B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *
 *  thrust::is_partitioned(A, A + 10, is_even()); // returns true
 *  thrust::is_partitioned(B, B + 10, is_even()); // returns false
 *  \endcode
 *
 *  \see \p partition
 */
template<typename InputIterator, typename Predicate>
  bool is_partitioned(InputIterator first,
                      InputIterator last,
                      Predicate pred);


/*! \} // end predicates
 *  \} // end reductions
 */

THRUST_NAMESPACE_END

#include <thrust/detail/partition.inl>

