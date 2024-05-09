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


/*! \file thrust/copy.h
 *  \brief Copies elements from one range to another
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup algorithms
 */

/*! \addtogroup copying
 *  \ingroup algorithms
 *  \{
 */


/*! \p copy copies elements from the range [\p first, \p last) to the range
 *  [\p result, \p result + (\p last - \p first)). That is, it performs
 *  the assignments *\p result = *\p first, *(\p result + \c 1) = *(\p first + \c 1),
 *  and so on. Generally, for every integer \c n from \c 0 to \p last - \p first, \p copy
 *  performs the assignment *(\p result + \c n) = *(\p first + \c n). Unlike
 *  \c std::copy, \p copy offers no guarantee on order of operation.  As a result,
 *  calling \p copy with overlapping source and destination ranges has undefined
 *  behavior.
 *
 *  The return value is \p result + (\p last - \p first).
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to copy.
 *  \param last The end of the sequence to copy.
 *  \param result The destination sequence.
 *  \return The end of the destination sequence.
 *  \see https://en.cppreference.com/w/cpp/algorithm/copy
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and \c InputIterator's \c value_type must be convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre \p result may be equal to \p first, but \p result shall not be in the range <tt>[first, last)</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p copy
 *  to copy from one range to another using the \p thrust::device parallelization policy:
 *
 *  \code
 *  #include <thrust/copy.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *
 *  thrust::device_vector<int> vec0(100);
 *  thrust::device_vector<int> vec1(100);
 *  ...
 *
 *  thrust::copy(thrust::device, vec0.begin(), vec0.end(), vec1.begin());
 *
 *  // vec1 is now a copy of vec0
 *  \endcode
 */
template<typename DerivedPolicy, typename InputIterator, typename OutputIterator>
__host__ __device__
  OutputIterator copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      OutputIterator result);


/*! \p copy_n copies elements from the range <tt>[first, first + n)</tt> to the range
 *  <tt>[result, result + n)</tt>. That is, it performs the assignments <tt>*result = *first, *(result + 1) = *(first + 1)</tt>,
 *  and so on. Generally, for every integer \c i from \c 0 to \c n, \p copy
 *  performs the assignment *(\p result + \c i) = *(\p first + \c i). Unlike
 *  \c std::copy_n, \p copy_n offers no guarantee on order of operation. As a result,
 *  calling \p copy_n with overlapping source and destination ranges has undefined
 *  behavior.
 *
 *  The return value is \p result + \p n.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range to copy.
 *  \param n The number of elements to copy.
 *  \param result The beginning destination range.
 *  \return The end of the destination range.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and \c InputIterator's \c value_type must be convertible to \c OutputIterator's \c value_type.
 *  \tparam Size is an integral type.
 *  \tparam OutputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre \p result may be equal to \p first, but \p result shall not be in the range <tt>[first, first + n)</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p copy
 *  to copy from one range to another using the \p thrust::device parallelization policy:
 *
 *  \code
 *  #include <thrust/copy.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  size_t n = 100;
 *  thrust::device_vector<int> vec0(n);
 *  thrust::device_vector<int> vec1(n);
 *  ...
 *  thrust::copy_n(thrust::device, vec0.begin(), n, vec1.begin());
 *
 *  // vec1 is now a copy of vec0
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/copy_n
 *  \see thrust::copy
 */
template<typename DerivedPolicy, typename InputIterator, typename Size, typename OutputIterator>
__host__ __device__
  OutputIterator copy_n(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        InputIterator first,
                        Size n,
                        OutputIterator result);


	
/*! \p copy copies elements from the range [\p first, \p last) to the range
 *  [\p result, \p result + (\p last - \p first)). That is, it performs
 *  the assignments *\p result = *\p first, *(\p result + \c 1) = *(\p first + \c 1),
 *  and so on. Generally, for every integer \c n from \c 0 to \p last - \p first, \p copy
 *  performs the assignment *(\p result + \c n) = *(\p first + \c n). Unlike
 *  \c std::copy, \p copy offers no guarantee on order of operation.  As a result,
 *  calling \p copy with overlapping source and destination ranges has undefined
 *  behavior.
 *
 *  The return value is \p result + (\p last - \p first).
 *
 *  \param first The beginning of the sequence to copy.
 *  \param last The end of the sequence to copy.
 *  \param result The destination sequence.
 *  \return The end of the destination sequence.
 *  \see https://en.cppreference.com/w/cpp/algorithm/copy
 *
 *  \tparam InputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and \c InputIterator's \c value_type must be convertible to \c OutputIterator's \c value_type.
 *  \tparam OutputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre \p result may be equal to \p first, but \p result shall not be in the range <tt>[first, last)</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p copy
 *  to copy from one range to another.
 *
 *  \code
 *  #include <thrust/copy.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *
 *  thrust::device_vector<int> vec0(100);
 *  thrust::device_vector<int> vec1(100);
 *  ...
 *
 *  thrust::copy(vec0.begin(), vec0.end(),
 *               vec1.begin());
 *
 *  // vec1 is now a copy of vec0
 *  \endcode
 */
template<typename InputIterator, typename OutputIterator>
  OutputIterator copy(InputIterator first,
                      InputIterator last,
                      OutputIterator result);

/*! \p copy_n copies elements from the range <tt>[first, first + n)</tt> to the range
 *  <tt>[result, result + n)</tt>. That is, it performs the assignments <tt>*result = *first, *(result + 1) = *(first + 1)</tt>,
 *  and so on. Generally, for every integer \c i from \c 0 to \c n, \p copy
 *  performs the assignment *(\p result + \c i) = *(\p first + \c i). Unlike
 *  \c std::copy_n, \p copy_n offers no guarantee on order of operation. As a result,
 *  calling \p copy_n with overlapping source and destination ranges has undefined
 *  behavior.
 *
 *  The return value is \p result + \p n.
 *
 *  \param first The beginning of the range to copy.
 *  \param n The number of elements to copy.
 *  \param result The beginning destination range.
 *  \return The end of the destination range.
 *
 *  \tparam InputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a> and \c InputIterator's \c value_type must be convertible to \c OutputIterator's \c value_type.
 *  \tparam Size is an integral type.
 *  \tparam OutputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *
 *  \pre \p result may be equal to \p first, but \p result shall not be in the range <tt>[first, first + n)</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p copy
 *  to copy from one range to another.
 *
 *  \code
 *  #include <thrust/copy.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  size_t n = 100;
 *  thrust::device_vector<int> vec0(n);
 *  thrust::device_vector<int> vec1(n);
 *  ...
 *  thrust::copy_n(vec0.begin(), n, vec1.begin());
 *
 *  // vec1 is now a copy of vec0
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/copy_n
 *  \see thrust::copy
 */
template<typename InputIterator, typename Size, typename OutputIterator>
  OutputIterator copy_n(InputIterator first,
                        Size n,
                        OutputIterator result);

/*! \} // end copying
 */

/*! \addtogroup stream_compaction
 *  \{
 */


/*! This version of \p copy_if copies elements from the range <tt>[first,last)</tt>
 *  to a range beginning at \p result, except that any element which causes \p pred
 *  to be \c false is not copied. \p copy_if is stable, meaning that the relative
 *  order of elements that are copied is unchanged.
 *
 *  More precisely, for every integer \c n such that <tt>0 <= n < last-first</tt>,
 *  \p copy_if performs the assignment <tt>*result = *(first+n)</tt> and \p result
 *  is advanced one position if <tt>pred(*(first+n))</tt>. Otherwise, no assignment
 *  occurs and \p result is not advanced.
 *
 *  The algorithm's execution is parallelized as determined by \p system.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence from which to copy.
 *  \param last The end of the sequence from which to copy.
 *  \param result The beginning of the sequence into which to copy.
 *  \param pred The predicate to test on every value of the range <tt>[first, last)</tt>.
 *  \return <tt>result + n</tt>, where \c n is equal to the number of times \p pred
 *          evaluated to \c true in the range <tt>[first, last)</tt>.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *                        and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The ranges <tt>[first, last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p copy_if to perform stream compaction
 *  to copy even numbers to an output range using the \p thrust::host parallelization policy:
 *
 *  \code
 *  #include <thrust/copy.h>
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
 *  int result[4];
 *
 *  thrust::copy_if(thrust::host, V, V + N, result, is_even());
 *
 *  // V remains {-2, 0, -1, 0, 1, 2}
 *  // result is now {-2, 0, 0, 2}
 *  \endcode
 *
 *  \see \c remove_copy_if
 */
template<typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate>
__host__ __device__
  OutputIterator copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                         InputIterator first,
                         InputIterator last,
                         OutputIterator result,
                         Predicate pred);



/*! This version of \p copy_if copies elements from the range <tt>[first,last)</tt>
 *  to a range beginning at \p result, except that any element which causes \p pred
 *  to \c false is not copied. \p copy_if is stable, meaning that the relative
 *  order of elements that are copied is unchanged.
 *
 *  More precisely, for every integer \c n such that <tt>0 <= n < last-first</tt>,
 *  \p copy_if performs the assignment <tt>*result = *(first+n)</tt> and \p result
 *  is advanced one position if <tt>pred(*(first+n))</tt>. Otherwise, no assignment
 *  occurs and \p result is not advanced.
 *
 *  \param first The beginning of the sequence from which to copy.
 *  \param last The end of the sequence from which to copy.
 *  \param result The beginning of the sequence into which to copy.
 *  \param pred The predicate to test on every value of the range <tt>[first, last)</tt>.
 *  \return <tt>result + n</tt>, where \c n is equal to the number of times \p pred
 *          evaluated to \c true in the range <tt>[first, last)</tt>.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *                        and \p InputIterator's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The ranges <tt>[first, last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p copy_if to perform stream compaction
 *  to copy even numbers to an output range.
 *
 *  \code
 *  #include <thrust/copy.h>
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
 *  int result[4];
 *
 *  thrust::copy_if(V, V + N, result, is_even());
 *
 *  // V remains {-2, 0, -1, 0, 1, 2}
 *  // result is now {-2, 0, 0, 2}
 *  \endcode
 *
 *  \see \c remove_copy_if
 */
template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator first,
                         InputIterator last,
                         OutputIterator result,
                         Predicate pred);


/*! This version of \p copy_if copies elements from the range <tt>[first,last)</tt>
 *  to a range beginning at \p result, except that any element whose corresponding stencil
 *  element causes \p pred to be \c false is not copied. \p copy_if is stable, meaning
 *  that the relative order of elements that are copied is unchanged.
 *
 *  More precisely, for every integer \c n such that <tt>0 <= n < last-first</tt>,
 *  \p copy_if performs the assignment <tt>*result = *(first+n)</tt> and \p result
 *  is advanced one position if <tt>pred(*(stencil+n))</tt>. Otherwise, no assignment
 *  occurs and \p result is not advanced.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence from which to copy.
 *  \param last The end of the sequence from which to copy.
 *  \param stencil The beginning of the stencil sequence.
 *  \param result The beginning of the sequence into which to copy.
 *  \param pred The predicate to test on every value of the range <tt>[stencil, stencil + (last-first))</tt>.
 *  \return <tt>result + n</tt>, where \c n is equal to the number of times \p pred
 *          evaluated to \c true in the range <tt>[stencil, stencil + (last-first))</tt>.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *                         and \p InputIterator2's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/named_req/OutputIterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The ranges <tt>[first, last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *  \pre The ranges <tt>[stencil, stencil + (last - first))</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p copy_if to perform stream compaction
 *  to copy numbers to an output range when corresponding stencil elements are even using the \p thrust::host execution policy:
 *
 *  \code
 *  #include <thrust/copy.h>
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
 *  int N = 6;
 *  int data[N]    = { 0, 1,  2, 3, 4, 5};
 *  int stencil[N] = {-2, 0, -1, 0, 1, 2};
 *  int result[4];
 *
 *  thrust::copy_if(thrust::host, data, data + N, stencil, result, is_even());
 *
 *  // data remains    = { 0, 1,  2, 3, 4, 5};
 *  // stencil remains = {-2, 0, -1, 0, 1, 2};
 *  // result is now     { 0, 1,  3, 5}
 *  \endcode
 *
 *  \see \c remove_copy_if
 */
template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
__host__ __device__
  OutputIterator copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                         InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred);


/*! This version of \p copy_if copies elements from the range <tt>[first,last)</tt>
 *  to a range beginning at \p result, except that any element whose corresponding stencil
 *  element causes \p pred to be \c false is not copied. \p copy_if is stable, meaning
 *  that the relative order of elements that are copied is unchanged.
 *
 *  More precisely, for every integer \c n such that <tt>0 <= n < last-first</tt>,
 *  \p copy_if performs the assignment <tt>*result = *(first+n)</tt> and \p result
 *  is advanced one position if <tt>pred(*(stencil+n))</tt>. Otherwise, no assignment
 *  occurs and \p result is not advanced.
 *
 *  \param first The beginning of the sequence from which to copy.
 *  \param last The end of the sequence from which to copy.
 *  \param stencil The beginning of the stencil sequence.
 *  \param result The beginning of the sequence into which to copy.
 *  \param pred The predicate to test on every value of the range <tt>[stencil, stencil + (last-first))</tt>.
 *  \return <tt>result + n</tt>, where \c n is equal to the number of times \p pred
 *          evaluated to \c true in the range <tt>[stencil, stencil + (last-first))</tt>.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *                         and \p InputIterator2's \c value_type is convertible to \p Predicate's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/named_req/OutputIterator">Output Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre The ranges <tt>[first, last)</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *  \pre The ranges <tt>[stencil, stencil + (last - first))</tt> and <tt>[result, result + (last - first))</tt> shall not overlap.
 *
 *  The following code snippet demonstrates how to use \p copy_if to perform stream compaction
 *  to copy numbers to an output range when corresponding stencil elements are even:
 *
 *  \code
 *  #include <thrust/copy.h>
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
 *  int N = 6;
 *  int data[N]    = { 0, 1,  2, 3, 4, 5};
 *  int stencil[N] = {-2, 0, -1, 0, 1, 2};
 *  int result[4];
 *
 *  thrust::copy_if(data, data + N, stencil, result, is_even());
 *
 *  // data remains    = { 0, 1,  2, 3, 4, 5};
 *  // stencil remains = {-2, 0, -1, 0, 1, 2};
 *  // result is now     { 0, 1,  3, 5}
 *  \endcode
 *
 *  \see \c remove_copy_if
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred);

/*! \} // end stream_compaction
 */

THRUST_NAMESPACE_END

#include <thrust/detail/copy.h>
#include <thrust/detail/copy_if.h>

