/*
 *  Copyright 2008-2020 NVIDIA Corporation
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

/*! \file shuffle.h
 *  \brief Reorders range by a uniform random permutation
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup reordering
*  \ingroup algorithms
*
*  \addtogroup shuffling
*  \ingroup reordering
*  \{
*/


/*! \p shuffle reorders the elements <tt>[first, last)</tt> by a uniform pseudorandom permutation, defined by
 *  random engine \p g.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to shuffle.
 *  \param last The end of the sequence to shuffle.
 *  \param g A UniformRandomBitGenerator
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam RandomIterator is a random access iterator
 *  \tparam URBG is a uniform random bit generator
 *
 *  The following code snippet demonstrates how to use \p shuffle to create a random permutation
 *  using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/shuffle.h>
 *  #include <thrust/random.h>
 *  #include <thrust/execution_policy.h>
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::default_random_engine g;
 *  thrust::shuffle(thrust::host, A, A + N, g);
 *  // A is now {6, 5, 8, 7, 2, 1, 4, 3, 10, 9}
 *  \endcode
 *
 *  \see \p shuffle_copy
 */
template <typename DerivedPolicy, typename RandomIterator, typename URBG>
__host__ __device__ void shuffle(
    const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
    RandomIterator first, RandomIterator last, URBG&& g);

/*! \p shuffle reorders the elements <tt>[first, last)</tt> by a uniform pseudorandom permutation, defined by
 *  random engine \p g.
 *
 *  \param first The beginning of the sequence to shuffle.
 *  \param last The end of the sequence to shuffle.
 *  \param g A UniformRandomBitGenerator
 *
 *  \tparam RandomIterator is a random access iterator
 *  \tparam URBG is a uniform random bit generator
 *
 *  The following code snippet demonstrates how to use \p shuffle to create a random permutation.
 *
 *  \code
 *  #include <thrust/shuffle.h>
 *  #include <thrust/random.h>
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::default_random_engine g;
 *  thrust::shuffle(A, A + N, g);
 *  // A is now {6, 5, 8, 7, 2, 1, 4, 3, 10, 9}
 *  \endcode
 *
 *  \see \p shuffle_copy
 */
template <typename RandomIterator, typename URBG>
__host__ __device__ void shuffle(RandomIterator first, RandomIterator last,
                                 URBG&& g);

/*! shuffle_copy differs from shuffle only in that the reordered sequence is written to different output sequences, rather than in place.
 *  \p shuffle_copy reorders the elements <tt>[first, last)</tt> by a uniform pseudorandom permutation, defined by
 *  random engine \p g.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.

 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to shuffle.
 *  \param last The end of the sequence to shuffle.
 *  \param result Destination of shuffled sequence
 *  \param g A UniformRandomBitGenerator
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam RandomIterator is a random access iterator
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam URBG is a uniform random bit generator
 *
 *  The following code snippet demonstrates how to use \p shuffle_copy to create a random permutation.
 *
 *  \code
 *  #include <thrust/shuffle.h>
 *  #include <thrust/random.h>
 *  #include <thrust/execution_policy.h>
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  int result[10];
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::default_random_engine g;
 *  thrust::shuffle_copy(thrust::host, A, A + N, result, g);
 *  // result is now {6, 5, 8, 7, 2, 1, 4, 3, 10, 9}
 *  \endcode
 *
 *  \see \p shuffle
 */
template <typename DerivedPolicy, typename RandomIterator,
          typename OutputIterator, typename URBG>
__host__ __device__ void shuffle_copy(
    const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
    RandomIterator first, RandomIterator last, OutputIterator result, URBG&& g);

/*! shuffle_copy differs from shuffle only in that the reordered sequence is written to different output sequences, rather than in place.
 *\p shuffle_copy reorders the elements <tt>[first, last)</tt> by a uniform pseudorandom permutation, defined by
 *  random engine \p g.
 *
 *  \param first The beginning of the sequence to shuffle.
 *  \param last The end of the sequence to shuffle.
 *  \param result Destination of shuffled sequence
 *  \param g A UniformRandomBitGenerator
 *
 *  \tparam RandomIterator is a random access iterator
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam URBG is a uniform random bit generator
 *
 *  The following code snippet demonstrates how to use \p shuffle_copy to create a random permutation.
 *
 *  \code
 *  #include <thrust/shuffle.h>
 *  #include <thrust/random.h>
 *  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
 *  int result[10];
 *  const int N = sizeof(A)/sizeof(int);
 *  thrust::default_random_engine g;
 *  thrust::shuffle_copy(A, A + N, result, g);
 *  // result is now {6, 5, 8, 7, 2, 1, 4, 3, 10, 9}
 *  \endcode
 *
 *  \see \p shuffle
 */
template <typename RandomIterator, typename OutputIterator, typename URBG>
__host__ __device__ void shuffle_copy(RandomIterator first, RandomIterator last,
                                      OutputIterator result, URBG&& g);

THRUST_NAMESPACE_END

#include <thrust/detail/shuffle.inl>
#endif
