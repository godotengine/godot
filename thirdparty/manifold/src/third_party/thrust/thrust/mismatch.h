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


/*! \file mismatch.h
 *  \brief Search for differences between ranges
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup algorithms
 */

/*! \addtogroup searching
 *  \ingroup algorithms
 *  \{
 */


/*! \p mismatch finds the first position where the two ranges <tt>[first1, last1)</tt>
 *  and <tt>[first2, first2 + (last1 - first1))</tt> differ. The two versions of 
 *  \p mismatch use different tests for whether elements differ.
 *
 *  This version of \p mismatch finds the first iterator \c i in <tt>[first1, last1)</tt>
 *  such that <tt>*i == *(first2 + (i - first1))</tt> is \c false. The return value is a
 *  \c pair whose first element is \c i and whose second element is <tt>*(first2 + (i - first1))</tt>.
 *  If no such iterator \c i exists, the return value is a \c pair whose first element
 *  is \c last1 and whose second element is <tt>*(first2 + (last1 - first1))</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first sequence.
 *  \param last1  The end of the first sequence.
 *  \param first2 The beginning of the second sequence.
 *  \return The first position where the sequences differ.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *          and \p InputIterator1's \c value_type is equality comparable to \p InputIterator2's \c value_type.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *
 *  \code
 *  #include <thrust/mismatch.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> vec1(4);
 *  thrust::device_vector<int> vec2(4);
 *
 *  vec1[0] = 0;  vec2[0] = 0; 
 *  vec1[1] = 5;  vec2[1] = 5;
 *  vec1[2] = 3;  vec2[2] = 8;
 *  vec1[3] = 7;  vec2[3] = 7;
 *
 *  typedef thrust::device_vector<int>::iterator Iterator;
 *  thrust::pair<Iterator,Iterator> result;
 *
 *  result = thrust::mismatch(thrust::device, vec1.begin(), vec1.end(), vec2.begin());
 *
 *  // result.first  is vec1.begin() + 2
 *  // result.second is vec2.begin() + 2
 *  \endcode
 *
 *  \see find
 *  \see find_if
 */
template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2>
__host__ __device__
thrust::pair<InputIterator1, InputIterator2> mismatch(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                                      InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2);


/*! \p mismatch finds the first position where the two ranges <tt>[first1, last1)</tt>
 * and <tt>[first2, first2 + (last1 - first1))</tt> differ. The two versions of 
 * \p mismatch use different tests for whether elements differ.
 *
 * This version of \p mismatch finds the first iterator \c i in <tt>[first1, last1)</tt>
 * such that <tt>*i == *(first2 + (i - first1))</tt> is \c false. The return value is a
 * \c pair whose first element is \c i and whose second element is <tt>*(first2 + (i - first1))</tt>.
 * If no such iterator \c i exists, the return value is a \c pair whose first element
 * is \c last1 and whose second element is <tt>*(first2 + (last1 - first1))</tt>.
 *
 *  \param first1 The beginning of the first sequence.
 *  \param last1  The end of the first sequence.
 *  \param first2 The beginning of the second sequence.
 *  \return The first position where the sequences differ.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *          and \p InputIterator1's \c value_type is equality comparable to \p InputIterator2's \c value_type.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *
 *  \code
 *  #include <thrust/mismatch.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> vec1(4);
 *  thrust::device_vector<int> vec2(4);
 *
 *  vec1[0] = 0;  vec2[0] = 0; 
 *  vec1[1] = 5;  vec2[1] = 5;
 *  vec1[2] = 3;  vec2[2] = 8;
 *  vec1[3] = 7;  vec2[3] = 7;
 *
 *  typedef thrust::device_vector<int>::iterator Iterator;
 *  thrust::pair<Iterator,Iterator> result;
 *
 *  result = thrust::mismatch(vec1.begin(), vec1.end(), vec2.begin());
 *
 *  // result.first  is vec1.begin() + 2
 *  // result.second is vec2.begin() + 2
 *  \endcode
 *
 *  \see find
 *  \see find_if
 */
template <typename InputIterator1, typename InputIterator2>
thrust::pair<InputIterator1, InputIterator2> mismatch(InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2);


/*! \p mismatch finds the first position where the two ranges <tt>[first1, last1)</tt>
 *  and <tt>[first2, first2 + (last1 - first1))</tt> differ. The two versions of 
 *  \p mismatch use different tests for whether elements differ.
 *
 *  This version of \p mismatch finds the first iterator \c i in <tt>[first1, last1)</tt>
 *  such that <tt>pred(\*i, \*(first2 + (i - first1))</tt> is \c false. The return value is a
 *  \c pair whose first element is \c i and whose second element is <tt>*(first2 + (i - first1))</tt>.
 *  If no such iterator \c i exists, the return value is a \c pair whose first element is
 *  \c last1 and whose second element is <tt>*(first2 + (last1 - first1))</tt>.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first sequence.
 *  \param last1  The end of the first sequence.
 *  \param first2 The beginning of the second sequence.
 *  \param pred   The binary predicate to compare elements.
 *  \return The first position where the sequences differ.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Input Iterator</a>.
 *
 *  \code
 *  #include <thrust/mismatch.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> vec1(4);
 *  thrust::device_vector<int> vec2(4);
 *
 *  vec1[0] = 0;  vec2[0] = 0; 
 *  vec1[1] = 5;  vec2[1] = 5;
 *  vec1[2] = 3;  vec2[2] = 8;
 *  vec1[3] = 7;  vec2[3] = 7;
 *
 *  typedef thrust::device_vector<int>::iterator Iterator;
 *  thrust::pair<Iterator,Iterator> result;
 *
 *  result = thrust::mismatch(thrust::device, vec1.begin(), vec1.end(), vec2.begin(), thrust::equal_to<int>());
 *
 *  // result.first  is vec1.begin() + 2
 *  // result.second is vec2.begin() + 2
 *  \endcode
 *
 *  \see find
 *  \see find_if
 */
template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
__host__ __device__
thrust::pair<InputIterator1, InputIterator2> mismatch(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                                      InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2,
                                                      BinaryPredicate pred);


/*! \p mismatch finds the first position where the two ranges <tt>[first1, last1)</tt>
 * and <tt>[first2, first2 + (last1 - first1))</tt> differ. The two versions of 
 * \p mismatch use different tests for whether elements differ.
 *
 * This version of \p mismatch finds the first iterator \c i in <tt>[first1, last1)</tt>
 * such that <tt>pred(\*i, \*(first2 + (i - first1))</tt> is \c false. The return value is a
 * \c pair whose first element is \c i and whose second element is <tt>*(first2 + (i - first1))</tt>.
 * If no such iterator \c i exists, the return value is a \c pair whose first element is
 * \c last1 and whose second element is <tt>*(first2 + (last1 - first1))</tt>.
 *
 *  \param first1 The beginning of the first sequence.
 *  \param last1  The end of the first sequence.
 *  \param first2 The beginning of the second sequence.
 *  \param pred   The binary predicate to compare elements.
 *  \return The first position where the sequences differ.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Input Iterator</a>.
 *
 *  \code
 *  #include <thrust/mismatch.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> vec1(4);
 *  thrust::device_vector<int> vec2(4);
 *
 *  vec1[0] = 0;  vec2[0] = 0; 
 *  vec1[1] = 5;  vec2[1] = 5;
 *  vec1[2] = 3;  vec2[2] = 8;
 *  vec1[3] = 7;  vec2[3] = 7;
 *
 *  typedef thrust::device_vector<int>::iterator Iterator;
 *  thrust::pair<Iterator,Iterator> result;
 *
 *  result = thrust::mismatch(vec1.begin(), vec1.end(), vec2.begin(), thrust::equal_to<int>());
 *
 *  // result.first  is vec1.begin() + 2
 *  // result.second is vec2.begin() + 2
 *  \endcode
 *
 *  \see find
 *  \see find_if
 */
template <typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
thrust::pair<InputIterator1, InputIterator2> mismatch(InputIterator1 first1,
                                                      InputIterator1 last1,
                                                      InputIterator2 first2,
                                                      BinaryPredicate pred);

/*! \} // end searching
 */

THRUST_NAMESPACE_END

#include <thrust/detail/mismatch.inl>
