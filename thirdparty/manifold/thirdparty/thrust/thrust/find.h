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


/*! \file find.h
 *  \brief Locating values in (unsorted) ranges
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup algorithms
 */

/*! \addtogroup searching
 *  \ingroup algorithms
 *  \{
 */


/*! \p find returns the first iterator \c i in the range 
 *  <tt>[first, last)</tt> such that <tt>*i == value</tt>
 *  or \c last if no such iterator exists.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first Beginning of the sequence to search.
 *  \param last End of the sequence to search.
 *  \param value The value to find.
 *  \return The first iterator \c i such that <tt>*i == value</tt> or \c last.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *          and \p InputIterator's \c value_type is equality comparable to type \c T.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">EqualityComparable</a>. 
 *
 *  \code
 *  #include <thrust/find.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  thrust::device_vector<int> input(4);
 *
 *  input[0] = 0;
 *  input[1] = 5;
 *  input[2] = 3;
 *  input[3] = 7;
 *
 *  thrust::device_vector<int>::iterator iter;
 *
 *  iter = thrust::find(thrust::device, input.begin(), input.end(), 3); // returns input.first() + 2
 *  iter = thrust::find(thrust::device, input.begin(), input.end(), 5); // returns input.first() + 1
 *  iter = thrust::find(thrust::device, input.begin(), input.end(), 9); // returns input.end()
 *  \endcode
 *
 *  \see find_if
 *  \see mismatch
 */
template<typename DerivedPolicy, typename InputIterator, typename T>
__host__ __device__
InputIterator find(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   InputIterator first,
                   InputIterator last,
                   const T& value);


/*! \p find returns the first iterator \c i in the range 
 *  <tt>[first, last)</tt> such that <tt>*i == value</tt>
 *  or \c last if no such iterator exists.
 *
 *  \param first Beginning of the sequence to search.
 *  \param last End of the sequence to search.
 *  \param value The value to find.
 *  \return The first iterator \c i such that <tt>*i == value</tt> or \c last.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *          and \p InputIterator's \c value_type is equality comparable to type \c T.
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">EqualityComparable</a>. 
 *
 *  \code
 *  #include <thrust/find.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> input(4);
 *
 *  input[0] = 0;
 *  input[1] = 5;
 *  input[2] = 3;
 *  input[3] = 7;
 *
 *  thrust::device_vector<int>::iterator iter;
 *
 *  iter = thrust::find(input.begin(), input.end(), 3); // returns input.first() + 2
 *  iter = thrust::find(input.begin(), input.end(), 5); // returns input.first() + 1
 *  iter = thrust::find(input.begin(), input.end(), 9); // returns input.end()
 *  \endcode
 *
 *  \see find_if
 *  \see mismatch
 */
template <typename InputIterator, typename T>
InputIterator find(InputIterator first,
                   InputIterator last,
                   const T& value);


/*! \p find_if returns the first iterator \c i in the range 
 *  <tt>[first, last)</tt> such that <tt>pred(*i)</tt> is \c true
 *  or \c last if no such iterator exists.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first Beginning of the sequence to search.
 *  \param last End of the sequence to search.
 *  \param pred A predicate used to test range elements.
 *  \return The first iterator \c i such that <tt>pred(*i)</tt> is \c true, or \c last.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \code
 *  #include <thrust/find.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *
 *  struct greater_than_four
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x > 4;
 *    }
 *  };
 *
 *  struct greater_than_ten
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x > 10;
 *    }
 *  };
 *
 *  ...
 *  thrust::device_vector<int> input(4);
 *
 *  input[0] = 0;
 *  input[1] = 5;
 *  input[2] = 3;
 *  input[3] = 7;
 *
 *  thrust::device_vector<int>::iterator iter;
 *
 *  iter = thrust::find_if(thrust::device, input.begin(), input.end(), greater_than_four()); // returns input.first() + 1
 *
 *  iter = thrust::find_if(thrust::device, input.begin(), input.end(), greater_than_ten());  // returns input.end()
 *  \endcode
 *
 *  \see find
 *  \see find_if_not
 *  \see mismatch
 */
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__host__ __device__
InputIterator find_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred);


/*! \p find_if returns the first iterator \c i in the range 
 *  <tt>[first, last)</tt> such that <tt>pred(*i)</tt> is \c true
 *  or \c last if no such iterator exists.
 *
 *  \param first Beginning of the sequence to search.
 *  \param last End of the sequence to search.
 *  \param pred A predicate used to test range elements.
 *  \return The first iterator \c i such that <tt>pred(*i)</tt> is \c true, or \c last.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \code
 *  #include <thrust/find.h>
 *  #include <thrust/device_vector.h>
 *
 *  struct greater_than_four
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x > 4;
 *    }
 *  };
 *
 *  struct greater_than_ten
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x > 10;
 *    }
 *  };
 *
 *  ...
 *  thrust::device_vector<int> input(4);
 *
 *  input[0] = 0;
 *  input[1] = 5;
 *  input[2] = 3;
 *  input[3] = 7;
 *
 *  thrust::device_vector<int>::iterator iter;
 *
 *  iter = thrust::find_if(input.begin(), input.end(), greater_than_four()); // returns input.first() + 1
 *
 *  iter = thrust::find_if(input.begin(), input.end(), greater_than_ten());  // returns input.end()
 *  \endcode
 *
 *  \see find
 *  \see find_if_not
 *  \see mismatch
 */
template <typename InputIterator, typename Predicate>
InputIterator find_if(InputIterator first,
                      InputIterator last,
                      Predicate pred);


/*! \p find_if_not returns the first iterator \c i in the range 
 *  <tt>[first, last)</tt> such that <tt>pred(*i)</tt> is \c false
 *  or \c last if no such iterator exists.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first Beginning of the sequence to search.
 *  \param last End of the sequence to search.
 *  \param pred A predicate used to test range elements.
 *  \return The first iterator \c i such that <tt>pred(*i)</tt> is \c false, or \c last.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \code
 *  #include <thrust/find.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *
 *  struct greater_than_four
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x > 4;
 *    }
 *  };
 *
 *  struct greater_than_ten
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x > 10;
 *    }
 *  };
 *
 *  ...
 *  thrust::device_vector<int> input(4);
 *
 *  input[0] = 0;
 *  input[1] = 5;
 *  input[2] = 3;
 *  input[3] = 7;
 *
 *  thrust::device_vector<int>::iterator iter;
 *
 *  iter = thrust::find_if_not(thrust::device, input.begin(), input.end(), greater_than_four()); // returns input.first()
 *
 *  iter = thrust::find_if_not(thrust::device, input.begin(), input.end(), greater_than_ten());  // returns input.first()
 *  \endcode
 *
 *  \see find
 *  \see find_if
 *  \see mismatch
 */
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__host__ __device__
InputIterator find_if_not(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          InputIterator first,
                          InputIterator last,
                          Predicate pred);


/*! \p find_if_not returns the first iterator \c i in the range 
 *  <tt>[first, last)</tt> such that <tt>pred(*i)</tt> is \c false
 *  or \c last if no such iterator exists.
 *
 *  \param first Beginning of the sequence to search.
 *  \param last End of the sequence to search.
 *  \param pred A predicate used to test range elements.
 *  \return The first iterator \c i such that <tt>pred(*i)</tt> is \c false, or \c last.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \code
 *  #include <thrust/find.h>
 *  #include <thrust/device_vector.h>
 *
 *  struct greater_than_four
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x > 4;
 *    }
 *  };
 *
 *  struct greater_than_ten
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x > 10;
 *    }
 *  };
 *
 *  ...
 *  thrust::device_vector<int> input(4);
 *
 *  input[0] = 0;
 *  input[1] = 5;
 *  input[2] = 3;
 *  input[3] = 7;
 *
 *  thrust::device_vector<int>::iterator iter;
 *
 *  iter = thrust::find_if_not(input.begin(), input.end(), greater_than_four()); // returns input.first()
 *
 *  iter = thrust::find_if_not(input.begin(), input.end(), greater_than_ten());  // returns input.first()
 *  \endcode
 *
 *  \see find
 *  \see find_if
 *  \see mismatch
 */
template <typename InputIterator, typename Predicate>
InputIterator find_if_not(InputIterator first,
                          InputIterator last,
                          Predicate pred);

/*! \} // end searching
 */

THRUST_NAMESPACE_END

#include <thrust/detail/find.inl>
