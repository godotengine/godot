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


/*! \file thrust/transform.h
 *  \brief Transforms input ranges using a function object
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup algorithms
 */

/*! \addtogroup transformations
 *  \ingroup algorithms
 *  \{
 */


/*! This version of \p transform applies a unary function to each element
 *  of an input sequence and stores the result in the corresponding 
 *  position in an output sequence.  Specifically, for each iterator 
 *  <tt>i</tt> in the range [\p first, \p last) the operation 
 *  <tt>op(*i)</tt> is performed and the result is assigned to <tt>*o</tt>,
 *  where <tt>o</tt> is the corresponding output iterator in the range
 *  [\p result, \p result + (\p last - \p first) ).  The input and
 *  output sequences may coincide, resulting in an in-place transformation.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *    
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param op The transformation operation.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to \c UnaryFunction's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam UnaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>
 *                              and \c UnaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, last)</tt> shall not overlap the range <tt>[result, result + (last - first))</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform to negate a range in-place
 *  using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
 * 
 *  thrust::negate<int> op;
 *
 *  thrust::transform(thrust::host, data, data + 10, data, op); // in-place transformation
 *
 *  // data is now {5, 0, -2, 3, -2, -4, 0, 1, -2, -8};
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/transform
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
__host__ __device__
  OutputIterator transform(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction op);

	
/*! This version of \p transform applies a unary function to each element
 *  of an input sequence and stores the result in the corresponding 
 *  position in an output sequence.  Specifically, for each iterator 
 *  <tt>i</tt> in the range [\p first, \p last) the operation 
 *  <tt>op(*i)</tt> is performed and the result is assigned to <tt>*o</tt>,
 *  where <tt>o</tt> is the corresponding output iterator in the range
 *  [\p result, \p result + (\p last - \p first) ).  The input and
 *  output sequences may coincide, resulting in an in-place transformation.
 *    
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param op The tranformation operation.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator's \c value_type is convertible to \c UnaryFunction's \c argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam UnaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>
 *                              and \c UnaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, last)</tt> shall not overlap the range <tt>[result, result + (last - first))</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform
 *
 *  \code
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  
 *  int data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
 * 
 *  thrust::negate<int> op;
 *
 *  thrust::transform(data, data + 10, data, op); // in-place transformation
 *
 *  // data is now {5, 0, -2, 3, -2, -4, 0, 1, -2, -8};
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/transform
 */
template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction op);


/*! This version of \p transform applies a binary function to each pair
 *  of elements from two input sequences and stores the result in the
 *  corresponding position in an output sequence.  Specifically, for
 *  each iterator <tt>i</tt> in the range [\p first1, \p last1) and 
 *  <tt>j = first + (i - first1)</tt> in the range [\p first2, \p last2)
 *  the operation <tt>op(*i,*j)</tt> is performed and the result is 
 *  assigned to <tt>*o</tt>,  where <tt>o</tt> is the corresponding
 *  output iterator in the range [\p result, \p result + (\p last - \p first) ).
 *  The input and output sequences may coincide, resulting in an 
 *  in-place transformation.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *    
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input sequence.
 *  \param last1 The end of the first input sequence.
 *  \param first2 The beginning of the second input sequence.
 *  \param result The beginning of the output sequence.
 *  \param op The tranformation operation.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator1's \c value_type is convertible to \c BinaryFunction's \c first_argument_type.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator2's \c value_type is convertible to \c BinaryFunction's \c second_argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam BinaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                              and \c BinaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first1 may equal \p result, but the range <tt>[first1, last1)</tt> shall not overlap the range <tt>[result, result + (last1 - first1))</tt> otherwise.
 *  \pre \p first2 may equal \p result, but the range <tt>[first2, first2 + (last1 - first1))</tt> shall not overlap the range <tt>[result, result + (last1 - first1))</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform to compute the sum of two
 *  ranges using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int input1[6] = {-5,  0,  2,  3,  2,  4};
 *  int input2[6] = { 3,  6, -2,  1,  2,  3};
 *  int output[6];
 * 
 *  thrust::plus<int> op;
 *
 *  thrust::transform(thrust::host, input1, input1 + 6, input2, output, op);
 *
 *  // output is now {-2,  6,  0,  4,  4,  7};
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/transform
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
__host__ __device__
  OutputIterator transform(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op);


/*! This version of \p transform applies a binary function to each pair
 *  of elements from two input sequences and stores the result in the
 *  corresponding position in an output sequence.  Specifically, for
 *  each iterator <tt>i</tt> in the range [\p first1, \p last1) and 
 *  <tt>j = first + (i - first1)</tt> in the range [\p first2, \p last2)
 *  the operation <tt>op(*i,*j)</tt> is performed and the result is 
 *  assigned to <tt>*o</tt>,  where <tt>o</tt> is the corresponding
 *  output iterator in the range [\p result, \p result + (\p last - \p first) ).
 *  The input and output sequences may coincide, resulting in an 
 *  in-place transformation.
 *    
 *  \param first1 The beginning of the first input sequence.
 *  \param last1 The end of the first input sequence.
 *  \param first2 The beginning of the second input sequence.
 *  \param result The beginning of the output sequence.
 *  \param op The tranformation operation.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator1's \c value_type is convertible to \c BinaryFunction's \c first_argument_type.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                        and \c InputIterator2's \c value_type is convertible to \c BinaryFunction's \c second_argument_type.
 *  \tparam OutputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output Iterator</a>.
 *  \tparam BinaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                              and \c BinaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *
 *  \pre \p first1 may equal \p result, but the range <tt>[first1, last1)</tt> shall not overlap the range <tt>[result, result + (last1 - first1))</tt> otherwise.
 *  \pre \p first2 may equal \p result, but the range <tt>[first2, first2 + (last1 - first1))</tt> shall not overlap the range <tt>[result, result + (last1 - first1))</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform
 *
 *  \code
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  
 *  int input1[6] = {-5,  0,  2,  3,  2,  4};
 *  int input2[6] = { 3,  6, -2,  1,  2,  3};
 *  int output[6];
 * 
 *  thrust::plus<int> op;
 *
 *  thrust::transform(input1, input1 + 6, input2, output, op);
 *
 *  // output is now {-2,  6,  0,  4,  4,  7};
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/transform
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op);


/*! This version of \p transform_if conditionally applies a unary function
 *  to each element of an input sequence and stores the result in the corresponding 
 *  position in an output sequence if the corresponding position in the input sequence
 *  satifies a predicate. Otherwise, the corresponding position in the
 *  output sequence is not modified.
 *
 *  Specifically, for each iterator <tt>i</tt> in the range <tt>[first, last)</tt> the
 *  predicate <tt>pred(*i)</tt> is evaluated. If this predicate
 *  evaluates to \c true, the result of <tt>op(*i)</tt> is assigned to <tt>*o</tt>,
 *  where <tt>o</tt> is the corresponding output iterator in the range
 *  <tt>[result, result + (last - first) )</tt>. Otherwise, <tt>op(*i)</tt> is
 *  not evaluated and no assignment occurs. The input and output sequences may coincide,
 *  resulting in an in-place transformation.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *    
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param op The tranformation operation.
 *  \param pred The predicate operation.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *                        and \c InputIterator's \c value_type is convertible to \c Predicate's \c argument_type,
 *                        and \c InputIterator's \c value_type is convertible to \c UnaryFunction's \c argument_type.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam UnaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>
 *                        and \c UnaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, last)</tt> shall not overlap the range <tt>[result, result + (last - first))</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform_if to negate the odd-valued
 *  elements of a range using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int data[10]    = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
 *
 *  struct is_odd
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x % 2;
 *    }
 *  };
 * 
 *  thrust::negate<int> op;
 *  thrust::identity<int> identity;
 *
 *  // negate odd elements
 *  thrust::transform_if(thrust::host, data, data + 10, data, op, is_odd()); // in-place transformation
 *
 *  // data is now {5, 0, 2, 3, 2, 4, 0, 1, 2, 8};
 *  \endcode
 *
 *  \see thrust::transform
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
__host__ __device__
  ForwardIterator transform_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                               InputIterator first, InputIterator last,
                               ForwardIterator result,
                               UnaryFunction op,
                               Predicate pred);


/*! This version of \p transform_if conditionally applies a unary function
 *  to each element of an input sequence and stores the result in the corresponding 
 *  position in an output sequence if the corresponding position in the input sequence
 *  satifies a predicate. Otherwise, the corresponding position in the
 *  output sequence is not modified.
 *
 *  Specifically, for each iterator <tt>i</tt> in the range <tt>[first, last)</tt> the
 *  predicate <tt>pred(*i)</tt> is evaluated. If this predicate
 *  evaluates to \c true, the result of <tt>op(*i)</tt> is assigned to <tt>*o</tt>,
 *  where <tt>o</tt> is the corresponding output iterator in the range
 *  <tt>[result, result + (last - first) )</tt>. Otherwise, <tt>op(*i)</tt> is
 *  not evaluated and no assignment occurs. The input and output sequences may coincide,
 *  resulting in an in-place transformation.
 *    
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param result The beginning of the output sequence.
 *  \param op The tranformation operation.
 *  \param pred The predicate operation.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>,
 *                        and \c InputIterator's \c value_type is convertible to \c Predicate's \c argument_type,
 *                        and \c InputIterator's \c value_type is convertible to \c UnaryFunction's \c argument_type.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam UnaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>
 *                        and \c UnaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, last)</tt> shall not overlap the range <tt>[result, result + (last - first))</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform_if:
 *
 *  \code
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  
 *  int data[10]    = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
 *
 *  struct is_odd
 *  {
 *    __host__ __device__
 *    bool operator()(int x)
 *    {
 *      return x % 2;
 *    }
 *  };
 * 
 *  thrust::negate<int> op;
 *  thrust::identity<int> identity;
 *
 *  // negate odd elements
 *  thrust::transform_if(data, data + 10, data, op, is_odd()); // in-place transformation
 *
 *  // data is now {5, 0, 2, 3, 2, 4, 0, 1, 2, 8};
 *  \endcode
 *
 *  \see thrust::transform
 */
template<typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator first, InputIterator last,
                               ForwardIterator result,
                               UnaryFunction op,
                               Predicate pred);


/*! This version of \p transform_if conditionally applies a unary function
 *  to each element of an input sequence and stores the result in the corresponding 
 *  position in an output sequence if the corresponding position in a stencil sequence
 *  satisfies a predicate. Otherwise, the corresponding position in the
 *  output sequence is not modified.
 *
 *  Specifically, for each iterator <tt>i</tt> in the range <tt>[first, last)</tt> the
 *  predicate <tt>pred(*s)</tt> is evaluated, where <tt>s</tt> is the corresponding input
 *  iterator in the range <tt>[stencil, stencil + (last - first) )</tt>. If this predicate
 *  evaluates to \c true, the result of <tt>op(*i)</tt> is assigned to <tt>*o</tt>,
 *  where <tt>o</tt> is the corresponding output iterator in the range
 *  <tt>[result, result + (last - first) )</tt>. Otherwise, <tt>op(*i)</tt> is
 *  not evaluated and no assignment occurs. The input and output sequences may coincide,
 *  resulting in an in-place transformation.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *    
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param stencil The beginning of the stencil sequence.
 *  \param result The beginning of the output sequence.
 *  \param op The tranformation operation.
 *  \param pred The predicate operation.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator1's \c value_type is convertible to \c UnaryFunction's \c argument_type.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator2's \c value_type is convertible to \c Predicate's \c argument_type.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam UnaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>
 *                        and \c UnaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, last)</tt> shall not overlap the range <tt>[result, result + (last - first))</tt> otherwise.
 *  \pre \p stencil may equal \p result, but the range <tt>[stencil, stencil + (last - first))</tt> shall not overlap the range <tt>[result, result + (last - first))</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform_if using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int data[10]    = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
 *  int stencil[10] = { 1, 0, 1,  0, 1, 0, 1,  0, 1, 0};
 * 
 *  thrust::negate<int> op;
 *  thrust::identity<int> identity;
 *
 *  thrust::transform_if(thrust::host, data, data + 10, stencil, data, op, identity); // in-place transformation
 *
 *  // data is now {5, 0, -2, -3, -2,  4, 0, -1, -2,  8};
 *  \endcode
 *
 *  \see thrust::transform
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
__host__ __device__
  ForwardIterator transform_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                               InputIterator1 first, InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction op,
                               Predicate pred);


/*! This version of \p transform_if conditionally applies a unary function
 *  to each element of an input sequence and stores the result in the corresponding 
 *  position in an output sequence if the corresponding position in a stencil sequence
 *  satisfies a predicate. Otherwise, the corresponding position in the
 *  output sequence is not modified.
 *
 *  Specifically, for each iterator <tt>i</tt> in the range <tt>[first, last)</tt> the
 *  predicate <tt>pred(*s)</tt> is evaluated, where <tt>s</tt> is the corresponding input
 *  iterator in the range <tt>[stencil, stencil + (last - first) )</tt>. If this predicate
 *  evaluates to \c true, the result of <tt>op(*i)</tt> is assigned to <tt>*o</tt>,
 *  where <tt>o</tt> is the corresponding output iterator in the range
 *  <tt>[result, result + (last - first) )</tt>. Otherwise, <tt>op(*i)</tt> is
 *  not evaluated and no assignment occurs. The input and output sequences may coincide,
 *  resulting in an in-place transformation.
 *    
 *  \param first The beginning of the input sequence.
 *  \param last The end of the input sequence.
 *  \param stencil The beginning of the stencil sequence.
 *  \param result The beginning of the output sequence.
 *  \param op The tranformation operation.
 *  \param pred The predicate operation.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator1's \c value_type is convertible to \c UnaryFunction's \c argument_type.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator2's \c value_type is convertible to \c Predicate's \c argument_type.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam UnaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/unary_function">Unary Function</a>
 *                        and \c UnaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre \p first may equal \p result, but the range <tt>[first, last)</tt> shall not overlap the range <tt>[result, result + (last - first))</tt> otherwise.
 *  \pre \p stencil may equal \p result, but the range <tt>[stencil, stencil + (last - first))</tt> shall not overlap the range <tt>[result, result + (last - first))</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform_if:
 *
 *  \code
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  
 *  int data[10]    = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
 *  int stencil[10] = { 1, 0, 1,  0, 1, 0, 1,  0, 1, 0};
 * 
 *  thrust::negate<int> op;
 *  thrust::identity<int> identity;
 *
 *  thrust::transform_if(data, data + 10, stencil, data, op, identity); // in-place transformation
 *
 *  // data is now {5, 0, -2, -3, -2,  4, 0, -1, -2,  8};
 *  \endcode
 *
 *  \see thrust::transform
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first, InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction op,
                               Predicate pred);


/*! This version of \p transform_if conditionally applies a binary function
 *  to each pair of elements from two input sequences and stores the result in the corresponding 
 *  position in an output sequence if the corresponding position in a stencil sequence
 *  satifies a predicate. Otherwise, the corresponding position in the
 *  output sequence is not modified.
 *
 *  Specifically, for each iterator <tt>i</tt> in the range <tt>[first1, last1)</tt> and 
 *  <tt>j = first2 + (i - first1)</tt> in the range <tt>[first2, first2 + (last1 - first1) )</tt>,
 *  the predicate <tt>pred(*s)</tt> is evaluated, where <tt>s</tt> is the corresponding input
 *  iterator in the range <tt>[stencil, stencil + (last1 - first1) )</tt>. If this predicate
 *  evaluates to \c true, the result of <tt>binary_op(*i,*j)</tt> is assigned to <tt>*o</tt>,
 *  where <tt>o</tt> is the corresponding output iterator in the range
 *  <tt>[result, result + (last1 - first1) )</tt>. Otherwise, <tt>binary_op(*i,*j)</tt> is
 *  not evaluated and no assignment occurs. The input and output sequences may coincide,
 *  resulting in an in-place transformation.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *    
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input sequence.
 *  \param last1 The end of the first input sequence.
 *  \param first2 The beginning of the second input sequence.
 *  \param stencil The beginning of the stencil sequence.
 *  \param result The beginning of the output sequence.
 *  \param binary_op The transformation operation.
 *  \param pred The predicate operation.
 *  \return The end of the output sequence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator1's \c value_type is convertible to \c BinaryFunction's \c first_argument_type.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator2's \c value_type is convertible to \c BinaryFunction's \c second_argument_type.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam BinaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                         and \c BinaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre \p first1 may equal \p result, but the range <tt>[first1, last1)</tt> shall not overlap the range <tt>[result, result + (last1 - first1))</tt> otherwise.
 *  \pre \p first2 may equal \p result, but the range <tt>[first2, first2 + (last1 - first1))</tt> shall not overlap the range <tt>[result, result + (last1 - first1))</tt> otherwise.
 *  \pre \p stencil may equal \p result, but the range <tt>[stencil, stencil + (last1 - first1))</tt> shall not overlap the range <tt>[result, result + (last1 - first1))</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform_if using the \p thrust::host
 *  execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  
 *  int input1[6]  = {-5,  0,  2,  3,  2,  4};
 *  int input2[6]  = { 3,  6, -2,  1,  2,  3};
 *  int stencil[8] = { 1,  0,  1,  0,  1,  0};
 *  int output[6];
 * 
 *  thrust::plus<int> op;
 *  thrust::identity<int> identity;
 *
 *  thrust::transform_if(thrust::host, input1, input1 + 6, input2, stencil, output, op, identity);
 *
 *  // output is now {-2,  0,  0,  3,  4,  4};
 *  \endcode
 *
 *  \see thrust::transform
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
__host__ __device__
  ForwardIterator transform_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                               InputIterator1 first1, InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred);


/*! This version of \p transform_if conditionally applies a binary function
 *  to each pair of elements from two input sequences and stores the result in the corresponding 
 *  position in an output sequence if the corresponding position in a stencil sequence
 *  satifies a predicate. Otherwise, the corresponding position in the
 *  output sequence is not modified.
 *
 *  Specifically, for each iterator <tt>i</tt> in the range <tt>[first1, last1)</tt> and 
 *  <tt>j = first2 + (i - first1)</tt> in the range <tt>[first2, first2 + (last1 - first1) )</tt>,
 *  the predicate <tt>pred(*s)</tt> is evaluated, where <tt>s</tt> is the corresponding input
 *  iterator in the range <tt>[stencil, stencil + (last1 - first1) )</tt>. If this predicate
 *  evaluates to \c true, the result of <tt>binary_op(*i,*j)</tt> is assigned to <tt>*o</tt>,
 *  where <tt>o</tt> is the corresponding output iterator in the range
 *  <tt>[result, result + (last1 - first1) )</tt>. Otherwise, <tt>binary_op(*i,*j)</tt> is
 *  not evaluated and no assignment occurs. The input and output sequences may coincide,
 *  resulting in an in-place transformation.
 *    
 *  \param first1 The beginning of the first input sequence.
 *  \param last1 The end of the first input sequence.
 *  \param first2 The beginning of the second input sequence.
 *  \param stencil The beginning of the stencil sequence.
 *  \param result The beginning of the output sequence.
 *  \param binary_op The transformation operation.
 *  \param pred The predicate operation.
 *  \return The end of the output sequence.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator1's \c value_type is convertible to \c BinaryFunction's \c first_argument_type.
 *  \tparam InputIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>
 *                         and \c InputIterator2's \c value_type is convertible to \c BinaryFunction's \c second_argument_type.
 *  \tparam ForwardIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward Iterator</a>.
 *  \tparam BinaryFunction is a model of <a href="https://en.cppreference.com/w/cpp/utility/functional/binary_function">Binary Function</a>
 *                         and \c BinaryFunction's \c result_type is convertible to \c OutputIterator's \c value_type.
 *  \tparam Predicate is a model of <a href="https://en.cppreference.com/w/cpp/concepts/predicate">Predicate</a>.
 *
 *  \pre \p first1 may equal \p result, but the range <tt>[first1, last1)</tt> shall not overlap the range <tt>[result, result + (last1 - first1))</tt> otherwise.
 *  \pre \p first2 may equal \p result, but the range <tt>[first2, first2 + (last1 - first1))</tt> shall not overlap the range <tt>[result, result + (last1 - first1))</tt> otherwise.
 *  \pre \p stencil may equal \p result, but the range <tt>[stencil, stencil + (last1 - first1))</tt> shall not overlap the range <tt>[result, result + (last1 - first1))</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p transform_if:
 *
 *  \code
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  
 *  int input1[6]  = {-5,  0,  2,  3,  2,  4};
 *  int input2[6]  = { 3,  6, -2,  1,  2,  3};
 *  int stencil[8] = { 1,  0,  1,  0,  1,  0};
 *  int output[6];
 * 
 *  thrust::plus<int> op;
 *  thrust::identity<int> identity;
 *
 *  thrust::transform_if(input1, input1 + 6, input2, stencil, output, op, identity);
 *
 *  // output is now {-2,  0,  0,  3,  4,  4};
 *  \endcode
 *
 *  \see thrust::transform
 */
template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
  ForwardIterator transform_if(InputIterator1 first1, InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred);


/*! \} // end transformations
 */

THRUST_NAMESPACE_END

#include <thrust/detail/transform.inl>
