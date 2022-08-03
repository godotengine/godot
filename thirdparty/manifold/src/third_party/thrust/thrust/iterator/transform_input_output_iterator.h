/*
 *  Copyright 2020 NVIDIA Corporation
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

/*! \file thrust/iterator/transform_input_output_iterator.h
 *  \brief An iterator which adapts another iterator by applying transform
 *         functions when reading and writing dereferenced values.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/transform_input_output_iterator.inl>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup iterators
 *  \{
 */

/*! \addtogroup fancyiterator Fancy Iterators
 *  \ingroup iterators
 *  \{
 */

/*! \p transform_input_output_iterator is a special kind of iterator which applies
 * transform functions when reading from or writing to dereferenced values.
 * This iterator is useful for algorithms that operate on a type that needs to
 * be serialized/deserialized from values in another iterator, avoiding the
 * need to materialize intermediate results in memory. This also enables the
 * transform functions to be fused with the operations that read and write to
 * the `transform_input_output_iterator`.
 *
 * The following code snippet demonstrates how to create a
 * \p transform_input_output_iterator which performs different transformations when
 * reading from and writing to the iterator.
 *
 * \code
 * #include <thrust/iterator/transform_input_output_iterator.h>
 * #include <thrust/device_vector.h>
 *
 *  int main()
 *  {
 *    const size_t size = 4;
 *    thrust::device_vector<float> v(size);
 *
 *    // Write 1.0f, 2.0f, 3.0f, 4.0f to vector
 *    thrust::sequence(v.begin(), v.end(), 1);
 *
 *    // Iterator that returns negated values and writes squared values
 *    auto iter = thrust::make_transform_input_output_iterator(v.begin(),
 *        thrust::negate<float>{}, thrust::square<float>{});
 * 
 *    // Iterator negates values when reading
 *    std::cout << iter[0] << " ";  // -1.0f;
 *    std::cout << iter[1] << " ";  // -2.0f;
 *    std::cout << iter[2] << " ";  // -3.0f;
 *    std::cout << iter[3] << "\n"; // -4.0f;
 *
 *    // Write 1.0f, 2.0f, 3.0f, 4.0f to iterator
 *    thrust::sequence(iter, iter + size, 1);
 *
 *    // Values were squared before writing to vector
 *    std::cout << v[0] << " ";  // 1.0f;
 *    std::cout << v[1] << " ";  // 4.0f;
 *    std::cout << v[2] << " ";  // 9.0f;
 *    std::cout << v[3] << "\n"; // 16.0f;
 *
 *  }
 * \endcode
 *
 * \see make_transform_input_output_iterator
 */

template <typename InputFunction, typename OutputFunction, typename Iterator>
  class transform_input_output_iterator
    : public detail::transform_input_output_iterator_base<InputFunction, OutputFunction, Iterator>::type
{

  /*! \cond
   */

  public:

    typedef typename
    detail::transform_input_output_iterator_base<InputFunction, OutputFunction, Iterator>::type
    super_t;

    friend class thrust::iterator_core_access;
  /*! \endcond
   */

  /*! This constructor takes as argument a \c Iterator an \c InputFunction and an
   * \c OutputFunction and copies them to a new \p transform_input_output_iterator
   *
   * \param io An \c Iterator pointing to where the input to \c InputFunction
   *           will be read from and the result of \c OutputFunction will be written to
   * \param input_function An \c InputFunction to be executed on values read from the iterator
   * \param output_function An \c OutputFunction to be executed on values written to the iterator
   */
    __host__ __device__
    transform_input_output_iterator(Iterator const& io, InputFunction input_function, OutputFunction output_function)
      : super_t(io), input_function(input_function), output_function(output_function)
    {
    }

    /*! \cond
     */
  private:

    __host__ __device__
    typename super_t::reference dereference() const
    {
      return detail::transform_input_output_iterator_proxy<
        InputFunction, OutputFunction, Iterator
      >(this->base_reference(), input_function, output_function);
    }

    InputFunction input_function;
    OutputFunction output_function;

    /*! \endcond
     */
}; // end transform_input_output_iterator

/*! \p make_transform_input_output_iterator creates a \p transform_input_output_iterator from
 *  an \c Iterator a \c InputFunction and a \c OutputFunction
 *
 * \param io An \c Iterator pointing to where the input to \c InputFunction
 *           will be read from and the result of \c OutputFunction will be written to
 * \param input_function An \c InputFunction to be executed on values read from the iterator
 * \param output_function An \c OutputFunction to be executed on values written to the iterator
 *  \see transform_input_output_iterator
 */
template <typename InputFunction, typename OutputFunction, typename Iterator>
transform_input_output_iterator<InputFunction, OutputFunction, Iterator>
__host__ __device__
make_transform_input_output_iterator(Iterator io, InputFunction input_function, OutputFunction output_function)
{
    return transform_input_output_iterator<InputFunction, OutputFunction, Iterator>(io, input_function, output_function);
} // end make_transform_input_output_iterator

/*! \} // end fancyiterators
 */

/*! \} // end iterators
 */

THRUST_NAMESPACE_END

