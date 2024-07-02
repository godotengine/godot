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

/*! \file temporary_array.h
 *  \brief Container-like class temporary storage inside algorithms.
 */

#pragma once

#include <thrust/detail/config.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{

// Forward declare temporary_array, as it's used by the CUDA copy backend, which
// is included in contiguous_storage's definition.
template<typename T, typename System>
  class temporary_array;

} // end detail
THRUST_NAMESPACE_END

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/tagged_iterator.h>
#include <thrust/detail/contiguous_storage.h>
#include <thrust/detail/allocator/temporary_allocator.h>
#include <thrust/detail/allocator/no_throw_allocator.h>
#include <thrust/detail/memory_wrapper.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{


template<typename T, typename System>
  class temporary_array
    : public contiguous_storage<
               T,
               no_throw_allocator<
                 temporary_allocator<T,System>
               >
             >
{
  private:
    typedef contiguous_storage<
      T,
      no_throw_allocator<
        temporary_allocator<T,System>
      >
    > super_t;

    // to help out the constructor
    typedef no_throw_allocator<temporary_allocator<T,System> > alloc_type;

  public:
    typedef typename super_t::size_type size_type;

    __host__ __device__
    temporary_array(thrust::execution_policy<System> &system);

    __host__ __device__
    temporary_array(thrust::execution_policy<System> &system, size_type n);

    // provide a kill-switch to explicitly avoid initialization
    __host__ __device__
    temporary_array(int uninit, thrust::execution_policy<System> &system, size_type n);

    template<typename InputIterator>
    __host__ __device__
    temporary_array(thrust::execution_policy<System> &system,
                    InputIterator first,
                    size_type n);

    template<typename InputIterator, typename InputSystem>
    __host__ __device__
    temporary_array(thrust::execution_policy<System> &system,
                    thrust::execution_policy<InputSystem> &input_system,
                    InputIterator first,
                    size_type n);

    template<typename InputIterator>
    __host__ __device__
    temporary_array(thrust::execution_policy<System> &system,
                    InputIterator first,
                    InputIterator last);

    template<typename InputSystem, typename InputIterator>
    __host__ __device__
    temporary_array(thrust::execution_policy<System> &system,
                    thrust::execution_policy<InputSystem> &input_system,
                    InputIterator first,
                    InputIterator last);

    __host__ __device__
    ~temporary_array();
}; // end temporary_array


// XXX eliminate this when we do ranges for real
template<typename Iterator, typename System>
  class tagged_iterator_range
{
  public:
    typedef thrust::detail::tagged_iterator<Iterator,System> iterator;

    template<typename Ignored1, typename Ignored2>
    tagged_iterator_range(const Ignored1 &, const Ignored2 &, Iterator first, Iterator last)
      : m_begin(first),
        m_end(last)
    {}

    iterator begin(void) const { return m_begin; }
    iterator end(void) const { return m_end; }

  private:
    iterator m_begin, m_end;
};


// if FromSystem is convertible to ToSystem, then just make a shallow
// copy of the range. else, use a temporary_array
// note that the resulting iterator is explicitly tagged with ToSystem either way
template<typename Iterator, typename FromSystem, typename ToSystem>
  struct move_to_system_base
    : public eval_if<
        is_convertible<
          FromSystem,
          ToSystem
        >::value,
        identity_<
          tagged_iterator_range<Iterator,ToSystem>
        >,
        identity_<
          temporary_array<
            typename thrust::iterator_value<Iterator>::type,
            ToSystem
          >
        >
      >
{};


template<typename Iterator, typename FromSystem, typename ToSystem>
  class move_to_system
    : public move_to_system_base<
        Iterator,
        FromSystem,
        ToSystem
      >::type
{
  typedef typename move_to_system_base<Iterator,FromSystem,ToSystem>::type super_t;

  public:
    move_to_system(thrust::execution_policy<FromSystem> &from_system,
                   thrust::execution_policy<ToSystem> &to_system,
                   Iterator first,
                   Iterator last)
      : super_t(to_system, from_system, first, last) {}
};


} // end detail
THRUST_NAMESPACE_END

#include <thrust/detail/temporary_array.inl>

