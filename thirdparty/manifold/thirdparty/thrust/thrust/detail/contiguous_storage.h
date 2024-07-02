/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

#pragma once

#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/config.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

struct copy_allocator_t {};

// XXX parameter T is redundant with parameter Alloc
template<typename T, typename Alloc>
  class contiguous_storage
{
  private:
    typedef thrust::detail::allocator_traits<Alloc> alloc_traits;

  public:
    typedef Alloc                                      allocator_type;
    typedef T                                          value_type;
    typedef typename alloc_traits::pointer             pointer;
    typedef typename alloc_traits::const_pointer       const_pointer;
    typedef typename alloc_traits::size_type           size_type;
    typedef typename alloc_traits::difference_type     difference_type;
    typedef typename alloc_traits::reference           reference;
    typedef typename alloc_traits::const_reference     const_reference;

    typedef thrust::detail::normal_iterator<pointer>       iterator;
    typedef thrust::detail::normal_iterator<const_pointer> const_iterator;

    __thrust_exec_check_disable__
    __host__ __device__
    explicit contiguous_storage(const allocator_type &alloc = allocator_type());

    __thrust_exec_check_disable__
    __host__ __device__
    explicit contiguous_storage(size_type n, const allocator_type &alloc = allocator_type());

    __thrust_exec_check_disable__
    __host__ __device__
    explicit contiguous_storage(copy_allocator_t, const contiguous_storage &other);

    __thrust_exec_check_disable__
    __host__ __device__
    explicit contiguous_storage(copy_allocator_t, const contiguous_storage &other, size_type n);

    __thrust_exec_check_disable__
    __host__ __device__
    ~contiguous_storage();

    __host__ __device__
    size_type size() const;

    __host__ __device__
    size_type max_size() const;

    __host__ __device__
    pointer data();

    __host__ __device__
    const_pointer data() const;

    __host__ __device__
    iterator begin();

    __host__ __device__
    const_iterator begin() const;

    __host__ __device__
    iterator end();

    __host__ __device__
    const_iterator end() const;

    __host__ __device__
    reference operator[](size_type n);

    __host__ __device__
    const_reference operator[](size_type n) const;

    __host__ __device__
    allocator_type get_allocator() const;

    // note that allocate does *not* automatically call deallocate
    __host__ __device__
    void allocate(size_type n);

    __host__ __device__
    void deallocate();

    __host__ __device__
    void swap(contiguous_storage &x);

    __host__ __device__
    void default_construct_n(iterator first, size_type n);

    __host__ __device__
    void uninitialized_fill_n(iterator first, size_type n, const value_type &value);

    template<typename InputIterator>
    __host__ __device__
    iterator uninitialized_copy(InputIterator first, InputIterator last, iterator result);

    template<typename System, typename InputIterator>
    __host__ __device__
    iterator uninitialized_copy(thrust::execution_policy<System> &from_system,
                                InputIterator first,
                                InputIterator last,
                                iterator result);

    template<typename InputIterator, typename Size>
    __host__ __device__
    iterator uninitialized_copy_n(InputIterator first, Size n, iterator result);

    template<typename System, typename InputIterator, typename Size>
    __host__ __device__
    iterator uninitialized_copy_n(thrust::execution_policy<System> &from_system,
                                  InputIterator first,
                                  Size n,
                                  iterator result);

    __host__ __device__
    void destroy(iterator first, iterator last);

    __host__ __device__
    void deallocate_on_allocator_mismatch(const contiguous_storage &other);

    __host__ __device__
    void destroy_on_allocator_mismatch(const contiguous_storage &other,
        iterator first, iterator last);

    __host__ __device__
    void set_allocator(const allocator_type &alloc);

    __host__ __device__
    bool is_allocator_not_equal(const allocator_type &alloc) const;

    __host__ __device__
    bool is_allocator_not_equal(const contiguous_storage &other) const;

    __host__ __device__
    void propagate_allocator(const contiguous_storage &other);

#if THRUST_CPP_DIALECT >= 2011
    __host__ __device__
    void propagate_allocator(contiguous_storage &other);

    // allow move assignment for a sane implementation of allocator propagation
    // on move assignment
    __host__ __device__
    contiguous_storage &operator=(contiguous_storage &&other);
#endif

  private:
    // XXX we could inherit from this to take advantage of empty base class optimization
    allocator_type m_allocator;

    iterator m_begin;

    size_type m_size;

    // disallow assignment
    contiguous_storage &operator=(const contiguous_storage &x);

    __host__ __device__
    void swap_allocators(true_type, const allocator_type &);

    __host__ __device__
    void swap_allocators(false_type, allocator_type &);

    __host__ __device__
    bool is_allocator_not_equal_dispatch(true_type, const allocator_type &) const;

    __host__ __device__
    bool is_allocator_not_equal_dispatch(false_type, const allocator_type &) const;

    __host__ __device__
    void deallocate_on_allocator_mismatch_dispatch(true_type, const contiguous_storage &other);

    __host__ __device__
    void deallocate_on_allocator_mismatch_dispatch(false_type, const contiguous_storage &other);

    __host__ __device__
    void destroy_on_allocator_mismatch_dispatch(true_type, const contiguous_storage &other,
        iterator first, iterator last);

    __host__ __device__
    void destroy_on_allocator_mismatch_dispatch(false_type, const contiguous_storage &other,
        iterator first, iterator last);

    __host__ __device__
    void propagate_allocator_dispatch(true_type, const contiguous_storage &other);

    __host__ __device__
    void propagate_allocator_dispatch(false_type, const contiguous_storage &other);

#if THRUST_CPP_DIALECT >= 2011
    __host__ __device__
    void propagate_allocator_dispatch(true_type, contiguous_storage &other);

    __host__ __device__
    void propagate_allocator_dispatch(false_type, contiguous_storage &other);
#endif
}; // end contiguous_storage

} // end detail

template<typename T, typename Alloc>
__host__ __device__
void swap(detail::contiguous_storage<T,Alloc> &lhs, detail::contiguous_storage<T,Alloc> &rhs);

THRUST_NAMESPACE_END

#include <thrust/detail/contiguous_storage.inl>

