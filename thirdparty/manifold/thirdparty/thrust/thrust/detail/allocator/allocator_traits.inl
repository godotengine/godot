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

#include <thrust/detail/config.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/type_traits/is_call_possible.h>
#include <thrust/detail/integer_traits.h>

#if THRUST_CPP_DIALECT >= 2011
  #include <thrust/detail/type_deduction.h>
#endif

#include <thrust/detail/memory_wrapper.h>
#include <new>

THRUST_NAMESPACE_BEGIN
namespace detail
{

#if THRUST_CPP_DIALECT >= 2011

// std::allocator's member functions are deprecated in C++17 and removed in
// C++20, so we can't just use the generic implementation for allocator_traits
// that calls the allocator's member functions.
// Instead, specialize allocator_traits for std::allocator and defer to
// std::allocator_traits<std::allocator> and let the STL do whatever it needs
// to for the current c++ version. Manually forward the calls to suppress
// host/device warnings.
template <typename T>
struct allocator_traits<std::allocator<T>>
  : public std::allocator_traits<std::allocator<T>>
{
private:
  using superclass = std::allocator_traits<std::allocator<T>>;

public:
  using allocator_type = typename superclass::allocator_type;
  using value_type = typename superclass::value_type;
  using pointer = typename superclass::pointer;
  using const_pointer = typename superclass::const_pointer;
  using void_pointer = typename superclass::void_pointer;
  using const_void_pointer = typename superclass::const_void_pointer;
  using difference_type = typename superclass::difference_type;
  using size_type = typename superclass::size_type;
  using propagate_on_container_swap = typename superclass::propagate_on_container_swap;
  using propagate_on_container_copy_assignment =
    typename superclass::propagate_on_container_copy_assignment;
  using propagate_on_container_move_assignment =
    typename superclass::propagate_on_container_move_assignment;

  // std::allocator_traits added this in C++17, but thrust::allocator_traits defines
  // it unconditionally.
  using is_always_equal = typename eval_if<
      allocator_traits_detail::has_is_always_equal<allocator_type>::value,
      allocator_traits_detail::nested_is_always_equal<allocator_type>,
      is_empty<allocator_type>
    >::type;

  // std::allocator_traits doesn't provide these, but
  // thrust::detail::allocator_traits does. These used to be part of the
  // std::allocator API but were deprecated in C++17.
  using reference = typename thrust::detail::pointer_traits<pointer>::reference;
  using const_reference = typename thrust::detail::pointer_traits<const_pointer>::reference;

  template <typename U>
  using rebind_alloc = std::allocator<U>;
  template <typename U>
  using rebind_traits = allocator_traits<std::allocator<U>>;

  __thrust_exec_check_disable__
  __host__ __device__
  static pointer allocate(allocator_type &a, size_type n)
  {
    return superclass::allocate(a, n);
  }

  __thrust_exec_check_disable__
  __host__ __device__
  static pointer allocate(allocator_type &a, size_type n, const_void_pointer hint)
  {
    return superclass::allocate(a, n, hint);
  }

  __thrust_exec_check_disable__
  __host__ __device__
  static void deallocate(allocator_type &a, pointer p, size_type n)
  {
    superclass::deallocate(a, p, n);
  }

  __thrust_exec_check_disable__
  template <typename U, typename ...Args>
  __host__ __device__
  static void construct(allocator_type &a, U *p, Args&&... args)
  {
    superclass::construct(a, p, THRUST_FWD(args)...);
  }

  __thrust_exec_check_disable__
  template <typename U>
  __host__ __device__
  static void destroy(allocator_type &a, U *p)
  {
    superclass::destroy(a, p);
  }

  __thrust_exec_check_disable__
  __host__ __device__
  static size_type max_size(const allocator_type &a)
  {
    return superclass::max_size(a);
  }
};

#endif //  C++11

namespace allocator_traits_detail
{

__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_allocate_with_hint_impl, allocate)

template<typename Alloc>
  class has_member_allocate_with_hint
{
  typedef typename allocator_traits<Alloc>::pointer            pointer;
  typedef typename allocator_traits<Alloc>::size_type          size_type;
  typedef typename allocator_traits<Alloc>::const_void_pointer const_void_pointer;

  public:
    typedef typename has_member_allocate_with_hint_impl<Alloc, pointer(size_type,const_void_pointer)>::type type;
    static const bool value = type::value;
};

template<typename Alloc>
__host__ __device__
  typename enable_if<
    has_member_allocate_with_hint<Alloc>::value,
    typename allocator_traits<Alloc>::pointer
  >::type
    allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n, typename allocator_traits<Alloc>::const_void_pointer hint)
{
  return a.allocate(n,hint);
}

template<typename Alloc>
__host__ __device__
  typename disable_if<
    has_member_allocate_with_hint<Alloc>::value,
    typename allocator_traits<Alloc>::pointer
  >::type
    allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n, typename allocator_traits<Alloc>::const_void_pointer)
{
  return a.allocate(n);
}


__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_construct1_impl, construct)

template<typename Alloc, typename T>
  struct has_member_construct1
    : has_member_construct1_impl<Alloc, void(T*)>
{};

__thrust_exec_check_disable__
template<typename Alloc, typename T>
  inline __host__ __device__
    typename enable_if<
      has_member_construct1<Alloc,T>::value
    >::type
      construct(Alloc &a, T *p)
{
  a.construct(p);
}

__thrust_exec_check_disable__
template<typename Alloc, typename T>
  inline __host__ __device__
    typename disable_if<
      has_member_construct1<Alloc,T>::value
    >::type
      construct(Alloc &, T *p)
{
  ::new(static_cast<void*>(p)) T();
}


__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_construct2_impl, construct)

template<typename Alloc, typename T, typename Arg1>
  struct has_member_construct2
    : has_member_construct2_impl<Alloc, void(T*,const Arg1 &)>
{};

__thrust_exec_check_disable__
template<typename Alloc, typename T, typename Arg1>
  inline __host__ __device__
    typename enable_if<
      has_member_construct2<Alloc,T,Arg1>::value
    >::type
      construct(Alloc &a, T *p, const Arg1 &arg1)
{
  a.construct(p,arg1);
}

__thrust_exec_check_disable__
template<typename Alloc, typename T, typename Arg1>
  inline __host__ __device__
    typename disable_if<
      has_member_construct2<Alloc,T,Arg1>::value
    >::type
      construct(Alloc &, T *p, const Arg1 &arg1)
{
  ::new(static_cast<void*>(p)) T(arg1);
}

#if THRUST_CPP_DIALECT >= 2011

__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_constructN_impl, construct)

template<typename Alloc, typename T, typename... Args>
  struct has_member_constructN
    : has_member_constructN_impl<Alloc, void(T*, Args...)>
{};

__thrust_exec_check_disable__
template<typename Alloc, typename T, typename... Args>
  inline __host__ __device__
    typename enable_if<
      has_member_constructN<Alloc, T, Args...>::value
    >::type
      construct(Alloc &a, T* p, Args&&... args)
{
  a.construct(p, THRUST_FWD(args)...);
}

__thrust_exec_check_disable__
template<typename Alloc, typename T, typename... Args>
  inline __host__ __device__
    typename disable_if<
      has_member_constructN<Alloc, T, Args...>::value
    >::type
      construct(Alloc &, T* p, Args&&... args)
{
  ::new(static_cast<void*>(p)) T(THRUST_FWD(args)...);
}

#endif

__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_destroy_impl, destroy)

template<typename Alloc, typename T>
  struct has_member_destroy
    : has_member_destroy_impl<Alloc, void(T*)>
{};

__thrust_exec_check_disable__
template<typename Alloc, typename T>
  inline __host__ __device__
    typename enable_if<
      has_member_destroy<Alloc,T>::value
    >::type
      destroy(Alloc &a, T *p)
{
  a.destroy(p);
}

__thrust_exec_check_disable__
template<typename Alloc, typename T>
  inline __host__ __device__
    typename disable_if<
      has_member_destroy<Alloc,T>::value
    >::type
      destroy(Alloc &, T *p)
{
  p->~T();
}


__THRUST_DEFINE_IS_CALL_POSSIBLE(has_member_max_size_impl, max_size)

template<typename Alloc>
  class has_member_max_size
{
  typedef typename allocator_traits<Alloc>::size_type size_type;

  public:
    typedef typename has_member_max_size_impl<Alloc, size_type(void)>::type type;
    static const bool value = type::value;
};

template<typename Alloc>
__host__ __device__
  typename enable_if<
    has_member_max_size<Alloc>::value,
    typename allocator_traits<Alloc>::size_type
  >::type
    max_size(const Alloc &a)
{
  return a.max_size();
}

template<typename Alloc>
__host__ __device__
  typename disable_if<
    has_member_max_size<Alloc>::value,
    typename allocator_traits<Alloc>::size_type
  >::type
    max_size(const Alloc &)
{
  typedef typename allocator_traits<Alloc>::size_type size_type;
  return thrust::detail::integer_traits<size_type>::const_max;
}

template<typename Alloc>
__host__ __device__
  typename enable_if<
    has_member_system<Alloc>::value,
    typename allocator_system<Alloc>::type &
  >::type
    system(Alloc &a)
{
  // return the allocator's system
  return a.system();
}

template<typename Alloc>
__host__ __device__
  typename disable_if<
    has_member_system<Alloc>::value,
    typename allocator_system<Alloc>::type
  >::type
    system(Alloc &)
{
  // return a copy of a value-initialized system
  return typename allocator_system<Alloc>::type();
}


} // end allocator_traits_detail


template<typename Alloc>
__host__ __device__
  typename allocator_traits<Alloc>::pointer
    allocator_traits<Alloc>
      ::allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n)
{
  struct workaround_warnings
  {
    __thrust_exec_check_disable__
    static __host__ __device__
    typename allocator_traits<Alloc>::pointer
      allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n)
    {
      return a.allocate(n);
    }
  };

  return workaround_warnings::allocate(a, n);
}

template<typename Alloc>
__host__ __device__
  typename allocator_traits<Alloc>::pointer
    allocator_traits<Alloc>
      ::allocate(Alloc &a, typename allocator_traits<Alloc>::size_type n, typename allocator_traits<Alloc>::const_void_pointer hint)
{
  return allocator_traits_detail::allocate(a, n, hint);
}

template<typename Alloc>
__host__ __device__
  void allocator_traits<Alloc>
    ::deallocate(Alloc &a, typename allocator_traits<Alloc>::pointer p, typename allocator_traits<Alloc>::size_type n)
{
  struct workaround_warnings
  {
    __thrust_exec_check_disable__
    static __host__ __device__
    void deallocate(Alloc &a, typename allocator_traits<Alloc>::pointer p, typename allocator_traits<Alloc>::size_type n)
    {
      return a.deallocate(p,n);
    }
  };

  return workaround_warnings::deallocate(a,p,n);
}

template<typename Alloc>
  template<typename T>
  __host__ __device__
    void allocator_traits<Alloc>
      ::construct(allocator_type &a, T *p)
{
  return allocator_traits_detail::construct(a,p);
}

template<typename Alloc>
  template<typename T, typename Arg1>
  __host__ __device__
    void allocator_traits<Alloc>
      ::construct(allocator_type &a, T *p, const Arg1 &arg1)
{
  return allocator_traits_detail::construct(a,p,arg1);
}

#if THRUST_CPP_DIALECT >= 2011

template<typename Alloc>
  template<typename T, typename... Args>
  __host__ __device__
    void allocator_traits<Alloc>
      ::construct(allocator_type &a, T *p, Args&&... args)
{
  return allocator_traits_detail::construct(a, p, THRUST_FWD(args)...);
}

#endif

template<typename Alloc>
  template<typename T>
  __host__ __device__
    void allocator_traits<Alloc>
      ::destroy(allocator_type &a, T *p)
{
  return allocator_traits_detail::destroy(a,p);
}

template<typename Alloc>
__host__ __device__
  typename allocator_traits<Alloc>::size_type
    allocator_traits<Alloc>
      ::max_size(const allocator_type &a)
{
  return allocator_traits_detail::max_size(a);
}

template<typename Alloc>
__host__ __device__
  typename allocator_system<Alloc>::get_result_type
    allocator_system<Alloc>
      ::get(Alloc &a)
{
  return allocator_traits_detail::system(a);
}


} // end detail
THRUST_NAMESPACE_END

