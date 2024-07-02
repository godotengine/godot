// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/type_deduction.h>
#include <thrust/detail/memory_algorithms.h>
#include <thrust/detail/allocator/allocator_traits.h>

#include <utility>
#include <thrust/detail/memory_wrapper.h>

THRUST_NAMESPACE_BEGIN

// wg21.link/p0316r0

///////////////////////////////////////////////////////////////////////////////

namespace detail
{

template <typename Allocator, typename Pointer>
void allocator_delete_impl(
  Allocator const& alloc, Pointer p, std::false_type
)
{
  using traits = typename detail::allocator_traits<
    typename std::remove_cv<
      typename std::remove_reference<Allocator>::type
    >::type
  >;

  typename traits::allocator_type alloc_T(alloc);

  if (nullptr != pointer_traits<Pointer>::get(p))
  {
    traits::destroy(alloc_T, thrust::raw_pointer_cast(p));
    traits::deallocate(alloc_T, p, 1);
  }
}

template <typename Allocator, typename Pointer>
void allocator_delete_impl(
  Allocator const& alloc, Pointer p, std::true_type
)
{
  using traits = typename detail::allocator_traits<
    typename std::remove_cv<
      typename std::remove_reference<Allocator>::type
    >::type
  >;

  typename traits::allocator_type alloc_T(alloc);

  if (nullptr != pointer_traits<Pointer>::get(p))
  {
    traits::deallocate(alloc_T, p, 1);
  }
}

} // namespace detail

template <typename T, typename Allocator, bool Uninitialized = false>
struct allocator_delete final
{
  using allocator_type
    = typename std::remove_cv<
        typename std::remove_reference<Allocator>::type
      >::type::template rebind<T>::other;
  using pointer = typename detail::allocator_traits<allocator_type>::pointer;

  template <typename UAllocator>
  allocator_delete(UAllocator&& other) noexcept
    : alloc_(THRUST_FWD(other))
  {}

  template <typename U, typename UAllocator>
  allocator_delete(
      allocator_delete<U, UAllocator> const& other
    ) noexcept
    : alloc_(other.get_allocator())
  {}
  template <typename U, typename UAllocator>
  allocator_delete(
      allocator_delete<U, UAllocator>&& other
    ) noexcept
    : alloc_(std::move(other.get_allocator()))
  {}

  template <typename U, typename UAllocator>
  allocator_delete& operator=(
    allocator_delete<U, UAllocator> const& other
  ) noexcept
  {
    alloc_ = other.get_allocator();
    return *this;
  }
  template <typename U, typename UAllocator>
  allocator_delete& operator=(
    allocator_delete<U, UAllocator>&& other
  ) noexcept
  {
    alloc_ = std::move(other.get_allocator());
    return *this;
  }

  void operator()(pointer p)
  {
    std::integral_constant<bool, Uninitialized> ic;

    detail::allocator_delete_impl(get_allocator(), p, ic);
  }

  allocator_type& get_allocator() noexcept { return alloc_; }
  allocator_type const& get_allocator() const noexcept { return alloc_; }

  void swap(allocator_delete& other) noexcept
  {
    using std::swap;
    swap(alloc_, other.alloc_);
  }

private:
  allocator_type alloc_;
};

template <typename T, typename Allocator>
using uninitialized_allocator_delete = allocator_delete<T, Allocator, true>;

namespace detail {

template <typename Allocator, typename Pointer, typename Size>
void array_allocator_delete_impl(
  Allocator const& alloc, Pointer p, Size count, std::false_type
)
{
  using traits = typename detail::allocator_traits<
    typename std::remove_cv<
      typename std::remove_reference<Allocator>::type
    >::type
  >;

  typename traits::allocator_type alloc_T(alloc);

  if (nullptr != pointer_traits<Pointer>::get(p))
  {
    destroy_n(alloc_T, p, count);
    traits::deallocate(alloc_T, p, count);
  }
}

template <typename Allocator, typename Pointer, typename Size>
void array_allocator_delete_impl(
  Allocator const& alloc, Pointer p, Size count, std::true_type
)
{
  using traits = typename detail::allocator_traits<
    typename std::remove_cv<
      typename std::remove_reference<Allocator>::type
    >::type
  >;

  typename traits::allocator_type alloc_T(alloc);

  if (nullptr != pointer_traits<Pointer>::get(p))
  {
    traits::deallocate(alloc_T, p, count);
  }
}

} // namespace detail

template <typename T, typename Allocator, bool Uninitialized = false>
struct array_allocator_delete final
{
  using allocator_type
    = typename std::remove_cv<
        typename std::remove_reference<Allocator>::type
      >::type::template rebind<T>::other;
  using pointer = typename detail::allocator_traits<allocator_type>::pointer;

  template <typename UAllocator>
  array_allocator_delete(UAllocator&& other, std::size_t n) noexcept
    : alloc_(THRUST_FWD(other)), count_(n)
  {}

  template <typename U, typename UAllocator>
  array_allocator_delete(
      array_allocator_delete<U, UAllocator> const& other
    ) noexcept
    : alloc_(other.get_allocator()), count_(other.count_)
  {}
  template <typename U, typename UAllocator>
  array_allocator_delete(
      array_allocator_delete<U, UAllocator>&& other
    ) noexcept
    : alloc_(std::move(other.get_allocator())), count_(other.count_)
  {}

  template <typename U, typename UAllocator>
  array_allocator_delete& operator=(
    array_allocator_delete<U, UAllocator> const& other
  ) noexcept
  {
    alloc_ = other.get_allocator();
    count_ = other.count_;
    return *this;
  }
  template <typename U, typename UAllocator>
  array_allocator_delete& operator=(
    array_allocator_delete<U, UAllocator>&& other
  ) noexcept
  {
    alloc_ = std::move(other.get_allocator());
    count_ = other.count_;
    return *this;
  }

  void operator()(pointer p)
  {
    std::integral_constant<bool, Uninitialized> ic;

    detail::array_allocator_delete_impl(get_allocator(), p, count_, ic);
  }

  allocator_type& get_allocator() noexcept { return alloc_; }
  allocator_type const& get_allocator() const noexcept { return alloc_; }

  void swap(array_allocator_delete& other) noexcept
  {
    using std::swap;
    swap(alloc_, other.alloc_);
    swap(count_, other.count_);
  }

private:
  allocator_type alloc_;
  std::size_t    count_;
};
  
template <typename T, typename Allocator>
using uninitialized_array_allocator_delete
  = array_allocator_delete<T, Allocator, true>;

///////////////////////////////////////////////////////////////////////////////

template <typename Pointer, typename Lambda>
struct tagged_deleter : Lambda
{
  __host__ __device__
  tagged_deleter(Lambda&& l) : Lambda(THRUST_FWD(l)) {}

  using pointer = Pointer;
};

template <typename Pointer, typename Lambda>
__host__ __device__
tagged_deleter<Pointer, Lambda>
make_tagged_deleter(Lambda&& l)
{
  return tagged_deleter<Pointer, Lambda>(THRUST_FWD(l));
}

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename Allocator, typename... Args>
__host__
std::unique_ptr<
  T,
  allocator_delete<
    T
  , typename detail::allocator_traits<
      typename std::remove_cv<
        typename std::remove_reference<Allocator>::type
      >::type
    >::template rebind_traits<T>::allocator_type
  >
>
allocate_unique(
  Allocator const& alloc, Args&&... args
)
{
  using traits = typename detail::allocator_traits<
    typename std::remove_cv<
      typename std::remove_reference<Allocator>::type
    >::type
  >::template rebind_traits<T>;

  typename traits::allocator_type alloc_T(alloc);

  auto hold_deleter = make_tagged_deleter<typename traits::pointer>(
    [&alloc_T] (typename traits::pointer p) {
      traits::deallocate(alloc_T, p, 1);
    }
  );
  using hold_t = std::unique_ptr<T, decltype(hold_deleter)>;
  auto hold = hold_t(traits::allocate(alloc_T, 1), hold_deleter);

  traits::construct(
    alloc_T, thrust::raw_pointer_cast(hold.get()), THRUST_FWD(args)...
  );
  auto deleter = allocator_delete<T, typename traits::allocator_type>(alloc);
  return std::unique_ptr<T, decltype(deleter)>
    (hold.release(), std::move(deleter));
}

template <typename T, typename Allocator>
__host__
std::unique_ptr<
  T,
  uninitialized_allocator_delete<
    T
  , typename detail::allocator_traits<
      typename std::remove_cv<
        typename std::remove_reference<Allocator>::type
      >::type
    >::template rebind_traits<T>::allocator_type
  >
>
uninitialized_allocate_unique(
  Allocator const& alloc
)
{
  using traits = typename detail::allocator_traits<
    typename std::remove_cv<
      typename std::remove_reference<Allocator>::type
    >::type
  >::template rebind_traits<T>;

  typename traits::allocator_type alloc_T(alloc);

  auto hold_deleter = make_tagged_deleter<typename traits::pointer>(
    [&alloc_T] (typename traits::pointer p) {
      traits::deallocate(alloc_T, p, 1);
    }
  );
  using hold_t = std::unique_ptr<T, decltype(hold_deleter)>;
  auto hold = hold_t(traits::allocate(alloc_T, 1), hold_deleter);

  auto deleter = uninitialized_allocator_delete<
    T, typename traits::allocator_type
  >(alloc_T);
  return std::unique_ptr<T, decltype(deleter)>
    (hold.release(), std::move(deleter));
}

template <typename T, typename Allocator, typename Size, typename... Args>
__host__
std::unique_ptr<
  T[],
  array_allocator_delete<
    T
  , typename detail::allocator_traits<
      typename std::remove_cv<
        typename std::remove_reference<Allocator>::type
      >::type
    >::template rebind_traits<T>::allocator_type
  >
>
allocate_unique_n(
  Allocator const& alloc, Size n, Args&&... args
)
{
  using traits = typename detail::allocator_traits<
    typename std::remove_cv<
      typename std::remove_reference<Allocator>::type
    >::type
  >::template rebind_traits<T>;

  typename traits::allocator_type alloc_T(alloc);

  auto hold_deleter = make_tagged_deleter<typename traits::pointer>(
    [n, &alloc_T] (typename traits::pointer p) {
      traits::deallocate(alloc_T, p, n);
    }
  );
  using hold_t = std::unique_ptr<T[], decltype(hold_deleter)>;
  auto hold = hold_t(traits::allocate(alloc_T, n), hold_deleter);

  uninitialized_construct_n_with_allocator(
    alloc_T, hold.get(), n, THRUST_FWD(args)...
  );
  auto deleter = array_allocator_delete<
    T, typename traits::allocator_type
  >(alloc_T, n);
  return std::unique_ptr<T[], decltype(deleter)>
    (hold.release(), std::move(deleter));
}

template <typename T, typename Allocator, typename Size>
__host__
std::unique_ptr<
  T[],
  uninitialized_array_allocator_delete<
    T
  , typename detail::allocator_traits<
      typename std::remove_cv<
        typename std::remove_reference<Allocator>::type
      >::type
    >::template rebind_traits<T>::allocator_type
  >
>
uninitialized_allocate_unique_n(
  Allocator const& alloc, Size n
)
{
  using traits = typename detail::allocator_traits<
    typename std::remove_cv<
      typename std::remove_reference<Allocator>::type
    >::type
  >::template rebind_traits<T>;

  typename traits::allocator_type alloc_T(alloc);

  auto hold_deleter = make_tagged_deleter<typename traits::pointer>(
    [n, &alloc_T] (typename traits::pointer p) {
      traits::deallocate(alloc_T, p, n);
    }
  );
  using hold_t = std::unique_ptr<T[], decltype(hold_deleter)>;
  auto hold = hold_t(traits::allocate(alloc_T, n), hold_deleter);

  auto deleter = uninitialized_array_allocator_delete<
    T, typename traits::allocator_type
  >(alloc_T, n);
  return std::unique_ptr<T[], decltype(deleter)>
    (hold.release(), std::move(deleter));
}

///////////////////////////////////////////////////////////////////////////////

THRUST_NAMESPACE_END

#endif // THRUST_CPP_DIALECT >= 2011

