// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

// TODO: Split into more granular headers (move unique_stream/unique_marker to
// another header, etc).

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp14_required.h>

#if THRUST_CPP_DIALECT >= 2014

#include <thrust/optional.h>
#include <thrust/detail/type_deduction.h>
#include <thrust/type_traits/integer_sequence.h>
#include <thrust/type_traits/remove_cvref.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/detail/tuple_algorithms.h>
#include <thrust/allocate_unique.h>
#include <thrust/detail/static_assert.h>
#include <thrust/detail/execute_with_dependencies.h>
#include <thrust/detail/event_error.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/system/cuda/future.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/cuda/detail/get_value.h>

#include <type_traits>
#include <thrust/detail/memory_wrapper.h>

THRUST_NAMESPACE_BEGIN

// Forward declaration.
struct new_stream_t;

namespace system { namespace cuda { namespace detail
{

///////////////////////////////////////////////////////////////////////////////

struct nonowning_t final {};

THRUST_INLINE_CONSTANT nonowning_t nonowning{};

///////////////////////////////////////////////////////////////////////////////

struct marker_deleter final
{
  __host__
  void operator()(CUevent_st* e) const
  {
    if (nullptr != e)
      thrust::cuda_cub::throw_on_error(cudaEventDestroy(e));
  }
};

///////////////////////////////////////////////////////////////////////////////

struct unique_marker final
{
  using native_handle_type = CUevent_st*;

private:
  std::unique_ptr<CUevent_st, marker_deleter> handle_;

public:
  /// \brief Create a new stream and construct a handle to it. When the handle
  ///        is destroyed, the stream is destroyed.
  __host__
  unique_marker()
    : handle_(nullptr, marker_deleter())
  {
    native_handle_type e;
    thrust::cuda_cub::throw_on_error(
      cudaEventCreateWithFlags(&e, cudaEventDisableTiming)
    );
    handle_.reset(e);
  }

  __thrust_exec_check_disable__
  unique_marker(unique_marker const&) = delete;
  __thrust_exec_check_disable__
  unique_marker(unique_marker&&) = default;
  __thrust_exec_check_disable__
  unique_marker& operator=(unique_marker const&) = delete;
  __thrust_exec_check_disable__
  unique_marker& operator=(unique_marker&&) = default;

  __thrust_exec_check_disable__
  ~unique_marker() = default;

  __host__
  auto get() const
  THRUST_DECLTYPE_RETURNS(native_handle_type(handle_.get()));
  __host__
  auto native_handle() const
  THRUST_DECLTYPE_RETURNS(native_handle_type(handle_.get()));

  __host__
  bool valid() const noexcept { return bool(handle_); }

  __host__
  bool ready() const
  {
    cudaError_t const err = cudaEventQuery(handle_.get());

    if (cudaErrorNotReady == err)
      return false;

    // Throw on any other error.
    thrust::cuda_cub::throw_on_error(err);

    return true;
  }

  __host__
  void wait() const
  {
    thrust::cuda_cub::throw_on_error(cudaEventSynchronize(handle_.get()));
  }

  __host__
  bool operator==(unique_marker const& other) const
  {
    return other.handle_ == handle_;
  }

  __host__
  bool operator!=(unique_marker const& other) const
  {
    return !(other == *this);
  }
};

///////////////////////////////////////////////////////////////////////////////

struct stream_deleter final
{
  __host__
  void operator()(CUstream_st* s) const
  {
    if (nullptr != s)
      thrust::cuda_cub::throw_on_error(cudaStreamDestroy(s));
  }
};

struct stream_conditional_deleter final
{
private:
  bool cond_;

public:
  __host__
  constexpr stream_conditional_deleter() noexcept
    : cond_(true) {}

  __host__
  explicit constexpr stream_conditional_deleter(nonowning_t) noexcept
    : cond_(false) {}

  __host__
  void operator()(CUstream_st* s) const
  {
    if (cond_ && nullptr != s)
    {
      thrust::cuda_cub::throw_on_error(cudaStreamDestroy(s));
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

struct unique_stream final
{
  using native_handle_type = CUstream_st*;

private:
  std::unique_ptr<CUstream_st, stream_conditional_deleter> handle_;

public:
  /// \brief Create a new stream and construct a handle to it. When the handle
  ///        is destroyed, the stream is destroyed.
  __host__
  unique_stream()
    : handle_(nullptr, stream_conditional_deleter())
  {
    native_handle_type s;
    thrust::cuda_cub::throw_on_error(
      cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)
    );
    handle_.reset(s);
  }

  /// \brief Construct a non-owning handle to an existing stream. When the
  ///        handle is destroyed, the stream is not destroyed.
  __host__
  explicit unique_stream(nonowning_t, native_handle_type handle)
    : handle_(handle, stream_conditional_deleter(nonowning))
  {}

  __thrust_exec_check_disable__
  unique_stream(unique_stream const&) = delete;

  // GCC 10 complains if this is defaulted. See NVIDIA/thrust#1269.
  __thrust_exec_check_disable__
  __host__ unique_stream(unique_stream &&o) noexcept
    : handle_(std::move(o.handle_))
  {}

  __thrust_exec_check_disable__
  unique_stream& operator=(unique_stream const&) = delete;
  __thrust_exec_check_disable__
  unique_stream& operator=(unique_stream&&) = default;

  __thrust_exec_check_disable__
  ~unique_stream() = default;

  __host__
  auto get() const
  THRUST_DECLTYPE_RETURNS(native_handle_type(handle_.get()));
  __host__
  auto native_handle() const
  THRUST_DECLTYPE_RETURNS(native_handle_type(handle_.get()));

  __host__
  bool valid() const noexcept { return bool(handle_); }

  __host__
  bool ready() const
  {
    cudaError_t const err = cudaStreamQuery(handle_.get());

    if (cudaErrorNotReady == err)
      return false;

    // Throw on any other error.
    thrust::cuda_cub::throw_on_error(err);

    return true;
  }

  __host__
  void wait() const
  {
    thrust::cuda_cub::throw_on_error(
      cudaStreamSynchronize(handle_.get())
    );
  }

  __host__
  void depend_on(unique_marker& e)
  {
    thrust::cuda_cub::throw_on_error(
      cudaStreamWaitEvent(handle_.get(), e.get(), 0)
    );
  }

  __host__
  void depend_on(unique_stream& s)
  {
    if (s != *this)
    {
      unique_marker e;
      s.record(e);
      depend_on(e);
    }
  }

  __host__
  void record(unique_marker& e)
  {
    thrust::cuda_cub::throw_on_error(cudaEventRecord(e.get(), handle_.get()));
  }

  __host__
  bool operator==(unique_stream const& other) const
  {
    return other.handle_ == handle_;
  }

  __host__
  bool operator!=(unique_stream const& other) const
  {
    return !(other == *this);
  }
};

///////////////////////////////////////////////////////////////////////////////

// Inheritance hierarchy of future/event shared state types.

struct async_signal;

template <typename KeepAlives>
struct async_keep_alives /* : virtual async_signal */;

template <typename T>
struct async_value /* : virtual async_signal */;

template <typename T, typename Pointer, typename KeepAlives>
struct async_addressable_value_with_keep_alives
/* : async_value<T>, async_keep_alives<KeepAlives> */;

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename Pointer>
struct weak_promise;

template <typename X, typename XPointer = pointer<X>>
struct unique_eager_future_promise_pair final
{
  unique_eager_future<X>    future;
  weak_promise<X, XPointer> promise;
};

struct acquired_stream final
{
  unique_stream stream;
  optional<std::size_t> const acquired_from;
  // `acquired_from` contains the index in the tuple of dependencies from which
  // the stream was acquired. If `acquired_from` is empty, no stream could be
  // acquired from a dependency, and then the stream was newly created.
};

// Precondition: `device` is the current CUDA device.
template <typename X, typename Y, typename Deleter>
__host__
optional<unique_stream>
try_acquire_stream(int device, std::unique_ptr<Y, Deleter>&) noexcept;

// Precondition: `device` is the current CUDA device.
inline __host__
optional<unique_stream>
try_acquire_stream(int, unique_stream& stream) noexcept;

// Precondition: `device` is the current CUDA device.
inline __host__
optional<unique_stream>
try_acquire_stream(int device, ready_event&) noexcept;

// Precondition: `device` is the current CUDA device.
template <typename X>
inline __host__
optional<unique_stream>
try_acquire_stream(int device, ready_future<X>&) noexcept;

// Precondition: `device` is the current CUDA device.
inline __host__
optional<unique_stream>
try_acquire_stream(int device, unique_eager_event& parent) noexcept;

// Precondition: `device` is the current CUDA device.
template <typename X>
__host__
optional<unique_stream>
try_acquire_stream(int device, unique_eager_future<X>& parent) noexcept;

template <typename... Dependencies>
__host__
acquired_stream acquire_stream(int device, Dependencies&... deps) noexcept;
  
template <typename... Dependencies>
__host__
unique_eager_event
make_dependent_event(
  std::tuple<Dependencies...>&& deps
);

template <
  typename X, typename XPointer
, typename ComputeContent, typename... Dependencies
>
__host__
unique_eager_future_promise_pair<X, XPointer>
make_dependent_future(ComputeContent&& cc, std::tuple<Dependencies...>&& deps);

///////////////////////////////////////////////////////////////////////////////

struct async_signal
{
protected:
  unique_stream stream_;

public:
  // Constructs an `async_signal` which uses `stream`.
  __host__
  explicit async_signal(unique_stream&& stream)
    : stream_(std::move(stream))
  {}

  __host__
  virtual ~async_signal() {}

  unique_stream&       stream()       noexcept { return stream_; }
  unique_stream const& stream() const noexcept { return stream_; }
};

template <typename... KeepAlives>
struct async_keep_alives<std::tuple<KeepAlives...>> : virtual async_signal
{
  using keep_alives_type = std::tuple<KeepAlives...>;

protected:
  keep_alives_type keep_alives_;

public:
  // Constructs an `async_keep_alives` which uses `stream`, and keeps the
  // objects in the tuple `keep_alives` alive until the asynchronous signal is
  // destroyed.
  __host__
  explicit async_keep_alives(
    unique_stream&& stream, keep_alives_type&& keep_alives
  )
    : async_signal(std::move(stream))
    , keep_alives_(std::move(keep_alives))
  {}

  __host__
  virtual ~async_keep_alives() {}
};

template <typename T>
struct async_value : virtual async_signal
{
  using value_type        = T;
  using raw_const_pointer = value_type const*;

  // Constructs an `async_value` which uses `stream` and has no content.
  __host__
  explicit async_value(unique_stream stream)
    : async_signal(std::move(stream))
  {}

  __host__
  virtual ~async_value() {}

  __host__
  virtual bool valid_content() const noexcept { return false; }

  __host__
  virtual value_type get()
  {
    throw thrust::event_error(event_errc::no_state);
  }

  __host__
  virtual value_type extract()
  {
    throw thrust::event_error(event_errc::no_state);
  }

  // For testing only.
  #if defined(THRUST_ENABLE_FUTURE_RAW_DATA_MEMBER)
  __host__
  virtual raw_const_pointer raw_data() const
  {
    return nullptr;
  }
  #endif
};

template <typename T, typename Pointer, typename... KeepAlives>
struct async_addressable_value_with_keep_alives<
  T, Pointer, std::tuple<KeepAlives...>
> final
  : async_value<T>, async_keep_alives<std::tuple<KeepAlives...>>
{
  using value_type        = typename async_value<T>::value_type;
  using raw_const_pointer = typename async_value<T>::raw_const_pointer;

  using keep_alives_type
    = typename async_keep_alives<std::tuple<KeepAlives...>>::keep_alives_type;

  using pointer
    = typename thrust::detail::pointer_traits<Pointer>::template
      rebind<value_type>::other;
  using const_pointer
    = typename thrust::detail::pointer_traits<Pointer>::template
      rebind<value_type const>::other;

private:
  pointer content_;

public:
  // Constructs an `async_addressable_value_with_keep_alives` which uses
  // `stream`, keeps the objects in the tuple `keep_alives` alive until the
  // asynchronous value is destroyed, and determines the location of its
  // content by evaluating `compute_content(content_keep_alive)`.
  // NOTE: The use of a callback idiom is necessary if the content is stored in
  // place in the content keep alive object, in which case we need to get its
  // address after its been moved into the new signal we're constructing.
  // NOTE: NVCC has a bug that causes it to reorder our base class initializers
  // in generated host code, which leads to -Wreorder warnings.
  THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_BEGIN
  template <typename ComputeContent>
  __host__
  explicit async_addressable_value_with_keep_alives(
    unique_stream&&    stream
  , keep_alives_type&& keep_alives
  , ComputeContent&&   compute_content
  )
    : async_signal(std::move(stream))
    , async_value<T>(std::move(stream))
    , async_keep_alives<keep_alives_type>(
        std::move(stream), std::move(keep_alives)
      )
  {
    content_ = THRUST_FWD(compute_content)(std::get<0>(this->keep_alives_));
  }
  THRUST_DISABLE_CLANG_AND_GCC_INITIALIZER_REORDERING_WARNING_END

  __host__
  bool valid_content() const noexcept final override
  {
    return nullptr != content_;
  }

  // Precondition: `true == valid_content()`.
  __host__
  pointer data() 
  {
    if (!valid_content())
      throw thrust::event_error(event_errc::no_content);

    return content_;
  }

  // Precondition: `true == valid_content()`.
  __host__
  const_pointer data() const 
  {
    if (!valid_content())
      throw thrust::event_error(event_errc::no_content);

    return content_;
  }

  // Blocks.
  // Precondition: `true == valid_content()`.
  __host__
  value_type get() final override
  {
    this->stream().wait();
    return *data();
  }

  // Blocks.
  // Precondition: `true == valid_content()`.
  __host__
  value_type extract() final override
  {
    this->stream().wait();
    return std::move(*data());
  }

  // For testing only.
  #if defined(THRUST_ENABLE_FUTURE_RAW_DATA_MEMBER)
  __host__
  raw_const_pointer raw_data() const final override
  {
    return raw_pointer_cast(content_);
  }
  #endif
};

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename Pointer>
struct weak_promise final
{
  using value_type = typename async_value<T>::value_type;

  using pointer
    = typename thrust::detail::pointer_traits<Pointer>::template
      rebind<T>::other;
  using const_pointer
    = typename thrust::detail::pointer_traits<Pointer>::template
      rebind<T const>::other;

private:
  int device_ = 0;
  pointer content_;

  explicit weak_promise(int device_id, pointer content)
    : device_(device_id), content_(std::move(content))
  {}

public:
  __host__ __device__
  weak_promise() : device_(0), content_{} {}

  __thrust_exec_check_disable__
  weak_promise(weak_promise const&) = default;
  __thrust_exec_check_disable__
  weak_promise(weak_promise&&) = default;
  __thrust_exec_check_disable__
  weak_promise& operator=(weak_promise const&) = default;
  __thrust_exec_check_disable__
  weak_promise& operator=(weak_promise&&) = default;

  template <typename U>
  __host__ __device__
  void set_value(U&& value) &&
  {
    *content_ = THRUST_FWD(value);
  }

  template <
    typename X, typename XPointer
  , typename ComputeContent, typename... Dependencies
  >
  friend __host__
  unique_eager_future_promise_pair<X, XPointer>
  thrust::system::cuda::detail::make_dependent_future(
    ComputeContent&& cc, std::tuple<Dependencies...>&& deps
  );
};

///////////////////////////////////////////////////////////////////////////////

} // namespace detail

struct ready_event final
{
  ready_event() = default;

  template <typename U>
  __host__ __device__
  explicit ready_event(ready_future<U>) {}

  __host__ __device__
  static constexpr bool valid_content() noexcept { return true; }

  __host__ __device__
  static constexpr bool ready() noexcept { return true; }
};

template <typename T>
struct ready_future final
{
  using value_type        = T;
  using raw_const_pointer = T const*;

private:
  value_type value_;

public:
  __host__ __device__
  ready_future() : value_{} {}

  ready_future(ready_future&&) = default;
  ready_future(ready_future const&) = default;
  ready_future& operator=(ready_future&&) = default;
  ready_future& operator=(ready_future const&) = default;

  template <typename U>
  __host__ __device__
  explicit ready_future(U&& u) : value_(THRUST_FWD(u)) {}

  __host__ __device__
  static constexpr bool valid_content() noexcept { return true; }

  __host__ __device__
  static constexpr bool ready() noexcept { return true; }

  __host__ __device__
  value_type get() const
  {
    return value_;
  }

  THRUST_NODISCARD __host__ __device__
  value_type extract() 
  {
    return std::move(value_);
  }

  #if defined(THRUST_ENABLE_FUTURE_RAW_DATA_MEMBER)
  // For testing only.
  __host__ __device__
  raw_const_pointer data() const
  {
    return addressof(value_);
  }
  #endif
};

struct unique_eager_event final
{
protected:
  int device_ = 0;
  std::unique_ptr<detail::async_signal> async_signal_;

  __host__
  explicit unique_eager_event(
    int device_id, std::unique_ptr<detail::async_signal> async_signal
  )
    : device_(device_id), async_signal_(std::move(async_signal))
  {}

public:
  __host__
  unique_eager_event()
    : device_(0), async_signal_()
  {}

  unique_eager_event(unique_eager_event&&) = default;
  unique_eager_event(unique_eager_event const&) = delete;
  unique_eager_event& operator=(unique_eager_event&&) = default;
  unique_eager_event& operator=(unique_eager_event const&) = delete;

  // Any `unique_eager_future<T>` can be explicitly converted to a
  // `unique_eager_event<void>`.
  template <typename U>
  __host__
  explicit unique_eager_event(unique_eager_future<U>&& other)
    // NOTE: We upcast to `unique_ptr<async_signal>` here.
    : device_(other.where()), async_signal_(std::move(other.async_signal_))
  {}

  __host__
  // NOTE: We take `new_stream_t` by `const&` because it is incomplete here.
  explicit unique_eager_event(new_stream_t const&)
    : device_(0)
    , async_signal_(new detail::async_signal(detail::unique_stream{}))
  {
    thrust::cuda_cub::throw_on_error(cudaGetDevice(&device_));
  }

  __host__
  virtual ~unique_eager_event()
  {
    // FIXME: If we could asynchronously handle destruction of keep alives, we
    // could avoid doing this.
    if (valid_stream()) wait();
  }

  __host__
  bool valid_stream() const noexcept
  {
    return bool(async_signal_);
  }

  __host__
  bool ready() const noexcept
  {
    if (valid_stream())
      return stream().ready();
    else
      return false;
  }

  // Precondition: `true == valid_stream()`.
  __host__
  detail::unique_stream& stream()
  {
    if (!valid_stream())
      throw thrust::event_error(event_errc::no_state);

    return async_signal_->stream();
  }
  detail::unique_stream const& stream() const
  {
    if (!valid_stream())
      throw thrust::event_error(event_errc::no_state);

    return async_signal_->stream();
  }

  __host__
  int where() const noexcept { return device_; }

  // Precondition: `true == valid_stream()`.
  __host__
  void wait()
  {
    stream().wait();
  }

  friend __host__
  optional<detail::unique_stream>
  thrust::system::cuda::detail::try_acquire_stream(
    int device_id, unique_eager_event& parent
    ) noexcept;

  template <typename... Dependencies>
  friend __host__
  unique_eager_event
  thrust::system::cuda::detail::make_dependent_event(
    std::tuple<Dependencies...>&& deps
  );
};

template <typename T>
struct unique_eager_future final
{
  THRUST_STATIC_ASSERT_MSG(
    (!std::is_same<T, remove_cvref_t<void>>::value)
  , "`thrust::event` should be used to express valueless futures"
  );

  using value_type        = typename detail::async_value<T>::value_type;
  using raw_const_pointer = typename detail::async_value<T>::raw_const_pointer;

private:
  int device_ = 0;
  std::unique_ptr<detail::async_value<value_type>> async_signal_;

  __host__
  explicit unique_eager_future(
    int device_id, std::unique_ptr<detail::async_value<value_type>> async_signal
  )
    : device_(device_id), async_signal_(std::move(async_signal))
  {}

public:
  __host__
  unique_eager_future()
    : device_(0), async_signal_()
  {}

  unique_eager_future(unique_eager_future&&) = default;
  unique_eager_future(unique_eager_future const&) = delete;
  unique_eager_future& operator=(unique_eager_future&&) = default;
  unique_eager_future& operator=(unique_eager_future const&) = delete;

  __host__
  // NOTE: We take `new_stream_t` by `const&` because it is incomplete here.
  explicit unique_eager_future(new_stream_t const&)
    : device_(0)
    , async_signal_(new detail::async_value<value_type>(detail::unique_stream{}))
  {
    thrust::cuda_cub::throw_on_error(cudaGetDevice(&device_));
  }

  __host__
  ~unique_eager_future()
  {
    // FIXME: If we could asynchronously handle destruction of keep alives, we
    // could avoid doing this.
    if (valid_stream()) wait();
  }

  __host__
  bool valid_stream() const noexcept
  {
    return bool(async_signal_);
  }

  __host__
  bool valid_content() const noexcept
  {
    if (!valid_stream())
      return false;

    // We might have been constructed with `new_stream_t`, in which case we'd
    // have an async_value, but it doesn't have content.
    return async_signal_->valid_content();
  }

  // Precondition: `true == valid_stream()`.
  __host__
  bool ready() const noexcept
  {
    if (valid_stream())
      return stream().ready();
    else
      return false;
  }

  // Precondition: `true == valid_stream()`.
  __host__
  detail::unique_stream& stream()
  {
    if (!valid_stream())
      throw thrust::event_error(event_errc::no_state);

    return async_signal_->stream();
  }
  __host__
  detail::unique_stream const& stream() const
  {
    if (!valid_stream())
      throw thrust::event_error(event_errc::no_state);

    return async_signal_->stream();
  }

  __host__
  int where() const noexcept { return device_; }

  // Blocks.
  // Precondition: `true == valid_stream()`.
  __host__
  void wait()
  {
    stream().wait();
  }

  // Blocks.
  // Precondition: `true == valid_content()`.
  __host__
  value_type get()
  {
    if (!valid_content())
      throw thrust::event_error(event_errc::no_content);

    return async_signal_->get();
  }

  // Blocks.
  // Precondition: `true == valid_content()`.
  THRUST_NODISCARD __host__
  value_type extract()
  {
    if (!valid_content())
      throw thrust::event_error(event_errc::no_content);

    value_type tmp(async_signal_->extract());
    async_signal_.reset();
    return tmp;
  }

  // For testing only.
  #if defined(THRUST_ENABLE_FUTURE_RAW_DATA_MEMBER)
  // Precondition: `true == valid_stream()`.
  __host__
  raw_const_pointer raw_data() const
  {
    if (!valid_stream())
      throw thrust::event_error(event_errc::no_state);

    return async_signal_->raw_data();
  }
  #endif

  template <typename X>
  friend __host__
  optional<detail::unique_stream>
  thrust::system::cuda::detail::try_acquire_stream(
    int device_id, unique_eager_future<X>& parent
    ) noexcept;

  template <
    typename X, typename XPointer
  , typename ComputeContent, typename... Dependencies
  >
  friend __host__
  detail::unique_eager_future_promise_pair<X, XPointer>
  thrust::system::cuda::detail::make_dependent_future(
    ComputeContent&& cc, std::tuple<Dependencies...>&& deps
  );

  friend struct unique_eager_event;
};

///////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename X, typename Deleter>
__host__
optional<unique_stream>
try_acquire_stream(int, std::unique_ptr<X, Deleter>&) noexcept
{
  // There's no stream to acquire!
  return {};
}

inline __host__
optional<unique_stream>
try_acquire_stream(int, unique_stream& stream) noexcept
{
  return {std::move(stream)};
}

inline __host__
optional<unique_stream>
try_acquire_stream(int, ready_event&) noexcept
{
  // There's no stream to acquire!
  return {};
}

template <typename X>
__host__
optional<unique_stream>
try_acquire_stream(int, ready_future<X>&) noexcept
{
  // There's no stream to acquire!
  return {};
}

__host__
optional<unique_stream>
try_acquire_stream(int device_id, unique_eager_event& parent) noexcept
{
  // We have unique ownership, so we can always steal the stream if the future
  // has one as long as they are on the same device as us.
  if (parent.valid_stream())
    if (device_id == parent.device_)
      return std::move(parent.async_signal_->stream());

  return {};
}

template <typename X>
__host__
optional<unique_stream>
try_acquire_stream(int device_id, unique_eager_future<X>& parent) noexcept
{
  // We have unique ownership, so we can always steal the stream if the future
  // has one as long as they are on the same device as us.
  if (parent.valid_stream())
    if (device_id == parent.device_)
      return std::move(parent.async_signal_->stream());

  return {};
}

///////////////////////////////////////////////////////////////////////////////

template <typename... Dependencies>
__host__
acquired_stream acquire_stream_impl(
  int, std::tuple<Dependencies...>&, index_sequence<>
) noexcept
{
  // We tried to take a stream from all of our dependencies and failed every
  // time, so we need to make a new stream.
  return {unique_stream{}, {}};
}

template <typename... Dependencies, std::size_t I0, std::size_t... Is>
__host__
acquired_stream acquire_stream_impl(
  int device_id
, std::tuple<Dependencies...>& deps, index_sequence<I0, Is...>
) noexcept
{
  auto tr = try_acquire_stream(device_id, std::get<I0>(deps));

  if (tr)
    return {std::move(*tr), {I0}};
  else
    return acquire_stream_impl(device_id, deps, index_sequence<Is...>{});
}

template <typename... Dependencies>
__host__
acquired_stream acquire_stream(
  int device_id
, std::tuple<Dependencies...>& deps
) noexcept
{
  return acquire_stream_impl(
    device_id, deps, make_index_sequence<sizeof...(Dependencies)>{}
  );
}

///////////////////////////////////////////////////////////////////////////////

template <typename X, typename Deleter>
__host__
void create_dependency(
  unique_stream&, std::unique_ptr<X, Deleter>&
) noexcept
{}

inline __host__
void create_dependency(
  unique_stream&, ready_event&
) noexcept
{}

template <typename T>
__host__
void create_dependency(
  unique_stream&, ready_future<T>&
) noexcept
{}

inline __host__
void create_dependency(
  unique_stream& child, unique_stream& parent
)
{
  child.depend_on(parent);
}

inline __host__
void create_dependency(
  unique_stream& child, unique_eager_event& parent
)
{
  child.depend_on(parent.stream());
}

template <typename X>
__host__
void create_dependency(
  unique_stream& child, unique_eager_future<X>& parent
)
{
  child.depend_on(parent.stream());
}

template <typename... Dependencies>
__host__
void create_dependencies_impl(
  acquired_stream&
, std::tuple<Dependencies...>&, index_sequence<>
)
{}

template <typename... Dependencies, std::size_t I0, std::size_t... Is>
__host__
void create_dependencies_impl(
  acquired_stream& as
, std::tuple<Dependencies...>& deps, index_sequence<I0, Is...>
)
{
  // We only need to wait on the current dependency if we didn't steal our
  // stream from it.
  if (!as.acquired_from || *as.acquired_from != I0)
  {
    create_dependency(as.stream, std::get<I0>(deps));
  }

  create_dependencies_impl(as, deps, index_sequence<Is...>{});
}

template <typename... Dependencies>
__host__
void create_dependencies(acquired_stream& as, std::tuple<Dependencies...>& deps)
{
  create_dependencies_impl(
    as, deps, make_index_sequence<sizeof...(Dependencies)>{}
  );
}

///////////////////////////////////////////////////////////////////////////////

// Metafunction that determine which `Dependencies` need to be kept alive.
// Returns the result as an `index_sequence` of indices into the parameter
// pack.
template <typename Tuple, typename Indices>
  struct find_keep_alives_impl;
template <typename Tuple>
  using find_keep_alives
    = typename find_keep_alives_impl<
        Tuple, make_index_sequence<std::tuple_size<Tuple>::value>
      >::type;

template <>
struct find_keep_alives_impl<
  std::tuple<>, index_sequence<>
>
{
  using type = index_sequence<>;
};

// User-provided stream.
template <
  typename... Dependencies
, std::size_t I0, std::size_t... Is
>
struct find_keep_alives_impl<
  std::tuple<unique_stream, Dependencies...>, index_sequence<I0, Is...>
>
{
  // Nothing to keep alive, skip this index.
  using type = typename find_keep_alives_impl<
    std::tuple<Dependencies...>, index_sequence<Is...>
  >::type;
};

template <
  typename... Dependencies
, std::size_t I0, std::size_t... Is
>
struct find_keep_alives_impl<
  std::tuple<ready_event, Dependencies...>, index_sequence<I0, Is...>
>
{
  // Nothing to keep alive, skip this index.
  using type = typename find_keep_alives_impl<
    std::tuple<Dependencies...>, index_sequence<Is...>
  >::type;
};

template <
  typename T, typename... Dependencies
, std::size_t I0, std::size_t... Is
>
struct find_keep_alives_impl<
  std::tuple<ready_future<T>, Dependencies...>, index_sequence<I0, Is...>
>
{
  // Add this index to the list.
  using type = integer_sequence_push_front<
    std::size_t, I0
  , typename find_keep_alives_impl<
      std::tuple<Dependencies...>, index_sequence<Is...>
    >::type
  >;
};

template <
  typename... Dependencies
, std::size_t I0, std::size_t... Is
>
struct find_keep_alives_impl<
  std::tuple<unique_eager_event, Dependencies...>
, index_sequence<I0, Is...>
>
{
  // Add this index to the list.
  using type = integer_sequence_push_front<
    std::size_t, I0
  , typename find_keep_alives_impl<
      std::tuple<Dependencies...>, index_sequence<Is...>
    >::type
  >;
};

template <
  typename X, typename... Dependencies
, std::size_t I0, std::size_t... Is
>
struct find_keep_alives_impl<
  std::tuple<unique_eager_future<X>, Dependencies...>
, index_sequence<I0, Is...>
>
{
  // Add this index to the list.
  using type = integer_sequence_push_front<
    std::size_t, I0
  , typename find_keep_alives_impl<
      std::tuple<Dependencies...>, index_sequence<Is...>
    >::type
  >;
};

// Content storage.
template <
  typename T, typename Deleter, typename... Dependencies
, std::size_t I0, std::size_t... Is
>
struct find_keep_alives_impl<
  std::tuple<std::unique_ptr<T, Deleter>, Dependencies...>
, index_sequence<I0, Is...>
>
{
  // Add this index to the list.
  using type = integer_sequence_push_front<
    std::size_t, I0
  , typename find_keep_alives_impl<
      std::tuple<Dependencies...>, index_sequence<Is...>
    >::type
  >;
};

///////////////////////////////////////////////////////////////////////////////

template <typename... Dependencies>
__host__
unique_eager_event make_dependent_event(std::tuple<Dependencies...>&& deps)
{
  int device_id = 0;
  thrust::cuda_cub::throw_on_error(cudaGetDevice(&device_id));

  // First, either steal a stream from one of our children or make a new one.
  auto as = acquire_stream(device_id, deps);

  // Then, make the stream we've acquired asynchronously wait on all of our
  // dependencies, except the one we stole the stream from.
  create_dependencies(as, deps);

  // Then, we determine which subset of dependencies need to be kept alive.
  auto ka = tuple_subset(
    std::move(deps)
  , find_keep_alives<std::tuple<Dependencies...>>{}
  );

  // Next, we create the asynchronous signal.
  using async_signal_type = async_keep_alives<decltype(ka)>;

  std::unique_ptr<async_signal_type> sig(
    new async_signal_type(std::move(as.stream), std::move(ka))
  );

  // Finally, we create the event object.
  return unique_eager_event(device_id, std::move(sig));
}

///////////////////////////////////////////////////////////////////////////////

template <
  typename X, typename XPointer
, typename ComputeContent, typename... Dependencies
>
__host__
unique_eager_future_promise_pair<X, XPointer>
make_dependent_future(ComputeContent&& cc, std::tuple<Dependencies...>&& deps)
{
  int device_id = 0;
  thrust::cuda_cub::throw_on_error(cudaGetDevice(&device_id));

  // First, either steal a stream from one of our children or make a new one.
  auto as = acquire_stream(device_id, deps);

  // Then, make the stream we've acquired asynchronously wait on all of our
  // dependencies, except the one we stole the stream from.
  create_dependencies(as, deps);

  // Then, we determine which subset of dependencies need to be kept alive.
  auto ka = tuple_subset(
    std::move(deps)
  , find_keep_alives<std::tuple<Dependencies...>>{}
  );

  // Next, we create the asynchronous value.
  using async_signal_type = async_addressable_value_with_keep_alives<
    X, XPointer, decltype(ka)
  >;

  std::unique_ptr<async_signal_type> sig(
    new async_signal_type(std::move(as.stream), std::move(ka), std::move(cc))
  );
 
  // Finally, we create the promise and future objects.
  weak_promise<X, XPointer> child_prom(device_id, sig->data());
  unique_eager_future<X> child_fut(device_id, std::move(sig));

  return unique_eager_future_promise_pair<X, XPointer>
    {std::move(child_fut), std::move(child_prom)};
}

} // namespace detail

///////////////////////////////////////////////////////////////////////////////

template <typename... Events>
__host__
unique_eager_event when_all(Events&&... evs)
// TODO: Constrain to events, futures, and maybe streams (currently allows keep
// alives).
{
  return detail::make_dependent_event(std::make_tuple(std::move(evs)...)); 
}

// ADL hook for transparent `.after` move support.
inline __host__
auto capture_as_dependency(unique_eager_event& dependency)
THRUST_DECLTYPE_RETURNS(std::move(dependency))

// ADL hook for transparent `.after` move support.
template <typename X>
__host__
auto capture_as_dependency(unique_eager_future<X>& dependency)
THRUST_DECLTYPE_RETURNS(std::move(dependency))

}} // namespace system::cuda

THRUST_NAMESPACE_END

#endif // C++14

