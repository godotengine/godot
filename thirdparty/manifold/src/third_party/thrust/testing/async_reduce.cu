#define THRUST_ENABLE_FUTURE_RAW_DATA_MEMBER

#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2014

#include <unittest/unittest.h>
#include <unittest/util_async.h>

#include <thrust/async/reduce.h>
#include <thrust/async/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <typename T>
struct custom_plus
{
  __host__ __device__
  T operator()(T lhs, T rhs) const
  {
    return lhs + rhs;
  }
};

#define DEFINE_STATEFUL_ASYNC_REDUCE_INVOKER(                                 \
    NAME, MEMBERS, CTOR, DTOR, VALIDATE, ...                                  \
  )                                                                           \
  template <typename T>                                                       \
  struct NAME                                                                 \
  {                                                                           \
    MEMBERS                                                                   \
                                                                              \
    NAME() { CTOR }                                                           \
                                                                              \
    ~NAME() { DTOR }                                                          \
                                                                              \
    template <typename Event>                                                 \
    void validate_event(Event& e)                                             \
    {                                                                         \
      THRUST_UNUSED_VAR(e);                                                   \
      VALIDATE                                                                \
    }                                                                         \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    auto operator()(                                                          \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::reduce(                                                \
        __VA_ARGS__                                                           \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

#define DEFINE_ASYNC_REDUCE_INVOKER(NAME, ...)                                \
  DEFINE_STATEFUL_ASYNC_REDUCE_INVOKER(                                       \
    NAME                                                                      \
  , THRUST_PP_EMPTY(), THRUST_PP_EMPTY(), THRUST_PP_EMPTY(), THRUST_PP_EMPTY()\
  , __VA_ARGS__                                                               \
  )                                                                           \
  /**/

#define DEFINE_SYNC_REDUCE_INVOKER(NAME, ...)                                 \
  template <typename T>                                                       \
  struct NAME                                                                 \
  {                                                                           \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    auto operator()(                                                          \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_RETURNS(                                                           \
      ::thrust::reduce(                                                       \
        __VA_ARGS__                                                           \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker
, THRUST_FWD(first), THRUST_FWD(last)
);
DEFINE_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device
, thrust::device
, THRUST_FWD(first), THRUST_FWD(last)
);
DEFINE_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_allocator
, thrust::device(thrust::device_allocator<void>{})
, THRUST_FWD(first), THRUST_FWD(last)
);
DEFINE_STATEFUL_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_on
  // Members.
, cudaStream_t stream_;
  // Constructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)
  );
  // Destructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamDestroy(stream_)
  );
  // `validate_event` member.
, ASSERT_EQUAL_QUIET(stream_, e.stream().native_handle());
  // Arguments to `thrust::async::reduce`.
, thrust::device.on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
);
DEFINE_STATEFUL_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_allocator_on
  // Members.
, cudaStream_t stream_;
  // Constructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)
  );
  // Destructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamDestroy(stream_)
  );
  // `validate_event` member.
, ASSERT_EQUAL_QUIET(stream_, e.stream().native_handle());
  // Arguments to `thrust::async::reduce`.
, thrust::device(thrust::device_allocator<void>{}).on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
);

DEFINE_SYNC_REDUCE_INVOKER(
  reduce_sync_invoker
, THRUST_FWD(first), THRUST_FWD(last)
);

DEFINE_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_init
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
);
DEFINE_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_init
, thrust::device
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
);
DEFINE_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_allocator_init
, thrust::device(thrust::device_allocator<void>{})
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
);
DEFINE_STATEFUL_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_on_init
  // Members.
, cudaStream_t stream_;
  // Constructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)
  );
  // Destructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamDestroy(stream_)
  );
  // `validate_event` member.
, ASSERT_EQUAL_QUIET(stream_, e.stream().native_handle());
  // Arguments to `thrust::async::reduce`.
, thrust::device.on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
);
DEFINE_STATEFUL_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_allocator_on_init
  // Members.
, cudaStream_t stream_;
  // Constructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)
  );
  // Destructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamDestroy(stream_)
  );
  // `validate_event` member.
, ASSERT_EQUAL_QUIET(stream_, e.stream().native_handle());
  // Arguments to `thrust::async::reduce`.
, thrust::device(thrust::device_allocator<void>{}).on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
);

DEFINE_SYNC_REDUCE_INVOKER(
  reduce_sync_invoker_init
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
);

DEFINE_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_init_plus
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, thrust::plus<T>()
);
DEFINE_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_init_plus
, thrust::device
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, thrust::plus<T>()
);
DEFINE_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_allocator_init_plus
, thrust::device(thrust::device_allocator<void>{})
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, thrust::plus<T>()
);
DEFINE_STATEFUL_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_on_init_plus
  // Members.
, cudaStream_t stream_;
  // Constructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)
  );
  // Destructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamDestroy(stream_)
  );
  // `validate_event` member.
, ASSERT_EQUAL_QUIET(stream_, e.stream().native_handle());
  // Arguments to `thrust::async::reduce`.
, thrust::device.on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, thrust::plus<T>()
);
DEFINE_STATEFUL_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_allocator_on_init_plus
  // Members.
, cudaStream_t stream_;
  // Constructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)
  );
  // Destructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamDestroy(stream_)
  );
  // `validate_event` member.
, ASSERT_EQUAL_QUIET(stream_, e.stream().native_handle());
  // Arguments to `thrust::async::reduce`.
, thrust::device(thrust::device_allocator<void>{}).on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, thrust::plus<T>()
);

DEFINE_SYNC_REDUCE_INVOKER(
  reduce_sync_invoker_init_plus
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, thrust::plus<T>()
);

DEFINE_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_init_custom_plus
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, custom_plus<T>()
);
DEFINE_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_init_custom_plus
, thrust::device
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, custom_plus<T>()
);
DEFINE_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_allocator_init_custom_plus
, thrust::device(thrust::device_allocator<void>{})
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, custom_plus<T>()
);
DEFINE_STATEFUL_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_on_init_custom_plus
  // Members.
, cudaStream_t stream_;
  // Constructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)
  );
  // Destructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamDestroy(stream_)
  );
  // `validate_event` member.
, ASSERT_EQUAL_QUIET(stream_, e.stream().native_handle());
  // Arguments to `thrust::async::reduce`.
, thrust::device.on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, custom_plus<T>()
);
DEFINE_STATEFUL_ASYNC_REDUCE_INVOKER(
  reduce_async_invoker_device_allocator_on_init_custom_plus
  // Members.
, cudaStream_t stream_;
  // Constructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)
  );
  // Destructor.
, thrust::cuda_cub::throw_on_error(
    cudaStreamDestroy(stream_)
  );
  // `validate_event` member.
, ASSERT_EQUAL_QUIET(stream_, e.stream().native_handle());
  // Arguments to `thrust::async::reduce`.
, thrust::device(thrust::device_allocator<void>{}).on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, custom_plus<T>()
);

DEFINE_SYNC_REDUCE_INVOKER(
  reduce_sync_invoker_init_custom_plus
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, custom_plus<T>()
);

///////////////////////////////////////////////////////////////////////////////

template <
  template <typename> class AsyncReduceInvoker
, template <typename> class SyncReduceInvoker
>
struct test_async_reduce
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
      thrust::device_vector<T> d0a(h0);
      thrust::device_vector<T> d0b(h0);
      thrust::device_vector<T> d0c(h0);
      thrust::device_vector<T> d0d(h0);

      AsyncReduceInvoker<T> invoke_async;
      SyncReduceInvoker<T>  invoke_sync;

      ASSERT_EQUAL(h0, d0a);
      ASSERT_EQUAL(h0, d0b);
      ASSERT_EQUAL(h0, d0c);
      ASSERT_EQUAL(h0, d0d);

      auto f0a = invoke_async(d0a.begin(), d0a.end());
      auto f0b = invoke_async(d0b.begin(), d0b.end());
      auto f0c = invoke_async(d0c.begin(), d0c.end());
      auto f0d = invoke_async(d0d.begin(), d0d.end());

      invoke_async.validate_event(f0a);
      invoke_async.validate_event(f0b);
      invoke_async.validate_event(f0c);
      invoke_async.validate_event(f0d);

      // This potentially runs concurrently with the copies.
      auto const r0 = invoke_sync(h0.begin(), h0.end());

      auto const r1a = TEST_FUTURE_VALUE_RETRIEVAL(f0a);
      auto const r1b = TEST_FUTURE_VALUE_RETRIEVAL(f0b);
      auto const r1c = TEST_FUTURE_VALUE_RETRIEVAL(f0c);
      auto const r1d = TEST_FUTURE_VALUE_RETRIEVAL(f0d);

      ASSERT_EQUAL(r0, r1a);
      ASSERT_EQUAL(r0, r1b);
      ASSERT_EQUAL(r0, r1c);
      ASSERT_EQUAL(r0, r1d);
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker
    , reduce_sync_invoker
    >::tester
  )
, NumericTypes
, test_async_reduce
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device
    , reduce_sync_invoker
    >::tester
  )
, NumericTypes
, test_async_reduce_policy
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_allocator
    , reduce_sync_invoker
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_allocator
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_on
    , reduce_sync_invoker
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_on
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_allocator_on
    , reduce_sync_invoker
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_allocator_on
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_init
    , reduce_sync_invoker_init
    >::tester
  )
, NumericTypes
, test_async_reduce_init
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_init
    , reduce_sync_invoker_init
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_init
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_allocator_init
    , reduce_sync_invoker_init
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_allocator_init
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_on_init
    , reduce_sync_invoker_init
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_on_init
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_allocator_on_init
    , reduce_sync_invoker_init
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_allocator_on_init
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_init_plus
    , reduce_sync_invoker_init_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_init_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_init_plus
    , reduce_sync_invoker_init_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_init_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_allocator_init_plus
    , reduce_sync_invoker_init_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_allocator_init_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_on_init_plus
    , reduce_sync_invoker_init_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_on_init_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_allocator_on_init_plus
    , reduce_sync_invoker_init_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_allocator_on_init_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_init_custom_plus
    , reduce_sync_invoker_init_custom_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_init_custom_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_init_custom_plus
    , reduce_sync_invoker_init_custom_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_init_custom_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_allocator_init_custom_plus
    , reduce_sync_invoker_init_custom_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_allocator_init_custom_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_on_init_custom_plus
    , reduce_sync_invoker_init_custom_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_on_init_custom_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce<
      reduce_async_invoker_device_allocator_on_init_custom_plus
    , reduce_sync_invoker_init_custom_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_policy_allocator_on_init_custom_plus
);

///////////////////////////////////////////////////////////////////////////////

template <
  template <typename> class AsyncReduceInvoker
, template <typename> class SyncReduceInvoker
>
struct test_async_reduce_counting_iterator
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()()
    {
      constexpr std::size_t n = 15 * sizeof(T);

      ASSERT_LEQUAL(T(n), unittest::truncate_to_max_representable<T>(n));

      thrust::counting_iterator<T> first(0);
      thrust::counting_iterator<T> last(n);

      AsyncReduceInvoker<T> invoke_async;
      SyncReduceInvoker<T>  invoke_sync;

      auto f0a = invoke_async(first, last);
      auto f0b = invoke_async(first, last);
      auto f0c = invoke_async(first, last);
      auto f0d = invoke_async(first, last);

      invoke_async.validate_event(f0a);
      invoke_async.validate_event(f0b);
      invoke_async.validate_event(f0c);
      invoke_async.validate_event(f0d);

      // This potentially runs concurrently with the copies.
      auto const r0 = invoke_sync(first, last);

      auto const r1a = TEST_FUTURE_VALUE_RETRIEVAL(f0a);
      auto const r1b = TEST_FUTURE_VALUE_RETRIEVAL(f0b);
      auto const r1c = TEST_FUTURE_VALUE_RETRIEVAL(f0c);
      auto const r1d = TEST_FUTURE_VALUE_RETRIEVAL(f0d);

      ASSERT_EQUAL(r0, r1a);
      ASSERT_EQUAL(r0, r1b);
      ASSERT_EQUAL(r0, r1c);
      ASSERT_EQUAL(r0, r1d);
    }
  };
};
DECLARE_GENERIC_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_counting_iterator<
      reduce_async_invoker
    , reduce_sync_invoker
    >::tester
  )
, BuiltinNumericTypes
, test_async_reduce_counting_iterator
);
DECLARE_GENERIC_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_counting_iterator<
      reduce_async_invoker_device
    , reduce_sync_invoker
    >::tester
  )
, BuiltinNumericTypes
, test_async_reduce_policy_counting_iterator
);
DECLARE_GENERIC_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_counting_iterator<
      reduce_async_invoker_init
    , reduce_sync_invoker_init
    >::tester
  )
, BuiltinNumericTypes
, test_async_reduce_counting_iterator_init
);
DECLARE_GENERIC_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_counting_iterator<
      reduce_async_invoker_device_init
    , reduce_sync_invoker_init
    >::tester
  )
, BuiltinNumericTypes
, test_async_reduce_policy_counting_iterator_init
);
DECLARE_GENERIC_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_counting_iterator<
      reduce_async_invoker_init_plus
    , reduce_sync_invoker_init_plus
    >::tester
  )
, BuiltinNumericTypes
, test_async_reduce_counting_iterator_init_plus
);
DECLARE_GENERIC_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_counting_iterator<
      reduce_async_invoker_device_init_plus
    , reduce_sync_invoker_init_plus
    >::tester
  )
, BuiltinNumericTypes
, test_async_reduce_policy_counting_iterator_init_plus
);
DECLARE_GENERIC_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_counting_iterator<
      reduce_async_invoker_init_custom_plus
    , reduce_sync_invoker_init_custom_plus
    >::tester
  )
, BuiltinNumericTypes
, test_async_reduce_counting_iterator_init_custom_plus
);
DECLARE_GENERIC_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_counting_iterator<
      reduce_async_invoker_device_init_custom_plus
    , reduce_sync_invoker_init_custom_plus
    >::tester
  )
, BuiltinNumericTypes
, test_async_reduce_policy_counting_iterator_init_custom_plus
);

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_async_reduce_using
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0a(h0);
    thrust::device_vector<T> d0b(h0);

    ASSERT_EQUAL(h0, d0a);
    ASSERT_EQUAL(h0, d0b);

    thrust::device_future<T> f0a;
    thrust::device_future<T> f0b;

    // When you import the customization points into the global namespace,
    // they should be selected instead of the synchronous algorithms.
    {
      using namespace thrust::async;
      f0a = reduce(d0a.begin(), d0a.end());
    }
    {
      using thrust::async::reduce;
      f0b = reduce(d0b.begin(), d0b.end());
    }

    // ADL should find the synchronous algorithms.
    // This potentially runs concurrently with the copies.
    T const r0 = reduce(h0.begin(), h0.end());

    T const r1a = TEST_FUTURE_VALUE_RETRIEVAL(f0a);
    T const r1b = TEST_FUTURE_VALUE_RETRIEVAL(f0b);

    ASSERT_EQUAL(r0, r1a);
    ASSERT_EQUAL(r0, r1b);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(
  test_async_reduce_using
, NumericTypes
);

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_async_reduce_after
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0(h0);

    ASSERT_EQUAL(h0, d0);

    auto f0 = thrust::async::reduce(
      d0.begin(), d0.end()
    );

    ASSERT_EQUAL(true, f0.valid_stream());
 
    auto const f0_stream = f0.stream().native_handle();

    auto f1 = thrust::async::reduce(
      thrust::device.after(f0), d0.begin(), d0.end()
    );

    // Verify that double consumption of a future produces an exception.
    ASSERT_THROWS_EQUAL(
      auto x = thrust::async::reduce(
        thrust::device.after(f0), d0.begin(), d0.end()
      );
      THRUST_UNUSED_VAR(x)
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_state)
    );

    ASSERT_EQUAL_QUIET(f0_stream, f1.stream().native_handle());

    auto after_policy2 = thrust::device.after(f1);

    auto f2 = thrust::async::reduce(
      after_policy2, d0.begin(), d0.end()
    );

    // Verify that double consumption of a policy produces an exception.
    ASSERT_THROWS_EQUAL(
      auto x = thrust::async::reduce(
        after_policy2, d0.begin(), d0.end()
      );
      THRUST_UNUSED_VAR(x)
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_state)
    );

    ASSERT_EQUAL_QUIET(f0_stream, f2.stream().native_handle());

    // This potentially runs concurrently with the copies.
    T const r0 = thrust::reduce(h0.begin(), h0.end());

    T const r1 = TEST_FUTURE_VALUE_RETRIEVAL(f2);

    ASSERT_EQUAL(r0, r1);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(
  test_async_reduce_after
, NumericTypes
);

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_async_reduce_on_then_after
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0(h0);

    ASSERT_EQUAL(h0, d0);

    cudaStream_t stream;
    thrust::cuda_cub::throw_on_error(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
    );

    auto f0 = thrust::async::reduce(
      thrust::device.on(stream), d0.begin(), d0.end()
    );

    ASSERT_EQUAL_QUIET(stream, f0.stream().native_handle());

    auto f1 = thrust::async::reduce(
      thrust::device.after(f0), d0.begin(), d0.end()
    );

    // Verify that double consumption of a future produces an exception.
    ASSERT_THROWS_EQUAL(
      auto x = thrust::async::reduce(
        thrust::device.after(f0), d0.begin(), d0.end()
      );
      THRUST_UNUSED_VAR(x)
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_state)
    );

    ASSERT_EQUAL_QUIET(stream, f1.stream().native_handle());

    auto after_policy2 = thrust::device.after(f1);

    auto f2 = thrust::async::reduce(
      after_policy2, d0.begin(), d0.end()
    );

    // Verify that double consumption of a policy produces an exception.
    ASSERT_THROWS_EQUAL(
      auto x = thrust::async::reduce(
        after_policy2, d0.begin(), d0.end()
      );
      THRUST_UNUSED_VAR(x)
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_state)
    );

    ASSERT_EQUAL_QUIET(stream, f2.stream().native_handle());

    // This potentially runs concurrently with the copies.
    T const r0 = thrust::reduce(h0.begin(), h0.end());

    T const r1 = TEST_FUTURE_VALUE_RETRIEVAL(f2);

    ASSERT_EQUAL(r0, r1);

    thrust::cuda_cub::throw_on_error(
      cudaStreamDestroy(stream)
    );
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(
  test_async_reduce_on_then_after
, NumericTypes
);

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_async_reduce_allocator_on_then_after
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0(h0);

    ASSERT_EQUAL(h0, d0);

    cudaStream_t stream0;
    thrust::cuda_cub::throw_on_error(
      cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking)
    );

    cudaStream_t stream1;
    thrust::cuda_cub::throw_on_error(
      cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking)
    );

    auto f0 = thrust::async::reduce(
      thrust::device(thrust::device_allocator<void>{}).on(stream0)
    , d0.begin(), d0.end()
    );

    ASSERT_EQUAL_QUIET(stream0, f0.stream().native_handle());

    auto f1 = thrust::async::reduce(
      thrust::device(thrust::device_allocator<void>{}).after(f0)
    , d0.begin(), d0.end()
    );

    ASSERT_THROWS_EQUAL(
      auto x = thrust::async::reduce(
        thrust::device(thrust::device_allocator<void>{}).after(f0)
      , d0.begin(), d0.end()
      );
      THRUST_UNUSED_VAR(x)
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_state)
    );

    ASSERT_EQUAL_QUIET(stream0, f1.stream().native_handle());

    auto f2 = thrust::async::reduce(
      thrust::device(thrust::device_allocator<void>{}).on(stream1).after(f1)
    , d0.begin(), d0.end()
    );

    ASSERT_THROWS_EQUAL(
      auto x = thrust::async::reduce(
        thrust::device(thrust::device_allocator<void>{}).on(stream1).after(f1)
      , d0.begin(), d0.end()
      );
      THRUST_UNUSED_VAR(x)
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_state)
    );

    KNOWN_FAILURE;
    // FIXME: The below fails because you can't combine allocator attachment,
    // `.on`, and `.after`.
    // The `#if 0` can be removed once the KNOWN_FAILURE is resolved.
#if 0
    ASSERT_EQUAL_QUIET(stream1, f2.stream().native_handle());

    // This potentially runs concurrently with the copies.
    T const r0 = thrust::reduce(h0.begin(), h0.end());

    T const r1 = TEST_FUTURE_VALUE_RETRIEVAL(f2);

    ASSERT_EQUAL(r0, r1);

    thrust::cuda_cub::throw_on_error(cudaStreamDestroy(stream0));
    thrust::cuda_cub::throw_on_error(cudaStreamDestroy(stream1));
#endif
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(
  test_async_reduce_allocator_on_then_after
, NumericTypes
);

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_async_reduce_caching
{
  __host__
  void operator()(std::size_t n)
  {
    constexpr std::int64_t m = 32;

    thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0(h0);

    ASSERT_EQUAL(h0, d0);

    T const* f0_raw_data;

    {
      // Perform one reduction to ensure there's an entry in the caching
      // allocator.
      auto f0 = thrust::async::reduce(d0.begin(), d0.end());

      TEST_EVENT_WAIT(f0);

      f0_raw_data = f0.raw_data();
    }

    for (std::int64_t i = 0; i < m; ++i)
    {
      auto f1 = thrust::async::reduce(d0.begin(), d0.end());

      ASSERT_EQUAL(true, f1.valid_stream());
      ASSERT_EQUAL(true, f1.valid_content());

      ASSERT_EQUAL_QUIET(f0_raw_data, f1.raw_data());

      // This potentially runs concurrently with the copies.
      T const r0 = thrust::reduce(h0.begin(), h0.end());

      T const r1 = TEST_FUTURE_VALUE_RETRIEVAL(f1);

      ASSERT_EQUAL(r0, r1);
    }
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(
  test_async_reduce_caching
, NumericTypes
);

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_async_copy_then_reduce
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0a(unittest::random_integers<T>(n));
    thrust::host_vector<T>   h0b(unittest::random_integers<T>(n));
    thrust::host_vector<T>   h0c(unittest::random_integers<T>(n));
    thrust::host_vector<T>   h0d(unittest::random_integers<T>(n));

    thrust::device_vector<T> d0a(n);
    thrust::device_vector<T> d0b(n);
    thrust::device_vector<T> d0c(n);
    thrust::device_vector<T> d0d(n);

    auto f0a = thrust::async::copy(h0a.begin(), h0a.end(), d0a.begin());
    auto f0b = thrust::async::copy(h0b.begin(), h0b.end(), d0b.begin());
    auto f0c = thrust::async::copy(h0c.begin(), h0c.end(), d0c.begin());
    auto f0d = thrust::async::copy(h0d.begin(), h0d.end(), d0d.begin());

    ASSERT_EQUAL(true, f0a.valid_stream());
    ASSERT_EQUAL(true, f0b.valid_stream());
    ASSERT_EQUAL(true, f0c.valid_stream());
    ASSERT_EQUAL(true, f0d.valid_stream());

    auto const f0a_stream = f0a.stream().native_handle();
    auto const f0b_stream = f0b.stream().native_handle();
    auto const f0c_stream = f0c.stream().native_handle();
    auto const f0d_stream = f0d.stream().native_handle();

    auto f1a = thrust::async::reduce(
      thrust::device.after(f0a), d0a.begin(), d0a.end()
    );
    auto f1b = thrust::async::reduce(
      thrust::device.after(f0b), d0b.begin(), d0b.end()
    );
    auto f1c = thrust::async::reduce(
      thrust::device.after(f0c), d0c.begin(), d0c.end()
    );
    auto f1d = thrust::async::reduce(
      thrust::device.after(f0d), d0d.begin(), d0d.end()
    );

    ASSERT_EQUAL(false, f0a.valid_stream());
    ASSERT_EQUAL(false, f0b.valid_stream());
    ASSERT_EQUAL(false, f0c.valid_stream());
    ASSERT_EQUAL(false, f0d.valid_stream());

    ASSERT_EQUAL(true, f1a.valid_stream());
    ASSERT_EQUAL(true, f1a.valid_content());
    ASSERT_EQUAL(true, f1b.valid_stream());
    ASSERT_EQUAL(true, f1b.valid_content());
    ASSERT_EQUAL(true, f1c.valid_stream());
    ASSERT_EQUAL(true, f1c.valid_content());
    ASSERT_EQUAL(true, f1d.valid_stream());
    ASSERT_EQUAL(true, f1d.valid_content());

    // Verify that streams were stolen.
    ASSERT_EQUAL_QUIET(f0a_stream, f1a.stream().native_handle());
    ASSERT_EQUAL_QUIET(f0b_stream, f1b.stream().native_handle());
    ASSERT_EQUAL_QUIET(f0c_stream, f1c.stream().native_handle());
    ASSERT_EQUAL_QUIET(f0d_stream, f1d.stream().native_handle());

    // This potentially runs concurrently with the copies.
    T const r0 = thrust::reduce(h0a.begin(), h0a.end());

    T const r1a = TEST_FUTURE_VALUE_RETRIEVAL(f1a);
    T const r1b = TEST_FUTURE_VALUE_RETRIEVAL(f1b);
    T const r1c = TEST_FUTURE_VALUE_RETRIEVAL(f1c);
    T const r1d = TEST_FUTURE_VALUE_RETRIEVAL(f1d);

    ASSERT_EQUAL(r0, r1a);
    ASSERT_EQUAL(r0, r1b);
    ASSERT_EQUAL(r0, r1c);
    ASSERT_EQUAL(r0, r1d);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(
  test_async_copy_then_reduce
, BuiltinNumericTypes
);

///////////////////////////////////////////////////////////////////////////////

// TODO: when_all from reductions.

#endif

