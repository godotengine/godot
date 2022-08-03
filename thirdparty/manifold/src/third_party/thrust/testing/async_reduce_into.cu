#define THRUST_ENABLE_FUTURE_RAW_DATA_MEMBER

#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2014

#include <unittest/unittest.h>
#include <unittest/util_async.h>

#include <thrust/async/reduce.h>
#include <thrust/async/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_make_unique.h>

template <typename T>
struct custom_plus
{
  __host__ __device__
  T operator()(T lhs, T rhs) const
  {
    return lhs + rhs;
  }
};

#define DEFINE_STATEFUL_ASYNC_REDUCE_INTO_INVOKER(                            \
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
      typename ForwardIt, typename Sentinel, typename OutputIt                \
    >                                                                         \
    __host__                                                                  \
    auto operator()(                                                          \
      ForwardIt&& first, Sentinel&& last, OutputIt&& output                   \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::reduce_into(                                           \
        __VA_ARGS__                                                           \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

#define DEFINE_ASYNC_REDUCE_INTO_INVOKER(NAME, ...)                           \
  DEFINE_STATEFUL_ASYNC_REDUCE_INTO_INVOKER(                                  \
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

DEFINE_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
);
DEFINE_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device
, thrust::device
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
);
DEFINE_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_allocator
, thrust::device(thrust::device_allocator<void>{})
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
);
DEFINE_STATEFUL_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_on
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
  // Arguments to `thrust::async::reduce_into`.
, thrust::device.on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
);
DEFINE_STATEFUL_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_allocator_on
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
  // Arguments to `thrust::async::reduce_into`.
, thrust::device(thrust::device_allocator<void>{}).on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
);

DEFINE_SYNC_REDUCE_INVOKER(
  reduce_sync_invoker
, THRUST_FWD(first), THRUST_FWD(last)
);

DEFINE_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_init
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
);
DEFINE_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_init
, thrust::device
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
);
DEFINE_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_allocator_init
, thrust::device(thrust::device_allocator<void>{})
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
);
DEFINE_STATEFUL_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_on_init
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
  // Arguments to `thrust::async::reduce_into`.
, thrust::device.on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
);
DEFINE_STATEFUL_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_allocator_on_init
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
  // Arguments to `thrust::async::reduce_into`.
, thrust::device(thrust::device_allocator<void>{}).on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
);

DEFINE_SYNC_REDUCE_INVOKER(
  reduce_sync_invoker_init
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
);

DEFINE_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_init_plus
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
, thrust::plus<T>()
);
DEFINE_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_init_plus
, thrust::device
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
, thrust::plus<T>()
);
DEFINE_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_allocator_init_plus
, thrust::device(thrust::device_allocator<void>{})
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
, thrust::plus<T>()
);
DEFINE_STATEFUL_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_on_init_plus
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
  // Arguments to `thrust::async::reduce_into`.
, thrust::device.on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
, thrust::plus<T>()
);
DEFINE_STATEFUL_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_allocator_on_init_plus
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
  // Arguments to `thrust::async::reduce_into`.
, thrust::device(thrust::device_allocator<void>{}).on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
, thrust::plus<T>()
);

DEFINE_SYNC_REDUCE_INVOKER(
  reduce_sync_invoker_init_plus
, THRUST_FWD(first), THRUST_FWD(last)
, unittest::random_integer<T>()
, thrust::plus<T>()
);

DEFINE_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_init_custom_plus
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
, custom_plus<T>()
);
DEFINE_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_init_custom_plus
, thrust::device
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
, custom_plus<T>()
);
DEFINE_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_allocator_init_custom_plus
, thrust::device(thrust::device_allocator<void>{})
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
, custom_plus<T>()
);
DEFINE_STATEFUL_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_on_init_custom_plus
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
  // Arguments to `thrust::async::reduce_into`.
, thrust::device.on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, unittest::random_integer<T>()
, custom_plus<T>()
);
DEFINE_STATEFUL_ASYNC_REDUCE_INTO_INVOKER(
  reduce_into_async_invoker_device_allocator_on_init_custom_plus
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
  // Arguments to `thrust::async::reduce_into`.
, thrust::device(thrust::device_allocator<void>{}).on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
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
  template <typename> class AsyncReduceIntoInvoker
, template <typename> class SyncReduceIntoInvoker
>
struct test_async_reduce_into
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

      auto s0a = thrust::device_make_unique<T>();
      auto s0b = thrust::device_make_unique<T>();
      auto s0c = thrust::device_make_unique<T>();
      auto s0d = thrust::device_make_unique<T>();

      auto const s0a_ptr = s0a.get();
      auto const s0b_ptr = s0b.get();
      auto const s0c_ptr = s0c.get();
      auto const s0d_ptr = s0d.get();

      AsyncReduceIntoInvoker<T> invoke_async;
      SyncReduceIntoInvoker<T>  invoke_sync;

      ASSERT_EQUAL(h0, d0a);
      ASSERT_EQUAL(h0, d0b);
      ASSERT_EQUAL(h0, d0c);
      ASSERT_EQUAL(h0, d0d);

      auto f0a = invoke_async(d0a.begin(), d0a.end(), s0a_ptr);
      auto f0b = invoke_async(d0b.begin(), d0b.end(), s0b_ptr);
      auto f0c = invoke_async(d0c.begin(), d0c.end(), s0c_ptr);
      auto f0d = invoke_async(d0d.begin(), d0d.end(), s0d_ptr);

      invoke_async.validate_event(f0a);
      invoke_async.validate_event(f0b);
      invoke_async.validate_event(f0c);
      invoke_async.validate_event(f0d);

      // This potentially runs concurrently with the copies.
      auto const r0 = invoke_sync(h0.begin(), h0.end());

      TEST_EVENT_WAIT(thrust::when_all(f0a, f0b, f0c, f0d));

      ASSERT_EQUAL(r0, *s0a_ptr);
      ASSERT_EQUAL(r0, *s0b_ptr);
      ASSERT_EQUAL(r0, *s0c_ptr);
      ASSERT_EQUAL(r0, *s0d_ptr);
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker
    , reduce_sync_invoker
    >::tester
  )
, NumericTypes
, test_async_reduce_into
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device
    , reduce_sync_invoker
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_allocator
    , reduce_sync_invoker
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_allocator
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_on
    , reduce_sync_invoker
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_on
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_allocator_on
    , reduce_sync_invoker
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_allocator_on
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_init
    , reduce_sync_invoker_init
    >::tester
  )
, NumericTypes
, test_async_reduce_into_init
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_init
    , reduce_sync_invoker_init
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_init
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_allocator_init
    , reduce_sync_invoker_init
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_allocator_init
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_on_init
    , reduce_sync_invoker_init
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_on_init
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_allocator_on_init
    , reduce_sync_invoker_init
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_allocator_on_init
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_init_plus
    , reduce_sync_invoker_init_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_into_init_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_init_plus
    , reduce_sync_invoker_init_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_init_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_allocator_init_plus
    , reduce_sync_invoker_init_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_allocator_init_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_on_init_plus
    , reduce_sync_invoker_init_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_on_init_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_allocator_on_init_plus
    , reduce_sync_invoker_init_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_allocator_on_init_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_init_custom_plus
    , reduce_sync_invoker_init_custom_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_into_init_custom_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_init_custom_plus
    , reduce_sync_invoker_init_custom_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_init_custom_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_allocator_init_custom_plus
    , reduce_sync_invoker_init_custom_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_allocator_init_custom_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_on_init_custom_plus
    , reduce_sync_invoker_init_custom_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_on_init_custom_plus
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_reduce_into<
      reduce_into_async_invoker_device_allocator_on_init_custom_plus
    , reduce_sync_invoker_init_custom_plus
    >::tester
  )
, NumericTypes
, test_async_reduce_into_policy_allocator_on_init_custom_plus
);

#endif

