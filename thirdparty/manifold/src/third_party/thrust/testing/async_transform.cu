#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2014

#include <unittest/unittest.h>
#include <unittest/util_async.h>

#include <thrust/async/transform.h>
#include <thrust/async/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <typename T>
struct divide_by_2
{
  __host__ __device__
  T operator()(T x) const
  {
    return x / 2;
  }
};

#define DEFINE_STATEFUL_ASYNC_TRANSFORM_UNARY_INVOKER(                        \
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
    , typename UnaryOperation                                                 \
    >                                                                         \
    __host__                                                                  \
    auto operator()(                                                          \
      ForwardIt&& first, Sentinel&& last, OutputIt&& output                   \
    , UnaryOperation&& op                                                     \
    )                                                                         \
    THRUST_DECLTYPE_RETURNS(                                                  \
      ::thrust::async::transform(                                             \
        __VA_ARGS__                                                           \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

#define DEFINE_ASYNC_TRANSFORM_UNARY_INVOKER(NAME, ...)                       \
  DEFINE_STATEFUL_ASYNC_TRANSFORM_UNARY_INVOKER(                              \
    NAME                                                                      \
  , THRUST_PP_EMPTY(), THRUST_PP_EMPTY(), THRUST_PP_EMPTY(), THRUST_PP_EMPTY()\
  , __VA_ARGS__                                                               \
  )                                                                           \
  /**/

#define DEFINE_SYNC_TRANSFORM_UNARY_INVOKER(NAME, ...)                        \
  template <typename T>                                                       \
  struct NAME                                                                 \
  {                                                                           \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel, typename OutputIt                \
    , typename UnaryOperation                                                 \
    >                                                                         \
    __host__                                                                  \
    auto operator()(                                                          \
      ForwardIt&& first, Sentinel&& last, OutputIt&& output                   \
    , UnaryOperation&& op                                                     \
    )                                                                         \
    THRUST_RETURNS(                                                           \
      ::thrust::transform(                                                    \
        __VA_ARGS__                                                           \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, THRUST_FWD(op)
);
DEFINE_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker_device
, thrust::device
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, THRUST_FWD(op)
);
DEFINE_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker_device_allocator
, thrust::device(thrust::device_allocator<void>{})
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, THRUST_FWD(op)
);
DEFINE_STATEFUL_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker_device_on
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
  // Arguments to `thrust::async::transform`.
, thrust::device.on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, THRUST_FWD(op)
);
DEFINE_STATEFUL_ASYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_async_invoker_device_allocator_on
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
  // Arguments to `thrust::async::transform`.
, thrust::device(thrust::device_allocator<void>{}).on(stream_)
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, THRUST_FWD(op)
);

DEFINE_SYNC_TRANSFORM_UNARY_INVOKER(
  transform_unary_sync_invoker
, THRUST_FWD(first), THRUST_FWD(last)
, THRUST_FWD(output)
, THRUST_FWD(op)
);

///////////////////////////////////////////////////////////////////////////////

template <
  template <typename> class AsyncTransformUnaryInvoker
, template <typename> class SyncTransformUnaryInvoker
, template <typename> class UnaryOperation
>
struct test_async_transform_unary
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

      thrust::host_vector<T>   h1(n);

      thrust::device_vector<T> d1a(n);
      thrust::device_vector<T> d1b(n);
      thrust::device_vector<T> d1c(n);
      thrust::device_vector<T> d1d(n);

      AsyncTransformUnaryInvoker<T> invoke_async;
      SyncTransformUnaryInvoker<T>  invoke_sync;

      UnaryOperation<T> op;

      ASSERT_EQUAL(h0, d0a);
      ASSERT_EQUAL(h0, d0b);
      ASSERT_EQUAL(h0, d0c);
      ASSERT_EQUAL(h0, d0d);

      auto f0a = invoke_async(d0a.begin(), d0a.end(), d1a.begin(), op);
      auto f0b = invoke_async(d0b.begin(), d0b.end(), d1b.begin(), op);
      auto f0c = invoke_async(d0c.begin(), d0c.end(), d1c.begin(), op);
      auto f0d = invoke_async(d0d.begin(), d0d.end(), d1d.begin(), op);

      invoke_async.validate_event(f0a);
      invoke_async.validate_event(f0b);
      invoke_async.validate_event(f0c);
      invoke_async.validate_event(f0d);

      // This potentially runs concurrently with the copies.
      invoke_sync(h0.begin(), h0.end(), h1.begin(), op);

      TEST_EVENT_WAIT(thrust::when_all(f0a, f0b, f0c, f0d));

      ASSERT_EQUAL(h0, d0a);
      ASSERT_EQUAL(h0, d0b);
      ASSERT_EQUAL(h0, d0c);
      ASSERT_EQUAL(h0, d0d);

      ASSERT_EQUAL(h1, d1a);
      ASSERT_EQUAL(h1, d1b);
      ASSERT_EQUAL(h1, d1c);
      ASSERT_EQUAL(h1, d1d);
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary<
      transform_unary_async_invoker
    , transform_unary_sync_invoker
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_divide_by_2
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary<
      transform_unary_async_invoker_device
    , transform_unary_sync_invoker
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_policy_divide_by_2
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary<
      transform_unary_async_invoker_device_allocator
    , transform_unary_sync_invoker
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_policy_allocator_divide_by_2
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary<
      transform_unary_async_invoker_device_on
    , transform_unary_sync_invoker
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_policy_on_divide_by_2
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary<
      transform_unary_async_invoker_device_allocator_on
    , transform_unary_sync_invoker
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_policy_allocator_on_divide_by_2
);

///////////////////////////////////////////////////////////////////////////////

template <
  template <typename> class AsyncTransformUnaryInvoker
, template <typename> class SyncTransformUnaryInvoker
, template <typename> class UnaryOperation
>
struct test_async_transform_unary_inplace
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

      AsyncTransformUnaryInvoker<T> invoke_async;
      SyncTransformUnaryInvoker<T>  invoke_sync;

      UnaryOperation<T> op;

      ASSERT_EQUAL(h0, d0a);
      ASSERT_EQUAL(h0, d0b);
      ASSERT_EQUAL(h0, d0c);
      ASSERT_EQUAL(h0, d0d);

      auto f0a = invoke_async(d0a.begin(), d0a.end(), d0a.begin(), op);
      auto f0b = invoke_async(d0b.begin(), d0b.end(), d0b.begin(), op);
      auto f0c = invoke_async(d0c.begin(), d0c.end(), d0c.begin(), op);
      auto f0d = invoke_async(d0d.begin(), d0d.end(), d0d.begin(), op);

      invoke_async.validate_event(f0a);
      invoke_async.validate_event(f0b);
      invoke_async.validate_event(f0c);
      invoke_async.validate_event(f0d);

      // This potentially runs concurrently with the copies.
      invoke_sync(h0.begin(), h0.end(), h0.begin(), op);

      TEST_EVENT_WAIT(thrust::when_all(f0a, f0b, f0c, f0d));

      ASSERT_EQUAL(h0, d0a);
      ASSERT_EQUAL(h0, d0b);
      ASSERT_EQUAL(h0, d0c);
      ASSERT_EQUAL(h0, d0d);
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary_inplace<
      transform_unary_async_invoker
    , transform_unary_sync_invoker
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_inplace_divide_by_2
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary_inplace<
      transform_unary_async_invoker_device
    , transform_unary_sync_invoker
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_inplace_policy_divide_by_2
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary_inplace<
      transform_unary_async_invoker_device_allocator
    , transform_unary_sync_invoker
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_inplace_policy_allocator_divide_by_2
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary_inplace<
      transform_unary_async_invoker_device_on
    , transform_unary_sync_invoker
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_inplace_policy_on_divide_by_2
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary_inplace<
      transform_unary_async_invoker_device_allocator_on
    , transform_unary_sync_invoker
    , divide_by_2
    >::tester
  )
, NumericTypes
, test_async_transform_unary_inplace_policy_allocator_on_divide_by_2
);

///////////////////////////////////////////////////////////////////////////////

template <
  template <typename> class AsyncTransformUnaryInvoker
, template <typename> class SyncTransformUnaryInvoker
, template <typename> class UnaryOperation
>
struct test_async_transform_unary_counting_iterator
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

      thrust::host_vector<T>   h0(n);

      thrust::device_vector<T> d0a(n);
      thrust::device_vector<T> d0b(n);
      thrust::device_vector<T> d0c(n);
      thrust::device_vector<T> d0d(n);

      AsyncTransformUnaryInvoker<T> invoke_async;
      SyncTransformUnaryInvoker<T>  invoke_sync;

      UnaryOperation<T> op;

      auto f0a = invoke_async(first, last, d0a.begin(), op);
      auto f0b = invoke_async(first, last, d0b.begin(), op);
      auto f0c = invoke_async(first, last, d0c.begin(), op);
      auto f0d = invoke_async(first, last, d0d.begin(), op);

      invoke_async.validate_event(f0a);
      invoke_async.validate_event(f0b);
      invoke_async.validate_event(f0c);
      invoke_async.validate_event(f0d);

      // This potentially runs concurrently with the copies.
      invoke_sync(first, last, h0.begin(), op);

      TEST_EVENT_WAIT(thrust::when_all(f0a, f0b, f0c, f0d));

      ASSERT_EQUAL(h0, d0a);
      ASSERT_EQUAL(h0, d0b);
      ASSERT_EQUAL(h0, d0c);
      ASSERT_EQUAL(h0, d0d);
    }
  };
};
DECLARE_GENERIC_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary_counting_iterator<
      transform_unary_async_invoker
    , transform_unary_sync_invoker
    , divide_by_2
    >::tester
  )
, BuiltinNumericTypes
, test_async_transform_unary_counting_iterator_divide_by_2
);
DECLARE_GENERIC_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_transform_unary_counting_iterator<
      transform_unary_async_invoker_device
    , transform_unary_sync_invoker
    , divide_by_2
    >::tester
  )
, BuiltinNumericTypes
, test_async_transform_unary_counting_iterator_policy_divide_by_2
);

///////////////////////////////////////////////////////////////////////////////

template <
  template <typename> class UnaryOperation
>
struct test_async_transform_using
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

      thrust::host_vector<T>   h1(n);

      thrust::device_vector<T> d1a(n);
      thrust::device_vector<T> d1b(n);

      UnaryOperation<T> op;

      ASSERT_EQUAL(h0, d0a);
      ASSERT_EQUAL(h0, d0b);

      thrust::device_event f0a;
      thrust::device_event f0b;

      // When you import the customization points into the global namespace,
      // they should be selected instead of the synchronous algorithms.
      {
        using namespace thrust::async;
        f0a = transform(d0a.begin(), d0a.end(), d1a.begin(), op);
      }
      {
        using thrust::async::transform;
        f0b = transform(d0b.begin(), d0b.end(), d1b.begin(), op);
      }

      // ADL should find the synchronous algorithms.
      // This potentially runs concurrently with the copies.
      transform(h0.begin(), h0.end(), h1.begin(), op);

      TEST_EVENT_WAIT(thrust::when_all(f0a, f0b));

      ASSERT_EQUAL(h0, d0a);
      ASSERT_EQUAL(h0, d0b);

      ASSERT_EQUAL(h1, d1a);
      ASSERT_EQUAL(h1, d1b);
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND(test_async_transform_using<divide_by_2>::tester)
, NumericTypes
, test_async_transform_using_divide_by_2
);

///////////////////////////////////////////////////////////////////////////////

#endif

