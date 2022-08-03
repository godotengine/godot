#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2014

#include <unittest/unittest.h>
#include <unittest/util_async.h>

#include <thrust/limits.h>
#include <thrust/async/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define DEFINE_ASYNC_COPY_CALLABLE(name, ...)                                 \
  struct THRUST_PP_CAT2(name, _fn)                                            \
  {                                                                           \
    template <typename ForwardIt, typename Sentinel, typename OutputIt>       \
    __host__                                                                  \
    auto operator()(                                                          \
      ForwardIt&& first, Sentinel&& last, OutputIt&& output                   \
    ) const                                                                   \
    THRUST_RETURNS(                                                           \
      ::thrust::async::copy(                                                  \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last), THRUST_FWD(output)               \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy
);

DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_host,   thrust::host
);
DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_device, thrust::device
);

DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_host_to_device,    thrust::host,   thrust::device
);
DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_device_to_host,    thrust::device, thrust::host
);
DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_host_to_host,      thrust::host,   thrust::host
);
DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_device_to_device,  thrust::device, thrust::device
);

#undef DEFINE_ASYNC_COPY_CALLABLE

///////////////////////////////////////////////////////////////////////////////

template <typename AsyncCopyCallable>
struct test_async_copy_host_to_device
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
      thrust::device_vector<T> d0(n);

      auto f0 = AsyncCopyCallable{}(
        h0.begin(), h0.end(), d0.begin()
      );

      f0.wait();

      ASSERT_EQUAL(h0, d0);
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_host_to_device<invoke_async_copy_fn>::tester
, BuiltinNumericTypes
, test_async_copy_trivially_relocatable_elements_host_to_device
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_host_to_device<invoke_async_copy_host_to_device_fn>::tester
, BuiltinNumericTypes
, test_async_copy_trivially_relocatable_elements_host_to_device_policies
);

///////////////////////////////////////////////////////////////////////////////

template <typename AsyncCopyCallable>
struct test_async_copy_device_to_host
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
      thrust::host_vector<T>   h1(n);
      thrust::device_vector<T> d0(n);

      thrust::copy(h0.begin(), h0.end(), d0.begin());

      ASSERT_EQUAL(h0, d0);

      auto f0 = AsyncCopyCallable{}(
        d0.begin(), d0.end(), h1.begin()
      );

      f0.wait();

      ASSERT_EQUAL(h0, d0);
      ASSERT_EQUAL(d0, h1);
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_device_to_host<invoke_async_copy_fn>::tester
, BuiltinNumericTypes
, test_async_copy_trivially_relocatable_elements_device_to_host
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_device_to_host<invoke_async_copy_device_to_host_fn>::tester
, BuiltinNumericTypes
, test_async_copy_trivially_relocatable_elements_device_to_host_policies
);

///////////////////////////////////////////////////////////////////////////////

template <typename AsyncCopyCallable>
struct test_async_copy_device_to_device
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
      thrust::device_vector<T> d0(n);
      thrust::device_vector<T> d1(n);

      thrust::copy(h0.begin(), h0.end(), d0.begin());

      ASSERT_EQUAL(h0, d0);

      auto f0 = AsyncCopyCallable{}(
        d0.begin(), d0.end(), d1.begin()
      );

      f0.wait();

      ASSERT_EQUAL(h0, d0);
      ASSERT_EQUAL(d0, d1);
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_device_to_device<invoke_async_copy_fn>::tester
, NumericTypes
, test_async_copy_device_to_device
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_device_to_device<invoke_async_copy_device_fn>::tester
, NumericTypes
, test_async_copy_device_to_device_policy
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_device_to_device<invoke_async_copy_device_to_device_fn>::tester
, NumericTypes
, test_async_copy_device_to_device_policies
);

///////////////////////////////////////////////////////////////////////////////

// Non ContiguousIterator input.
template <typename AsyncCopyCallable>
struct test_async_copy_counting_iterator_input_to_device_vector
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::counting_iterator<T> first(0);
      thrust::counting_iterator<T> last(
        unittest::truncate_to_max_representable<T>(n)
      );

      thrust::device_vector<T> d0(n);
      thrust::device_vector<T> d1(n);

      thrust::copy(first, last, d0.begin());

      auto f0 = AsyncCopyCallable{}(
        first, last, d1.begin()
      );

      f0.wait();

      ASSERT_EQUAL(d0, d1);
    }
  };
};
// TODO: Re-add custom_numeric when it supports counting iterators.
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_counting_iterator_input_to_device_vector<
    invoke_async_copy_fn
  >::tester
, BuiltinNumericTypes
, test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_device
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_counting_iterator_input_to_device_vector<
    invoke_async_copy_device_fn
  >::tester
, BuiltinNumericTypes
, test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_device_policy
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_counting_iterator_input_to_device_vector<
    invoke_async_copy_device_to_device_fn
  >::tester
, BuiltinNumericTypes
, test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_device_policies
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_counting_iterator_input_to_device_vector<
    invoke_async_copy_host_to_device_fn
  >::tester
, BuiltinNumericTypes
, test_async_copy_counting_iterator_input_host_to_device_policies
);

///////////////////////////////////////////////////////////////////////////////

// Non ContiguousIterator input.
template <typename AsyncCopyCallable>
struct test_async_copy_counting_iterator_input_to_host_vector
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::counting_iterator<T> first(0);
      thrust::counting_iterator<T> last(
        unittest::truncate_to_max_representable<T>(n)
      );

      thrust::host_vector<T> d0(n);
      thrust::host_vector<T> d1(n);

      thrust::copy(first, last, d0.begin());

      auto f0 = AsyncCopyCallable{}(
        first, last, d1.begin()
      );

      f0.wait();

      ASSERT_EQUAL(d0, d1);

      #if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_INTEL)
      // ICC fails this for some unknown reason - see #1468.
      KNOWN_FAILURE;
      #endif
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_counting_iterator_input_to_host_vector<
    invoke_async_copy_fn
  >::tester
, BuiltinNumericTypes
, test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_host
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_counting_iterator_input_to_host_vector<
    invoke_async_copy_device_to_host_fn
  >::tester
, BuiltinNumericTypes
, test_async_copy_counting_iterator_input_trivially_relocatable_elements_device_to_host_policies
);

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_async_copy_roundtrip
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0(n);

    auto e0 = thrust::async::copy(
      thrust::host, thrust::device
    , h0.begin(), h0.end(), d0.begin()
    );

    auto e1 = thrust::async::copy(
      thrust::device.after(e0), thrust::host
    , d0.begin(), d0.end(), h0.begin()
    );

    TEST_EVENT_WAIT(e1);

    ASSERT_EQUAL(h0, d0);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  test_async_copy_roundtrip
, BuiltinNumericTypes
, test_async_copy_trivially_relocatable_elements_roundtrip
);

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_async_copy_after
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0(unittest::random_integers<T>(n));
    thrust::host_vector<T>   h1(n);
    thrust::device_vector<T> d0(n);
    thrust::device_vector<T> d1(n);
    thrust::device_vector<T> d2(n);

    auto e0 = thrust::async::copy(
      h0.begin(), h0.end(), d0.begin()
    );

    ASSERT_EQUAL(true, e0.valid_stream());

    auto const e0_stream = e0.stream().native_handle();

    auto e1 = thrust::async::copy(
      thrust::device.after(e0), d0.begin(), d0.end(), d1.begin()
    );

    // Verify that double consumption of a future produces an exception.
    ASSERT_THROWS_EQUAL(
      auto x = thrust::async::copy(
        thrust::device.after(e0), d0.begin(), d0.end(), d1.begin()
      );
      THRUST_UNUSED_VAR(x)
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_state)
    );

    ASSERT_EQUAL_QUIET(e0_stream, e1.stream().native_handle());

    auto after_policy2 = thrust::device.after(e1);

    auto e2 = thrust::async::copy(
      thrust::host, after_policy2
    , h0.begin(), h0.end(), d2.begin()
    );

    // Verify that double consumption of a policy produces an exception.
    ASSERT_THROWS_EQUAL(
      auto x = thrust::async::copy(
        thrust::host, after_policy2
      , h0.begin(), h0.end(), d2.begin()
      );
      THRUST_UNUSED_VAR(x)
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_state)
    );

    ASSERT_EQUAL_QUIET(e0_stream, e2.stream().native_handle());

    auto e3 = thrust::async::copy(
      thrust::device.after(e2), thrust::host
    , d1.begin(), d1.end(), h1.begin()
    );

    ASSERT_EQUAL_QUIET(e0_stream, e3.stream().native_handle());

    TEST_EVENT_WAIT(e3);

    ASSERT_EQUAL(h0, h1);
    ASSERT_EQUAL(h0, d0);
    ASSERT_EQUAL(h0, d1);
    ASSERT_EQUAL(h0, d2);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(
  test_async_copy_after
, BuiltinNumericTypes
);

///////////////////////////////////////////////////////////////////////////////

// TODO: device_to_device NonContiguousIterator output (discard_iterator).

// TODO: host_to_device non trivially relocatable.

// TODO: device_to_host non trivially relocatable.

// TODO: host_to_device NonContiguousIterator input (counting_iterator).

// TODO: host_to_device NonContiguousIterator output (discard_iterator).

// TODO: device_to_host NonContiguousIterator input (counting_iterator).

// TODO: device_to_host NonContiguousIterator output (discard_iterator).

// TODO: Mixed types, needs loosening of `is_trivially_relocatable_to` logic.

// TODO: H->D copy, then dependent D->H copy (round trip).
// Can't do this today because we can't do cross-system with explicit policies.

#endif

