#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2014

#include <unittest/unittest.h>
#include <unittest/util_async.h>

#include <thrust/future.h>

struct mock {};

using future_value_types = unittest::type_list<
  char
, signed char
, unsigned char
, short
, unsigned short
, int
, unsigned int
, long
, unsigned long
, long long
, unsigned long long
, float
, double
, custom_numeric
, float2
, mock
>;

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_future_default_constructed
{
  __host__
  void operator()()
  {
    THRUST_STATIC_ASSERT(
      (std::is_same<
        thrust::future<decltype(thrust::device), T>
      , thrust::unique_eager_future<decltype(thrust::device), T>
      >::value)
    );

    THRUST_STATIC_ASSERT(
      (std::is_same<
        thrust::future<decltype(thrust::device), T>
      , thrust::device_future<T>
      >::value)
    );

    THRUST_STATIC_ASSERT(
      (std::is_same<
        thrust::device_future<T>
      , thrust::device_unique_eager_future<T>
      >::value)
    );

    thrust::device_future<T> f0;

    ASSERT_EQUAL(false, f0.valid_stream());
    ASSERT_EQUAL(false, f0.valid_content());

    ASSERT_THROWS_EQUAL(
      f0.wait()
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_state)
    );

    ASSERT_THROWS_EQUAL(
      f0.stream()
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_state)
    );

    ASSERT_THROWS_EQUAL(
      f0.get()
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_content)
    );

    ASSERT_THROWS_EQUAL(
      THRUST_UNUSED_VAR(f0.extract())
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_content)
    );
  }
};
DECLARE_GENERIC_UNITTEST_WITH_TYPES(
  test_future_default_constructed
, future_value_types
);

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_future_new_stream
{
  __host__
  void operator()()
  {
    auto f0 = thrust::device_future<T>(thrust::new_stream);

    ASSERT_EQUAL(true,  f0.valid_stream());
    ASSERT_EQUAL(false, f0.valid_content());

    ASSERT_NOT_EQUAL_QUIET(nullptr, f0.stream().native_handle());    

    TEST_EVENT_WAIT(f0);

    ASSERT_EQUAL(true, f0.ready());

    ASSERT_THROWS_EQUAL(
      f0.get()
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_content)
    );

    ASSERT_THROWS_EQUAL(
      THRUST_UNUSED_VAR(f0.extract())
    , thrust::event_error
    , thrust::event_error(thrust::event_errc::no_content)
    );
  }
};
DECLARE_GENERIC_UNITTEST_WITH_TYPES(
  test_future_new_stream
, future_value_types
);

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_future_convert_to_event
{
  __host__
  void operator()()
  {
    auto f0 = thrust::device_future<T>(thrust::new_stream);

    auto const f0_stream = f0.stream().native_handle();

    ASSERT_EQUAL(true,  f0.valid_stream());
    ASSERT_EQUAL(false, f0.valid_content());

    ASSERT_NOT_EQUAL_QUIET(nullptr, f0_stream);

    auto f1 = thrust::device_event(std::move(f0));

    ASSERT_EQUAL(false, f0.valid_stream());
    ASSERT_EQUAL(true,  f1.valid_stream());

    ASSERT_EQUAL(f0_stream, f1.stream().native_handle());
  }
};
DECLARE_GENERIC_UNITTEST_WITH_TYPES(
  test_future_convert_to_event
, future_value_types
);

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct test_future_when_all
{
  __host__
  void operator()()
  {
    // Create futures with new streams.
    auto f0 = thrust::device_future<T>(thrust::new_stream);
    auto f1 = thrust::device_future<T>(thrust::new_stream);
    auto f2 = thrust::device_future<T>(thrust::new_stream);
    auto f3 = thrust::device_future<T>(thrust::new_stream);
    auto f4 = thrust::device_future<T>(thrust::new_stream);
    auto f5 = thrust::device_future<T>(thrust::new_stream);
    auto f6 = thrust::device_future<T>(thrust::new_stream);
    auto f7 = thrust::device_future<T>(thrust::new_stream);

    auto const f0_stream = f0.stream().native_handle();

    ASSERT_EQUAL(true, f0.valid_stream());
    ASSERT_EQUAL(true, f1.valid_stream());
    ASSERT_EQUAL(true, f2.valid_stream());
    ASSERT_EQUAL(true, f3.valid_stream());
    ASSERT_EQUAL(true, f4.valid_stream());
    ASSERT_EQUAL(true, f5.valid_stream());
    ASSERT_EQUAL(true, f6.valid_stream());
    ASSERT_EQUAL(true, f7.valid_stream());

    ASSERT_EQUAL(false, f0.valid_content());
    ASSERT_EQUAL(false, f1.valid_content());
    ASSERT_EQUAL(false, f2.valid_content());
    ASSERT_EQUAL(false, f3.valid_content());
    ASSERT_EQUAL(false, f4.valid_content());
    ASSERT_EQUAL(false, f5.valid_content());
    ASSERT_EQUAL(false, f6.valid_content());
    ASSERT_EQUAL(false, f7.valid_content());

    ASSERT_NOT_EQUAL_QUIET(nullptr, f0_stream);
    ASSERT_NOT_EQUAL_QUIET(nullptr, f1.stream().native_handle());
    ASSERT_NOT_EQUAL_QUIET(nullptr, f2.stream().native_handle());
    ASSERT_NOT_EQUAL_QUIET(nullptr, f3.stream().native_handle());
    ASSERT_NOT_EQUAL_QUIET(nullptr, f4.stream().native_handle());
    ASSERT_NOT_EQUAL_QUIET(nullptr, f5.stream().native_handle());
    ASSERT_NOT_EQUAL_QUIET(nullptr, f6.stream().native_handle());
    ASSERT_NOT_EQUAL_QUIET(nullptr, f7.stream().native_handle());

    auto e0 = thrust::when_all(f0, f1, f2, f3, f4, f5, f6, f7);

    ASSERT_EQUAL(false, f0.valid_stream());
    ASSERT_EQUAL(false, f1.valid_stream());
    ASSERT_EQUAL(false, f2.valid_stream());
    ASSERT_EQUAL(false, f3.valid_stream());
    ASSERT_EQUAL(false, f4.valid_stream());
    ASSERT_EQUAL(false, f5.valid_stream());
    ASSERT_EQUAL(false, f6.valid_stream());
    ASSERT_EQUAL(false, f7.valid_stream());

    ASSERT_EQUAL(false, f0.valid_content());
    ASSERT_EQUAL(false, f1.valid_content());
    ASSERT_EQUAL(false, f2.valid_content());
    ASSERT_EQUAL(false, f3.valid_content());
    ASSERT_EQUAL(false, f4.valid_content());
    ASSERT_EQUAL(false, f5.valid_content());
    ASSERT_EQUAL(false, f6.valid_content());
    ASSERT_EQUAL(false, f7.valid_content());

    ASSERT_EQUAL(true,  e0.valid_stream());

    ASSERT_EQUAL(f0_stream, e0.stream().native_handle());

    TEST_EVENT_WAIT(e0);

    ASSERT_EQUAL(false, f0.ready());
    ASSERT_EQUAL(false, f1.ready());
    ASSERT_EQUAL(false, f2.ready());
    ASSERT_EQUAL(false, f3.ready());
    ASSERT_EQUAL(false, f4.ready());
    ASSERT_EQUAL(false, f5.ready());
    ASSERT_EQUAL(false, f6.ready());
    ASSERT_EQUAL(false, f7.ready());

    ASSERT_EQUAL(true,  e0.ready());
  }
};
DECLARE_GENERIC_UNITTEST_WITH_TYPES(
  test_future_when_all
, future_value_types
);

///////////////////////////////////////////////////////////////////////////////

#endif

