#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2014

#include <unittest/unittest.h>
#include <unittest/util_async.h>

#include <thrust/event.h>

///////////////////////////////////////////////////////////////////////////////

__host__
void test_event_default_constructed()
{
  THRUST_STATIC_ASSERT(
    (std::is_same<
      thrust::event<decltype(thrust::device)>
    , thrust::unique_eager_event<decltype(thrust::device)>
    >::value)
  );

  THRUST_STATIC_ASSERT(
    (std::is_same<
      thrust::event<decltype(thrust::device)>
    , thrust::device_event
    >::value)
  );

  THRUST_STATIC_ASSERT(
    (std::is_same<
      thrust::device_event
    , thrust::device_unique_eager_event
    >::value)
  );

  thrust::device_event e0;

  ASSERT_EQUAL(false, e0.valid_stream());

  ASSERT_THROWS_EQUAL(
    e0.wait()
  , thrust::event_error
  , thrust::event_error(thrust::event_errc::no_state)
  );

  ASSERT_THROWS_EQUAL(
    e0.stream()
  , thrust::event_error
  , thrust::event_error(thrust::event_errc::no_state)
  );
}
DECLARE_UNITTEST(test_event_default_constructed);

///////////////////////////////////////////////////////////////////////////////

__host__
void test_event_new_stream()
{
  auto e0 = thrust::device_event(thrust::new_stream);

  ASSERT_EQUAL(true, e0.valid_stream());

  ASSERT_NOT_EQUAL_QUIET(nullptr, e0.stream().native_handle());    

  e0.wait();

  ASSERT_EQUAL(true, e0.ready());
}
DECLARE_UNITTEST(test_event_new_stream);

///////////////////////////////////////////////////////////////////////////////

__host__
void test_event_linear_chaining()
{
  constexpr std::int64_t n = 1024;

  // Create a new stream.
  auto e0 = thrust::when_all();

  auto const e0_stream = e0.stream().native_handle();

  ASSERT_EQUAL(true, e0.valid_stream());

  ASSERT_NOT_EQUAL_QUIET(nullptr, e0_stream);

  thrust::device_event e1;

  for (std::int64_t i = 0; i < n; ++i)
  {
    ASSERT_EQUAL(true,  e0.valid_stream());

    ASSERT_EQUAL(false, e1.valid_stream());
    ASSERT_EQUAL(false, e1.ready());

    ASSERT_EQUAL_QUIET(e0_stream, e0.stream().native_handle());

    e1 = thrust::when_all(e0);

    ASSERT_EQUAL(false, e0.valid_stream());
    ASSERT_EQUAL(false, e0.ready());

    ASSERT_EQUAL(true,  e1.valid_stream());

    ASSERT_EQUAL(e0_stream, e1.stream().native_handle());

    std::swap(e0, e1);
  }
}
DECLARE_UNITTEST(test_event_linear_chaining);

///////////////////////////////////////////////////////////////////////////////

__host__
void test_event_when_all()
{
  // Create events with new streams.
  auto e0 = thrust::when_all();
  auto e1 = thrust::when_all();
  auto e2 = thrust::when_all();
  auto e3 = thrust::when_all();
  auto e4 = thrust::when_all();
  auto e5 = thrust::when_all();
  auto e6 = thrust::when_all();
  auto e7 = thrust::when_all();

  auto const e0_stream = e0.stream().native_handle();

  ASSERT_EQUAL(true, e0.valid_stream());
  ASSERT_EQUAL(true, e1.valid_stream());
  ASSERT_EQUAL(true, e2.valid_stream());
  ASSERT_EQUAL(true, e3.valid_stream());
  ASSERT_EQUAL(true, e4.valid_stream());
  ASSERT_EQUAL(true, e5.valid_stream());
  ASSERT_EQUAL(true, e6.valid_stream());
  ASSERT_EQUAL(true, e7.valid_stream());

  ASSERT_NOT_EQUAL_QUIET(nullptr, e0_stream);
  ASSERT_NOT_EQUAL_QUIET(nullptr, e1.stream().native_handle());
  ASSERT_NOT_EQUAL_QUIET(nullptr, e2.stream().native_handle());
  ASSERT_NOT_EQUAL_QUIET(nullptr, e3.stream().native_handle());
  ASSERT_NOT_EQUAL_QUIET(nullptr, e4.stream().native_handle());
  ASSERT_NOT_EQUAL_QUIET(nullptr, e5.stream().native_handle());
  ASSERT_NOT_EQUAL_QUIET(nullptr, e6.stream().native_handle());
  ASSERT_NOT_EQUAL_QUIET(nullptr, e7.stream().native_handle());

  auto e8 = thrust::when_all(e0, e1, e2, e3, e4, e5, e6, e7);

  ASSERT_EQUAL(false, e0.valid_stream());
  ASSERT_EQUAL(false, e1.valid_stream());
  ASSERT_EQUAL(false, e2.valid_stream());
  ASSERT_EQUAL(false, e3.valid_stream());
  ASSERT_EQUAL(false, e4.valid_stream());
  ASSERT_EQUAL(false, e5.valid_stream());
  ASSERT_EQUAL(false, e6.valid_stream());
  ASSERT_EQUAL(false, e7.valid_stream());

  ASSERT_EQUAL(true, e8.valid_stream());

  ASSERT_EQUAL(e0_stream, e8.stream().native_handle());

  e8.wait();

  ASSERT_EQUAL(false, e0.ready());
  ASSERT_EQUAL(false, e1.ready());
  ASSERT_EQUAL(false, e2.ready());
  ASSERT_EQUAL(false, e3.ready());
  ASSERT_EQUAL(false, e4.ready());
  ASSERT_EQUAL(false, e5.ready());
  ASSERT_EQUAL(false, e6.ready());
  ASSERT_EQUAL(false, e7.ready());

  ASSERT_EQUAL(true,  e8.ready());
}
DECLARE_UNITTEST(test_event_when_all);

///////////////////////////////////////////////////////////////////////////////
 
#endif

