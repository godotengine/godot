#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp14_required.h>

#if THRUST_CPP_DIALECT >= 2014

#include <unittest/unittest.h>

#include <thrust/future.h>

#define TEST_EVENT_WAIT(e)                                                    \
  ::unittest::test_event_wait(e, __FILE__, __LINE__)                          \
  /**/

#define TEST_FUTURE_VALUE_RETRIEVAL(f)                                        \
  ::unittest::test_future_value_retrieval(f, __FILE__, __LINE__)              \
  /**/

namespace unittest
{

template <typename Event>
__host__
void test_event_wait(
  Event&& e, std::string const& filename = "unknown", int lineno = -1
)
{
  ASSERT_EQUAL_WITH_FILE_AND_LINE(true, e.valid_stream(), filename, lineno);

  e.wait();
  e.wait();

  ASSERT_EQUAL_WITH_FILE_AND_LINE(true, e.valid_stream(), filename, lineno);
  ASSERT_EQUAL_WITH_FILE_AND_LINE(true, e.ready(), filename, lineno);
}

template <typename Future>
__host__
auto test_future_value_retrieval(
  Future&& f, std::string const& filename = "unknown", int lineno = -1
) -> decltype(f.extract())
{
  ASSERT_EQUAL_WITH_FILE_AND_LINE(true, f.valid_stream(), filename, lineno);
  ASSERT_EQUAL_WITH_FILE_AND_LINE(true, f.valid_content(), filename, lineno);

  auto const r0 = f.get();
  auto const r1 = f.get();

  ASSERT_EQUAL_WITH_FILE_AND_LINE(true, f.ready(), filename, lineno);
  ASSERT_EQUAL_WITH_FILE_AND_LINE(true, f.valid_stream(), filename, lineno);
  ASSERT_EQUAL_WITH_FILE_AND_LINE(true, f.valid_content(), filename, lineno);
  ASSERT_EQUAL_WITH_FILE_AND_LINE(r0, r1, filename, lineno);

  auto const r2 = f.extract();

  ASSERT_THROWS_EQUAL_WITH_FILE_AND_LINE(
    auto x = f.extract();
    THRUST_UNUSED_VAR(x)
  , thrust::event_error
  , thrust::event_error(thrust::event_errc::no_content)
  , filename, lineno
  );

  ASSERT_EQUAL_WITH_FILE_AND_LINE(false, f.ready(), filename, lineno);
  ASSERT_EQUAL_WITH_FILE_AND_LINE(false, f.valid_stream(), filename, lineno);
  ASSERT_EQUAL_WITH_FILE_AND_LINE(false, f.valid_content(), filename, lineno);
  ASSERT_EQUAL_WITH_FILE_AND_LINE(r2, r1, filename, lineno);
  ASSERT_EQUAL_WITH_FILE_AND_LINE(r2, r0, filename, lineno);

  return r2;
}

} // namespace unittest

#endif // THRUST_CPP_DIALECT >= 2014
