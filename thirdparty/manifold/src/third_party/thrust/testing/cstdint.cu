#include <unittest/unittest.h>
#include <thrust/detail/cstdint.h>

#include <limits>

void TestStandardIntegerTypes(void)
{
  ASSERT_EQUAL(sizeof(thrust::detail::int8_t),   1lu);
  ASSERT_EQUAL(sizeof(thrust::detail::int16_t),  2lu);
  ASSERT_EQUAL(sizeof(thrust::detail::int32_t),  4lu);
  ASSERT_EQUAL(sizeof(thrust::detail::int64_t),  8lu);
  ASSERT_EQUAL(sizeof(thrust::detail::uint8_t),  1lu);
  ASSERT_EQUAL(sizeof(thrust::detail::uint16_t), 2lu);
  ASSERT_EQUAL(sizeof(thrust::detail::uint32_t), 4lu);
  ASSERT_EQUAL(sizeof(thrust::detail::uint64_t), 8lu);

  ASSERT_EQUAL(sizeof(thrust::detail::intptr_t),  sizeof(void *));
  ASSERT_EQUAL(sizeof(thrust::detail::uintptr_t), sizeof(void *));

  ASSERT_EQUAL(std::numeric_limits<thrust::detail::int8_t >::is_integer,   true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::int16_t>::is_integer,   true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::int32_t>::is_integer,   true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::int64_t>::is_integer,   true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::uint8_t >::is_integer,  true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::uint16_t>::is_integer,  true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::uint32_t>::is_integer,  true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::uint64_t>::is_integer,  true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::int8_t >::is_signed,    true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::int16_t>::is_signed,    true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::int32_t>::is_signed,    true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::int64_t>::is_signed,    true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::uint8_t >::is_signed,   false);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::uint16_t>::is_signed,   false);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::uint32_t>::is_signed,   false);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::uint64_t>::is_signed,   false);
  
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::intptr_t>::is_integer,  true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::uintptr_t>::is_integer, true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::intptr_t>::is_signed,   true);
  ASSERT_EQUAL(std::numeric_limits<thrust::detail::uintptr_t>::is_signed,  false);
}
DECLARE_UNITTEST(TestStandardIntegerTypes);

