#include <unittest/unittest.h>
#include <thrust/detail/mpl/math.h>

void TestLog2(void)
{
    unsigned int result;
    
    result = thrust::detail::mpl::math::log2<  1>::value;   ASSERT_EQUAL(result, 0lu);
    result = thrust::detail::mpl::math::log2<  2>::value;   ASSERT_EQUAL(result, 1lu);
    result = thrust::detail::mpl::math::log2<  3>::value;   ASSERT_EQUAL(result, 1lu);
    result = thrust::detail::mpl::math::log2<  4>::value;   ASSERT_EQUAL(result, 2lu);
    result = thrust::detail::mpl::math::log2<  5>::value;   ASSERT_EQUAL(result, 2lu);
    result = thrust::detail::mpl::math::log2<  6>::value;   ASSERT_EQUAL(result, 2lu);
    result = thrust::detail::mpl::math::log2<  7>::value;   ASSERT_EQUAL(result, 2lu);
    result = thrust::detail::mpl::math::log2<  8>::value;   ASSERT_EQUAL(result, 3lu);
    result = thrust::detail::mpl::math::log2<  9>::value;   ASSERT_EQUAL(result, 3lu);
    result = thrust::detail::mpl::math::log2< 15>::value;   ASSERT_EQUAL(result, 3lu);
    result = thrust::detail::mpl::math::log2< 16>::value;   ASSERT_EQUAL(result, 4lu);
    result = thrust::detail::mpl::math::log2< 17>::value;   ASSERT_EQUAL(result, 4lu);
    result = thrust::detail::mpl::math::log2<127>::value;   ASSERT_EQUAL(result, 6lu);
    result = thrust::detail::mpl::math::log2<128>::value;   ASSERT_EQUAL(result, 7lu);
    result = thrust::detail::mpl::math::log2<129>::value;   ASSERT_EQUAL(result, 7lu);
    result = thrust::detail::mpl::math::log2<256>::value;   ASSERT_EQUAL(result, 8lu);
    result = thrust::detail::mpl::math::log2<511>::value;   ASSERT_EQUAL(result, 8lu);
    result = thrust::detail::mpl::math::log2<512>::value;   ASSERT_EQUAL(result, 9lu);
}
DECLARE_UNITTEST(TestLog2);

