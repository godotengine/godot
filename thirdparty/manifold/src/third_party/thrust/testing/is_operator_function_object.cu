#include <unittest/unittest.h>
#include <thrust/detail/static_assert.h>
#include <thrust/type_traits/is_operator_less_or_greater_function_object.h>
#include <thrust/type_traits/is_operator_plus_function_object.h>

#if THRUST_CPP_DIALECT >= 2014
THRUST_STATIC_ASSERT((thrust::is_operator_less_function_object<
  std::less<>
>::value));

THRUST_STATIC_ASSERT((thrust::is_operator_greater_function_object<
  std::greater<>
>::value));

THRUST_STATIC_ASSERT((thrust::is_operator_less_or_greater_function_object<
  std::less<>
>::value));

THRUST_STATIC_ASSERT((thrust::is_operator_less_or_greater_function_object<
  std::greater<>
>::value));

THRUST_STATIC_ASSERT((thrust::is_operator_plus_function_object<
  std::plus<>
>::value));
#endif

template <typename T>
__host__
void test_is_operator_less_function_object()
{
  THRUST_STATIC_ASSERT((thrust::is_operator_less_function_object<
    thrust::less<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<
    thrust::greater<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<
    thrust::less_equal<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<
    thrust::greater_equal<T>
  >::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_less_function_object<
    std::less<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<
    std::greater<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<
    std::less_equal<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<
    std::greater_equal<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<
    T
  >::value));
}
DECLARE_GENERIC_UNITTEST(test_is_operator_less_function_object);

template <typename T>
__host__
void test_is_operator_greater_function_object()
{
  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<
    thrust::less<T>
  >::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_greater_function_object<
    thrust::greater<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<
    thrust::less_equal<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<
    thrust::greater_equal<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<
    std::less<T>
  >::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_greater_function_object<
    std::greater<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<
    std::less_equal<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<
    std::greater_equal<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<
    T
  >::value));
}
DECLARE_GENERIC_UNITTEST(test_is_operator_greater_function_object);

template <typename T>
__host__
void test_is_operator_less_or_greater_function_object()
{
  THRUST_STATIC_ASSERT((thrust::is_operator_less_or_greater_function_object<
    thrust::less<T>
  >::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_less_or_greater_function_object<
    thrust::greater<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_or_greater_function_object<
    thrust::less_equal<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_or_greater_function_object<
    thrust::greater_equal<T>
  >::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_less_or_greater_function_object<
    std::less<T>
  >::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_less_or_greater_function_object<
    std::greater<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_or_greater_function_object<
    std::less_equal<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_or_greater_function_object<
    std::greater_equal<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_or_greater_function_object<
    T
  >::value));
}
DECLARE_GENERIC_UNITTEST(test_is_operator_less_or_greater_function_object);

template <typename T>
__host__
void test_is_operator_plus_function_object()
{
  THRUST_STATIC_ASSERT((thrust::is_operator_plus_function_object<
    thrust::plus<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<
    thrust::minus<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<
    thrust::less<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<
    thrust::greater<T>
  >::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_plus_function_object<
    std::plus<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<
    std::minus<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<
    std::less<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<
    std::greater<T>
  >::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<
    T
  >::value));
}
DECLARE_GENERIC_UNITTEST(test_is_operator_plus_function_object);

