#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/iterator/retag.h>

template <class Vector>
void TestIsSortedSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(4);
    v[0] = 0; v[1] = 5; v[2] = 8; v[3] = 0;

    ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 0), true);
    ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 1), true);

    // the following line crashes gcc 4.3
#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 3)
    // do nothing
#else
    // compile this line on other compilers
    ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 2), true);
#endif // GCC

    ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 3), true);
    ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 4), false);

    ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 3, thrust::less<T>()),    true);

    ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 1, thrust::greater<T>()), true);
    ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 4, thrust::greater<T>()), false);

    ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.end()), false);
}
DECLARE_VECTOR_UNITTEST(TestIsSortedSimple);

template <class Vector>
void TestIsSortedRepeatedElements(void)
{
  Vector v(10);

  v[0] = 0;
  v[1] = 1;
  v[2] = 1;
  v[3] = 2;
  v[4] = 3;
  v[5] = 4;
  v[6] = 5;
  v[7] = 5;
  v[8] = 5;
  v[9] = 6;

  ASSERT_EQUAL(true, thrust::is_sorted(v.begin(), v.end()));
}
DECLARE_VECTOR_UNITTEST(TestIsSortedRepeatedElements);


template <class Vector>
void TestIsSorted(void)
{
    typedef typename Vector::value_type T;

    const size_t n = (1 << 16) + 13;

    Vector v = unittest::random_integers<T>(n);

    v[0] = 1;
    v[1] = 0;

    ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.end()), false);

    thrust::sort(v.begin(), v.end());

    ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.end()), true);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestIsSorted);


template<typename InputIterator>
bool is_sorted(my_system &system, InputIterator /*first*/, InputIterator)
{
  system.validate_dispatch();
  return false;
}

void TestIsSortedDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::is_sorted(sys,
                    vec.begin(),
                    vec.end());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestIsSortedDispatchExplicit);


template<typename InputIterator>
bool is_sorted(my_tag, InputIterator first, InputIterator)
{
  *first = 13;
  return false;
}

void TestIsSortedDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::is_sorted(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.end()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestIsSortedDispatchImplicit);

