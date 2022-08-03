#include <unittest/unittest.h>
#include <thrust/partition.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>

template<typename T>
struct is_even
{
  __host__ __device__
  bool operator()(T x) const { return ((int) x % 2) == 0; }
};

template<typename Vector>
void TestIsPartitionedSimple(void)
{
  typedef typename Vector::value_type T;

  Vector v(4);
  v[0] = 1; v[1] = 1; v[2] = 1; v[3] = 0;

  // empty partition
  ASSERT_EQUAL_QUIET(true, thrust::is_partitioned(v.begin(), v.begin(), thrust::identity<T>()));

  // one element true partition
  ASSERT_EQUAL_QUIET(true, thrust::is_partitioned(v.begin(), v.begin() + 1, thrust::identity<T>()));

  // just true partition
  ASSERT_EQUAL_QUIET(true, thrust::is_partitioned(v.begin(), v.begin() + 2, thrust::identity<T>()));

  // both true & false partitions
  ASSERT_EQUAL_QUIET(true, thrust::is_partitioned(v.begin(), v.end(), thrust::identity<T>()));

  // one element false partition
  ASSERT_EQUAL_QUIET(true, thrust::is_partitioned(v.begin() + 3, v.end(), thrust::identity<T>()));

  v[0] = 1; v[1] = 0; v[2] = 1; v[3] = 1;

  // not partitioned
  ASSERT_EQUAL_QUIET(false, thrust::is_partitioned(v.begin(), v.end(), thrust::identity<T>()));
}
DECLARE_VECTOR_UNITTEST(TestIsPartitionedSimple);

template <class Vector>
void TestIsPartitioned(void)
{
  typedef typename Vector::value_type T;

  const size_t n = (1 << 16) + 13;

  Vector v = unittest::random_integers<T>(n);

  v[0] = 1;
  v[1] = 0;

  ASSERT_EQUAL(false, thrust::is_partitioned(v.begin(), v.end(), is_even<T>()));

  thrust::partition(v.begin(), v.end(), is_even<T>());

  ASSERT_EQUAL(true, thrust::is_partitioned(v.begin(), v.end(), is_even<T>()));
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestIsPartitioned);


template<typename InputIterator, typename Predicate>
bool is_partitioned(my_system &system, InputIterator /*first*/, InputIterator, Predicate)
{
  system.validate_dispatch();
  return false;
}

void TestIsPartitionedDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::is_partitioned(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestIsPartitionedDispatchExplicit);


template<typename InputIterator, typename Predicate>
bool is_partitioned(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return false;
}

void TestIsPartitionedDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::is_partitioned(thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.end()),
                         0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestIsPartitionedDispatchImplicit);

