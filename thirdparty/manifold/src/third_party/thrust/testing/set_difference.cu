#include <unittest/unittest.h>
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/iterator/retag.h>


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_difference(my_system &system,
                              InputIterator1,
                              InputIterator1,
                              InputIterator2,
                              InputIterator2,
                              OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

void TestSetDifferenceDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::set_difference(sys,
                         vec.begin(),
                         vec.begin(),
                         vec.begin(),
                         vec.begin(),
                         vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSetDifferenceDispatchExplicit);


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_difference(my_tag,
                              InputIterator1,
                              InputIterator1,
                              InputIterator2,
                              InputIterator2,
                              OutputIterator result)
{
  *result = 13;
  return result;
}

void TestSetDifferenceDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::set_difference(thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSetDifferenceDispatchImplicit);


template<typename Vector>
void TestSetDifferenceSimple(void)
{
  typedef typename Vector::iterator Iterator;

  Vector a(4), b(5);

  a[0] = 0; a[1] = 2; a[2] = 4; a[3] = 5;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4; b[4] = 6;

  Vector ref(2);
  ref[0] = 2; ref[1] = 5;

  Vector result(2);

  Iterator end = thrust::set_difference(a.begin(), a.end(),
                                        b.begin(), b.end(),
                                        result.begin());

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}
DECLARE_VECTOR_UNITTEST(TestSetDifferenceSimple);


template<typename T>
void TestSetDifference(const size_t n)
{
  size_t sizes[]   = {0, 1, n / 2, n, n + 1, 2 * n};
  size_t num_sizes = sizeof(sizes) / sizeof(size_t);

  thrust::host_vector<T> random = unittest::random_integers<unittest::int8_t>(n + *thrust::max_element(sizes, sizes + num_sizes));

  thrust::host_vector<T> h_a(random.begin(), random.begin() + n);
  thrust::host_vector<T> h_b(random.begin() + n, random.end());
  
  thrust::stable_sort(h_a.begin(), h_a.end());
  thrust::stable_sort(h_b.begin(), h_b.end());
  
  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  for (size_t i = 0; i < num_sizes; i++)
  {
    size_t size = sizes[i];
    
    thrust::host_vector<T>   h_result(n + size);
    thrust::device_vector<T> d_result(n + size);

    typename thrust::host_vector<T>::iterator   h_end;
    typename thrust::device_vector<T>::iterator d_end;
    
    h_end = thrust::set_difference(h_a.begin(), h_a.end(),
                                   h_b.begin(), h_b.begin() + size,
                                   h_result.begin());
    h_result.resize(h_end - h_result.begin());

    d_end = thrust::set_difference(d_a.begin(), d_a.end(),
                                   d_b.begin(), d_b.begin() + size,
                                   d_result.begin());
    d_result.resize(d_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }
}
DECLARE_VARIABLE_UNITTEST(TestSetDifference);


template<typename T>
void TestSetDifferenceEquivalentRanges(const size_t n)
{
  thrust::host_vector<T> temp = unittest::random_integers<T>(n);
  thrust::host_vector<T> h_a = temp; thrust::sort(h_a.begin(), h_a.end());
  thrust::host_vector<T> h_b = h_a;

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T>   h_result(n);
  thrust::device_vector<T> d_result(n);

  typename thrust::host_vector<T>::iterator   h_end;
  typename thrust::device_vector<T>::iterator d_end;
  
  h_end = thrust::set_difference(h_a.begin(), h_a.end(),
                                 h_b.begin(), h_b.end(),
                                 h_result.begin());
  h_result.resize(h_end - h_result.begin());

  d_end = thrust::set_difference(d_a.begin(), d_a.end(),
                                 d_b.begin(), d_b.end(),
                                 d_result.begin());

  d_result.resize(d_end - d_result.begin());

  ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestSetDifferenceEquivalentRanges);


template<typename T>
void TestSetDifferenceMultiset(const size_t n)
{
  thrust::host_vector<T> vec = unittest::random_integers<T>(2 * n);

  // restrict elements to [min,13)
  for(typename thrust::host_vector<T>::iterator i = vec.begin();
      i != vec.end();
      ++i)
  {
    int temp = static_cast<int>(*i);
    temp %= 13;
    *i = temp;
  }

  thrust::host_vector<T> h_a(vec.begin(), vec.begin() + n);
  thrust::host_vector<T> h_b(vec.begin() + n, vec.end());

  thrust::sort(h_a.begin(), h_a.end());
  thrust::sort(h_b.begin(), h_b.end());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);

  typename thrust::host_vector<T>::iterator h_end;
  typename thrust::device_vector<T>::iterator d_end;
  
  h_end = thrust::set_difference(h_a.begin(), h_a.end(),
                                 h_b.begin(), h_b.end(),
                                 h_result.begin());
  h_result.resize(h_end - h_result.begin());

  d_end = thrust::set_difference(d_a.begin(), d_a.end(),
                                 d_b.begin(), d_b.end(),
                                 d_result.begin());

  d_result.resize(d_end - d_result.begin());

  ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestSetDifferenceMultiset);

// FIXME: disabled on Windows, because it causes a failure on the internal CI system in one specific configuration.
// That failure will be tracked in a new NVBug, this is disabled to unblock submitting all the other changes.
#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
void TestSetDifferenceWithBigIndexesHelper(int magnitude)
{
    thrust::counting_iterator<long long> begin(0);
    thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
    thrust::counting_iterator<long long> end_longer = end + 1;
    ASSERT_EQUAL(thrust::distance(begin, end), 1ll << magnitude);

    thrust::device_vector<long long> result;
    result.resize(1);
    thrust::set_difference(thrust::device, begin, end_longer, begin, end, result.begin());

    thrust::host_vector<long long> expected;
    expected.push_back(*end);

    ASSERT_EQUAL(result, expected);
}

void TestSetDifferenceWithBigIndexes()
{
    TestSetDifferenceWithBigIndexesHelper(30);
    TestSetDifferenceWithBigIndexesHelper(31);
    TestSetDifferenceWithBigIndexesHelper(32);
    TestSetDifferenceWithBigIndexesHelper(33);
}
DECLARE_UNITTEST(TestSetDifferenceWithBigIndexes);
#endif
