#include <unittest/unittest.h>
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

template<typename Vector>
void TestSetSymmetricDifferenceDescendingSimple(void)
{
  typedef typename Vector::value_type T;
  typedef typename Vector::iterator Iterator;

  Vector a(4), b(5);

  a[0] = 6; a[1] = 4; a[2] = 2; a[3] = 0;
  b[0] = 7; b[1] = 4; b[2] = 3; b[3] = 3; b[4] = 0;

  Vector ref(5);
  ref[0] = 7; ref[1] = 6; ref[2] = 3; ref[3] = 3; ref[4] = 2;

  Vector result(5);

  Iterator end = thrust::set_symmetric_difference(a.begin(), a.end(),
                                                  b.begin(), b.end(),
                                                  result.begin(),
                                                  thrust::greater<T>());

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}
DECLARE_VECTOR_UNITTEST(TestSetSymmetricDifferenceDescendingSimple);


template<typename T>
void TestSetSymmetricDifferenceDescending(const size_t n)
{
  thrust::host_vector<T> temp = unittest::random_integers<T>(2 * n);
  thrust::host_vector<T> h_a(temp.begin(), temp.begin() + n);
  thrust::host_vector<T> h_b(temp.begin() + n, temp.end());

  thrust::sort(h_a.begin(), h_a.end(), thrust::greater<T>());
  thrust::sort(h_b.begin(), h_b.end(), thrust::greater<T>());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T> h_result(h_a.size() + h_b.size());
  thrust::device_vector<T> d_result(h_result.size());

  typename thrust::host_vector<T>::iterator h_end;
  typename thrust::device_vector<T>::iterator d_end;
  
  h_end = thrust::set_symmetric_difference(h_a.begin(), h_a.end(),
                                           h_b.begin(), h_b.end(),
                                           h_result.begin(),
                                           thrust::greater<T>());
  h_result.erase(h_end, h_result.end());

  d_end = thrust::set_symmetric_difference(d_a.begin(), d_a.end(),
                                           d_b.begin(), d_b.end(),
                                           d_result.begin(),
                                           thrust::greater<T>());

  d_result.erase(d_end, d_result.end());

  ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestSetSymmetricDifferenceDescending);

