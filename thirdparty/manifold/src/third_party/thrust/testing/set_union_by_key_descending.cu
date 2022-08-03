#include <unittest/unittest.h>
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

template<typename Vector>
void TestSetUnionByKeyDescendingSimple(void)
{
  typedef typename Vector::value_type T;
  typedef typename Vector::iterator Iterator;

  Vector a_key(3), b_key(4);
  Vector a_val(3), b_val(4);

  a_key[0] = 4; a_key[1] = 2; a_key[2] = 0;
  a_val[0] = 0; a_val[1] = 0; a_val[2] = 0;

  b_key[0] = 4; b_key[1] = 3; b_key[2] = 3; b_key[3] = 0;
  b_val[0] = 1; b_val[1] = 1; b_val[2] = 1; b_val[3] = 1;

  Vector ref_key(5), ref_val(5);
  ref_key[0] = 4; ref_key[1] = 3; ref_key[2] = 3; ref_key[3] = 2; ref_key[4] = 0;
  ref_val[0] = 0; ref_val[1] = 1; ref_val[2] = 1; ref_val[3] = 0; ref_val[4] = 0;

  Vector result_key(5), result_val(5);

  thrust::pair<Iterator,Iterator> end =
    thrust::set_union_by_key(a_key.begin(), a_key.end(),
                             b_key.begin(), b_key.end(),
                             a_val.begin(),
                             b_val.begin(),
                             result_key.begin(),
                             result_val.begin(),
                             thrust::greater<T>());

  ASSERT_EQUAL_QUIET(result_key.end(), end.first);
  ASSERT_EQUAL_QUIET(result_val.end(), end.second);
  ASSERT_EQUAL(ref_key, result_key);
  ASSERT_EQUAL(ref_val, result_val);
}
DECLARE_VECTOR_UNITTEST(TestSetUnionByKeyDescendingSimple);


template<typename T>
void TestSetUnionByKeyDescending(const size_t n)
{
  thrust::host_vector<T> temp = unittest::random_integers<T>(2 * n);
  thrust::host_vector<T> h_a_key(temp.begin(), temp.begin() + n);
  thrust::host_vector<T> h_b_key(temp.begin() + n, temp.end());

  thrust::sort(h_a_key.begin(), h_a_key.end(), thrust::greater<T>());
  thrust::sort(h_b_key.begin(), h_b_key.end(), thrust::greater<T>());

  thrust::host_vector<T> h_a_val = unittest::random_integers<T>(h_a_key.size());
  thrust::host_vector<T> h_b_val = unittest::random_integers<T>(h_b_key.size());

  thrust::device_vector<T> d_a_key = h_a_key;
  thrust::device_vector<T> d_b_key = h_b_key;

  thrust::device_vector<T> d_a_val = h_a_val;
  thrust::device_vector<T> d_b_val = h_b_val;

  size_t max_size = h_a_key.size() + h_b_key.size();
  thrust::host_vector<T>   h_result_key(max_size), h_result_val(max_size);
  thrust::device_vector<T> d_result_key(max_size), d_result_val(max_size);

  thrust::pair<
    typename thrust::host_vector<T>::iterator,
    typename thrust::host_vector<T>::iterator
  > h_end;

  thrust::pair<
    typename thrust::device_vector<T>::iterator,
    typename thrust::device_vector<T>::iterator
  > d_end;
  
  h_end = thrust::set_union_by_key(h_a_key.begin(), h_a_key.end(),
                                   h_b_key.begin(), h_b_key.end(),
                                   h_a_val.begin(),
                                   h_b_val.begin(),
                                   h_result_key.begin(),
                                   h_result_val.begin(),
                                   thrust::greater<T>());
  h_result_key.erase(h_end.first,  h_result_key.end());
  h_result_val.erase(h_end.second, h_result_val.end());

  d_end = thrust::set_union_by_key(d_a_key.begin(), d_a_key.end(),
                                   d_b_key.begin(), d_b_key.end(),
                                   d_a_val.begin(),
                                   d_b_val.begin(),
                                   d_result_key.begin(),
                                   d_result_val.begin(),
                                   thrust::greater<T>());
  d_result_key.erase(d_end.first,  d_result_key.end());
  d_result_val.erase(d_end.second, d_result_val.end());

  ASSERT_EQUAL(h_result_key, d_result_key);
  ASSERT_EQUAL(h_result_val, d_result_val);
}
DECLARE_VARIABLE_UNITTEST(TestSetUnionByKeyDescending);

