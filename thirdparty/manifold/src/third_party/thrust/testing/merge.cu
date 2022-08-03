#include <unittest/unittest.h>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>

template<typename Vector>
void TestMergeSimple(void)
{
  typedef typename Vector::iterator Iterator;

  Vector a(3), b(4);

  a[0] = 0; a[1] = 2; a[2] = 4;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4;

  Vector ref(7);
  ref[0] = 0;
  ref[1] = 0;
  ref[2] = 2;
  ref[3] = 3;
  ref[4] = 3;
  ref[5] = 4;
  ref[6] = 4;

  Vector result(7);

  Iterator end = thrust::merge(a.begin(), a.end(),
                               b.begin(), b.end(),
                               result.begin());

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}
DECLARE_VECTOR_UNITTEST(TestMergeSimple);


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator merge(my_system &system,
                     InputIterator1,
                     InputIterator1,
                     InputIterator2,
                     InputIterator2,
                     OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

void TestMergeDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::merge(sys,
                vec.begin(),
                vec.begin(),
                vec.begin(),
                vec.begin(),
                vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestMergeDispatchExplicit);


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator merge(my_tag,
                     InputIterator1,
                     InputIterator1,
                     InputIterator2,
                     InputIterator2,
                     OutputIterator result)
{
  *result = 13;
  return result;
}

void TestMergeDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::merge(thrust::retag<my_tag>(vec.begin()),
                thrust::retag<my_tag>(vec.begin()),
                thrust::retag<my_tag>(vec.begin()),
                thrust::retag<my_tag>(vec.begin()),
                thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestMergeDispatchImplicit);


template<typename T>
  void TestMerge(size_t n)
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
    
    h_end = thrust::merge(h_a.begin(), h_a.end(),
                          h_b.begin(), h_b.begin() + size,
                          h_result.begin());
    h_result.resize(h_end - h_result.begin());

    d_end = thrust::merge(d_a.begin(), d_a.end(),
                          d_b.begin(), d_b.begin() + size,
                          d_result.begin());
    d_result.resize(d_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }
}
DECLARE_VARIABLE_UNITTEST(TestMerge);


template<typename T>
  void TestMergeToDiscardIterator(size_t n)
{
  thrust::host_vector<T> h_a = unittest::random_integers<T>(n);
  thrust::host_vector<T> h_b = unittest::random_integers<T>(n);

  thrust::stable_sort(h_a.begin(), h_a.end());
  thrust::stable_sort(h_b.begin(), h_b.end());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::discard_iterator<> h_result = 
    thrust::merge(h_a.begin(), h_a.end(),
                  h_b.begin(), h_b.end(),
                  thrust::make_discard_iterator());

  thrust::discard_iterator<> d_result = 
    thrust::merge(d_a.begin(), d_a.end(),
                  d_b.begin(), d_b.end(),
                  thrust::make_discard_iterator());

  thrust::discard_iterator<> reference(2 * n);

  ASSERT_EQUAL_QUIET(reference, h_result);
  ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestMergeToDiscardIterator);


template<typename T>
  void TestMergeDescending(size_t n)
{
  thrust::host_vector<T> h_a = unittest::random_integers<T>(n);
  thrust::host_vector<T> h_b = unittest::random_integers<T>(n);

  thrust::stable_sort(h_a.begin(), h_a.end(), thrust::greater<T>());
  thrust::stable_sort(h_b.begin(), h_b.end(), thrust::greater<T>());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T> h_result(h_a.size() + h_b.size());
  thrust::device_vector<T> d_result(d_a.size() + d_b.size());

  typename thrust::host_vector<T>::iterator h_end;
  typename thrust::device_vector<T>::iterator d_end;
  
  h_end = thrust::merge(h_a.begin(), h_a.end(),
                        h_b.begin(), h_b.end(),
                        h_result.begin(),
                        thrust::greater<T>());

  d_end = thrust::merge(d_a.begin(), d_a.end(),
                        d_b.begin(), d_b.end(),
                        d_result.begin(),
                        thrust::greater<T>());

  ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestMergeDescending);

