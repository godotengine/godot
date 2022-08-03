#include <unittest/unittest.h>
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_union(my_system &system,
                         InputIterator1,
                         InputIterator1,
                         InputIterator2,
                         InputIterator2,
                         OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

void TestSetUnionDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::set_union(sys,
                    vec.begin(),
                    vec.begin(),
                    vec.begin(),
                    vec.begin(),
                    vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSetUnionDispatchExplicit);


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
OutputIterator set_union(my_tag,
                         InputIterator1,
                         InputIterator1,
                         InputIterator2,
                         InputIterator2,
                         OutputIterator result)
{
  *result = 13;
  return result;
}

void TestSetUnionDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::set_union(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSetUnionDispatchImplicit);


template<typename Vector>
void TestSetUnionSimple(void)
{
  typedef typename Vector::iterator Iterator;

  Vector a(3), b(4);

  a[0] = 0; a[1] = 2; a[2] = 4;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4;

  Vector ref(5);
  ref[0] = 0; ref[1] = 2; ref[2] = 3; ref[3] = 3; ref[4] = 4;

  Vector result(5);

  Iterator end = thrust::set_union(a.begin(), a.end(),
                                   b.begin(), b.end(),
                                   result.begin());

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}
DECLARE_VECTOR_UNITTEST(TestSetUnionSimple);


template<typename Vector>
void TestSetUnionWithEquivalentElementsSimple(void)
{
  typedef typename Vector::iterator Iterator;

  Vector a(3), b(5);

  a[0] = 0; a[1] = 2; a[2] = 2;
  b[0] = 0; b[1] = 2; b[2] = 2; b[3] = 2; b[4] = 3;

  Vector ref(5);
  ref[0] = 0; ref[1] = 2; ref[2] = 2; ref[3] = 2; ref[4] = 3;

  Vector result(5);

  Iterator end = thrust::set_union(a.begin(), a.end(),
                                   b.begin(), b.end(),
                                   result.begin());

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}
DECLARE_VECTOR_UNITTEST(TestSetUnionWithEquivalentElementsSimple);


template<typename T>
void TestSetUnion(const size_t n)
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
    
    h_end = thrust::set_union(h_a.begin(), h_a.end(),
                              h_b.begin(), h_b.begin() + size,
                              h_result.begin());
    h_result.resize(h_end - h_result.begin());

    d_end = thrust::set_union(d_a.begin(), d_a.end(),
                              d_b.begin(), d_b.begin() + size,
                              d_result.begin());
    d_result.resize(d_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }
}
DECLARE_VARIABLE_UNITTEST(TestSetUnion);

template<typename T>
void TestSetUnionToDiscardIterator(const size_t n)
{
  thrust::host_vector<T> temp = unittest::random_integers<T>(2 * n);
  thrust::host_vector<T> h_a(temp.begin(), temp.begin() + n);
  thrust::host_vector<T> h_b(temp.begin() + n, temp.end());

  thrust::sort(h_a.begin(), h_a.end());
  thrust::sort(h_b.begin(), h_b.end());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::discard_iterator<> h_result;
  thrust::discard_iterator<> d_result;

  thrust::host_vector<T> h_reference(2 * n);
  typename thrust::host_vector<T>::iterator h_end = 
    thrust::set_union(h_a.begin(), h_a.end(),
                      h_b.begin(), h_b.end(),
                      h_reference.begin());
  h_reference.erase(h_end, h_reference.end());
  
  h_result = thrust::set_union(h_a.begin(), h_a.end(),
                               h_b.begin(), h_b.end(),
                               thrust::make_discard_iterator());

  d_result = thrust::set_union(d_a.begin(), d_a.end(),
                               d_b.begin(), d_b.end(),
                               thrust::make_discard_iterator());

  thrust::discard_iterator<> reference(h_reference.size());

  ASSERT_EQUAL_QUIET(reference, h_result);
  ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestSetUnionToDiscardIterator);

