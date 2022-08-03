#include <unittest/unittest.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

struct make_pair_functor
{
  template<typename T1, typename T2>
  __host__ __device__
    thrust::pair<T1,T2> operator()(const T1 &x, const T2 &y)
  {
    return thrust::make_pair(x,y);
  } // end operator()()
}; // end make_pair_functor


template <typename T>
  struct TestPairStableSortByKey
{
  void operator()(const size_t n)
  {
    typedef thrust::pair<T,T> P;

    // host arrays
    thrust::host_vector<T>   h_p1 = unittest::random_integers<T>(n);
    thrust::host_vector<T>   h_p2 = unittest::random_integers<T>(n);
    thrust::host_vector<P>   h_pairs(n);

    thrust::host_vector<int> h_values(n);
    thrust::sequence(h_values.begin(), h_values.end());

    // zip up pairs on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_pairs.begin(), make_pair_functor());

    // device arrays
    thrust::device_vector<P>   d_pairs = h_pairs;
    thrust::device_vector<int> d_values = h_values;

    // sort on the host
    thrust::stable_sort_by_key(h_pairs.begin(), h_pairs.end(), h_values.begin());

    // sort on the device
    thrust::stable_sort_by_key(d_pairs.begin(), d_pairs.end(), d_values.begin());

    ASSERT_EQUAL_QUIET(h_pairs,  d_pairs);
    ASSERT_EQUAL(h_values, d_values);
  }
};
VariableUnitTest<TestPairStableSortByKey, unittest::type_list<unittest::int8_t,unittest::int16_t,unittest::int32_t> > TestPairStableSortByKeyInstance;

