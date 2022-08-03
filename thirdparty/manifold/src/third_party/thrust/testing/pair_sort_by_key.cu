#include <unittest/unittest.h>
#include <thrust/pair.h>
#include <thrust/sort.h>

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
  struct TestPairStableSort
{
  void operator()(const size_t n)
  {
    typedef thrust::pair<T,T> P;

    thrust::host_vector<T>   h_p1 = unittest::random_integers<T>(n);
    thrust::host_vector<T>   h_p2 = unittest::random_integers<T>(n);
    thrust::host_vector<P>   h_pairs(n);

    // zip up pairs on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_pairs.begin(), make_pair_functor());

    thrust::device_vector<P> d_pairs = h_pairs;

    // sort on the host
    thrust::stable_sort(h_pairs.begin(), h_pairs.end());

    // sort on the device
    thrust::stable_sort(d_pairs.begin(), d_pairs.end());

    ASSERT_EQUAL_QUIET(h_pairs, d_pairs);
  }
};
VariableUnitTest<TestPairStableSort, unittest::type_list<unittest::int8_t,unittest::int16_t,unittest::int32_t> > TestPairStableSortInstance;

