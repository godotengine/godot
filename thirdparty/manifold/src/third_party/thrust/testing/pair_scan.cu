#include <unittest/unittest.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/scan.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <unittest/cuda/testframework.h>
#endif

struct make_pair_functor
{
  template<typename T1, typename T2>
  __host__ __device__
    thrust::pair<T1,T2> operator()(const T1 &x, const T2 &y)
  {
    return thrust::make_pair(x,y);
  } // end operator()()
}; // end make_pair_functor


struct add_pairs
{
  template <typename Pair1, typename Pair2>
  __host__ __device__
    Pair1 operator()(const Pair1 &x, const Pair2 &y)
  {
    return thrust::make_pair(x.first + y.first, x.second + y.second);
  } // end operator()
}; // end add_pairs


template <typename T>
  struct TestPairScan
{
  void operator()(const size_t n)
  {
    typedef thrust::pair<T,T> P;

    thrust::host_vector<T>   h_p1 = unittest::random_integers<T>(n);
    thrust::host_vector<T>   h_p2 = unittest::random_integers<T>(n);
    thrust::host_vector<P>   h_pairs(n);
    thrust::host_vector<P>   h_output(n);

    // zip up pairs on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_pairs.begin(), make_pair_functor());

    thrust::device_vector<T> d_p1 = h_p1;
    thrust::device_vector<T> d_p2 = h_p2;
    thrust::device_vector<P> d_pairs = h_pairs;
    thrust::device_vector<P> d_output(n);

    P init = thrust::make_pair(13,13);

    // scan with plus
    thrust::inclusive_scan(h_pairs.begin(), h_pairs.end(), h_output.begin(), add_pairs());
    thrust::inclusive_scan(d_pairs.begin(), d_pairs.end(), d_output.begin(), add_pairs());
    ASSERT_EQUAL_QUIET(h_output, d_output);

    // scan with maximum (thrust issue #69)
    thrust::inclusive_scan(h_pairs.begin(), h_pairs.end(), h_output.begin(), thrust::maximum<P>());
    thrust::inclusive_scan(d_pairs.begin(), d_pairs.end(), d_output.begin(), thrust::maximum<P>());
    ASSERT_EQUAL_QUIET(h_output, d_output);

    // scan with plus
    thrust::exclusive_scan(h_pairs.begin(), h_pairs.end(), h_output.begin(), init, add_pairs());
    thrust::exclusive_scan(d_pairs.begin(), d_pairs.end(), d_output.begin(), init, add_pairs());
    ASSERT_EQUAL_QUIET(h_output, d_output);
    
    // scan with maximum (thrust issue #69)
    thrust::exclusive_scan(h_pairs.begin(), h_pairs.end(), h_output.begin(), init, thrust::maximum<P>());
    thrust::exclusive_scan(d_pairs.begin(), d_pairs.end(), d_output.begin(), init, thrust::maximum<P>());
    ASSERT_EQUAL_QUIET(h_output, d_output);
  }
};
VariableUnitTest<TestPairScan, unittest::type_list<unittest::int8_t,unittest::int16_t,unittest::int32_t> > TestPairScanInstance;

