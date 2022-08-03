#include <unittest/unittest.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/scan.h>

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
    // Need cast to undo integer promotion, decltype(char{} + char{}) == int
    using P1T1 = typename Pair1::first_type;
    using P1T2 = typename Pair1::second_type;
    return thrust::make_pair(static_cast<P1T1>(x.first + y.first),
                             static_cast<P1T2>(x.second + y.second));
  } // end operator()
}; // end add_pairs


template <typename T>
  struct TestPairScanByKey
{
  void operator()(const size_t n)
  {
    typedef thrust::pair<T,T> P;

    thrust::host_vector<T>   h_p1 = unittest::random_integers<T>(n);
    thrust::host_vector<T>   h_p2 = unittest::random_integers<T>(n);
    thrust::host_vector<P>   h_pairs(n);

    // zip up pairs on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_pairs.begin(), make_pair_functor());

    thrust::device_vector<T> d_p1 = h_p1;
    thrust::device_vector<T> d_p2 = h_p2;
    thrust::device_vector<P> d_pairs = h_pairs;

    thrust::host_vector<T>   h_keys = unittest::random_integers<bool>(n);
    thrust::device_vector<T> d_keys = h_keys;

    P init = thrust::make_pair(T{13}, T{13});

    // scan on the host
    thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_pairs.begin(), h_pairs.begin(), init, thrust::equal_to<T>(), add_pairs());

    // scan on the device
    thrust::exclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_pairs.begin(), d_pairs.begin(), init, thrust::equal_to<T>(), add_pairs());

    ASSERT_EQUAL_QUIET(h_pairs, d_pairs);
  }
};
VariableUnitTest<TestPairScanByKey, unittest::type_list<unittest::int8_t,unittest::int16_t,unittest::int32_t> > TestPairScanByKeyInstance;

