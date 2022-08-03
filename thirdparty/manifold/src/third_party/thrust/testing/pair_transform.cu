#include <unittest/unittest.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>

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
  struct TestPairTransform
{
  void operator()(const size_t n)
  {
    typedef thrust::pair<T,T> P;

    thrust::host_vector<T>   h_p1 = unittest::random_integers<T>(n);
    thrust::host_vector<T>   h_p2 = unittest::random_integers<T>(n);
    thrust::host_vector<P>   h_result(n);

    thrust::device_vector<T> d_p1 = h_p1;
    thrust::device_vector<T> d_p2 = h_p2;
    thrust::device_vector<P> d_result(n);

    // zip up pairs on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_result.begin(), make_pair_functor());

    // zip up pairs on the device
    thrust::transform(d_p1.begin(), d_p1.end(), d_p2.begin(), d_result.begin(), make_pair_functor());

    ASSERT_EQUAL_QUIET(h_result, d_result);
    
    // add pairs on the host
    thrust::transform(h_result.begin(), h_result.end(), h_result.begin(), h_result.begin(), add_pairs());
                                                                          
    // add pairs on the device                                            
    thrust::transform(d_result.begin(), d_result.end(), d_result.begin(), d_result.begin(), add_pairs());

    ASSERT_EQUAL_QUIET(h_result, d_result);
  }
}; // end TestPairZip
VariableUnitTest<TestPairTransform, unittest::type_list<unittest::int8_t,unittest::int16_t,unittest::int32_t> > TestPairTransformInstance;

