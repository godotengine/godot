#include <unittest/unittest.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <unittest/cuda/testframework.h>
#endif

using namespace unittest;

template<typename Tuple>
struct TuplePlus
{
  __host__ __device__
  Tuple operator()(Tuple x, Tuple y) const
  {
    using namespace thrust;
    return make_tuple(get<0>(x) + get<0>(y),
                      get<1>(x) + get<1>(y));
  }
}; // end TuplePlus


template <typename T>
struct TestZipIteratorReduceByKey
{
  void operator()(const size_t n)
  {
    using namespace thrust;

    host_vector<T> h_data0 = unittest::random_integers<bool>(n);
    host_vector<T> h_data1 = unittest::random_integers<T>(n);
    host_vector<T> h_data2 = unittest::random_integers<T>(n);

    device_vector<T> d_data0 = h_data0;
    device_vector<T> d_data1 = h_data1;
    device_vector<T> d_data2 = h_data2;

    typedef tuple<T,T> Tuple;

    // integer key, tuple value
    {
      host_vector<T> h_data3(n,0);
      host_vector<T> h_data4(n,0);
      host_vector<T> h_data5(n,0);
      device_vector<T> d_data3(n,0);
      device_vector<T> d_data4(n,0);
      device_vector<T> d_data5(n,0);

      // run on host
      reduce_by_key
          ( h_data0.begin(), h_data0.end(),
            make_zip_iterator(make_tuple(h_data1.begin(), h_data2.begin())),
            h_data3.begin(),
            make_zip_iterator(make_tuple(h_data4.begin(), h_data5.begin())),
            equal_to<T>(),
            TuplePlus<Tuple>());

      // run on device
      reduce_by_key
          ( d_data0.begin(), d_data0.end(),
            make_zip_iterator(make_tuple(d_data1.begin(), d_data2.begin())),
            d_data3.begin(),
            make_zip_iterator(make_tuple(d_data4.begin(), d_data5.begin())),
            equal_to<T>(),
            TuplePlus<Tuple>());

      ASSERT_EQUAL(h_data3, d_data3);
      ASSERT_EQUAL(h_data4, d_data4);
      ASSERT_EQUAL(h_data5, d_data5);
    }
    
    // The tests below get miscompiled on Tesla hw for 8b types

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    if(const CUDATestDriver *driver = dynamic_cast<const CUDATestDriver*>(&UnitTestDriver::s_driver()))
    {
      if(typeid(T) == typeid(unittest::uint8_t) && driver->current_device_architecture() < 200)
      {
        KNOWN_FAILURE;
      } // end if
    } // end if
#endif

    // tuple key, tuple value
    {
      host_vector<T> h_data3(n,0);
      host_vector<T> h_data4(n,0);
      host_vector<T> h_data5(n,0);
      host_vector<T> h_data6(n,0);
      device_vector<T> d_data3(n,0);
      device_vector<T> d_data4(n,0);
      device_vector<T> d_data5(n,0);
      device_vector<T> d_data6(n,0);

      // run on host
      reduce_by_key
          ( make_zip_iterator(make_tuple(h_data0.begin(), h_data0.begin())),
            make_zip_iterator(make_tuple(h_data0.end(),   h_data0.end())),
            make_zip_iterator(make_tuple(h_data1.begin(), h_data2.begin())),
            make_zip_iterator(make_tuple(h_data3.begin(), h_data4.begin())),
            make_zip_iterator(make_tuple(h_data5.begin(), h_data6.begin())),
            equal_to<Tuple>(),
            TuplePlus<Tuple>());

      // run on device
      reduce_by_key
          ( make_zip_iterator(make_tuple(d_data0.begin(), d_data0.begin())),
            make_zip_iterator(make_tuple(d_data0.end(),   d_data0.end())),
            make_zip_iterator(make_tuple(d_data1.begin(), d_data2.begin())),
            make_zip_iterator(make_tuple(d_data3.begin(), d_data4.begin())),
            make_zip_iterator(make_tuple(d_data5.begin(), d_data6.begin())),
            equal_to<Tuple>(),
            TuplePlus<Tuple>());

      ASSERT_EQUAL(h_data3, d_data3);
      ASSERT_EQUAL(h_data4, d_data4);
      ASSERT_EQUAL(h_data5, d_data5);
      ASSERT_EQUAL(h_data6, d_data6);
    }
  }
};
VariableUnitTest<TestZipIteratorReduceByKey, UnsignedIntegralTypes> TestZipIteratorReduceByKeyInstance;

