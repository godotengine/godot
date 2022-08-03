#include <unittest/unittest.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

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
}; // end SumTuple


template <typename T>
struct TestZipIteratorReduce
{
  void operator()(const size_t n)
  {
    using namespace thrust;

    host_vector<T> h_data0 = unittest::random_samples<T>(n);
    host_vector<T> h_data1 = unittest::random_samples<T>(n);

    device_vector<T> d_data0 = h_data0;
    device_vector<T> d_data1 = h_data1;

    typedef tuple<T,T> Tuple;

    // run on host
    Tuple h_result = reduce( make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin())),
                             make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end())),
                             make_tuple<T,T>(0,0),
                             TuplePlus<Tuple>());

    // run on device
    Tuple d_result = reduce( make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin())),
                             make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end())),
                             make_tuple<T,T>(0,0),
                             TuplePlus<Tuple>());

    ASSERT_EQUAL(get<0>(h_result), get<0>(d_result));
    ASSERT_EQUAL(get<1>(h_result), get<1>(d_result));
  }
};
VariableUnitTest<TestZipIteratorReduce, IntegralTypes> TestZipIteratorReduceInstance;



