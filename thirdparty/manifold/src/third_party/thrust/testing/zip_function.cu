#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC)

#include <unittest/unittest.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>

#include <iostream>

using namespace unittest;

struct SumThree
{
  template <typename T1, typename T2, typename T3>
  __host__ __device__
  auto operator()(T1 x, T2 y, T3 z) const
  THRUST_DECLTYPE_RETURNS(x + y + z)
}; // end SumThree

struct SumThreeTuple
{
  template <typename Tuple>
  __host__ __device__
  auto operator()(Tuple x) const
  THRUST_DECLTYPE_RETURNS(thrust::get<0>(x) + thrust::get<1>(x) + thrust::get<2>(x))
}; // end SumThreeTuple

template <typename T>
struct TestZipFunctionTransform
{
  void operator()(const size_t n)
  {
    using namespace thrust;

    host_vector<T> h_data0 = unittest::random_samples<T>(n);
    host_vector<T> h_data1 = unittest::random_samples<T>(n);
    host_vector<T> h_data2 = unittest::random_samples<T>(n);

    device_vector<T> d_data0 = h_data0;
    device_vector<T> d_data1 = h_data1;
    device_vector<T> d_data2 = h_data2;

    host_vector<T>   h_result_tuple(n);
    host_vector<T>   h_result_zip(n);
    device_vector<T> d_result_zip(n);

    // Tuple base case
    transform(make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin(), h_data2.begin())),
              make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end(),   h_data2.end())),
              h_result_tuple.begin(),
              SumThreeTuple{});
    // Zip Function
    transform(make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin(), h_data2.begin())),
              make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end(),   h_data2.end())),
              h_result_zip.begin(),
              make_zip_function(SumThree{}));
    transform(make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin(), d_data2.begin())),
              make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end(),   d_data2.end())),
              d_result_zip.begin(),
              make_zip_function(SumThree{}));

    ASSERT_EQUAL(h_result_tuple, h_result_zip);
    ASSERT_EQUAL(h_result_tuple, d_result_zip);
  }
};
VariableUnitTest<TestZipFunctionTransform, ThirtyTwoBitTypes> TestZipFunctionTransformInstance;

#endif // THRUST_CPP_DIALECT
