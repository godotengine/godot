#include <unittest/unittest.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

using namespace unittest;

struct MakeTupleFunctor
{
  template<typename T1, typename T2>
  __host__ __device__
  thrust::tuple<T1,T2> operator()(T1 &lhs, T2 &rhs)
  {
    return thrust::make_tuple(lhs, rhs);
  }
};

template<int N>
struct GetFunctor
{
  template<typename Tuple>
  __host__ __device__
  typename thrust::access_traits<
    typename thrust::tuple_element<N, Tuple>::type
  >::const_type
  operator()(const Tuple &t)
  {
    return thrust::get<N>(t);
  }
};

template <typename T>
struct TestTupleStableSort
{
  void operator()(const size_t n)
  {
     using namespace thrust;

     host_vector<T> h_keys   = random_integers<T>(n);
     host_vector<T> h_values = random_integers<T>(n);

     // zip up the data
     host_vector< tuple<T,T> > h_tuples(n);
     transform(h_keys.begin(),   h_keys.end(),
               h_values.begin(), h_tuples.begin(),
               MakeTupleFunctor());

     // copy to device
     device_vector< tuple<T,T> > d_tuples = h_tuples;

     // sort on host
     stable_sort(h_tuples.begin(), h_tuples.end());

     // sort on device
     stable_sort(d_tuples.begin(), d_tuples.end());

     ASSERT_EQUAL(true, is_sorted(d_tuples.begin(), d_tuples.end()));

     // select keys
     transform(h_tuples.begin(), h_tuples.end(), h_keys.begin(), GetFunctor<0>());

     device_vector<T> d_keys(h_keys.size());
     transform(d_tuples.begin(), d_tuples.end(), d_keys.begin(), GetFunctor<0>());

     // select values
     transform(h_tuples.begin(), h_tuples.end(), h_values.begin(), GetFunctor<1>());
     
     device_vector<T> d_values(h_values.size());
     transform(d_tuples.begin(), d_tuples.end(), d_values.begin(), GetFunctor<1>());

     ASSERT_ALMOST_EQUAL(h_keys, d_keys);
     ASSERT_ALMOST_EQUAL(h_values, d_values);
  }
};
VariableUnitTest<TestTupleStableSort, unittest::type_list<unittest::int8_t,unittest::int16_t,unittest::int32_t> > TestTupleStableSortInstance;

