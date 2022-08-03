#include <unittest/unittest.h>
#include <thrust/tuple.h>
#include <thrust/generate.h>
#include <thrust/swap.h>

using namespace unittest;

template <typename T>
struct TestTupleConstructor
{
  void operator()(void)
  {
    using namespace thrust;

    host_vector<T> data = random_integers<T>(10);

    tuple<T> t1(data[0]);
    ASSERT_EQUAL(data[0], get<0>(t1));

    tuple<T,T> t2(data[0], data[1]);
    ASSERT_EQUAL(data[0], get<0>(t2));
    ASSERT_EQUAL(data[1], get<1>(t2));

    tuple<T,T,T> t3(data[0], data[1], data[2]);
    ASSERT_EQUAL(data[0], get<0>(t3));
    ASSERT_EQUAL(data[1], get<1>(t3));
    ASSERT_EQUAL(data[2], get<2>(t3));

    tuple<T,T,T,T> t4(data[0], data[1], data[2], data[3]);
    ASSERT_EQUAL(data[0], get<0>(t4));
    ASSERT_EQUAL(data[1], get<1>(t4));
    ASSERT_EQUAL(data[2], get<2>(t4));
    ASSERT_EQUAL(data[3], get<3>(t4));

    tuple<T,T,T,T,T> t5(data[0], data[1], data[2], data[3], data[4]);
    ASSERT_EQUAL(data[0], get<0>(t5));
    ASSERT_EQUAL(data[1], get<1>(t5));
    ASSERT_EQUAL(data[2], get<2>(t5));
    ASSERT_EQUAL(data[3], get<3>(t5));
    ASSERT_EQUAL(data[4], get<4>(t5));

    tuple<T,T,T,T,T,T> t6(data[0], data[1], data[2], data[3], data[4], data[5]);
    ASSERT_EQUAL(data[0], get<0>(t6));
    ASSERT_EQUAL(data[1], get<1>(t6));
    ASSERT_EQUAL(data[2], get<2>(t6));
    ASSERT_EQUAL(data[3], get<3>(t6));
    ASSERT_EQUAL(data[4], get<4>(t6));
    ASSERT_EQUAL(data[5], get<5>(t6));

    tuple<T,T,T,T,T,T,T> t7(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
    ASSERT_EQUAL(data[0], get<0>(t7));
    ASSERT_EQUAL(data[1], get<1>(t7));
    ASSERT_EQUAL(data[2], get<2>(t7));
    ASSERT_EQUAL(data[3], get<3>(t7));
    ASSERT_EQUAL(data[4], get<4>(t7));
    ASSERT_EQUAL(data[5], get<5>(t7));
    ASSERT_EQUAL(data[6], get<6>(t7));

    tuple<T,T,T,T,T,T,T,T> t8(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    ASSERT_EQUAL(data[0], get<0>(t8));
    ASSERT_EQUAL(data[1], get<1>(t8));
    ASSERT_EQUAL(data[2], get<2>(t8));
    ASSERT_EQUAL(data[3], get<3>(t8));
    ASSERT_EQUAL(data[4], get<4>(t8));
    ASSERT_EQUAL(data[5], get<5>(t8));
    ASSERT_EQUAL(data[6], get<6>(t8));
    ASSERT_EQUAL(data[7], get<7>(t8));

    tuple<T,T,T,T,T,T,T,T,T> t9(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    ASSERT_EQUAL(data[0], get<0>(t9));
    ASSERT_EQUAL(data[1], get<1>(t9));
    ASSERT_EQUAL(data[2], get<2>(t9));
    ASSERT_EQUAL(data[3], get<3>(t9));
    ASSERT_EQUAL(data[4], get<4>(t9));
    ASSERT_EQUAL(data[5], get<5>(t9));
    ASSERT_EQUAL(data[6], get<6>(t9));
    ASSERT_EQUAL(data[7], get<7>(t9));
    ASSERT_EQUAL(data[8], get<8>(t9));

    tuple<T,T,T,T,T,T,T,T,T,T> t10(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
    ASSERT_EQUAL(data[0], get<0>(t10));
    ASSERT_EQUAL(data[1], get<1>(t10));
    ASSERT_EQUAL(data[2], get<2>(t10));
    ASSERT_EQUAL(data[3], get<3>(t10));
    ASSERT_EQUAL(data[4], get<4>(t10));
    ASSERT_EQUAL(data[5], get<5>(t10));
    ASSERT_EQUAL(data[6], get<6>(t10));
    ASSERT_EQUAL(data[7], get<7>(t10));
    ASSERT_EQUAL(data[8], get<8>(t10));
    ASSERT_EQUAL(data[9], get<9>(t10));
  }
};
SimpleUnitTest<TestTupleConstructor, BuiltinNumericTypes> TestTupleConstructorInstance;

template <typename T>
struct TestMakeTuple
{
  void operator()(void)
  {
    using namespace thrust;

    host_vector<T> data = random_integers<T>(10);

    tuple<T> t1 = make_tuple(data[0]);
    ASSERT_EQUAL(data[0], get<0>(t1));

    tuple<T,T> t2 = make_tuple(data[0], data[1]);
    ASSERT_EQUAL(data[0], get<0>(t2));
    ASSERT_EQUAL(data[1], get<1>(t2));

    tuple<T,T,T> t3 = make_tuple(data[0], data[1], data[2]);
    ASSERT_EQUAL(data[0], get<0>(t3));
    ASSERT_EQUAL(data[1], get<1>(t3));
    ASSERT_EQUAL(data[2], get<2>(t3));

    tuple<T,T,T,T> t4 = make_tuple(data[0], data[1], data[2], data[3]);
    ASSERT_EQUAL(data[0], get<0>(t4));
    ASSERT_EQUAL(data[1], get<1>(t4));
    ASSERT_EQUAL(data[2], get<2>(t4));
    ASSERT_EQUAL(data[3], get<3>(t4));

    tuple<T,T,T,T,T> t5 = make_tuple(data[0], data[1], data[2], data[3], data[4]);
    ASSERT_EQUAL(data[0], get<0>(t5));
    ASSERT_EQUAL(data[1], get<1>(t5));
    ASSERT_EQUAL(data[2], get<2>(t5));
    ASSERT_EQUAL(data[3], get<3>(t5));
    ASSERT_EQUAL(data[4], get<4>(t5));

    tuple<T,T,T,T,T,T> t6 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5]);
    ASSERT_EQUAL(data[0], get<0>(t6));
    ASSERT_EQUAL(data[1], get<1>(t6));
    ASSERT_EQUAL(data[2], get<2>(t6));
    ASSERT_EQUAL(data[3], get<3>(t6));
    ASSERT_EQUAL(data[4], get<4>(t6));
    ASSERT_EQUAL(data[5], get<5>(t6));

    tuple<T,T,T,T,T,T,T> t7 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
    ASSERT_EQUAL(data[0], get<0>(t7));
    ASSERT_EQUAL(data[1], get<1>(t7));
    ASSERT_EQUAL(data[2], get<2>(t7));
    ASSERT_EQUAL(data[3], get<3>(t7));
    ASSERT_EQUAL(data[4], get<4>(t7));
    ASSERT_EQUAL(data[5], get<5>(t7));
    ASSERT_EQUAL(data[6], get<6>(t7));

    tuple<T,T,T,T,T,T,T,T> t8 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    ASSERT_EQUAL(data[0], get<0>(t8));
    ASSERT_EQUAL(data[1], get<1>(t8));
    ASSERT_EQUAL(data[2], get<2>(t8));
    ASSERT_EQUAL(data[3], get<3>(t8));
    ASSERT_EQUAL(data[4], get<4>(t8));
    ASSERT_EQUAL(data[5], get<5>(t8));
    ASSERT_EQUAL(data[6], get<6>(t8));
    ASSERT_EQUAL(data[7], get<7>(t8));

    tuple<T,T,T,T,T,T,T,T,T> t9 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    ASSERT_EQUAL(data[0], get<0>(t9));
    ASSERT_EQUAL(data[1], get<1>(t9));
    ASSERT_EQUAL(data[2], get<2>(t9));
    ASSERT_EQUAL(data[3], get<3>(t9));
    ASSERT_EQUAL(data[4], get<4>(t9));
    ASSERT_EQUAL(data[5], get<5>(t9));
    ASSERT_EQUAL(data[6], get<6>(t9));
    ASSERT_EQUAL(data[7], get<7>(t9));
    ASSERT_EQUAL(data[8], get<8>(t9));

    tuple<T,T,T,T,T,T,T,T,T,T> t10 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
    ASSERT_EQUAL(data[0], get<0>(t10));
    ASSERT_EQUAL(data[1], get<1>(t10));
    ASSERT_EQUAL(data[2], get<2>(t10));
    ASSERT_EQUAL(data[3], get<3>(t10));
    ASSERT_EQUAL(data[4], get<4>(t10));
    ASSERT_EQUAL(data[5], get<5>(t10));
    ASSERT_EQUAL(data[6], get<6>(t10));
    ASSERT_EQUAL(data[7], get<7>(t10));
    ASSERT_EQUAL(data[8], get<8>(t10));
    ASSERT_EQUAL(data[9], get<9>(t10));
  }
};
SimpleUnitTest<TestMakeTuple, BuiltinNumericTypes> TestMakeTupleInstance;

template <typename T>
struct TestTupleGet
{
  void operator()(void)
  {
    using namespace thrust;
    host_vector<T> data = random_integers<T>(10);

    tuple<T> t1(data[0]);
    ASSERT_EQUAL(data[0], thrust::get<0>(t1));

    tuple<T,T> t2(data[0], data[1]);
    ASSERT_EQUAL(data[0], thrust::get<0>(t2));
    ASSERT_EQUAL(data[1], thrust::get<1>(t2));

    tuple<T,T,T> t3 = make_tuple(data[0], data[1], data[2]);
    ASSERT_EQUAL(data[0], thrust::get<0>(t3));
    ASSERT_EQUAL(data[1], thrust::get<1>(t3));
    ASSERT_EQUAL(data[2], thrust::get<2>(t3));

    tuple<T,T,T,T> t4 = make_tuple(data[0], data[1], data[2], data[3]);
    ASSERT_EQUAL(data[0], thrust::get<0>(t4));
    ASSERT_EQUAL(data[1], thrust::get<1>(t4));
    ASSERT_EQUAL(data[2], thrust::get<2>(t4));
    ASSERT_EQUAL(data[3], thrust::get<3>(t4));

    tuple<T,T,T,T,T> t5 = make_tuple(data[0], data[1], data[2], data[3], data[4]);
    ASSERT_EQUAL(data[0], thrust::get<0>(t5));
    ASSERT_EQUAL(data[1], thrust::get<1>(t5));
    ASSERT_EQUAL(data[2], thrust::get<2>(t5));
    ASSERT_EQUAL(data[3], thrust::get<3>(t5));
    ASSERT_EQUAL(data[4], thrust::get<4>(t5));

    tuple<T,T,T,T,T,T> t6 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5]);
    ASSERT_EQUAL(data[0], thrust::get<0>(t6));
    ASSERT_EQUAL(data[1], thrust::get<1>(t6));
    ASSERT_EQUAL(data[2], thrust::get<2>(t6));
    ASSERT_EQUAL(data[3], thrust::get<3>(t6));
    ASSERT_EQUAL(data[4], thrust::get<4>(t6));
    ASSERT_EQUAL(data[5], thrust::get<5>(t6));

    tuple<T,T,T,T,T,T,T> t7 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
    ASSERT_EQUAL(data[0], thrust::get<0>(t7));
    ASSERT_EQUAL(data[1], thrust::get<1>(t7));
    ASSERT_EQUAL(data[2], thrust::get<2>(t7));
    ASSERT_EQUAL(data[3], thrust::get<3>(t7));
    ASSERT_EQUAL(data[4], thrust::get<4>(t7));
    ASSERT_EQUAL(data[5], thrust::get<5>(t7));
    ASSERT_EQUAL(data[6], thrust::get<6>(t7));

    tuple<T,T,T,T,T,T,T,T> t8 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    ASSERT_EQUAL(data[0], thrust::get<0>(t8));
    ASSERT_EQUAL(data[1], thrust::get<1>(t8));
    ASSERT_EQUAL(data[2], thrust::get<2>(t8));
    ASSERT_EQUAL(data[3], thrust::get<3>(t8));
    ASSERT_EQUAL(data[4], thrust::get<4>(t8));
    ASSERT_EQUAL(data[5], thrust::get<5>(t8));
    ASSERT_EQUAL(data[6], thrust::get<6>(t8));
    ASSERT_EQUAL(data[7], thrust::get<7>(t8));

    tuple<T,T,T,T,T,T,T,T,T> t9 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    ASSERT_EQUAL(data[0], thrust::get<0>(t9));
    ASSERT_EQUAL(data[1], thrust::get<1>(t9));
    ASSERT_EQUAL(data[2], thrust::get<2>(t9));
    ASSERT_EQUAL(data[3], thrust::get<3>(t9));
    ASSERT_EQUAL(data[4], thrust::get<4>(t9));
    ASSERT_EQUAL(data[5], thrust::get<5>(t9));
    ASSERT_EQUAL(data[6], thrust::get<6>(t9));
    ASSERT_EQUAL(data[7], thrust::get<7>(t9));
    ASSERT_EQUAL(data[8], thrust::get<8>(t9));

    tuple<T,T,T,T,T,T,T,T,T,T> t10 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
    ASSERT_EQUAL(data[0], thrust::get<0>(t10));
    ASSERT_EQUAL(data[1], thrust::get<1>(t10));
    ASSERT_EQUAL(data[2], thrust::get<2>(t10));
    ASSERT_EQUAL(data[3], thrust::get<3>(t10));
    ASSERT_EQUAL(data[4], thrust::get<4>(t10));
    ASSERT_EQUAL(data[5], thrust::get<5>(t10));
    ASSERT_EQUAL(data[6], thrust::get<6>(t10));
    ASSERT_EQUAL(data[7], thrust::get<7>(t10));
    ASSERT_EQUAL(data[8], thrust::get<8>(t10));
    ASSERT_EQUAL(data[9], thrust::get<9>(t10));
  }
};
SimpleUnitTest<TestTupleGet, BuiltinNumericTypes> TestTupleGetInstance;



template <typename T>
struct TestTupleComparison
{
  void operator()(void)
  {
    using namespace thrust;

    tuple<T,T,T,T,T> lhs(0, 0, 0, 0, 0), rhs(0, 0, 0, 0, 0);

    // equality
    ASSERT_EQUAL(true,  lhs == rhs);
    get<0>(rhs) = 1;
    ASSERT_EQUAL(false,  lhs == rhs);

    // inequality
    ASSERT_EQUAL(true,  lhs != rhs);
    lhs = rhs;
    ASSERT_EQUAL(false, lhs != rhs);

    // less than
    lhs = make_tuple(0,0,0,0,0);
    rhs = make_tuple(0,0,1,0,0);
    ASSERT_EQUAL(true,  lhs < rhs);
    get<0>(lhs) = 2;
    ASSERT_EQUAL(false, lhs < rhs);

    // less than equal
    lhs = make_tuple(0,0,0,0,0);
    rhs = lhs;
    ASSERT_EQUAL(true,  lhs <= rhs); // equal
    get<2>(rhs) = 1;
    ASSERT_EQUAL(true,  lhs <= rhs); // less than
    get<2>(lhs) = 2;
    ASSERT_EQUAL(false, lhs <= rhs);

    // greater than
    lhs = make_tuple(1,0,0,0,0);
    rhs = make_tuple(0,1,1,1,1);
    ASSERT_EQUAL(true,  lhs > rhs);
    get<0>(rhs) = 2;
    ASSERT_EQUAL(false, lhs > rhs);

    // greater than equal
    lhs = make_tuple(0,0,0,0,0);
    rhs = lhs;
    ASSERT_EQUAL(true,  lhs >= rhs); // equal
    get<4>(lhs) = 1;
    ASSERT_EQUAL(true,  lhs >= rhs); // greater than
    get<3>(rhs) = 1;
    ASSERT_EQUAL(false, lhs >= rhs);
  }
};
SimpleUnitTest<TestTupleComparison, NumericTypes> TestTupleComparisonInstance;


template <typename T>
struct TestTupleTieFunctor
{
  __host__ __device__
  void clear(T *data) const
  {
    for(int i = 0; i < 10; ++i)
      data[i] = 13;
  }

  __host__ __device__
  bool operator()() const
  {
    using namespace thrust;

    bool result = true;

    T data[10];
    clear(data);

    // 17 and not 0 to avoid triggering custom_numeric's `operator void *` and a comparison with a null pointer
    // TODO: get this back from 17 to 0 once C++11 is on everywhere and that operator on custom_numeric is changed
    // to an explicit operator bool
    tie(data[0]) = make_tuple(17);
    result &= data[0] == 17;
    clear(data);

    tie(data[0], data[1]) = make_tuple(17,1);
    result &= data[0] == 17;
    result &= data[1] == 1;
    clear(data);

    tie(data[0], data[1], data[2]) = make_tuple(17,1,2);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    clear(data);

    tie(data[0], data[1], data[2], data[3]) = make_tuple(17,1,2,3);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4]) = make_tuple(17,1,2,3,4);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5]) = make_tuple(17,1,2,3,4,5);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6]) = make_tuple(17,1,2,3,4,5,6);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]) = make_tuple(17,1,2,3,4,5,6,7);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    result &= data[7] == 7;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]) = make_tuple(17,1,2,3,4,5,6,7,8);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    result &= data[7] == 7;
    result &= data[8] == 8;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]) = make_tuple(17,1,2,3,4,5,6,7,8,9);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    result &= data[7] == 7;
    result &= data[8] == 8;
    result &= data[9] == 9;
    clear(data);

    return result;
  }
};

template <typename T>
struct TestTupleTie
{
  void operator()(void)
  {
    thrust::host_vector<bool> h_result(1);
    thrust::generate(h_result.begin(), h_result.end(), TestTupleTieFunctor<T>());

    thrust::device_vector<bool> d_result(1);
    thrust::generate(d_result.begin(), d_result.end(), TestTupleTieFunctor<T>());

    ASSERT_EQUAL(true, h_result[0]);
    ASSERT_EQUAL(true, d_result[0]);
  }
};
SimpleUnitTest<TestTupleTie, NumericTypes> TestTupleTieInstance;

void TestTupleSwap(void)
{
  int a = 7;
  int b = 13;
  int c = 42;

  int x = 77;
  int y = 1313;
  int z = 4242;

  thrust::tuple<int,int,int> t1(a,b,c);
  thrust::tuple<int,int,int> t2(x,y,z);

  thrust::swap(t1,t2);

  ASSERT_EQUAL(x, thrust::get<0>(t1));
  ASSERT_EQUAL(y, thrust::get<1>(t1));
  ASSERT_EQUAL(z, thrust::get<2>(t1));
  ASSERT_EQUAL(a, thrust::get<0>(t2));
  ASSERT_EQUAL(b, thrust::get<1>(t2));
  ASSERT_EQUAL(c, thrust::get<2>(t2));


  typedef thrust::tuple<user_swappable,user_swappable,user_swappable,user_swappable> swappable_tuple;

  thrust::host_vector<swappable_tuple>   h_v1(1), h_v2(1);
  thrust::device_vector<swappable_tuple> d_v1(1), d_v2(1);

  thrust::swap_ranges(h_v1.begin(), h_v1.end(), h_v2.begin());
  thrust::swap_ranges(d_v1.begin(), d_v1.end(), d_v2.begin());

  swappable_tuple ref(user_swappable(true),user_swappable(true),user_swappable(true),user_swappable(true));

  ASSERT_EQUAL_QUIET(ref, h_v1[0]);
  ASSERT_EQUAL_QUIET(ref, h_v1[0]);
  ASSERT_EQUAL_QUIET(ref, (swappable_tuple)d_v1[0]);
  ASSERT_EQUAL_QUIET(ref, (swappable_tuple)d_v1[0]);
}
DECLARE_UNITTEST(TestTupleSwap);


