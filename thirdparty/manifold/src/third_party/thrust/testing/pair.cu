#include <unittest/unittest.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/swap.h>

template <typename T>
struct TestPairManipulation
{
  void operator()(void)
  {
    typedef thrust::pair<T,T> P;

    // test null constructor
    P p1;
    ASSERT_EQUAL(T(0), p1.first);
    ASSERT_EQUAL(T(0), p1.second);

    // test individual value manipulation
    p1.first  = T(1);
    p1.second = T(2);
    ASSERT_EQUAL(T(1), p1.first);
    ASSERT_EQUAL(T(2), p1.second);

    // test copy constructor
    P p2(p1);
    ASSERT_EQUAL(p1.first,  p2.first);
    ASSERT_EQUAL(p1.second, p2.second);

    // test copy from std::pair constructor
    std::pair<T,T> sp(p1.first, p1.second);
    ASSERT_EQUAL(p1.first,  sp.first);
    ASSERT_EQUAL(p1.second, sp.second);

    // test initialization
    P p3 = p2;
    ASSERT_EQUAL(p2.first,  p3.first);
    ASSERT_EQUAL(p2.second, p3.second);

    // test initialization from std::pair
    P p4 = sp;
    ASSERT_EQUAL(sp.first,  p4.first);
    ASSERT_EQUAL(sp.second, p4.second);

    // test copy from pair
    p4.first  = T(2);
    p4.second = T(3);
    
    P p5;
    p5 = p4;
    ASSERT_EQUAL(p4.first,  p5.first);
    ASSERT_EQUAL(p4.second, p5.second);

    // test copy from std::pair
    sp.first  = T(4);
    sp.second = T(5);

    P p6;
    p6 = sp;
    ASSERT_EQUAL(sp.first,  p6.first);
    ASSERT_EQUAL(sp.second, p6.second);

    // test initialization from make_pair
    P p7 = thrust::make_pair(T(6),T(7));
    ASSERT_EQUAL(T(6), p7.first);
    ASSERT_EQUAL(T(7), p7.second);

    // test copy from make_pair
    p7 = thrust::make_pair(T(8),T(9));
    ASSERT_EQUAL(T(8), p7.first);
    ASSERT_EQUAL(T(9), p7.second);
  }
};
SimpleUnitTest<TestPairManipulation, NumericTypes> TestPairManipulationInstance;


template <typename T>
struct TestPairComparison
{
  void operator()(void)
  {
    typedef thrust::pair<T,T> P;

    P x, y;

    // test operator ==
    x.first = x.second = y.first = y.second = T(0);
    ASSERT_EQUAL(true, x == y);
    ASSERT_EQUAL(true, y == x);

    x.first = y.first = y.second = T(0);
    x.second = T(1);
    ASSERT_EQUAL(false, x == y);
    ASSERT_EQUAL(false, y == x);

    // test operator<
    x.first  = T(0); x.second = T(0);
    y.first  = T(0); y.second = T(0);
    ASSERT_EQUAL(false, x < y);
    ASSERT_EQUAL(false, y < x);

    x.first  = T(0); x.second = T(1);
    y.first  = T(2); y.second = T(3);
    ASSERT_EQUAL(true,  x < y);
    ASSERT_EQUAL(false, y < x);

    x.first  = T(0); x.second = T(0);
    y.first  = T(0); y.second = T(1);
    ASSERT_EQUAL(true,  x < y);
    ASSERT_EQUAL(false, y < x);

    x.first  = T(0); x.second = T(1);
    y.first  = T(0); y.second = T(2);
    ASSERT_EQUAL(true,  x < y);
    ASSERT_EQUAL(false, y < x);

    // test operator!=
    x.first = y.first = y.second = T(0);
    x.second = T(1);
    ASSERT_EQUAL(true, x != y);
    ASSERT_EQUAL(true, y != x);

    x.first = x.second = y.first = y.second = T(0);
    ASSERT_EQUAL(false, x != y);
    ASSERT_EQUAL(false, y != x);

    // test operator>
    x.first  = T(0); x.second = T(0);
    y.first  = T(0); y.second = T(0);
    ASSERT_EQUAL(false, x > y);
    ASSERT_EQUAL(false, y > x);

    x.first  = T(2); x.second = T(3);
    y.first  = T(0); y.second = T(1);
    ASSERT_EQUAL(true,  x > y);
    ASSERT_EQUAL(false, y > x);

    x.first  = T(0); x.second = T(1);
    y.first  = T(0); y.second = T(0);
    ASSERT_EQUAL(true,  x > y);
    ASSERT_EQUAL(false, y > x);

    x.first  = T(0); x.second = T(2);
    y.first  = T(0); y.second = T(1);
    ASSERT_EQUAL(true,  x > y);
    ASSERT_EQUAL(false, y > x);


    // test operator <=
    x.first = x.second = y.first = y.second = T(0);
    ASSERT_EQUAL(true, x <= y);
    ASSERT_EQUAL(true, y <= x);

    x.first = y.first = y.second = T(0);
    x.second = T(1);
    ASSERT_EQUAL(false, x <= y);

    x.first  = T(0); x.second = T(1);
    y.first  = T(2); y.second = T(3);
    ASSERT_EQUAL(true,  x <= y);
    ASSERT_EQUAL(false, y <= x);

    x.first  = T(0); x.second = T(0);
    y.first  = T(0); y.second = T(1);
    ASSERT_EQUAL(true,  x <= y);
    ASSERT_EQUAL(false, y <= x);

    x.first  = T(0); x.second = T(1);
    y.first  = T(0); y.second = T(2);
    ASSERT_EQUAL(true,  x <= y);
    ASSERT_EQUAL(false, y <= x);


    // test operator >=
    x.first = x.second = y.first = y.second = T(0);
    ASSERT_EQUAL(true, x >= y);
    ASSERT_EQUAL(true, y >= x);

    x.first = x.second = y.first = T(0);
    y.second = T(1);
    ASSERT_EQUAL(false, x >= y);

    x.first  = T(2); x.second = T(3);
    y.first  = T(0); y.second = T(1);
    ASSERT_EQUAL(true,  x >= y);
    ASSERT_EQUAL(false, y >= x);

    x.first  = T(0); x.second = T(1);
    y.first  = T(0); y.second = T(0);
    ASSERT_EQUAL(true,  x >= y);
    ASSERT_EQUAL(false, y >= x);

    x.first  = T(0); x.second = T(2);
    y.first  = T(0); y.second = T(1);
    ASSERT_EQUAL(true,  x >= y);
    ASSERT_EQUAL(false, y >= x);
  }
};
SimpleUnitTest<TestPairComparison, NumericTypes> TestPairComparisonInstance;


template<typename T>
struct TestPairGet
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_integers<T>(2);

    thrust::pair<T,T> p(data[0], data[1]);

    ASSERT_EQUAL(data[0], thrust::get<0>(p));
    ASSERT_EQUAL(data[1], thrust::get<1>(p));
  }
};
SimpleUnitTest<TestPairGet, BuiltinNumericTypes> TestPairGetInstance;

using PairConstVolatileTypes =
    unittest::type_list<thrust::pair<int, float>, thrust::pair<int, float> const,
                        thrust::pair<int, float> const volatile>;

template <typename Pair> 
struct TestPairTupleSize
{
  void operator()()
  {
    ASSERT_EQUAL(2, static_cast<int>(thrust::tuple_size<Pair>::value));
  }
};
SimpleUnitTest<TestPairTupleSize, PairConstVolatileTypes> TestPairTupleSizeInstance;


void TestPairTupleElement(void)
{
  using type0 = thrust::tuple_element<0, thrust::pair<int, float> >::type;
  using type1 = thrust::tuple_element<1, thrust::pair<int, float> >::type;
  static_assert(std::is_same<int, type0>::value,"");
  static_assert(std::is_same<float, type1>::value,"");

  using c_type0 = thrust::tuple_element<0, thrust::pair<int, float> const>::type;
  using c_type1 = thrust::tuple_element<1, thrust::pair<int, float> const>::type;
  static_assert(std::is_same<int const, c_type0>::value,"");
  static_assert(std::is_same<float const, c_type1>::value,"");

  using v_type0 = thrust::tuple_element<0, thrust::pair<int, float> volatile>::type;
  using v_type1 = thrust::tuple_element<1, thrust::pair<int, float> volatile>::type;
  static_assert(std::is_same<int volatile, v_type0>::value,"");
  static_assert(std::is_same<float volatile, v_type1>::value,"");

  using cv_type0 = thrust::tuple_element<0, thrust::pair<int, float> const volatile>::type;
  using cv_type1 = thrust::tuple_element<1, thrust::pair<int, float> const volatile>::type;
  static_assert(std::is_same<int const volatile, cv_type0>::value,"");
  static_assert(std::is_same<float const volatile, cv_type1>::value,"");
};
DECLARE_UNITTEST(TestPairTupleElement);


void TestPairSwap(void)
{
  int x = 7;
  int y = 13;

  int z = 42;
  int w = 0;

  thrust::pair<int,int> a(x,y);
  thrust::pair<int,int> b(z,w);

  thrust::swap(a,b);

  ASSERT_EQUAL(z, a.first);
  ASSERT_EQUAL(w, a.second);
  ASSERT_EQUAL(x, b.first);
  ASSERT_EQUAL(y, b.second);


  typedef thrust::pair<user_swappable,user_swappable> swappable_pair;

  thrust::host_vector<swappable_pair>   h_v1(1), h_v2(1);
  thrust::device_vector<swappable_pair> d_v1(1), d_v2(1);

  thrust::swap_ranges(h_v1.begin(), h_v1.end(), h_v2.begin());
  thrust::swap_ranges(d_v1.begin(), d_v1.end(), d_v2.begin());

  swappable_pair ref(user_swappable(true), user_swappable(true));

  ASSERT_EQUAL_QUIET(ref, h_v1[0]);
  ASSERT_EQUAL_QUIET(ref, h_v1[0]);
  ASSERT_EQUAL_QUIET(ref, (swappable_pair)d_v1[0]);
  ASSERT_EQUAL_QUIET(ref, (swappable_pair)d_v1[0]);
}
DECLARE_UNITTEST(TestPairSwap);

