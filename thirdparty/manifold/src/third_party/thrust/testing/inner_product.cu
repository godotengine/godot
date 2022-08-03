#include <unittest/unittest.h>
#include <thrust/inner_product.h>

#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>

template <class Vector>
void TestInnerProductSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v1(3);
    Vector v2(3);
    v1[0] =  1; v1[1] = -2; v1[2] =  3;
    v2[0] = -4; v2[1] =  5; v2[2] =  6;

    T init = 3;
    T result = thrust::inner_product(v1.begin(), v1.end(), v2.begin(), init);
    ASSERT_EQUAL(result, 7);
}
DECLARE_VECTOR_UNITTEST(TestInnerProductSimple);


template <typename InputIterator1, typename InputIterator2, typename OutputType>
int inner_product(my_system &system, InputIterator1, InputIterator1, InputIterator2, OutputType)
{
    system.validate_dispatch();
    return 13;
}

void TestInnerProductDispatchExplicit()
{
    thrust::device_vector<int> vec;

    my_system sys(0);
    thrust::inner_product(sys,
                          vec.begin(),
                          vec.end(),
                          vec.begin(),
                          0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestInnerProductDispatchExplicit);


template <typename InputIterator1, typename InputIterator2, typename OutputType>
int inner_product(my_tag, InputIterator1, InputIterator1, InputIterator2, OutputType)
{
    return 13;
}

void TestInnerProductDispatchImplicit()
{
    thrust::device_vector<int> vec;

    int result = thrust::inner_product(thrust::retag<my_tag>(vec.begin()),
                                       thrust::retag<my_tag>(vec.end()),
                                       thrust::retag<my_tag>(vec.begin()),
                                       0);

    ASSERT_EQUAL(13, result);
}
DECLARE_UNITTEST(TestInnerProductDispatchImplicit);

template <class Vector>
void TestInnerProductWithOperator(void)
{
    typedef typename Vector::value_type T;

    Vector v1(3);
    Vector v2(3);
    v1[0] =  1; v1[1] = -2; v1[2] =  3;
    v2[0] = -1; v2[1] =  3; v2[2] =  6;

    // compute (v1 - v2) and perform a multiplies reduction
    T init = 3;
    T result = thrust::inner_product(v1.begin(), v1.end(), v2.begin(), init, 
                                      thrust::multiplies<T>(), thrust::minus<T>());
    ASSERT_EQUAL(result, 90);
}
DECLARE_VECTOR_UNITTEST(TestInnerProductWithOperator);

template <typename T>
struct TestInnerProduct
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T> h_v1 = unittest::random_integers<T>(n);
        thrust::host_vector<T> h_v2 = unittest::random_integers<T>(n);

        thrust::device_vector<T> d_v1 = h_v1;
        thrust::device_vector<T> d_v2 = h_v2;

        T init = 13;

        T expected = thrust::inner_product(h_v1.begin(), h_v1.end(), h_v2.begin(), init);
        T result   = thrust::inner_product(d_v1.begin(), d_v1.end(), d_v2.begin(), init);

        ASSERT_EQUAL(expected, result);
    }
};
VariableUnitTest<TestInnerProduct, IntegralTypes> TestInnerProductInstance;

struct only_set_when_both_expected
{
    long long expected;
    bool * flag;

    __device__
    long long operator()(long long x, long long y)
    {
        if (x == expected && y == expected)
        {
            *flag = true;
        }

        return x == y;
    }
};

void TestInnerProductWithBigIndexesHelper(int magnitude)
{
    thrust::counting_iterator<long long> begin(1);
    thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
    ASSERT_EQUAL(thrust::distance(begin, end), 1ll << magnitude);

    thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
    *has_executed = false;

    only_set_when_both_expected fn = { (1ll << magnitude) - 1,
        thrust::raw_pointer_cast(has_executed) };

    ASSERT_EQUAL(thrust::inner_product(
        thrust::device,
        begin, end,
        begin,
        0ll,
        thrust::plus<long long>(),
        fn), (1ll << magnitude));

    bool has_executed_h = *has_executed;
    thrust::device_free(has_executed);

    ASSERT_EQUAL(has_executed_h, true);
}

void TestInnerProductWithBigIndexes()
{
    TestInnerProductWithBigIndexesHelper(30);
    TestInnerProductWithBigIndexesHelper(31);
    TestInnerProductWithBigIndexesHelper(32);
    TestInnerProductWithBigIndexesHelper(33);
}
DECLARE_UNITTEST(TestInnerProductWithBigIndexes);

void TestInnerProductPlaceholders()
{ // Regression test for NVIDIA/thrust#1178
  using namespace thrust::placeholders;

  thrust::device_vector<float> v1(100, 1.f);
  thrust::device_vector<float> v2(100, 1.f);

  auto result = thrust::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0f,
                                      thrust::plus<float>{},
                                      _1 * _2 + 1.0f);

  ASSERT_ALMOST_EQUAL(result, 200.f);
}
DECLARE_UNITTEST(TestInnerProductPlaceholders);
