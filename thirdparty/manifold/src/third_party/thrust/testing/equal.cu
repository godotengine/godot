#include <unittest/unittest.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

template <class Vector>
void TestEqualSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v1(5);
    Vector v2(5);
    v1[0] = 5; v1[1] = 2; v1[2] = 0; v1[3] = 0; v1[4] = 0;
    v2[0] = 5; v2[1] = 2; v2[2] = 0; v2[3] = 6; v2[4] = 1;

    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.end(), v1.begin()), true);
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.end(), v2.begin()), false);
    ASSERT_EQUAL(thrust::equal(v2.begin(), v2.end(), v2.begin()), true);
    
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.begin() + 0, v1.begin()), true);
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.begin() + 1, v1.begin()), true);
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.begin() + 3, v2.begin()), true);
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.begin() + 4, v2.begin()), false);
    
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.end(), v2.begin(), thrust::less_equal<T>()), true);
    ASSERT_EQUAL(thrust::equal(v1.begin(), v1.end(), v2.begin(), thrust::greater<T>()),    false);
}
DECLARE_VECTOR_UNITTEST(TestEqualSimple);

template <typename T>
void TestEqual(const size_t n)
{
    thrust::host_vector<T>   h_data1 = unittest::random_samples<T>(n);
    thrust::host_vector<T>   h_data2 = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data1 = h_data1;
    thrust::device_vector<T> d_data2 = h_data2;

    //empty ranges
    ASSERT_EQUAL(thrust::equal(h_data1.begin(), h_data1.begin(), h_data1.begin()), true);
    ASSERT_EQUAL(thrust::equal(d_data1.begin(), d_data1.begin(), d_data1.begin()), true);
    
    //symmetric cases
    ASSERT_EQUAL(thrust::equal(h_data1.begin(), h_data1.end(), h_data1.begin()), true);
    ASSERT_EQUAL(thrust::equal(d_data1.begin(), d_data1.end(), d_data1.begin()), true);

    if (n > 0)
    {
        h_data1[0] = 0; h_data2[0] = 1;
        d_data1[0] = 0; d_data2[0] = 1;

        //different vectors
        ASSERT_EQUAL(thrust::equal(h_data1.begin(), h_data1.end(), h_data2.begin()), false);
        ASSERT_EQUAL(thrust::equal(d_data1.begin(), d_data1.end(), d_data2.begin()), false);

        //different predicates
        ASSERT_EQUAL(thrust::equal(h_data1.begin(), h_data1.begin() + 1, h_data2.begin(), thrust::less<T>()), true);
        ASSERT_EQUAL(thrust::equal(d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), thrust::less<T>()), true);
        ASSERT_EQUAL(thrust::equal(h_data1.begin(), h_data1.begin() + 1, h_data2.begin(), thrust::greater<T>()), false);
        ASSERT_EQUAL(thrust::equal(d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), thrust::greater<T>()), false);
    }
}
DECLARE_VARIABLE_UNITTEST(TestEqual);

template<typename InputIterator1, typename InputIterator2>
bool equal(my_system &system, InputIterator1 /*first*/, InputIterator1, InputIterator2)
{
    system.validate_dispatch();
    return false;
}

void TestEqualDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::equal(sys,
                  vec.begin(),
                  vec.end(),
                  vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestEqualDispatchExplicit);


template<typename InputIterator1, typename InputIterator2>
bool equal(my_tag, InputIterator1 first, InputIterator1, InputIterator2)
{
    *first = 13;
    return false;
}

void TestEqualDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::equal(thrust::retag<my_tag>(vec.begin()),
                  thrust::retag<my_tag>(vec.end()),
                  thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestEqualDispatchImplicit);

struct only_set_when_both_expected
{
    long long expected;
    bool * flag;

    __device__
    bool operator()(long long x, long long y)
    {
        if (x == expected && y == expected)
        {
            *flag = true;
        }

        return x == y;
    }
};

void TestEqualWithBigIndexesHelper(int magnitude)
{
    thrust::counting_iterator<long long> begin(1);
    thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
    ASSERT_EQUAL(thrust::distance(begin, end), 1ll << magnitude);

    thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
    *has_executed = false;

    only_set_when_both_expected fn = { (1ll << magnitude) - 1,
        thrust::raw_pointer_cast(has_executed) };

    ASSERT_EQUAL(thrust::equal(thrust::device, begin, end, begin, fn), true);

    bool has_executed_h = *has_executed;
    thrust::device_free(has_executed);

    ASSERT_EQUAL(has_executed_h, true);
}

void TestEqualWithBigIndexes()
{
    TestEqualWithBigIndexesHelper(30);
    TestEqualWithBigIndexesHelper(31);
    TestEqualWithBigIndexesHelper(32);
    TestEqualWithBigIndexesHelper(33);
}
DECLARE_UNITTEST(TestEqualWithBigIndexes);
