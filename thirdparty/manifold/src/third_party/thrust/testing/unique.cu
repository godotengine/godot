#include <unittest/unittest.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>


template <typename ForwardIterator>
ForwardIterator unique(my_system &system,
                       ForwardIterator first,
                       ForwardIterator)
{
    system.validate_dispatch();
    return first;
}

void TestUniqueDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::unique(sys, vec.begin(), vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestUniqueDispatchExplicit);


template <typename ForwardIterator>
ForwardIterator unique(my_tag,
                       ForwardIterator first,
                       ForwardIterator)
{
    *first = 13;
    return first;
}


void TestUniqueDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::unique(thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestUniqueDispatchImplicit);


template <typename InputIterator,
          typename OutputIterator>
OutputIterator unique_copy(my_system &system,
                           InputIterator,
                           InputIterator,
                           OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

void TestUniqueCopyDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::unique_copy(sys, vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestUniqueCopyDispatchExplicit);


template <typename InputIterator,
          typename OutputIterator>
OutputIterator unique_copy(my_tag,
                           InputIterator,
                           InputIterator,
                           OutputIterator result)
{
    *result = 13;
    return result;
}

void TestUniqueCopyDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::unique_copy(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestUniqueCopyDispatchImplicit);


template <typename ForwardIterator>
typename thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(my_system &system,
                 ForwardIterator,
                 ForwardIterator)
{
    system.validate_dispatch();
    return 0;
}

void TestUniqueCountDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::unique_count(sys, vec.begin(), vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestUniqueCountDispatchExplicit);


template <typename ForwardIterator>
typename thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(my_tag,
                 ForwardIterator,
                 ForwardIterator)
{
    return 13;
}

void TestUniqueCountDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    auto result = thrust::unique_count(
        thrust::retag<my_tag>(vec.begin()),
        thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, result);
}
DECLARE_UNITTEST(TestUniqueCountDispatchImplicit);


template<typename T>
struct is_equal_div_10_unique
{
    __host__ __device__
    bool operator()(const T x, const T& y) const { return ((int) x / 10) == ((int) y / 10); }
};

template<typename Vector>
void TestUniqueSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(10);
    data[0] = 11; 
    data[1] = 11; 
    data[2] = 12;
    data[3] = 20; 
    data[4] = 29; 
    data[5] = 21; 
    data[6] = 21; 
    data[7] = 31; 
    data[8] = 31; 
    data[9] = 37; 

    typename Vector::iterator new_last;
    
    new_last = thrust::unique(data.begin(), data.end());

    ASSERT_EQUAL(new_last - data.begin(), 7);
    ASSERT_EQUAL(data[0], 11);
    ASSERT_EQUAL(data[1], 12);
    ASSERT_EQUAL(data[2], 20);
    ASSERT_EQUAL(data[3], 29);
    ASSERT_EQUAL(data[4], 21);
    ASSERT_EQUAL(data[5], 31);
    ASSERT_EQUAL(data[6], 37);

    new_last = thrust::unique(data.begin(), new_last, is_equal_div_10_unique<T>());

    ASSERT_EQUAL(new_last - data.begin(), 3);
    ASSERT_EQUAL(data[0], 11);
    ASSERT_EQUAL(data[1], 20);
    ASSERT_EQUAL(data[2], 31);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestUniqueSimple);


template<typename T>
struct TestUnique
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_data = unittest::random_integers<bool>(n);
        thrust::device_vector<T> d_data = h_data;

        typename thrust::host_vector<T>::iterator   h_new_last;
        typename thrust::device_vector<T>::iterator d_new_last;

        h_new_last = thrust::unique(h_data.begin(), h_data.end());
        d_new_last = thrust::unique(d_data.begin(), d_data.end());

        ASSERT_EQUAL(h_new_last - h_data.begin(), d_new_last - d_data.begin());
        
        h_data.resize(h_new_last - h_data.begin());
        d_data.resize(d_new_last - d_data.begin());

        ASSERT_EQUAL(h_data, d_data);
    }
};
VariableUnitTest<TestUnique, IntegralTypes> TestUniqueInstance;


template<typename Vector>
void TestUniqueCopySimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(10);
    data[0] = 11; 
    data[1] = 11; 
    data[2] = 12;
    data[3] = 20; 
    data[4] = 29; 
    data[5] = 21; 
    data[6] = 21; 
    data[7] = 31; 
    data[8] = 31; 
    data[9] = 37; 
    
    Vector output(10, -1);

    typename Vector::iterator new_last;
    
    new_last = thrust::unique_copy(data.begin(), data.end(), output.begin());

    ASSERT_EQUAL(new_last - output.begin(), 7);
    ASSERT_EQUAL(output[0], 11);
    ASSERT_EQUAL(output[1], 12);
    ASSERT_EQUAL(output[2], 20);
    ASSERT_EQUAL(output[3], 29);
    ASSERT_EQUAL(output[4], 21);
    ASSERT_EQUAL(output[5], 31);
    ASSERT_EQUAL(output[6], 37);

    new_last = thrust::unique_copy(output.begin(), new_last, data.begin(), is_equal_div_10_unique<T>());

    ASSERT_EQUAL(new_last - data.begin(), 3);
    ASSERT_EQUAL(data[0], 11);
    ASSERT_EQUAL(data[1], 20);
    ASSERT_EQUAL(data[2], 31);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestUniqueCopySimple);


template<typename T>
struct TestUniqueCopy
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_data = unittest::random_integers<bool>(n);
        thrust::device_vector<T> d_data = h_data;
        
        thrust::host_vector<T>   h_output(n);
        thrust::device_vector<T> d_output(n);

        typename thrust::host_vector<T>::iterator   h_new_last;
        typename thrust::device_vector<T>::iterator d_new_last;

        h_new_last = thrust::unique_copy(h_data.begin(), h_data.end(), h_output.begin());
        d_new_last = thrust::unique_copy(d_data.begin(), d_data.end(), d_output.begin());

        ASSERT_EQUAL(h_new_last - h_output.begin(), d_new_last - d_output.begin());
        
        h_data.resize(h_new_last - h_output.begin());
        d_data.resize(d_new_last - d_output.begin());

        ASSERT_EQUAL(h_output, d_output);
    }
};
VariableUnitTest<TestUniqueCopy, IntegralTypes> TestUniqueCopyInstance;


template<typename T>
struct TestUniqueCopyToDiscardIterator
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_data = unittest::random_integers<bool>(n);
        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T> h_unique = h_data;
        h_unique.erase(thrust::unique(h_unique.begin(), h_unique.end()), h_unique.end());

        thrust::discard_iterator<> reference(h_unique.size());

        typename thrust::host_vector<T>::iterator   h_new_last;
        typename thrust::device_vector<T>::iterator d_new_last;

        thrust::discard_iterator<> h_result =
          thrust::unique_copy(h_data.begin(), h_data.end(), thrust::make_discard_iterator());

        thrust::discard_iterator<> d_result =
          thrust::unique_copy(d_data.begin(), d_data.end(), thrust::make_discard_iterator());

        ASSERT_EQUAL_QUIET(reference, h_result);
        ASSERT_EQUAL_QUIET(reference, d_result);
    }
};
VariableUnitTest<TestUniqueCopyToDiscardIterator, IntegralTypes> TestUniqueCopyToDiscardIteratorInstance;


template <typename Vector>
void TestUniqueCountSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(10);
    data[0] = 11;
    data[1] = 11;
    data[2] = 12;
    data[3] = 20;
    data[4] = 29;
    data[5] = 21;
    data[6] = 21;
    data[7] = 31;
    data[8] = 31;
    data[9] = 37;

    int count = thrust::unique_count(data.begin(), data.end());

    ASSERT_EQUAL(count, 7);

    int div_10_count = thrust::unique_count(data.begin(), data.end(), is_equal_div_10_unique<T>());

    ASSERT_EQUAL(div_10_count, 3);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestUniqueCountSimple);

template <typename T>
struct TestUniqueCount
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T> h_data = unittest::random_integers<bool>(n);
        thrust::device_vector<T> d_data = h_data;

        int h_count{};
        int d_count{};

        h_count = thrust::unique_count(h_data.begin(), h_data.end());
        d_count = thrust::unique_count(d_data.begin(), d_data.end());

        ASSERT_EQUAL(h_count, d_count);
    }
};
VariableUnitTest<TestUniqueCount, IntegralTypes> TestUniqueCountInstance;
