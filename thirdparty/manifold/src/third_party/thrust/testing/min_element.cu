#include <unittest/unittest.h>
#include <thrust/extrema.h>
#include <thrust/iterator/retag.h>

template <class Vector>
void TestMinElementSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(6);
    data[0] = 3;
    data[1] = 5;
    data[2] = 1;
    data[3] = 2;
    data[4] = 5;
    data[5] = 1;

    ASSERT_EQUAL( *thrust::min_element(data.begin(), data.end()), 1);
    ASSERT_EQUAL( thrust::min_element(data.begin(), data.end()) - data.begin(), 2);
    
    ASSERT_EQUAL( *thrust::min_element(data.begin(), data.end(), thrust::greater<T>()), 5);
    ASSERT_EQUAL( thrust::min_element(data.begin(), data.end(), thrust::greater<T>()) - data.begin(), 1);
}
DECLARE_VECTOR_UNITTEST(TestMinElementSimple);

template <class Vector>
void TestMinElementWithTransform(void)
{
    typedef typename Vector::value_type T;

    Vector data(6);
    data[0] = 3;
    data[1] = 5;
    data[2] = 1;
    data[3] = 2;
    data[4] = 5;
    data[5] = 1;

    ASSERT_EQUAL( *thrust::min_element(
          thrust::make_transform_iterator(data.begin(), thrust::negate<T>()),
          thrust::make_transform_iterator(data.end(),   thrust::negate<T>())), -5);
    ASSERT_EQUAL( *thrust::min_element(
          thrust::make_transform_iterator(data.begin(), thrust::negate<T>()),
          thrust::make_transform_iterator(data.end(),   thrust::negate<T>()),
          thrust::greater<T>()), -1);
    
}
DECLARE_VECTOR_UNITTEST(TestMinElementWithTransform);

template<typename T>
void TestMinElement(const size_t n)
{
    thrust::host_vector<T> h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    typename thrust::host_vector<T>::iterator   h_min = thrust::min_element(h_data.begin(), h_data.end());
    typename thrust::device_vector<T>::iterator d_min = thrust::min_element(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_min - h_data.begin(), d_min - d_data.begin());
    
    typename thrust::host_vector<T>::iterator   h_max = thrust::min_element(h_data.begin(), h_data.end(), thrust::greater<T>());
    typename thrust::device_vector<T>::iterator d_max = thrust::min_element(d_data.begin(), d_data.end(), thrust::greater<T>());

    ASSERT_EQUAL(h_max - h_data.begin(), d_max - d_data.begin());
}
DECLARE_VARIABLE_UNITTEST(TestMinElement);


template<typename ForwardIterator>
ForwardIterator min_element(my_system &system, ForwardIterator first, ForwardIterator)
{
    system.validate_dispatch();
    return first;
}

void TestMinElementDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::min_element(sys, vec.begin(), vec.end());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestMinElementDispatchExplicit);


template<typename ForwardIterator>
ForwardIterator min_element(my_tag, ForwardIterator first, ForwardIterator)
{
    *first = 13;
    return first;
}

void TestMinElementDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::min_element(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestMinElementDispatchImplicit);

void TestMinElementWithBigIndexesHelper(int magnitude)
{
    thrust::counting_iterator<long long> begin(1);
    thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
    ASSERT_EQUAL(thrust::distance(begin, end), 1ll << magnitude);

    ASSERT_EQUAL(
        *thrust::min_element(thrust::device, begin, end, thrust::greater<long long>()),
        (1ll << magnitude));
}

void TestMinElementWithBigIndexes()
{
    TestMinElementWithBigIndexesHelper(30);
    TestMinElementWithBigIndexesHelper(31);
    TestMinElementWithBigIndexesHelper(32);
    TestMinElementWithBigIndexesHelper(33);
}
DECLARE_UNITTEST(TestMinElementWithBigIndexes);
