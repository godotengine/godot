#include <unittest/unittest.h>
#include <thrust/extrema.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>

template <class Vector>
void TestMaxElementSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(6);
    data[0] = 3;
    data[1] = 5;
    data[2] = 1;
    data[3] = 2;
    data[4] = 5;
    data[5] = 1;

    ASSERT_EQUAL( *thrust::max_element(data.begin(), data.end()), 5);
    ASSERT_EQUAL( thrust::max_element(data.begin(), data.end()) - data.begin(), 1);
    
    ASSERT_EQUAL( *thrust::max_element(data.begin(), data.end(), thrust::greater<T>()), 1);
    ASSERT_EQUAL( thrust::max_element(data.begin(), data.end(), thrust::greater<T>()) - data.begin(), 2);
}
DECLARE_VECTOR_UNITTEST(TestMaxElementSimple);

template <class Vector>
void TestMaxElementWithTransform(void)
{
    typedef typename Vector::value_type T;

    Vector data(6);
    data[0] = 3;
    data[1] = 5;
    data[2] = 1;
    data[3] = 2;
    data[4] = 5;
    data[5] = 1;

    ASSERT_EQUAL( *thrust::max_element(
          thrust::make_transform_iterator(data.begin(), thrust::negate<T>()),
          thrust::make_transform_iterator(data.end(),   thrust::negate<T>())), -1);
    ASSERT_EQUAL( *thrust::max_element(
          thrust::make_transform_iterator(data.begin(), thrust::negate<T>()),
          thrust::make_transform_iterator(data.end(),   thrust::negate<T>()),
          thrust::greater<T>()), -5);
    
}
DECLARE_VECTOR_UNITTEST(TestMaxElementWithTransform);

template<typename T>
void TestMaxElement(const size_t n)
{
    thrust::host_vector<T> h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    typename thrust::host_vector<T>::iterator   h_max = thrust::max_element(h_data.begin(), h_data.end());
    typename thrust::device_vector<T>::iterator d_max = thrust::max_element(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_max - h_data.begin(), d_max - d_data.begin());
    
    typename thrust::host_vector<T>::iterator   h_min = thrust::max_element(h_data.begin(), h_data.end(), thrust::greater<T>());
    typename thrust::device_vector<T>::iterator d_min = thrust::max_element(d_data.begin(), d_data.end(), thrust::greater<T>());

    ASSERT_EQUAL(h_min - h_data.begin(), d_min - d_data.begin());
}
DECLARE_VARIABLE_UNITTEST(TestMaxElement);


template<typename ForwardIterator>
ForwardIterator max_element(my_system &system, ForwardIterator first, ForwardIterator)
{
    system.validate_dispatch();
    return first;
}

void TestMaxElementDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::max_element(sys, vec.begin(), vec.end());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestMaxElementDispatchExplicit);


template<typename ForwardIterator>
ForwardIterator max_element(my_tag, ForwardIterator first, ForwardIterator)
{
    *first = 13;
    return first;
}

void TestMaxElementDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::max_element(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestMaxElementDispatchImplicit);

void TestMaxElementWithBigIndexesHelper(int magnitude)
{
    thrust::counting_iterator<long long> begin(1);
    thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
    ASSERT_EQUAL(thrust::distance(begin, end), 1ll << magnitude);

    ASSERT_EQUAL(*thrust::max_element(thrust::device, begin, end), (1ll << magnitude));
}

void TestMaxElementWithBigIndexes()
{
    TestMaxElementWithBigIndexesHelper(30);
    TestMaxElementWithBigIndexesHelper(31);
    TestMaxElementWithBigIndexesHelper(32);
    TestMaxElementWithBigIndexesHelper(33);
}
DECLARE_UNITTEST(TestMaxElementWithBigIndexes);
