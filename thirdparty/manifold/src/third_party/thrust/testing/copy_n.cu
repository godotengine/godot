#include <unittest/unittest.h>
#include <thrust/copy.h>

#include <list>
#include <iterator>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>

void TestCopyNFromConstIterator(void)
{
    typedef int T;

    std::vector<T> v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    std::vector<int>::const_iterator begin = v.begin();

    // copy to host_vector
    thrust::host_vector<T> h(5, (T) 10);
    thrust::host_vector<T>::iterator h_result = thrust::copy_n(begin, h.size(), h.begin());
    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);
    ASSERT_EQUAL(h[2], 2);
    ASSERT_EQUAL(h[3], 3);
    ASSERT_EQUAL(h[4], 4);
    ASSERT_EQUAL_QUIET(h_result, h.end());

    // copy to device_vector
    thrust::device_vector<T> d(5, (T) 10);
    thrust::device_vector<T>::iterator d_result = thrust::copy_n(begin, d.size(), d.begin());
    ASSERT_EQUAL(d[0], 0);
    ASSERT_EQUAL(d[1], 1);
    ASSERT_EQUAL(d[2], 2);
    ASSERT_EQUAL(d[3], 3);
    ASSERT_EQUAL(d[4], 4);
    ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_UNITTEST(TestCopyNFromConstIterator);

void TestCopyNToDiscardIterator(void)
{
    typedef int T;

    thrust::host_vector<T> h_input(5, 1);
    thrust::device_vector<T> d_input = h_input;

    // copy from host_vector
    thrust::discard_iterator<> h_result =
      thrust::copy_n(h_input.begin(), h_input.size(), thrust::make_discard_iterator());

    // copy from device_vector
    thrust::discard_iterator<> d_result =
      thrust::copy_n(d_input.begin(), d_input.size(), thrust::make_discard_iterator());

    thrust::discard_iterator<> reference(5);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_UNITTEST(TestCopyNToDiscardIterator);

template <class Vector>
void TestCopyNMatchingTypes(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    // copy to host_vector
    thrust::host_vector<T> h(5, (T) 10);
    typename thrust::host_vector<T>::iterator h_result = thrust::copy_n(v.begin(), v.size(), h.begin());
    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);
    ASSERT_EQUAL(h[2], 2);
    ASSERT_EQUAL(h[3], 3);
    ASSERT_EQUAL(h[4], 4);
    ASSERT_EQUAL_QUIET(h_result, h.end());

    // copy to device_vector
    thrust::device_vector<T> d(5, (T) 10);
    typename thrust::device_vector<T>::iterator d_result = thrust::copy_n(v.begin(), v.size(), d.begin());
    ASSERT_EQUAL(d[0], 0);
    ASSERT_EQUAL(d[1], 1);
    ASSERT_EQUAL(d[2], 2);
    ASSERT_EQUAL(d[3], 3);
    ASSERT_EQUAL(d[4], 4);
    ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_VECTOR_UNITTEST(TestCopyNMatchingTypes);

template <class Vector>
void TestCopyNMixedTypes(void)
{
    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    // copy to host_vector with different type
    thrust::host_vector<float> h(5, (float) 10);
    typename thrust::host_vector<float>::iterator h_result = thrust::copy_n(v.begin(), v.size(), h.begin());

    ASSERT_EQUAL(h[0], 0);
    ASSERT_EQUAL(h[1], 1);
    ASSERT_EQUAL(h[2], 2);
    ASSERT_EQUAL(h[3], 3);
    ASSERT_EQUAL(h[4], 4);
    ASSERT_EQUAL_QUIET(h_result, h.end());

    // copy to device_vector with different type
    thrust::device_vector<float> d(5, (float) 10);
    typename thrust::device_vector<float>::iterator d_result = thrust::copy_n(v.begin(), v.size(), d.begin());
    ASSERT_EQUAL(d[0], 0);
    ASSERT_EQUAL(d[1], 1);
    ASSERT_EQUAL(d[2], 2);
    ASSERT_EQUAL(d[3], 3);
    ASSERT_EQUAL(d[4], 4);
    ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestCopyNMixedTypes);


void TestCopyNVectorBool(void)
{
    std::vector<bool> v(3);
    v[0] = true; v[1] = false; v[2] = true;

    thrust::host_vector<bool> h(3);
    thrust::device_vector<bool> d(3);
    
    thrust::copy_n(v.begin(), v.size(), h.begin());
    thrust::copy_n(v.begin(), v.size(), d.begin());

    ASSERT_EQUAL(h[0], true);
    ASSERT_EQUAL(h[1], false);
    ASSERT_EQUAL(h[2], true);

    ASSERT_EQUAL(d[0], true);
    ASSERT_EQUAL(d[1], false);
    ASSERT_EQUAL(d[2], true);
}
DECLARE_UNITTEST(TestCopyNVectorBool);


template <class Vector>
void TestCopyNListTo(void)
{
    typedef typename Vector::value_type T;

    // copy from list to Vector
    std::list<T> l;
    l.push_back(0);
    l.push_back(1);
    l.push_back(2);
    l.push_back(3);
    l.push_back(4);
   
    Vector v(l.size());

    typename Vector::iterator v_result = thrust::copy_n(l.begin(), l.size(), v.begin());

    ASSERT_EQUAL(v[0], T(0));
    ASSERT_EQUAL(v[1], T(1));
    ASSERT_EQUAL(v[2], T(2));
    ASSERT_EQUAL(v[3], T(3));
    ASSERT_EQUAL(v[4], T(4));
    ASSERT_EQUAL_QUIET(v_result, v.end());

    l.clear();

    thrust::copy_n(v.begin(), v.size(), std::back_insert_iterator< std::list<T> >(l));

    ASSERT_EQUAL(l.size(), 5lu);

    typename std::list<T>::const_iterator iter = l.begin();
    ASSERT_EQUAL(*iter, T(0));  iter++;
    ASSERT_EQUAL(*iter, T(1));  iter++;
    ASSERT_EQUAL(*iter, T(2));  iter++;
    ASSERT_EQUAL(*iter, T(3));  iter++;
    ASSERT_EQUAL(*iter, T(4));  iter++;
}
DECLARE_VECTOR_UNITTEST(TestCopyNListTo);


template <typename Vector>
void TestCopyNCountingIterator(void)
{
    typedef typename Vector::value_type T;

    thrust::counting_iterator<T> iter(1);

    Vector vec(4);

    thrust::copy_n(iter, 4, vec.begin());

    ASSERT_EQUAL(vec[0], T(1));
    ASSERT_EQUAL(vec[1], T(2));
    ASSERT_EQUAL(vec[2], T(3));
    ASSERT_EQUAL(vec[3], T(4));
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestCopyNCountingIterator);

template <typename Vector>
void TestCopyNZipIterator(void)
{
    typedef typename Vector::value_type T;

    Vector v1(3); v1[0] = 1; v1[1] = 2; v1[2] = 3;
    Vector v2(3); v2[0] = 4; v2[1] = 5; v2[2] = 6; 
    Vector v3(3, T(0));
    Vector v4(3, T(0));

    thrust::copy_n(thrust::make_zip_iterator(thrust::make_tuple(v1.begin(),v2.begin())),
                   3,
                   thrust::make_zip_iterator(thrust::make_tuple(v3.begin(),v4.begin())));

    ASSERT_EQUAL(v1, v3);
    ASSERT_EQUAL(v2, v4);
};
DECLARE_VECTOR_UNITTEST(TestCopyNZipIterator);

template <typename Vector>
void TestCopyNConstantIteratorToZipIterator(void)
{
    typedef typename Vector::value_type T;

    Vector v1(3, T(0));
    Vector v2(3, T(0));

    thrust::copy_n(thrust::make_constant_iterator(thrust::tuple<T,T>(4,7)),
                   v1.size(),
                   thrust::make_zip_iterator(thrust::make_tuple(v1.begin(),v2.begin())));

    ASSERT_EQUAL(v1[0], T(4));
    ASSERT_EQUAL(v1[1], T(4));
    ASSERT_EQUAL(v1[2], T(4));
    ASSERT_EQUAL(v2[0], T(7));
    ASSERT_EQUAL(v2[1], T(7));
    ASSERT_EQUAL(v2[2], T(7));
};
DECLARE_VECTOR_UNITTEST(TestCopyNConstantIteratorToZipIterator);

template<typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(my_system &system, InputIterator, Size, OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

void TestCopyNDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::copy_n(sys,
                   vec.begin(),
                   1,
                   vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestCopyNDispatchExplicit);


template<typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(my_tag, InputIterator, Size, OutputIterator result)
{
    *result = 13;
    return result;
}

void TestCopyNDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::copy_n(thrust::retag<my_tag>(vec.begin()),
                   1,
                   thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestCopyNDispatchImplicit);

