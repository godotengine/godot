#include <unittest/unittest.h>
#include <thrust/mismatch.h>
#include <thrust/iterator/retag.h>

template <class Vector>
void TestMismatchSimple(void)
{
    Vector a(4); Vector b(4);
    a[0] = 1; b[0] = 1;
    a[1] = 2; b[1] = 2;
    a[2] = 3; b[2] = 4;
    a[3] = 4; b[3] = 3;

    ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).first  - a.begin(), 2);
    ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 2);

    b[2] = 3;
    
    ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).first  - a.begin(), 3);
    ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 3);
    
    b[3] = 4;
    
    ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).first  - a.begin(), 4);
    ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 4);
}
DECLARE_VECTOR_UNITTEST(TestMismatchSimple);


template <typename InputIterator1, typename InputIterator2>
thrust::pair<InputIterator1, InputIterator2> mismatch(my_system &system,
                                                      InputIterator1 first,
                                                      InputIterator1,
                                                      InputIterator2)
{
    system.validate_dispatch();
    return thrust::make_pair(first,first);
}

void TestMismatchDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::mismatch(sys,
                     vec.begin(),
                     vec.begin(),
                     vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestMismatchDispatchExplicit);


template <typename InputIterator1, typename InputIterator2>
thrust::pair<InputIterator1, InputIterator2> mismatch(my_tag,
                                                      InputIterator1 first,
                                                      InputIterator1,
                                                      InputIterator2)
{
    *first = 13;
    return thrust::make_pair(first,first);
}

void TestMismatchDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::mismatch(thrust::retag<my_tag>(vec.begin()),
                     thrust::retag<my_tag>(vec.begin()),
                     thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestMismatchDispatchImplicit);

