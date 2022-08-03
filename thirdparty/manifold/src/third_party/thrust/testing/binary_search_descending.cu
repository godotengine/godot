#include <unittest/unittest.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

#include <thrust/sequence.h>
#include <thrust/sort.h>

//////////////////////
// Scalar Functions //
//////////////////////

template <class Vector>
void TestScalarLowerBoundDescendingSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 8;
    vec[1] = 7;
    vec[2] = 5;
    vec[3] = 2;
    vec[4] = 0;

    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::lower_bound(vec.begin(), vec.end(), T{0}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::lower_bound(vec.begin(), vec.end(), T{1}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::lower_bound(vec.begin(), vec.end(), T{2}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::lower_bound(vec.begin(), vec.end(), T{3}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::lower_bound(vec.begin(), vec.end(), T{4}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::lower_bound(vec.begin(), vec.end(), T{5}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::lower_bound(vec.begin(), vec.end(), T{6}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 1, thrust::lower_bound(vec.begin(), vec.end(), T{7}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::lower_bound(vec.begin(), vec.end(), T{8}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::lower_bound(vec.begin(), vec.end(), T{9}, thrust::greater<T>()));
}
DECLARE_VECTOR_UNITTEST(TestScalarLowerBoundDescendingSimple);


template <class Vector>
void TestScalarUpperBoundDescendingSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 8;
    vec[1] = 7;
    vec[2] = 5;
    vec[3] = 2;
    vec[4] = 0;

    ASSERT_EQUAL_QUIET(vec.begin() + 5, thrust::upper_bound(vec.begin(), vec.end(), T{0}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::upper_bound(vec.begin(), vec.end(), T{1}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::upper_bound(vec.begin(), vec.end(), T{2}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::upper_bound(vec.begin(), vec.end(), T{3}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::upper_bound(vec.begin(), vec.end(), T{4}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::upper_bound(vec.begin(), vec.end(), T{5}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::upper_bound(vec.begin(), vec.end(), T{6}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::upper_bound(vec.begin(), vec.end(), T{7}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 1, thrust::upper_bound(vec.begin(), vec.end(), T{8}, thrust::greater<T>()));
    ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::upper_bound(vec.begin(), vec.end(), T{9}, thrust::greater<T>()));
}
DECLARE_VECTOR_UNITTEST(TestScalarUpperBoundDescendingSimple);


template <class Vector>
void TestScalarBinarySearchDescendingSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 8;
    vec[1] = 7;
    vec[2] = 5;
    vec[3] = 2;
    vec[4] = 0;

    ASSERT_EQUAL(true,  thrust::binary_search(vec.begin(), vec.end(), T{0}, thrust::greater<T>()));
    ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), T{1}, thrust::greater<T>()));
    ASSERT_EQUAL(true,  thrust::binary_search(vec.begin(), vec.end(), T{2}, thrust::greater<T>()));
    ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), T{3}, thrust::greater<T>()));
    ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), T{4}, thrust::greater<T>()));
    ASSERT_EQUAL(true,  thrust::binary_search(vec.begin(), vec.end(), T{5}, thrust::greater<T>()));
    ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), T{6}, thrust::greater<T>()));
    ASSERT_EQUAL(true,  thrust::binary_search(vec.begin(), vec.end(), T{7}, thrust::greater<T>()));
    ASSERT_EQUAL(true,  thrust::binary_search(vec.begin(), vec.end(), T{8}, thrust::greater<T>()));
    ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), T{9}, thrust::greater<T>()));
}
DECLARE_VECTOR_UNITTEST(TestScalarBinarySearchDescendingSimple);


template <class Vector>
void TestScalarEqualRangeDescendingSimple(void)
{
    typedef typename Vector::value_type T;

    Vector vec(5);

    vec[0] = 8;
    vec[1] = 7;
    vec[2] = 5;
    vec[3] = 2;
    vec[4] = 0;

    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::equal_range(vec.begin(), vec.end(), T{0}, thrust::greater<T>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::equal_range(vec.begin(), vec.end(), T{1}, thrust::greater<T>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), T{2}, thrust::greater<T>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), T{3}, thrust::greater<T>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), T{4}, thrust::greater<T>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::equal_range(vec.begin(), vec.end(), T{5}, thrust::greater<T>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::equal_range(vec.begin(), vec.end(), T{6}, thrust::greater<T>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 1, thrust::equal_range(vec.begin(), vec.end(), T{7}, thrust::greater<T>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::equal_range(vec.begin(), vec.end(), T{8}, thrust::greater<T>()).first);
    ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::equal_range(vec.begin(), vec.end(), T{9}, thrust::greater<T>()).first);

    ASSERT_EQUAL_QUIET(vec.begin() + 5, thrust::equal_range(vec.begin(), vec.end(), T{0}, thrust::greater<T>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::equal_range(vec.begin(), vec.end(), T{1}, thrust::greater<T>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::equal_range(vec.begin(), vec.end(), T{2}, thrust::greater<T>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), T{3}, thrust::greater<T>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), T{4}, thrust::greater<T>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), T{5}, thrust::greater<T>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::equal_range(vec.begin(), vec.end(), T{6}, thrust::greater<T>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::equal_range(vec.begin(), vec.end(), T{7}, thrust::greater<T>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 1, thrust::equal_range(vec.begin(), vec.end(), T{8}, thrust::greater<T>()).second);
    ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::equal_range(vec.begin(), vec.end(), T{9}, thrust::greater<T>()).second);
}
DECLARE_VECTOR_UNITTEST(TestScalarEqualRangeDescendingSimple);

