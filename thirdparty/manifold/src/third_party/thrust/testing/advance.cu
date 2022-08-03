#include <unittest/unittest.h>
#include <thrust/advance.h>
#include <thrust/sequence.h>

// TODO expand this with other iterator types (forward, bidirectional, etc.)

template <typename Vector>
void TestAdvance()
{
    typedef typename Vector::value_type T;
    typedef typename Vector::iterator Iterator;

    Vector v(10);
    thrust::sequence(v.begin(), v.end());

    Iterator i = v.begin();

    thrust::advance(i, 1);

    ASSERT_EQUAL(*i, T(1));
    
    thrust::advance(i, 8);

    ASSERT_EQUAL(*i, T(9));
    
    thrust::advance(i, -4);

    ASSERT_EQUAL(*i, T(5));
}
DECLARE_VECTOR_UNITTEST(TestAdvance);

template <typename Vector>
void TestNext()
{
    typedef typename Vector::value_type T;
    typedef typename Vector::iterator Iterator;

    Vector v(10);
    thrust::sequence(v.begin(), v.end());

    Iterator const i0 = v.begin();

    Iterator const i1 = thrust::next(i0);

    ASSERT_EQUAL(*i0, T(0));
    ASSERT_EQUAL(*i1, T(1));
    
    Iterator const i2 = thrust::next(i1, 8);

    ASSERT_EQUAL(*i0, T(0));
    ASSERT_EQUAL(*i1, T(1));
    ASSERT_EQUAL(*i2, T(9));
    
    Iterator const i3 = thrust::next(i2, -4);

    ASSERT_EQUAL(*i0, T(0));
    ASSERT_EQUAL(*i1, T(1));
    ASSERT_EQUAL(*i2, T(9));
    ASSERT_EQUAL(*i3, T(5));
}
DECLARE_VECTOR_UNITTEST(TestNext);

template <typename Vector>
void TestPrev()
{
    typedef typename Vector::value_type T;
    typedef typename Vector::iterator Iterator;

    Vector v(10);
    thrust::sequence(v.begin(), v.end());

    Iterator const i0 = v.end();

    Iterator const i1 = thrust::prev(i0);

    ASSERT_EQUAL_QUIET(i0, v.end());
    ASSERT_EQUAL(*i1, T(9));
    
    Iterator const i2 = thrust::prev(i1, 8);

    ASSERT_EQUAL_QUIET(i0, v.end());
    ASSERT_EQUAL(*i1, T(9));
    ASSERT_EQUAL(*i2, T(1));
    
    Iterator const i3 = thrust::prev(i2, -4);

    ASSERT_EQUAL_QUIET(i0, v.end());
    ASSERT_EQUAL(*i1, T(9));
    ASSERT_EQUAL(*i2, T(1));
    ASSERT_EQUAL(*i3, T(5));
}
DECLARE_VECTOR_UNITTEST(TestPrev);

