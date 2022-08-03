#include <unittest/unittest.h>
#include <thrust/detail/trivial_sequence.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <thrust/iterator/zip_iterator.h> 

template <typename Iterator>
void test(Iterator first, Iterator last)
{
    typedef typename thrust::iterator_system<Iterator>::type System;
    System system;
    thrust::detail::trivial_sequence<Iterator,System> ts(system, first, last);
    typedef typename thrust::iterator_traits<Iterator>::value_type ValueType;
    
    ASSERT_EQUAL_QUIET((ValueType) ts.begin()[0], ValueType(0, 11)); 
    ASSERT_EQUAL_QUIET((ValueType) ts.begin()[1], ValueType(2, 11)); 
    ASSERT_EQUAL_QUIET((ValueType) ts.begin()[2], ValueType(1, 13)); 
    ASSERT_EQUAL_QUIET((ValueType) ts.begin()[3], ValueType(0, 10)); 
    ASSERT_EQUAL_QUIET((ValueType) ts.begin()[4], ValueType(1, 12)); 

    ts.begin()[0] = ValueType(0,0);
    ts.begin()[1] = ValueType(0,0);
    ts.begin()[2] = ValueType(0,0);
    ts.begin()[3] = ValueType(0,0);
    ts.begin()[4] = ValueType(0,0);

    typedef typename thrust::detail::trivial_sequence<Iterator,System>::iterator_type TrivialIterator;

    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<Iterator>::value,        false);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<TrivialIterator>::value,  true);
}

template <class Vector>
void TestTrivialSequence(void)
{
    Vector A(5);  A[0] =  0;  A[1] =  2;  A[2] =  1;  A[3] =  0;  A[4] =  1;  
    Vector B(5);  B[0] = 11;  B[1] = 11;  B[2] = 13;  B[3] = 10;  B[4] = 12;

    test(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end())));

    // ensure that values weren't modified
    ASSERT_EQUAL(A[0], 0);  ASSERT_EQUAL(B[0], 11); 
    ASSERT_EQUAL(A[1], 2);  ASSERT_EQUAL(B[1], 11); 
    ASSERT_EQUAL(A[2], 1);  ASSERT_EQUAL(B[2], 13); 
    ASSERT_EQUAL(A[3], 0);  ASSERT_EQUAL(B[3], 10); 
    ASSERT_EQUAL(A[4], 1);  ASSERT_EQUAL(B[4], 12); 
}
DECLARE_VECTOR_UNITTEST(TestTrivialSequence);

