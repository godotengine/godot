#include <unittest/unittest.h>

#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

template <class Vector>
void TestSortPermutationIterator(void)
{
  typedef typename Vector::iterator Iterator;

  Vector A(10);
  A[0] = 2;
  A[1] = 9;
  A[2] = 0;
  A[3] = 1;
  A[4] = 5;
  A[5] = 3;
  A[6] = 8;
  A[7] = 6;
  A[8] = 7;
  A[9] = 4;

  strided_range<Iterator> S(A.begin(), A.end(), 2);

  thrust::sort(S.begin(), S.end());

  ASSERT_EQUAL(A[0], 0);
  ASSERT_EQUAL(A[1], 9);
  ASSERT_EQUAL(A[2], 2);
  ASSERT_EQUAL(A[3], 1);
  ASSERT_EQUAL(A[4], 5);
  ASSERT_EQUAL(A[5], 3);
  ASSERT_EQUAL(A[6], 7);
  ASSERT_EQUAL(A[7], 6);
  ASSERT_EQUAL(A[8], 8);
  ASSERT_EQUAL(A[9], 4);
}
DECLARE_VECTOR_UNITTEST(TestSortPermutationIterator);

template <class Vector>
void TestStableSortPermutationIterator(void)
{
  typedef typename Vector::iterator Iterator;

  Vector A(10);
  A[0] = 2;
  A[1] = 9;
  A[2] = 0;
  A[3] = 1;
  A[4] = 5;
  A[5] = 3;
  A[6] = 8;
  A[7] = 6;
  A[8] = 7;
  A[9] = 4;

  strided_range<Iterator> S(A.begin(), A.end(), 2);

  thrust::stable_sort(S.begin(), S.end());

  ASSERT_EQUAL(A[0], 0);
  ASSERT_EQUAL(A[1], 9);
  ASSERT_EQUAL(A[2], 2);
  ASSERT_EQUAL(A[3], 1);
  ASSERT_EQUAL(A[4], 5);
  ASSERT_EQUAL(A[5], 3);
  ASSERT_EQUAL(A[6], 7);
  ASSERT_EQUAL(A[7], 6);
  ASSERT_EQUAL(A[8], 8);
  ASSERT_EQUAL(A[9], 4);
}
DECLARE_VECTOR_UNITTEST(TestStableSortPermutationIterator);

template <class Vector>
void TestSortByKeyPermutationIterator(void)
{
  typedef typename Vector::iterator Iterator;

  Vector A(10), B(10);
  A[0] = 2; B[0] = 0;
  A[1] = 9; B[1] = 1;
  A[2] = 0; B[2] = 2;
  A[3] = 1; B[3] = 3;
  A[4] = 5; B[4] = 4;
  A[5] = 3; B[5] = 5;
  A[6] = 8; B[6] = 6;
  A[7] = 6; B[7] = 7;
  A[8] = 7; B[8] = 8;
  A[9] = 4; B[9] = 9;
  
  strided_range<Iterator> S(A.begin(), A.end(), 2);
  strided_range<Iterator> T(B.begin(), B.end(), 2);

  thrust::sort_by_key(S.begin(), S.end(), T.begin());

  ASSERT_EQUAL(A[0], 0);
  ASSERT_EQUAL(A[1], 9);
  ASSERT_EQUAL(A[2], 2);
  ASSERT_EQUAL(A[3], 1);
  ASSERT_EQUAL(A[4], 5);
  ASSERT_EQUAL(A[5], 3);
  ASSERT_EQUAL(A[6], 7);
  ASSERT_EQUAL(A[7], 6);
  ASSERT_EQUAL(A[8], 8);
  ASSERT_EQUAL(A[9], 4);
  
  ASSERT_EQUAL(B[0], 2);
  ASSERT_EQUAL(B[1], 1);
  ASSERT_EQUAL(B[2], 0);
  ASSERT_EQUAL(B[3], 3);
  ASSERT_EQUAL(B[4], 4);
  ASSERT_EQUAL(B[5], 5);
  ASSERT_EQUAL(B[6], 8);
  ASSERT_EQUAL(B[7], 7);
  ASSERT_EQUAL(B[8], 6);
  ASSERT_EQUAL(B[9], 9);
}
DECLARE_VECTOR_UNITTEST(TestSortByKeyPermutationIterator);

template <class Vector>
void TestStableSortByKeyPermutationIterator(void)
{
  typedef typename Vector::iterator Iterator;

  Vector A(10), B(10);
  A[0] = 2; B[0] = 0;
  A[1] = 9; B[1] = 1;
  A[2] = 0; B[2] = 2;
  A[3] = 1; B[3] = 3;
  A[4] = 5; B[4] = 4;
  A[5] = 3; B[5] = 5;
  A[6] = 8; B[6] = 6;
  A[7] = 6; B[7] = 7;
  A[8] = 7; B[8] = 8;
  A[9] = 4; B[9] = 9;
  
  strided_range<Iterator> S(A.begin(), A.end(), 2);
  strided_range<Iterator> T(B.begin(), B.end(), 2);

  thrust::stable_sort_by_key(S.begin(), S.end(), T.begin());

  ASSERT_EQUAL(A[0], 0);
  ASSERT_EQUAL(A[1], 9);
  ASSERT_EQUAL(A[2], 2);
  ASSERT_EQUAL(A[3], 1);
  ASSERT_EQUAL(A[4], 5);
  ASSERT_EQUAL(A[5], 3);
  ASSERT_EQUAL(A[6], 7);
  ASSERT_EQUAL(A[7], 6);
  ASSERT_EQUAL(A[8], 8);
  ASSERT_EQUAL(A[9], 4);
  
  ASSERT_EQUAL(B[0], 2);
  ASSERT_EQUAL(B[1], 1);
  ASSERT_EQUAL(B[2], 0);
  ASSERT_EQUAL(B[3], 3);
  ASSERT_EQUAL(B[4], 4);
  ASSERT_EQUAL(B[5], 5);
  ASSERT_EQUAL(B[6], 8);
  ASSERT_EQUAL(B[7], 7);
  ASSERT_EQUAL(B[8], 6);
  ASSERT_EQUAL(B[9], 9);
}
DECLARE_VECTOR_UNITTEST(TestStableSortByKeyPermutationIterator);

