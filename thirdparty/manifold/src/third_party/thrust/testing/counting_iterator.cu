#include <unittest/unittest.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/detail/cstdint.h>


THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

template <typename T>
void TestCountingDefaultConstructor(void)
{
  thrust::counting_iterator<T> iter0;
  ASSERT_EQUAL(*iter0, T{});
}
DECLARE_GENERIC_UNITTEST(TestCountingDefaultConstructor);

void TestCountingIteratorCopyConstructor(void)
{
    thrust::counting_iterator<int> iter0(100);

    thrust::counting_iterator<int> iter1(iter0);

    ASSERT_EQUAL_QUIET(iter0, iter1);
    ASSERT_EQUAL(*iter0, *iter1);

    // construct from related space
    thrust::counting_iterator<int, thrust::host_system_tag> h_iter = iter0;
    ASSERT_EQUAL(*iter0, *h_iter);

    thrust::counting_iterator<int, thrust::device_system_tag> d_iter = iter0;
    ASSERT_EQUAL(*iter0, *d_iter);
}
DECLARE_UNITTEST(TestCountingIteratorCopyConstructor);


void TestCountingIteratorIncrement(void)
{
    thrust::counting_iterator<int> iter(0);

    ASSERT_EQUAL(*iter, 0);

    iter++;

    ASSERT_EQUAL(*iter, 1);
    
    iter++;
    iter++;
    
    ASSERT_EQUAL(*iter, 3);

    iter += 5;
    
    ASSERT_EQUAL(*iter, 8);

    iter -= 10;
    
    ASSERT_EQUAL(*iter, -2);
}
DECLARE_UNITTEST(TestCountingIteratorIncrement);


void TestCountingIteratorComparison(void)
{
    thrust::counting_iterator<int> iter1(0);
    thrust::counting_iterator<int> iter2(0);

    ASSERT_EQUAL(iter1 - iter2, 0);
    ASSERT_EQUAL(iter1 == iter2, true);

    iter1++;
    
    ASSERT_EQUAL(iter1 - iter2, 1);
    ASSERT_EQUAL(iter1 == iter2, false);
   
    iter2++;

    ASSERT_EQUAL(iter1 - iter2, 0);
    ASSERT_EQUAL(iter1 == iter2, true);
  
    iter1 += 100;
    iter2 += 100;

    ASSERT_EQUAL(iter1 - iter2, 0);
    ASSERT_EQUAL(iter1 == iter2, true);
}
DECLARE_UNITTEST(TestCountingIteratorComparison);


void TestCountingIteratorFloatComparison(void)
{
    thrust::counting_iterator<float> iter1(0);
    thrust::counting_iterator<float> iter2(0);

    ASSERT_EQUAL(iter1 - iter2, 0);
    ASSERT_EQUAL(iter1 == iter2, true);
    ASSERT_EQUAL(iter1 <  iter2, false);
    ASSERT_EQUAL(iter2 <  iter1, false);

    iter1++;
    
    ASSERT_EQUAL(iter1 - iter2, 1);
    ASSERT_EQUAL(iter1 == iter2, false);
    ASSERT_EQUAL(iter2 < iter1, true); 
    ASSERT_EQUAL(iter1 < iter2, false); 
   
    iter2++;

    ASSERT_EQUAL(iter1 - iter2, 0);
    ASSERT_EQUAL(iter1 == iter2, true);
    ASSERT_EQUAL(iter1 < iter2, false);
    ASSERT_EQUAL(iter2 < iter1, false);
  
    iter1 += 100;
    iter2 += 100;

    ASSERT_EQUAL(iter1 - iter2, 0);
    ASSERT_EQUAL(iter1 == iter2, true);
    ASSERT_EQUAL(iter1 < iter2, false);
    ASSERT_EQUAL(iter2 < iter1, false);


    thrust::counting_iterator<float> iter3(0);
    thrust::counting_iterator<float> iter4(0.5);

    ASSERT_EQUAL(iter3 - iter4, 0);
    ASSERT_EQUAL(iter3 == iter4, true);
    ASSERT_EQUAL(iter3 < iter4, false);
    ASSERT_EQUAL(iter4 < iter3, false);

    iter3++; // iter3 = 1.0, iter4 = 0.5
    
    ASSERT_EQUAL(iter3 - iter4, 0);
    ASSERT_EQUAL(iter3 == iter4, true);
    ASSERT_EQUAL(iter3 < iter4, false);
    ASSERT_EQUAL(iter4 < iter3, false);
   
    iter4++; // iter3 = 1.0, iter4 = 1.5

    ASSERT_EQUAL(iter3 - iter4, 0);
    ASSERT_EQUAL(iter3 == iter4, true);
    ASSERT_EQUAL(iter3 < iter4, false);
    ASSERT_EQUAL(iter4 < iter3, false);

    iter4++; // iter3 = 1.0, iter4 = 2.5

    ASSERT_EQUAL(iter3 - iter4, -1);
    ASSERT_EQUAL(iter4 - iter3,  1);
    ASSERT_EQUAL(iter3 == iter4, false);
    ASSERT_EQUAL(iter3 < iter4, true);
    ASSERT_EQUAL(iter4 < iter3, false);
}
DECLARE_UNITTEST(TestCountingIteratorFloatComparison);


void TestCountingIteratorDistance(void)
{
    thrust::counting_iterator<int> iter1(0);
    thrust::counting_iterator<int> iter2(5);

    ASSERT_EQUAL(thrust::distance(iter1, iter2), 5);

    iter1++;
    
    ASSERT_EQUAL(thrust::distance(iter1, iter2), 4);
   
    iter2 += 100;

    ASSERT_EQUAL(thrust::distance(iter1, iter2), 104);
}
DECLARE_UNITTEST(TestCountingIteratorDistance);


void TestCountingIteratorUnsignedType(void)
{
    thrust::counting_iterator<unsigned int> iter0(0);
    thrust::counting_iterator<unsigned int> iter1(5);

    ASSERT_EQUAL(iter1 - iter0,  5);
    ASSERT_EQUAL(iter0 - iter1, -5);
    ASSERT_EQUAL(iter0 != iter1, true);
    ASSERT_EQUAL(iter0 <  iter1, true);
    ASSERT_EQUAL(iter1 <  iter0, false);
}
DECLARE_UNITTEST(TestCountingIteratorUnsignedType);


void TestCountingIteratorLowerBound(void)
{
    size_t n = 10000;
    const size_t M = 100;

    thrust::host_vector<unsigned int> h_data = unittest::random_integers<unsigned int>(n);
    for(unsigned int i = 0; i < n; ++i)
      h_data[i] %= M;

    thrust::sort(h_data.begin(), h_data.end());

    thrust::device_vector<unsigned int> d_data = h_data;

    thrust::counting_iterator<unsigned int> search_begin(0);
    thrust::counting_iterator<unsigned int> search_end(M);


    thrust::host_vector<unsigned int> h_result(M);
    thrust::device_vector<unsigned int> d_result(M);


    thrust::lower_bound(h_data.begin(), h_data.end(), search_begin, search_end, h_result.begin());

    thrust::lower_bound(d_data.begin(), d_data.end(), search_begin, search_end, d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_UNITTEST(TestCountingIteratorLowerBound);

void TestCountingIteratorDifference(void)
{
    typedef thrust::counting_iterator<thrust::detail::uint64_t> Iterator;
    typedef thrust::iterator_difference<Iterator>::type Difference;

    Difference diff = std::numeric_limits<thrust::detail::uint32_t>::max() + 1;

    Iterator first(0);
    Iterator last = first + diff;

    ASSERT_EQUAL(diff, last - first);
}
DECLARE_UNITTEST(TestCountingIteratorDifference);

THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END
