#include <unittest/unittest.h>

#include <thrust/system/detail/internal/decompose.h>

using thrust::system::detail::internal::uniform_decomposition;

void TestUniformDecomposition(void)
{
  {
    uniform_decomposition<int> ud(10, 10, 1);
   
    // [0,10)
    ASSERT_EQUAL(ud.size(), 1);
    ASSERT_EQUAL(ud[0].begin(),   0);
    ASSERT_EQUAL(ud[0].end(),    10);
    ASSERT_EQUAL(ud[0].size(),   10);
  }
  
  {
    uniform_decomposition<int> ud(10, 20, 1);
   
    // [0,10)
    ASSERT_EQUAL(ud.size(), 1);
    ASSERT_EQUAL(ud[0].begin(),  0);
    ASSERT_EQUAL(ud[0].end(),   10);
    ASSERT_EQUAL(ud[0].size(),  10);
  }

  {
    uniform_decomposition<int> ud(8, 5, 2);
   
    // [0,5)[5,8)
    ASSERT_EQUAL(ud.size(), 2);
    ASSERT_EQUAL(ud[0].begin(),  0);
    ASSERT_EQUAL(ud[0].end(),    5);
    ASSERT_EQUAL(ud[0].size(),   5);
    ASSERT_EQUAL(ud[1].begin(),  5);
    ASSERT_EQUAL(ud[1].end(),    8);
    ASSERT_EQUAL(ud[1].size(),   3);
  }
  
  {
    uniform_decomposition<int> ud(8, 5, 3);
   
    // [0,5)[5,8)
    ASSERT_EQUAL(ud.size(), 2);
    ASSERT_EQUAL(ud[0].begin(),  0);
    ASSERT_EQUAL(ud[0].end(),    5);
    ASSERT_EQUAL(ud[0].size(),   5);
    ASSERT_EQUAL(ud[1].begin(),  5);
    ASSERT_EQUAL(ud[1].end(),    8);
    ASSERT_EQUAL(ud[1].size(),   3);
  }

  {
    uniform_decomposition<int> ud(10, 1, 2);
   
    // [0,5)[5,10)
    ASSERT_EQUAL(ud.size(), 2);
    ASSERT_EQUAL(ud[0].begin(),  0);
    ASSERT_EQUAL(ud[0].end(),    5);
    ASSERT_EQUAL(ud[0].size(),   5);
    ASSERT_EQUAL(ud[1].begin(),  5);
    ASSERT_EQUAL(ud[1].end(),   10);
    ASSERT_EQUAL(ud[1].size(),   5);
  }

  {
    // [0,4)[4,8)[8,10)
    uniform_decomposition<int> ud(10, 2, 3);   

    ASSERT_EQUAL(ud.size(), 3);
    ASSERT_EQUAL(ud[0].begin(),  0);
    ASSERT_EQUAL(ud[0].end(),    4);
    ASSERT_EQUAL(ud[0].size(),   4);
    ASSERT_EQUAL(ud[1].begin(),  4);
    ASSERT_EQUAL(ud[1].end(),    8);
    ASSERT_EQUAL(ud[1].size(),   4);
    ASSERT_EQUAL(ud[2].begin(),  8);
    ASSERT_EQUAL(ud[2].end(),   10);
    ASSERT_EQUAL(ud[2].size(),   2);
  }
}
DECLARE_UNITTEST(TestUniformDecomposition);
