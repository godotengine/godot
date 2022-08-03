#include <unittest/unittest.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

void TestConstantIteratorConstructFromConvertibleSystem(void)
{
  using namespace thrust;

  constant_iterator<int> default_system(13);

  constant_iterator<int, use_default, host_system_tag> host_system = default_system;
  ASSERT_EQUAL(*default_system, *host_system);

  constant_iterator<int, use_default, device_system_tag> device_system = default_system;
  ASSERT_EQUAL(*default_system, *device_system);
}
DECLARE_UNITTEST(TestConstantIteratorConstructFromConvertibleSystem);

void TestConstantIteratorIncrement(void)
{
    using namespace thrust;

    constant_iterator<int> lhs(0,0);
    constant_iterator<int> rhs(0,0);

    ASSERT_EQUAL(0, lhs - rhs);

    lhs++;

    ASSERT_EQUAL(1, lhs - rhs);
    
    lhs++;
    lhs++;
    
    ASSERT_EQUAL(3, lhs - rhs);

    lhs += 5;
    
    ASSERT_EQUAL(8, lhs - rhs);

    lhs -= 10;
    
    ASSERT_EQUAL(-2, lhs - rhs);
}
DECLARE_UNITTEST(TestConstantIteratorIncrement);

void TestConstantIteratorIncrementBig(void)
{
    long long int n = 10000000000ULL;

    thrust::constant_iterator<long long int> begin(1);
    thrust::constant_iterator<long long int> end = begin + n;

    ASSERT_EQUAL(thrust::distance(begin, end), n);
}
DECLARE_UNITTEST(TestConstantIteratorIncrementBig);

void TestConstantIteratorComparison(void)
{
    using namespace thrust;

    constant_iterator<int> iter1(0);
    constant_iterator<int> iter2(0);

    ASSERT_EQUAL(0, iter1 - iter2);
    ASSERT_EQUAL(true, iter1 == iter2);

    iter1++;
    
    ASSERT_EQUAL(1, iter1 - iter2);
    ASSERT_EQUAL(false, iter1 == iter2);
   
    iter2++;

    ASSERT_EQUAL(0, iter1 - iter2);
    ASSERT_EQUAL(true, iter1 == iter2);
  
    iter1 += 100;
    iter2 += 100;

    ASSERT_EQUAL(0, iter1 - iter2);
    ASSERT_EQUAL(true, iter1 == iter2);
}
DECLARE_UNITTEST(TestConstantIteratorComparison);


void TestMakeConstantIterator(void)
{
    using namespace thrust;

    // test one argument version
    constant_iterator<int> iter0 = make_constant_iterator<int>(13);

    ASSERT_EQUAL(13, *iter0);

    // test two argument version
    constant_iterator<int,thrust::detail::intmax_t> iter1 = make_constant_iterator<int,thrust::detail::intmax_t>(13, 7);

    ASSERT_EQUAL(13, *iter1);
    ASSERT_EQUAL(7, iter1 - iter0);
}
DECLARE_UNITTEST(TestMakeConstantIterator);


template<typename Vector>
void TestConstantIteratorCopy(void)
{
  using namespace thrust;

  using ValueType = typename Vector::value_type;
  using ConstIter = constant_iterator<ValueType>;

  Vector result(4);

  ConstIter first = make_constant_iterator<ValueType>(7);
  ConstIter last  = first + result.size();
  thrust::copy(first, last, result.begin());

  ASSERT_EQUAL(7, result[0]);
  ASSERT_EQUAL(7, result[1]);
  ASSERT_EQUAL(7, result[2]);
  ASSERT_EQUAL(7, result[3]);
};
DECLARE_VECTOR_UNITTEST(TestConstantIteratorCopy);


template<typename Vector>
void TestConstantIteratorTransform(void)
{
  using namespace thrust;

  typedef typename Vector::value_type T;
  typedef constant_iterator<T> ConstIter;

  Vector result(4);

  ConstIter first1 = make_constant_iterator<T>(7);
  ConstIter last1  = first1 + result.size();
  ConstIter first2 = make_constant_iterator<T>(3);

  thrust::transform(first1, last1, result.begin(), thrust::negate<T>());

  ASSERT_EQUAL(-7, result[0]);
  ASSERT_EQUAL(-7, result[1]);
  ASSERT_EQUAL(-7, result[2]);
  ASSERT_EQUAL(-7, result[3]);
  
  thrust::transform(first1, last1, first2, result.begin(), thrust::plus<T>());

  ASSERT_EQUAL(10, result[0]);
  ASSERT_EQUAL(10, result[1]);
  ASSERT_EQUAL(10, result[2]);
  ASSERT_EQUAL(10, result[3]);
};
DECLARE_VECTOR_UNITTEST(TestConstantIteratorTransform);


void TestConstantIteratorReduce(void)
{
  using namespace thrust;

  typedef int T;
  typedef constant_iterator<T> ConstIter;

  ConstIter first = make_constant_iterator<T>(7);
  ConstIter last  = first + 4;

  T sum = thrust::reduce(first, last);

  ASSERT_EQUAL(sum, 4 * 7);
};
DECLARE_UNITTEST(TestConstantIteratorReduce);

