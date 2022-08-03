#include <unittest/unittest.h>

#include <thrust/detail/type_traits.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/device_ptr.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

struct non_pod
{
  // non-pods can have constructors
  non_pod(void)
  {}

  int x; int y;
};

void TestIsPlainOldData(void)
{
    // primitive types
    ASSERT_EQUAL((bool)thrust::detail::is_pod<bool>::value, true);

    ASSERT_EQUAL((bool)thrust::detail::is_pod<char>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed char>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned char>::value, true);
    
    ASSERT_EQUAL((bool)thrust::detail::is_pod<short>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed short>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned short>::value, true);

    ASSERT_EQUAL((bool)thrust::detail::is_pod<int>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed int>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned int>::value, true);
    
    ASSERT_EQUAL((bool)thrust::detail::is_pod<long>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed long>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned long>::value, true);
    
    ASSERT_EQUAL((bool)thrust::detail::is_pod<long long>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed long long>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned long long>::value, true);
    
    ASSERT_EQUAL((bool)thrust::detail::is_pod<float>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<double>::value, true);
    
    // void
    ASSERT_EQUAL((bool)thrust::detail::is_pod<void>::value, true);

    // structs
    ASSERT_EQUAL((bool)thrust::detail::is_pod<non_pod>::value, false);

    // pointers
    ASSERT_EQUAL((bool)thrust::detail::is_pod<char *>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<int *>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<int **>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<non_pod *>::value, true);

    // const types
    ASSERT_EQUAL((bool)thrust::detail::is_pod<const int>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<const int *>::value, true);
}
DECLARE_UNITTEST(TestIsPlainOldData);

void TestIsContiguousIterator(void)
{
    typedef thrust::host_vector<int>   HostVector;
    typedef thrust::device_vector<int> DeviceVector;
    
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator< int * >::value, true);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator< thrust::device_ptr<int> >::value, true);


    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<HostVector::iterator>::value, true);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<HostVector::const_iterator>::value, true);

    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<DeviceVector::iterator>::value, true);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<DeviceVector::const_iterator>::value, true);

    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator< thrust::device_ptr<int> >::value, true);

    typedef thrust::tuple< HostVector::iterator,   HostVector::iterator   > HostIteratorTuple;

    typedef thrust::constant_iterator<int> ConstantIterator;
    typedef thrust::counting_iterator<int> CountingIterator;
    typedef thrust::transform_iterator<thrust::identity<int>, HostVector::iterator > TransformIterator;
    typedef thrust::zip_iterator< HostIteratorTuple >  ZipIterator;

    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<ConstantIterator>::value,  false);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<CountingIterator>::value,  false);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<TransformIterator>::value, false);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<ZipIterator>::value,       false);

}
DECLARE_UNITTEST(TestIsContiguousIterator);

void TestIsCommutative(void)
{
  { typedef int T; typedef thrust::plus<T>        Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef int T; typedef thrust::multiplies<T>  Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef int T; typedef thrust::minimum<T>     Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef int T; typedef thrust::maximum<T>     Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef int T; typedef thrust::logical_or<T>  Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef int T; typedef thrust::logical_and<T> Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef int T; typedef thrust::bit_or<T>      Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef int T; typedef thrust::bit_and<T>     Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef int T; typedef thrust::bit_xor<T>     Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  
  { typedef char      T; typedef thrust::plus<T>  Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef short     T; typedef thrust::plus<T>  Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef long      T; typedef thrust::plus<T>  Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef long long T; typedef thrust::plus<T>  Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef float     T; typedef thrust::plus<T>  Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { typedef double    T; typedef thrust::plus<T>  Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  
  { typedef int   T; typedef thrust::minus<T>   Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false); }
  { typedef int   T; typedef thrust::divides<T> Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false); }
  { typedef float T; typedef thrust::divides<T> Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false); }
  { typedef float T; typedef thrust::minus<T>   Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false); }
  
  { typedef thrust::tuple<int,int> T; typedef thrust::plus<T>  Op; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false); }
}
DECLARE_UNITTEST(TestIsCommutative);

