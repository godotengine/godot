#include <unittest/unittest.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/retag.h>


template<typename InputIterator, typename ForwardIterator>
ForwardIterator uninitialized_copy(my_system &system,
                                   InputIterator,
                                   InputIterator,
                                   ForwardIterator result)
{
    system.validate_dispatch();
    return result;
}

void TestUninitializedCopyDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::uninitialized_copy(sys,
                               vec.begin(),
                               vec.begin(),
                               vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestUninitializedCopyDispatchExplicit);


template<typename InputIterator, typename ForwardIterator>
ForwardIterator uninitialized_copy(my_tag,
                                   InputIterator,
                                   InputIterator,
                                   ForwardIterator result)
{
    *result = 13;
    return result;
}

void TestUninitializedCopyDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::uninitialized_copy(thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestUninitializedCopyDispatchImplicit);


template<typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator uninitialized_copy_n(my_system &system,
                                     InputIterator,
                                     Size,
                                     ForwardIterator result)
{
    system.validate_dispatch();
    return result;
}

void TestUninitializedCopyNDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::uninitialized_copy_n(sys,
                                 vec.begin(),
                                 vec.size(),
                                 vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestUninitializedCopyNDispatchExplicit);


template<typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator uninitialized_copy_n(my_tag,
                                     InputIterator,
                                     Size,
                                     ForwardIterator result)
{
    *result = 13;
    return result;
}

void TestUninitializedCopyNDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::uninitialized_copy_n(thrust::retag<my_tag>(vec.begin()),
                                 vec.size(),
                                 thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestUninitializedCopyNDispatchImplicit);


template <class Vector>
void TestUninitializedCopySimplePOD(void)
{
    Vector v1(5);
    v1[0] = 0; v1[1] = 1; v1[2] = 2; v1[3] = 3; v1[4] = 4;

    // copy to Vector
    Vector v2(5);
    thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());
    ASSERT_EQUAL(v2[0], 0);
    ASSERT_EQUAL(v2[1], 1);
    ASSERT_EQUAL(v2[2], 2);
    ASSERT_EQUAL(v2[3], 3);
    ASSERT_EQUAL(v2[4], 4);
}
DECLARE_VECTOR_UNITTEST(TestUninitializedCopySimplePOD);


template<typename Vector>
void TestUninitializedCopyNSimplePOD(void)
{
    Vector v1(5);
    v1[0] = 0; v1[1] = 1; v1[2] = 2; v1[3] = 3; v1[4] = 4;

    // copy to Vector
    Vector v2(5);
    thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());
    ASSERT_EQUAL(v2[0], 0);
    ASSERT_EQUAL(v2[1], 1);
    ASSERT_EQUAL(v2[2], 2);
    ASSERT_EQUAL(v2[3], 3);
    ASSERT_EQUAL(v2[4], 4);
}
DECLARE_VECTOR_UNITTEST(TestUninitializedCopyNSimplePOD);


struct CopyConstructTest
{
  __host__ __device__
  CopyConstructTest(void)
    :copy_constructed_on_host(false),
     copy_constructed_on_device(false)
  {}

  __host__ __device__
  CopyConstructTest(const CopyConstructTest &)
  {
#if __CUDA_ARCH__
    copy_constructed_on_device = true;
    copy_constructed_on_host   = false;
#else
    copy_constructed_on_device = false;
    copy_constructed_on_device = true;
#endif
  }

  __host__ __device__
  CopyConstructTest &operator=(const CopyConstructTest &x)
  {
    copy_constructed_on_host   = x.copy_constructed_on_host;
    copy_constructed_on_device = x.copy_constructed_on_device;
    return *this;
  }

  bool copy_constructed_on_host;
  bool copy_constructed_on_device;
};


struct TestUninitializedCopyNonPODDevice
{
  void operator()(const size_t)
  {
    typedef CopyConstructTest T;

    thrust::device_vector<T> v1(5), v2(5);

    T x;
    ASSERT_EQUAL(false, x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);

    x = v1[0];
    ASSERT_EQUAL(false, x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);

    thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());

    x = v2[0];
    ASSERT_EQUAL(true,  x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);
  }
};
DECLARE_UNITTEST(TestUninitializedCopyNonPODDevice);


struct TestUninitializedCopyNNonPODDevice
{
  void operator()(const size_t)
  {
    typedef CopyConstructTest T;

    thrust::device_vector<T> v1(5), v2(5);

    T x;
    ASSERT_EQUAL(false, x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);

    x = v1[0];
    ASSERT_EQUAL(false, x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);

    thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());

    x = v2[0];
    ASSERT_EQUAL(true,  x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);
  }
};
DECLARE_UNITTEST(TestUninitializedCopyNNonPODDevice);


struct TestUninitializedCopyNonPODHost
{
  void operator()(const size_t)
  {
    typedef CopyConstructTest T;

    thrust::host_vector<T> v1(5), v2(5);

    T x;
    ASSERT_EQUAL(false, x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);

    x = v1[0];
    ASSERT_EQUAL(false, x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);

    thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());

    x = v2[0];
    ASSERT_EQUAL(false, x.copy_constructed_on_device);
    ASSERT_EQUAL(true,  x.copy_constructed_on_host);
  }
};
DECLARE_UNITTEST(TestUninitializedCopyNonPODHost);


struct TestUninitializedCopyNNonPODHost
{
  void operator()(const size_t)
  {
    typedef CopyConstructTest T;

    thrust::host_vector<T> v1(5), v2(5);

    T x;
    ASSERT_EQUAL(false, x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);

    x = v1[0];
    ASSERT_EQUAL(false, x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);

    thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());

    x = v2[0];
    ASSERT_EQUAL(false, x.copy_constructed_on_device);
    ASSERT_EQUAL(true,  x.copy_constructed_on_host);
  }
};
DECLARE_UNITTEST(TestUninitializedCopyNNonPODHost);

