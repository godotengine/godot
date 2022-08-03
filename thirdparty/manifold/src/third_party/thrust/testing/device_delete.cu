#include <unittest/unittest.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>

struct Foo
{
  __host__ __device__
  Foo(void)
    :set_me_upon_destruction(0)
  {}

  __host__ __device__
  ~Foo(void)
  {
#ifdef __CUDA_ARCH__
    // __device__ overload
    if(set_me_upon_destruction != 0)
      *set_me_upon_destruction = true;
#endif
  }

  bool *set_me_upon_destruction;
};

#if !defined(__QNX__)
void TestDeviceDeleteDestructorInvocation(void)
{
  KNOWN_FAILURE;
//
//  thrust::device_vector<bool> destructor_flag(1, false);
//
//  thrust::device_ptr<Foo> foo_ptr  = thrust::device_new<Foo>();
//
//  Foo exemplar;
//  exemplar.set_me_upon_destruction = thrust::raw_pointer_cast(&destructor_flag[0]);
//  *foo_ptr = exemplar;
//
//  ASSERT_EQUAL(false, destructor_flag[0]);
//
//  thrust::device_delete(foo_ptr);
//
//  ASSERT_EQUAL(true, destructor_flag[0]);
}
DECLARE_UNITTEST(TestDeviceDeleteDestructorInvocation);
#endif
