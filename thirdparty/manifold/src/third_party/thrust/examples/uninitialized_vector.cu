// Occasionally, it is advantageous to avoid initializing the individual
// elements of a device_vector. For example, the default behavior of
// zero-initializing numeric data may introduce undesirable overhead.
// This example demonstrates how to avoid default construction of a
// device_vector's data by using a custom allocator.

#include <thrust/device_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <cassert>

// uninitialized_allocator is an allocator which
// derives from device_allocator and which has a
// no-op construct member function
template<typename T>
  struct uninitialized_allocator
    : thrust::device_allocator<T>
{
  // the default generated constructors and destructors are implicitly
  // marked __host__ __device__, but the current Thrust device_allocator
  // can only be constructed and destroyed on the host; therefore, we
  // define these as host only
  __host__
  uninitialized_allocator() {}
  __host__
  uninitialized_allocator(const uninitialized_allocator & other)
    : thrust::device_allocator<T>(other) {}
  __host__
  ~uninitialized_allocator() {}

#if THRUST_CPP_DIALECT >= 2011
  uninitialized_allocator & operator=(const uninitialized_allocator &) = default;
#endif

  // for correctness, you should also redefine rebind when you inherit
  // from an allocator type; this way, if the allocator is rebound somewhere,
  // it's going to be rebound to the correct type - and not to its base
  // type for U
  template<typename U>
  struct rebind
  {
    typedef uninitialized_allocator<U> other;
  };

  // note that construct is annotated as
  // a __host__ __device__ function
  __host__ __device__
  void construct(T *)
  {
    // no-op
  }
};

// to make a device_vector which does not initialize its elements,
// use uninitialized_allocator as the 2nd template parameter
typedef thrust::device_vector<float, uninitialized_allocator<float> > uninitialized_vector;

int main()
{
  uninitialized_vector vec(10);

  // the initial value of vec's 10 elements is undefined

  // resize without default value does not initialize elements
  vec.resize(20);

  // resize with default value does initialize elements
  vec.resize(30, 13);

  // the value of elements [0,20) is still undefined
  // but the value of elements [20,30) is 13:

  using namespace thrust::placeholders;
  assert(thrust::all_of(vec.begin() + 20, vec.end(), _1 == 13));

  return 0;
}

