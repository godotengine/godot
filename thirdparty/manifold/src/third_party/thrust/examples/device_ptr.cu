#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include <thrust/sequence.h>
#include <thrust/reduce.h>

#include <cassert>
#include <iostream>

int main(void)
{
  // allocate memory buffer to store 10 integers on the device
  thrust::device_ptr<int> d_ptr = thrust::device_malloc<int>(10);

  // device_ptr supports pointer arithmetic 
  thrust::device_ptr<int> first = d_ptr;
  thrust::device_ptr<int> last  = d_ptr + 10;
  std::cout << "device array contains " << (last - first) << " values\n";
  
  // algorithms work as expected
  thrust::sequence(first, last);
  std::cout << "sum of values is " << thrust::reduce(first, last) << "\n";
  
  // device memory can be read and written transparently
  d_ptr[0] = 10;
  d_ptr[1] = 11;
  d_ptr[2] = d_ptr[0] + d_ptr[1];

  // device_ptr can be converted to a "raw" pointer for use in other APIs and kernels, etc.
  int * raw_ptr = thrust::raw_pointer_cast(d_ptr);

  // note: raw_ptr cannot necessarily be accessed by the host!

  // conversely, raw pointers can be wrapped
  thrust::device_ptr<int> wrapped_ptr = thrust::device_pointer_cast(raw_ptr);

  // back to where we started
  assert(wrapped_ptr == d_ptr);
  (void)wrapped_ptr; // for when NDEBUG is defined

  // deallocate device memory
  thrust::device_free(d_ptr);

  return 0;
}

