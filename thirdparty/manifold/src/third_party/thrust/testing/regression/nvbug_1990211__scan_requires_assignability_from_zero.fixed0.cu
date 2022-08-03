#include <thrust/scan.h>
#include <thrust/device_ptr.h>

struct uint2_adder 
{ 
  __host__ __device__ uint2 operator()(uint2 a, uint2 b) {  
    return make_uint2(a.x + b.x, a.y + b.y); 
  } 
}; 
  
int main() {  
  int num_elements = 32;  
  uint2 *input = NULL, *output = NULL;
  const uint2 zero = make_uint2(0,0);  
  
  thrust::exclusive_scan(thrust::device_ptr<uint2>((uint2*)input), 
                         thrust::device_ptr<uint2>((uint2*)input + num_elements), 
                         thrust::device_ptr<uint2>(output), zero, uint2_adder());  
  
  return 0;  
}
 
