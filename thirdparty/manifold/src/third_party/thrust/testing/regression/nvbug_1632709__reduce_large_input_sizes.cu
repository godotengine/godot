#include <thrust/reduce.h> 
#include <thrust/iterator/constant_iterator.h> 

#include <assert.h>
#include <iostream>
 
int main()
{ 
  long long n = 10000000000; 

  long long r = thrust::reduce(
    thrust::constant_iterator<long long>(0)
  , thrust::constant_iterator<long long>(n)
  ); 

  std::cout << r << std::endl;

  assert(r == n);
}
 
