#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
   
struct greater_than_5 
{
  template <typename T>
  __host__ __device__
  bool operator()(T val)
  {
    return abs(val) > 5;
  }
};
 
int main()
{
  typedef thrust::complex<float> T;

  thrust::device_vector<T> d(10);
  thrust::sequence(d.begin(), d.end());
  thrust::device_vector<T> r(10);

  thrust::counting_iterator<int> c_begin(0); 
  thrust::counting_iterator<int> c_end(c_begin + 10); 

  thrust::device_vector<int> idxs(10);

  thrust::copy_if(
    thrust::make_zip_iterator(thrust::make_tuple(c_begin, d.begin()))
  , thrust::make_zip_iterator(thrust::make_tuple(c_end, d.end()))
  , d.begin()
  , thrust::make_zip_iterator(thrust::make_tuple(idxs.begin(), r.begin()))
  , greater_than_5{}
  );
}
