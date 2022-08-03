#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <iostream>

struct Functor 
{
  template<class Tuple>
  __host__ __device__
  float operator()(const Tuple& tuple) const
  {
    const float x = thrust::get<0>(tuple);
    const float y = thrust::get<1>(tuple);
    return x*y*2.0f / 3.0f;
  }
};

int main(void)
{
  float u[4] = { 4 , 3,  2,   1};
  float v[4] = {-1,  1,  1,  -1};
  int idx[3] = {3, 0, 1};
  float w[3] = {0, 0, 0};

  thrust::device_vector<float> U(u, u + 4);
  thrust::device_vector<float> V(v, v + 4);
  thrust::device_vector<int> IDX(idx, idx + 3);
  thrust::device_vector<float> W(w, w + 3);

  // gather multiple elements and apply a function before writing result in memory
  thrust::gather(
      IDX.begin(), IDX.end(),
      thrust::make_zip_iterator(thrust::make_tuple(U.begin(), V.begin())),
      thrust::make_transform_output_iterator(W.begin(), Functor()));

  std::cout << "result= [ ";
  for (size_t i = 0; i < 3; i++)
    std::cout << W[i] <<  " ";
  std::cout << "] \n";

  return 0;
}

