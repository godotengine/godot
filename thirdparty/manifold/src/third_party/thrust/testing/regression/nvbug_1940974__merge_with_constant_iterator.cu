#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

struct comp
{
  template<typename Tuple1, typename Tuple2>
  __host__ __device__
  bool operator()(const Tuple1& t1, const Tuple2& t2) 
  {
    return thrust::get<0>(t1) == thrust::get<1>(t2);
  }
};

int main()
{
    typedef thrust::device_vector<int> Vector;

    Vector second(10), third(5), fourth(5), indices(15);

    thrust::merge_by_key(thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<int>(12), second.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<int>(12), second.begin())) + 10, 
                         thrust::make_zip_iterator(thrust::make_tuple(third.begin(), fourth.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(third.begin(), fourth.begin())) + 5,
                         thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(10),
                         thrust::make_discard_iterator(),
                         indices.begin(),
                         comp());

    return 0;
}
 
