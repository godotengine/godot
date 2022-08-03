#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <iostream>
#include <iomanip>

// define a 2d float vector
typedef thrust::tuple<float,float> vec2;

// return a random vec2 in [0,1)^2
vec2 make_random_vec2(void)
{
  static thrust::default_random_engine rng;
  static thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
  float x = u01(rng);
  float y = u01(rng);
  return vec2(x,y);
}

// hash a point in the unit square to the index of
// the grid bucket that contains it
struct point_to_bucket_index : public thrust::unary_function<vec2,unsigned int>
{
  unsigned int width;  // buckets in the x dimension (grid spacing = 1/width)
  unsigned int height; // buckets in the y dimension (grid spacing = 1/height)

  __host__ __device__
  point_to_bucket_index(unsigned int width, unsigned int height)
    : width(width), height(height) {}

  __host__ __device__
  unsigned int operator()(const vec2& v) const
  {
    // find the raster indices of p's bucket
    unsigned int x = static_cast<unsigned int>(thrust::get<0>(v) * width);
    unsigned int y = static_cast<unsigned int>(thrust::get<1>(v) * height);

    // return the bucket's linear index
    return y * width + x;
  }

};

int main(void)
{
  const size_t N = 1000000;

  // allocate some random points in the unit square on the host
  thrust::host_vector<vec2> h_points(N);
  thrust::generate(h_points.begin(), h_points.end(), make_random_vec2);

  // transfer to device
  thrust::device_vector<vec2> points = h_points;

  // allocate storage for a 2D grid
  // of dimensions w x h
  unsigned int w = 200, h = 100;

  // the grid data structure keeps a range per grid bucket:
  // each bucket_begin[i] indexes the first element of bucket i's list of points
  // each bucket_end[i] indexes one past the last element of bucket i's list of points
  thrust::device_vector<unsigned int> bucket_begin(w*h);
  thrust::device_vector<unsigned int> bucket_end(w*h);

  // allocate storage for each point's bucket index
  thrust::device_vector<unsigned int> bucket_indices(N);

  // transform the points to their bucket indices
  thrust::transform(points.begin(),
                    points.end(),
                    bucket_indices.begin(),
                    point_to_bucket_index(w,h));

  // sort the points by their bucket index
  thrust::sort_by_key(bucket_indices.begin(),
                      bucket_indices.end(),
                      points.begin());

  // find the beginning of each bucket's list of points
  thrust::counting_iterator<unsigned int> search_begin(0);
  thrust::lower_bound(bucket_indices.begin(),
                      bucket_indices.end(),
                      search_begin,
                      search_begin + w*h,
                      bucket_begin.begin());

  // find the end of each bucket's list of points
  thrust::upper_bound(bucket_indices.begin(),
                      bucket_indices.end(),
                      search_begin,
                      search_begin + w*h,
                      bucket_end.begin());

  // write out bucket (150, 50)'s list of points
  unsigned int bucket_idx = 50 * w + 150;
  std::cout << "bucket (150, 50)'s list of points:" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  for(unsigned int point_idx = bucket_begin[bucket_idx];
      point_idx != bucket_end[bucket_idx];
      ++point_idx)
  {
    vec2 p = points[point_idx];
    std::cout << "(" << thrust::get<0>(p) << "," << thrust::get<1>(p) << ")" << std::endl;
  }

  return 0;
}

