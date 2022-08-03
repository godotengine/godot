#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <iostream>
#include <iomanip>

// This example computes a summed area table using segmented scan
// http://en.wikipedia.org/wiki/Summed_area_table



// convert a linear index to a linear index in the transpose 
struct transpose_index : public thrust::unary_function<size_t,size_t>
{
  size_t m, n;

  __host__ __device__
  transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

  __host__ __device__
  size_t operator()(size_t linear_index)
  {
      size_t i = linear_index / n;
      size_t j = linear_index % n;

      return m * j + i;
  }
};

// convert a linear index to a row index
struct row_index : public thrust::unary_function<size_t,size_t>
{
  size_t n;
  
  __host__ __device__
  row_index(size_t _n) : n(_n) {}

  __host__ __device__
  size_t operator()(size_t i)
  {
      return i / n;
  }
};

// transpose an M-by-N array
template <typename T>
void transpose(size_t m, size_t n, thrust::device_vector<T>& src, thrust::device_vector<T>& dst)
{
  thrust::counting_iterator<size_t> indices(0);
  
  thrust::gather
    (thrust::make_transform_iterator(indices, transpose_index(n, m)),
     thrust::make_transform_iterator(indices, transpose_index(n, m)) + dst.size(),
     src.begin(),
     dst.begin());
}


// scan the rows of an M-by-N array
template <typename T>
void scan_horizontally(size_t n, thrust::device_vector<T>& d_data)
{
  thrust::counting_iterator<size_t> indices(0);

  thrust::inclusive_scan_by_key
    (thrust::make_transform_iterator(indices, row_index(n)),
     thrust::make_transform_iterator(indices, row_index(n)) + d_data.size(),
     d_data.begin(),
     d_data.begin());
}

// print an M-by-N array
template <typename T>
void print(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
  thrust::host_vector<T> h_data = d_data;

  for(size_t i = 0; i < m; i++)
  {
    for(size_t j = 0; j < n; j++)
      std::cout << std::setw(8) << h_data[i * n + j] << " ";
    std::cout << "\n";
  }
}

int main(void)
{
  size_t m = 3; // number of rows
  size_t n = 4; // number of columns

  // 2d array stored in row-major order [(0,0), (0,1), (0,2) ... ]
  thrust::device_vector<int> data(m * n, 1);

  std::cout << "[step 0] initial array" << std::endl;
  print(m, n, data);

  std::cout << "[step 1] scan horizontally" << std::endl;
  scan_horizontally(n, data);
  print(m, n, data);

  std::cout << "[step 2] transpose array" << std::endl;
  thrust::device_vector<int> temp(m * n);
  transpose(m, n, data, temp);
  print(n, m, temp);

  std::cout << "[step 3] scan transpose horizontally" << std::endl;
  scan_horizontally(m, temp);
  print(n, m, temp);

  std::cout << "[step 4] transpose the transpose" << std::endl;
  transpose(n, m, temp, data);
  print(m, n, data);

  return 0;
}
