#include <thrust/sort.h>
#include <thrust/device_ptr.h>
int main() {
  const int N = 100;
  thrust::device_ptr<int> input_key_A1;
  thrust::device_ptr<float> input_val_A1;
  thrust::device_ptr<int> input_key_B1;
  thrust::device_ptr<float> input_val_B1;
  thrust::device_ptr<int> output_key;
  thrust::device_ptr<float> output_val;

  // use key tuples (with one element to keep it simple)
  auto input_key_tuple_A = thrust::make_tuple(input_key_A1);
  auto input_key_tuple_B = thrust::make_tuple(input_key_B1);
  auto output_key_tuple = thrust::make_tuple(output_key);
  // use zip iterator to zip together elements of a tuple (each is an iterator)
  auto zip_it_A = thrust::make_zip_iterator(input_key_tuple_A);
  auto zip_it_B = thrust::make_zip_iterator(input_key_tuple_B);
  auto zip_it_out = thrust::make_zip_iterator(output_key_tuple);

  // does NOT compile in CUDA 9.1 (compiles fine in CUDA 8)
  thrust::merge_by_key(zip_it_A, zip_it_A + N, zip_it_B, zip_it_B + N, input_val_A1, input_val_B1, zip_it_out, output_val);

  return 0;
}

