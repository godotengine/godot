#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_input_output_iterator.h>
#include <thrust/sequence.h>
#include <iostream>

// Base 2 fixed point
class ScaledInteger
{
  int value_;
  int scale_;

public:
  __host__ __device__
  ScaledInteger(int value, int scale): value_{value}, scale_{scale} {}

  __host__ __device__
  int value() const { return value_; }

  __host__ __device__
  ScaledInteger rescale(int scale) const
  {
    int shift = scale - scale_;
    int result = shift < 0 ? value_ << (-shift) : value_ >> shift;
    return ScaledInteger{result, scale};
  }

  __host__ __device__
  friend ScaledInteger operator+(ScaledInteger a, ScaledInteger b)
  {
    // Rescale inputs to the lesser of the two scales
    if (b.scale_ < a.scale_)
      a = a.rescale(b.scale_);
    else if (a.scale_ < b.scale_)
      b = b.rescale(a.scale_);
    return ScaledInteger{a.value_ + b.value_, a.scale_};
  }
};

struct ValueToScaledInteger
{
  int scale;

  __host__ __device__
  ScaledInteger operator()(const int& value) const
  {
    return ScaledInteger{value, scale};
  }
};

struct ScaledIntegerToValue
{
  int scale;

  __host__ __device__
  int operator()(const ScaledInteger& scaled) const
  {
    return scaled.rescale(scale).value();
  }
};

int main(void)
{
  const size_t size = 4;
  thrust::device_vector<int> A(size);
  thrust::device_vector<int> B(size);
  thrust::device_vector<int> C(size);

  thrust::sequence(A.begin(), A.end(), 1);
  thrust::sequence(B.begin(), B.end(), 5);

  const int A_scale = 16; // Values in A are left shifted by 16
  const int B_scale = 8;  // Values in B are left shifted by 8
  const int C_scale = 4;  // Values in C are left shifted by 4

  auto A_begin = thrust::make_transform_input_output_iterator(A.begin(),
                    ValueToScaledInteger{A_scale}, ScaledIntegerToValue{A_scale});
  auto A_end   = thrust::make_transform_input_output_iterator(A.end(),
                    ValueToScaledInteger{A_scale}, ScaledIntegerToValue{A_scale});
  auto B_begin = thrust::make_transform_input_output_iterator(B.begin(),
                    ValueToScaledInteger{B_scale}, ScaledIntegerToValue{B_scale});
  auto C_begin = thrust::make_transform_input_output_iterator(C.begin(),
                    ValueToScaledInteger{C_scale}, ScaledIntegerToValue{C_scale});

  // Sum A and B as ScaledIntegers, storing the scaled result in C
  thrust::transform(A_begin, A_end, B_begin, C_begin, thrust::plus<ScaledInteger>{});

  thrust::host_vector<int> A_h(A);
  thrust::host_vector<int> B_h(B);
  thrust::host_vector<int> C_h(C);

  std::cout << std::hex;

  std::cout << "Expected [ ";
  for (size_t i = 0; i < size; i++) {
    const int expected = ((A_h[i] << A_scale) + (B_h[i] << B_scale)) >> C_scale;
    std::cout << expected <<  " ";
  }
  std::cout << "] \n";

  std::cout << "Result   [ ";
  for (size_t i = 0; i < size; i++) {
    std::cout << C_h[i] <<  " ";
  }
  std::cout << "] \n";

  return 0;
}

